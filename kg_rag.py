"""
kg_rag.py - Knowledge-Graph RAG layer for the Laptop Shopping Assistant
=========================================================================

Adds a graph-structured retrieval path alongside the existing dense/sparse
(RRF) retrieval in agent_functions.py. Nothing here requires a graph
database — it's a NetworkX in-memory graph, built once from the same
laptop records loaded straight out of Qdrant's `laptops_chunked` collection
(the same collection chunk_qdrant_pytorch.py embeds into, and the same
source agent_functions.py's build_vector_store reads from — no local pkl
file anywhere) + the existing keyword feature cache, and persisted to disk
so it isn't rebuilt every run. Swap `_build_networkx_graph` for a Neo4j/AGE-
backed builder later if you outgrow in-memory.

NOTE: the standalone MCP server (mcp_server.py) that used to expose this
module's tools to external MCP clients has been removed. Everything here
is now driven only from agent_functions.py / the admin routes in
agent_app.py.

Additional GraphRAG-style capabilities in this module:

  - Document structure graphing  -> Chunk nodes per laptop description,
                                     linked to the Laptop and to each other
                                     in reading order (HAS_CHUNK / NEXT_CHUNK)
  - Flattened vector subspaces   -> a flat numpy matrix over KG entity
                                     nodes (deterministic hashed features),
                                     for cheap nearest-neighbour lookups
                                     without a full embedding call
  - Non-incremental indexing     -> `reindex_non_incremental()` always
                                     rebuilds from scratch (drops the old
                                     cache/graph rather than patching it)
  - Local-level retrieval        -> `local_search()`, a tight 1-hop,
                                     entity-centric walk (vs. the wider
                                     multi-hop `kg_retrieve`)
  - One-time retrieval           -> `one_time_retrieve()` memoizes a
                                     retrieval per query key so the same
                                     turn never re-walks the graph twice
  - Literal sequential mapping   -> `literal_sequential_map()`, an
                                     order-preserving exact-match token
                                     -> node mapping with no PageRank/fuzz

Pipeline this module adds:

    user query
        │
        ▼
    entity_linking()       -> pulls out {gpu tier, cpu tier, ram tier,
                               brand, price ceiling, use-case} mentioned
        │
        ▼
    extract_subgraph()     -> walks the KG from linked entity nodes,
                               `hops` steps out, returns a subgraph
        │
        ▼
    subgraph_to_triplets() -> subgraph edges -> (subject, predicate, object)
                               triplet strings, ranked by a PPR-style score
        │
        ▼
    triplets_to_context()  -> triplets -> natural-language context chunks,
                               to be fused with (or replace) the vector
                               search context before the LLM compare step
        │
        ▼
    evaluate_kg_rag()       -> graph-specific RAG metrics: triplet
                               precision/recall against the query's linked
                               entities, path relevance, context coverage.

Public API used by agent_functions.py:
    build_knowledge_graph(collection_name) -> builds/loads the graph once
    kg_retrieve(requirements, req_string, top_k) -> dict with triplets,
                                                     context, laptop ids
    evaluate_kg_rag(question, triplets, ranked_laptop_ids) -> metrics dict
    graph_stats()                        -> summary dict (for MCP/admin)
    explain_subgraph(laptop_id)          -> triplets touching one laptop
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================
KG_CACHE_PATH   = "kg_cache.gpickle.json"   # our own edge-list serialization
KG_HOPS         = 2                          # subgraph radius from linked entities
KG_TOP_TRIPLETS = 25                         # triplets returned per query
KG_TOP_LAPTOPS  = 8                          # laptop candidates handed back
KG_SCHEMA_VERSION = 2                        # invalidate pre-entity-extraction caches

# Qdrant connection — must match QDRANT_HOST/PORT/COLLECTION in
# agent_functions.py. The laptop catalog for the graph is read straight out
# of this collection (no local pkl file); overridable via env var so this
# module and agent_functions.py can't silently point at different Qdrant
# collections.
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "laptops_chunked")

_qdrant_client: Optional[QdrantClient] = None


def _get_qdrant_client() -> Optional[QdrantClient]:
    """Lazily create (and cache) a Qdrant client for catalog loading."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    try:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        _qdrant_client.get_collections()
        return _qdrant_client
    except Exception as e:
        logger.error(f"KG build: could not connect to Qdrant: {e}")
        _qdrant_client = None
        return None


def _load_laptops_from_qdrant(collection_name: str) -> List[dict]:
    """
    Reconstruct one flat laptop record per laptop_id (id, name, price,
    description) by scrolling the chunked Qdrant collection and keeping
    the lowest chunk_index payload per laptop_id, using full_description
    as the description. Mirrors agent_functions.load_catalog_from_qdrant —
    kept as a separate lightweight copy here so this module has no import
    dependency on agent_functions (avoids a circular import).
    """
    client = _get_qdrant_client()
    if not client:
        return []

    by_laptop: Dict[str, dict] = {}
    try:
        next_offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                laptop_id = str(payload.get("laptop_id", ""))
                if not laptop_id:
                    continue
                chunk_index = payload.get("chunk_index", 0)
                existing = by_laptop.get(laptop_id)
                if existing is None or chunk_index < existing.get("_chunk_index", 0):
                    by_laptop[laptop_id] = {
                        "id": laptop_id,
                        "name": payload.get("name", ""),
                        "price": payload.get("price", 0),
                        "description": payload.get("full_description", payload.get("description", "")),
                        "_chunk_index": chunk_index,
                    }
            if next_offset is None:
                break
    except Exception as e:
        logger.error(f"KG build: could not load catalog from Qdrant collection '{collection_name}': {e}")
        return []

    return [{k: v for k, v in l.items() if k != "_chunk_index"} for l in by_laptop.values()]

# tiers we already compute via _KW_RULES / _FEATURE_CACHE in agent_functions.py
_FEATURES = ["GPU intensity", "Display quality", "Portability",
             "Multitasking", "Processing speed"]

_BRAND_TOKENS = ["dell", "hp", "lenovo", "asus", "acer", "apple", "msi",
                  "razer", "samsung", "lg", "microsoft", "huawei"]

_USE_CASE_TOKENS = {
    "gaming":      ["gaming", "fps", "esports"],
    "creator":     ["editing", "rendering", "design", "creator", "video"],
    "business":    ["business", "office", "productivity", "enterprise"],
    "student":     ["student", "college", "budget"],
    "development": ["developer", "coding", "programming", "devops"],
}

PRICE_BANDS = [
    ("budget",   0,       50_000),
    ("mid",      50_000,  100_000),
    ("premium",  100_000, 200_000),
    ("flagship", 200_000, float("inf")),
]

# =============================================================================
# MODULE STATE
# =============================================================================
_GRAPH: Optional[nx.MultiDiGraph] = None
_LAPTOP_INDEX: Dict[str, dict] = {}   # laptop_id -> laptop record

# flattened vector subspace: parallel arrays, row i of _FLAT_MATRIX <-> _FLAT_IDS[i]
_FLAT_MATRIX: Optional[np.ndarray] = None
_FLAT_IDS: List[str] = []
FLAT_VECTOR_DIM = 64

# one-time retrieval memo — cleared whenever the graph is rebuilt
_RETRIEVAL_CACHE: Dict[str, dict] = {}


# =============================================================================
# BUILD
# =============================================================================
def _price_band(price: float) -> str:
    for name, lo, hi in PRICE_BANDS:
        if lo <= price < hi:
            return name
    return "flagship"


def _extract_entities_from_description(desc: str) -> dict:
    lower = desc.lower()
    brand = next((b for b in _BRAND_TOKENS if b in lower), "unknown_brand")
    use_cases = [uc for uc, kws in _USE_CASE_TOKENS.items()
                 if any(kw in lower for kw in kws)]
    return {"brand": brand, "use_cases": use_cases}


def _normalise(value: str) -> str:
    """Stable entity IDs link spelling/case variants without an extra model."""
    value = value.lower().replace("nvidia geforce", "").replace("geforce", "")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _extract_spec_triplets(name: str, desc: str) -> List[Tuple[str, str]]:
    """Extract catalogue entities/relations from specifications, not keywords.

    These regular expressions target common structured laptop fields and are
    deliberately cheap enough to run once during graph construction.
    """
    text = f"{name} {desc}".lower()
    patterns = {
        "HAS_GPU": r"\b((?:nvidia )?(?:geforce )?(?:rtx|gtx)\s*\d{3,4}(?:\s*ti)?|radeon\s+[\w ]{2,20}|iris\s+xe)\b",
        "HAS_CPU": r"\b((?:intel )?core\s*i[3579](?:-\d{4,5}[a-z]*)?|(?:amd )?ryzen\s*[3579](?:\s*\d{4,5}[a-z]*)?|apple\s*m[1-4](?:\s*(?:pro|max))?)\b",
        "HAS_RAM": r"\b(\d{1,3}\s*gb\s*(?:ddr[45]|lpddr\w*)?\s*ram)\b",
        "HAS_STORAGE": r"\b((?:\d+(?:\.\d+)?\s*(?:tb|gb))\s*(?:ssd|nvme|storage))\b",
        "HAS_DISPLAY": r"\b((?:\d{3,4}\s*[x×]\s*\d{3,4})|(?:oled|ips|mini led))\b",
        "HAS_REFRESH_RATE": r"\b(\d{2,3}\s*hz)\b",
        "WEIGHS": r"\b(\d(?:\.\d+)?\s*kg)\b",
    }
    triplets = []
    for predicate, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.I):
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            triplets.append((predicate, value))
    return list(dict.fromkeys(triplets))


def _add_document_structure(g: nx.MultiDiGraph, laptop_id: str, desc: str,
                             max_chunks: int = 12) -> None:
    """
    Document-structure graphing: split a laptop's description into
    sentence-level Chunk nodes and wire them into the graph two ways —

      Laptop -HAS_CHUNK-> Chunk_i          (which doc a chunk belongs to)
      Chunk_i -NEXT_CHUNK-> Chunk_{i+1}     (reading order within the doc)

    This gives the graph an explicit "document layer" underneath the
    entity layer (Tier/Brand/UseCase/PriceBand), so retrieval can walk
    from a laptop down into the literal text that justifies it, not just
    up into the abstract tiers it was tagged with.
    """
    if not desc:
        return
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", desc) if s.strip()]
    prev_chunk = None
    for i, sent in enumerate(sentences[:max_chunks]):
        chunk_id = f"{laptop_id}::chunk::{i}"
        g.add_node(chunk_id, type="Chunk", text=sent, order=i, laptop_id=laptop_id)
        g.add_edge(laptop_id, chunk_id, predicate="HAS_CHUNK")
        if prev_chunk is not None:
            g.add_edge(prev_chunk, chunk_id, predicate="NEXT_CHUNK")
        prev_chunk = chunk_id


def _build_networkx_graph(laptops: List[dict], feature_cache: Dict[str, dict]) -> nx.MultiDiGraph:
    """
    Node types:  Laptop, GPUTier, DisplayTier, PortabilityTier,
                 MultitaskTier, ProcessingTier, Brand, PriceBand, UseCase
    Edge (triplet) shape: (laptop_id, PREDICATE, entity_id)

    Predicates: HAS_GPU_TIER, HAS_DISPLAY_TIER, HAS_PORTABILITY_TIER,
                HAS_MULTITASK_TIER, HAS_PROCESSING_TIER, MADE_BY,
                IN_PRICE_BAND, SUITED_FOR, SIMILAR_TO (laptop-laptop,
                added post-hoc based on shared tier/brand overlap)
    """
    g = nx.MultiDiGraph(schema_version=KG_SCHEMA_VERSION)

    for laptop in laptops:
        lid  = str(laptop.get("id"))
        name = laptop.get("name", lid)
        desc = laptop.get("description", "")
        price = float(laptop.get("price", 0) or 0)

        g.add_node(lid, type="Laptop", name=name, price=price, description=desc)

        feats = feature_cache.get(desc, {})
        for feat in _FEATURES:
            tier = feats.get(feat)
            if not tier:
                continue
            tier_node = f"{feat}::{tier}"
            g.add_node(tier_node, type="Tier", feature=feat, tier=tier)
            predicate = "HAS_" + feat.split()[0].upper() + "_TIER"
            g.add_edge(lid, tier_node, predicate=predicate)

        ents = _extract_entities_from_description(desc)
        brand_node = f"Brand::{ents['brand']}"
        g.add_node(brand_node, type="Brand", name=ents["brand"])
        g.add_edge(lid, brand_node, predicate="MADE_BY")

        band = _price_band(price)
        band_node = f"PriceBand::{band}"
        g.add_node(band_node, type="PriceBand", name=band)
        g.add_edge(lid, band_node, predicate="IN_PRICE_BAND")

        for uc in ents["use_cases"]:
            uc_node = f"UseCase::{uc}"
            g.add_node(uc_node, type="UseCase", name=uc)
            g.add_edge(lid, uc_node, predicate="SUITED_FOR")

        for predicate, value in _extract_spec_triplets(name, desc):
            entity_type = predicate.removeprefix("HAS_").removesuffix("S")
            entity_id = f"{entity_type}::{_normalise(value)}"
            g.add_node(entity_id, type=entity_type, name=value)
            g.add_edge(lid, entity_id, predicate=predicate)

        _add_document_structure(g, lid, desc)

    # laptop-laptop SIMILAR_TO edges: laptops sharing >=3 tier/brand nodes.
    # Built via inverted index (group by shared node) instead of O(n^2)
    # all-pairs comparison, so this stays fast for catalogs of any size.
    laptop_ids = [n for n, d in g.nodes(data=True) if d.get("type") == "Laptop"]
    shared_neighbors: Dict[str, set] = {
        lid: {v for _, v, k in g.out_edges(lid, keys=True)
              if g.nodes[v].get("type") in ("Tier", "Brand")}
        for lid in laptop_ids
    }

    # inverted index: entity node -> laptops that have it
    node_to_laptops: Dict[str, List[str]] = {}
    for lid, nodes in shared_neighbors.items():
        for node in nodes:
            node_to_laptops.setdefault(node, []).append(lid)

    pair_overlap: Dict[Tuple[str, str], int] = {}
    MAX_BUCKET = 200  # skip buckets so large they'd blow up pair counts anyway
    for node, lids in node_to_laptops.items():
        if len(lids) > MAX_BUCKET:
            continue
        for i, a in enumerate(lids):
            for b in lids[i + 1:]:
                key = (a, b) if a < b else (b, a)
                pair_overlap[key] = pair_overlap.get(key, 0) + 1

    added = 0
    for (a, b), overlap in pair_overlap.items():
        if overlap >= 3:
            g.add_edge(a, b, predicate="SIMILAR_TO", weight=overlap)
            g.add_edge(b, a, predicate="SIMILAR_TO", weight=overlap)
            added += 1
    logger.info(f"   KG similarity pass: {len(pair_overlap)} candidate pairs, "
                f"{added} SIMILAR_TO links")

    logger.info(f"   KG built: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
                f"({len(laptop_ids)} laptops)")
    return g


def _save_graph_cache(g: nx.MultiDiGraph, path: str = KG_CACHE_PATH) -> None:
    try:
        data = nx.node_link_data(g)
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"   KG cached to {path}")
    except Exception as e:
        logger.warning(f"   Could not cache KG to {path}: {e}")


def _load_graph_cache(path: str = KG_CACHE_PATH) -> Optional[nx.MultiDiGraph]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        g = nx.node_link_graph(data, directed=True, multigraph=True)
        if g.graph.get("schema_version") != KG_SCHEMA_VERSION:
            logger.info("   KG cache schema is outdated; rebuilding enriched graph")
            return None
        logger.info(f"   KG loaded from cache {path} "
                    f"({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)")
        return g
    except Exception as e:
        logger.warning(f"   Could not load KG cache {path}: {e}")
        return None


def build_knowledge_graph(collection_name: str = QDRANT_COLLECTION,
                            feature_cache: Optional[Dict[str, dict]] = None,
                            force_rebuild: bool = False,
                            cache_path: str = KG_CACHE_PATH) -> nx.MultiDiGraph:
    """
    Build once, cache in-process AND on disk. Subsequent process starts load
    the cached graph instead of re-deriving it from disk, same pattern as
    the existing `laptop_features_cache.json`. Pass force_rebuild=True (or
    delete the cache file) after the underlying dataset changes.

    collection_name defaults to the same `laptops_chunked` Qdrant collection
    that chunk_qdrant_pytorch.py populates and that agent_functions.py's
    vector search reads from — the laptop catalog is loaded straight from
    there (see _load_laptops_from_qdrant above), so the KG can never
    disagree with the vector index on which/how many laptops exist.
    """
    global _GRAPH, _LAPTOP_INDEX
    if _GRAPH is not None and not force_rebuild:
        return _GRAPH

    # always need the laptop index in memory for kg_retrieve() to return
    # full laptop records, cache or no cache
    laptops = _load_laptops_from_qdrant(collection_name)
    _LAPTOP_INDEX = {str(l.get("id")): l for l in laptops}
    if not laptops:
        logger.error(f"KG build: no laptops loaded from Qdrant collection '{collection_name}'")

    if not force_rebuild:
        cached = _load_graph_cache(cache_path)
        if cached is not None:
            _GRAPH = cached
            return _GRAPH

    fc = feature_cache or {}
    _GRAPH = _build_networkx_graph(laptops, fc)
    _save_graph_cache(_GRAPH, cache_path)
    _RETRIEVAL_CACHE.clear()
    return _GRAPH


def reindex_non_incremental(collection_name: str = QDRANT_COLLECTION,
                             feature_cache: Optional[Dict[str, dict]] = None,
                             cache_path: str = KG_CACHE_PATH) -> nx.MultiDiGraph:
    """
    Non-incremental indexing: always throw the whole graph away and build
    a fresh one from Qdrant collection `collection_name`, rather than
    trying to diff/patch the existing graph or cache file. There is no
    per-laptop upsert path in this module by design — every reindex is a
    full rebuild, which keeps the graph's derived structures (SIMILAR_TO
    edges, price bands, chunk ordering) always internally consistent with
    the current dataset instead of accumulating drift from partial updates.
    """
    global _GRAPH
    logger.info(f"🔄 [KG REINDEX] non-incremental rebuild starting (collection={collection_name})")
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"   Removed stale cache {cache_path}")
        except OSError as e:
            logger.warning(f"   Could not remove stale KG cache {cache_path}: {e}")
    _GRAPH = None
    _RETRIEVAL_CACHE.clear()
    graph = build_knowledge_graph(collection_name=collection_name, feature_cache=feature_cache,
                                   force_rebuild=True, cache_path=cache_path)
    build_flattened_vector_subspace()
    logger.info(f"✅ [KG REINDEX] complete — {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges")
    return graph


# =============================================================================
# ENTITY LINKING  (query text/requirements -> KG node ids)
# =============================================================================
def entity_linking(requirements: dict, req_string: str = "") -> List[str]:
    """Map structured requirements (+ raw text) to KG node ids to seed the walk."""
    if _GRAPH is None:
        return []

    seeds: List[str] = []
    for feat in _FEATURES:
        tier = (requirements.get(feat) or "").lower()
        if tier:
            node = f"{feat}::{tier}"
            if node in _GRAPH:
                seeds.append(node)

    budget = requirements.get("Budget")
    if budget:
        band = _price_band(float(budget))
        node = f"PriceBand::{band}"
        if node in _GRAPH:
            seeds.append(node)

    lower = req_string.lower()
    for brand in _BRAND_TOKENS:
        node = f"Brand::{brand}"
        if brand in lower and node in _GRAPH:
            seeds.append(node)
    for brand in requirements.get("Brand", []) or []:
        node = f"Brand::{_normalise(str(brand))}"
        if node in _GRAPH:
            seeds.append(node)
    for uc in _USE_CASE_TOKENS:
        node = f"UseCase::{uc}"
        if uc in lower and node in _GRAPH:
            seeds.append(node)
    for uc in requirements.get("Use cases", []) or []:
        node = f"UseCase::{_normalise(str(uc))}"
        if node in _GRAPH:
            seeds.append(node)

    return list(dict.fromkeys(seeds))  # dedupe, preserve order


# =============================================================================
# SUBGRAPH EXTRACTION
# =============================================================================
def extract_subgraph(seed_nodes: List[str], hops: int = KG_HOPS) -> nx.MultiDiGraph:
    """Ego-graph union over all seeds, `hops` steps, undirected traversal
    (a laptop node is reachable from a tier node and vice versa)."""
    if _GRAPH is None or not seed_nodes:
        return nx.MultiDiGraph()

    undirected = _GRAPH.to_undirected(as_view=True)
    nodes: set = set()
    for seed in seed_nodes:
        if seed not in undirected:
            continue
        ego = nx.ego_graph(undirected, seed, radius=hops)
        nodes |= set(ego.nodes())

    return _GRAPH.subgraph(nodes).copy()


# =============================================================================
# TRIPLET EXTRACTION + RANKING
# =============================================================================
def subgraph_to_triplets(subgraph: nx.MultiDiGraph, seed_nodes: List[str],
                           top_k: int = KG_TOP_TRIPLETS) -> List[dict]:
    """
    Turn subgraph edges into ranked (subject, predicate, object) triplets.
    Ranking = personalized PageRank seeded at the linked entities, so
    triplets closer to what the user actually asked for surface first.
    """
    if subgraph.number_of_nodes() == 0:
        return []

    personalization = {n: (1.0 if n in seed_nodes else 0.0) for n in subgraph.nodes()}
    if not any(personalization.values()):
        personalization = None  # fall back to uniform PageRank

    try:
        scores = nx.pagerank(subgraph, personalization=personalization, weight="weight")
    except Exception:
        scores = {n: 1.0 for n in subgraph.nodes()}

    triplets = []
    for u, v, data in subgraph.edges(data=True):
        subj = _node_label(subgraph, u)
        obj  = _node_label(subgraph, v)
        pred = data.get("predicate", "RELATED_TO")
        score = (scores.get(u, 0) + scores.get(v, 0)) / 2
        triplets.append({"subject": subj, "predicate": pred, "object": obj,
                          "subject_id": u, "object_id": v, "score": round(score, 6)})

    triplets.sort(key=lambda t: t["score"], reverse=True)
    return triplets[:top_k]


def _node_label(g: nx.MultiDiGraph, node_id: str) -> str:
    data = g.nodes[node_id]
    if data.get("type") == "Laptop":
        return data.get("name", node_id)
    return data.get("name") or data.get("tier") or node_id.split("::")[-1]


def triplets_to_context(triplets: List[dict]) -> List[str]:
    """Natural-language sentences, suitable as RAG context chunks."""
    verb = {
        "MADE_BY": "is made by", "IN_PRICE_BAND": "falls in the",
        "SUITED_FOR": "is well suited for", "SIMILAR_TO": "is similar to",
    }
    out = []
    for t in triplets:
        pred = t["predicate"]
        if pred.startswith("HAS_") and pred.endswith("_TIER"):
            feat = pred[4:-5].replace("_", " ").title()
            out.append(f"{t['subject']} has {t['object']}-tier {feat}.")
        elif pred == "IN_PRICE_BAND":
            out.append(f"{t['subject']} falls in the {t['object']} price band.")
        else:
            out.append(f"{t['subject']} {verb.get(pred, pred.lower())} {t['object']}.")
    return out


def find_laptop_node_by_name(name: str) -> Optional[str]:
    """Exact (case-insensitive) then substring match against laptop names
    already in the graph. Used by pipelines that only have a laptop name
    string (side_compare, upgrade) rather than structured requirements."""
    if _GRAPH is None or not name:
        return None
    lower = name.strip().lower()
    for lid, data in _GRAPH.nodes(data=True):
        if data.get("type") == "Laptop" and data.get("name", "").strip().lower() == lower:
            return lid
    for lid, data in _GRAPH.nodes(data=True):
        if data.get("type") == "Laptop" and lower in data.get("name", "").strip().lower():
            return lid
    return None


def kg_retrieve_for_names(names: List[str], hops: int = 1,
                            top_k: int = KG_TOP_TRIPLETS) -> dict:
    """
    KG retrieval seeded directly at specific laptop names, for pipelines
    that already know which laptop(s) they're talking about (side_compare)
    rather than deriving requirements from a gather conversation.
    """
    seeds = [lid for lid in (find_laptop_node_by_name(n) for n in names) if lid]
    if not seeds:
        return {"seed_nodes": [], "triplets": [], "context": [], "laptop_ids": [], "laptops": []}

    sub = extract_subgraph(seeds, hops=hops)
    triplets = subgraph_to_triplets(sub, seeds, top_k=top_k)
    context = triplets_to_context(triplets)
    return {
        "seed_nodes": seeds,
        "triplets": triplets,
        "context": context,
        "laptop_ids": seeds,
        "laptops": [_LAPTOP_INDEX[lid] for lid in seeds if lid in _LAPTOP_INDEX],
    }


def kg_retrieve_free_text(text: str, top_k: int = KG_TOP_TRIPLETS,
                            max_laptop_neighbors: int = KG_TOP_LAPTOPS) -> dict:
    """
    KG retrieval for pipelines that only have unstructured text (upgrade
    advisor's "what laptop do you currently have" description) rather than
    a structured requirements dict. Classifies the text with the same
    keyword tiers used elsewhere, links to tier/brand/use-case entities,
    and additionally tries to match it directly to a known laptop node
    (covers "I have a Dell XPS 15" naming an exact catalog model).
    """
    if _GRAPH is None:
        return {"seed_nodes": [], "triplets": [], "context": [], "laptop_ids": [], "laptops": []}

    lower = text.lower()
    seeds: List[str] = []

    # crude tier detection reusing the same feature vocabulary as agent_functions
    tier_hits = {
        "GPU intensity":    ["rtx", "gtx", "vram", "graphics"],
        "Processing speed": ["core i", "ryzen", "m1", "m2", "m3", "celeron"],
        "Multitasking":     ["gb ram", "ram"],
    }
    for feat in _FEATURES:
        for tier_word in ("high", "medium", "low"):
            node = f"{feat}::{tier_word}"
            # only seed a tier we can plausibly infer is mentioned at all
            if feat in tier_hits and any(k in lower for k in tier_hits[feat]) and node in _GRAPH:
                seeds.append(node)
                break

    for brand in _BRAND_TOKENS:
        node = f"Brand::{brand}"
        if brand in lower and node in _GRAPH:
            seeds.append(node)
    for uc in _USE_CASE_TOKENS:
        node = f"UseCase::{uc}"
        if uc in lower and node in _GRAPH:
            seeds.append(node)

    # direct laptop-name match (e.g. user names their exact current model)
    direct_match = find_laptop_node_by_name(text)
    if direct_match:
        seeds.append(direct_match)

    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        return {"seed_nodes": [], "triplets": [], "context": [], "laptop_ids": [], "laptops": []}

    sub = extract_subgraph(seeds, hops=KG_HOPS)
    triplets = subgraph_to_triplets(sub, seeds, top_k=top_k)
    context = triplets_to_context(triplets)

    laptop_scores: Counter = Counter()
    for t in triplets:
        for node_id in (t["subject_id"], t["object_id"]):
            if node_id in _GRAPH and _GRAPH.nodes[node_id].get("type") == "Laptop":
                laptop_scores[node_id] += t["score"]
    ranked_ids = [lid for lid, _ in laptop_scores.most_common(max_laptop_neighbors)]

    return {
        "seed_nodes": seeds,
        "triplets": triplets,
        "context": context,
        "laptop_ids": ranked_ids,
        "laptops": [_LAPTOP_INDEX[lid] for lid in ranked_ids if lid in _LAPTOP_INDEX],
    }


# =============================================================================
# TOP-LEVEL RETRIEVAL ENTRY POINT (what search_node calls)
# =============================================================================
def kg_retrieve(requirements: dict, req_string: str = "",
                 top_k: int = KG_TOP_LAPTOPS) -> dict:
    """
    Returns:
        {
          "seed_nodes": [...],
          "triplets": [ {subject, predicate, object, score}, ... ],
          "context": [ "sentence", ... ],
          "laptop_ids": ["3", "17", ...]   # ranked by triplet support
          "laptops": [ full laptop dicts in the same order ]
        }
    """
    seeds = entity_linking(requirements, req_string)
    if not seeds:
        return {"seed_nodes": [], "triplets": [], "context": [], "laptop_ids": [], "laptops": []}

    sub = extract_subgraph(seeds, hops=KG_HOPS)
    triplets = subgraph_to_triplets(sub, seeds, top_k=KG_TOP_TRIPLETS)
    context = triplets_to_context(triplets)

    # rank laptops by how much triplet "mass" touches them
    laptop_scores: Counter = Counter()
    for t in triplets:
        for node_id in (t["subject_id"], t["object_id"]):
            if node_id in _GRAPH and _GRAPH.nodes[node_id].get("type") == "Laptop":
                laptop_scores[node_id] += t["score"]

    ranked_ids = [lid for lid, _ in laptop_scores.most_common(top_k)]
    laptops = [_LAPTOP_INDEX[lid] for lid in ranked_ids if lid in _LAPTOP_INDEX]

    return {
        "seed_nodes": seeds,
        "triplets": triplets,
        "context": context,
        "laptop_ids": ranked_ids,
        "laptops": laptops,
    }


def fuse_kg_with_vector_results(kg_result: dict, vector_ranked: List[dict],
                                  kg_weight: float = 0.4) -> List[dict]:
    """
    Blend KG-derived laptop ranking with the existing dense/sparse RRF ranking
    (the `ranked` list produced in search_node). Same RRF-style reciprocal
    rank fusion, just across two retrieval systems instead of two indexes.
    """
    kg_ranks = {lid: i for i, lid in enumerate(kg_result.get("laptop_ids", []))}
    vec_ranks = {str(l.get("id")): i for i, l in enumerate(vector_ranked)}

    all_ids = set(kg_ranks) | set(vec_ranks)
    fused = []
    by_id = {str(l.get("id")): l for l in vector_ranked}
    by_id.update({lid: lap for lid, lap in zip(kg_result.get("laptop_ids", []),
                                                 kg_result.get("laptops", []))})

    for lid in all_ids:
        k = 60
        rrf = 0.0
        if lid in kg_ranks:
            rrf += kg_weight * (1.0 / (k + kg_ranks[lid] + 1))
        if lid in vec_ranks:
            rrf += (1 - kg_weight) * (1.0 / (k + vec_ranks[lid] + 1))
        laptop = dict(by_id.get(lid, {}))
        laptop["kg_fused_score"] = rrf
        fused.append(laptop)

    fused.sort(key=lambda l: l.get("kg_fused_score", 0), reverse=True)
    return fused


# =============================================================================
# GRAPH-AWARE RAG METRICS
# =============================================================================
def evaluate_kg_rag(seed_nodes: List[str], triplets: List[dict],
                     answer: str) -> dict:
    """
    Graph-specific RAG metrics computed directly from the retrieved triplets
    and the LLM's answer:

      - triplet_coverage:   fraction of retrieved triplets whose subject or
                             object text actually appears in the answer
                             (crude grounding proxy, no LLM call needed)
      - seed_recall:        fraction of linked query entities that are
                             represented anywhere in the returned triplets
      - avg_path_score:     mean PageRank-derived score of triplets used,
                             i.e. how "central" the retrieved facts are to
                             what the user asked, vs noise from the subgraph
    """
    if not triplets:
        return {"triplet_coverage": 0.0, "seed_recall": 0.0, "avg_path_score": 0.0}

    lower_answer = answer.lower()
    mentioned = sum(
        1 for t in triplets
        if t["subject"].lower() in lower_answer or t["object"].lower() in lower_answer
    )
    triplet_coverage = round(mentioned / len(triplets), 3)

    touched_nodes = {t["subject_id"] for t in triplets} | {t["object_id"] for t in triplets}
    seed_recall = round(
        sum(1 for s in seed_nodes if s in touched_nodes) / max(len(seed_nodes), 1), 3
    )

    avg_path_score = round(sum(t["score"] for t in triplets) / len(triplets), 6)

    return {
        "triplet_coverage": triplet_coverage,
        "seed_recall": seed_recall,
        "avg_path_score": avg_path_score,
    }


# =============================================================================
# INTROSPECTION (used by /admin routes and the MCP server)
# =============================================================================
def graph_stats() -> dict:
    if _GRAPH is None:
        return {"built": False}
    type_counts = Counter(d.get("type", "?") for _, d in _GRAPH.nodes(data=True))
    pred_counts = Counter(d.get("predicate", "?") for _, _, d in _GRAPH.edges(data=True))
    return {
        "built": True,
        "nodes": _GRAPH.number_of_nodes(),
        "edges": _GRAPH.number_of_edges(),
        "node_types": dict(type_counts),
        "predicates": dict(pred_counts),
        "laptops_indexed": len(_LAPTOP_INDEX),
    }


def explain_subgraph(laptop_id: str, hops: int = 1) -> List[dict]:
    """All triplets touching a single laptop — useful for 'why was this
    laptop recommended' explanations."""
    if _GRAPH is None or laptop_id not in _GRAPH:
        return []
    sub = extract_subgraph([laptop_id], hops=hops)
    return subgraph_to_triplets(sub, [laptop_id], top_k=50)


# =============================================================================
# FLATTENED VECTOR SUBSPACES
# =============================================================================
def _hash_embed(text: str, dim: int = FLAT_VECTOR_DIM) -> np.ndarray:
    """
    Deterministic hashed feature vector for a node label — no embedding
    model call needed. Every token is hashed into a bucket in a fixed-size
    vector (the "flattened subspace"), sign-weighted so opposite tokens
    don't just cancel to zero. Good enough for coarse nearest-neighbour
    lookups over KG entity nodes; swap in a real embedder if you need
    semantic (not lexical) similarity.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx  = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_flattened_vector_subspace(dim: int = FLAT_VECTOR_DIM) -> dict:
    """
    Flatten every non-Chunk KG node into a single dense matrix (one row
    per node) instead of leaving similarity implicit in graph topology.
    This is the "flattened vector subspace": a plain 2D numpy array plus
    a parallel id list, cheap to keep in memory and to brute-force search
    with a dot product — no vector DB required for entity-level lookups.
    """
    global _FLAT_MATRIX, _FLAT_IDS
    if _GRAPH is None:
        _FLAT_MATRIX, _FLAT_IDS = None, []
        return {"built": False, "rows": 0}

    ids, rows = [], []
    for node_id, data in _GRAPH.nodes(data=True):
        if data.get("type") == "Chunk":
            continue  # chunks live in the document layer, not the entity subspace
        label = _node_label(_GRAPH, node_id)
        ids.append(node_id)
        rows.append(_hash_embed(f"{data.get('type', '')} {label}", dim=dim))

    _FLAT_MATRIX = np.vstack(rows) if rows else np.zeros((0, dim), dtype=np.float32)
    _FLAT_IDS = ids
    logger.info(f"   Flattened vector subspace built: {len(ids)} rows x {dim} dims")
    return {"built": True, "rows": len(ids), "dim": dim}


def flat_vector_search(query_text: str, top_k: int = 10) -> List[dict]:
    """Brute-force cosine search over the flattened vector subspace."""
    if _FLAT_MATRIX is None or _FLAT_MATRIX.shape[0] == 0:
        build_flattened_vector_subspace()
    if _FLAT_MATRIX is None or _FLAT_MATRIX.shape[0] == 0:
        logger.info(f"🔎 [FLAT VECTOR SEARCH] '{query_text}' — subspace empty, 0 results")
        return []

    q = _hash_embed(query_text, dim=_FLAT_MATRIX.shape[1])
    sims = _FLAT_MATRIX @ q  # rows are already unit-normed, so this is cosine sim
    top_idx = np.argsort(-sims)[:top_k]
    results = [
        {"node_id": _FLAT_IDS[i], "label": _node_label(_GRAPH, _FLAT_IDS[i]), "score": round(float(sims[i]), 4)}
        for i in top_idx if sims[i] > 0
    ]
    logger.info(f"🔎 [FLAT VECTOR SEARCH] '{query_text}' — {len(results)} hits "
                f"(top: {results[0]['label'] if results else 'none'})")
    return results


# =============================================================================
# LOCAL-LEVEL RETRIEVAL
# =============================================================================
def local_search(entity_text: str, top_k: int = KG_TOP_TRIPLETS) -> dict:
    """
    Local-level retrieval: a tight, single-entity, single-hop walk — the
    graph-RAG equivalent of "just this node's neighbours", as opposed to
    `kg_retrieve`'s wider multi-seed, multi-hop (KG_HOPS) global walk.
    Seeds at one entity (a laptop name, brand, tier, or use-case string),
    resolved via literal matching first and the flattened subspace as a
    fallback, then returns only its immediate neighbourhood.
    """
    if _GRAPH is None:
        return {"seed_node": None, "triplets": [], "context": [], "laptop_ids": [], "laptops": []}

    seed = find_laptop_node_by_name(entity_text)
    if not seed:
        lower = entity_text.lower()
        for prefix, tokens in (("Brand::", _BRAND_TOKENS), ("UseCase::", list(_USE_CASE_TOKENS))):
            match = next((t for t in tokens if t in lower and f"{prefix}{t}" in _GRAPH), None)
            if match:
                seed = f"{prefix}{match}"
                break
    if not seed:
        hits = flat_vector_search(entity_text, top_k=1)
        seed = hits[0]["node_id"] if hits else None
    if not seed or seed not in _GRAPH:
        logger.info(f"📍 [LOCAL SEARCH] '{entity_text}' — no seed node resolved")
        return {"seed_node": None, "triplets": [], "context": [], "laptop_ids": [], "laptops": []}

    sub = extract_subgraph([seed], hops=1)
    triplets = subgraph_to_triplets(sub, [seed], top_k=top_k)
    context = triplets_to_context(triplets)
    laptop_ids = [n for n in sub.nodes if sub.nodes[n].get("type") == "Laptop"]
    logger.info(f"📍 [LOCAL SEARCH] '{entity_text}' -> seed={seed} — "
                f"{len(triplets)} triplets, {len(laptop_ids)} laptops")

    return {
        "seed_node": seed,
        "triplets": triplets,
        "context": context,
        "laptop_ids": laptop_ids,
        "laptops": [_LAPTOP_INDEX[lid] for lid in laptop_ids if lid in _LAPTOP_INDEX],
    }


# =============================================================================
# ONE-TIME RETRIEVAL
# =============================================================================
def one_time_retrieve(cache_key: str, requirements: dict, req_string: str = "",
                       top_k: int = KG_TOP_LAPTOPS) -> dict:
    """
    Memoized wrapper around `kg_retrieve`: the graph walk for a given
    `cache_key` (e.g. the turn/session id + normalized query) runs exactly
    once. Later calls with the same key return the cached result instead
    of re-walking the graph — useful when several nodes in the LangGraph
    pipeline (search_node, followup_node, evaluate) would otherwise all
    ask for the same KG context within one turn. The cache is cleared
    automatically on every rebuild/reindex.
    """
    if cache_key in _RETRIEVAL_CACHE:
        logger.info(f"♻️  [ONE-TIME RETRIEVE] cache HIT for key='{cache_key}'")
        return _RETRIEVAL_CACHE[cache_key]
    result = kg_retrieve(requirements, req_string=req_string, top_k=top_k)
    _RETRIEVAL_CACHE[cache_key] = result
    logger.info(f"🆕 [ONE-TIME RETRIEVE] cache MISS for key='{cache_key}' — "
                f"walked graph, {len(result.get('triplets', []))} triplets cached")
    return result


# =============================================================================
# LITERAL SEQUENTIAL MAPPING
# =============================================================================
def literal_sequential_map(text: str) -> List[dict]:
    """
    Order-preserving, exact-match mapping from query tokens to KG nodes —
    no PageRank, no fuzz, no hop expansion. Walks the text left to right
    and, for each token/bigram in the order it appears, records the first
    KG node whose id or label matches it literally. Useful when you want
    a deterministic, explainable trace of "which words in the query hit
    which graph nodes" rather than a ranked/inferred set of seeds.
    """
    if _GRAPH is None or not text:
        return []

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    candidates = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]

    # literal label -> node_id index, built once per call (graph is small)
    label_index: Dict[str, str] = {}
    for node_id, data in _GRAPH.nodes(data=True):
        if data.get("type") == "Chunk":
            continue
        label = _node_label(_GRAPH, node_id).lower()
        label_index.setdefault(label, node_id)
        label_index.setdefault(str(node_id).lower(), node_id)

    mapped, seen_nodes = [], set()
    for tok in candidates:
        node_id = label_index.get(tok)
        if node_id and node_id not in seen_nodes:
            mapped.append({"token": tok, "node_id": node_id,
                            "label": _node_label(_GRAPH, node_id),
                            "type": _GRAPH.nodes[node_id].get("type")})
            seen_nodes.add(node_id)
    logger.info(f"🔗 [LITERAL SEQUENTIAL MAP] '{text}' -> {len(mapped)} node(s): "
                f"{[m['token'] + '->' + m['label'] for m in mapped]}")
    return mapped
