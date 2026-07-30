"""
functions.py — Laptop Shopping Assistant (Full LangChain Orchestration)

LangChain is used for EVERYTHING:
  • LangGraph          → orchestrates the entire conversation flow as a state graph
  • LCEL (|)           → every LLM call is a prompt | llm | parser chain
  • @tool              → moderation, intent, extraction, search, recommendation as LC tools
  • ChatOllama         → LLM inference
  • OllamaEmbeddings   → embeddings
  • RunnablePassthrough / RunnableLambda → glue between steps
  • StrOutputParser / JsonOutputParser   → structured output parsing
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import requests
from collections import defaultdict
from typing import Annotated, Dict, List, Set, TypedDict

import time
import faiss
import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
OLLAMA_MODEL    = "llama3.1"
EMBEDDING_MODEL = "nomic-embed-text"
CTX_WINDOW      = 8192

MAX_MODERATION   = 20
MAX_INTENT       = 20
MAX_REQ_STRING   = 80
MAX_FUNC_CALLING = 60
MAX_HYDE         = 120
MAX_CHAT         = 400
MAX_RECO         = 600

# Shared model instances
# Shared model instances — num_predict must be in constructor, NOT via .bind()
_llm      = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=CTX_WINDOW, num_predict=MAX_CHAT)
_llm_chat = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=CTX_WINDOW, num_predict=MAX_CHAT)
_llm_reco = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=CTX_WINDOW, num_predict=MAX_RECO)
_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Keep model loaded in VRAM indefinitely (avoids eviction between long requests)
# Set via Ollama API after warm-up. -1 = never unload.
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_KEEP_ALIVE = -1


# =============================================================================
# GRAPH STATE
# =============================================================================
class AssistantState(TypedDict):
    messages:               Annotated[list[BaseMessage], add_messages]
    user_input:             str
    moderation_result:      str          # "ok" | "flagged"
    intent_confirmed:       bool
    requirement_string:     str
    user_requirements:      dict
    top_3_laptops:          str          # JSON string
    validated_laptops:      list
    recommendation:         str
    phase:                  str          # "gather" | "recommend" | "followup" | "end"


# =============================================================================
# VECTOR STORE (loaded once)
# =============================================================================
_VECTOR_STORE: list[dict] = []
_FAISS_INDEX        = None   # chunk-based index (from disk) — NOT used for search
_LAPTOP_FAISS_INDEX = None   # per-laptop index built at startup — used for dense search
_VS_BUILT     = False


_FEATURE_CACHE_PATH = "laptop_features_cache.json"

# In-memory feature cache: description -> {GPU intensity, Display quality, ...}
# Populated at startup so precision lookups are O(1) dict reads, not LLM calls.
_FEATURE_CACHE: dict[str, dict] = {}


def _load_feature_cache() -> dict:
    if os.path.exists(_FEATURE_CACHE_PATH):
        try:
            with open(_FEATURE_CACHE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_feature_cache():
    try:
        with open(_FEATURE_CACHE_PATH, "w") as f:
            json.dump(_FEATURE_CACHE, f)
    except OSError:
        pass


_KW_RULES: dict[str, dict[str, list[str]]] = {
    "GPU intensity": {
        "high": [
            "rtx 4090", "rtx 4080", "rtx 4070", "rtx 4060",
            "rtx 3080", "rtx 3070", "rtx 3060",
            "rx 7900", "rx 7800", "rx 6800", "rx 6700",
            "dedicated gpu", "discrete gpu",
            "16gb vram", "16 gb vram", "12gb vram", "12 gb vram",
            "8gb vram", "8 gb vram", "6gb vram", "6 gb vram",
        ],
        "medium": [
            "rtx 3050", "mx550", "mx450", "mx350",
            "gtx 1650", "gtx 1660",
            "rx 6600", "rx 6500",
            "4gb vram", "4 gb vram", "iris xe", "arc a",
        ],
        "low": [
            "intel uhd", "intel hd", "integrated graphics",
            "integrated gpu", "no dedicated", "onboard",
            "vega 8", "vega 7", "radeon graphics", "iris plus",
        ],
    },
    "Display quality": {
        "high": [
            "4k", "oled", "amoled", "retina", "3840", "2880",
            "mini-led", "miniled", "120hz", "144hz", "165hz", "240hz",
            "hdr", "dolby vision", "2560x1600", "2560x1440", "qhd", "wqhd",
            "2560 x 1600", "2560 x 1440", "2k resolution",
        ],
        "medium": [
            "fhd", "1920x1080", "1920 x 1080", "full hd", "1080p",
            "ips", "60hz", "90hz",
        ],
        "low": [
            "hd+", "1366x768", "1366 x 768", "hd ready",
            "tn panel", "720p", "768p",
        ],
    },
    "Portability": {
        "high": [
            "ultrabook", "ultra book", "under 1 kg", "under 1kg",
            "0.9 kg", "0.9kg", "1.0 kg", "1.0kg",
            "1.1 kg", "1.1kg", "1.2 kg", "1.2kg",
            "1.3 kg", "1.3kg", "1.4 kg", "1.4kg",
            "thin and light", "thin & light", "slim", "compact",
            "macbook air", "ultraslim", "featherlight",
        ],
        "medium": [
            "1.5 kg", "1.5kg", "1.6 kg", "1.6kg",
            "1.7 kg", "1.7kg", "1.8 kg", "1.8kg",
            "1.9 kg", "1.9kg", "2.0 kg", "2.0kg",
            "2.1 kg", "2.1kg", "2.2 kg", "2.2kg",
        ],
        "low": [
            "2.5 kg", "2.5kg", "2.6 kg", "2.6kg",
            "2.7 kg", "2.7kg", "2.8 kg", "2.8kg",
            "2.9 kg", "2.9kg", "3 kg", "3.0 kg", "3.0kg",
            "desktop replacement", "workstation",
            "17 inch", "17-inch", "17.3",
        ],
    },
    "Multitasking": {
        "high": [
            "64gb ram", "64 gb ram", "64 gb", "64gb",
            "32gb ram", "32 gb ram", "32 gb", "32gb",
            "lpddr5x", "lpddr5",
        ],
        "medium": [
            "16gb ram", "16 gb ram", "16 gb", "16gb",
            "lpddr4x", "ddr5",
        ],
        "low": [
            "8gb ram", "8 gb ram", "8 gb", "8gb",
            "4gb ram", "4 gb ram", "4 gb", "4gb",
            "lpddr4", "ddr4",
        ],
    },
    "Processing speed": {
        "high": [
            "core i9", "intel i9", "i9-", "i9 ",
            "ryzen 9", "ryzen9",
            "m3 pro", "m3 max", "m2 pro", "m2 max", "m1 pro", "m1 max",
            "xeon", "core ultra 9", "snapdragon x elite",
        ],
        "medium": [
            "core i7", "intel i7", "i7-", "i7 ",
            "ryzen 7", "ryzen7",
            "m3", "m2", "m1",
            "core ultra 7", "snapdragon x plus",
        ],
        "low": [
            "core i5", "intel i5", "i5-", "i5 ",
            "core i3", "intel i3", "i3-", "i3 ",
            "ryzen 5", "ryzen5", "ryzen 3", "ryzen3",
            "celeron", "pentium", "core ultra 5", "snapdragon 8cx",
        ],
    },
}


def _classify_one(laptop: dict) -> tuple[str, dict]:
    """
    Keyword-based laptop classifier — runs in microseconds, zero LLM calls.
    Checks high/low first; falls back to medium if neither matches.
    """
    desc = laptop["description"]
    if desc in _FEATURE_CACHE:
        return desc, _FEATURE_CACHE[desc]

    lower = desc.lower()
    features: dict[str, str] = {}
    for feature, tiers in _KW_RULES.items():
        matched = "medium"  # default fallback
        for tier in ("high", "low"):   # explicit tiers checked first
            if any(kw in lower for kw in tiers[tier]):
                matched = tier
                break
        features[feature] = matched

    return desc, features


def _preclassify_all_laptops():
    """
    Classify every laptop via keyword rules — O(N·K) string scans,
    zero LLM calls. 10 000 laptops completes in under 1 second.
    Already-cached laptops are skipped instantly.
    """
    uncached = [l for l in _VECTOR_STORE if l["description"] not in _FEATURE_CACHE]
    if not uncached:
        print(f"[FeatureCache] All {len(_VECTOR_STORE)} laptops already cached — skipping.")
        return

    total = len(uncached)
    print(f"[FeatureCache] Classifying {total} laptops with keyword rules "
          f"({len(_VECTOR_STORE) - total} already cached)...")

    for laptop in uncached:
        desc, features = _classify_one(laptop)
        _FEATURE_CACHE[desc] = features

    _save_feature_cache()
    print(f"[FeatureCache] Done. {len(_FEATURE_CACHE)} total entries saved to disk.")


def build_vector_store(
    faiss_path: str = "data/index.faiss",
    pkl_path:   str = "data/index.pkl",
):
    global _VECTOR_STORE, _FAISS_INDEX, _LAPTOP_FAISS_INDEX, _VS_BUILT, _FEATURE_CACHE
    if _VS_BUILT:
        return

    # Load chunk-based index from disk (kept for reference, not used for search)
    _FAISS_INDEX = faiss.read_index(faiss_path)
    print(f"[VectorStore] chunk index: {_FAISS_INDEX.ntotal} vectors, dim={_FAISS_INDEX.d}")

    with open(pkl_path, "rb") as f:
        _VECTOR_STORE = pickle.load(f)
    print(f"[VectorStore] {len(_VECTOR_STORE)} laptops loaded.")

    # Build a per-laptop FAISS index from full_embedding (one vector per laptop).
    # This fixes the chunk-index/laptop-list positional mismatch:
    # chunk index has 25k vectors but _VECTOR_STORE has 10k laptops, so
    # FAISS idx != laptop position. Per-laptop index keeps them aligned.
    dim = len(_VECTOR_STORE[0]["full_embedding"])
    laptop_index = faiss.IndexFlatIP(dim)
    embeddings = np.array(
        [l["full_embedding"] for l in _VECTOR_STORE], dtype="float32"
    )
    faiss.normalize_L2(embeddings)
    laptop_index.add(embeddings)
    _LAPTOP_FAISS_INDEX = laptop_index
    print(f"[VectorStore] per-laptop index: {laptop_index.ntotal} vectors, dim={dim}")

    # Load cache from disk first, then classify anything missing
    _FEATURE_CACHE = _load_feature_cache()
    _preclassify_all_laptops()

    _VS_BUILT = True


# =============================================================================
# LCEL CHAINS (every LLM call is a chain)
# =============================================================================

# ── Moderation chain ──────────────────────────────────────────────────────────
_moderation_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a content moderation classifier for a laptop shopping assistant. "
         "Flag ONLY hate speech, sexual content involving minors, weapon instructions, "
         "or threats of violence. Normal shopping messages must NEVER be flagged. "
         "Respond with ONLY valid JSON: {{\"flagged\": true}} or {{\"flagged\": false}}"),
        ("human", "{text}"),
    ])
    | ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=CTX_WINDOW, num_predict=MAX_MODERATION)
    | StrOutputParser()
)

# ── Intent confirmation chain ─────────────────────────────────────────────────
_intent_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "Check if ALL 6 laptop requirements are captured with clear values:\n"
         "1. GPU intensity (low/medium/high)\n"
         "2. Display quality (low/medium/high)\n"
         "3. Portability (low/medium/high)\n"
         "4. Multitasking (low/medium/high)\n"
         "5. Processing speed (low/medium/high)\n"
         "6. Budget (a real number >= 25000)\n"
         "If even ONE is missing or vague, set all_captured to false.\n"
         "Respond ONLY with valid JSON: {{\"all_captured\": true}} or {{\"all_captured\": false}}"),
        ("human", "{assistant_reply}"),
    ])
    | ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=CTX_WINDOW, num_predict=MAX_INTENT)
    | StrOutputParser()
)

# ── Requirement string chain ──────────────────────────────────────────────────
_req_string_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "Extract laptop requirements and output ONLY this exact sentence format:\n"
         "I need a laptop with <gpu> GPU intensity, <display> display quality, "
         "<portability> portability, <multitasking> multitasking, "
         "<processing> processing speed and a budget of <number>.\n"
         "Values must be: low/medium/high. Budget must be digits only (INR). "
         "No markdown, no explanation, nothing else."),
        ("human", "{text}"),
    ])
    | ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=CTX_WINDOW, num_predict=MAX_REQ_STRING)
    | StrOutputParser()
)

# ── Structured extraction chain ───────────────────────────────────────────────
_extraction_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "Extract laptop requirements from text. "
         "Respond ONLY with valid JSON matching exactly:\n"
         '{{"GPU intensity":"low|medium|high","Display quality":"low|medium|high",'
         '"Portability":"low|medium|high","Multitasking":"low|medium|high",'
         '"Processing speed":"low|medium|high","Budget":<integer>}}'),
        ("human", "{text}"),
    ])
    | ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=CTX_WINDOW, num_predict=MAX_FUNC_CALLING)
    | StrOutputParser()
)

# ── Laptop feature classification chain ───────────────────────────────────────
_feature_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "Classify this laptop description. "
         "Respond ONLY with valid JSON:\n"
         '{{"GPU intensity":"low|medium|high","Display quality":"low|medium|high",'
         '"Portability":"low|medium|high","Multitasking":"low|medium|high",'
         '"Processing speed":"low|medium|high"}}\n\n'
         "GPU: low=integrated/UHD, medium=Radeon/Iris/M1, high=RTX\n"
         "Display: low=<FHD, medium=FHD, high=4K/Retina\n"
         "Portability: high=<1.51kg, medium=1.51-2.51kg, low=>2.51kg\n"
         "Multitasking: low=8/12GB, medium=16GB, high=32/64GB\n"
         "Processing: low=i3/Ryzen3, medium=i5/Ryzen5, high=i7/Ryzen7+"),
        ("human", "{description}"),
    ])
    | ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=CTX_WINDOW, num_predict=MAX_FUNC_CALLING)
    | StrOutputParser()
)

# ── HyDE chain ────────────────────────────────────────────────────────────────
_hyde_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a laptop spec writer. Given requirements, write a 2-3 sentence "
         "realistic product description using technical language (GPU model, RAM, CPU, "
         "weight in kg, display type). No price. No markdown. Plain prose only."),
        ("human", "{requirement}"),
    ])
    | ChatOllama(model=OLLAMA_MODEL, temperature=0.3, num_ctx=CTX_WINDOW, num_predict=MAX_HYDE)
    | StrOutputParser()
)

# ── Conversation gathering chain ──────────────────────────────────────────────
_gather_system = """You are an intelligent laptop shopping assistant. Ask relevant 
questions to determine the user's needs across these 6 dimensions:
GPU intensity, Display quality, Portability, Multitasking, Processing speed, Budget.

Values for all except Budget must be: low, medium, or high.
Budget must be a number >= 25000 INR.

Once you have all 6 values, summarize them as:
"I need a laptop with <gpu> GPU intensity, <display> display quality, <portability> portability, 
<multitasking> multitasking, <processing> processing speed and a budget of <number>."

Start with a friendly welcome message."""

# ── Recommendation chain ──────────────────────────────────────────────────────
_reco_system = """You are an intelligent laptop expert. The user's recommended laptops are: {products}

Present them clearly in decreasing order of price:
1. <Name> : <Key specs>, Price: Rs.<price>
2. ...

Then answer any follow-up questions the user has about these laptops."""


# =============================================================================
# LANGCHAIN @tool DEFINITIONS
# =============================================================================

@tool
def moderation_tool(text: str) -> str:
    """Check if user input is safe. Returns 'ok' or 'flagged'."""
    raw = _moderation_chain.invoke({"text": text})
    try:
        flagged = json.loads(raw).get("flagged", False)
    except (json.JSONDecodeError, AttributeError):
        flagged = False
    result = "flagged" if flagged else "ok"
    print(f"[MODERATION] {text[:60]!r} → {result}")
    return result


@tool
def intent_confirmation_tool(assistant_reply: str) -> bool:
    """Check if all 6 laptop requirements have been captured."""
    raw = _intent_chain.invoke({"assistant_reply": assistant_reply})
    try:
        confirmed = json.loads(raw).get("all_captured", False)
    except (json.JSONDecodeError, AttributeError):
        confirmed = False
    numbers   = [int(n.replace(",", "")) for n in re.findall(r"[\d,]{4,}", assistant_reply)]
    has_budget = any(n >= 1000 for n in numbers)
    if not has_budget:
        confirmed = False
    print(f"[INTENT] confirmed={confirmed}, has_budget={has_budget}")
    return confirmed


_REQ_PATTERN = re.compile(
    r"I need a laptop with\s+(\w+)\s+GPU intensity[,\s]+(\w+)\s+display quality[,\s]+"
    r"(\w+)\s+portability[,\s]+(\w+)\s+multitasking[,\s]+(\w+)\s+processing speed"
    r"[\s\w]*budget\s+of\s+([\d,]+)",
    re.IGNORECASE,
)

_VALID_TIERS = {"low", "medium", "high"}


@tool
def requirement_string_tool(text: str) -> str:
    """
    Extract the canonical requirement sentence from the assistant reply.
    Uses regex first (instant, deterministic) — falls back to LLM only if
    the pattern isn't found (e.g. the model phrased it differently).
    """
    m = _REQ_PATTERN.search(text)
    if m:
        gpu, display, port, multi, proc, budget = m.groups()
        budget_clean = budget.replace(",", "")
        result = (
            f"I need a laptop with {gpu.lower()} GPU intensity, "
            f"{display.lower()} display quality, {port.lower()} portability, "
            f"{multi.lower()} multitasking, {proc.lower()} processing speed "
            f"and a budget of {budget_clean}."
        )
        print(f"[REQ STRING] regex → {result}")
        return result

    # Fallback: ask the LLM to reformat
    result = _req_string_chain.invoke({"text": text})
    print(f"[REQ STRING] llm fallback → {result}")
    return result


@tool
def extraction_tool(text: str) -> dict:
    """
    Extract structured requirements dict from requirement string.
    Validates tier values and sanitises the budget (must be >= 25000 INR).
    Falls back to regex budget extraction if the LLM returns a bad number.
    """
    raw = _extraction_chain.invoke({"text": text})
    try:
        p = json.loads(raw)
    except json.JSONDecodeError:
        p = {}

    def _tier(key: str) -> str:
        v = str(p.get(key, "medium")).lower()
        return v if v in _VALID_TIERS else "medium"

    # Parse budget — strip commas/symbols, coerce to int
    raw_budget = p.get("Budget", 0)
    try:
        budget = int(str(raw_budget).replace(",", "").replace("₹", "").strip())
    except (ValueError, TypeError):
        budget = 0

    # Sanity check: INR laptop prices are always >= 25 000
    # If extraction returned garbage, pull the largest number from the text directly
    if budget < 25_000:
        candidates = [int(n.replace(",", "")) for n in re.findall(r"[\d,]{5,}", text)]
        candidates = [n for n in candidates if n >= 25_000]
        budget = max(candidates) if candidates else budget

    result = {
        "GPU intensity":    _tier("GPU intensity"),
        "Display quality":  _tier("Display quality"),
        "Portability":      _tier("Portability"),
        "Multitasking":     _tier("Multitasking"),
        "Processing speed": _tier("Processing speed"),
        "Budget":           budget,
    }
    print(f"[EXTRACTION] {result}")
    return result


@tool
def hyde_tool(requirement: str) -> str:
    """Generate a hypothetical laptop description for better semantic search."""
    doc = _hyde_chain.invoke({"requirement": requirement})
    print(f"[HYDE] {doc}")
    return doc


@tool
def embed_tool(text: str) -> list:
    """Embed text using OllamaEmbeddings."""
    return _embedder.embed_query(text)


@tool
def dense_search_tool(query_embedding: list, budget: int, top_k: int = 10) -> list:
    """
    Search per-laptop FAISS index for nearest embeddings within budget.
    Uses _LAPTOP_FAISS_INDEX (one vector per laptop, position == _VECTOR_STORE index)
    instead of the chunk-based on-disk index (25k vectors vs 10k laptops mismatch).
    """
    index = _LAPTOP_FAISS_INDEX
    if index is None:
        print("[DENSE] laptop index not ready")
        return []
    dim = index.d
    q = list(query_embedding[:dim]) + [0.0] * max(0, dim - len(query_embedding))
    q_np = np.array([q], dtype="float32")
    faiss.normalize_L2(q_np)
    search_k = min(top_k * 10, index.ntotal)
    distances, indices = index.search(q_np, search_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_VECTOR_STORE):
            continue
        laptop = _VECTOR_STORE[int(idx)]
        if laptop["price"] > budget:
            continue
        results.append({"laptop": laptop, "score": float(dist)})
        if len(results) >= top_k:
            break
    print(f"[DENSE] {len(results)} results")
    return results


@tool
def bm25_search_tool(query: str, budget: int, top_k: int = 10) -> list:
    """BM25 keyword search within budget."""
    index: dict[str, list] = defaultdict(list)
    for laptop in _VECTOR_STORE:
        terms = laptop["description"].lower().split()
        tf_map: dict[str, int] = defaultdict(int)
        for t in terms:
            tf_map[t] += 1
        for term, tf in tf_map.items():
            index[term].append((laptop["id"], tf, len(terms)))

    affordable  = {l["id"]: l for l in _VECTOR_STORE if l["price"] <= budget}
    avg_doc_len = sum(len(l["description"].split()) for l in _VECTOR_STORE) / max(len(_VECTOR_STORE), 1)
    N, k1, b    = len(_VECTOR_STORE), 1.5, 0.75
    scores: dict[int, float] = {}
    for term in set(query.lower().split()):
        postings = index.get(term, [])
        df = len(postings)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        for (lid, tf, doc_len) in postings:
            if lid not in affordable:
                continue
            tf_norm     = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
            scores[lid] = scores.get(lid, 0.0) + idf * tf_norm
    id_to_laptop = {l["id"]: l for l in _VECTOR_STORE}
    results = sorted(
        [{"laptop": id_to_laptop[lid], "score": s} for lid, s in scores.items()],
        key=lambda x: x["score"], reverse=True,
    )
    print(f"[BM25] {len(results)} results")
    return results[:top_k]


@tool
def rrf_tool(dense_results: list, sparse_results: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion — merge dense and sparse ranked lists."""
    rrf_scores: dict[int, float] = defaultdict(float)
    laptop_map: dict[int, dict]  = {}
    for ranked_list in [dense_results, sparse_results]:
        for rank, item in enumerate(ranked_list, start=1):
            lid               = item["laptop"]["id"]
            rrf_scores[lid]   += 1.0 / (k + rank)
            laptop_map[lid]   = item["laptop"]
    merged = [{"laptop": laptop_map[lid], "rrf_score": s} for lid, s in rrf_scores.items()]
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged


@tool
def feature_extraction_tool(description: str) -> dict:
    """
    Return pre-classified feature ratings for a laptop description.
    Always an O(1) cache read after startup. Fallback for unseen descriptions
    uses keyword rules (microseconds) — never calls the LLM.
    """
    if description in _FEATURE_CACHE:
        return _FEATURE_CACHE[description]
    # Fallback: keyword classify on the fly and cache for next time
    _, features = _classify_one({"description": description})
    _FEATURE_CACHE[description] = features
    _save_feature_cache()
    return features


@tool
def score_and_rank_tool(candidates: list, user_requirements: dict) -> str:
    """Score candidate laptops against user requirements and return top 3 as JSON."""
    mappings = {"low": 0, "medium": 1, "high": 2}
    scored   = []
    for item in candidates:
        laptop   = item["laptop"]
        features = feature_extraction_tool.invoke({"description": laptop["description"]})
        score    = sum(
            1 for k, uv in user_requirements.items()
            if k.lower() != "budget"
            and mappings.get((features.get(k) or "").lower(), -1) >= mappings.get(uv.lower(), -1)
        )
        scored.append({
            "Name":        laptop["name"],
            "Description": laptop["description"],
            "Price":       laptop["price"],
            "Score":       score,
            "RRF_Score":   item.get("rrf_score", 0),
        })
    scored.sort(key=lambda x: (x["Score"], x["RRF_Score"]), reverse=True)
    return json.dumps(scored[:3])


@tool
def validation_tool(laptop_json: str) -> list:
    """
    Keep laptops with Score >= 2 (at least 2 of 5 features match).
    Falls back to the best-scoring item if nothing meets threshold so we
    never return an empty list to the user.
    """
    data = json.loads(laptop_json)
    validated = [item for item in data if item["Score"] >= 2]
    if not validated and data:
        validated = [max(data, key=lambda x: (x["Score"], x.get("RRF_Score", 0)))]
        print(f"[VALIDATION] {len(data)} -> 0 passed; returning best available (score={validated[0]['Score']})")
    else:
        print(f"[VALIDATION] {len(data)} -> {len(validated)} passed")
    return validated


# =============================================================================
# HYBRID SEARCH RUNNABLE (LCEL pipeline combining all search tools)
# =============================================================================
def _hybrid_search_pipeline(req_string: str, requirements: dict, top_k: int = 10) -> list:
    """
    Full hybrid search as an LCEL-style pipeline:
    req_string → HyDE → embed → dense+sparse → RRF → candidates
    """
    budget = int(requirements.get("Budget", 0)) or 10_000_000

    # Step 1: HyDE
    hyp_doc = hyde_tool.invoke({"requirement": req_string})

    # Step 2: Embed both
    stored_dim = len(_VECTOR_STORE[0]["full_embedding"]) if _VECTOR_STORE else None
    def _trim(v):
        if stored_dim is None: return v
        return (v + [0.0] * stored_dim)[:stored_dim]

    hyp_emb = _trim(_embedder.embed_query(hyp_doc))
    raw_emb = _trim(_embedder.embed_query(req_string))
    blended = list((np.array(hyp_emb) + np.array(raw_emb)) / 2)

    # Step 3: Dense + Sparse search
    dense_r  = dense_search_tool.invoke({"query_embedding": blended, "budget": budget, "top_k": top_k})
    sparse_r = bm25_search_tool.invoke({"query": hyp_doc + " " + req_string, "budget": budget, "top_k": top_k})

    # Step 4: RRF
    fused = rrf_tool.invoke({"dense_results": dense_r, "sparse_results": sparse_r})
    return fused[:top_k]


# =============================================================================
# PRECISION EVALUATION  (fast — zero LLM calls after startup)
# =============================================================================

# Per-call match-score cache: requirements_key → {laptop_id → score}
_MATCH_SCORE_CACHE: dict[str, dict[int, int]] = {}


def _requirements_key(requirements: Dict) -> str:
    return json.dumps(dict(sorted(requirements.items())))


def _build_match_scores(requirements: Dict) -> dict[int, int]:
    """
    Score every laptop in one pass using _FEATURE_CACHE (pure dict reads).
    Result cached so the same requirements dict is never scored twice.
    Zero LLM calls — all features were pre-classified at startup.
    """
    key = _requirements_key(requirements)
    if key in _MATCH_SCORE_CACHE:
        return _MATCH_SCORE_CACHE[key]

    mappings  = {"low": 0, "medium": 1, "high": 2}
    req_items = [(k, v) for k, v in requirements.items() if k.lower() != "budget"]
    scores: dict[int, int] = {}

    for laptop in _VECTOR_STORE:
        features = _FEATURE_CACHE.get(laptop["description"], {})
        scores[laptop["id"]] = sum(
            1 for k, uv in req_items
            if mappings.get((features.get(k) or "").lower(), -1)
            >= mappings.get(uv.lower(), -1)
        )

    _MATCH_SCORE_CACHE[key] = scores
    return scores


def get_match_score(laptop_id: int, requirements: Dict) -> int:
    """O(1) — reads from pre-built score map."""
    return _build_match_scores(requirements).get(laptop_id, 0)


def get_laptops_with_any_match(requirements: Dict) -> Set[int]:
    scores = _build_match_scores(requirements)
    return {lid for lid, s in scores.items() if s >= 1}


def get_relevant_laptops(requirements: Dict) -> Set[int]:
    """Laptops meeting ALL requirements within budget. O(N) dict scan, zero LLM calls."""
    budget    = int(requirements.get("Budget", 0)) or 10_000_000
    req_items = [(k, v) for k, v in requirements.items() if k.lower() != "budget"]
    total_req = len(req_items)
    scores    = _build_match_scores(requirements)
    return {
        l["id"] for l in _VECTOR_STORE
        if l["price"] <= budget and scores.get(l["id"], 0) == total_req
    }


def evaluate_search_precision(
    user_requirements: Dict,
    user_requirement_string: str,
    top_k: int = 10,
) -> Dict:
    """
    Precision evaluation using the same HyDE + blended embedding pipeline
    as the actual recommendation search, so the metric reflects real performance.

    Pipeline (mirrors _hybrid_search_pipeline exactly):
      1. HyDE  : generate a hypothetical spec document from the requirement string
      2. Embed : embed both HyDE doc and raw requirement string, blend them
      3. Dense : FAISS search with blended embedding
      4. Sparse: BM25 on (hyp_doc + req_string) — same as production
      5. RRF   : fuse dense + sparse ranked lists
      6. Precision: compare each result set against the pre-classified relevant set
    """
    import time

    print("\n" + "=" * 80)
    print(" PRECISION EVALUATION: SEMANTIC vs KEYWORD SEARCH")
    print("=" * 80)
    t0 = time.time()

    budget       = int(user_requirements.get("Budget", 0)) or 10_000_000
    relevant_ids = get_relevant_laptops(user_requirements)
    affordable   = [l for l in _VECTOR_STORE if l["price"] <= budget]
    print(f"   Total: {len(_VECTOR_STORE)} | Budget: {len(affordable)} | "
          f"Perfect matches: {len(relevant_ids)} [{time.time()-t0:.2f}s]")

    # Step 1: HyDE — generate a realistic laptop spec description
    hyp_doc = hyde_tool.invoke({"requirement": user_requirement_string})
    print(f"[HYDE] {hyp_doc[:200]}")

    # Step 2: Embed HyDE doc + raw string, blend (same as production pipeline)
    stored_dim = len(_VECTOR_STORE[0]["full_embedding"]) if _VECTOR_STORE else None
    def _trim(v):
        if stored_dim is None: return v
        return (v + [0.0] * stored_dim)[:stored_dim]

    hyp_emb    = _trim(_embedder.embed_query(hyp_doc))
    raw_emb    = _trim(_embedder.embed_query(user_requirement_string))
    blended    = list((np.array(hyp_emb) + np.array(raw_emb)) / 2)
    bm25_query = hyp_doc + " " + user_requirement_string   # same as production

    # Step 3 & 4: Dense + Sparse search with production queries
    dense_r  = dense_search_tool.invoke({"query_embedding": blended, "budget": budget, "top_k": top_k})
    sparse_r = bm25_search_tool.invoke({"query": bm25_query, "budget": budget, "top_k": top_k})

    # Step 5: RRF fusion
    fused_r = rrf_tool.invoke({"dense_results": dense_r[:top_k], "sparse_results": sparse_r[:top_k]})[:top_k]

    d_ids = {i["laptop"]["id"] for i in dense_r}
    s_ids = {i["laptop"]["id"] for i in sparse_r}
    f_ids = {i["laptop"]["id"] for i in fused_r}

    print(f"[DENSE] {len(d_ids)} results")
    print(f"[BM25] {len(s_ids)} results")

    # Step 6: Precision — strict (all 5 match) and relaxed (>=3 match)
    def _precision(ids: set) -> tuple[float, float]:
        if not ids:
            return 0.0, 0.0
        strict  = ids & relevant_ids
        relaxed = {lid for lid in ids if get_match_score(lid, user_requirements) >= 3}
        return len(strict) / len(ids) * 100, len(relaxed) / len(ids) * 100

    dsp, drp = _precision(d_ids)
    ssp, srp = _precision(s_ids)
    fsp, frp = _precision(f_ids)

    winner_names = {"semantic": "SEMANTIC (DENSE)", "keyword": "KEYWORD (SPARSE)", "rrf_fused": "RRF FUSED"}
    winner = max({"semantic": dsp, "keyword": ssp, "rrf_fused": fsp}.items(), key=lambda x: x[1])[0]
    print(f"   Semantic {dsp:.1f}% | Keyword {ssp:.1f}% | RRF {fsp:.1f}%  →  WINNER: {winner_names[winner]}")
    print(f"   Total eval time: {time.time()-t0:.2f}s")

    return {
        "semantic_strict_precision":   dsp, "semantic_relaxed_precision":  drp,
        "keyword_strict_precision":    ssp, "keyword_relaxed_precision":   srp,
        "rrf_fused_strict_precision":  fsp, "rrf_fused_relaxed_precision": frp,
        "winner": winner, "winner_name": winner_names[winner],
        "relevant_count":     len(relevant_ids),
        "semantic_retrieved": len(d_ids),
        "keyword_retrieved":  len(s_ids),
        "fused_retrieved":    len(f_ids),
    }


# =============================================================================
# LANGGRAPH NODES
# =============================================================================

def node_moderation(state: AssistantState) -> AssistantState:
    """Node: check user input for unsafe content."""
    result = moderation_tool.invoke({"text": state["user_input"]})
    return {**state, "moderation_result": result}


def node_gather(state: AssistantState) -> AssistantState:
    """Node: continue requirement-gathering conversation."""
    msgs = [SystemMessage(content=_gather_system)] + state["messages"]
    chain = _llm_chat | StrOutputParser()
    reply = chain.invoke(msgs)
    print(f"\n[GATHER] {reply}")
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=reply)],
    }


def node_intent_check(state: AssistantState) -> AssistantState:
    """Node: check if the last assistant reply captured all requirements."""
    last_ai = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), ""
    )
    confirmed = intent_confirmation_tool.invoke({"assistant_reply": last_ai})
    return {**state, "intent_confirmed": confirmed}


def node_extract_requirements(state: AssistantState) -> AssistantState:
    """Node: extract requirement string + structured dict from last AI message."""
    last_ai = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), ""
    )
    req_string   = requirement_string_tool.invoke({"text": last_ai})
    requirements = extraction_tool.invoke({"text": req_string})
    return {**state, "requirement_string": req_string, "user_requirements": requirements}


def node_search(state: AssistantState) -> AssistantState:
    """Node: run full hybrid search pipeline (HyDE + FAISS + BM25 + RRF)."""
    candidates  = _hybrid_search_pipeline(state["requirement_string"], state["user_requirements"])
    top_3_json  = score_and_rank_tool.invoke({"candidates": candidates, "user_requirements": state["user_requirements"]})
    validated   = validation_tool.invoke({"laptop_json": top_3_json})
    return {**state, "top_3_laptops": top_3_json, "validated_laptops": validated}


def node_recommend(state: AssistantState) -> AssistantState:
    """Node: generate recommendation summary from validated laptops."""
    if not state["validated_laptops"]:
        reply = "Sorry, no laptops matched your requirements. Please try different criteria."
        return {
            **state,
            "recommendation": reply,
            "messages": state["messages"] + [AIMessage(content=reply)],
            "phase": "end",
        }

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", _reco_system),
            ("human",  "Please summarize these laptop recommendations for me."),
        ])
        | _llm_reco
        | StrOutputParser()
    )
    reply = chain.invoke({"products": state["validated_laptops"]})
    print(f"\n[RECOMMEND] {reply}")
    return {
        **state,
        "recommendation": reply,
        "messages": state["messages"] + [AIMessage(content=reply)],
        "phase": "followup",
    }


def node_followup(state: AssistantState) -> AssistantState:
    """Node: answer follow-up questions about recommended laptops."""
    system = _reco_system.format(products=state["validated_laptops"])
    msgs   = [SystemMessage(content=system)] + state["messages"]
    chain  = _llm_reco | StrOutputParser()
    reply  = chain.invoke(msgs)
    print(f"\n[FOLLOWUP] {reply}")
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=reply)],
    }


def node_flagged(state: AssistantState) -> AssistantState:
    """Node: handle flagged content — end conversation."""
    return {**state, "phase": "end"}


# =============================================================================
# LANGGRAPH EDGES (routing logic)
# =============================================================================

def route_after_moderation(state: AssistantState) -> str:
    if state["moderation_result"] == "flagged":
        return "flagged"
    return state.get("phase", "gather")


def route_after_intent(state: AssistantState) -> str:
    if state["intent_confirmed"]:
        return "extract_requirements"
    return "gather"


def route_after_search(state: AssistantState) -> str:
    return "recommend"


# =============================================================================
# BUILD LANGGRAPH
# =============================================================================
def build_graph() -> StateGraph:
    graph = StateGraph(AssistantState)

    # Add nodes
    graph.add_node("moderation",          node_moderation)
    graph.add_node("gather",              node_gather)
    graph.add_node("intent_check",        node_intent_check)
    graph.add_node("extract_requirements",node_extract_requirements)
    graph.add_node("search",              node_search)
    graph.add_node("recommend",           node_recommend)
    graph.add_node("followup",            node_followup)
    graph.add_node("flagged",             node_flagged)

    # Entry point
    graph.set_entry_point("moderation")

    # Edges
    graph.add_conditional_edges(
        "moderation",
        route_after_moderation,
        {
            "flagged": "flagged",
            "gather":  "gather",
            "followup":"followup",
        }
    )
    graph.add_edge("gather",               "intent_check")
    graph.add_conditional_edges(
        "intent_check",
        route_after_intent,
        {
            "extract_requirements": "extract_requirements",
            "gather":               END,   # return to Flask; more info needed
        }
    )
    graph.add_edge("extract_requirements", "search")
    graph.add_edge("search",               "recommend")
    graph.add_edge("recommend",            END)
    graph.add_edge("followup",             END)
    graph.add_edge("flagged",              END)

    return graph.compile()


# Compiled graph — import and use in app.py
laptop_graph = build_graph()


# =============================================================================
# OLLAMA RETRY HELPER
# =============================================================================
def _invoke_with_retry(chain, msgs, retries: int = 6, delay: float = 5.0) -> str:
    """
    Invoke a LangChain chain with retry logic to handle Ollama's empty-stream
    error that occurs when the model is first loaded into memory.
    Raises the last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = chain.invoke(msgs)
            if result and result.strip():
                return result
            raise ValueError("Ollama returned empty content.")
        except (ValueError, Exception) as e:
            last_exc = e
            if attempt < retries:
                print(f"[OLLAMA RETRY] Attempt {attempt}/{retries} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"[OLLAMA RETRY] All {retries} attempts failed.")
    raise last_exc


def _call_ollama_rest(messages: list, max_tokens: int = MAX_RECO) -> str:
    """
    Call Ollama /api/chat on CPU only (num_gpu=0), stream=False.
    Raises ValueError if all attempts fail.
    """
    for attempt in range(1, 7):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0,
                        "num_gpu": 0,   # CPU only — no VRAM used
                    },
                },
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "").strip()
            if not text:
                raise ValueError("Empty response from Ollama")
            return text
        except Exception as e:
            print(f"[CPU] Attempt {attempt}/6 failed: {e}. Retrying in 5s...")
            time.sleep(5)
    raise ValueError("All CPU attempts failed.")


def warm_up_ollama() -> None:
    """No-op: warm-up not needed for CPU-only inference."""
    print("[CPU MODE] Skipping warm-up — running entirely on CPU.")


# =============================================================================
# PUBLIC API (called from app.py — same signatures as before)
# =============================================================================

def initialize_conversation() -> list[dict]:
    return []   # LangGraph manages state; no manual system message list needed


def initialize_conv_reco(products) -> list[dict]:
    """
    Build the initial conversation for the recommendation phase.
    Injects a system message with the validated laptop data and a
    user turn asking for a summary — so get_chat_model_completions_reco
    always receives a non-empty messages list.
    """
    if isinstance(products, list):
        products_str = json.dumps(products, indent=2)
    else:
        products_str = str(products)

    system_msg = _reco_system.format(products=products_str)
    return [
        {"role": "system",    "content": system_msg},
        {"role": "user",      "content": "Please summarize these laptop recommendations for me."},
    ]


def get_chat_model_completions(conversation: list) -> str:
    """Gather requirements using CPU-only Ollama inference."""
    rest_msgs = [{"role": "system", "content": _gather_system}]
    for m in conversation:
        role = m.get("role", "")
        if role in ("user", "assistant"):
            rest_msgs.append({"role": role, "content": m.get("content", "")})
    return _call_ollama_rest(rest_msgs, max_tokens=MAX_CHAT)


def get_chat_model_completions_reco(conversation: list) -> str:
    """Generate laptop recommendations using CPU-only Ollama inference."""
    rest_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in conversation
        if m.get("role") in ("system", "user", "assistant")
    ]
    return _call_ollama_rest(rest_msgs, max_tokens=MAX_RECO)


def moderation_check(text: str) -> str:
    result = moderation_tool.invoke({"text": text})
    return "Flagged" if result == "flagged" else "Not Flagged"


def intent_confirmation_layer(reply: str) -> str:
    return "Yes" if intent_confirmation_tool.invoke({"assistant_reply": reply}) else "No"


def get_user_requirement_string(reply: str) -> str:
    return requirement_string_tool.invoke({"text": reply})


def get_chat_completions_func_calling(text: str, include_budget: bool) -> dict:
    result = extraction_tool.invoke({"text": text})
    if not include_budget:
        result["Budget"] = 0
    return result


def compare_laptops_with_user(user_requirements: dict, user_requirement_string: str = "") -> str:
    if int(user_requirements.get("Budget", 0)) <= 0:
        user_requirements["Budget"] = 10_000_000
    candidates = _hybrid_search_pipeline(user_requirement_string, user_requirements, top_k=20)
    return score_and_rank_tool.invoke({"candidates": candidates, "user_requirements": user_requirements})


def recommendation_validation(laptop_json: str) -> list:
    return validation_tool.invoke({"laptop_json": laptop_json})


def extract_laptop_features(description: str) -> dict:
    """O(1) cache read — all laptops pre-classified at startup."""
    return _FEATURE_CACHE.get(description, feature_extraction_tool.invoke({"description": description}))


def extract_user_info(GPU_intensity, Display_quality, Portability, Multitasking, Processing_speed, Budget) -> dict:
    return {
        "GPU intensity":    GPU_intensity,
        "Display quality":  Display_quality,
        "Portability":      Portability,
        "Multitasking":     Multitasking,
        "Processing speed": Processing_speed,
        "Budget":           Budget,
    }


def _dicts_to_lc(messages: list[dict]) -> list[BaseMessage]:
    out = []
    for m in messages:
        role, content = m.get("role"), m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


# =============================================================================
# CONTEXT-BASED CHUNKING (used during index building — unchanged)
# =============================================================================
CHUNK_TOPICS = {
    "GPU":          ["gpu", "graphics", "rtx", "gtx", "radeon", "vram", "nvidia", "amd gpu", "intel arc", "uhd graphics"],
    "Display":      ["display", "screen", "resolution", "fhd", "4k", "retina", "oled", "ips", "nits", "hz", "refresh"],
    "Portability":  ["weight", "kg", "portable", "thin", "light", "slim", "compact", "battery"],
    "Multitasking": ["ram", "gb ram", "memory", "multitask"],
    "Processing":   ["cpu", "processor", "core i", "ryzen", "intel", "amd", "ghz", "m1", "m2", "m3"],
}


def context_based_chunking(description: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"[.,;]\s*", description) if s.strip()]
    buckets: dict[str, list[str]] = {t: [] for t in CHUNK_TOPICS}
    misc: list[str] = []
    for sentence in sentences:
        lower, matched = sentence.lower(), False
        for topic, keywords in CHUNK_TOPICS.items():
            if any(kw in lower for kw in keywords):
                buckets[topic].append(sentence)
                matched = True
                break
        if not matched:
            misc.append(sentence)
    chunks = [f"{t} info: " + ". ".join(s) for t, s in buckets.items() if s]
    if misc:
        chunks.append("General info: " + ". ".join(misc))
    chunks.append(description)
    return chunks


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    build_vector_store("data/index.faiss", "data/index.pkl")

    # Test the full graph with a sample user message
    initial_state: AssistantState = {
        "messages":           [HumanMessage(content="Hi, I need a laptop for gaming")],
        "user_input":         "Hi, I need a laptop for gaming",
        "moderation_result":  "",
        "intent_confirmed":   False,
        "requirement_string": "",
        "user_requirements":  {},
        "top_3_laptops":      "",
        "validated_laptops":  [],
        "recommendation":     "",
        "phase":              "gather",
    }
    result = laptop_graph.invoke(initial_state)
    print("\n[FINAL STATE]")
    for m in result["messages"]:
        print(f"  {type(m).__name__}: {m.content[:100]}")