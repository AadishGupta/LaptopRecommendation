"""
agent_functions.py - Laptop Shopping Assistant (LangGraph Multi-Agent + Orchestrator)
Updated to use Llama 3.1 via local generate API
Full implementation with all agents - 2500+ lines
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import logging
import datetime
import random
import uuid
from collections import defaultdict
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Any, Union

import numpy as np
try:
    import torch
    from sentence_transformers import CrossEncoder
except ImportError:  # Keep the existing retriever usable in minimal installs.
    torch = None
    CrossEncoder = None
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, Filter, FieldCondition, Range, VectorParams, PointStruct, MatchValue
)
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict

from langchain_ollama import OllamaLLM

import kg_rag
import live_pricing

# PDF generation
from reportlab.lib.pagesizes import A4, letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def clean_llm_response(response: str) -> str:
    """
    Clean an LLM response by stripping any <think>...</think> block and
    leftover "Thinking..." artifacts.

    Kept as a cheap safety net even though llama3.1 (the current model)
    doesn't emit these — if the model is ever swapped for a reasoning model
    again, every call site that already calls this stays correct with no
    further changes needed.
    """
    if not response:
        return ""
    # Remove closed <think>...</think> blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # Remove a truncated/unclosed <think> block (ran out of tokens mid-thought)
    response = re.sub(r"<think>.*$", "", response, flags=re.DOTALL)
    response = re.sub(r"Thinking\.\.\.\s*", "", response)
    return response.strip()


# =============================================================================
# CONFIGURATION - Llama 3.1 Local
# =============================================================================

# Model Configuration
OLLAMA_MODEL = "llama3.1:latest"   # single model for everything — no reasoning pass, no <think> overhead
EMBEDDING_MODEL = "nomic-embed-text"
CTX_WINDOW = 4096

MAX_MODERATION = 1024
MAX_INTENT = 1024
MAX_REQ_STRING = 400
MAX_FUNC_CALLING = 300
MAX_HYDE = 400
MAX_CHAT = 1500
MAX_RECO = 1200
MAX_COMPARE = 1200
MAX_ORCHESTRATOR = 800
MAX_UPGRADE = 1200
MAX_SIDE_COMPARE = 1400
MAX_PDF_DESCRIPTION = 1400

# GPU Optimization Settings
OLLAMA_NUM_GPU = 1
OLLAMA_NUM_THREAD = 4


QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "laptops_chunked")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


CASE_MEMORY_COLLECTION = "case_memory"
CASE_MEMORY_TOP_K = 4          
CASE_MEMORY_MIN_SCORE = 0.55  
CASE_MEMORY_MIN_WRITE_SCORE = 0.65
CASE_MEMORY_DUPLICATE_SCORE = 0.94
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_BATCH_SIZE = 16
RERANK_CANDIDATES = 30
RERANK_TOP_K = 10


class _CrossEncoderSingleton:
    """Process-wide, best-effort reranker; never blocks startup on failure."""
    _model = None
    _failed = False

    @classmethod
    def get(cls):
        if cls._model is not None or cls._failed:
            return cls._model
        if CrossEncoder is None:
            cls._failed = True
            logger.warning("Cross-encoder unavailable; using RRF ranking only")
            return None
        try:
            device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
            cls._model = CrossEncoder(RERANK_MODEL, device=device, max_length=256)
            logger.info("Loaded cross-encoder reranker on %s", device)
        except Exception as exc:
            cls._failed = True
            logger.warning("Cross-encoder unavailable; using RRF ranking only: %s", exc)
        return cls._model

# PDF Configuration
PDF_OUTPUT_DIR = "static/reports"

# Currency Conversion
USD_TO_INR = 85

# Feature cache path
_FEATURE_CACHE_PATH = "laptop_features_cache.json"
_KG_CACHE_PATH = "kg_cache.gpickle.json"


RAG_EVAL_LOG_PATH = "rag_eval_log.jsonl"


def _log_rag_turn(
    pipeline: str,
    question: str,
    contexts: List[str],
    answer: str,
    requirements: Optional[dict] = None,
    recommended: Optional[dict] = None,
    ranked: Optional[List[dict]] = None,
) -> None:
    """
    Append one completed RAG turn to a plain JSONL log for later, fully
    offline RAGAS scoring (see ragas_eval.py / rag_evaluation.py). Pure
    stdlib (json + file append) — no ragas/datasets import here, so this
    never pulls RAGAS into the live app's import graph. Never raises — a
    failed log write should not break the user-facing turn.

    requirements/recommended/ranked are optional and additive: they carry
    the extracted requirement dict, the top-ranked laptop for this turn,
    and (when available) the full top-K candidate list in ranked order.
    rag_evaluation.py's log-scoring path needs requirements+recommended to
    compute ground-truth correctness, and additionally needs `ranked` (the
    top-K list, each item minimally {"name","price","description"}) to
    compute real Recall@K/Precision@K/MRR/nDCG@K — without `ranked` those
    retrieval metrics fall back to scoring against a single-item list
    (just `recommended`), which trivializes them.
    """
    try:
        with open(RAG_EVAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "pipeline": pipeline,
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "requirements": requirements,
                "recommended": recommended,
                "ranked": ranked,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }) + "\n")
    except Exception as e:
        logger.warning(f"   ⚠️ [RAG LOG] could not write turn to {RAG_EVAL_LOG_PATH}: {e}")


# =============================================================================
# LLM FACTORY
# =============================================================================

class LLMFactory:
    """Factory for creating LLM instances. Single model (llama3.1) for
    everything — no reasoning pass, so no <think> token overhead on any
    call, and only one model needs to stay resident in VRAM."""

    @classmethod
    def get_llm(
        cls,
        max_tokens: int,
        temperature: float = 0.3,
        format_json: bool = False,
        use_reasoning: Optional[bool] = None,  # kept for call-site compatibility; no longer changes model
    ) -> OllamaLLM:
        """
        Get an LLM instance using the Ollama generate API (llama3.1 for all tasks).
        """
        if format_json:
            system_prompt = "You are a helpful assistant. Respond ONLY with valid JSON. Do not include thinking tags."
        else:
            system_prompt = "You are a helpful assistant. Keep responses clear and concise. Do not include thinking tags."

        return OllamaLLM(
            model=OLLAMA_MODEL,
            temperature=temperature,
            num_predict=max_tokens,
            num_ctx=CTX_WINDOW,
            system=system_prompt,
            format="json" if format_json else "",
            keep_alive="30m",
        )
    
    @classmethod
    def get_embeddings(cls) -> OllamaEmbeddings:
        """Get the embedding model instance with GPU support."""
        return OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            num_gpu=OLLAMA_NUM_GPU,
        )


_embedder = LLMFactory.get_embeddings()


# =============================================================================
# QDRANT CLIENT
# =============================================================================

class QdrantClientManager:
    _client: Optional[QdrantClient] = None
    _initialized = False

    @classmethod
    def get_client(cls) -> Optional[QdrantClient]:
        if not cls._initialized:
            try:
                cls._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                cls._client.get_collections()
                logger.info("✅ Connected to Qdrant")
            except Exception as e:
                logger.error(f"⚠️ Qdrant unavailable: {e}")
                cls._client = None
            cls._initialized = True
        return cls._client


def get_qdrant_client() -> Optional[QdrantClient]:
    return QdrantClientManager.get_client()


# =============================================================================
# CASE MEMORY (Memento-style episodic case bank)
# =============================================================================
#
# Non-parametric CBR, following Memento §4.2 (Eq. 12/13): every completed
# turn is written as a case (query, what we recommended, how well it scored),
# and future similar queries retrieve the top-K most similar past cases by
# cosine similarity to use as extra context. No training required — this is
# the "cheap, ships-today" half of Memento; the learned-Q parametric variant
# is a natural follow-up once there's enough logged volume.

_CASE_EMBED_DIM = 384  # matches the trim used for the laptop search collections
_CASE_COLLECTION_READY = False


def _trim_embedding(vec: List[float], dim: int = _CASE_EMBED_DIM) -> List[float]:
    """Pad/truncate an embedding to a fixed size so it fits the collection's
    vector config, regardless of what the underlying embed model returns."""
    return (list(vec) + [0.0] * dim)[:dim]


def _ensure_case_memory_collection(client: QdrantClient) -> bool:
    """Create the case_memory collection on first use. Cheap to call
    repeatedly — short-circuits once it's confirmed to exist."""
    global _CASE_COLLECTION_READY
    if _CASE_COLLECTION_READY:
        return True
    try:
        existing = {c.name for c in client.get_collections().collections}
        if CASE_MEMORY_COLLECTION not in existing:
            client.create_collection(
                collection_name=CASE_MEMORY_COLLECTION,
                vectors_config=VectorParams(size=_CASE_EMBED_DIM, distance=Distance.COSINE),
            )
            logger.info(f"🧠 [CASE MEMORY] created collection '{CASE_MEMORY_COLLECTION}'")
        _CASE_COLLECTION_READY = True
        return True
    except Exception as e:
        logger.warning(f"   [CASE MEMORY] could not ensure collection: {e}")
        return False


def write_case(
    pipeline: str,
    query_text: str,
    requirements: dict,
    summary: dict,
    correctness: Optional[float] = None,
    action: Optional[dict] = None,
    result: Optional[dict] = None,
    allow_low_quality: bool = False,
) -> None:
    """Append a completed turn to the case bank.

    CHANGES FROM ORIGINAL:

    1. QUALITY GATE (no-filter problem):
       Cases whose correctness is KNOWN and below CASE_MEMORY_MIN_WRITE_SCORE are
       silently dropped. Writing them would mean the retriever surfaces bad
       examples as if they were positive demonstrations. Cases without a score
       (correctness=None, live app without ground truth) are always written —
       they contribute as weak-prior "unvalidated" context and are ranked below
       scored ones at read time.

    2. REQUIREMENTS-AXIS EMBEDDING (wrong-axis problem):
       The embedding key is now _requirements_to_canonical_string(requirements)
       — a normalized tier profile string — instead of the raw natural-language
       query_text. Two queries that extracted the same tiers but used different
       phrasing ("gaming rig under 85k" vs "ML workstation 85000 rupees") now
       embed close together and retrieve the same past cases. query_text is still
       stored in the payload for display purposes.

    3. CORRECTNESS IN PAYLOAD:
       The verified score (or None) is stored so read-time retrieval can
       quality-rank results and so _format_case_context can show it to the LLM.

    Never raises — a failed write must not break the user-facing turn.
    """
    # Store only verified, high-quality cases. An unscored answer is useful
    # conversationally but must not become training evidence for future turns.
    if correctness is None:
        logger.info("ðŸ§  [CASE MEMORY] skipping unverified case")
        return
    if correctness < CASE_MEMORY_MIN_WRITE_SCORE and not allow_low_quality:
        logger.info(
            f"🧠 [CASE MEMORY] skipping write — "
            f"correctness={correctness:.3f} < threshold={CASE_MEMORY_MIN_WRITE_SCORE}"
        )
        return

    client = get_qdrant_client()
    if not client or not query_text.strip():
        return
    if not _ensure_case_memory_collection(client):
        return

    try:
        # Embed on requirements tier profile (not surface query text) so
        # retrieval similarity reflects what the user actually NEEDS, not how
        # they happened to phrase the request.
        embed_text = (
            _requirements_to_canonical_string(requirements)
            if requirements
            else query_text
        )
        vec = _trim_embedding(_embedder.embed_query(embed_text))
        # Selective retention: a near-identical requirements profile is not a
        # new case. Keep the existing higher-quality/diverse example instead.
        existing = client.query_points(
            collection_name=CASE_MEMORY_COLLECTION, query=vec, limit=1,
            query_filter=Filter(
                must=[FieldCondition(key="pipeline", match=MatchValue(value=pipeline))]
            ),
            with_payload=True,
        ).points
        case_id = str(uuid.uuid4())
        if existing and existing[0].score >= CASE_MEMORY_DUPLICATE_SCORE:
            old_quality = existing[0].payload.get("correctness")
            # A verified evaluation result is more valuable than the live
            # turn's unvalidated copy. Keep it so the 200-question pass can
            # progressively replace weak examples with quality-scored cases.
            if not (correctness is not None and old_quality is None) and (
                correctness is None or (old_quality is not None and old_quality >= correctness)
            ):
                logger.info("ðŸ§  [CASE MEMORY] skipping near-duplicate case")
                return
            # Upgrade the live/unvalidated copy in place instead of creating
            # a second semantically identical case.
            case_id = existing[0].id
        score_tag = f"correctness={correctness:.3f}" if correctness is not None else "correctness=unvalidated"
        client.upsert(
            collection_name=CASE_MEMORY_COLLECTION,
            points=[PointStruct(
                id=case_id,
                vector=vec,
                payload={
                    "pipeline": pipeline,
                    "query_text": query_text,
                    "requirements": requirements,
                    "summary": summary,
                    "correctness": correctness,          # NEW: stored for read-time ranking
                    # Retrieval-planner cases carry the complete strategy and
                    # outcome. Existing recommendation cases leave these empty.
                    "action": action or {},
                    "result": result or {},
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                },
            )],
        )
        logger.info(
            f"🧠 [CASE MEMORY] wrote case pipeline='{pipeline}' "
            f"{score_tag} q='{query_text[:60]}'"
        )
    except Exception as e:
        logger.warning(f"   [CASE MEMORY] write failed: {e}")


def retrieve_similar_cases(
    query_text: str,
    top_k: int = CASE_MEMORY_TOP_K,
    pipeline: Optional[str] = None,
    requirements: Optional[dict] = None,
) -> List[dict]:
    """Memento non-parametric Read (Eq. 13): top-K cases by cosine similarity.

    CHANGES FROM ORIGINAL:

    1. REQUIREMENTS-AXIS EMBEDDING (wrong-axis problem):
       When `requirements` is provided, the query vector is built from
       _requirements_to_canonical_string(requirements) — the same canonical
       tier profile used at write time — instead of the natural-language
       query_text. This ensures retrieval similarity reflects requirement-tier
       proximity, not surface-text proximity. Pass the state["requirements"]
       dict here; query_text is still accepted as a text fallback.

    2. QUALITY FILTERING (no-filter problem):
       - Cases with a stored correctness below CASE_MEMORY_MIN_WRITE_SCORE are
         excluded (they should never have been written, but this guards against
         schema migrations or direct DB inserts).
       - Fetches top_k * 3 candidates first, then quality-filters, then caps
         at top_k so quality filtering doesn't leave the caller with fewer
         results than expected when plenty of good cases exist.

    3. QUALITY-AWARE RANKING:
       After filtering, results are re-ranked by a combined signal:
         rank_score = vector_similarity * quality_weight
       where quality_weight = correctness if scored, 0.55 if unvalidated.
       This means a good case (correctness=0.9, sim=0.70) will outrank a
       mediocre-quality case (correctness=0.45, sim=0.85 — but the latter
       was excluded anyway) and also outrank an unvalidated case (sim=0.85,
       effective_quality=0.55 → combined=0.47 vs 0.63).

    Returns [] on any failure or when nothing beats CASE_MEMORY_MIN_SCORE.
    """
    client = get_qdrant_client()
    if not client or not query_text.strip():
        return []
    if not _ensure_case_memory_collection(client):
        return []

    try:
        # Embed on the requirements axis if structured requirements are available.
        embed_text = (
            _requirements_to_canonical_string(requirements)
            if requirements
            else query_text
        )
        vec = _trim_embedding(_embedder.embed_query(embed_text))

        query_filter = None
        if pipeline:
            query_filter = Filter(
                must=[FieldCondition(key="pipeline", match=MatchValue(value=pipeline))]
            )

        # Fetch more candidates than needed so quality filtering has headroom.
        results = client.query_points(
            collection_name=CASE_MEMORY_COLLECTION,
            query=vec,
            query_filter=query_filter,
            limit=top_k * 3,
            with_payload=True,
        ).points

        cases = []
        for r in results:
            if r.score < CASE_MEMORY_MIN_SCORE:
                continue
            stored_correctness = r.payload.get("correctness")  # float or None
            # Exclude cases we know were wrong (shouldn't exist post-quality-gate,
            # but guard against legacy data / direct DB writes).
            if (
                stored_correctness is not None
                and stored_correctness < CASE_MEMORY_MIN_WRITE_SCORE
                and pipeline != RETRIEVAL_PLANNER_PIPELINE
            ):
                continue
            # Rank by combined similarity × quality weight.
            # Legacy unvalidated points can remain on disk, but they must
            # never influence CBR adaptation after this quality-only change.
            if stored_correctness is None:
                continue
            quality_weight = stored_correctness
            cases.append({
                "score": r.score,
                "_rank_score": r.score * quality_weight,   # used for sorting only
                "query_text": r.payload.get("query_text", ""),
                "requirements": r.payload.get("requirements", {}),
                "summary": r.payload.get("summary", {}),
                "correctness": stored_correctness,
                "action": r.payload.get("action", {}),
                "result": r.payload.get("result", {}),
                "timestamp": r.payload.get("timestamp", ""),
            })

        # Re-rank by combined signal: high-quality + similar cases first.
        cases.sort(key=lambda c: c["_rank_score"], reverse=True)
        cases = cases[:top_k]
        # Strip internal sort key before returning.
        for c in cases:
            c.pop("_rank_score", None)

        if cases:
            scored_n = sum(1 for c in cases if c["correctness"] is not None)
            logger.info(
                f"🧠 [CASE MEMORY] retrieved {len(cases)} case(s) "
                f"({scored_n} quality-scored) for '{query_text[:60]}'"
            )
        return cases
    except Exception as e:
        logger.warning(f"   [CASE MEMORY] retrieval failed: {e}")
        return []


RETRIEVAL_PLANNER_PIPELINE = "retrieval_planner"
DEFAULT_RETRIEVAL_ACTION = {
    "kg_weight": 0.30,
    "hyde_enabled": True,
    "dense_top_k": 30,
    "sparse_top_k": 30,
    "reranker_enabled": True,
    "reranker_top_k": RERANK_TOP_K,
}


def _normalise_retrieval_action(action: Optional[dict]) -> dict:
    """Return a safe, complete retrieval strategy.

    New parameters can be added to DEFAULT_RETRIEVAL_ACTION without changing
    the planner policy; they become part of the stored action automatically.
    """
    merged = dict(DEFAULT_RETRIEVAL_ACTION)
    if isinstance(action, dict):
        merged.update({key: value for key, value in action.items() if value is not None})
    merged["kg_weight"] = min(1.0, max(0.0, float(merged["kg_weight"])))
    merged["hyde_enabled"] = bool(merged["hyde_enabled"])
    merged["reranker_enabled"] = bool(merged["reranker_enabled"])
    merged["dense_top_k"] = max(1, int(merged["dense_top_k"]))
    merged["sparse_top_k"] = max(1, int(merged["sparse_top_k"]))
    merged["reranker_top_k"] = max(1, int(merged["reranker_top_k"]))
    return merged


class RetrievalPlanner:
    """Memento-inspired, non-parametric policy over evaluated retrieval cases.

    A case is (state, action, result, reward). The state is the canonical
    requirement profile; cosine similarity retrieves analogous states. For
    each observed action, the policy estimates expected reward with:
        sum(similarity * correctness) / sum(similarity).
    """

    def retrieve_similar_cases(self, query_text: str, requirements: dict) -> List[dict]:
        return retrieve_similar_cases(
            query_text,
            top_k=CASE_MEMORY_TOP_K,
            pipeline=RETRIEVAL_PLANNER_PIPELINE,
            requirements=requirements,
        )

    def select_best_action(self, cases: List[dict]) -> Tuple[dict, dict]:
        grouped: Dict[str, dict] = {}
        for case in cases:
            action = case.get("action") or {}
            reward = case.get("correctness")
            similarity = float(case.get("score", 0.0) or 0.0)
            if not action or reward is None or similarity <= 0:
                continue
            normalised = _normalise_retrieval_action(action)
            key = json.dumps(normalised, sort_keys=True, separators=(",", ":"))
            bucket = grouped.setdefault(key, {"action": normalised, "numerator": 0.0, "denominator": 0.0, "case_count": 0})
            bucket["numerator"] += similarity * float(reward)
            bucket["denominator"] += similarity
            bucket["case_count"] += 1

        if not grouped:
            return dict(DEFAULT_RETRIEVAL_ACTION), {
                "policy": "default_no_similar_evaluated_cases",
                "candidate_actions": [],
            }

        candidates = []
        for bucket in grouped.values():
            weighted_reward = bucket["numerator"] / bucket["denominator"]
            candidates.append({
                "action": bucket["action"],
                "weighted_reward": round(weighted_reward, 6),
                "case_count": bucket["case_count"],
                "similarity_mass": round(bucket["denominator"], 6),
            })
        candidates.sort(key=lambda item: (item["weighted_reward"], item["similarity_mass"]), reverse=True)
        return candidates[0]["action"], {
            "policy": "similarity_weighted_reward",
            "selected_weighted_reward": candidates[0]["weighted_reward"],
            "candidate_actions": candidates,
        }

    def plan(self, query_text: str, requirements: dict) -> Tuple[dict, List[dict], dict]:
        cases = self.retrieve_similar_cases(query_text, requirements)
        action, policy = self.select_best_action(cases)
        logger.info(
            "[RETRIEVAL PLANNER] policy=%s cases=%d kg_weight=%.2f hyde=%s reranker=%s",
            policy["policy"], len(cases), action["kg_weight"], action["hyde_enabled"], action["reranker_enabled"],
        )
        return action, cases, policy


_retrieval_planner = RetrievalPlanner()


def store_retrieval_case(state: LaptopState, correctness: float, reward_source: str = "ground_truth") -> None:
    """Retain an evaluated retrieval episode for future case-based planning."""
    requirements = state.get("requirements", {}) or {}
    action = state.get("retrieval_action", {}) or {}
    if not requirements or not action:
        return
    ranked = state.get("ranked_laptops", []) or []
    search_results = state.get("search_results", []) or []
    result = {
        "retrieved_document_ids": [str(item.get("laptop", {}).get("id")) for item in search_results if item.get("laptop", {}).get("id") is not None],
        "retrieved_laptop_ids": [str(item.get("id")) for item in ranked if item.get("id") is not None],
        "final_answer": state.get("comparison_analysis", ""),
        "reward_source": reward_source,
    }
    query_text = state.get("requirement_string", "") or json.dumps(requirements, sort_keys=True)
    write_case(
        pipeline=RETRIEVAL_PLANNER_PIPELINE,
        query_text=query_text,
        requirements=requirements,
        summary={
            "best_overall": (state.get("best_overall") or {}).get("name", ""),
            "top_3": [lap.get("name", "") for lap in ranked[:3]],
            "reward_source": reward_source,
        },
        correctness=float(correctness),
        action=_normalise_retrieval_action(action),
        result=result,
        allow_low_quality=True,
    )


def compute_proxy_reward(state: LaptopState) -> float:
    """Estimate live retrieval quality when no ground truth is available.

    This is deliberately based on observable constraints, not LLM self-judging:
    feature-tier compliance (60%), budget compliance (20%), enough viable
    candidates (10%), and explicit GPU/refresh constraints (10%). Ground-truth
    evaluation replaces this signal during offline policy search.
    """
    requirements = state.get("requirements", {}) or {}
    ranked = state.get("ranked_laptops", []) or []
    if not ranked:
        return 0.0
    best = ranked[0]
    req_items = [(key, value) for key, value in requirements.items() if key in _KW_RULES]
    feature_score = float(best.get("score", 0) or 0) / max(len(req_items), 1)
    budget = float(requirements.get("Budget", 0) or 0)
    budget_score = 1.0 if not budget or float(best.get("price", 0) or 0) <= budget else 0.0
    coverage_score = min(len(ranked), 3) / 3.0
    text = f"{best.get('name', '')} {best.get('description', '')}".lower()
    constraints = []
    required_gpu = str(requirements.get("Required GPU", "") or "").lower()
    if required_gpu:
        constraints.append(1.0 if required_gpu in text else 0.0)
    minimum_refresh = int(requirements.get("Minimum refresh rate", 0) or 0)
    if minimum_refresh:
        rates = [int(value) for value in re.findall(r"\b(\d{2,3})\s*hz\b", text, re.IGNORECASE)]
        constraints.append(1.0 if any(rate >= minimum_refresh for rate in rates) else 0.0)
    constraint_score = sum(constraints) / len(constraints) if constraints else 1.0
    return round(0.60 * feature_score + 0.20 * budget_score + 0.10 * coverage_score + 0.10 * constraint_score, 4)


def _requirements_to_canonical_string(requirements: dict) -> str:
    """Convert structured requirements into a normalized tier-profile string
    used as the EMBEDDING KEY for case memory.

    WHY THIS EXISTS (axis problem):
    The original code embedded the natural-language `req_string` ("I need a
    laptop with high GPU intensity, medium portability, budget 85000"). Two
    queries with identical extracted tiers but different phrasing ("gaming rig
    under 85k" vs "ML workstation 85000 rupees") would NOT retrieve the same
    past cases, even though the same retrieval strategy applies to both.

    This function reduces requirements to a short, deterministic string whose
    embedding captures only the tier profile:
        "gpu:high display:medium portability:high multitasking:medium
         processing:high budget_tier:medium"

    Two queries with the same tiers → nearly identical embeddings → the same
    high-quality past cases bubble to the top regardless of how the user phrased
    the original request. This is the axis we actually care about for CBR.
    """
    budget = requirements.get("Budget", 0)
    # Approximate tier mapping. These boundaries don't need to be pixel-perfect —
    # they just need to be consistent between write and read time so the same
    # query always maps to the same embedding. The correctness_scoring module
    # uses catalog-derived boundaries; those can't be called here without
    # importing Qdrant, so we use fixed INR brackets that match the typical
    # catalog distribution (low < ₹50k, medium ₹50k–₹1L, high > ₹1L).
    if budget <= 50_000:
        budget_tier = "low"
    elif budget <= 100_000:
        budget_tier = "medium"
    else:
        budget_tier = "high"

    parts = [
        f"gpu:{requirements.get('GPU intensity', 'medium').lower()}",
        f"display:{requirements.get('Display quality', 'medium').lower()}",
        f"portability:{requirements.get('Portability', 'medium').lower()}",
        f"multitasking:{requirements.get('Multitasking', 'medium').lower()}",
        f"processing:{requirements.get('Processing speed', 'medium').lower()}",
        f"budget_tier:{budget_tier}",
    ]
    return " ".join(parts)


def _format_case_context(cases: List[dict]) -> str:
    """Turn retrieved cases into a quality-annotated prompt block.

    WHAT CHANGED (implicit-usage problem):
    The original code injected cases with no quality signal — the LLM had no
    way to know whether the past recommendation was correct or garbage. Now each
    case is labeled with its verified correctness score so the LLM can treat
    high-scoring cases as positive examples and moderate ones with appropriate
    skepticism. Cases without a score (live app, no ground-truth available) are
    marked as unvalidated rather than pretending they were fine.

    Sections:
      ✓ HIGH-QUALITY  — correctness ≥ 0.75  → use as positive examples
      ~ MODERATE      — 0.55 ≤ correctness < 0.75 → adapt carefully
      ? UNVALIDATED   — correctness is None  → treat as weak prior only
    """
    if not cases:
        return ""

    good     = [c for c in cases if c.get("correctness") is not None and c["correctness"] >= 0.75]
    moderate = [c for c in cases if c.get("correctness") is not None and 0.55 <= c["correctness"] < 0.75]
    unknown  = [c for c in cases if c.get("correctness") is None]

    lines = ["── Similar past requests (quality-annotated) ──"]

    if good:
        lines.append("✓ HIGH-QUALITY — these recommendations were verified correct:")
        for i, c in enumerate(good, 1):
            summ = c.get("summary", {})
            top3 = ", ".join(summ.get("top_3", [])) or summ.get("best_overall", "N/A")
            lines.append(
                f"  {i}. Profile: {c.get('query_text', '')[:180]}\n"
                f"     → Best pick: {summ.get('best_overall', 'N/A')}  "
                f"Also considered: {top3}  [correctness={c['correctness']:.2f}]"
            )

    if moderate:
        lines.append("~ MODERATE — partially correct; adapt to current requirements:")
        for i, c in enumerate(moderate, 1):
            summ = c.get("summary", {})
            lines.append(
                f"  {i}. Profile: {c.get('query_text', '')[:180]}\n"
                f"     → Suggested: {summ.get('best_overall', 'N/A')}  "
                f"[correctness={c['correctness']:.2f} — verify it fits]"
            )

    if unknown:
        lines.append("? UNVALIDATED — no ground-truth score; treat as weak prior only:")
        for i, c in enumerate(unknown, 1):
            summ = c.get("summary", {})
            lines.append(
                f"  {i}. Profile: {c.get('query_text', '')[:180]}\n"
                f"     → Suggested: {summ.get('best_overall', 'N/A')}  [score unknown]"
            )

    lines.append(
        "Prioritize ✓ cases as positive examples. "
        "Current requirements may differ — adapt, never blindly copy."
    )
    return "\n".join(lines)


def _adapt_from_cases(candidates: List[dict], requirements: dict, cases: List[dict]) -> Tuple[List[dict], List[str]]:
    """CBR adaptation: transfer only validated constraint evidence, not prose.

    A prior answer is never injected into generation.  A case can only give a
    small tie-break boost to a current candidate it recommended when its
    structured profile overlaps the current profile.
    """
    if not cases or not candidates:
        return candidates, []
    boosts: Dict[str, float] = defaultdict(float)
    notes: List[str] = []
    keys = tuple(_KW_RULES)
    for case in cases:
        old = case.get("requirements", {})
        overlap = sum(old.get(k) == requirements.get(k) for k in keys) / len(keys)
        quality = case.get("correctness") or 0.55
        if overlap < 0.6 or quality < CASE_MEMORY_MIN_WRITE_SCORE:
            continue
        best = case.get("summary", {}).get("best_overall", "").strip().lower()
        if not best:
            continue
        for candidate in candidates:
            if candidate.get("name", "").strip().lower() == best:
                boosts[str(candidate.get("id"))] += 0.05 * overlap * quality
        notes.append(f"validated prior case overlaps {overlap:.0%}; rechecked against current budget and constraints")
    adapted = [dict(c, case_adaptation_boost=boosts.get(str(c.get("id")), 0.0)) for c in candidates]
    adapted.sort(key=lambda c: (c.get("score", 0), c.get("reranker_score", c.get("rrf_score", 0)) + c["case_adaptation_boost"]), reverse=True)
    return adapted, notes[:2]


# =============================================================================
# FEATURE CACHE
# =============================================================================

_FEATURE_CACHE: Dict[str, dict] = {}
_VS_BUILT = False

def _flex_pattern(kw: str) -> str:
    """Turn a plain keyword phrase into a regex tolerant of whitespace AND
    small filler words a human might naturally add — '32gb ram' also matches
    '32 GB RAM', '32GB of RAM', etc. Splits into digit/letter tokens and
    allows up to 2 filler words between each."""
    tokens = re.findall(r"\d+|[a-z]+", kw.lower())
    return r"\s*(?:\w+\s+){0,2}".join(re.escape(t) for t in tokens)


def _kw_matches(kw: str, text: str) -> bool:
    """Whitespace-tolerant replacement for the old brittle `kw in text` check."""
    return bool(re.search(_flex_pattern(kw), text))


_KW_RULES = {
    "GPU intensity": {
        "high": ["rtx 5090", "rtx 5080", "rtx 5070", "rtx 5060", "rtx 4090", "rtx 4080", "rtx 4070", "rtx 4060", "rx 7900", "rx 8900", "8gb vram", "16gb vram"],
        "medium": ["rtx 3050", "mx550", "gtx 1650", "rx 6600", "iris xe"],
        "low": ["intel uhd", "integrated graphics", "vega 8"],
    },
    "Display quality": {
        "high": ["4k", "oled", "retina", "120hz", "144hz", "2560x1600"],
        "medium": ["fhd", "1920x1080", "ips", "1080p"],
        "low": ["hd+", "1366x768", "tn panel", "720p"],
    },
    "Portability": {
        "high": ["ultrabook", "under 1 kg", "thin and light", "slim"],
        "medium": ["1.5 kg", "1.8 kg", "2.0 kg"],
        "low": ["2.5 kg", "3 kg", "workstation", "17 inch"],
    },
    "Multitasking": {
        "high": ["64gb ram", "32gb ram", "lpddr5x"],
        "medium": ["16gb ram", "lpddr4x"],
        "low": ["8gb ram", "4gb ram", "lpddr4"],
    },
    "Processing speed": {
        "high": ["core i9", "ryzen 9", "m3 pro", "m2 pro"],
        "medium": ["core i7", "ryzen 7", "m3", "m2"],
        "low": ["core i5", "ryzen 5", "core i3", "celeron"],
    },
}


def _classify_one(name: str, description: str) -> dict:
    """Classify a laptop into feature tiers using whitespace-tolerant keyword
    matching against BOTH its structured name/spec string and its
    description. The name usually carries exact specs
    ("...32GB/1TB SSD/8GB RTX 5070..."), while description is often
    LLM-generated prose that paraphrases them ("32 GB of RAM") — checking
    only description, with rigid substring matching, silently misclassified
    real 32GB-RAM laptops as 'medium' multitasking.
    """
    text = f"{name} {description}".lower()
    features = {}
    for feature, tiers in _KW_RULES.items():
        matched = "medium"
        for tier in ("high", "low"):
            if any(_kw_matches(kw, text) for kw in tiers[tier]):
                matched = tier
                break
        features[feature] = matched
    return features


def load_catalog_from_qdrant(collection_name: str = None) -> List[dict]:
    """
    Reconstruct the flat laptop catalog (one record per laptop_id: id, name,
    price, description) directly from Qdrant, instead of a local pkl file.

    `laptops_chunked` (see chunk_qdrant_pytorch.py) stores several points
    (chunks) per laptop, each payload carrying laptop_id/name/price/
    full_description — so we scroll the whole collection and keep only the
    first chunk (chunk_index 0, falling back to the first point seen) per
    laptop_id, using full_description as that laptop's description. This
    keeps the feature cache / knowledge graph always in sync with whatever
    catalog is actually indexed in Qdrant, with nothing to fall out of sync
    on disk.
    """
    collection_name = collection_name or QDRANT_COLLECTION
    client = get_qdrant_client()
    if not client:
        logger.error("   Could not load catalog: no Qdrant client available")
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
        logger.error(f"   Could not load catalog from Qdrant collection '{collection_name}': {e}")
        return []

    laptops = [{k: v for k, v in l.items() if k != "_chunk_index"} for l in by_laptop.values()]
    return laptops


def build_vector_store(collection_name: str = None):
    """
    Build or load the vector store and knowledge graph.

    This app's vector search is pure Qdrant (see QDRANT_COLLECTION /
    get_qdrant_client below) — there is no FAISS index anywhere in the
    live pipeline, and no local pkl catalog either. The laptop catalog used
    for the feature cache and the knowledge graph is reconstructed straight
    from Qdrant's `laptops_chunked` collection (see load_catalog_from_qdrant
    above), the same collection the actual vectors live in — so there's a
    single source of truth and nothing to drift out of sync.
    """
    global _VS_BUILT, _FEATURE_CACHE

    if _VS_BUILT:
        return

    # Load once at startup, rather than paying model construction cost on the
    # first user turn. It remains optional for offline/minimal environments.
    _CrossEncoderSingleton.get()
    collection_name = collection_name or QDRANT_COLLECTION
    logger.info("📚 Initialising vector store …")

    # Load the catalog up front so we can sanity-check every cache/index
    # against it below, instead of trusting on-disk caches blindly.
    laptops = load_catalog_from_qdrant(collection_name)
    logger.info(f"   💻 Catalog (Qdrant): {len(laptops)} laptops from '{collection_name}'")

    # Load feature cache if exists
    if os.path.exists(_FEATURE_CACHE_PATH):
        try:
            with open(_FEATURE_CACHE_PATH) as f:
                _FEATURE_CACHE = json.load(f)
            logger.info(f"   Loaded {len(_FEATURE_CACHE)} cached features")
        except Exception:
            _FEATURE_CACHE = {}

    # Rebuild if empty OR if it disagrees with the current Qdrant catalog — an on-disk
    # cache surviving a catalog swap (e.g. 11k -> 1000 laptops) would
    # otherwise be loaded as-is forever, since "not _FEATURE_CACHE" only
    # catches a missing/empty file, not a stale one.
    cache_matches_catalog = (
        laptops
        and _FEATURE_CACHE
        and all(l.get("description", "") in _FEATURE_CACHE for l in laptops[:20])
    )
    if _FEATURE_CACHE and laptops and len(_FEATURE_CACHE) != len(laptops):
        logger.warning(
            f"   ⚠️ Feature cache has {len(_FEATURE_CACHE)} entries but Qdrant has "
            f"{len(laptops)} laptops — cache looks stale (from a different catalog build)."
        )
    stale_cache = bool(_FEATURE_CACHE) and not cache_matches_catalog
    if not _FEATURE_CACHE or not cache_matches_catalog:
        try:
            _FEATURE_CACHE = {}
            for laptop in laptops:
                desc = laptop.get("description", "")
                _FEATURE_CACHE[desc] = _classify_one(laptop.get("name", ""), desc)
            
            with open(_FEATURE_CACHE_PATH, "w") as f:
                json.dump(_FEATURE_CACHE, f)
            
            logger.info(f"   Built cache for {len(_FEATURE_CACHE)} laptops"
                        f"{' (rebuilt — stale cache detected)' if stale_cache else ''}")
        except Exception as e:
            logger.error(f"Could not build feature cache: {e}")

    # Connect to Qdrant
    client = get_qdrant_client()
    if client:
        try:
            info = client.get_collection(QDRANT_COLLECTION)
            points_count = getattr(info, "points_count", None)
            if points_count is None:
                points_count = getattr(info, "vectors_count", None)  # older qdrant-client versions
            logger.info(f"   Qdrant collection '{QDRANT_COLLECTION}' ready "
                        f"({points_count} points indexed)")
            if points_count is not None and laptops:
                # laptops_chunked holds several points (chunks) per laptop
                # (see chunk_qdrant_pytorch.py), so points_count is expected
                # to be a multiple of len(laptops), NOT equal to it — only
                # flag ratios that indicate a genuinely different catalog:
                # fewer points than laptops (impossible if every laptop
                # produced >=1 chunk), or suspiciously close to a 1:1 ratio
                # (suggests an unchunked collection / wrong catalog size).
                ratio = points_count / len(laptops)
                if points_count < len(laptops):
                    logger.warning(
                        f"   ⚠️ Qdrant has only {points_count} points but {len(laptops)} distinct "
                        f"laptop_ids were found — that's fewer chunks than laptops, which "
                        f"shouldn't happen. Re-run chunk_qdrant_pytorch.py to re-chunk collection "
                        f"'{collection_name}'."
                    )
                elif ratio < 1.5:
                    logger.warning(
                        f"   ⚠️ Qdrant has {points_count} points for {len(laptops)} laptops "
                        f"(~{ratio:.1f}x) — that's a suspiciously low chunks-per-laptop ratio for "
                        f"a chunked collection. Double check '{collection_name}' was built as chunked."
                    )
        except Exception as e:
            logger.warning(f"   Collection not found: {e}")

    # Build knowledge graph
    kg_rag.build_knowledge_graph(collection_name=collection_name, feature_cache=_FEATURE_CACHE)
    kg_rag.build_flattened_vector_subspace()
    kg_stats = kg_rag.graph_stats()
    logger.info(f"   KG stats: {kg_stats}")
    kg_laptops_indexed = kg_stats.get("laptops_indexed")
    if laptops and kg_laptops_indexed is not None and kg_laptops_indexed != len(laptops):
        logger.warning(
            f"   ⚠️ KG has {kg_laptops_indexed} laptops indexed but Qdrant currently has "
            f"{len(laptops)} — kg_cache.gpickle.json is likely stale from a previous "
            f"catalog build. Delete it and rerun to force a rebuild from '{collection_name}'."
        )

    _VS_BUILT = True


# =============================================================================
# LCEL CHAINS
# =============================================================================

def create_chain(system_prompt: str, max_tokens: int, temperature: float = 0.3, format_json: bool = False, use_reasoning: Optional[bool] = None):
    """Create a chain with model routing (see LLMFactory.get_llm)."""
    llm = LLMFactory.get_llm(max_tokens, temperature, format_json, use_reasoning=use_reasoning)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


# Moderation chain - FIXED escaping
_moderation_chain = create_chain(
    "You are a content moderation classifier. Respond ONLY with valid JSON: {{'flagged': true}} or {{'flagged': false}}",
    MAX_MODERATION,
    0.0,
    True
)

# Intent detection chain - FIXED escaping
_intent_chain = create_chain(
    "Check if ALL 6 laptop requirements (GPU intensity, Display quality, Portability, Multitasking, Processing speed, Budget) are captured. Respond ONLY with valid JSON: {{'all_captured': true}} or {{'all_captured': false}}",
    MAX_INTENT,
    0.0,
    True
)

# Requirement string extraction chain — plain-text extraction, no reasoning needed
#
# BUG FIXED: the previous prompt showed <gpu>/<budget>/etc as bracket
# placeholders describing the FORMAT to fill in. Llama 3.1 (unlike
# DeepSeek-R1's reasoning pass) took that literally and echoed the bracket
# syntax back verbatim — e.g. "a budget of <5k>." — instead of substituting
# the actual value, which then failed number parsing downstream and silently
# fell back to the default 100000. A concrete worked example (few-shot) is
# what makes a non-reasoning model reliably substitute real values instead
# of parroting the template.
_req_string_chain = create_chain(
    "Extract laptop requirements from the conversation and rewrite them into "
    "ONE sentence with REAL values substituted in (never output angle "
    "brackets or placeholder text).\n\n"
    "Example input: \"I want a gaming laptop, best GPU and display, budget "
    "around 5k dollars\"\n"
    "Example output: I need a laptop with high GPU intensity, high display "
    "quality, medium portability, high multitasking, high processing speed "
    "and a budget of 5000.\n\n"
    "Rules:\n"
    "- GPU/display/portability/multitasking/processing must each be exactly "
    "one of: low, medium, high\n"
    "- Budget must be a plain number in the ORIGINAL currency's smallest "
    "common unit the user mentioned (e.g. \"5k dollars\" -> 5000, "
    "\"1.5 lakh\" -> 150000), never a placeholder, never a word\n"
    "- Output ONLY the one sentence, no explanation, no brackets",
    MAX_REQ_STRING,
    0.3,
    use_reasoning=False,
)

# Structured requirement extraction chain - FIXED escaping
_extraction_chain = create_chain(
    'Extract laptop requirements. Respond ONLY with valid JSON matching exactly: {{"GPU intensity":"low|medium|high","Display quality":"low|medium|high","Portability":"low|medium|high","Multitasking":"low|medium|high","Processing speed":"low|medium|high","Budget":<integer>}}',
    MAX_FUNC_CALLING,
    0.0,
    True
)

# HyDE (Hypothetical Document Embeddings) chain — templated generation, no reasoning needed
_hyde_chain = create_chain(
    "You are a laptop spec writer. Write a 2-3 sentence realistic product description. No price. Plain prose.",
    MAX_HYDE,
    0.0,  # deterministic HyDE makes retrieval/evaluation reproducible
    use_reasoning=False,
)

# System prompts
_gather_system = (
    "You are a laptop shopping assistant. Collect these six requirements before "
    "searching: GPU intensity, display quality, portability, multitasking, "
    "processing speed, and budget. Ask one short question at a time and never "
    "assume a preference that the user has not stated."
)

_REQUIREMENT_ORDER = (
    "GPU intensity", "Display quality", "Portability", "Multitasking",
    "Processing speed", "Budget",
)

_REQUIREMENT_QUESTIONS = {
    "GPU intensity": "How much graphics performance do you need: low, medium, or high?",
    "Display quality": "What display quality do you want: low, medium, or high?",
    "Portability": "How portable should it be: low, medium, or high?",
    "Multitasking": "How much multitasking do you need: low, medium, or high?",
    "Processing speed": "What processing speed do you need: low, medium, or high?",
    "Budget": "What is your maximum budget? Please include the amount and currency.",
}

# Plain-language meaning of each tier, fed to the LLM below so it can ask a
# natural question without ever saying the words "low", "medium", "high".
_REQUIREMENT_HINTS = {
    "GPU intensity": "low = no gaming/3D work, medium = casual gaming or occasional editing, high = heavy gaming, 3D rendering, or AI/ML work",
    "Display quality": "low = a basic screen is fine, medium = a decent everyday screen, high = a crisp, color-accurate screen for photo/video/creative work",
    "Portability": "low = mostly stays on a desk, medium = carried around sometimes, high = carried around all day and needs to be thin and light",
    "Multitasking": "low = a few tabs/apps at once, medium = normal daily multitasking, high = many heavy apps and tabs open at the same time",
    "Processing speed": "low = basic browsing and documents, medium = everyday multitasking and some heavier apps, high = demanding compiling, editing, or number-crunching work",
}

# System prompt for rephrasing a requirement axis as one natural, specific
# question about real-world usage — never as a "low/medium/high?" menu.
_creative_question_chain = create_chain(
    "You are a friendly laptop shopping assistant gathering one requirement at "
    "a time from a customer. You'll be given a requirement axis and what its "
    "low/medium/high tiers mean in plain terms. Ask ONE short, natural, "
    "conversational question (max 20 words) that gets the customer to reveal "
    "where they land on that axis — WITHOUT ever using the words 'low', "
    "'medium', or 'high', and without presenting it as a multiple-choice menu. "
    "Ask about what they actually do with the laptop instead. "
    "Output ONLY the question, nothing else.",
    120,
    0.8,
    use_reasoning=False,
)

# Generated once per feature per process and reused — keeps the UX creative
# without paying an LLM round trip on every single turn.
_creative_question_cache: Dict[str, str] = {}


def _ask_requirement_creatively(feature: str) -> str:
    """
    Return a natural-language question for `feature` instead of the blunt
    "low, medium, or high?" prompt. Falls back to the static question in
    _REQUIREMENT_QUESTIONS if the LLM is unavailable, empty, too long, or
    still leaks a tier word. Downstream answer parsing (_answer_tier /
    _explicit_tier_in_message) reads the user's reply, not the question text,
    so this rephrasing never affects requirement extraction correctness.
    """
    if feature == "Budget":
        return _REQUIREMENT_QUESTIONS[feature]

    cached = _creative_question_cache.get(feature)
    if cached:
        return cached

    question = None
    try:
        raw = _creative_question_chain.invoke({
            "input": f"Requirement axis: {feature}. Tier meanings: {_REQUIREMENT_HINTS[feature]}."
        })
        candidate = clean_llm_response(raw).strip().strip('"')
        if candidate and len(candidate) <= 200 and not re.search(r"\b(low|medium|high)\b", candidate, re.IGNORECASE):
            question = candidate
            logger.info(f"   [CREATIVE QUESTION] {feature!r} -> {question!r}")
        else:
            logger.warning(f"   [CREATIVE QUESTION] unusable for {feature!r}, using fallback: {candidate!r}")
    except Exception as exc:
        logger.warning(f"   [CREATIVE QUESTION] generation failed for {feature!r}: {exc}")

    question = question or _REQUIREMENT_QUESTIONS[feature]
    _creative_question_cache[feature] = question
    return question


# Fallback classifier for when neither the literal low/medium/high check nor
# the keyword rules can read a tier out of the user's answer — most commonly
# a plain "yes"/"no" reply to a creative question, whose meaning depends on
# how that specific question was phrased (e.g. "yes" to "do you travel with
# it a lot?" implies HIGH portability). Given both the question and the
# answer, the LLM can resolve that; static rules can't.
_answer_tier_llm_chain = create_chain(
    "You classify a customer's short answer to a laptop-shopping question into "
    "exactly one of: low, medium, high, unclear. You'll get the requirement "
    "axis, what its tiers mean, the question the customer was asked, and their "
    "answer. Map the answer onto the tier it implies for that axis — e.g. a "
    "plain 'yes' to a question about wanting/needing the high-tier trait means "
    "'high'; a plain 'no' to that framing means 'low'. If you genuinely can't "
    "tell, answer 'unclear'. Output ONLY that one word, nothing else.",
    20,
    0.0,
    use_reasoning=False,
)


def _answer_tier_via_llm(feature: str, question_asked: str, text: str) -> Optional[str]:
    """Classify text against the specific question_asked. Returns None if the
    LLM is unavailable or genuinely unsure — caller should re-ask, not guess."""
    try:
        raw = _answer_tier_llm_chain.invoke({
            "input": (
                f"Requirement axis: {feature}. Tier meanings: {_REQUIREMENT_HINTS.get(feature, '')}.\n"
                f"Question asked: {question_asked}\n"
                f"Customer answer: {text}"
            )
        })
        tier = clean_llm_response(raw).strip().lower()
        result = tier if tier in {"low", "medium", "high"} else None
        logger.info(f"   [ANSWER CLASSIFY] {feature!r} answer {text!r} -> {result!r}")
        return result
    except Exception as exc:
        logger.warning(f"   [ANSWER CLASSIFY] failed for {feature!r}: {exc}")
        return None


def _answer_tier(text: str) -> Optional[str]:
    """Accept a short guided answer such as 'high' or 'medium performance'."""
    match = re.search(r"\b(low|medium|high)\b", text.lower())
    return match.group(1) if match else None


# Whole-message fallback for axes the regex/keyword rules in
# _explicit_tier_in_message miss entirely — casual phrasing like "alot of
# multitasking" or "best processor" that isn't one of the hardcoded patterns.
# "unstated" is a required output option specifically so the model doesn't
# guess a preference for an axis the customer never actually addressed.
_missing_tiers_chain = create_chain(
    "You extract ONLY explicitly-stated laptop preferences from a customer's "
    "message. You'll be given a list of requirement axes to check. For EACH "
    "axis, respond with 'low', 'medium', 'high', or 'unstated' if the message "
    "does not clearly state or imply a preference for that axis. Do NOT guess "
    "from general assumptions — e.g. 'gaming' alone says nothing about "
    "portability. Casual phrasing counts as explicit: 'alot of multitasking' "
    "= high, 'best processor' = high, 'portability isn't an issue' = low. "
    "Respond ONLY with valid JSON mapping each given axis name to one of: "
    "low, medium, high, unstated.",
    150,
    0.0,
    True,  # format_json
)


def _extract_explicit_tiers_llm(features: List[str], text: str) -> Dict[str, str]:
    """LLM fallback for axes the regex/keyword rules didn't catch. Only
    returns axes the message actually addresses — 'unstated' axes are
    dropped, never guessed, so silence never invents a fake preference."""
    if not features:
        return {}
    try:
        raw = _missing_tiers_chain.invoke({
            "input": f"Axes to check: {features}\nCustomer message: {text}"
        })
        raw = clean_llm_response(raw)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception as exc:
        logger.warning(f"   [TIER EXTRACT] failed: {exc}")
        return {}

    result = {}
    for feature in features:
        tier = str(data.get(feature, "unstated")).lower()
        if tier in {"low", "medium", "high"}:
            result[feature] = tier
            logger.info(f"   [TIER EXTRACT] {feature!r} -> {tier!r} from message")
    return result


def _explicit_tier_in_message(feature: str, text: str) -> Optional[str]:
    """Return a tier only when the user named a preference/spec for this axis."""
    lower = text.lower()
    names = {
        "GPU intensity": ("gpu", "graphics", "vram"),
        "Display quality": ("display", "screen", "resolution", "refresh rate"),
        "Portability": ("portable", "portability", "weight", "lightweight"),
        "Multitasking": ("multitasking", "multitask", "ram", "memory"),
        "Processing speed": ("processing", "processor", "cpu", "performance"),
    }
    for tier in ("low", "medium", "high"):
        if re.search(rf"\b{tier}\b.*\b(?:{'|'.join(names[feature])})\b", lower) or re.search(rf"\b(?:{'|'.join(names[feature])})\b.*\b{tier}\b", lower):
            return tier
    for tier, terms in _KW_RULES[feature].items():
        if any(_kw_matches(term, lower) for term in terms):
            return tier

    # Common natural-language and shorthand forms users put in a single
    # message (for example: "lightweight, 16 GB RAM, i7").  These are
    # explicit specifications, not defaults inferred from a use case.
    shorthand_patterns = {
        "GPU intensity": {
            "high": (r"\bbest\s+(?:gpu|graphics)\b", r"\brtx\s*40\d{2}\b"),
            "medium": (r"\bmid[- ]?range\s+(?:gpu|graphics)\b",),
            "low": (r"\bno\s+(?:dedicated\s+)?gpu\b",),
        },
        "Display quality": {
            "high": (r"\bbest\s+(?:display|screen)\b", r"\bhigh[- ]?resolution\b"),
            "medium": (r"\bgood\s+(?:display|screen)\b",),
            "low": (r"\bbasic\s+(?:display|screen)\b",),
        },
        "Portability": {
            "high": (r"\blightweight\b", r"\btravel(?:ling)?\b", r"\beasy\s+to\s+carry\b"),
            "low": (r"\bdesktop replacement\b", r"\bdon'?t care about weight\b"),
        },
        "Multitasking": {
            "high": (r"\b(?:32|64)\s*gb(?:\s+(?:ram|memory))?\b", r"\bheavy multitasking\b"),
            "medium": (r"\b16\s*gb(?:\s+(?:ram|memory))?\b",),
            "low": (r"\b(?:4|8)\s*gb(?:\s+(?:ram|memory))?\b",),
        },
        "Processing speed": {
            "high": (r"\b(?:core\s+)?i9\b", r"\bryzen\s*9\b", r"\bfast\s+(?:cpu|processor)\b"),
            "medium": (r"\b(?:core\s+)?i7\b", r"\bryzen\s*7\b"),
            "low": (r"\b(?:core\s+)?i[35]\b", r"\bryzen\s*5\b"),
        },
    }
    for tier, patterns in shorthand_patterns[feature].items():
        if any(re.search(pattern, lower) for pattern in patterns):
            return tier
    return None


def _collect_requirements_from_turn(existing: dict, pending: str, text: str) -> dict:
    """Merge explicit preferences from one user turn into the intake profile."""
    collected = dict(existing)
    if pending == "Budget":
        budget = parse_budget(text)
        if budget is not None:
            collected["Budget"] = budget
    elif pending:
        tier = _answer_tier(text)
        if not tier:
            tier = _explicit_tier_in_message(pending, text)
        if not tier:
            question_asked = _creative_question_cache.get(pending, _REQUIREMENT_QUESTIONS[pending])
            tier = _answer_tier_via_llm(pending, question_asked, text)
        if tier:
            collected[pending] = tier

    # A detailed message can answer multiple questions at once. Regex/keyword
    # rules catch common phrasing ("lightweight", "16gb ram", "i7"); an LLM
    # fallback catches casual phrasing they miss ("alot of multitasking",
    # "best processor", "portability isn't an issue") for whatever axes are
    # still unfilled. The fallback only ever fills axes the message actually
    # addresses — it must say "unstated" rather than guess, so a generic use
    # case like "gaming" alone still doesn't invent a portability preference.
    still_missing = [f for f in _REQUIREMENT_ORDER[:-1] if f not in collected]
    for feature in list(still_missing):
        tier = _explicit_tier_in_message(feature, text)
        if tier:
            collected[feature] = tier
            still_missing.remove(feature)
    if still_missing:
        collected.update(_extract_explicit_tiers_llm(still_missing, text))

    if "Budget" not in collected:
        budget = parse_budget(text)
        if budget is not None:
            collected["Budget"] = budget
    return collected


def _requirements_to_sentence(requirements: dict) -> str:
    return (
        "I need a laptop with "
        f"{requirements['GPU intensity']} GPU intensity, "
        f"{requirements['Display quality']} display quality, "
        f"{requirements['Portability']} portability, "
        f"{requirements['Multitasking']} multitasking, "
        f"{requirements['Processing speed']} processing speed and a budget of "
        f"{requirements['Budget']}."
    )

_reco_system = (
    "You are an intelligent laptop expert. The user's recommended laptops are:\n{products}\n\n"
    "Present them clearly in decreasing order of price."
)


# =============================================================================
# STATE - Full TypedDict
# =============================================================================

class LaptopState(TypedDict, total=False):
    """Complete state schema for the laptop shopping assistant."""
    
    # LangGraph message history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Raw user text for the current turn
    user_input: str
    
    # Moderation result: "ok" | "flagged"
    moderation_result: str
    
    # Whether all requirements have been gathered
    requirements_complete: bool
    
    # Structured user requirements
    requirement_string: str
    requirements: dict
    # Explicit answers collected by the guided intake. These are kept separate
    # from `requirements`, which later also contains retrieval-only fields.
    collected_requirements: dict
    pending_requirement: str
    requirement_retry_count: int
    
    # Conversation history in dict format (for Flask render)
    conversation_bot: List[dict]
    
    # Recommendation phase
    top_3_laptops: Optional[str]
    conversation_reco: List[dict]
    
    # Search + compare outputs
    search_results: List[dict]
    ranked_laptops: List[dict]
    top_k_laptops: List[dict]
    comparison_table: str
    comparison_analysis: str
    best_overall: dict
    
    # Flow control
    phase: str  # "gather" | "search" | "compare" | "followup" | "end" | "side_compare" | "upgrade" | "pdf_report"
    error: str
    last_response: str
    
    # Orchestrator intent
    orchestrator_intent: str  # "recommend" | "side_compare" | "upgrade" | "pdf_report" | "followup"
    
    # Side-by-side comparison agent
    compare_laptops: List[str]
    compare_keywords: List[str]
    compare_candidates: dict
    side_compare_result: str
    
    # Upgrade advisor agent
    current_laptop: str
    upgrade_advice: str
    
    # PDF report agent
    pdf_path: str
    pdf_url: str
    
    # KG-RAG context
    kg_seed_nodes: List[str]
    kg_triplets: List[dict]
    kg_context: List[str]
    kg_weight_used: float   # the _KG_WEIGHT search_node actually blended with this turn
    kg_changed_top: bool    # whether KG fusion changed the #1 ranked laptop vs vector-only RRF
    
    # Case memory (Memento-style CBR) — similar past cases retrieved for
    # this turn's recommendation, kept around for debugging/admin visibility
    case_context: List[dict]
    retrieval_metrics: dict
    retrieval_attribution: dict
    retrieval_retry_count: int
    retrieval_action: dict
    retrieval_action_override: dict  # offline evaluation only
    offline_evaluation: bool
    retrieval_planner_cases: List[dict]
    retrieval_planner_policy: dict
    
    # KG metrics
    kg_metrics: dict
    kg_literal_map: List[dict]


# =============================================================================
# ORCHESTRATOR NODE - Full Implementation
# =============================================================================

_ORCHESTRATOR_SYSTEM = """You are a router for a laptop shopping assistant.
Classify the user's intent into EXACTLY one of:
  - "side_compare"  : user wants a head-to-head comparison of two specific laptops
  - "upgrade"       : user describes their current laptop and asks if they should upgrade
  - "pdf_report"    : user asks for a PDF, report, download, or export of results.
                      ANY mention of "pdf", "download", "generate report", "export",
                      "create report", "save report" MUST map here.
                      NEVER answer PDF generation requests yourself — always return "pdf_report".
  - "recommend"     : user wants laptop recommendations (default)
  - "followup"      : user is asking a follow-up question after a recommendation was made

CRITICAL: If the user says anything like "generate a pdf", "download report", "give me a pdf",
"create a report", "export results" — you MUST return {"intent": "pdf_report"} and nothing else.

Respond ONLY with valid JSON: {"intent": "<one of the above>"}
"""

_PIPELINE_PHASE_LOCK = {
    "gather": "recommend",
    "search": "recommend",
    "compare": "recommend",
    "side_compare": "side_compare",
    "side_compare_clarify": "side_compare",
}


def orchestrator_node(state: LaptopState) -> LaptopState:
    """
    Master router: reads user_input and current phase, decides which agent
    pipeline handles this turn.
    
    FIX (pipeline-switching bug):
      Previously the orchestrator sent ONLY the user's latest message to the LLM
      for classification, with no awareness of what pipeline was already running.
      Mid-flow answers like "rtx 3080 yes and 1 tb" were classified as "upgrade"
      because they look like hardware specs, even though we were mid-recommendation.
    
      Now: if a pipeline is actively running (phase lock), we honour that pipeline
      and only allow switching to pdf_report (explicit export) or side_compare
      (explicit new comparison request with "vs" keyword) — never to upgrade.
    """
    logger.info("🧭 [ORCHESTRATOR NODE] started")

    user_input = state.get("user_input", "")
    logger.info(f"   📝 User input: {user_input!r}")
    phase = state.get("phase", "gather")
    lower_input = user_input.lower()

    # ── 1. PDF shortcut — always wins regardless of phase ────────────────────
    _pdf_keywords = ("pdf", "download", "export", "generate report", "create report", "save report")
    if any(kw in lower_input for kw in _pdf_keywords):
        logger.info("   PDF keyword shortcut — routing to pdf_report")
        return {**state, "orchestrator_intent": "pdf_report"}

    # ── 2. Phase lock — if a pipeline is mid-flow, stay in it ────────────────
    locked_intent = _PIPELINE_PHASE_LOCK.get(phase)
    if locked_intent:
        # The only allowed escape from a locked pipeline is an explicit new
        # comparison request (must contain "vs" or "versus" to be unambiguous)
        is_new_compare = (
            locked_intent != "side_compare"
            and ("vs" in lower_input or "versus" in lower_input or "compare" in lower_input)
            and " vs " in lower_input  # require actual two-sided compare syntax
        )
        if not is_new_compare:
            logger.info(f"   Phase lock active: phase='{phase}' → forcing intent='{locked_intent}'")
            return {**state, "orchestrator_intent": locked_intent}

    # ── 3. Followup lock — already have recommendations, keep chatting ────────
    if phase == "followup" and state.get("top_3_laptops"):
        # Allow side_compare and pdf_report to break out; block upgrade hijack
        if not ("vs" in lower_input or "versus" in lower_input):
            logger.info("   Followup lock active — staying in followup")
            return {**state, "orchestrator_intent": "followup"}

    # ── 4. LLM classification for fresh / unambiguous turns ──────────────────
    try:
        llm = LLMFactory.get_llm(MAX_ORCHESTRATOR, temperature=0.0, format_json=True)
        raw = llm.invoke(f"User input: {user_input}\n\nClassify the intent and return JSON.")
        
        # Clean the response
        raw = clean_llm_response(raw)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            intent = json.loads(json_match.group()).get("intent", "recommend")
        else:
            intent = "recommend"
    except Exception as e:
        logger.warning(f"   Orchestrator LLM failed: {e}, defaulting to recommend")
        intent = "recommend"

    logger.info(f"   Orchestrated intent: {intent}")
    return {**state, "orchestrator_intent": intent}


def route_after_orchestrator(state: LaptopState) -> str:
    """Route to the appropriate node based on orchestrator intent."""
    intent = state.get("orchestrator_intent", "recommend")
    phase = state.get("phase", "gather")

    # Mid-clarification: user is responding to laptop options we presented
    if phase == "side_compare_clarify":
        return "side_compare_clarify_node"

    if intent == "side_compare":
        return "side_compare_parse_node"
    if intent == "upgrade":
        return "upgrade_node"
    if intent == "pdf_report":
        return "pdf_report_node"
    if intent == "followup" and state.get("top_3_laptops"):
        return "followup_node"
    if phase == "followup" and state.get("top_3_laptops"):
        return "followup_node"
    
    return "conversation_node"


# =============================================================================
# CONVERSATION NODE - Full Implementation
# =============================================================================

_WELCOME_MESSAGE = """\
👋 Hi! I'm your **Laptop Shopping Assistant** — here's what I can help you with:

🔍 **Find the right laptop** — tell me your use case, budget, and preferences and I'll recommend the best options from our database.

⚖️ **Side-by-side comparison** — say something like *"Compare Dell XPS 15 vs MacBook Pro 14"* and I'll give you a detailed spec breakdown.

⬆️ **Upgrade advice** — tell me your current laptop and I'll let you know if it's worth upgrading.

📄 **PDF report** — once I've made recommendations, ask for a PDF and I'll generate a downloadable report.

---
To get started, tell me what you'll mainly use the laptop for and your approximate budget! 💬\
"""


def conversation_node(state: LaptopState) -> LaptopState:
    """
    Conversation node: handles natural language conversation with the user
    to gather requirements.
    """
    logger.info("💬 [CONVERSATION NODE] started")

    user_input = state["user_input"]
    conv_bot = list(state.get("conversation_bot", []))
    messages = list(state.get("messages", []))

    # ── First turn: show welcome message instead of calling the LLM ──────────
    if not messages:
        conv_bot.append({"bot": _WELCOME_MESSAGE})
        return {
            **state,
            "messages": [HumanMessage(content=user_input)],
            "conversation_bot": conv_bot,
            "last_response": _WELCOME_MESSAGE,
        }

    collected = _collect_requirements_from_turn(
        state.get("collected_requirements", {}),
        state.get("pending_requirement", ""),
        user_input,
    )
    missing = next((key for key in _REQUIREMENT_ORDER if key not in collected), None)

    prev_pending = state.get("pending_requirement", "")
    retry_count = state.get("requirement_retry_count", 0)
    retry_count = retry_count + 1 if (missing and missing == prev_pending) else 0

    if missing:
        # After two unclear answers on the same axis, drop the creative
        # phrasing and ask the plain low/medium/high question directly —
        # guarantees the conversation can always terminate instead of
        # repeating the same creative question forever.
        response = (
            _REQUIREMENT_QUESTIONS[missing] if retry_count >= 2
            else _ask_requirement_creatively(missing)
        )
        complete = False
        requirements = state.get("requirements", {})
        requirement_string = state.get("requirement_string", "")
        phase = "gather"
    else:
        # Only start retrieval once all six values were explicitly supplied.
        user_history = "\n".join(
            m.content for m in messages if isinstance(m, HumanMessage)
        )
        user_history = f"{user_history}\n{user_input}".strip()
        requirements = _decompose_query(user_history, dict(collected))
        requirement_string = _requirements_to_sentence(collected)
        response = "Thanks — I have all your requirements. Let me find the best laptops for you…"
        complete = True
        phase = "search"

    new_messages = [HumanMessage(content=user_input), AIMessage(content=response)]
    conv_bot.append({"user": user_input})
    conv_bot.append({"bot": response})

    return {
        **state,
        "messages": new_messages,
        "conversation_bot": conv_bot,
        "last_response": response,
        "collected_requirements": collected,
        "pending_requirement": missing or "",
        "requirement_retry_count": retry_count,
        "requirements_complete": complete,
        "requirements": requirements,
        "requirement_string": requirement_string,
        "phase": phase,
    }


def moderation_node(state: LaptopState) -> LaptopState:
    """
    Moderation node: checks for inappropriate content.
    """
    logger.info("🛡️ [MODERATION NODE] started")

    texts_to_check = [state.get("user_input", "")]
    last = state.get("last_response", "")
    if last:
        texts_to_check.append(last)

    flagged = False
    for text in texts_to_check:
        try:
            raw = _moderation_chain.invoke({"input": text})
            raw = clean_llm_response(raw)
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                if json.loads(json_match.group()).get("flagged", False):
                    flagged = True
                    break
        except Exception as e:
            logger.warning(f"   Moderation check failed: {e}")

    result = "flagged" if flagged else "ok"
    logger.info(f"   moderation_result={result}")
    return {**state, "moderation_result": result}


def intent_node(state: LaptopState) -> LaptopState:
    """
    Intent node: determines if all requirements have been gathered.
    """
    logger.info("🎯 [INTENT NODE] started")

    messages = list(state.get("messages", []))
    
    # Check if this is the welcome message (first turn)
    if not messages or len(messages) <= 1:
        logger.info("   First turn - not detecting requirements yet")
        return {**state, "requirements_complete": False, "phase": "gather"}

    # conversation_node is the single authority for intake completion.  The
    # old budget-only shortcut below inferred the other five axes and started
    # a search as soon as it found any number in the chat.
    if state.get("requirements_complete"):
        return state
    return {**state, "requirements_complete": False, "phase": "gather"}
    
    full_convo = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
    )

    # IMPORTANT: budget/requirement detection must only ever look at what the
    # USER actually typed, never at the assistant's own generated replies.
    # Bug found: convo_text used to be full_convo (user+assistant mixed) plus
    # last_response (the assistant's own just-generated reply). Since the
    # assistant's reply is free-text LLM output, it can hallucinate a garbled
    # number while trying to restate the user's budget back to them (e.g.
    # mangling "5k dollars" into something like "₹425" in its confirmation
    # message) — and because that hallucinated number could match an
    # earlier-priority pattern (₹ before k), it would silently override the
    # real number the user actually stated. Only HumanMessage content is
    # ground truth for what the user said.
    user_only_text = "\n".join(
        m.content for m in messages if isinstance(m, HumanMessage)
    )
    last_response = state.get("last_response", "")
    convo_text = user_only_text
    convo_lower = convo_text.lower()

    # ── Budget detection ───────────────────────────────────────────────────────
    # NOTE: budget parsing used to be duplicated inline here (and again further
    # below), each copy drifting independently as bugs got patched in one copy
    # but not the other (e.g. one copy lacked USD->INR conversion for "5k
    # dollars"). Now delegates entirely to the single _detect_budget_number
    # helper — one implementation, no drift.
    has_budget_number = parse_budget(convo_text) is not None
    has_budget = has_budget_number

    # Only trigger if user has actually given a real number
    if has_budget and len(messages) > 1:
        logger.info("   Budget detected in conversation")

        _detected_budget = parse_budget(convo_text)

        try:
            req_string = _extract_req_string(user_only_text)
            requirements = _extract_requirements(req_string)
        except Exception as e:
            logger.warning(f"   Extraction failed: {e}")
            # Fall back to defaults, but keep the budget the user actually stated
            requirements = {
                "GPU intensity": "medium",
                "Display quality": "medium",
                "Portability": "medium",
                "Multitasking": "medium",
                "Processing speed": "medium",
                "Budget": _detected_budget,
            }
            req_string = (
                f"I need a laptop with medium GPU intensity, medium display quality, "
                f"medium portability, medium multitasking, medium processing speed "
                f"and a budget of {_detected_budget}."
            )
        
        conv_bot = list(state.get("conversation_bot", []))
        conv_bot.append({"bot": "Thank you! Let me find the best laptops for you…"})
        
        return {
            **state,
            "requirements_complete": True,
            "requirement_string": req_string,
            "requirements": requirements,
            "conversation_bot": conv_bot,
            "phase": "search",
        }

    # ── Deterministic pre-check ───────────────────────────────────────────────
    # The 7B intent-classification LLM (_intent_chain) is unreliable and can
    # say "all_captured": false even when the user has clearly already given
    # everything needed, causing the bot to re-ask questions it doesn't need
    # answered. We only reach this point with has_budget_number == False (the
    # branch above already returns when it's True), so budget is genuinely
    # still missing — no amount of use-case detail changes that. There is
    # nothing to short-circuit here; skip the LLM entirely and ask directly.
    confirmed = False

    # We only reach this branch when has_budget_number was False — i.e. the
    # user has NOT stated an actual budget number yet. Regardless of what the
    # (unreliable, 7B) intent LLM claims, we never mark requirements complete
    # without a real number — otherwise we silently fabricate a budget.
    if not confirmed or not has_budget_number:
        conv_bot = list(state.get("conversation_bot", []))
        if not any("budget" in msg.get("bot", "").lower() for msg in conv_bot):
            conv_bot.append({"bot": "I need a bit more information. Could you please specify your budget? For example: 'My budget is 1.5 lakh' or 'Under ₹100,000'."})
        return {**state, "conversation_bot": conv_bot, "requirements_complete": False, "phase": "gather"}

    # NOTE: everything below this point was previously dead code — `confirmed`
    # is hardcoded to False above, so `not confirmed or not has_budget_number`
    # is always True, meaning the return above always fires. Removed the
    # unreachable duplicate budget-detection block and its own drifted copy
    # of the same bug (missing USD conversion for "k" values).


# Deprecated compatibility default. Search no longer reads this mutable global:
# RetrievalPlanner selects an action from similar evaluated cases per query.
_KG_WEIGHT = 0.3


def set_kg_weight(weight: float) -> None:
    """Deprecated compatibility setter; retained for external callers only."""
    global _KG_WEIGHT
    _KG_WEIGHT = weight


def get_kg_weight() -> float:
    return _KG_WEIGHT


def _rerank_candidates(query: str, candidates: List[dict], enabled: bool = True, top_k: Optional[int] = None) -> Tuple[List[dict], bool]:
    """Cross-encode candidates when the selected retrieval action enables it."""
    top_k = top_k or RERANK_TOP_K
    if not enabled:
        return candidates[:top_k], False
    model = _CrossEncoderSingleton.get()
    candidates = candidates[:RERANK_CANDIDATES]
    if model is None or not candidates:
        return candidates[:top_k], False
    try:
        passages = [
            f"{c.get('name', '')}. Price INR {c.get('price', '')}. {c.get('description', '')[:900]}"
            for c in candidates
        ]
        scores = model.predict([(query, passage) for passage in passages], batch_size=RERANK_BATCH_SIZE, show_progress_bar=False)
        score_values = [float(score) for score in scores]
        low, high = min(score_values), max(score_values)
        reranked = []
        max_requirement_score = max((c.get("score", 0) for c in candidates), default=1) or 1
        for candidate, score in zip(candidates, score_values):
            item = dict(candidate)
            item["reranker_score"] = float(score)
            semantic = (float(score) - low) / max(high - low, 1e-6)
            compliance = item.get("score", 0) / max_requirement_score
            # Prevent semantic similarity from promoting a weak-spec laptop
            # above candidates that satisfy more extracted requirements.
            item["final_rerank_score"] = 0.65 * semantic + 0.35 * compliance
            reranked.append(item)
        before = str(candidates[0].get("id"))
        reranked.sort(key=lambda item: item["final_rerank_score"], reverse=True)
        return reranked[:top_k], bool(reranked and str(reranked[0].get("id")) != before)
    except Exception as exc:
        logger.warning("Cross-encoder reranking failed; retaining RRF order: %s", exc)
        return candidates[:top_k], False


def search_node(state: LaptopState) -> LaptopState:
    """
    Search node: performs dense + sparse search with RRF fusion and KG integration.
    """
    logger.info("🔍 [SEARCH NODE] started")

    requirements = state.get("requirements", {})
    req_string = state.get("requirement_string", "")
    retrieval_query = requirements.get("Retrieval query", req_string)
    retrieval_started = time.perf_counter()
    budget = int(requirements.get("Budget", 0)) or 10_000_000
    # Contextual Memento policy: retrieval_memory delegates to the Qdrant case
    # bank, so this choice depends on analogous requirement states only.
    override = state.get("retrieval_action_override", {}) or {}
    if override:
        action = _normalise_retrieval_action(override)
        planner_cases = []
        planner_policy = {"policy": "offline_action_override", "candidate_actions": []}
        logger.info("[RETRIEVAL PLANNER] offline action override kg_weight=%.2f", action["kg_weight"])
    else:
        import retrieval_memory
        action, planner_policy, planner_cases = retrieval_memory.select_action(retrieval_query, requirements)
    stored_dim = 384

    def _trim(v):
        return (v + [0.0] * stored_dim)[:stored_dim]

    # The retrieval action is selected per-query from similar successful cases.
    if action["hyde_enabled"]:
        try:
            hyp_doc = _hyde_chain.invoke({"input": retrieval_query})
            hyp_doc = clean_llm_response(hyp_doc)
        except Exception as e:
            logger.warning(f"   HyDE failed: {e}, using raw requirement")
            hyp_doc = retrieval_query
    else:
        hyp_doc = retrieval_query
    
    hyp_emb = _trim(_embedder.embed_query(hyp_doc))
    raw_emb = _trim(_embedder.embed_query(retrieval_query))
    blended = list((np.array(hyp_emb) + np.array(raw_emb)) / 2)

    logger.debug(f"   [SEARCH NODE] HyDE doc: {hyp_doc[:120]!r}...")
    logger.debug(f"   [SEARCH NODE] req_string used for retrieval: {req_string[:120]!r}...")

    # Dense and sparse search
    dense_r = _dense_search(blended, budget, top_k=action["dense_top_k"])
    sparse_r = _sparse_search(hyp_doc + " " + retrieval_query, budget, top_k=action["sparse_top_k"])
    fused = _rrf_fuse(dense_r, sparse_r)

    # Deduplicate fused results
    aggregated: Dict[str, dict] = {}
    for item in fused:
        lid = item["laptop"]["id"]
        if lid not in aggregated:
            aggregated[lid] = {"laptop": item["laptop"], "rrf_score": 0.0, "count": 0}
        aggregated[lid]["rrf_score"] += item["rrf_score"]
        aggregated[lid]["count"] += 1
    
    fused_deduped = sorted(
        [{"laptop": v["laptop"], "rrf_score": v["rrf_score"] / max(v["count"], 1)}
         for v in aggregated.values()],
        key=lambda x: x["rrf_score"], reverse=True
    )

    logger.info(f"   [SEARCH NODE] {len(fused)} fused chunks -> {len(fused_deduped)} after dedup")

    # KG retrieval and fusion
    # Snapshot the pre-KG (vector-only RRF) top pick so we can tell afterward
    # whether KG fusion actually changed the winner — this is what lets
    # run_grounding_eval.py's retrieval_memory bandit attribute a correctness
    # score specifically to "did blending in KG help or hurt", instead of
    # only ever seeing the combined result.
    pre_kg_top_id = str(fused_deduped[0]["laptop"]["id"]) if fused_deduped else None
    kg_changed_top = False
    kg_cache_key = f"search::{retrieval_query}::{budget}"
    try:
        kg_result = kg_rag.one_time_retrieve(kg_cache_key, requirements, req_string=retrieval_query, top_k=15)
        if kg_result.get("laptop_ids"):
            logger.info(f"   KG fusion: {len(kg_result['laptop_ids'])} KG-ranked laptops blended into vector ranking")
            vector_laptops = [v["laptop"] for v in fused_deduped]
            kg_fused = kg_rag.fuse_kg_with_vector_results(kg_result, vector_laptops, kg_weight=action["kg_weight"])
            # KG can surface semantically related laptops outside the vector
            # filter. Preserve the user's budget as a hard constraint.
            kg_fused = [lap for lap in kg_fused if float(lap.get("price", 0) or 0) <= budget]
            by_id = {str(v["laptop"]["id"]): v for v in fused_deduped}
            fused_deduped = [
                {"laptop": by_id.get(str(lap.get("id")), {"laptop": lap})["laptop"],
                 "rrf_score": lap.get("kg_fused_score", by_id.get(str(lap.get("id")), {}).get("rrf_score", 0.0))}
                for lap in kg_fused
            ]
            post_kg_top_id = str(fused_deduped[0]["laptop"]["id"]) if fused_deduped else None
            kg_changed_top = pre_kg_top_id is not None and post_kg_top_id != pre_kg_top_id
            logger.info(f"   [KG FUSE] -> {len(fused_deduped)} chunks after KG blend (kg_weight={action['kg_weight']}, changed_top={kg_changed_top})")
            for i, item in enumerate(fused_deduped[:10], 1):
                lap = item["laptop"]
                logger.debug(f"      #{i} id={lap.get('id')} score={item['rrf_score']:.5f} name={lap.get('name')!r}")
        else:
            logger.info("   [KG FUSE] no KG laptop_ids returned, skipping blend")
    except Exception as e:
        logger.warning(f"   KG retrieval failed: {e}")
        kg_result = {"seed_nodes": [], "triplets": [], "context": []}

    # Score each laptop against requirements
    mappings = {"low": 0, "medium": 1, "high": 2}
    req_items = [(k, v) for k, v in requirements.items() if k in _KW_RULES]
    ranked = []
    
    for item in fused_deduped:
        laptop = item["laptop"]
        desc = laptop.get("description", "")
        features = _FEATURE_CACHE.get(desc, {})
        score = sum(
            1 for k, uv in req_items
            if mappings.get((features.get(k) or "").lower(), -1)
               >= mappings.get(uv.lower(), -1)
        )
        ranked.append({**laptop, "score": score, "rrf_score": item["rrf_score"]})

    ranked.sort(key=lambda x: (x["score"], x["rrf_score"]), reverse=True)
    required_gpu = requirements.get("Required GPU", "")
    if required_gpu:
        exact_gpu = [lap for lap in ranked if required_gpu.lower() in lap.get("description", "").lower() or required_gpu.lower() in lap.get("name", "").lower()]
        if exact_gpu:
            exact_ids = {str(lap.get("id")) for lap in exact_gpu}
            ranked = exact_gpu + [lap for lap in ranked if str(lap.get("id")) not in exact_ids]
    minimum_refresh = int(requirements.get("Minimum refresh rate", 0) or 0)
    if minimum_refresh:
        refresh_matches = [
            lap for lap in ranked
            if any(int(hz) >= minimum_refresh for hz in re.findall(r"\b(\d{2,3})\s*hz\b", f"{lap.get('name', '')} {lap.get('description', '')}", re.IGNORECASE))
        ]
        if refresh_matches:
            refresh_ids = {str(lap.get("id")) for lap in refresh_matches}
            ranked = refresh_matches + [lap for lap in ranked if str(lap.get("id")) not in refresh_ids]
    # One bounded corrective pass when the retrieval does not meet even half
    # of the explicitly requested feature tiers. This preserves the graph and
    # avoids agent loops/extra generation for confident requests.
    max_feature_score = max(len(req_items), 1)
    retry_count = state.get("retrieval_retry_count", 0)
    if ranked and ranked[0]["score"] < max_feature_score / 2 and retry_count < 1:
        corrective_query = retrieval_query + " exact requirements " + " ".join(f"{k} {v}" for k, v in req_items)
        corrective_dense = _dense_search(_trim(_embedder.embed_query(corrective_query)), budget, top_k=action["dense_top_k"])
        corrective_sparse = _sparse_search(corrective_query, budget, top_k=action["sparse_top_k"])
        corrective = _rrf_fuse(corrective_dense, corrective_sparse)
        if corrective:
            by_id = {str(x["id"]): x for x in ranked}
            for rank, item in enumerate(corrective, 1):
                lap = dict(item["laptop"])
                existing = by_id.get(str(lap["id"]), {})
                lap["rrf_score"] = existing.get("rrf_score", 0.0) + 1.0 / (60 + rank)
                lap["score"] = existing.get("score", sum(1 for k, uv in req_items if mappings.get((_FEATURE_CACHE.get(lap.get("description", ""), {}).get(k) or "").lower(), -1) >= mappings.get(uv.lower(), -1)))
                by_id[str(lap["id"])] = lap
            ranked = sorted(by_id.values(), key=lambda x: (x["score"], x["rrf_score"]), reverse=True)
            retry_count += 1

    pre_rerank_ids = [str(l.get("id")) for l in ranked[:action["reranker_top_k"]]]
    ranked, reranker_changed_top = _rerank_candidates(
        retrieval_query, ranked,
        enabled=action["reranker_enabled"], top_k=action["reranker_top_k"],
    )
    if required_gpu:
        reranked_gpu = [lap for lap in ranked if required_gpu.lower() in lap.get("description", "").lower() or required_gpu.lower() in lap.get("name", "").lower()]
        if reranked_gpu:
            gpu_ids = {str(lap.get("id")) for lap in reranked_gpu}
            ranked = reranked_gpu + [lap for lap in ranked if str(lap.get("id")) not in gpu_ids]
    if minimum_refresh:
        reranked_refresh = [
            lap for lap in ranked
            if any(int(hz) >= minimum_refresh for hz in re.findall(r"\b(\d{2,3})\s*hz\b", f"{lap.get('name', '')} {lap.get('description', '')}", re.IGNORECASE))
        ]
        if reranked_refresh:
            refresh_ids = {str(lap.get("id")) for lap in reranked_refresh}
            ranked = reranked_refresh + [lap for lap in ranked if str(lap.get("id")) not in refresh_ids]
    attribution = {
        "dense": bool(dense_r), "sparse": bool(sparse_r), "kg": kg_changed_top,
        "reranker": reranker_changed_top, "memory": False,
        "reranker_changed_top": reranker_changed_top,
        "pre_rerank_top_ids": pre_rerank_ids,
    }
    logger.info(f"   Retrieved {len(ranked)} ranked candidates")
    for i, lap in enumerate(ranked[:5], 1):
        logger.info(
            f"      top#{i} id={lap.get('id')} name={lap.get('name')!r} "
            f"req_score={lap.get('score')} rrf_score={lap.get('rrf_score'):.5f} price={lap.get('price')}"
        )
    logger.info("✅ [SEARCH NODE] complete")

    return {
        **state,
        "search_results": fused_deduped,
        "ranked_laptops": ranked,
        "top_k_laptops": ranked[:5],
        "phase": "compare",
        "kg_seed_nodes": kg_result.get("seed_nodes", []),
        "kg_triplets": kg_result.get("triplets", []),
        "kg_context": kg_result.get("context", []),
        "kg_weight_used": action["kg_weight"],
        "kg_changed_top": kg_changed_top,
        "retrieval_retry_count": retry_count,
        "retrieval_attribution": attribution,
        "retrieval_action": action,
        "retrieval_planner_cases": planner_cases,
        "retrieval_planner_policy": planner_policy,
        "retrieval_metrics": {"latency_ms": round((time.perf_counter() - retrieval_started) * 1000, 2), "candidate_count": len(ranked), "reranker_used": action["reranker_enabled"] and bool(_CrossEncoderSingleton.get())},
    }


def _prepare_top3_context(state: LaptopState) -> Optional[dict]:
    """
    Shared prep for both the default recommendation summary and the
    explicit full-comparison pipeline.

    Always takes the TOP 3 laptops in the order search_node already ranked
    them (best match first — score, then rrf/rerank), rather than filtering
    by an arbitrary score threshold first. Filtering before slicing could
    silently drop the #1-ranked laptop (e.g. score==1) and promote a
    lower-ranked one, which would contradict "top 3 in order of best".
    Returns None when there is nothing to recommend.
    """
    laptops = state.get("top_k_laptops", [])
    requirements = state.get("requirements", {})
    if not laptops:
        return None

    top3 = laptops[:3]

    # Live market price is a display-only enrichment layered on top of the
    # catalog price already used for budget filtering/ranking upstream — it
    # never changes `price`, only adds `live_price` (None if lookup fails or
    # SERPAPI_KEY isn't set, in which case the catalog price is shown as-is).
    live_pricing.enrich_with_live_prices(top3)

    # Case memory read (Memento Eq. 13): pull similar past requests + how
    # well they scored, before drafting this turn's recommendation.
    req_string_for_retrieval = state.get("requirement_string", "") or json.dumps(requirements)
    similar_cases = retrieve_similar_cases(
        req_string_for_retrieval,
        pipeline="recommend",
        requirements=requirements,
    )
    top3, case_adaptation_notes = _adapt_from_cases(top3, requirements, similar_cases)
    case_context_block = "\n".join(f"CBR adaptation: {note}." for note in case_adaptation_notes)
    attribution = dict(state.get("retrieval_attribution", {}))
    attribution["memory"] = bool(case_adaptation_notes)

    return {
        "valid": top3,
        "requirements": requirements,
        "req_string_for_retrieval": req_string_for_retrieval,
        "similar_cases": similar_cases,
        "case_context_block": case_context_block,
        "attribution": attribution,
    }


def _finalize_recommendation_state(
    state: LaptopState,
    *,
    valid: list,
    requirements: dict,
    analysis: str,
    conv_bot: list,
    conv_reco: list,
    comparison_table: str,
    req_string_for_retrieval: str,
    similar_cases: list,
    attribution: dict,
    kg_context: list,
    logged_answer: Optional[str] = None,
) -> LaptopState:
    """Shared tail: pick best-overall, log the turn, write/store case memory.

    logged_answer overrides what's written to the RAG eval log's "answer"
    field, for callers (e.g. compare_node) where the user-visible text is
    analysis + a table appended separately — state["comparison_analysis"]
    intentionally stays prose-only for other consumers (PDF generation,
    follow-up prompts), so this only affects what rag_evaluation.py sees,
    not app state.
    """
    best = valid[0] if valid else {}
    top_3 = json.dumps(valid)

    _log_rag_turn(
        pipeline="recommend",
        question=state.get("requirement_string", "") or json.dumps(requirements),
        contexts=[l.get("description", "") for l in valid] + kg_context,
        answer=logged_answer if logged_answer is not None else analysis,
        requirements=requirements,
        recommended=best,
        ranked=valid,
    )

    write_case(
        pipeline="recommend",
        query_text=req_string_for_retrieval,
        requirements=requirements,
        summary={
            "best_overall": best.get("name", ""),
            "top_3": [l.get("name", "") for l in valid],
        },
    )

    completed_state = {
        **state,
        "comparison_analysis": analysis,
        "best_overall": best,
        "top_3_laptops": top_3,
    }
    # Live learning: retain every completed recommendation with a measurable
    # proxy reward. Offline KG sweeps use a temporary action override and are
    # retained later with ground-truth correctness only for their winner.
    if not state.get("retrieval_action_override") and not state.get("offline_evaluation"):
        proxy_reward = compute_proxy_reward(completed_state)
        store_retrieval_case(completed_state, proxy_reward, reward_source="live_proxy")
        logger.info("[RETRIEVAL PLANNER] retained live proxy-reward case score=%.4f", proxy_reward)

    return {
        **completed_state,
        "comparison_table": comparison_table,
        "comparison_analysis": analysis,
        "best_overall": best,
        "top_3_laptops": top_3,
        "conversation_reco": conv_reco,
        "conversation_bot": conv_bot,
        "phase": "followup",
        "case_context": similar_cases,
        "retrieval_attribution": attribution,
    }


# Explicit "give me the full comparison" trigger — mirrors the pdf keyword
# shortcut in orchestrator_node. Checked against the message that completes
# the requirement-gathering turn (the one that actually reaches search_node).
# "vs"/"versus" two-laptop requests are intercepted earlier by the
# orchestrator's side_compare routing and never reach this check at all.
_FULL_COMPARE_KEYWORDS = (
    "compare", "comparison", "side by side", "side-by-side", "spec breakdown",
)


def route_after_search(state: LaptopState) -> str:
    """
    Default after search: a short best-first recommendation summary
    (recommend_summary_node) — NOT the full spec-table comparison.
    The full comparison (compare_node) only runs when the user actually
    asked to compare, same gating style as the pdf_report shortcut.
    """
    text = state.get("user_input", "").lower()
    if any(kw in text for kw in _FULL_COMPARE_KEYWORDS):
        logger.info("   Compare keyword detected — routing to full compare_node")
        return "compare_node"
    logger.info("   No compare keyword — routing to lightweight recommend_summary_node")
    return "recommend_summary_node"


def _sane_live_price(laptop: dict, low_ratio: float = 0.4, high_ratio: float = 2.5) -> Optional[dict]:
    """
    Guard against bad live-price scrapes (wrong product/variant matched,
    a truncated number parsed off the page, a currency mix-up, etc.).

    If the scraped figure is wildly out of line with our own catalog price,
    it's far more likely a bad match on live_pricing's side than a genuine
    steep discount — e.g. a ₹1,70,000-class laptop should not show up at
    ₹34,000. Showing that number erodes trust in the whole recommendation,
    so we fall back to the catalog price instead. Tune low_ratio/high_ratio
    if legitimate flash-sale discounts start getting discarded.
    """
    live = laptop.get("live_price")
    if not live or not live.get("price"):
        return None
    catalog_price = laptop.get("price", 0)
    if not catalog_price:
        return live  # nothing to sanity-check against — trust it
    ratio = live["price"] / catalog_price
    if ratio < low_ratio or ratio > high_ratio:
        logger.warning(
            "   Discarding implausible live price for %r: live=₹%s vs catalog=₹%s (source=%s, ratio=%.2fx)",
            laptop.get("name"), live["price"], catalog_price, live.get("source"), ratio,
        )
        return None
    return live


def _format_price_line(laptop: dict) -> str:
    """Deterministic price string — never hand this number to the LLM to restate."""
    live = _sane_live_price(laptop)
    if live:
        return f"₹{live['price']:,} (live, {live['source']})"
    return f"₹{laptop.get('price', 0):,}"


def recommend_summary_node(state: LaptopState) -> LaptopState:
    """
    Default post-search node: retrieves the top 3 ranked laptops (already
    best-first from search_node) and writes a short recommendation.
    No spec-comparison table — that's reserved for compare_node, which only
    runs when the user explicitly asks to compare.

    Output shape: a deterministic "Top Picks" list (name + price, built in
    code so numbers can't drift from what's actually in the catalog) followed
    by a short bullet-point summary from the LLM — never one free-form
    paragraph that risks restating figures (e.g. the budget) incorrectly.
    """
    logger.info("📝 [RECOMMEND SUMMARY NODE] started")

    ctx = _prepare_top3_context(state)
    if ctx is None:
        conv_bot = list(state.get("conversation_bot", []))
        conv_bot.append({"bot": "Sorry, no laptops matched your criteria. Please try again."})
        return {**state, "phase": "end", "conversation_bot": conv_bot}

    valid = ctx["valid"]
    requirements = ctx["requirements"]

    # Deterministic, best-first laptop list — built in code, not by the LLM,
    # so names/prices/rank order can never be mangled or hallucinated.
    picks_lines = [
        f"{i}. **{l.get('name', 'N/A')}** — {_format_price_line(l)}"
        for i, l in enumerate(valid, 1)
    ]
    picks_md = "## Top Picks\n\n" + "\n".join(picks_lines)

    try:
        llm = LLMFactory.get_llm(MAX_RECO, temperature=0.3)
        ranked_payload = [
            {"rank": i + 1, "name": l.get("name"), "score": l.get("score")}
            for i, l in enumerate(valid)
        ]
        prompt = (
            f"Here is a shortlist of laptops, already ranked best match first "
            f"(rank 1 = best fit):\n\n"
            f"User Requirements: {json.dumps(requirements, indent=2)}\n\n"
            f"Ranked Laptops:\n{json.dumps(ranked_payload, indent=2)}\n\n"
            f"{ctx['case_context_block'] + chr(10) + chr(10) if ctx['case_context_block'] else ''}"
            f"Write ONLY a bullet-point summary, one bullet per laptop in the same "
            f"rank order, each starting with the laptop's name in bold, explaining "
            f"in one short sentence why it fits (or doesn't quite fit) the "
            f"requirements. Do not restate prices or the budget figure — those are "
            f"shown separately. Do not add a title, a table, or any text before or "
            f"after the bullets."
        )
        bullet_summary = llm.invoke(prompt)
        bullet_summary = clean_llm_response(bullet_summary)
        if not bullet_summary or len(bullet_summary.strip()) < 10:
            bullet_summary = "\n".join(f"- **{l.get('name', 'N/A')}**: matches your requirements." for l in valid)
    except Exception as e:
        logger.error(f"   Recommend summary LLM failed: {e}")
        bullet_summary = "\n".join(f"- **{l.get('name', 'N/A')}**: matches your requirements." for l in valid)

    analysis = f"{picks_md}\n\n**Why these fit:**\n{bullet_summary}"

    best = valid[0]
    logger.info(f"   Best overall: {best.get('name')}")

    conv_bot = list(state.get("conversation_bot", []))
    conv_bot.append({"bot": analysis})

    products_str = json.dumps(valid, indent=2)
    conv_reco = [
        {"role": "system", "content": _reco_system.format(products=products_str)},
        {"role": "user", "content": "Please summarise these laptop recommendations for me."},
        {"role": "assistant", "content": analysis},
        {"role": "user", "content": f"This is my user profile: {state.get('requirement_string', '')}"},
        {"role": "assistant", "content": analysis},
    ]

    logger.info("✅ [RECOMMEND SUMMARY NODE] complete")

    return _finalize_recommendation_state(
        state,
        valid=valid,
        requirements=requirements,
        analysis=analysis,
        conv_bot=conv_bot,
        conv_reco=conv_reco,
        comparison_table="",
        req_string_for_retrieval=ctx["req_string_for_retrieval"],
        similar_cases=ctx["similar_cases"],
        attribution=ctx["attribution"],
        kg_context=state.get("kg_context", []),
    )


def compare_node(state: LaptopState) -> LaptopState:
    """
    Compare node: generates the FULL spec-comparison table + analysis for
    the top laptops. Only reached when the user explicitly asked to
    compare (see route_after_search) — same gating pattern as pdf_report.
    """
    logger.info("📊 [COMPARE NODE] started")

    ctx = _prepare_top3_context(state)
    if ctx is None:
        conv_bot = list(state.get("conversation_bot", []))
        conv_bot.append({"bot": "Sorry, no laptops matched your criteria. Please try again."})
        return {**state, "phase": "end", "conversation_bot": conv_bot}

    valid = ctx["valid"]
    requirements = ctx["requirements"]
    cols = min(3, len(valid))

    # Build comparison table
    table = "## Laptop Comparison\n\n"
    table += "| Feature | " + " | ".join(f"Laptop {i+1}" for i in range(cols)) + " |\n"
    table += "|---------|" + "|".join(["---------"] * cols) + "|\n"

    feat_rows = ["Name", "Price (INR)", "GPU intensity", "Display quality",
                 "Portability", "Multitasking", "Processing speed", "Score"]
    
    for feat in feat_rows:
        row = [feat]
        for laptop in valid[:cols]:
            if feat == "Name":
                row.append(laptop.get("name", "N/A"))
            elif feat == "Price (INR)":
                live = _sane_live_price(laptop)
                if live:
                    row.append(f"₹{live['price']:,} (live, {live['source']})")
                else:
                    row.append(f"₹{laptop.get('price', 0):,} (catalog)")
            elif feat == "Score":
                row.append(f"{laptop.get('score', 0)}/5")
            else:
                desc = laptop.get("description", "")
                feats = _FEATURE_CACHE.get(desc, {})
                row.append(feats.get(feat, "N/A"))
        table += "| " + " | ".join(row) + " |\n"

    # Generation receives only adaptation rationale, never a copied prior
    # answer/recommendation; current catalog evidence remains authoritative.
    case_context_block = ctx["case_context_block"]
    kg_context = state.get("kg_context", [])

    # Generate analysis with LLM
    try:
        llm = LLMFactory.get_llm(MAX_COMPARE, temperature=0.3)
        kg_context_block = "\n".join(kg_context[:10])
        
        prompt = (
            f"Compare these laptops against user requirements:\n\n"
            f"User Requirements: {json.dumps(requirements, indent=2)}\n\n"
            f"Top Laptops (already ranked best match first):\n{json.dumps([{'rank': i + 1, 'name': l.get('name'), 'price': l.get('price'), 'score': l.get('score')} for i, l in enumerate(valid)], indent=2)}\n\n"
            f"Knowledge-graph facts (brand, tiers, price band):\n{kg_context_block or '(no graph facts found)'}\n\n"
            f"{case_context_block + chr(10) + chr(10) if case_context_block else ''}"
            f"Provide:\n1. Brief summary of each laptop\n2. Which laptop best fits each requirement\n"
            f"3. Overall recommendation with justification\n4. Pros and cons of each"
        )
        
        analysis = llm.invoke(prompt)
        analysis = clean_llm_response(analysis)
        
        if not analysis or len(analysis.strip()) < 20:
            analysis = "Based on your requirements, here are the top laptop recommendations with their key specifications."
    except Exception as e:
        logger.error(f"   Compare LLM failed: {e}")
        analysis = "Here are the top laptop recommendations matching your criteria."

    recommendation = f"{analysis}\n\n{table}"
    conv_bot = list(state.get("conversation_bot", []))
    conv_bot.append({"bot": recommendation})

    products_str = json.dumps(valid, indent=2)
    conv_reco = [
        {"role": "system", "content": _reco_system.format(products=products_str)},
        {"role": "user", "content": "Please summarise these laptop recommendations for me."},
        {"role": "assistant", "content": recommendation},
        {"role": "user", "content": f"This is my user profile: {state.get('requirement_string', '')}"},
        {"role": "assistant", "content": recommendation},
    ]

    logger.info("✅ [COMPARE NODE] complete")

    return _finalize_recommendation_state(
        state,
        valid=valid,
        requirements=requirements,
        analysis=analysis,
        conv_bot=conv_bot,
        conv_reco=conv_reco,
        comparison_table=table,
        req_string_for_retrieval=ctx["req_string_for_retrieval"],
        similar_cases=ctx["similar_cases"],
        attribution=ctx["attribution"],
        kg_context=kg_context,
        logged_answer=recommendation,
    )


def followup_node(state: LaptopState) -> LaptopState:
    """
    Followup node: handles follow-up questions after recommendations.
    """
    logger.info("💬 [FOLLOWUP NODE] started")

    user_input = state.get("user_input", "")
    conv_reco = list(state.get("conversation_reco", []))
    conv_bot = list(state.get("conversation_bot", []))

    conv_reco.append({"role": "user", "content": user_input})
    conv_bot.append({"user": user_input})

    # Build context from conversation
    context_parts = []
    for m in conv_reco[-10:]:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            context_parts.append(f"User: {content}")
        elif role == "assistant":
            context_parts.append(f"Assistant: {content}")
    
    context = "\n".join(context_parts) if context_parts else "No previous context."
    context += f"\n\nCurrent question: {user_input}"

    try:
        llm = LLMFactory.get_llm(MAX_RECO, temperature=0.3)
        prompt = f"You are a helpful laptop shopping assistant. Answer the user's question based on the previous recommendations.\n\nConversation context:\n{context}\n\nAssistant response:"
        response = llm.invoke(prompt)
        response = clean_llm_response(response)
        
        if not response or len(response.strip()) < 10:
            response = "I'd be happy to help with that! Could you please clarify your question?"
    except Exception as e:
        logger.error(f"   Followup LLM failed: {e}")
        response = "I'd be happy to help with that. Could you please rephrase your question?"

    conv_reco.append({"role": "assistant", "content": response})
    conv_bot.append({"bot": response})

    return {
        **state,
        "conversation_reco": conv_reco,
        "conversation_bot": conv_bot,
        "phase": "followup",
    }


def moderation_end_node(state: LaptopState) -> LaptopState:
    """
    Moderation end node: resets the conversation when content is flagged.
    """
    logger.warning("🚫 [MODERATION END] Content flagged — resetting conversation")
    return {
        **state,
        "conversation_bot": [{"bot": "Your message was flagged. The conversation has been reset."}],
        "messages": [],
        "conversation_reco": [],
        "top_3_laptops": None,
        "requirements_complete": False,
        "collected_requirements": {},
        "pending_requirement": "",
        "requirement_retry_count": 0,
        "requirements": {},
        "requirement_string": "",
        "phase": "gather",
    }


# =============================================================================
# SIDE-BY-SIDE COMPARISON AGENT - Full Implementation
# =============================================================================

_PARSE_COMPARE_SYSTEM = """Extract the two laptop model names/keywords the user wants to compare.
Respond ONLY with valid JSON: {"laptop_a": "<name>", "laptop_b": "<name>"}
If you cannot find two names, use empty strings."""


def _get_all_laptop_names() -> List[str]:
    """
    Fetch all unique laptop names from Qdrant via scroll.
    """
    client = get_qdrant_client()
    if not client:
        return []
    
    try:
        all_names: set = set()
        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in results:
                name = r.payload.get("name", "").strip()
                if name:
                    all_names.add(name)
            if next_offset is None:
                break
            offset = next_offset
        return list(all_names)
    except Exception as e:
        logger.error(f"   scroll all names failed: {e}")
        return []


def _search_laptops_by_keyword(keyword: str, top_k: int = 4) -> List[str]:
    """
    Search for laptop names matching a keyword using substring matching.
    
    FIX (hallucination bug):
      The old code split the keyword into ALL words and matched ANY of them.
      "Dell XPS 15" → words = ["dell", "xps", "15"] → "dell" alone matched
      every Dell laptop in the database, returning irrelevant results.
      "MacBook Pro 14" → nothing matched → embedding fallback returned completely
      random laptops that have nothing to do with MacBook.
    
    New approach:
      1. Strip generic brand/filler words so only model-distinguishing terms remain
         (e.g. "Dell XPS 15" → ["xps", "15"]; "MacBook Pro" → ["macbook", "pro"]).
      2. Require ALL meaningful terms to appear in the laptop name (AND logic),
         not just any one of them (OR logic).
      3. If multi-word AND match finds nothing, fall back to single best term match.
      4. NO embedding fallback — returning semantically similar but wrong laptops
         is worse than returning nothing (caller shows "not found" message).
    """
    client = get_qdrant_client()
    if not client:
        return []

    # Words that alone are too generic to identify a specific model.
    _FILLER_WORDS = {
        "laptop", "notebook", "ultrabook", "computer", "pc", "gen",
        "the", "and", "with", "for", "best",
    }

    # Brands — useful only when no model-series word is available
    _BRAND_WORDS = {
        "dell", "hp", "lenovo", "asus", "acer", "apple", "msi",
        "samsung", "toshiba", "sony", "lg", "huawei", "honor",
        "microsoft", "razer", "gigabyte", "alienware", "vaio",
    }

    all_words = [w.lower() for w in keyword.split() if len(w) >= 2]
    meaningful = [w for w in all_words if w not in _FILLER_WORDS]
    brand_words = [w for w in meaningful if w in _BRAND_WORDS]
    model_words = [w for w in meaningful if w not in _BRAND_WORDS]

    logger.info(f"   Keyword='{keyword}' | meaningful={meaningful} | brand={brand_words} | model={model_words}")

    all_names = _get_all_laptop_names()

    # ── Stage 1: AND on model-series words + brand (most precise) ────────────
    if model_words and brand_words:
        combined = model_words + brand_words
        matches = [n for n in all_names if all(w in n.lower() for w in combined)]
        if matches:
            logger.info(f"   AND brand+model match for '{keyword}': {matches[:top_k]}")
            return matches[:top_k]

    # ── Stage 2: AND on model-series words only (ignore brand) ───────────────
    if model_words:
        matches = [n for n in all_names if all(w in n.lower() for w in model_words)]
        if matches:
            logger.info(f"   AND model-only match for '{keyword}': {matches[:top_k]}")
            return matches[:top_k]

    # ── Stage 3: brand-only query (e.g. bare "msi" or "dell") ───────────────
    if brand_words and not model_words:
        matches = [n for n in all_names if any(w in n.lower() for w in brand_words)]
        if matches:
            logger.info(f"   Brand-only OR match for '{keyword}': {matches[:top_k]}")
            return matches[:top_k]

    # ── Stage 4: single longest meaningful term (typo / partial match) ───────
    if meaningful:
        best_term = max(meaningful, key=len)
        matches = [n for n in all_names if best_term in n.lower()]
        if matches:
            logger.info(f"   Single-term fallback '{best_term}' for '{keyword}': {matches[:top_k]}")
            return matches[:top_k]

    # ── Nothing found — return empty so caller shows honest "not found" ───────
    logger.info(f"   No matches found for '{keyword}' — not in database")
    return []


def side_compare_parse_node(state: LaptopState) -> LaptopState:
    """
    Step 1 of side-compare pipeline.
    Extracts keywords from the user query, searches Qdrant for candidates,
    and either proceeds directly (single match) or asks the user to clarify.
    """
    logger.info("🔎 [SIDE COMPARE PARSE NODE] started")

    user_input = state.get("user_input", "")
    conv_bot = list(state.get("conversation_bot", []))
    conv_bot.append({"user": user_input})

    # ── Extract keywords ──────────────────────────────────────────────────────
    try:
        llm = LLMFactory.get_llm(MAX_ORCHESTRATOR, temperature=0.0, format_json=True)
        raw = llm.invoke(f"User input: {user_input}\n\nExtract two laptop names and return JSON.")
        raw = clean_llm_response(raw)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            keyword_a = parsed.get("laptop_a", "")
            keyword_b = parsed.get("laptop_b", "")
        else:
            keyword_a, keyword_b = "", ""
    except Exception as e:
        logger.warning(f"   Parse LLM failed: {e}")
        keyword_a, keyword_b = "", ""

    if not keyword_a or not keyword_b:
        conv_bot.append({"bot": (
            "I couldn't identify two laptop models to compare. "
            "Please say something like: *'Compare Dell XPS 15 vs MacBook Pro 14'.*"
        )})
        return {**state, "conversation_bot": conv_bot, "phase": "followup"}

    logger.info(f"   Keywords: '{keyword_a}' vs '{keyword_b}'")

    # ── Search Qdrant for each keyword ────────────────────────────────────────
    candidates_a = _search_laptops_by_keyword(keyword_a, top_k=4)
    candidates_b = _search_laptops_by_keyword(keyword_b, top_k=4)

    logger.info(f"   Candidates A: {candidates_a}")
    logger.info(f"   Candidates B: {candidates_b}")

    # ── Nothing found ─────────────────────────────────────────────────────────
    not_found = []
    if not candidates_a:
        not_found.append(keyword_a)
    if not candidates_b:
        not_found.append(keyword_b)
    
    if not_found:
        names = " and ".join(f"**{n}**" for n in not_found)
        sample_names = _get_all_laptop_names()
        sample_hint = ""
        if sample_names:
            sample = random.sample(sample_names, min(6, len(sample_names)))
            sample_hint = "\n\nHere are some examples of what's in our database:\n" + "\n".join(f"• {n}" for n in sorted(sample))
        conv_bot.append({"bot": (
            f"❌ I couldn't find any laptops matching {names} in our database. "
            f"The model name may be slightly different — try using the exact brand and series "
            f"(e.g. *'Asus VivoBook'*, *'Lenovo IdeaPad'*, *'HP Pavilion'*).{sample_hint}"
        )})
        return {**state, "conversation_bot": conv_bot, "phase": "followup"}

    # ── Exact single match on both sides — proceed immediately ────────────────
    if len(candidates_a) == 1 and len(candidates_b) == 1:
        logger.info("   Single match on both sides — skipping clarification")
        return {
            **state,
            "compare_laptops": [candidates_a[0], candidates_b[0]],
            "compare_keywords": [keyword_a, keyword_b],
            "compare_candidates": {},
            "conversation_bot": conv_bot,
            "phase": "side_compare",
        }

    # ── Multiple candidates — ask user to pick ────────────────────────────────
    msg = "I found a few options. Please tell me which ones you'd like to compare:\n\n"

    if len(candidates_a) == 1:
        msg += f"**For '{keyword_a}':** {candidates_a[0]}\n\n"
    else:
        msg += f"**For '{keyword_a}'** — pick one:\n"
        for i, name in enumerate(candidates_a, 1):
            msg += f"  {i}. {name}\n"
        msg += "\n"

    if len(candidates_b) == 1:
        msg += f"**For '{keyword_b}':** {candidates_b[0]}\n"
    else:
        msg += f"**For '{keyword_b}'** — pick one:\n"
        for i, name in enumerate(candidates_b, 1):
            msg += f"  {i}. {name}\n"

    msg += (
        "\nReply with your choices, e.g. *'1 and 2'*, *'option 3 and option 1'*, "
        "or the full model names."
    )

    conv_bot.append({"bot": msg})
    logger.info("   Awaiting user clarification")

    return {
        **state,
        "compare_keywords": [keyword_a, keyword_b],
        "compare_candidates": {"a": candidates_a, "b": candidates_b},
        "conversation_bot": conv_bot,
        "phase": "side_compare_clarify",
    }


def route_after_parse(state: LaptopState) -> str:
    """
    Route after parse node: go to agent only if we have two resolved laptop names.
    """
    phase = state.get("phase", "")
    compare_laptops = state.get("compare_laptops", [])
    if len(compare_laptops) == 2 and phase == "side_compare":
        return "side_compare_agent_node"
    return END


_CLARIFY_SYSTEM = """The user was shown two lists of laptop options and asked to pick one from each.
Given the candidate lists and the user's reply, identify which laptop they chose for list A and list B.
Respond ONLY with valid JSON: {"laptop_a": "<exact name from list A>", "laptop_b": "<exact name from list B>"}
If unclear, default to the first item in each list."""


def side_compare_clarify_node(state: LaptopState) -> LaptopState:
    """
    Step 2 of side-compare pipeline (only reached when clarification was needed).
    Resolves the user's selection to exact model names from the candidate lists,
    then hands off to side_compare_agent_node.
    
    FIX: resolution is constrained to the known candidate lists — the LLM can
    no longer hallucinate a name that was never offered to the user.
    """
    logger.info("✅ [SIDE COMPARE CLARIFY NODE] started")

    user_input = state.get("user_input", "")
    candidates = state.get("compare_candidates", {})
    keywords = state.get("compare_keywords", ["", ""])
    conv_bot = list(state.get("conversation_bot", []))
    conv_bot.append({"user": user_input})

    candidates_a = candidates.get("a", [])
    candidates_b = candidates.get("b", [])

    def _resolve(user_text: str, options: List[str]) -> str:
        """
        Resolve the user's reply to one of the known options.
        Priority:
          1. Substring match — handles pasted full names.
          2. Digit pick — handles '1', '2', 'option 3', etc.
          3. LLM fallback — output validated against options list.
          4. Default to first option.
        """
        lower = user_text.lower()

        # 1. Direct substring match
        for opt in options:
            if opt.lower() in lower or lower in opt.lower():
                return opt

        # 2. Digit selector
        digit_match = re.search(r'\b([1-9])\b', user_text)
        if digit_match:
            idx = int(digit_match.group(1)) - 1
            if 0 <= idx < len(options):
                return options[idx]

        # 3. LLM resolution — constrained to the known options
        try:
            llm = LLMFactory.get_llm(MAX_ORCHESTRATOR, temperature=0.0, format_json=True)
            raw = llm.invoke(f"Options: {options}\nUser reply: {user_text}\nReturn JSON: {{'selected': '<exact option>'}}")
            raw = clean_llm_response(raw)
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                selected = json.loads(json_match.group()).get("selected", "")
                for opt in options:
                    if opt.lower() == selected.lower() or selected.lower() in opt.lower():
                        return opt
        except Exception as e:
            logger.warning(f"   LLM resolution failed: {e}")

        # 4. Safe default
        return options[0] if options else ""

    laptop_a = _resolve(user_input, candidates_a)
    laptop_b = _resolve(user_input, candidates_b)

    logger.info(f"   Resolved: '{laptop_a}' vs '{laptop_b}'")

    if not laptop_a or not laptop_b:
        conv_bot.append({"bot": (
            "Sorry, I couldn't match your selection to any laptop in the list. "
            "Please reply with the number (e.g. *'4 and 1'*) or paste the full model name."
        )})
        return {**state, "conversation_bot": conv_bot, "phase": "side_compare_clarify"}

    conv_bot.append({"bot": f"Got it! Comparing **{laptop_a}** vs **{laptop_b}**…"})

    return {
        **state,
        "compare_laptops": [laptop_a, laptop_b],
        "compare_candidates": {},
        "conversation_bot": conv_bot,
        "phase": "side_compare",
    }


_SIDE_COMPARE_SYSTEM = """You are an expert laptop reviewer with deep technical knowledge.
The user wants a detailed side-by-side comparison of two specific laptops.

Structure your response as:

## {laptop_a} vs {laptop_b}

### 📋 Specs at a Glance
[Side-by-side spec table with: CPU, GPU, RAM, Storage, Display, Battery, Weight, Price range]

### ⚖️ Pros & Cons

**{laptop_a}**
✅ Pros: ...
❌ Cons: ...

**{laptop_b}**
✅ Pros: ...
❌ Cons: ...

### 🏆 Best for Each Use Case
- **Coding/Programming**: Winner — [name] because ...
- **Gaming**: Winner — [name] because ...
- **Video Editing**: Winner — [name] because ...
- **Machine Learning / AI**: Winner — [name] because ...
- **Portability / Travel**: Winner — [name] because ...

### 🎯 Overall Verdict
[2-3 sentences recommending who should buy which.]

IMPORTANT: Only use specs and details from the provided database context. Do NOT invent specs.
"""


def side_compare_agent_node(state: LaptopState) -> LaptopState:
    """
    Step 3 of side-compare pipeline.
    Fetches Qdrant context for the resolved laptop names and runs the LLM comparison.
    Only proceeds with laptops that exist in the database.
    """
    logger.info("📊 [SIDE COMPARE AGENT NODE] started")

    compare_laptops = state.get("compare_laptops", [])
    if len(compare_laptops) < 2:
        conv_bot = list(state.get("conversation_bot", []))
        conv_bot.append({"bot": "Please name two specific laptops to compare."})
        return {**state, "conversation_bot": conv_bot}

    laptop_a, laptop_b = compare_laptops[0], compare_laptops[1]

    def _fetch_context(name: str) -> str:
        """
        Fetch the database description for an exact laptop name.
        
        FIX: Use a scroll-based exact name match first (guaranteed to return
        the right laptop), falling back to embedding search only when not found.
        Trusting embedding similarity alone caused the wrong laptop's specs to
        be returned, leading to hallucinated comparison tables.
        """
        client = get_qdrant_client()
        if not client:
            return ""

        # Stage 1: scroll all records and match by exact name string
        try:
            offset = None
            while True:
                results, next_offset = client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    limit=250,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for r in results:
                    if r.payload.get("name", "").strip().lower() == name.strip().lower():
                        return r.payload.get("full_description", r.payload.get("description", ""))
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            logger.warning(f"   Exact scroll failed for '{name}': {e}")

        # Stage 2: embedding fallback (best-effort when exact name not found)
        logger.warning(f"   Exact match not found for '{name}', falling back to embedding search")
        try:
            emb = _embedder.embed_query(name)[:384]
            results = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=emb,
                limit=3,
                with_payload=True,
            ).points
            return " | ".join(
                r.payload.get("full_description", r.payload.get("description", ""))
                for r in results
            )
        except Exception as e:
            logger.error(f"   Embedding fallback failed for '{name}': {e}")
            return ""

    ctx_a = _fetch_context(laptop_a)
    ctx_b = _fetch_context(laptop_b)

    # KG context
    try:
        kg_result = kg_rag.kg_retrieve_for_names([laptop_a, laptop_b], hops=1)
        kg_context_block = "\n".join(kg_result["context"][:10])
    except Exception as e:
        logger.warning(f"   KG retrieval failed: {e}")
        kg_result = {"seed_nodes": [], "triplets": [], "context": []}
        kg_context_block = ""

    user_prompt = (
        f"Compare: {laptop_a} vs {laptop_b}\n\n"
        f"Database context for {laptop_a}:\n{ctx_a[:1500] if ctx_a else '(not found in database — do NOT invent specs)'}\n\n"
        f"Database context for {laptop_b}:\n{ctx_b[:1500] if ctx_b else '(not found in database — do NOT invent specs)'}\n\n"
        f"Knowledge-graph facts (brand, tiers, price band):\n{kg_context_block or '(no graph facts found)'}"
    )

    system = _SIDE_COMPARE_SYSTEM.format(laptop_a=laptop_a, laptop_b=laptop_b)
    
    try:
        llm = LLMFactory.get_llm(MAX_SIDE_COMPARE, temperature=0.3)
        response = llm.invoke(user_prompt)
        response = clean_llm_response(response)
        if not response or len(response.strip()) < 20:
            response = f"Here's a comparison of {laptop_a} and {laptop_b} based on available specs."
    except Exception as e:
        logger.error(f"   Side compare LLM failed: {e}")
        response = f"Here's the comparison of {laptop_a} vs {laptop_b}."

    logger.info("✅ [SIDE COMPARE AGENT NODE] complete")

    conv_bot = list(state.get("conversation_bot", []))
    conv_bot.append({"bot": response})

    conv_reco = list(state.get("conversation_reco", []))
    conv_reco += [
        {"role": "system", "content": f"You just compared {laptop_a} and {laptop_b} for the user."},
        {"role": "assistant", "content": response},
    ]

    # KG metrics
    try:
        kg_metrics = kg_rag.evaluate_kg_rag(
            seed_nodes=kg_result.get("seed_nodes", []),
            triplets=kg_result.get("triplets", []),
            answer=response,
        )
    except Exception as e:
        logger.warning(f"   KG metrics failed: {e}")
        kg_metrics = {}

    # Literal map
    try:
        literal_map = kg_rag.literal_sequential_map(f"{laptop_a} {laptop_b}")
        logger.info(f"   Literal sequential map: {len(literal_map)} token(s) matched to KG nodes")
    except Exception as e:
        logger.warning(f"   Literal map failed: {e}")
        literal_map = []

    # Log this turn for offline RAGAS scoring (see ragas_eval.py)
    _log_rag_turn(
        pipeline="side_compare",
        question=f"Compare: {laptop_a} vs {laptop_b}",
        contexts=[ctx_a, ctx_b] + kg_result.get("context", []),
        answer=response,
    )

    return {
        **state,
        "side_compare_result": response,
        "conversation_bot": conv_bot,
        "conversation_reco": conv_reco,
        "top_3_laptops": json.dumps([{"name": laptop_a}, {"name": laptop_b}]),
        "phase": "followup",
        "kg_seed_nodes": kg_result.get("seed_nodes", []),
        "kg_triplets": kg_result.get("triplets", []),
        "kg_context": kg_result.get("context", []),
        "kg_metrics": kg_metrics,
        "kg_literal_map": literal_map,
    }


# =============================================================================
# UPGRADE ADVISOR AGENT - Full Implementation
# =============================================================================

_UPGRADE_SYSTEM = """You are a laptop upgrade advisor. The user describes their current laptop.

Structure your response as:

## 🔄 Upgrade Analysis: {current}

### 📊 Current Performance Tier
[Rate the current laptop: Entry / Mid-range / High-end, with brief justification]

### 📈 Performance Gain if Upgraded
| Workload | Current | After Upgrade | Gain |
|----------|---------|---------------|------|
| Gaming   | ... | ... | ... |
| Editing  | ... | ... | ... |
| ML/AI    | ... | ... | ... |
| Multitask| ... | ... | ... |

### 💡 Worth Upgrading?
[Clear YES/NO/MAYBE with reasoning. Consider: age, workloads, budget.]

### 🏆 Better Alternatives
1. **[Model Name]** — [Why it's better, estimated price range]
2. **[Model Name]** — [Why it's better, estimated price range]
3. **[Model Name]** — [Why it's better, estimated price range]

### 🎯 Recommendation
[1-2 sentence actionable advice.]
"""


def upgrade_node(state: LaptopState) -> LaptopState:
    """
    Upgrade Advisor Agent: analyses the user's current laptop and advises.
    """
    logger.info("⬆️ [UPGRADE NODE] started")

    user_input = state.get("user_input", "")
    conv_bot = list(state.get("conversation_bot", []))
    conv_bot.append({"user": user_input})

    # Extract current laptop description from user input
    try:
        llm_parse = LLMFactory.get_llm(40, temperature=0.0, format_json=True)
        raw = llm_parse.invoke(f"Extract the laptop/GPU the user currently owns from: {user_input}\n\nReturn JSON: {{'current': '<description>'}}")
        raw = clean_llm_response(raw)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            current = json.loads(json_match.group()).get("current", user_input[:80])
        else:
            current = user_input[:80]
    except Exception as e:
        logger.warning(f"   Extract current laptop failed: {e}")
        current = user_input[:80]

    logger.info(f"   Current laptop: {current}")

    # Qdrant context — find similar/upgraded models
    def _fetch_upgrades(description: str) -> str:
        client = get_qdrant_client()
        if not client:
            return ""
        try:
            emb = _embedder.embed_query(description)[:384]
            results = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=emb,
                limit=3,
                with_payload=True,
            ).points
            return "\n".join(
                f"- {r.payload.get('name','')}: {r.payload.get('full_description', '')[:200]}"
                for r in results
            )
        except Exception as e:
            logger.warning(f"   Fetch upgrades failed: {e}")
            return ""

    qdrant_ctx = _fetch_upgrades(current)

    # KG context
    try:
        kg_result = kg_rag.kg_retrieve_free_text(current)
        if not kg_result.get("triplets"):
            logger.info("   KG free-text retrieval empty — falling back to local_search")
            local = kg_rag.local_search(current)
            kg_result = {**local, "seed_nodes": [local["seed_node"]] if local.get("seed_node") else []}
    except Exception as e:
        logger.warning(f"   KG retrieval failed: {e}")
        kg_result = {"seed_nodes": [], "triplets": [], "context": []}
    
    kg_context_block = "\n".join(kg_result.get("context", [])[:10])

    system = _UPGRADE_SYSTEM.format(current=current)
    prompt = (
        f"User's current laptop/GPU: {current}\n\n"
        f"Catalog context (potential upgrades from our database):\n{qdrant_ctx or 'Use your general knowledge.'}\n\n"
        f"Knowledge-graph facts about the current laptop's tier/brand and related models:\n"
        f"{kg_context_block or '(no graph facts found)'}\n\n"
        f"Full user message: {user_input}"
    )

    try:
        llm = LLMFactory.get_llm(MAX_UPGRADE, temperature=0.3)
        result = llm.invoke(prompt)
        result = clean_llm_response(result)
        if not result or len(result.strip()) < 20:
            result = f"Based on your current setup ({current}), here's my upgrade recommendation."
    except Exception as e:
        logger.error(f"   Upgrade LLM failed: {e}")
        result = f"Here's my upgrade advice for your current setup."

    logger.info("✅ [UPGRADE NODE] complete")

    conv_bot.append({"bot": result})

    conv_reco = list(state.get("conversation_reco", []))
    conv_reco += [
        {"role": "system", "content": f"You are a laptop upgrade advisor. You analysed the user's '{current}'."},
        {"role": "assistant", "content": result},
    ]

    # KG metrics
    try:
        kg_metrics = kg_rag.evaluate_kg_rag(
            seed_nodes=kg_result.get("seed_nodes", []),
            triplets=kg_result.get("triplets", []),
            answer=result,
        )
    except Exception as e:
        logger.warning(f"   KG metrics failed: {e}")
        kg_metrics = {}

    # Log this turn for offline RAGAS scoring (see ragas_eval.py)
    _log_rag_turn(
        pipeline="upgrade",
        question=user_input,
        contexts=([qdrant_ctx] if qdrant_ctx else []) + kg_result.get("context", []),
        answer=result,
    )

    return {
        **state,
        "current_laptop": current,
        "upgrade_advice": result,
        "conversation_bot": conv_bot,
        "conversation_reco": conv_reco,
        "top_3_laptops": json.dumps([{"name": "upgrade analysis"}]),
        "phase": "followup",
        "kg_seed_nodes": kg_result.get("seed_nodes", []),
        "kg_triplets": kg_result.get("triplets", []),
        "kg_context": kg_result.get("context", []),
        "kg_metrics": kg_metrics,
    }


# =============================================================================
# PDF REPORT AGENT - Full Implementation
# =============================================================================

def _build_pdf(state: LaptopState) -> str:
    """
    Generates a PDF report from the current session state, saves it to disk,
    and returns the file path.
    """
    os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"laptop_report_{timestamp}.pdf"
    filepath = os.path.join(PDF_OUTPUT_DIR, filename)

    # Build into a BytesIO buffer first
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           leftMargin=2*cm, rightMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#6c757d"),
        spaceAfter=16,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#0d6efd"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#6c757d"),
        spaceAfter=2,
    )

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("💻 Laptop Recommendation Report", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6")))
    story.append(Spacer(1, 12))

    # ── User Requirements ────────────────────────────────────────────────────
    requirements = state.get("requirements", {})
    requirement_str = state.get("requirement_string", "")
    if requirements:
        story.append(Paragraph("Your Requirements", section_style))
        if requirement_str:
            story.append(Paragraph(requirement_str, body_style))
            story.append(Spacer(1, 6))

        req_data = [["Dimension", "Level"]]
        for k, v in requirements.items():
            req_data.append([k, str(v).capitalize()])

        req_table = Table(req_data, colWidths=[9*cm, 7*cm])
        req_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(req_table)
        story.append(Spacer(1, 12))

    # ── Top Recommendations ──────────────────────────────────────────────────
    top_3_raw = state.get("top_3_laptops")
    top_3 = []
    if top_3_raw:
        try:
            top_3 = json.loads(top_3_raw)
        except Exception:
            pass

    if top_3:
        story.append(Paragraph("Top Recommended Laptops", section_style))
        for i, laptop in enumerate(top_3[:3], 1):
            name = laptop.get("name", f"Laptop {i}")
            live = _sane_live_price(laptop)
            price = live["price"] if live else laptop.get("price", 0)
            price_label = f"₹{price:,}" + (f" (live, {live['source']})" if live else " (catalog)")
            score = laptop.get("score", "N/A")
            desc = laptop.get("description", "")[:300]

            story.append(Paragraph(f"#{i}  {name}", ParagraphStyle(
                f"LapHead{i}",
                parent=styles["Heading3"],
                fontSize=11,
                textColor=colors.HexColor("#1a1a2e"),
                spaceBefore=8,
            )))
            if price:
                story.append(Paragraph(f"Price: {price_label}   |   Match Score: {score}/5", label_style))
            if desc:
                story.append(Paragraph(desc, body_style))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 6))

    # ── Comparison Analysis ──────────────────────────────────────────────────
    analysis = state.get("comparison_analysis", "") or state.get("side_compare_result", "") or state.get("upgrade_advice", "")
    if analysis:
        story.append(Paragraph("Detailed Analysis", section_style))
        # Strip markdown to plain text for PDF
        plain = re.sub(r"[#*`|>\-]+", "", analysis)
        plain = re.sub(r"\n{3,}", "\n\n", plain).strip()
        for para in plain.split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para, body_style))
        story.append(Spacer(1, 12))

    # ── Conversation History fallback ────────────────────────────────────────
    if not analysis and not top_3:
        conv_bot_history = state.get("conversation_bot", [])
        chat_entries = conv_bot_history[1:] if len(conv_bot_history) > 1 else []
        if chat_entries:
            story.append(Paragraph("Conversation Summary", section_style))
            for entry in chat_entries:
                if "user" in entry:
                    txt = re.sub(r"[#*`|>\-]+", "", str(entry["user"])).strip()
                    if txt:
                        story.append(Paragraph(
                            f"<b>You:</b> {txt}",
                            ParagraphStyle("UserMsg", parent=body_style,
                                           textColor=colors.HexColor("#0d6efd"), spaceAfter=4)
                        ))
                elif "bot" in entry:
                    txt = re.sub(r"<[^>]+>", "", str(entry["bot"]))
                    txt = re.sub(r"[#*`|>\-]+", "", txt)
                    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
                    for para in txt.split("\n\n"):
                        para = para.strip()
                        if para:
                            story.append(Paragraph(para, body_style))
                    story.append(Spacer(1, 6))
            story.append(Spacer(1, 12))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dee2e6")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report was generated by the Laptop Shopping Assistant.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#adb5bd"), alignment=TA_CENTER)
    ))

    doc.build(story)

    # Write the buffer contents to the actual file on disk
    buffer.seek(0)
    with open(filepath, "wb") as f:
        f.write(buffer.read())

    logger.info(f"   PDF saved to {filepath}")
    return filepath


def pdf_report_node(state: LaptopState) -> LaptopState:
    """
    PDF Report Agent: generates a downloadable PDF of the session results.
    """
    logger.info("📄 [PDF REPORT NODE] started")

    conv_bot = list(state.get("conversation_bot", []))
    user_input = state.get("user_input", "")
    conv_bot.append({"user": user_input})

    # has_data: any of the structured fields OR a meaningful conversation exists
    has_structured = bool(
        state.get("top_3_laptops")
        or state.get("side_compare_result")
        or state.get("upgrade_advice")
        or state.get("comparison_analysis")
    )
    has_conversation = len(state.get("conversation_bot", [])) > 1
    has_data = has_structured or has_conversation

    if not has_data:
        msg = ("There's nothing to export yet — please have a conversation first, "
               "then ask me to generate a PDF.")
        conv_bot.append({"bot": msg})
        return {**state, "conversation_bot": conv_bot}

    try:
        filepath = _build_pdf(state)
        filename = os.path.basename(filepath)
        pdf_url = f"/static/reports/{filename}"

        msg = (
            f"📄 <strong>Your PDF report is ready!</strong><br><br>"
            f"<a href='{pdf_url}' download target='_blank'>"
            f"⬇️ Download Report</a><br><br>"
            f"The report includes your requirements, top laptop recommendations, "
            f"and the full analysis."
        )
        conv_bot.append({"bot": msg})

        logger.info("✅ [PDF REPORT NODE] complete")
        return {
            **state,
            "pdf_path": filepath,
            "pdf_url": pdf_url,
            "conversation_bot": conv_bot,
            "phase": "followup",
        }
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        conv_bot.append({"bot": f"Sorry, I couldn't generate the PDF: {e}"})
        return {**state, "conversation_bot": conv_bot}


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_moderation(state: LaptopState) -> Literal["intent_node", "moderation_end_node"]:
    """Route after moderation node."""
    if state.get("moderation_result") == "flagged":
        return "moderation_end_node"
    return "intent_node"


def route_after_intent(state: LaptopState) -> Literal["conversation_node", "search_node"]:
    """Route after intent node."""
    if state.get("requirements_complete"):
        return "search_node"
    return "conversation_node"


def route_after_compare(state: LaptopState) -> Literal["followup_node", str]:
    """Route after compare node."""
    if state.get("phase") == "end":
        return END
    return "followup_node"


# =============================================================================
# BUILD GRAPH
# =============================================================================

def _build_graph():
    """Build the LangGraph state graph."""
    g = StateGraph(LaptopState)

    # ── Register all nodes ───────────────────────────────────────────────────
    g.add_node("orchestrator_node", orchestrator_node)

    # Original recommendation pipeline
    g.add_node("conversation_node", conversation_node)
    g.add_node("moderation_node", moderation_node)
    g.add_node("intent_node", intent_node)
    g.add_node("search_node", search_node)
    g.add_node("recommend_summary_node", recommend_summary_node)
    g.add_node("compare_node", compare_node)
    g.add_node("followup_node", followup_node)
    g.add_node("moderation_end_node", moderation_end_node)

    # New specialist agents
    g.add_node("side_compare_parse_node", side_compare_parse_node)
    g.add_node("side_compare_clarify_node", side_compare_clarify_node)
    g.add_node("side_compare_agent_node", side_compare_agent_node)
    g.add_node("upgrade_node", upgrade_node)
    g.add_node("pdf_report_node", pdf_report_node)

    # ── Entry point: always orchestrator ─────────────────────────────────────
    g.set_entry_point("orchestrator_node")

    # Orchestrator fans out
    g.add_conditional_edges(
        "orchestrator_node",
        route_after_orchestrator,
        {
            "conversation_node": "conversation_node",
            "followup_node": "followup_node",
            "side_compare_parse_node": "side_compare_parse_node",
            "side_compare_clarify_node": "side_compare_clarify_node",
            "upgrade_node": "upgrade_node",
            "pdf_report_node": "pdf_report_node",
        }
    )

    # ── Recommendation pipeline ──────────────────────────────────────────────
    g.add_edge("conversation_node", "moderation_node")
    g.add_conditional_edges("moderation_node", route_after_moderation, {
        "intent_node": "intent_node",
        "moderation_end_node": "moderation_end_node",
    })
    g.add_conditional_edges("intent_node", route_after_intent, {
        "conversation_node": END,
        "search_node": "search_node",
    })
    g.add_conditional_edges("search_node", route_after_search, {
        "compare_node": "compare_node",
        "recommend_summary_node": "recommend_summary_node",
    })
    g.add_conditional_edges("compare_node", route_after_compare, {
        "followup_node": END,
        END: END,
    })
    g.add_conditional_edges("recommend_summary_node", route_after_compare, {
        "followup_node": END,
        END: END,
    })
    g.add_edge("followup_node", END)
    g.add_edge("moderation_end_node", END)

    # ── Side-by-side comparison pipeline ────────────────────────────────────
    g.add_conditional_edges("side_compare_parse_node", route_after_parse, {
        "side_compare_agent_node": "side_compare_agent_node",
        END: END,
    })
    g.add_edge("side_compare_clarify_node", "side_compare_agent_node")
    g.add_edge("side_compare_agent_node", END)

    # ── Upgrade advisor pipeline ─────────────────────────────────────────────
    g.add_edge("upgrade_node", END)

    # ── PDF report pipeline ──────────────────────────────────────────────────
    g.add_edge("pdf_report_node", END)

    return g.compile()


# Compiled graph — single instance at module load
laptop_graph = _build_graph()


# =============================================================================
# HELPER SEARCH FUNCTIONS
# =============================================================================

def _dense_search(query_embedding: list, budget: int, top_k: int = 10) -> list:
    """Perform dense vector search in Qdrant."""
    client = get_qdrant_client()
    if client is None:
        return []
    
    filt = Filter(must=[FieldCondition(key="price", range=Range(lte=float(budget)))]) if budget > 0 else None
    
    try:
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            query_filter=filt,
            limit=top_k,
            with_payload=True,
        ).points

        out = [{
            "laptop": {
                # payload["laptop_id"] is the real laptop identity — r.id is
                # just a sequential counter over CHUNKS (see chunk_qdrant_pytorch.py:
                # PointStruct(id=idx, ...)), so two chunks of the same laptop
                # would otherwise look like two different laptops downstream
                # (RRF fusion dedups on this id). Falls back to r.id for any
                # non-chunked collection that doesn't set laptop_id.
                "id": r.payload.get("laptop_id", r.id),
                "name": r.payload.get("name", ""),
                "price": r.payload.get("price", 0),
                "description": r.payload.get("full_description", r.payload.get("description", "")),
            },
            "score": r.score,
        } for r in results]

        logger.info(f"   [DENSE SEARCH] budget<={budget} -> {len(out)} chunks retrieved")
        for i, item in enumerate(out, 1):
            lap = item["laptop"]
            desc_preview = (lap["description"] or "")[:80].replace("\n", " ")
            logger.debug(
                f"      #{i} id={lap['id']} score={item['score']:.4f} "
                f"name={lap['name']!r} price={lap['price']} desc={desc_preview!r}..."
            )
        return out
    except Exception as e:
        logger.error(f"Dense search error: {e}")
        return []


def _sparse_search(query: str, budget: int, top_k: int = 10) -> list:
    """Perform sparse (keyword-based) search in Qdrant."""
    client = get_qdrant_client()
    if client is None:
        return []
    
    filt = Filter(must=[FieldCondition(key="price", range=Range(lte=float(budget)))]) if budget > 0 else None
    
    try:
        scroll_results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=filt,
            limit=top_k * 3,
            with_payload=True,
        )
        
        query_terms = set(query.lower().split())
        scored = []
        
        for point in scroll_results[0]:
            desc = point.payload.get("full_description", point.payload.get("description", "")).lower()
            score = sum(1 for t in query_terms if t in desc) / max(len(query_terms), 1)
            scored.append({
                "laptop": {
                    # same laptop_id vs point.id distinction as _dense_search above
                    "id": point.payload.get("laptop_id", point.id),
                    "name": point.payload.get("name", ""),
                    "price": point.payload.get("price", 0),
                    "description": point.payload.get("full_description", point.payload.get("description", "")),
                },
                "score": score,
            })
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]

        logger.info(
            f"   [SPARSE SEARCH] scanned {len(scroll_results[0])} points, "
            f"{len(query_terms)} query terms -> top {len(top)} chunks kept"
        )
        for i, item in enumerate(top, 1):
            lap = item["laptop"]
            desc_preview = (lap["description"] or "")[:80].replace("\n", " ")
            logger.debug(
                f"      #{i} id={lap['id']} score={item['score']:.4f} "
                f"name={lap['name']!r} price={lap['price']} desc={desc_preview!r}..."
            )
        return top
    except Exception as e:
        logger.error(f"Sparse search error: {e}")
        return []


def _rrf_fuse(dense: list, sparse: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion for combining dense and sparse search results."""
    scores: Dict[str, float] = defaultdict(float)
    laptop_map = {}
    
    for lst in [dense, sparse]:
        for rank, item in enumerate(lst, 1):
            lid = item["laptop"]["id"]
            scores[lid] += 1.0 / (k + rank)
            if lid not in laptop_map:
                laptop_map[lid] = item["laptop"]
    
    merged = [{"laptop": laptop_map[lid], "rrf_score": s} for lid, s in scores.items()]
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)

    logger.info(
        f"   [RRF FUSE] dense={len(dense)} + sparse={len(sparse)} chunks -> "
        f"{len(merged)} unique after fusion"
    )
    for i, item in enumerate(merged[:10], 1):
        lap = item["laptop"]
        logger.debug(f"      #{i} id={lap['id']} rrf_score={item['rrf_score']:.5f} name={lap['name']!r}")

    return merged


# =============================================================================
# HELPER EXTRACTION FUNCTIONS
# =============================================================================

_REQ_PATTERN = re.compile(
    r"I need a laptop with\s+(\w+)\s+GPU intensity[,\s]+(\w+)\s+display quality[,\s]+"
    r"(\w+)\s+portability[,\s]+(\w+)\s+multitasking[,\s]+(\w+)\s+processing speed"
    r"[\s\w]*budget\s+of\s+([\d,]+)",
    re.IGNORECASE,
)


def parse_budget(text: str) -> Optional[int]:
    parsed = _parse_budget_units(text)
    if parsed is not None:
        return parsed
    """Deterministically find the budget number the user actually stated,
    across all the formats we support. Returns None if no number is found —
    callers must NOT invent a default budget when this returns None."""
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*lakh", text, re.IGNORECASE)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100000)

    inr_match = re.findall(r"₹\s*([\d,]+)", text)
    if inr_match:
        return int(inr_match[0].replace(",", ""))

    dollar_match = re.findall(r"\$\s*([\d,]+)", text)
    if dollar_match:
        return int(int(dollar_match[0].replace(",", "")) * USD_TO_INR)

    k_match = re.search(r"(\d+(?:\.\d+)?)\s*[kK]\b", text)
    if k_match:
        num = float(k_match.group(1)) * 1000
        # "5k dollars" vs "5k rupees" — assume USD if "dollar"/"$" nearby, else INR
        if re.search(r"dollar|usd|\$", text, re.IGNORECASE):
            num *= USD_TO_INR
        return int(num)

    plain_match = re.findall(r"\b(\d[\d,]{4,})\b", text)
    if plain_match:
        return max(int(n.replace(",", "")) for n in plain_match)

    return None


def _parse_budget_units(text: str) -> Optional[int]:
    """Parse currency/unit pairs without splitting ₹80k into the value 80."""
    pattern = re.compile(
        r"(?:(?P<currency>₹|rs\.?|inr|\$|usd|dollars?)\s*)?"
        r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>lakh|lac|k|thousand)?",
        re.IGNORECASE,
    )
    values: List[int] = []
    # '$' is a regex anchor/symbol boundary in some tokenizers; parse it
    # explicitly so "$750" reliably becomes INR 63,750.
    for amount_text, unit in re.findall(r"\$\s*(\d+(?:\.\d+)?)\s*(lakh|lac|k|thousand)?", text, re.IGNORECASE):
        multiplier = 100_000 if unit.lower() in {"lakh", "lac"} else 1_000 if unit.lower() in {"k", "thousand"} else 1
        values.append(int(float(amount_text) * multiplier * USD_TO_INR))
    for match in pattern.finditer(text.replace("$", "USD ")):
        amount = float(match.group("amount"))
        unit = (match.group("unit") or "").lower()
        currency = (match.group("currency") or "").lower()
        if not unit and not currency and amount < 1000:
            continue
        multiplier = 100_000 if unit in {"lakh", "lac"} else 1_000 if unit in {"k", "thousand"} else 1
        value = amount * multiplier
        if currency in {"$", "usd", "dollar", "dollars"}:
            value *= USD_TO_INR
        values.append(int(value))
    return max(values) if values else None


# Backwards-compatible private name for scripts that imported it. All new
# call sites use parse_budget(), the single budget source of truth.
def _detect_budget_number(text: str) -> Optional[int]:
    return parse_budget(text)


_BRAND_NAMES = ("dell", "hp", "lenovo", "asus", "acer", "apple", "msi", "razer", "samsung", "lg", "microsoft", "huawei")
_USE_CASE_SIGNALS = {
    "gaming": ("gaming", "esports", "fps", "aaa"),
    "creator": ("editing", "render", "design", "video", "photoshop"),
    "development": ("coding", "programming", "developer", "machine learning", "ml"),
    "business": ("office", "business", "productivity", "excel"),
    "student": ("student", "college", "school"),
}


def _decompose_query(text: str, requirements: dict) -> dict:
    """Cheap deterministic query decomposition used by every retriever.

    This deliberately augments the compatible five-tier schema instead of
    adding an LLM call.  It captures compound constraints as independently
    searchable fields and produces a canonical retrieval query.
    """
    lower = text.lower()
    out = dict(requirements)
    out["Brand"] = [b for b in _BRAND_NAMES if re.search(rf"\b{re.escape(b)}\b", lower)]
    out["Use cases"] = [name for name, terms in _USE_CASE_SIGNALS.items() if any(t in lower for t in terms)]
    gpu_match = re.search(r"\b(?:nvidia\s+)?(?:geforce\s+)?(rtx\s*\d{3,4}(?:\s*ti)?|gtx\s*\d{3,4}(?:\s*ti)?)\b", lower)
    out["Required GPU"] = re.sub(r"\s+", " ", gpu_match.group(1)).strip() if gpu_match else ""
    refresh_match = re.search(r"\b(\d{2,3})\s*hz\b", lower)
    out["Minimum refresh rate"] = int(refresh_match.group(1)) if refresh_match else 0
    out["Battery priority"] = "high" if any(t in lower for t in ("battery", "all day", "long battery")) else "medium"
    out["Portable priority"] = "high" if any(t in lower for t in ("portable", "travel", "lightweight", "thin")) else out.get("Portability", "medium")
    tokens = [
        f"budget under {out.get('Budget', '')}",
        *(f"brand {b}" for b in out["Brand"]),
        *(f"use {u}" for u in out["Use cases"]),
        *(f"required gpu {out['Required GPU']}" for _ in [0] if out["Required GPU"]),
        *(f"minimum refresh rate {out['Minimum refresh rate']} hz" for _ in [0] if out["Minimum refresh rate"]),
        *(f"{k} {out.get(k, 'medium')}" for k in ("GPU intensity", "Display quality", "Portability", "Multitasking", "Processing speed")),
        f"battery {out['Battery priority']}",
        f"original request {text}",
    ]
    out["Retrieval query"] = "; ".join(tokens)
    return out


def _extract_req_string(text: str) -> str:
    """Extract a structured requirement string from conversation text."""
    detected_budget = parse_budget(text)

    try:
        # If we can deterministically find a real budget number, use it
        # directly rather than routing through the LLM at all — this is
        # both faster and avoids the LLM echoing placeholder syntax instead
        # of substituting values (see BUG note on _req_string_chain above).
        if detected_budget is not None:
            return (
                f"I need a laptop with high GPU intensity, high display quality, "
                f"medium portability, medium multitasking, medium processing speed "
                f"and a budget of {detected_budget}."
            )

        result = _req_string_chain.invoke({"input": text})

        # Safety net: if the LLM echoed the placeholder syntax literally
        # (angle brackets) instead of substituting real values, this is a
        # broken result — do NOT use it, and do NOT silently default to
        # 100000. Fall through to the except block's honest fallback.
        if "<" in result or ">" in result:
            raise ValueError(f"LLM echoed placeholder syntax instead of real values: {result!r}")

        return result

    except Exception as e:
        logger.warning(f"   Req string extraction failed: {e}")
        # Use the actually-detected budget if we found one anywhere in the
        # text, even outside the try block's early-return path (e.g. if the
        # LLM path was attempted first for some other reason). Never
        # hardcode 100000 when we know better — only fall back to a
        # generic "medium" profile with no invented budget as a last resort.
        if detected_budget is not None:
            return (
                f"I need a laptop with high GPU intensity, high display quality, "
                f"medium portability, medium multitasking, medium processing speed "
                f"and a budget of {detected_budget}."
            )
        return "I need a laptop with medium GPU intensity, medium display quality, medium portability, medium multitasking, medium processing speed."


def _extract_requirements(text: str) -> dict:
    """Extract structured requirements from conversation text."""
    try:
        raw = _extraction_chain.invoke({"input": text})
        raw = clean_llm_response(raw)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            p = json.loads(json_match.group())
        else:
            p = {}
    except Exception as e:
        logger.warning(f"   Extraction failed: {e}")
        p = {}
    
    def _tier(key: str) -> str:
        v = str(p.get(key, "medium")).lower()
        return v if v in {"low", "medium", "high"} else "medium"

    # Budget detection — delegates to the single shared helper instead of
    # its own separate hardcoded-100000-default copy (which also had no "k"
    # handling at all, e.g. "5k" or "5k dollars" would never match any of its
    # patterns and silently fall back to 100000 regardless of what the user
    # said — this was the actual source of the Budget=100000 seen in search
    # cache keys even when a real budget had been extracted correctly
    # elsewhere).
    detected = parse_budget(text)
    budget = detected if detected is not None else 100000

    # Detect GPU intensity
    # The structured extractor is the primary signal. Deterministic explicit
    # spec mentions below may strengthen/override it, but must not discard it.
    gpu = _tier("GPU intensity")
    gpu_lower = text.lower()
    if re.search(r"\brtx\s*(?:40\d\d|30(?:[789]0))\b", gpu_lower) or any(kw in gpu_lower for kw in ["rtx 4090", "4090", "best gpu", "top gpu", "high end"]):
        gpu = "high"
    elif any(kw in gpu_lower for kw in ["rtx 3050", "3060", "mid range"]):
        gpu = "medium"
    elif any(kw in gpu_lower for kw in ["integrated", "uhd"]):
        gpu = "low"
    
    # Detect display quality
    display = _tier("Display quality")
    if any(kw in gpu_lower for kw in ["4k", "oled", "retina", "120hz", "144hz"]):
        display = "high"
    elif any(kw in gpu_lower for kw in ["fhd", "1080p", "ips"]):
        display = "medium"
    
    # Detect portability
    portability = _tier("Portability")
    if any(kw in gpu_lower for kw in ["ultrabook", "thin", "light", "under 1 kg"]):
        portability = "high"
    elif any(kw in gpu_lower for kw in ["workstation", "17 inch", "heavy"]):
        portability = "low"
    
    # Detect multitasking
    multitasking = _tier("Multitasking")
    if any(kw in gpu_lower for kw in ["64gb", "32gb"]):
        multitasking = "high"
    elif any(kw in gpu_lower for kw in ["16gb"]):
        multitasking = "medium"
    elif any(kw in gpu_lower for kw in ["8gb", "4gb"]):
        multitasking = "low"
    
    # Detect processing speed
    processing = _tier("Processing speed")
    if any(kw in gpu_lower for kw in ["i9", "ryzen 9", "m3 pro"]):
        processing = "high"
    elif any(kw in gpu_lower for kw in ["i7", "ryzen 7"]):
        processing = "medium"
    elif any(kw in gpu_lower for kw in ["i5", "ryzen 5", "celeron"]):
        processing = "low"
    
    requirements = {
        "GPU intensity": gpu,
        "Display quality": display,
        "Portability": portability,
        "Multitasking": multitasking,
        "Processing speed": processing,
        "Budget": budget,
    }
    return _decompose_query(text, requirements)


# =============================================================================
# INITIAL STATE FACTORY
# =============================================================================

def make_initial_state() -> LaptopState:
    """Create a blank state for a new session."""
    return LaptopState(
        messages=[],
        user_input="",
        moderation_result="ok",
        requirements_complete=False,
        requirement_string="",
        requirements={},
        collected_requirements={},
        pending_requirement="",
        requirement_retry_count=0,
        conversation_bot=[],
        top_3_laptops=None,
        conversation_reco=[],
        search_results=[],
        ranked_laptops=[],
        top_k_laptops=[],
        comparison_table="",
        comparison_analysis="",
        best_overall={},
        phase="gather",
        error="",
        last_response="",
        orchestrator_intent="recommend",
        compare_laptops=[],
        compare_keywords=[],
        compare_candidates={},
        side_compare_result="",
        current_laptop="",
        upgrade_advice="",
        pdf_path="",
        pdf_url="",
        kg_seed_nodes=[],
        kg_triplets=[],
        kg_context=[],
        case_context=[],
        retrieval_action={},
        retrieval_action_override={},
        offline_evaluation=False,
        retrieval_planner_cases=[],
        retrieval_planner_policy={},
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def run_turn(state: LaptopState, user_input: str) -> LaptopState:
    """Invoke the compiled LangGraph for one user turn."""
    updated = {**state, "user_input": user_input}
    result = laptop_graph.invoke(updated)
    return result
