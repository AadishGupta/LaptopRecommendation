"""
rag_evaluation.py — Unified RAG Evaluation Pipeline
=====================================================================
Combines every evaluation signal into ONE runnable script:

  1. Answer relevancy  — fast embedding similarity by default (~1-2 s/turn);
                         pass --ragas for full RAGAS library (slow).
                         Faithfulness is never run (NLI parse failures waste ~5 min/turn).
  2. LLM-as-a-Judge    — 9 criteria scored 1-10 each (see _JUDGE_CRITERIA)
  3. Ground Truth      — fuzzy correctness vs ground_truth.db
                         (field_agreement + price_closeness)
  4. Classification    — accuracy, precision, recall, F1 at threshold 0.65
  5. Retrieval         — Recall@K, Precision@K, MRR, nDCG@K
  6. Case Memory       — reads Qdrant case_memory, shows what was learned
  7. Recency windows   — last-1 / last-5-avg / last-10-avg for every metric
                         so you can see learning progress turn by turn

Input modes
-----------
  --mode log   score turns already in rag_eval_log.jsonl  (DEFAULT)
  --mode live  run eval_questions through the live app, log turns, then score
  --mode both  live first, then score the full log (incl. older turns)

Usage
-----
  python rag_evaluation.py                          # score existing log
  python rag_evaluation.py --mode live --limit 20  # 20 live questions
  python rag_evaluation.py --mode both --limit 50  # live + full log
  python rag_evaluation.py --no-ragas              # skip relevancy scoring entirely
  python rag_evaluation.py --ragas                 # full RAGAS library (slow; default is fast embeddings)
  python rag_evaluation.py --no-judge              # skip LLM judge
  python rag_evaluation.py --no-cache              # re-score even if cached

Outputs
-------
  unified_eval_report.json   — full machine-readable report
  unified_eval_report.txt    — human-readable summary with all tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Stub VertexAI at MODULE LOAD TIME — must happen before any ragas import
# anywhere in this process, including lazy imports inside functions.
# ---------------------------------------------------------------------------
def _stub_vertexai_now() -> None:
    import sys, types as _t
    key = "langchain_community.chat_models.vertexai"
    if key in sys.modules:
        return
    stub = _t.ModuleType(key)
    class _S:
        def __init__(self, *a, **k):
            raise RuntimeError("VertexAI stub — not used by this app")
    stub.ChatVertexAI = _S
    sys.modules[key] = stub

_stub_vertexai_now()   # runs immediately on import

# Suppress RAGAS's verbose output-parser retry warnings — they're expected
# when faithfulness is not used and clutter the console badly.
import logging as _logging
_logging.getLogger("ragas").setLevel(_logging.ERROR)
_logging.getLogger("ragas.metrics").setLevel(_logging.ERROR)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JUDGE_MODEL       = "gemini-3.1-flash-lite"
RAGAS_JUDGE_MODEL = "gemini-3.1-flash-lite"   # only used with --ragas (full library)
# Tried in order if JUDGE_MODEL 404s / is inaccessible for this key.
# NOTE (2026-07): gemini-2.5-flash / gemini-2.5-flash-lite have started 404ing
# early ("no longer available to new users") ahead of their official Oct 16
# 2026 shutdown — a known Google-side rollout issue, not a quota/key problem.
# gemini-2.0-flash / gemini-2.0-flash-lite were fully shut down June 1 2026
# (hence the 0-quota 429 if you try them). Use the Gemini 3.x line instead.
JUDGE_MODEL_FALLBACKS = ["gemini-3-flash", "gemini-2.5-flash-lite"]
EMBEDDING_MODEL   = "models/gemini-embedding-2"
CTX_WINDOW        = 4096

# Gemini API key: set the GEMINI_API_KEY (or GOOGLE_API_KEY) env var, or pass
# --gemini-api-key on the command line.
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

DEFAULT_LOG       = "rag_eval_log.jsonl"
DEFAULT_CACHE     = "rag_eval_score_cache.json"
DEFAULT_QUESTIONS = "eval_questions.json"
DEFAULT_GT_DB     = "ground_truth.db"
DEFAULT_OUT_JSON  = "unified_eval_report.json"
DEFAULT_OUT_TXT   = "unified_eval_report.txt"
RETRIEVAL_K       = 10
CORRECT_THRESHOLD = 0.65   # correctness >= this => "correct" prediction

DIMENSIONS = [
    "GPU intensity",
    "Display quality",
    "Portability",
    "Multitasking",
    "Processing speed",
]

# ---------------------------------------------------------------------------
# VertexAI stub — already applied at module load by _stub_vertexai_now()
# ---------------------------------------------------------------------------
def _stub_vertexai() -> None:
    _stub_vertexai_now()  # no-op after first call


# ---------------------------------------------------------------------------
# Gemini judge-LLM constructor — shared by the standalone judge and the
# full-RAGAS path.
#
# "Thinking" models spend part of max_output_tokens on hidden reasoning
# before writing the visible answer. With a small budget the reasoning
# alone can exhaust it, truncating the JSON we actually need and causing
# "Unterminated string" / "Expecting value" parse errors. Fix: minimize
# hidden reasoning and force clean JSON output (response_mime_type) so the
# whole budget goes to the answer.
#   - Gemini 3.x models control this via thinking_level ('low'/'medium'/'high')
#   - Gemini 2.5 models control this via thinking_budget (int token count, 0=off)
# We try each in turn and fall back to no thinking control at all if the
# installed langchain-google-genai version or model rejects both.
# ---------------------------------------------------------------------------
def _make_gemini_judge_llm(model_name: str, api_key: str, max_output_tokens: int = 1024):
    from langchain_google_genai import ChatGoogleGenerativeAI
    base_kwargs = dict(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
    )
    last_exc: Exception | None = None
    for extra in ({"thinking_level": "low"}, {"thinking_budget": 0}, {}):
        try:
            return ChatGoogleGenerativeAI(**extra, **base_kwargs)
        except Exception as exc:
            last_exc = exc
    raise last_exc


# ---------------------------------------------------------------------------
# Keyword classifier  (mirrors agent_functions._classify_one, no extra imports)
# ---------------------------------------------------------------------------
_KW_RULES = {
    "GPU intensity": {
        "high":   ["rtx 4090","rtx 4080","rtx 4070","rtx 4060","rx 7900","8gb vram","16gb vram"],
        "medium": ["rtx 3050","mx550","gtx 1650","rx 6600","iris xe"],
        "low":    ["intel uhd","integrated graphics","vega 8"],
    },
    "Display quality": {
        "high":   ["4k","oled","retina","120hz","144hz","2560x1600"],
        "medium": ["fhd","1920x1080","ips","1080p"],
        "low":    ["hd+","1366x768","tn panel","720p"],
    },
    "Portability": {
        "high":   ["ultrabook","under 1 kg","thin and light","slim"],
        "medium": ["1.5 kg","1.8 kg","2.0 kg"],
        "low":    ["2.5 kg","3 kg","workstation","17 inch"],
    },
    "Multitasking": {
        "high":   ["64gb ram","32gb ram","lpddr5x"],
        "medium": ["16gb ram","lpddr4x"],
        "low":    ["8gb ram","4gb ram","lpddr4"],
    },
    "Processing speed": {
        "high":   ["core i9","ryzen 9","m3 pro","m2 pro"],
        "medium": ["core i7","ryzen 7","m3","m2"],
        "low":    ["core i5","ryzen 5","core i3","celeron"],
    },
}

def _classify_one(description: str) -> dict:
    lower = (description or "").lower()
    out = {}
    for feat, tiers in _KW_RULES.items():
        matched = "medium"
        for tier in ("high", "low"):
            if any(kw in lower for kw in tiers[tier]):
                matched = tier
                break
        out[feat] = matched
    return out


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------
def _budget_ranges_from_db(db_path: str) -> Dict[str, Tuple[float, float]]:
    conn = sqlite3.connect(db_path)
    ranges: Dict[str, Tuple[float, float]] = {}
    for tier in ("low", "medium", "high"):
        row = conn.execute(
            "SELECT MIN(laptop_price), MAX(laptop_price) "
            "FROM ground_truth_qa WHERE price_tier = ?", (tier,)
        ).fetchone()
        ranges[tier] = (float(row[0] or 0), float(row[1] or 0))
    conn.close()
    return ranges

def _budget_tier(budget: float, ranges: Dict[str, Tuple[float, float]]) -> str:
    _, hi_low = ranges.get("low", (0, 0))
    if budget <= hi_low:
        return "low"
    _, hi_med = ranges.get("medium", (0, 0))
    if budget <= hi_med:
        return "medium"
    return "high"

def _combo_key(requirements: dict, ranges: Dict[str, Tuple[float, float]]) -> Tuple[str, dict]:
    combo = {d: str(requirements.get(d, "medium")).lower() for d in DIMENSIONS}
    bt = _budget_tier(float(requirements.get("Budget", 0)), ranges)
    combo["Budget tier"] = bt
    key = "|".join(f"{d}:{combo[d]}" for d in DIMENSIONS) + f"|Budget:{bt}"
    return key, combo


class _GTIndex:
    def __init__(self, db_path: str):
        self._by_combo: Dict[str, List[dict]] = {}
        if not os.path.exists(db_path):
            print(f"  WARNING: ground_truth.db not found at {db_path} — GT scoring disabled")
            return
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT combo_key, laptop_name, laptop_price, price_tier, "
            "match_score, requirements_json, context, ground_truth_answer "
            "FROM ground_truth_qa WHERE combo_key IS NOT NULL"
        )
        for r in cur.fetchall():
            row = dict(r)
            row["requirements"] = json.loads(row["requirements_json"] or "{}")
            self._by_combo.setdefault(row["combo_key"], []).append(row)
        conn.close()

    def lookup(self, key: str) -> List[dict]:
        return self._by_combo.get(key, [])

    @property
    def available(self) -> bool:
        return bool(self._by_combo)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_combo.values())

    @property
    def n_combos(self) -> int:
        return len(self._by_combo)


# ---------------------------------------------------------------------------
# Fallback parsers — reconstruct "requirements" / "recommended" / "ranked"
# from the raw question + answer text for log lines written before the
# app started persisting these fields explicitly. Question format is fixed:
# "... {level} {dimension}, {level} {dimension}, ... and a budget of {n}."
# Answer format is fixed: "## Top Picks\n\n1. **Name** — ₹Price ..."
# ---------------------------------------------------------------------------
_BUDGET_RE = re.compile(r"budget of\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
_PICK_RE = re.compile(
    r"^\s*\d+\.\s+\*\*(.+?)\*\*\s*[—\-–]\s*₹?\s*([\d,]+(?:\.\d+)?)",
    re.MULTILINE,
)


def _parse_requirements_from_question(question: str) -> Optional[dict]:
    if not question:
        return None
    req: Dict[str, str] = {}
    for dim in DIMENSIONS:
        m = re.search(rf"(low|medium|high)\s+{re.escape(dim.lower())}", question, re.IGNORECASE)
        if m:
            req[dim] = m.group(1).lower()
    bm = _BUDGET_RE.search(question)
    if bm:
        req["Budget"] = float(bm.group(1).replace(",", ""))
    # only usable if we recovered every dimension + budget
    if len(req) != len(DIMENSIONS) + 1:
        return None
    return req


def _find_context_description(name: str, contexts: List[str]) -> str:
    name_l = (name or "").strip().lower()
    if not name_l:
        return ""
    for ctx in contexts:
        if f"name: {name_l}" in ctx.lower():
            return ctx
    for ctx in contexts:
        if name_l in ctx.lower():
            return ctx
    return ""


def _parse_recommended_from_answer(answer: str, contexts: List[str]) -> Tuple[Optional[dict], List[dict]]:
    if not answer:
        return None, []
    picks: List[dict] = []
    for m in _PICK_RE.finditer(answer):
        name = m.group(1).strip()
        try:
            price = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        picks.append({
            "name": name,
            "price": price,
            "description": _find_context_description(name, contexts),
        })
    if not picks:
        return None, []
    return picks[0], picks


def _gt_score(
    recommended: dict,
    requirements: dict,
    gt_index: _GTIndex,
    ranges: Dict[str, Tuple[float, float]],
) -> Optional[dict]:
    if not gt_index.available:
        return None
    combo_key, combo = _combo_key(requirements, ranges)
    rows = gt_index.lookup(combo_key)
    if not rows:
        return None
    gt_row = rows[0]
    rec_features = _classify_one(recommended.get("description", ""))
    gt_ceiling = int(gt_row.get("match_score") or len(DIMENSIONS))
    fa = sum(1 for d in DIMENSIONS if rec_features.get(d) == combo.get(d))
    fa_frac = min(1.0, fa / gt_ceiling)
    tier_range = ranges.get(combo.get("Budget tier", "medium"), (0.0, 0.0))
    width = max(tier_range[1] - tier_range[0], 1.0)
    diff = abs(float(recommended.get("price", 0) or 0) - float(gt_row.get("laptop_price", 0) or 0))
    pc = max(0.0, 1.0 - diff / width)
    correctness = 0.7 * fa_frac + 0.3 * pc
    return {
        "correctness":             round(correctness, 4),
        "field_agreement":         fa,
        "field_agreement_ceiling": gt_ceiling,
        "field_agreement_frac":    round(fa_frac, 4),
        "price_closeness":         round(pc, 4),
        "combo_key":               combo_key,
        "combo":                   combo,
        "gt_laptop_name":          gt_row.get("laptop_name"),
        "gt_laptop_price":         float(gt_row.get("laptop_price") or 0),
        "recommended_name":        recommended.get("name"),
        "recommended_price":       float(recommended.get("price") or 0),
        "recommended_features":    rec_features,
        "gt_answer":               gt_row.get("ground_truth_answer", ""),
    }

# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def _retrieval_metrics(
    ranked: List[dict],
    gt_result: dict,
    ranges: Dict[str, Tuple[float, float]],
    k: int = RETRIEVAL_K,
) -> dict:
    combo   = gt_result["combo"]
    ceiling = gt_result["field_agreement_ceiling"]
    lo, hi  = ranges.get(combo.get("Budget tier", "medium"), (0.0, float("inf")))
    target  = (gt_result.get("gt_laptop_name") or "").strip().lower()

    relevance: List[float] = []
    exact_rank: Optional[int] = None
    first_rel: Optional[int] = None

    for rank, item in enumerate(ranked[:k], 1):
        feats = _classify_one(item.get("description", ""))
        agree = sum(feats.get(d) == combo.get(d) for d in DIMENSIONS)
        in_budget = lo <= float(item.get("price", 0) or 0) <= hi
        rel = min(1.0, agree / max(ceiling, 1)) if in_budget else 0.0
        relevance.append(rel)
        if rel >= 1.0 and first_rel is None:
            first_rel = rank
        if (item.get("name") or "").strip().lower() == target:
            exact_rank = rank

    while len(relevance) < k:
        relevance.append(0.0)

    dcg   = sum(r / math.log2(i + 2) for i, r in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg  = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    rel_count = sum(1 for r in relevance if r >= 1.0)

    return {
        f"recall_at_{k}":    float(first_rel is not None),
        f"precision_at_{k}": round(rel_count / k, 4),
        "mrr":               round(1.0 / first_rel, 4) if first_rel else 0.0,
        f"ndcg_at_{k}":      round(dcg / idcg, 4) if idcg else 0.0,
        "first_relevant_rank": first_rel,
        "exact_gt_rank":       exact_rank,
        "relevant_in_top_k":   rel_count,
    }


# ---------------------------------------------------------------------------
# LLM-as-a-Judge — 9 criteria
# ---------------------------------------------------------------------------
_JUDGE_CRITERIA = {
    "comprehensiveness": "Does the answer cover all relevant facts from the context?",
    "diversity":         "Does it draw on a variety of distinct facts/entities, not just one?",
    "empowerment":       "Does it give enough reasoning for the user to make their own decision?",
    "directness":        "Does it answer the exact question without padding or drift?",
    "factual_accuracy":  "Are all stated facts consistent with the provided context?",
    "specificity":       "Does it cite specific model names, prices, or specs rather than vague generalities?",
    "coherence":         "Is the answer well-structured, logical, and easy to follow?",
    "conciseness":       "Does it avoid unnecessary repetition or excessive length?",
    "helpfulness":       "Would a real user find this answer genuinely useful for buying a laptop?",
}

_JUDGE_PROMPT = (
    "You are an evaluation judge for a laptop recommendation RAG system.\n"
    "Score the ANSWER on these 9 criteria, each from 1 (very poor) to 10 (excellent):\n\n"
    + "\n".join(f"  {k}: {v}" for k, v in _JUDGE_CRITERIA.items())
    + "\n\nQUESTION:\n{question}\n\nCONTEXT (first 1500 chars):\n{context}\n\nANSWER:\n{answer}\n\n"
    "Respond ONLY with a valid JSON object, no markdown, no explanation:\n"
    '{{"comprehensiveness":<int>,"diversity":<int>,"empowerment":<int>,"directness":<int>,'
    '"factual_accuracy":<int>,"specificity":<int>,"coherence":<int>,'
    '"conciseness":<int>,"helpfulness":<int>,"rationale":"<one sentence>"}}'
)

_JUDGE_FIELDS = list(_JUDGE_CRITERIA.keys())


def _extract_json_object(text: str) -> dict:
    """Best-effort extraction of a JSON object from a possibly truncated/
    noisy LLM response. Tries a straight parse first, then falls back to
    slicing out the outermost {...} span, then to closing an unterminated
    string/object so a partial-but-truncated response can still be salvaged.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*$",         "", text, flags=re.DOTALL).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")
    candidate = text[start:]

    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Truncated mid-response (common when a model's reasoning/thinking
    # tokens eat the output budget before the JSON finishes). Try trimming
    # back to the last complete "key": value pair and closing the object.
    last_comma = candidate.rfind(",")
    while last_comma != -1:
        repaired = candidate[:last_comma] + "}"
        try:
            return json.loads(repaired)
        except Exception:
            last_comma = candidate.rfind(",", 0, last_comma)
    raise ValueError("could not repair truncated JSON response")


def _content_to_text(content) -> str:
    """Normalize a LangChain message .content value to plain text.

    Older Gemini models return .content as a plain string. Newer Gemini
    3.x models can return it as a list of content blocks, e.g.
    [{"type": "text", "text": "..."}], or a mix of strings and dicts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", "") or "")
        return "".join(parts)
    return str(content) if content is not None else ""


def _invoke_with_rate_limit_retry(llm, prompt: str, max_retries: int = 3):
    """Call llm.invoke, retrying on 429/RESOURCE_EXHAUSTED with the
    server-suggested delay (or a fallback backoff) instead of failing the
    turn outright. Free-tier keys can be limited to just a few requests
    per minute, so a single 429 is expected, not an error worth giving up on.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                raise
            last_exc = exc
            if attempt == max_retries:
                break
            m = re.search(r"retry in ([\d.]+)s", msg, flags=re.IGNORECASE)
            delay = float(m.group(1)) + 1.0 if m else (5.0 * (attempt + 1))
            print(f"    rate limited, waiting {delay:.0f}s before retry ({attempt+1}/{max_retries}) ...")
            time.sleep(delay)
    raise last_exc


def _llm_judge(question: str, context: str, answer: str, llm) -> dict:
    """Run 9-criterion LLM judge. Returns {} silently on any failure."""
    if llm is None or not answer.strip():
        return {}
    prompt = _JUDGE_PROMPT.format(
        question=question,
        context=context[:1500],
        answer=answer,
    )
    try:
        raw  = _invoke_with_rate_limit_retry(llm, prompt)
        text = _content_to_text(raw.content) if hasattr(raw, "content") else str(raw)
        parsed = _extract_json_object(text)
        scores = {f: int(parsed.get(f, 0)) for f in _JUDGE_FIELDS}
        valid  = [v for v in scores.values() if v > 0]
        scores["judge_avg"]     = round(sum(valid) / len(valid), 2) if valid else 0.0
        scores["judge_max"]     = max(valid) if valid else 0
        scores["judge_min"]     = min(valid) if valid else 0
        scores["rationale"]     = parsed.get("rationale", "")
        scores["n_criteria"]    = len(valid)
        # composite sub-scores
        scores["retrieval_quality"] = round(
            sum([scores["comprehensiveness"], scores["factual_accuracy"], scores["specificity"]]) / 3, 2
        )
        scores["answer_quality"] = round(
            sum([scores["directness"], scores["coherence"], scores["conciseness"], scores["helpfulness"]]) / 4, 2
        )
        return scores
    except Exception as exc:
        print(f"    WARNING judge failed: {exc}")
        return {}

# ---------------------------------------------------------------------------
# Relevancy scoring — fast embedding path (default) or full RAGAS (--ragas)
#
# Default: cosine similarity between question & answer embeddings (~1-2 s/turn).
# --ragas:  full RAGAS answer_relevancy (LLM question generation, ~2-5 min/turn).
# Faithfulness is never run here — its NLI prompts fail on local models and
# waste ~5 min/turn with RagasOutputParserException retries.
# ---------------------------------------------------------------------------
_ragas_llm_w = None
_ragas_emb_w = None
_gemini_emb = None

def _get_gemini_embeddings():
    global _gemini_emb
    if _gemini_emb is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        _gemini_emb = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY,
        )
    return _gemini_emb


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def _fast_relevancy_batch(items: List[Tuple[str, str]]) -> List[dict]:
    """
    Batch embedding relevancy for many (question, answer) pairs.
    Much faster than per-turn RAGAS — two embed calls per turn, no LLM judge.
    """
    if not items:
        return []
    emb = _get_gemini_embeddings()
    q_texts = [q[:500] for q, _ in items]
    a_texts = [a[:2000] for _, a in items]
    try:
        q_vecs = emb.embed_documents(q_texts)
        a_vecs = emb.embed_documents(a_texts)
    except Exception as exc:
        print(f"    WARNING fast relevancy batch failed: {exc}")
        return [{} for _ in items]

    out: List[dict] = []
    for qv, av in zip(q_vecs, a_vecs):
        sim = _cosine(qv, av)
        # map cosine [-1,1] → [0,1] for RAGAS-comparable scale
        ar = round(max(0.0, min(1.0, (sim + 1.0) / 2.0)), 3)
        out.append({
            "answer_relevancy": ar,
            "faithfulness":     None,
            "method":           "embedding",
        })
    return out


def _get_ragas_components():
    global _ragas_llm_w, _ragas_emb_w
    _stub_vertexai()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if _ragas_llm_w is None:
        judge = _make_gemini_judge_llm(RAGAS_JUDGE_MODEL, GEMINI_API_KEY, max_output_tokens=1024)
        _ragas_llm_w = LangchainLLMWrapper(judge)
    if _ragas_emb_w is None:
        _ragas_emb_w = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
        )
    return _ragas_llm_w, _ragas_emb_w


def _ragas_score_batch(turns: List[dict]) -> List[dict]:
    """
    Score multiple turns in ONE ragas_evaluate call (--ragas only).
    faithfulness excluded — NLI parse failures waste minutes per turn.
    """
    valid_idx: List[int] = []
    questions: List[str] = []
    contexts_list: List[List[str]] = []
    answers: List[str] = []
    for i, turn in enumerate(turns):
        ctx = [c[:800] for c in turn.get("contexts", []) if c and c.strip()][:3]
        ans = (turn.get("answer") or "").strip()
        if ctx and ans:
            valid_idx.append(i)
            questions.append(turn.get("question", ""))
            contexts_list.append(ctx)
            answers.append(ans[:3000])

    results: List[dict] = [{} for _ in turns]
    if not valid_idx:
        return results

    try:
        os.environ.setdefault("RAGAS_DISABLE_PROGRESS_BARS", "true")
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy
        from ragas.run_config import RunConfig
        from datasets import Dataset

        llm_w, emb_w = _get_ragas_components()
        dataset = Dataset.from_dict({
            "question": questions,
            "contexts": contexts_list,
            "answer":   answers,
        })
        result = ragas_evaluate(
            dataset,
            metrics=[answer_relevancy],
            llm=llm_w,
            embeddings=emb_w,
            run_config=RunConfig(timeout=180, max_workers=1, max_retries=1),
        )
        df = result.to_pandas()
        for j, orig_i in enumerate(valid_idx):
            ar = float(df.iloc[j]["answer_relevancy"])
            results[orig_i] = {
                "answer_relevancy": round(ar, 3) if not math.isnan(ar) else None,
                "faithfulness":     None,
                "method":           "ragas",
            }
    except Exception as exc:
        print(f"    WARNING ragas batch failed: {exc}")
    return results


# ---------------------------------------------------------------------------
# Score cache — skip re-scoring identical turns on repeat runs
# ---------------------------------------------------------------------------
def _turn_cache_key(turn: dict) -> str:
    payload = json.dumps({
        "question":  turn.get("question", ""),
        "answer":    (turn.get("answer") or "")[:800],
        "timestamp": turn.get("timestamp", ""),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _load_score_cache(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _batch_relevancy_for_indices(
    turns: List[dict],
    indices: List[int],
    use_full_ragas: bool,
) -> Dict[int, dict]:
    """Score relevancy for turn indices; dedupe identical Q+A within the batch."""
    if not indices:
        return {}

    key_to_rep: Dict[str, int] = {}
    rep_indices: List[int] = []
    orig_to_rep: Dict[int, int] = {}

    for i in indices:
        turn = turns[i]
        dedupe_key = json.dumps({
            "q": turn.get("question", ""),
            "a": (turn.get("answer") or "")[:800],
        }, sort_keys=True)
        if dedupe_key not in key_to_rep:
            key_to_rep[dedupe_key] = len(rep_indices)
            rep_indices.append(i)
        orig_to_rep[i] = key_to_rep[dedupe_key]

    rep_turns = [turns[i] for i in rep_indices]
    if use_full_ragas:
        rep_scores = _ragas_score_batch(rep_turns)
    else:
        pairs = [(t.get("question", ""), t.get("answer", "")) for t in rep_turns]
        rep_scores = _fast_relevancy_batch(pairs)

    out: Dict[int, dict] = {}
    for i in indices:
        rep_i = orig_to_rep[i]
        out[i] = rep_scores[rep_i] if rep_i < len(rep_scores) else {}
    return out


def _save_score_cache(path: str, cache: dict) -> None:
    try:
        Path(path).write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"  WARNING cache save failed: {exc}")


# ---------------------------------------------------------------------------
# Case memory reader
# ---------------------------------------------------------------------------
def _read_case_memory() -> List[dict]:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
    except Exception:
        return []
    cases: List[dict] = []
    try:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name="case_memory", limit=256,
                offset=offset, with_payload=True, with_vectors=False,
            )
            for p in points:
                pay = p.payload or {}
                cases.append({
                    "query_text":   pay.get("query_text", ""),
                    "pipeline":     pay.get("pipeline", ""),
                    "correctness":  pay.get("correctness"),
                    "timestamp":    pay.get("timestamp", ""),
                    "summary":      pay.get("summary", {}),
                    "requirements": pay.get("requirements", {}),
                })
            if offset is None:
                break
    except Exception as exc:
        print(f"  WARNING case_memory read: {exc}")
    return cases


def _case_memory_stats(cases: List[dict]) -> dict:
    if not cases:
        return {"total": 0}
    scored = [c for c in cases if c.get("correctness") is not None]
    vals   = [c["correctness"] for c in scored]
    above  = [v for v in vals if v >= CORRECT_THRESHOLD]
    by_pipe: Dict[str, List[float]] = defaultdict(list)
    for c in scored:
        by_pipe[c.get("pipeline", "unknown")].append(c["correctness"])
    return {
        "total":              len(cases),
        "scored":             len(scored),
        "unscored":           len(cases) - len(scored),
        "avg_correctness":    round(sum(vals)/len(vals), 4) if vals else None,
        "min_correctness":    round(min(vals), 4) if vals else None,
        "max_correctness":    round(max(vals), 4) if vals else None,
        "high_quality_cases": len(above),
        "high_quality_rate":  round(len(above)/len(scored), 4) if scored else None,
        "by_pipeline": {
            pipe: {"n": len(v), "avg": round(sum(v)/len(v), 4)}
            for pipe, v in by_pipe.items()
        },
    }

# ---------------------------------------------------------------------------
# Recency windows: last-1 / last-5-avg / last-10-avg
# ---------------------------------------------------------------------------
def _recency_windows(results: List[dict], field: str) -> dict:
    """
    Given a list of scored result dicts, extract the last-1, last-5, last-10
    values for `field` and compute averages. Works for any numeric field.
    """
    vals = [(i, r[field]) for i, r in enumerate(results) if r.get(field) is not None]
    if not vals:
        return {"last_1": None, "last_5_avg": None, "last_10_avg": None}

    def _avg(subset):
        v = [x[1] for x in subset]
        return round(sum(v) / len(v), 4) if v else None

    last_1  = vals[-1][1]
    last_5  = vals[-5:]
    last_10 = vals[-10:]
    return {
        "last_1":     round(last_1, 4),
        "last_5_avg": _avg(last_5),
        "last_10_avg": _avg(last_10),
        "last_5_n":   len(last_5),
        "last_10_n":  len(last_10),
    }


def _recency_block(results: List[dict]) -> dict:
    """Build recency windows for every key metric."""
    def _judge(field):
        return [(i, r.get("judge_scores", {}).get(field))
                for i, r in enumerate(results)]

    def _ragas(field):
        return [(i, r.get("ragas_scores", {}).get(field))
                for i, r in enumerate(results)]

    def _ret(field):
        return [(i, r.get("retrieval_metrics", {}).get(field))
                for i, r in enumerate(results)]

    def _window(pairs):
        pairs = [(i, v) for i, v in pairs if v is not None]
        if not pairs:
            return {"last_1": None, "last_5_avg": None, "last_10_avg": None}
        def avg(s): return round(sum(x[1] for x in s)/len(s), 4) if s else None
        return {
            "last_1":      round(pairs[-1][1], 4),
            "last_5_avg":  avg(pairs[-5:]),
            "last_10_avg": avg(pairs[-10:]),
        }

    block = {}
    # ground truth
    block["correctness"]      = _window([(i, r.get("correctness"))      for i, r in enumerate(results)])
    block["field_agreement_frac"] = _window([(i, r.get("field_agreement_frac")) for i, r in enumerate(results)])
    block["price_closeness"]  = _window([(i, r.get("price_closeness"))  for i, r in enumerate(results)])
    # retrieval
    block[f"recall_at_{RETRIEVAL_K}"]    = _window(_ret(f"recall_at_{RETRIEVAL_K}"))
    block[f"precision_at_{RETRIEVAL_K}"] = _window(_ret(f"precision_at_{RETRIEVAL_K}"))
    block["mrr"]                         = _window(_ret("mrr"))
    block[f"ndcg_at_{RETRIEVAL_K}"]      = _window(_ret(f"ndcg_at_{RETRIEVAL_K}"))
    # judge
    for f in _JUDGE_FIELDS + ["judge_avg", "retrieval_quality", "answer_quality"]:
        block[f"judge_{f}"] = _window(_judge(f))
    # ragas
    block["faithfulness"]     = _window(_ragas("faithfulness"))
    block["answer_relevancy"] = _window(_ragas("answer_relevancy"))
    return block

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _iso_now() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat()

def _append_to_log(path: str, record: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"  WARNING log append: {exc}")

def _safe_avg(vals: list) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 4) if v else None

def _safe_std(vals: list) -> Optional[float]:
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    mean = sum(v) / len(v)
    return round((sum((x - mean)**2 for x in v) / len(v)) ** 0.5, 4)


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------
def _binary_clf_metrics(results: List[dict]) -> dict:
    scored = [r for r in results if r.get("correctness") is not None]
    if not scored:
        return {"n": 0}
    pred_pos = [r for r in scored if r["correctness"] >= CORRECT_THRESHOLD]
    pred_neg = [r for r in scored if r["correctness"] <  CORRECT_THRESHOLD]
    def gt_pos(r):
        c = r.get("field_agreement_ceiling")
        return c is not None and c >= 4
    tp = sum(1 for r in pred_pos if gt_pos(r))
    fp = sum(1 for r in pred_pos if not gt_pos(r))
    fn = sum(1 for r in pred_neg if gt_pos(r))
    tn = sum(1 for r in pred_neg if not gt_pos(r))
    acc  = round((tp + tn) / len(scored), 4) if scored else None
    prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
    rec  = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
    f1   = round(2*prec*rec/(prec+rec), 4) if (prec and rec and (prec+rec) > 0) else None
    return {
        "n": len(scored), "threshold": CORRECT_THRESHOLD,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "positive_rate": round(len(pred_pos)/len(scored), 4),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(all_results: List[dict]) -> dict:
    gt_scored = [r for r in all_results if r.get("correctness") is not None]
    clf = _binary_clf_metrics(gt_scored)

    def _ret(field):
        return _safe_avg([r.get("retrieval_metrics", {}).get(field) for r in all_results])
    def _judge(field):
        return _safe_avg([r.get("judge_scores", {}).get(field) for r in all_results])
    def _ragas(field):
        return _safe_avg([r.get("ragas_scores", {}).get(field) for r in all_results])

    by_cat: Dict[str, list] = defaultdict(list)
    for r in all_results:
        by_cat[r.get("category", r.get("pipeline", "unknown"))].append(r)

    cat_summary = {}
    for cat, rows in sorted(by_cat.items()):
        cat_gt = [r for r in rows if r.get("correctness") is not None]
        cat_summary[cat] = {
            "n":                len(rows),
            "n_scored":         len(cat_gt),
            "avg_correctness":  _safe_avg([r["correctness"] for r in cat_gt]),
            "avg_fa":           _safe_avg([r.get("field_agreement") for r in cat_gt]),
            "avg_fa_frac":      _safe_avg([r.get("field_agreement_frac") for r in cat_gt]),
            "avg_pc":           _safe_avg([r.get("price_closeness") for r in cat_gt]),
            "avg_judge":        _safe_avg([r.get("judge_scores", {}).get("judge_avg") for r in rows]),
            "avg_faithfulness": _safe_avg([r.get("ragas_scores", {}).get("faithfulness") for r in rows]),
            "avg_ar":           _safe_avg([r.get("ragas_scores", {}).get("answer_relevancy") for r in rows]),
        }

    attr_sources = ("dense", "sparse", "kg", "memory", "reranker")
    attribution = {
        s: sum(1 for r in all_results if r.get("retrieval_attribution", {}).get(s))
        for s in attr_sources
    }

    c_vals = [r["correctness"] for r in gt_scored]
    return {
        "total_turns":     len(all_results),
        "gt_scored":       len(gt_scored),
        "unscored_no_gt":  sum(1 for r in all_results if "error" not in r and r.get("correctness") is None),
        "errors":          sum(1 for r in all_results if "error" in r),
        # correctness
        "avg_correctness": _safe_avg(c_vals),
        "std_correctness": _safe_std(c_vals),
        "min_correctness": round(min(c_vals), 4) if c_vals else None,
        "max_correctness": round(max(c_vals), 4) if c_vals else None,
        "avg_fa_frac":     _safe_avg([r.get("field_agreement_frac") for r in gt_scored]),
        "avg_price_closeness": _safe_avg([r.get("price_closeness") for r in gt_scored]),
        # classification
        "classification_metrics": clf,
        # retrieval
        f"avg_recall_at_{RETRIEVAL_K}":    _ret(f"recall_at_{RETRIEVAL_K}"),
        f"avg_precision_at_{RETRIEVAL_K}": _ret(f"precision_at_{RETRIEVAL_K}"),
        "avg_mrr":                         _ret("mrr"),
        f"avg_ndcg_at_{RETRIEVAL_K}":      _ret(f"ndcg_at_{RETRIEVAL_K}"),
        "avg_latency_ms":                  _safe_avg([r.get("latency_ms") for r in all_results]),
        # judge (all 9 + composites)
        **{f"avg_judge_{f}": _judge(f) for f in _JUDGE_FIELDS},
        "avg_judge_overall":          _judge("judge_avg"),
        "avg_judge_retrieval_quality":_judge("retrieval_quality"),
        "avg_judge_answer_quality":   _judge("answer_quality"),
        # ragas
        "avg_faithfulness":     _ragas("faithfulness"),
        "avg_answer_relevancy": _ragas("answer_relevancy"),
        # recency
        "recency_windows":      _recency_block(all_results),
        # breakdown
        "by_category":          cat_summary,
        "kg_changed_top_rate":  (
            round(sum(1 for r in gt_scored if r.get("kg_changed_top"))/len(gt_scored), 4)
            if gt_scored else None
        ),
        "retrieval_attribution_counts": attribution,
    }

# ---------------------------------------------------------------------------
# Score the JSONL log  (log mode)
# Only scores the last `score_last` turns — enough to compute last-1 /
# last-5-avg / last-10-avg windows without re-running all 100+ old turns.
# ---------------------------------------------------------------------------
def _score_log(
    log_path: str,
    use_relevancy: bool,
    use_full_ragas: bool,
    judge_llm,
    score_last: int = 10,
    cache_path: str = DEFAULT_CACHE,
    use_cache: bool = True,
    gt_index: Optional["_GTIndex"] = None,
    budget_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[dict]:
    path = Path(log_path)
    if not path.exists():
        print(f"  Log not found: {log_path}")
        return []

    all_turns = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_turns.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    turns = all_turns[-score_last:]
    print(f"\nLog has {len(all_turns)} total turns.")
    print(f"Scoring last {len(turns)} turn(s) (for last-1 / last-5 / last-10 windows) ...")

    cache = _load_score_cache(cache_path) if use_cache else {}
    n_cached = 0
    judge_mem: Dict[str, dict] = {}

    # Pre-compute relevancy in one batch (fast path) or one RAGAS call (--ragas)
    relevancy_by_idx: List[dict] = [{} for _ in turns]
    if use_relevancy:
        need_rel: List[int] = []
        for i, turn in enumerate(turns):
            key = _turn_cache_key(turn)
            cached = cache.get(key, {})
            if use_cache and cached.get("ragas_scores"):
                relevancy_by_idx[i] = cached["ragas_scores"]
                n_cached += 1
            elif (turn.get("answer") or "").strip() and any(
                c and c.strip() for c in turn.get("contexts", [])
            ):
                need_rel.append(i)

        if need_rel:
            n_unique = len({json.dumps({"q": turns[i].get("question",""), "a": (turns[i].get("answer") or "")[:800]}, sort_keys=True) for i in need_rel})
            if use_full_ragas:
                print(f"  Running full RAGAS on {n_unique} unique turn(s) ({len(need_rel)} total, slow) ...")
            else:
                print(f"  Fast embedding relevancy on {n_unique} unique turn(s) ({len(need_rel)} total) ...")
            scored_map = _batch_relevancy_for_indices(turns, need_rel, use_full_ragas)
            for orig_i in need_rel:
                relevancy_by_idx[orig_i] = scored_map.get(orig_i, {})

    if n_cached:
        print(f"  Reused cached scores for {n_cached} turn(s).")

    results: List[dict] = []

    for i, turn in enumerate(turns, 1):
        pipeline  = turn.get("pipeline", "unknown")
        question  = turn.get("question", "")
        contexts  = [c for c in turn.get("contexts", []) if c and c.strip()]
        answer    = turn.get("answer",   "")
        timestamp = turn.get("timestamp", "")
        cache_key = _turn_cache_key(turn)

        print(f"  [{i:>3}/{len(turns)}] {pipeline}  q={question[:55]}")

        judge_scores: dict = {}
        ragas_scores: dict = relevancy_by_idx[i - 1]

        if answer.strip():
            cached = cache.get(cache_key, {}) if use_cache else {}
            if use_cache and cached.get("judge_scores"):
                judge_scores = cached["judge_scores"]
            elif cache_key in judge_mem:
                judge_scores = judge_mem[cache_key]
            elif judge_llm is not None:
                judge_scores = _llm_judge(question, "\n".join(c[:800] for c in contexts[:3]), answer[:2500], judge_llm)
                judge_mem[cache_key] = judge_scores

        if use_cache and (judge_scores or ragas_scores):
            cache[cache_key] = {
                "question": question,
                "timestamp": timestamp,
                "judge_scores": judge_scores,
                "ragas_scores": ragas_scores,
            }

        # GT scoring — only possible for turns logged after this fix, which
        # persist "requirements" + "recommended" (+ "ranked") alongside the
        # answer. Older log lines won't have these keys and fall back to None.
        gt_result:   Optional[dict] = None
        retrieval_m: dict = {}
        requirements = turn.get("requirements") or _parse_requirements_from_question(question)
        recommended  = turn.get("recommended")
        ranked       = turn.get("ranked")
        if not recommended or not ranked:
            parsed_rec, parsed_ranked = _parse_recommended_from_answer(answer, contexts)
            recommended = recommended or parsed_rec
            ranked      = ranked or parsed_ranked
        ranked = ranked or ([recommended] if recommended else [])
        if gt_index is not None and gt_index.available and recommended and requirements:
            gt_result = _gt_score(recommended, requirements, gt_index, budget_ranges or {})
            if gt_result:
                retrieval_m = _retrieval_metrics(ranked, gt_result, budget_ranges or {})

        row = {
            "source":        "log",
            "pipeline":      pipeline,
            "category":      pipeline,
            "question":      question,
            "answer_length": len(answer),
            "n_contexts":    len(contexts),
            "timestamp":     timestamp,
            "latency_ms":    None,
            "correctness":             gt_result["correctness"]             if gt_result else None,
            "field_agreement":         gt_result["field_agreement"]         if gt_result else None,
            "field_agreement_frac":    gt_result["field_agreement_frac"]    if gt_result else None,
            "field_agreement_ceiling": gt_result["field_agreement_ceiling"] if gt_result else None,
            "price_closeness":         gt_result["price_closeness"]         if gt_result else None,
            "combo_key":               gt_result["combo_key"]               if gt_result else None,
            "recommended_name":        gt_result["recommended_name"]        if gt_result else (recommended.get("name") if recommended else None),
            "gt_laptop_name":          gt_result["gt_laptop_name"]          if gt_result else None,
            "retrieval_metrics":       retrieval_m,
            "retrieval_attribution":   {},
            "kg_changed_top":          None,
            # scored
            "judge_scores": judge_scores,
            "ragas_scores": ragas_scores,
        }
        results.append(row)

        parts = []
        js = judge_scores
        if js.get("judge_avg"):
            parts.append(f"judge_avg={js['judge_avg']}")
            parts.append(f"helpfulness={js.get('helpfulness')}")
            parts.append(f"factual_acc={js.get('factual_accuracy')}")
        rs = ragas_scores
        if rs.get("answer_relevancy") is not None:
            method = rs.get("method", "ragas")
            parts.append(f"AR={rs['answer_relevancy']} ({method})")
        print(f"         {' | '.join(parts) if parts else 'no scores (empty answer)'}")

    if use_cache:
        _save_score_cache(cache_path, cache)

    return results

# ---------------------------------------------------------------------------
# Live eval: push eval_questions through agent_functions.run_turn
# ---------------------------------------------------------------------------
def _stratified_sample(questions: List[dict], limit: int) -> List[dict]:
    by_cat: Dict[str, List[dict]] = {}
    for q in questions:
        by_cat.setdefault(q.get("category", "?"), []).append(q)
    cats = list(by_cat.keys())
    sampled: List[dict] = []
    i = 0
    while len(sampled) < limit and any(by_cat.values()):
        cat = cats[i % len(cats)]
        if by_cat[cat]:
            sampled.append(by_cat[cat].pop(0))
        i += 1
    return sampled[:limit]


def _run_live_eval(
    questions_path: str,
    gt_index: _GTIndex,
    budget_ranges: Dict[str, Tuple[float, float]],
    log_path: str,
    limit: Optional[int],
    use_relevancy: bool,
    use_full_ragas: bool,
    judge_llm,
    cache_path: str = DEFAULT_CACHE,
    use_cache: bool = True,
) -> List[dict]:
    import agent_functions as af

    print("\n" + "="*60)
    print("  Building vector store / KG ...")
    print("="*60)
    af.build_vector_store()

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if limit:
        questions = _stratified_sample(questions, limit)

    print(f"\nRunning {len(questions)} question(s) through live pipeline ...")
    live_results: List[dict] = []
    pending_for_rel: List[dict] = []
    pending_meta: List[dict] = []

    for i, item in enumerate(questions, 1):
        q        = item["question"]
        category = item.get("category", "unknown")
        print(f"\n[{i}/{len(questions)}] {category[:50]}")
        print(f"  Q: {q[:80]}")

        state = af.make_initial_state()
        state = af.run_turn(state, "Hello")

        t0 = time.perf_counter()
        try:
            state = af.run_turn(state, q)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            live_results.append({"source":"live","category":category,"question":q,"error":str(exc)})
            continue

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        requirements  = state.get("requirements", {}) or {}
        top_k_laptops = state.get("top_k_laptops") or []
        ranked        = state.get("ranked_laptops") or []
        recommended   = top_k_laptops[0] if top_k_laptops else None
        answer        = state.get("comparison_analysis") or state.get("last_response") or ""
        contexts      = [l.get("description","") for l in top_k_laptops[:5]]
        contexts     += state.get("kg_context", [])

        _append_to_log(log_path, {
            "pipeline": "recommend", "question": q,
            "contexts": contexts,    "answer":   answer,
            "timestamp": _iso_now(),
            "requirements": requirements,
            "recommended":  recommended,
            "ranked":       ranked[:RETRIEVAL_K],
        })

        gt_result:    Optional[dict] = None
        retrieval_m:  dict = {}
        if recommended and gt_index.available:
            gt_result = _gt_score(recommended, requirements, gt_index, budget_ranges)
            if gt_result:
                retrieval_m = _retrieval_metrics(ranked, gt_result, budget_ranges)
                try:
                    af.write_case(
                        pipeline="recommend",
                        query_text=state.get("requirement_string","") or q,
                        requirements=requirements,
                        summary={
                            "best_overall": recommended.get("name",""),
                            "top_3": [l.get("name","") for l in top_k_laptops[:3]],
                        },
                        correctness=gt_result["correctness"],
                    )
                except Exception:
                    pass

        judge_scores = _llm_judge(q, "\n".join(c[:800] for c in contexts[:3]), answer[:2500], judge_llm)

        turn_stub = {"question": q, "contexts": contexts, "answer": answer, "timestamp": _iso_now()}
        pending_for_rel.append(turn_stub)
        pending_meta.append({
            "source": "live", "category": category, "question": q,
            "answer_length": len(answer), "latency_ms": latency_ms,
            "n_contexts": len(contexts), "gt_result": gt_result,
            "retrieval_m": retrieval_m, "recommended": recommended,
            "state": state, "judge_scores": judge_scores,
            "turn_stub": turn_stub,
        })

    # Batch relevancy after all live turns (one embed batch, not N separate RAGAS calls)
    rel_scores: List[dict] = [{} for _ in pending_for_rel]
    if use_relevancy and pending_for_rel:
        indices = list(range(len(pending_for_rel)))
        if use_full_ragas:
            print(f"\n  Running full RAGAS on {len(pending_for_rel)} live turn(s) ...")
        else:
            print(f"\n  Fast embedding relevancy on {len(pending_for_rel)} live turn(s) ...")
        rel_map = _batch_relevancy_for_indices(pending_for_rel, indices, use_full_ragas)
        rel_scores = [rel_map.get(i, {}) for i in indices]

    cache = _load_score_cache(cache_path) if use_cache else {}
    for meta, rel in zip(pending_meta, rel_scores):
        gt_result   = meta["gt_result"]
        retrieval_m = meta["retrieval_m"]
        recommended = meta["recommended"]
        state       = meta["state"]
        judge_scores = meta["judge_scores"]
        q           = meta["question"]
        category    = meta["category"]

        row = {
            "source":        "live",
            "category":      category,
            "question":      q,
            "answer_length": meta["answer_length"],
            "latency_ms":    meta["latency_ms"],
            "n_contexts":    meta["n_contexts"],
            # GT
            "correctness":             gt_result["correctness"]             if gt_result else None,
            "field_agreement":         gt_result["field_agreement"]         if gt_result else None,
            "field_agreement_frac":    gt_result["field_agreement_frac"]    if gt_result else None,
            "field_agreement_ceiling": gt_result["field_agreement_ceiling"] if gt_result else None,
            "price_closeness":         gt_result["price_closeness"]         if gt_result else None,
            "combo_key":               gt_result["combo_key"]               if gt_result else None,
            "recommended_name":        gt_result["recommended_name"]        if gt_result else (recommended.get("name") if recommended else None),
            "recommended_price":       gt_result["recommended_price"]       if gt_result else (recommended.get("price") if recommended else None),
            "gt_laptop_name":          gt_result["gt_laptop_name"]          if gt_result else None,
            "gt_laptop_price":         gt_result["gt_laptop_price"]         if gt_result else None,
            "recommended_features":    gt_result.get("recommended_features") if gt_result else None,
            # retrieval + pipeline
            "retrieval_metrics":      retrieval_m,
            "retrieval_attribution":  dict(state.get("retrieval_attribution", {})),
            "kg_changed_top":         state.get("kg_changed_top"),
            "kg_weight_used":         state.get("kg_weight_used"),
            "phase":                  state.get("phase"),
            # scored
            "judge_scores": judge_scores,
            "ragas_scores": rel,
        }
        live_results.append(row)

        if use_cache:
            ck = _turn_cache_key(meta["turn_stub"])
            cache[ck] = {"judge_scores": judge_scores, "ragas_scores": rel}

        parts = []
        if gt_result:
            parts.append(f"correctness={gt_result['correctness']:.3f}  FA={gt_result['field_agreement']}/{gt_result['field_agreement_ceiling']}  PC={gt_result['price_closeness']:.3f}")
        js = judge_scores
        if js.get("judge_avg"):
            parts.append(f"judge_avg={js['judge_avg']}  helpfulness={js.get('helpfulness')}")
        if rel.get("answer_relevancy") is not None:
            parts.append(f"AR={rel['answer_relevancy']} ({rel.get('method', 'ragas')})")
        rm = retrieval_m
        if rm.get(f"ndcg_at_{RETRIEVAL_K}") is not None:
            parts.append(f"nDCG@10={rm[f'ndcg_at_{RETRIEVAL_K}']}")
        print(f"  => {chr(10).join(parts) if parts else 'no scores'}")

    if use_cache:
        _save_score_cache(cache_path, cache)

    return live_results

# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------
def _write_txt_report(summary: dict, case_stats: dict, all_results: List[dict], out_path: str) -> None:
    lines: List[str] = []

    def h(title):
        lines.append("\n" + "="*72)
        lines.append(f"  {title}")
        lines.append("="*72)

    def s(label, val, unit=""):
        v = f"{val}{unit}" if val is not None else "n/a"
        lines.append(f"  {label:<44} {v}")

    def rw(label, window_dict):
        w = window_dict or {}
        l1   = w.get("last_1",     "n/a")
        l5   = w.get("last_5_avg", "n/a")
        l10  = w.get("last_10_avg","n/a")
        lines.append(f"  {label:<44} last-1={l1}  last-5={l5}  last-10={l10}")

    lines.append("UNIFIED RAG EVALUATION REPORT")
    lines.append(f"Generated : {_iso_now()}")
    lines.append(f"Threshold : correctness >= {CORRECT_THRESHOLD} => 'correct'")

    h("OVERVIEW")
    s("Total turns evaluated",   summary["total_turns"])
    s("GT-scored turns",         summary["gt_scored"])
    s("Unscored (no GT combo)",  summary["unscored_no_gt"])
    s("Errors",                  summary["errors"])

    h("GROUND TRUTH CORRECTNESS  (0.7 x field_agreement + 0.3 x price_closeness)")
    s("Average correctness",        summary["avg_correctness"])
    s("Std dev correctness",        summary["std_correctness"])
    s("Min / Max correctness",      f"{summary['min_correctness']} / {summary['max_correctness']}")
    s("Avg field-agreement frac",   summary["avg_fa_frac"])
    s("Avg price closeness",        summary["avg_price_closeness"])

    h(f"CLASSIFICATION METRICS  (threshold={CORRECT_THRESHOLD})")
    clf = summary.get("classification_metrics", {})
    s("N scored",        clf.get("n"))
    s("TP / FP / FN / TN", f"{clf.get('tp')} / {clf.get('fp')} / {clf.get('fn')} / {clf.get('tn')}")
    s("Accuracy",        clf.get("accuracy"))
    s("Precision",       clf.get("precision"))
    s("Recall",          clf.get("recall"))
    s("F1 score",        clf.get("f1"))
    s("Positive rate",   clf.get("positive_rate"))

    h(f"RETRIEVAL METRICS  (top-{RETRIEVAL_K})")
    s(f"Avg Recall @ {RETRIEVAL_K}",    summary.get(f"avg_recall_at_{RETRIEVAL_K}"))
    s(f"Avg Precision @ {RETRIEVAL_K}", summary.get(f"avg_precision_at_{RETRIEVAL_K}"))
    s("Avg MRR",                        summary.get("avg_mrr"))
    s(f"Avg nDCG @ {RETRIEVAL_K}",      summary.get(f"avg_ndcg_at_{RETRIEVAL_K}"))
    s("Avg latency (ms)",               summary.get("avg_latency_ms"))

    h(f"LLM-AS-A-JUDGE  (1–10 each, {JUDGE_MODEL})")
    for f in _JUDGE_FIELDS:
        s(f"  {f}", summary.get(f"avg_judge_{f}"))
    s("Overall average",           summary.get("avg_judge_overall"))
    s("Retrieval quality composite", summary.get("avg_judge_retrieval_quality"))
    s("Answer quality composite",    summary.get("avg_judge_answer_quality"))

    h("ANSWER RELEVANCY  (embedding fast path by default; --ragas for full library)")
    s("Faithfulness",      summary.get("avg_faithfulness"), "  (disabled — NLI incompatible with local models)")
    s("Answer relevancy",  summary.get("avg_answer_relevancy"))

    h("RECENCY WINDOWS  (last-1 / last-5-avg / last-10-avg)")
    rw = _recency_block(all_results)
    lines.append(f"\n  {'Metric':<38} {'last-1':>8}  {'last-5-avg':>10}  {'last-10-avg':>11}")
    lines.append("  " + "-"*72)
    for key, w in rw.items():
        l1  = w.get("last_1",      "n/a")
        l5  = w.get("last_5_avg",  "n/a")
        l10 = w.get("last_10_avg", "n/a")
        lines.append(f"  {key:<38} {str(l1):>8}  {str(l5):>10}  {str(l10):>11}")

    h("CASE MEMORY  (Qdrant case_memory — what the system has learned)")
    s("Total cases stored",           case_stats.get("total"))
    s("Quality-scored cases",         case_stats.get("scored"))
    s("Unscored cases",               case_stats.get("unscored"))
    s("Avg stored correctness",       case_stats.get("avg_correctness"))
    s("Min / Max stored correctness", f"{case_stats.get('min_correctness')} / {case_stats.get('max_correctness')}")
    s(f"High-quality cases (>={CORRECT_THRESHOLD})", case_stats.get("high_quality_cases"))
    s("High-quality rate",            case_stats.get("high_quality_rate"))
    for pipe, pst in (case_stats.get("by_pipeline") or {}).items():
        s(f"  pipeline '{pipe}'  n={pst['n']}  avg", pst["avg"])

    h("KG FUSION")
    s("KG changed top result rate",  summary.get("kg_changed_top_rate"))
    for src, cnt in (summary.get("retrieval_attribution_counts") or {}).items():
        s(f"  {src} influenced turns", cnt)

    h("PER-CATEGORY BREAKDOWN")
    for cat, cs in (summary.get("by_category") or {}).items():
        lines.append(f"\n  {cat[:68]}")
        lines.append(f"    n={cs['n']}  scored={cs['n_scored']}  "
                     f"correctness={cs['avg_correctness']}  fa_frac={cs['avg_fa_frac']}  "
                     f"pc={cs['avg_pc']}  judge={cs['avg_judge']}  "
                     f"faith={cs['avg_faithfulness']}  AR={cs['avg_ar']}")

    scored_s = sorted(
        [r for r in all_results if r.get("correctness") is not None],
        key=lambda r: r["correctness"]
    )
    if scored_s:
        h("BEST 5 TURNS  (highest correctness)")
        for r in scored_s[-5:][::-1]:
            lines.append(f"  [{r['correctness']:.3f}]  {str(r.get('recommended_name','?'))[:42]:<42}  Q: {r['question'][:50]}")

        h("WORST 5 TURNS  (lowest correctness)")
        for r in scored_s[:5]:
            lines.append(f"  [{r['correctness']:.3f}]  {str(r.get('recommended_name','?'))[:42]:<42}  Q: {r['question'][:50]}")

    lines.append("\n" + "="*72)
    lines.append(f"  JSON report : {DEFAULT_OUT_JSON}")
    lines.append("="*72 + "\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Text report  -> {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    global GEMINI_API_KEY
    parser = argparse.ArgumentParser(
        description="Unified RAG evaluation: RAGAS + LLM judge + GT + case memory"
    )
    parser.add_argument("--mode", choices=["log","live","both"], default="log",
        help="log=score existing log (default)  live=run questions+score  both=live+full log")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--log",       default=DEFAULT_LOG)
    parser.add_argument("--gt-db",     default=DEFAULT_GT_DB)
    parser.add_argument("--out-json",  default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-txt",   default=DEFAULT_OUT_TXT)
    parser.add_argument("--limit",      type=int, default=None,
        help="Cap live questions (stratified across categories)")
    parser.add_argument("--score-last", type=int, default=10,
        help="In log mode: only score the last N turns from the log (default 10 — covers last-1/5/10 windows)")
    parser.add_argument("--cache", default=DEFAULT_CACHE,
        help="Score cache file (re-runs skip already-scored turns)")
    parser.add_argument("--no-cache", action="store_true",
        help="Ignore score cache and re-score everything")
    parser.add_argument("--ragas", action="store_true",
        help="Use full RAGAS answer_relevancy library (slow; ~2-5 min/turn)")
    parser.add_argument("--no-ragas", action="store_true",
        help="Skip answer relevancy scoring entirely")
    parser.add_argument("--no-judge",  action="store_true",
        help="Skip LLM-as-judge scoring")
    parser.add_argument("--gemini-api-key", default=GEMINI_API_KEY,
        help="Gemini API key (defaults to GEMINI_API_KEY/GOOGLE_API_KEY env var)")
    args = parser.parse_args()

    use_relevancy   = not args.no_ragas
    use_full_ragas  = args.ragas
    use_cache       = not args.no_cache

    # make the resolved key available to embedding/RAGAS helpers too
    GEMINI_API_KEY = args.gemini_api_key

    if use_relevancy:
        mode = "full RAGAS library (slow)" if use_full_ragas else "fast embedding similarity (default)"
        print(f"\nRelevancy scoring: {mode}")
    else:
        print("\nRelevancy scoring: disabled (--no-ragas)")

    # --- ground truth index ---
    print(f"\nLoading ground truth from {args.gt_db} ...")
    gt_index      = _GTIndex(args.gt_db)
    budget_ranges = _budget_ranges_from_db(args.gt_db) if gt_index.available else {}
    if gt_index.available:
        print(f"  {len(gt_index)} QA rows  |  {gt_index.n_combos} combos  |  ranges={budget_ranges}")

    # --- judge LLM ---
    judge_llm = None
    if not args.no_judge:
        if not args.gemini_api_key:
            print("\n  WARNING: no Gemini API key found (set GEMINI_API_KEY/GOOGLE_API_KEY "
                  "or pass --gemini-api-key) — judge disabled")
        else:
            candidates = [JUDGE_MODEL] + [m for m in JUDGE_MODEL_FALLBACKS if m != JUDGE_MODEL]
            for i, model_name in enumerate(candidates):
                print(f"\nLoading judge LLM ({model_name}) ...")
                try:
                    judge_llm = _make_gemini_judge_llm(model_name, args.gemini_api_key, max_output_tokens=1024)
                    # warm-up ping
                    judge_llm.invoke("ping")
                    print(f"  Judge LLM ready ({model_name}).")
                    break
                except Exception as exc:
                    is_last = (i == len(candidates) - 1)
                    if is_last:
                        print(f"  WARNING: could not load judge LLM: {exc}  (judge disabled)")
                        judge_llm = None
                    else:
                        print(f"  {model_name} unavailable ({exc}); trying next fallback ...")

    # --- full RAGAS availability check (only when --ragas) ---
    if use_full_ragas:
        print(f"\nChecking RAGAS library ({RAGAS_JUDGE_MODEL} judge) ...")
        _stub_vertexai()
        try:
            import ragas  # noqa
            print("  RAGAS library available.")
        except ImportError:
            print("  WARNING: ragas not installed (pip install ragas) — falling back to fast embedding relevancy.")
            use_full_ragas = False

    # --- run live if requested ---
    live_results: List[dict] = []
    if args.mode in ("live", "both"):
        live_results = _run_live_eval(
            questions_path=args.questions,
            gt_index=gt_index,
            budget_ranges=budget_ranges,
            log_path=args.log,
            limit=args.limit,
            use_relevancy=use_relevancy,
            use_full_ragas=use_full_ragas,
            judge_llm=judge_llm,
            cache_path=args.cache,
            use_cache=use_cache,
        )

    # --- score log turns ---
    log_results: List[dict] = []
    if args.mode in ("log", "both"):
        log_results = _score_log(
            log_path=args.log,
            use_relevancy=use_relevancy,
            use_full_ragas=use_full_ragas,
            judge_llm=judge_llm,
            score_last=args.score_last,
            cache_path=args.cache,
            use_cache=use_cache,
            gt_index=gt_index,
            budget_ranges=budget_ranges,
        )

    all_results = live_results + log_results

    if not all_results:
        print("\nNothing to score. Run with --mode live or check your log path.")
        return

    # --- case memory ---
    print("\nReading case memory from Qdrant ...")
    cases      = _read_case_memory()
    case_stats = _case_memory_stats(cases)
    print(f"  {case_stats.get('total',0)} cases  "
          f"(scored={case_stats.get('scored',0)}  avg={case_stats.get('avg_correctness')})")

    # --- aggregate ---
    print("\nAggregating all metrics ...")
    summary = _aggregate(all_results)
    summary["case_memory_stats"] = case_stats

    # --- write outputs ---
    report = {
        "summary":     summary,
        "case_memory": case_stats,
        "results":     all_results,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON report  -> {args.out_json}")

    _write_txt_report(summary, case_stats, all_results, args.out_txt)

    # --- condensed console print ---
    print("\n" + "="*72)
    print("  UNIFIED EVAL SUMMARY")
    print("="*72)

    def _p(label, val, unit=""):
        print(f"  {label:<44} {val if val is not None else 'n/a'}{unit}")

    _p("Total turns",           summary["total_turns"])
    _p("GT-scored turns",       summary["gt_scored"])
    print()
    _p("Avg correctness",       summary["avg_correctness"])
    _p("Accuracy",              summary["classification_metrics"].get("accuracy"))
    _p("Precision",             summary["classification_metrics"].get("precision"))
    _p("Recall",                summary["classification_metrics"].get("recall"))
    _p("F1",                    summary["classification_metrics"].get("f1"))
    print()
    _p(f"Recall@{RETRIEVAL_K}",       summary.get(f"avg_recall_at_{RETRIEVAL_K}"))
    _p(f"Precision@{RETRIEVAL_K}",    summary.get(f"avg_precision_at_{RETRIEVAL_K}"))
    _p("MRR",                         summary.get("avg_mrr"))
    _p(f"nDCG@{RETRIEVAL_K}",         summary.get(f"avg_ndcg_at_{RETRIEVAL_K}"))
    print()
    _p("Judge overall (1-10)",  summary.get("avg_judge_overall"))
    _p("Judge helpfulness",     summary.get("avg_judge_helpfulness"))
    _p("Judge factual accuracy",summary.get("avg_judge_factual_accuracy"))
    _p("Answer relevancy",      summary.get("avg_answer_relevancy"))
    print()
    _p("Case memory total",     case_stats.get("total"))
    _p("Case memory avg corr",  case_stats.get("avg_correctness"))
    _p(f"High-quality (>={CORRECT_THRESHOLD})", case_stats.get("high_quality_cases"))

    # recency
    rw = summary.get("recency_windows", {})
    print()
    print("  RECENCY  (last-1 / last-5-avg / last-10-avg)")
    for key in ["correctness", "judge_judge_avg", "faithfulness", f"ndcg_at_{RETRIEVAL_K}"]:
        w = rw.get(key, {})
        print(f"  {key:<38}  {w.get('last_1','n/a')}  /  {w.get('last_5_avg','n/a')}  /  {w.get('last_10_avg','n/a')}")

    print("="*72)


if __name__ == "__main__":
    main()