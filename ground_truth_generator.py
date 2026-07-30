"""
ground_truth_generator.py - Hypothetical ground-truth QA generation (Llama 3)

Standalone module, deliberately kept OUT of agent_functions.py so importing/
running the live app never triggers generation. Generation only happens when
you explicitly call one of the generate_* functions or run this file directly.

STRATEGIES
----------
"grid" (default) — the full combinatorial requirement space: every combination
    of low/medium/high across the 5 criteria (GPU intensity, Display quality,
    Portability, Multitasking, Processing speed) x 3 budget tiers
    (low/medium/high, split by price terciles across the catalog).
    3^5 x 3 = 729 combos. For each combo, the closest-matching real laptop in
    the catalog is found and used to ground a question + ground-truth answer
    (honestly noting any mismatch if no perfect match exists).

    python ground_truth_generator.py --strategy grid --n 1
    python ground_truth_generator.py --strategy grid --n 1 --limit-combos 30   # quick test slice

    PERFORMANCE: combos are sent GRID_BATCH_SIZE at a time (default 6, env
    var GRID_BATCH_SIZE) in a single LLM call instead of one call per combo.
    Together this turns ~729 sequential calls into ~120 batched calls.
    Any combo a batch doesn't cover is retried solo automatically, so
    coverage/resume behavior is unchanged — only the wall-clock time is
    different.

"representative" — top-N cheapest / mean-range / median-range / most-expensive
    laptops (price-stratified sample of the catalog itself, not the
    requirement space).

    python ground_truth_generator.py --strategy representative --n 3

"all" / "first_n" — every laptop / first N laptops in file order.

Results are stored in a SQLite database (ground_truth.db by default) instead
of a JSON file, so re-running the app / re-importing this module never
regenerates anything — the DB is the source of truth, and generation is
resumable (already-processed combos/laptops are skipped unless --no-resume).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import statistics
from contextlib import contextmanager
from itertools import product
from typing import Dict, List, Optional, Tuple

import requests
from qdrant_client import QdrantClient

# Reuse the existing keyword-tier classifier + feature cache path from the
# main app instead of duplicating the rules.
from agent_functions import _classify_one, CTX_WINDOW, _FEATURE_CACHE_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Override with `set GROUND_TRUTH_MODEL=llama3:latest` (Windows) or
# `export GROUND_TRUTH_MODEL=llama3:latest` (mac/Linux) if your local Ollama
# tag differs — check with `ollama list`.
GROUND_TRUTH_MODEL = os.environ.get("GROUND_TRUTH_MODEL", "llama3:latest")

# Ceiling only — doesn't cost VRAM/time unless actually generated up to
# (Ollama stops at EOS well before this). Sized for a full GRID_BATCH_SIZE
# batch's worth of QA pairs, not just one.
MAX_GROUND_TRUTH = int(os.environ.get("MAX_GROUND_TRUTH", 2200))

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "laptops")

GROUND_TRUTH_DB_PATH = "ground_truth.db"
DEFAULT_TOP_N_PER_TIER = 10                 # cheapest / mean / median / most-expensive, each
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
GROUND_TRUTH_KEEP_ALIVE = os.environ.get("GROUND_TRUTH_KEEP_ALIVE", "30m")  # don't unload mid-run

# Combos folded into a single LLM call for the "grid" strategy. Cuts 729
# sequential calls down to ~ceil(pending/effective_batch_size). Auto-shrinks
# when --n > 1 so total QA items/call (and completion tokens) stay roughly
# constant — see _effective_batch_size().
GRID_BATCH_SIZE = int(os.environ.get("GRID_BATCH_SIZE", 6))

# Kept equal to the live app's CTX_WINDOW by default (VRAM-safe on 6GB).
# Raise only if you have headroom to spare — num_ctx (not num_predict) is
# what drives KV-cache VRAM, per the note in agent_functions.py.
GROUND_TRUTH_CTX = int(os.environ.get("GROUND_TRUTH_CTX", CTX_WINDOW))

# The 5 criteria dimensions, matching _classify_one's output keys exactly.
DIMENSIONS = ["GPU intensity", "Display quality", "Portability", "Multitasking", "Processing speed"]
TIERS = ["low", "medium", "high"]
BUDGET_TIERS = ["low", "medium", "high"]
MEDIUM_BUDGET_CAP = 90_000   # medium/high budget split, fixed rather than derived from the mean
# Recalibrated for the reduced 1000-laptop catalog (2026 sample):
#   min=9,990  median=52,994  mean=68,935  Q3=81,990  max=500,000
# Old value (100,000) was tuned for the original ~11k catalog and now sits
# above Q3, so "high budget" only captured the top ~10-15% instead of a
# meaningful third. 90,000 keeps it a round, human-meaningful number while
# tracking the new distribution more sensibly.

# =============================================================================
# OLLAMA DIRECT API CALLS
# =============================================================================

def _ollama_generate(prompt: str, model: str = GROUND_TRUTH_MODEL) -> str:
    """Direct call to Ollama's generate endpoint with streaming disabled."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": MAX_GROUND_TRUTH,
            "num_ctx": GROUND_TRUTH_CTX,
        },
        "keep_alive": GROUND_TRUTH_KEEP_ALIVE,
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except requests.exceptions.Timeout:
        logger.error(f"   ⏱️  Ollama timeout after 120s for model {model}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"   ❌ Ollama request failed: {e}")
        raise


def list_ollama_models() -> List[str]:
    """Query Ollama for locally-pulled model tags. [] if Ollama is unreachable."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        logger.warning(f"   Could not reach Ollama at {OLLAMA_BASE_URL} to list models: {e}")
        return []


def _ensure_model_available(model: str) -> None:
    """
    Fail fast with a clear message instead of silently 404-ing on every
    single laptop/combo. If Ollama can't be reached at all, we let the real
    call surface its own error rather than blocking here.
    """
    models = list_ollama_models()
    if not models:
        return
    if model not in models:
        raise RuntimeError(
            f"Ollama model '{model}' is not pulled locally.\n"
            f"  Available models: {', '.join(models) if models else '(none)'}\n"
            f"  Fix: ollama pull {model.split(':')[0]}\n"
            f"  Or set the exact tag: set GROUND_TRUTH_MODEL={models[0] if models else '<tag>'}  (Windows)"
        )


# =============================================================================
# FEATURE CLASSIFICATION HELPERS (shared by both strategies)
# =============================================================================

# =============================================================================
# QDRANT DATA LOADER
# =============================================================================

def _load_laptops_from_qdrant(
    host: str = QDRANT_HOST, port: int = QDRANT_PORT, collection_name: str = QDRANT_COLLECTION
) -> List[dict]:
    """Fetch all laptops from Qdrant using an offset scroll loop."""
    logger.info(f"   📡 Connecting to Qdrant at {host}:{port} (collection={collection_name})...")
    try:
        client = QdrantClient(host=host, port=port)
        laptops = []
        next_offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
            for record in records:
                payload = record.payload or {}
                laptops.append({
                    "name": payload.get("name", "Unknown"),
                    "price": payload.get("price", 0),
                    "description": payload.get("description", ""),
                })
            if not next_offset:
                break
        return laptops
    except Exception as e:
        logger.error(f"❌ Failed to fetch laptop catalog from Qdrant ({host}:{port}/{collection_name}): {e}")
        return []


def _load_feature_cache() -> Dict[str, dict]:
    if os.path.exists(_FEATURE_CACHE_PATH):
        try:
            with open(_FEATURE_CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _classify_all_laptops(
    laptops: List[dict], feature_cache: Dict[str, dict]
) -> List[Tuple[dict, dict]]:
    """Return [(laptop, features)] for every laptop, using the cache where possible."""
    out = []
    for laptop in laptops:
        desc = laptop.get("description", "")
        features = feature_cache.get(desc) or _classify_one(desc)
        out.append((laptop, features))
    return out


# =============================================================================
# PRICE-TIER SAMPLING ("representative" strategy)
# =============================================================================

def select_representative_laptops(
    laptops: List[dict], top_n: int = DEFAULT_TOP_N_PER_TIER
) -> List[Tuple[dict, str]]:
    """
    Pick a price-stratified sample instead of the whole catalog:
      - top_n cheapest
      - top_n closest to the mean price   ("mean_range")
      - top_n closest to the median price ("median_range")
      - top_n most expensive

    Mean and median are kept as separate tiers (not merged) because laptop
    price distributions are usually right-skewed by a handful of premium
    models — the mean gets pulled up above the median, so each picks out a
    genuinely different "typical" laptop.
    """
    valid = [l for l in laptops if isinstance(l.get("price"), (int, float)) and l.get("price", 0) > 0]

    if not valid:
        logger.warning("   No laptops with a valid price found — falling back to first N laptops")
        return [(l, "unknown") for l in laptops[:top_n]]

    by_price = sorted(valid, key=lambda l: l["price"])
    cheapest = by_price[:top_n]
    most_expensive = by_price[-top_n:]

    mean_price = statistics.mean(l["price"] for l in valid)
    median_price = statistics.median(l["price"] for l in valid)

    mean_range = sorted(valid, key=lambda l: abs(l["price"] - mean_price))[:top_n]
    median_range = sorted(valid, key=lambda l: abs(l["price"] - median_price))[:top_n]

    logger.info(
        f"   💰 Price stats — min={by_price[0]['price']}, max={by_price[-1]['price']}, "
        f"mean={mean_price:.0f}, median={median_price:.0f}"
    )

    seen = set()
    selected: List[Tuple[dict, str]] = []
    for tier, group in [
        ("expensive", most_expensive),
        ("mean_range", mean_range),
        ("median_range", median_range),
        ("cheapest", cheapest),
    ]:
        for laptop in group:
            key = laptop.get("name") or laptop.get("description", "")[:50]
            if key in seen:
                continue
            seen.add(key)
            selected.append((laptop, tier))

    logger.info(
        f"   📊 Selected {len(selected)} laptops "
        f"({len(cheapest)} cheapest + {len(mean_range)} mean-range + "
        f"{len(median_range)} median-range + {len(most_expensive)} expensive, deduplicated)"
    )
    return selected


# =============================================================================
# FULL REQUIREMENT GRID ("grid" strategy)
# =============================================================================

def compute_budget_ranges(
    laptops: List[dict], medium_cap: float = MEDIUM_BUDGET_CAP
) -> Dict[str, Tuple[float, float]]:
    """
    Split budget into low / medium / high using the dataset's own median for
    the low/medium split, and a fixed cap for the medium/high split:

      low    = min_price .. median
      medium = median    .. medium_cap   (default ₹100,000)
      high   = medium_cap .. max_price

    The low/medium boundary still reflects the dataset (median price), but
    medium/high is pinned at a fixed, human-meaningful cutoff instead of the
    mean — on a right-skewed catalog the mean can land surprisingly close to
    the median and doesn't match what "high budget" intuitively means for a
    laptop shopper.
    """
    prices = [
        l["price"] for l in laptops
        if isinstance(l.get("price"), (int, float)) and l.get("price", 0) > 0
    ]
    if not prices:
        return {"low": (0, 0), "medium": (0, 0), "high": (0, 0)}

    min_p, max_p = min(prices), max(prices)
    median_p = statistics.median(prices)
    mean_p = statistics.mean(prices)

    # Guard against a pathological catalog where the median itself exceeds the cap.
    low_medium_split = min(median_p, medium_cap)

    ranges = {
        "low": (min_p, low_medium_split),
        "medium": (low_medium_split, medium_cap),
        "high": (medium_cap, max_p),
    }
    logger.info(
        f"   💰 Dataset price stats — min=₹{min_p:.0f}, max=₹{max_p:.0f}, "
        f"mean=₹{mean_p:.0f}, median=₹{median_p:.0f}"
    )
    logger.info(
        f"   💰 Budget tiers (median low/mid split, fixed ₹{medium_cap:.0f} mid/high cap) — "
        f"low: ₹{ranges['low'][0]:.0f}-{ranges['low'][1]:.0f}, "
        f"medium: ₹{ranges['medium'][0]:.0f}-{ranges['medium'][1]:.0f}, "
        f"high: ₹{ranges['high'][0]:.0f}-{ranges['high'][1]:.0f}"
    )
    return ranges


def log_dimension_distribution(laptops_with_features: List[Tuple[dict, dict]]) -> Dict[str, Dict[str, int]]:
    """
    GPU intensity / Display quality / Portability / Multitasking / Processing
    speed are categorical (keyword-classified low/medium/high), so there's no
    mean/median to compute — instead, show the REAL count of laptops in each
    tier per dimension across the dataset, so it's visible upfront which
    combos in the grid will have plenty of real matches vs. almost none
    (e.g. "high GPU intensity" + "low budget" may barely exist together).
    """
    counts: Dict[str, Dict[str, int]] = {d: {"low": 0, "medium": 0, "high": 0} for d in DIMENSIONS}
    for _, features in laptops_with_features:
        for d in DIMENSIONS:
            tier = features.get(d, "medium")
            if tier not in counts[d]:
                counts[d][tier] = 0
            counts[d][tier] += 1

    logger.info("   📐 Dataset distribution per criterion:")
    for d in DIMENSIONS:
        c = counts[d]
        logger.info(f"      {d}: low={c.get('low', 0)}, medium={c.get('medium', 0)}, high={c.get('high', 0)}")
    return counts


def build_requirement_grid() -> List[dict]:
    """Every combination of low/medium/high across the 5 dimensions + budget tier. 3^5 x 3 = 729."""
    grid = []
    for gpu, disp, port, multi, speed, budget in product(TIERS, TIERS, TIERS, TIERS, TIERS, BUDGET_TIERS):
        grid.append({
            "GPU intensity": gpu,
            "Display quality": disp,
            "Portability": port,
            "Multitasking": multi,
            "Processing speed": speed,
            "Budget tier": budget,
        })
    return grid


def _combo_key(combo: dict) -> str:
    """Stable unique string for a requirement combo, used for resume/dedup."""
    return "|".join(f"{d}:{combo[d]}" for d in DIMENSIONS) + f"|Budget:{combo['Budget tier']}"


def _match_score(features: dict, combo: dict) -> int:
    """How many of the 5 dimensions this laptop's real tiers match the target combo."""
    return sum(1 for d in DIMENSIONS if features.get(d) == combo[d])


def find_best_match(
    laptops_with_features: List[Tuple[dict, dict]],
    combo: dict,
    budget_range: Tuple[float, float],
) -> Tuple[dict, dict, int, bool]:
    """
    Find the closest real laptop to a requirement combo: prefer laptops within
    the target budget range, then maximize how many of the 5 dimensions match.
    Returns (laptop, features, match_score 0-5, was_within_budget).

    NOTE: kept around for the "representative"/"all"/"first_n" call sites and
    for reference — the grid strategy itself now goes through
    _bucket_by_feature_vector() + find_best_match_bucketed() below, which is
    the same logic but ~4-8x fewer comparisons across all 729 combos (see
    that function's docstring for why).
    """
    lo, hi = budget_range
    in_budget = [lf for lf in laptops_with_features if lo <= lf[0].get("price", 0) <= hi]
    pool = in_budget if in_budget else laptops_with_features

    best = max(pool, key=lambda lf: _match_score(lf[1], combo))
    score = _match_score(best[1], combo)
    within_budget = best in in_budget
    return best[0], best[1], score, within_budget


def _feature_vector(features: dict) -> tuple:
    """Features as an ordered tuple (matches DIMENSIONS order) so vectors are hashable/comparable."""
    return tuple(features.get(d, "medium") for d in DIMENSIONS)


def _combo_vector(combo: dict) -> tuple:
    return tuple(combo[d] for d in DIMENSIONS)


def _bucket_by_feature_vector(
    pool: List[Tuple[dict, dict]]
) -> Dict[tuple, List[Tuple[dict, dict]]]:
    """
    Group laptops by their exact (low/medium/high)^5 feature vector. There
    are only 3^5 = 243 *possible* vectors, and real catalogs collapse onto far
    fewer than that in practice (many laptops share the same tier profile) —
    so this is almost always a big reduction from "however many laptops are
    in the pool" down to "however many distinct vectors actually occur".
    """
    buckets: Dict[tuple, List[Tuple[dict, dict]]] = {}
    for lf in pool:
        buckets.setdefault(_feature_vector(lf[1]), []).append(lf)
    return buckets


def find_best_match_bucketed(
    buckets: Dict[tuple, List[Tuple[dict, dict]]],
    fallback_buckets: Dict[tuple, List[Tuple[dict, dict]]],
    combo: dict,
) -> Tuple[dict, dict, int, bool]:
    """
    Same contract/output as find_best_match, but takes pre-bucketed pools
    (see _bucket_by_feature_vector) instead of re-scanning every laptop.

    Why this is faster: find_best_match does one full O(pool_size) filter +
    O(pool_size) max-scan PER combo, so 729 combos over a 1000-laptop catalog
    is ~729,000 dict-comparison operations. Bucketing by feature vector once
    (O(pool_size), done a handful of times total — once per budget tier, not
    once per combo) turns each combo's lookup into a scan over only the
    distinct vectors present (<=243, usually far fewer) instead of every
    laptop. The catalog is only ever iterated when building the buckets.
    """
    combo_vec = _combo_vector(combo)
    within_budget = bool(buckets)
    pool = buckets if buckets else fallback_buckets

    best_vec = max(
        pool.keys(),
        key=lambda v: sum(1 for a, b in zip(v, combo_vec) if a == b),
    )
    score = sum(1 for a, b in zip(best_vec, combo_vec) if a == b)
    laptop, features = pool[best_vec][0]
    return laptop, features, score, within_budget


def _build_grid_prompt(
    combo: dict, laptop: dict, features: dict, match_score: int, within_budget: bool, n: int
) -> str:
    """Prompt Llama 3 for n Q/A pairs where this laptop is the best available answer to `combo`."""
    name = laptop.get("name", "Unknown laptop")
    price = laptop.get("price", 0)
    desc = laptop.get("description", "")[:800]

    req_string = (
        f"I need a laptop with {combo['GPU intensity']} GPU intensity, "
        f"{combo['Display quality']} display quality, "
        f"{combo['Portability']} portability, "
        f"{combo['Multitasking']} multitasking, "
        f"{combo['Processing speed']} processing speed, "
        f"and a {combo['Budget tier']} budget."
    )

    mismatch_note = ""
    if match_score < len(DIMENSIONS) or not within_budget:
        mismatched = [d for d in DIMENSIONS if features.get(d) != combo[d]]
        mismatch_note = (
            f"\nIMPORTANT: this laptop does NOT perfectly satisfy every criterion. "
            f"Mismatched dimensions: {mismatched or 'none'}. "
            f"It is {'within' if within_budget else 'outside'} the target budget tier. "
            f"Every ground_truth_answer must honestly acknowledge this is the closest "
            f"available match rather than claiming a perfect fit — do not pretend it "
            f"satisfies criteria it doesn't."
        )

    return (
        f"Target shopper requirement:\n\"{req_string}\"\n\n"
        f"Best available laptop match:\n"
        f"Laptop: {name}\n"
        f"Price: ₹{price}\n"
        f"Description: {desc}\n"
        f"Its actual feature tiers: {json.dumps(features)}\n"
        f"{mismatch_note}\n\n"
        f"Generate exactly {n} distinct hypothetical (question, ground_truth_answer) pairs "
        f"that a shopper with the above requirement might ask, for which this laptop is the "
        f"best/ideal available answer. Vary phrasing and angle (conversational, spec-driven, "
        f"budget-driven). Every ground_truth_answer must be fully supported by the laptop info "
        f"above (no invented specs).\n\n"
        f"Respond ONLY with a JSON array of exactly {n} objects, each shaped as:\n"
        f'{{"question": "...", "ground_truth_answer": "..."}}'
    )


def _effective_batch_size(n_per_combo: int, batch_size: int = GRID_BATCH_SIZE) -> int:
    """
    Shrink the batch as n_per_combo grows so total QA items/call (and thus
    completion tokens) stays roughly constant and inside GROUND_TRUTH_CTX.
    e.g. batch_size=6: n=1 -> 6 combos/call, n=2 -> 3 combos/call, n=3 -> 2.
    """
    return max(1, batch_size // max(1, n_per_combo))


def _build_batch_grid_prompt(slots: List[dict], n: int) -> str:
    """
    Same contract as _build_grid_prompt but for several combos in one call —
    the main lever for cutting ~729 sequential LLM round-trips down to
    ~729/effective_batch_size. Each slot is a dict with keys: index, combo,
    laptop, features, score, within_budget. The model is told to tag every
    item it returns with 'index' so responses route back to the right combo.
    Laptop descriptions are trimmed tighter than the single-combo prompt
    (500 vs 800 chars) since several are packed into one call.
    """
    blocks = []
    for slot in slots:
        combo, laptop, features = slot["combo"], slot["laptop"], slot["features"]
        score, within_budget = slot["score"], slot["within_budget"]
        name = laptop.get("name", "Unknown laptop")
        price = laptop.get("price", 0)
        desc = laptop.get("description", "")[:500]

        req_string = (
            f"{combo['GPU intensity']} GPU intensity, {combo['Display quality']} display quality, "
            f"{combo['Portability']} portability, {combo['Multitasking']} multitasking, "
            f"{combo['Processing speed']} processing speed, {combo['Budget tier']} budget"
        )

        mismatch_note = ""
        if score < len(DIMENSIONS) or not within_budget:
            mismatched = [d for d in DIMENSIONS if features.get(d) != combo[d]]
            mismatch_note = (
                f" NOT a perfect match — mismatched on {mismatched or 'none'}, "
                f"{'within' if within_budget else 'outside'} target budget. Every answer for "
                f"this item must honestly acknowledge that rather than claiming a perfect fit."
            )

        blocks.append(
            f"[{slot['index']}] Requirement: I need a laptop with {req_string}.\n"
            f"    Laptop: {name} | Price: \u20b9{price} | Tiers: {json.dumps(features)}\n"
            f"    Description: {desc}{mismatch_note}"
        )

    joined = "\n\n".join(blocks)
    total_items = len(slots) * n
    return (
        f"Below are {len(slots)} independent shopper requirement/laptop pairs, each tagged "
        f"with an [index]. For EACH one, generate exactly {n} distinct hypothetical "
        f"(question, ground_truth_answer) pairs a shopper with that requirement might ask, "
        f"for which that item's laptop is the best/ideal available answer. Vary phrasing and "
        f"angle across items (conversational, spec-driven, budget-driven). Every "
        f"ground_truth_answer must be fully supported by that item's laptop info (no invented "
        f"specs), and must honestly flag any noted mismatch.\n\n"
        f"{joined}\n\n"
        f"Respond ONLY with a single flat JSON array covering ALL items above (do not nest or "
        f"group by index), each element shaped as:\n"
        f'{{"index": <int, the item\'s index above>, "question": "...", "ground_truth_answer": "..."}}\n'
        f"Total elements expected: {total_items}."
    )


# =============================================================================
# SHARED PROMPT PARSING
# =============================================================================

def _parse_ground_truth_response(raw: str) -> List[dict]:
    """Strip any prose and pull the JSON array out of a Llama 3 response."""
    # Clean up any extra text before/after the JSON
    cleaned = raw.strip()
    # Try to find JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
    except Exception as e:
        logger.warning(f"   Ground truth JSON parse failed: {e}")
        return []
    return items if isinstance(items, list) else []


def _parse_batch_ground_truth_response(raw: str, valid_indices: set) -> Dict[int, List[dict]]:
    """Like _parse_ground_truth_response, but buckets items by their 'index' field
    so each combo in the batch gets only the items that belong to it."""
    cleaned = raw.strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    out: Dict[int, List[dict]] = {}
    if not match:
        return out
    try:
        items = json.loads(match.group())
    except Exception as e:
        logger.warning(f"   Batch ground truth JSON parse failed: {e}")
        return out
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if isinstance(idx, str) and idx.strip().lstrip("-").isdigit():
            idx = int(idx)
        if idx not in valid_indices:
            continue
        if not (item.get("question") and item.get("ground_truth_answer")):
            continue
        out.setdefault(idx, []).append({
            "question": item["question"],
            "ground_truth_answer": item["ground_truth_answer"],
        })
    return out


def _build_ground_truth_prompt(laptop: dict, features: dict, n: int) -> str:
    """Per-laptop prompt (used by representative/all/first_n): questions matching THIS laptop's own tiers."""
    name = laptop.get("name", "Unknown laptop")
    price = laptop.get("price", 0)
    desc = laptop.get("description", "")[:800]

    req_string = (
        f"I need a laptop with {features.get('GPU intensity', 'medium')} GPU intensity, "
        f"{features.get('Display quality', 'medium')} display quality, "
        f"{features.get('Portability', 'medium')} portability, "
        f"{features.get('Multitasking', 'medium')} multitasking, "
        f"{features.get('Processing speed', 'medium')} processing speed "
        f"and a budget of {price}."
    )

    return (
        f"Laptop: {name}\n"
        f"Price: ₹{price}\n"
        f"Description: {desc}\n"
        f"Feature tiers: {json.dumps(features)}\n\n"
        f"This laptop is the ideal match for a shopper whose requirements map to:\n"
        f"\"{req_string}\"\n\n"
        f"Generate exactly {n} distinct hypothetical (question, ground_truth_answer) pairs "
        f"a real shopper might ask that THIS laptop is the correct/ideal answer to. "
        f"Vary phrasing and angle (some conversational, some spec-driven, some budget-driven) "
        f"but every question must be answerable using only the laptop info above, and every "
        f"ground_truth_answer must be fully supported by that info (no invented specs).\n\n"
        f"Respond ONLY with a JSON array of exactly {n} objects, each shaped as:\n"
        f'{{"question": "...", "ground_truth_answer": "..."}}'
    )


# =============================================================================
# DATABASE
# =============================================================================

@contextmanager
def _connect(db_path: str = GROUND_TRUTH_DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = GROUND_TRUTH_DB_PATH) -> None:
    """Create the ground_truth_qa table if it doesn't exist yet, and migrate older DBs."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ground_truth_qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combo_key TEXT,
                laptop_name TEXT NOT NULL,
                laptop_price INTEGER,
                price_tier TEXT,
                match_score INTEGER,
                question TEXT NOT NULL,
                ground_truth_answer TEXT NOT NULL,
                requirements_json TEXT,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Migrate DBs created before combo_key/match_score existed.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(ground_truth_qa)")}
        for col, coltype in [("combo_key", "TEXT"), ("match_score", "INTEGER")]:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE ground_truth_qa ADD COLUMN {col} {coltype}")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_gt_laptop_name ON ground_truth_qa(laptop_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gt_combo_key ON ground_truth_qa(combo_key)")
        conn.commit()


def _laptop_already_generated(conn: sqlite3.Connection, laptop_name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM ground_truth_qa WHERE laptop_name = ? AND combo_key IS NULL LIMIT 1",
        (laptop_name,),
    )
    return cur.fetchone() is not None


def _delete_laptop_rows(conn: sqlite3.Connection, laptop_name: str) -> None:
    conn.execute("DELETE FROM ground_truth_qa WHERE laptop_name = ? AND combo_key IS NULL", (laptop_name,))


def _combo_already_generated(conn: sqlite3.Connection, combo_key: str) -> bool:
    cur = conn.execute("SELECT 1 FROM ground_truth_qa WHERE combo_key = ? LIMIT 1", (combo_key,))
    return cur.fetchone() is not None


def _delete_combo_rows(conn: sqlite3.Connection, combo_key: str) -> None:
    conn.execute("DELETE FROM ground_truth_qa WHERE combo_key = ?", (combo_key,))


def _insert_qa_items(
    conn: sqlite3.Connection,
    laptop_name: str,
    laptop_price: int,
    price_tier: str,
    context: str,
    items: List[dict],
    combo_key: Optional[str] = None,
    requirements: Optional[dict] = None,
    match_score: Optional[int] = None,
) -> None:
    rows = []
    for item in items:
        if not (item.get("question") and item.get("ground_truth_answer")):
            continue
        
        reqs = requirements if requirements is not None else item.get("requirements", {})
        reqs_json = json.dumps(reqs) if isinstance(reqs, dict) else reqs
        
        rows.append((
            combo_key,
            laptop_name,
            laptop_price,
            price_tier,
            match_score,
            item.get("question", ""),
            item.get("ground_truth_answer", ""),
            reqs_json,
            context,
        ))
    conn.executemany(
        """
        INSERT INTO ground_truth_qa
            (combo_key, laptop_name, laptop_price, price_tier, match_score,
             question, ground_truth_answer, requirements_json, context)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def load_ground_truth_dataset(db_path: str = GROUND_TRUTH_DB_PATH) -> List[dict]:
    """Read every stored QA row back out as a flat list of dicts."""
    if not os.path.exists(db_path):
        return []
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT combo_key, laptop_name, laptop_price, price_tier, match_score, "
            "question, ground_truth_answer, requirements_json, context, created_at "
            "FROM ground_truth_qa ORDER BY id"
        )
        rows = cur.fetchall()

    results = []
    for r in rows:
        try:
            requirements = json.loads(r["requirements_json"]) if r["requirements_json"] else {}
        except Exception:
            requirements = {}
        results.append({
            "combo_key": r["combo_key"],
            "laptop_name": r["laptop_name"],
            "laptop_price": r["laptop_price"],
            "price_tier": r["price_tier"],
            "match_score": r["match_score"],
            "question": r["question"],
            "ground_truth_answer": r["ground_truth_answer"],
            "requirements": requirements,
            "context": r["context"],
            "created_at": r["created_at"],
        })
    return results


def dataset_stats(db_path: str = GROUND_TRUTH_DB_PATH) -> dict:
    if not os.path.exists(db_path):
        return {"laptops_covered": 0, "qa_count": 0, "combos_covered": 0, "by_tier": {}}
    with _connect(db_path) as conn:
        qa_count = conn.execute("SELECT COUNT(*) FROM ground_truth_qa").fetchone()[0]
        laptops_covered = conn.execute(
            "SELECT COUNT(DISTINCT laptop_name) FROM ground_truth_qa"
        ).fetchone()[0]
        combos_covered = conn.execute(
            "SELECT COUNT(DISTINCT combo_key) FROM ground_truth_qa WHERE combo_key IS NOT NULL"
        ).fetchone()[0]
        by_tier = dict(conn.execute(
            "SELECT price_tier, COUNT(*) FROM ground_truth_qa GROUP BY price_tier"
        ).fetchall())
    return {
        "laptops_covered": laptops_covered,
        "qa_count": qa_count,
        "combos_covered": combos_covered,
        "by_tier": by_tier,
    }


# =============================================================================
# GENERATION ENTRY POINTS
# =============================================================================

def generate_ground_truth_grid(
    qdrant_host: str = QDRANT_HOST,
    qdrant_port: int = QDRANT_PORT,
    collection_name: str = QDRANT_COLLECTION,
    db_path: str = GROUND_TRUTH_DB_PATH,
    n_per_combo: int = 1,
    limit_combos: Optional[int] = None,
    resume: bool = True,
    batch_size: Optional[int] = None,
) -> List[dict]:
    """
    Full combinatorial requirement grid: every low/medium/high combination
    across GPU intensity / Display quality / Portability / Multitasking /
    Processing speed, crossed with low/medium/high budget tiers (price
    terciles across the catalog). 3^5 x 3 = 729 combos by default.

    For each combo, finds the closest-matching real laptop (within the
    target budget tier if possible, maximizing dimension matches otherwise)
    and asks Llama 3 to write n_per_combo (question, ground_truth_answer)
    pairs grounded in that laptop's real specs — honestly noting any
    mismatch rather than claiming a perfect fit.

    Combos are processed batch_size at a time (default GRID_BATCH_SIZE,
    auto-shrunk for n_per_combo > 1) — one LLM call covers several combos
    instead of one call per combo, which is what actually makes 729 combos
    tractable on a single 6GB-VRAM GPU. Any combo whose batch doesn't come
    back with a parseable item for it is retried solo (same prompt as the
    old one-call-per-combo path) so nothing silently gets skipped.

    limit_combos truncates the grid (useful for a quick test run before
    committing to all 729). resume=True (default) skips combos that already
    have rows in the DB.
    """
    init_db(db_path)
    _ensure_model_available(GROUND_TRUTH_MODEL)

    laptops = _load_laptops_from_qdrant(qdrant_host, qdrant_port, collection_name)
    if not laptops:
        return load_ground_truth_dataset(db_path)

    logger.info(f"   💻 Loaded catalog: {len(laptops)} laptops from Qdrant "
                f"({qdrant_host}:{qdrant_port}/{collection_name})")

    feature_cache = _load_feature_cache()
    laptops_with_features = _classify_all_laptops(laptops, feature_cache)
    budget_ranges = compute_budget_ranges(laptops)
    log_dimension_distribution(laptops_with_features)

    # Bucket once per budget tier (not once per combo) — this is what find_best_match_bucketed
    # searches against below, instead of rescanning the full 1000-laptop catalog 729 times.
    full_buckets = _bucket_by_feature_vector(laptops_with_features)
    tier_buckets: Dict[str, Dict[tuple, List[Tuple[dict, dict]]]] = {}
    for tier in BUDGET_TIERS:
        lo, hi = budget_ranges[tier]
        in_budget_pool = [lf for lf in laptops_with_features if lo <= lf[0].get("price", 0) <= hi]
        tier_buckets[tier] = _bucket_by_feature_vector(in_budget_pool)
        logger.info(
            f"   🔍 Budget tier '{tier}': searching {len(in_budget_pool)}/{len(laptops)} laptops "
            f"in ₹{lo:.0f}-{hi:.0f} ({len(tier_buckets[tier])} distinct feature-tier combos)"
        )

    grid = build_requirement_grid()
    if limit_combos:
        grid = grid[:limit_combos]

    logger.info(f"   🧮 Requirement grid: {len(grid)} combos "
                f"({len(TIERS)}^{len(DIMENSIONS)} x {len(BUDGET_TIERS)} budget tiers)")

    eff_batch = _effective_batch_size(n_per_combo, batch_size or GRID_BATCH_SIZE)
    logger.info(f"   📦 {eff_batch} combos/call (ctx={GROUND_TRUTH_CTX}) — was 1 combo/call")

    with _connect(db_path) as conn:
        # Resolve resume/skip + best-match laptop up front (cheap, CPU-only)
        # so batches only ever contain combos that actually need generation.
        pending: List[dict] = []
        for combo in grid:
            key = _combo_key(combo)
            if resume and _combo_already_generated(conn, key):
                continue
            if not resume:
                _delete_combo_rows(conn, key)
            tier = combo["Budget tier"]
            laptop, features, score, within_budget = find_best_match_bucketed(
                tier_buckets[tier], full_buckets, combo
            )
            pending.append({
                "key": key, "combo": combo, "laptop": laptop, "features": features,
                "score": score, "within_budget": within_budget,
            })

        skipped = len(grid) - len(pending)
        if skipped:
            logger.info(f"   ⏭  skipping {skipped} already-generated combos")

        n_batches = max(1, -(-len(pending) // eff_batch))  # ceil div
        done = 0
        for b in range(0, len(pending), eff_batch):
            chunk = pending[b: b + eff_batch]
            for idx, slot in enumerate(chunk):
                slot["index"] = idx
            batch_num = b // eff_batch + 1

            try:
                raw = _ollama_generate(_build_batch_grid_prompt(chunk, n_per_combo))
                by_index = _parse_batch_ground_truth_response(raw, valid_indices=set(range(len(chunk))))
            except Exception as e:
                logger.warning(f"   [batch {batch_num}/{n_batches}] ❌ batch call failed: {e}")
                by_index = {}

            for slot in chunk:
                items = by_index.get(slot["index"], [])
                if not items:
                    # Batch didn't cover this one (parse failure, model skipped
                    # it, etc.) — fall back to the original single-combo call
                    # rather than losing it for the whole run.
                    logger.info(f"   [batch {batch_num}/{n_batches}] ↩  retrying {slot['key']} solo")
                    try:
                        raw_solo = _ollama_generate(_build_grid_prompt(
                            slot["combo"], slot["laptop"], slot["features"],
                            slot["score"], slot["within_budget"], n_per_combo,
                        ))
                        items = _parse_ground_truth_response(raw_solo)
                    except Exception as e:
                        logger.warning(f"   [batch {batch_num}/{n_batches}] ❌ solo retry failed for {slot['key']}: {e}")
                        items = []

                _insert_qa_items(
                    conn,
                    laptop_name=slot["laptop"].get("name", "unknown"),
                    laptop_price=slot["laptop"].get("price", 0),
                    price_tier=slot["combo"]["Budget tier"],
                    context=slot["laptop"].get("description", ""),
                    items=items,
                    combo_key=slot["key"],
                    requirements=slot["combo"],
                    match_score=slot["score"],
                )
                done += 1
                logger.info(
                    f"   [{done}/{len(pending)}] ✅ {slot['key']} -> {slot['laptop'].get('name')} "
                    f"(match {slot['score']}/{len(DIMENSIONS)}"
                    f"{', in budget' if slot['within_budget'] else ', OUT of budget'}): "
                    f"{len(items)} QA pairs"
                )

    results = load_ground_truth_dataset(db_path)
    logger.info(f"📝 Grid generation done: {len(results)} QA pairs total in {db_path}")
    return results


def generate_ground_truth_dataset(
    qdrant_host: str = QDRANT_HOST,
    qdrant_port: int = QDRANT_PORT,
    collection_name: str = QDRANT_COLLECTION,
    db_path: str = GROUND_TRUTH_DB_PATH,
    n_per_laptop: int = 2,
    strategy: str = "grid",   # "grid" | "representative" | "all" | "first_n"
    top_n_per_tier: int = DEFAULT_TOP_N_PER_TIER,
    limit: Optional[int] = None,          # used by "first_n"
    limit_combos: Optional[int] = None,   # used by "grid"
    resume: bool = True,
    batch_size: Optional[int] = None,     # used by "grid"; None = GRID_BATCH_SIZE
) -> List[dict]:
    """
    Dispatch to the requested generation strategy. See module docstring for
    what each strategy does. "grid" (full requirement-combo coverage) is the
    default and recommended approach; the others sample the laptop catalog
    directly instead of the requirement space.
    """
    if strategy == "grid":
        return generate_ground_truth_grid(
            qdrant_host=qdrant_host, qdrant_port=qdrant_port, collection_name=collection_name,
            db_path=db_path, n_per_combo=n_per_laptop,
            limit_combos=limit_combos, resume=resume, batch_size=batch_size,
        )

    init_db(db_path)
    _ensure_model_available(GROUND_TRUTH_MODEL)

    laptops = _load_laptops_from_qdrant(qdrant_host, qdrant_port, collection_name)
    if not laptops:
        return load_ground_truth_dataset(db_path)

    logger.info(f"   💻 Loaded catalog: {len(laptops)} laptops from Qdrant "
                f"({qdrant_host}:{qdrant_port}/{collection_name})")

    if strategy == "representative":
        selected = select_representative_laptops(laptops, top_n=top_n_per_tier)
    elif strategy == "first_n":
        selected = [(l, "n/a") for l in (laptops[:limit] if limit else laptops)]
    elif strategy == "all":
        selected = [(l, "n/a") for l in laptops]
    else:
        raise ValueError(f"Unknown strategy: {strategy!r} (expected grid/representative/all/first_n)")

    logger.info(f"   🔍 Searching/generating from {len(selected)}/{len(laptops)} laptops (strategy={strategy})")

    feature_cache = _load_feature_cache()

    with _connect(db_path) as conn:
        for i, (laptop, price_tier) in enumerate(selected, 1):
            name = laptop.get("name", f"laptop_{i}")

            if resume and _laptop_already_generated(conn, name):
                logger.info(f"   [{i}/{len(selected)}] ⏭  skip (already generated): {name} [{price_tier}]")
                continue
            if not resume:
                _delete_laptop_rows(conn, name)

            desc = laptop.get("description", "")
            features = feature_cache.get(desc) or _classify_one(desc)

            prompt = _build_ground_truth_prompt(laptop, features, n_per_laptop)

            try:
                raw = _ollama_generate(prompt)
                items = _parse_ground_truth_response(raw)
            except Exception as e:
                logger.warning(f"   [{i}/{len(selected)}] ❌ generation failed for {name}: {e}")
                items = []

            _insert_qa_items(
                conn, laptop_name=name, laptop_price=laptop.get("price", 0),
                price_tier=price_tier, context=desc, items=items, requirements=features,
            )
            logger.info(f"   [{i}/{len(selected)}] ✅ {name} [{price_tier}]: {len(items)} QA pairs")

    results = load_ground_truth_dataset(db_path)
    logger.info(f"📝 Ground truth generation done: {len(results)} QA pairs total in {db_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate hypothetical ground-truth QA pairs via Llama 3")
    parser.add_argument("--n", type=int, default=1, help="QA pairs per combo/laptop (default 1)")
    parser.add_argument(
        "--strategy", choices=["grid", "representative", "all", "first_n"], default="grid",
        help="grid (default) = full low/medium/high x 5-criteria x budget-tier combinatorial coverage; "
             "representative = cheapest/mean/median/expensive top-N laptop sample; "
             "all = whole catalog; first_n = first --limit laptops",
    )
    parser.add_argument(
        "--top-n-per-tier", type=int, default=DEFAULT_TOP_N_PER_TIER,
        help="Cheapest/mean/median/expensive laptops each (only for --strategy representative, default 10)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max laptops (only for --strategy first_n)")
    parser.add_argument(
        "--limit-combos", type=int, default=None,
        help="Truncate the 729-combo grid to the first N (only for --strategy grid; good for a quick test run)",
    )
    parser.add_argument("--qdrant-host", type=str, default=QDRANT_HOST, help="Qdrant service host endpoint")
    parser.add_argument("--qdrant-port", type=int, default=QDRANT_PORT, help="Qdrant service port endpoint")
    parser.add_argument("--collection", type=str, default=QDRANT_COLLECTION, help="Target Qdrant collection name")
    parser.add_argument("--db-path", type=str, default=GROUND_TRUTH_DB_PATH)
    parser.add_argument("--no-resume", action="store_true", help="Wipe & regenerate selected combos/laptops")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Combos per LLM call for --strategy grid (default {GRID_BATCH_SIZE}, "
             f"auto-shrinks when --n > 1). Lower this if you hit context-length issues.",
    )
    args = parser.parse_args()

    results = generate_ground_truth_dataset(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection,
        db_path=args.db_path,
        n_per_laptop=args.n,
        strategy=args.strategy,
        top_n_per_tier=args.top_n_per_tier,
        limit=args.limit,
        limit_combos=args.limit_combos,
        resume=not args.no_resume,
        batch_size=args.batch_size,
    )
    print(f"\nGenerated/loaded {len(results)} ground-truth QA pairs -> {args.db_path}")