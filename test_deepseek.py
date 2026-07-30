"""
ground_truth_generator.py - Hypothetical ground-truth QA generation (DeepSeek-R1)

Standalone module, deliberately kept OUT of agent_functions.py so importing/
running the live app never triggers generation. This only runs QA generation
when you explicitly call generate_ground_truth_dataset(...) or run this file
directly:

    python ground_truth_generator.py --n 3                     # default: representative sample
    python ground_truth_generator.py --n 3 --top-n-per-tier 15  # 15 cheap + 15 mid + 15 expensive
    python ground_truth_generator.py --strategy all             # every laptop (slow on big catalogs)
    python ground_truth_generator.py --strategy first_n --limit 20
    python ground_truth_generator.py --no-resume                # wipe & regenerate everything

SAMPLING STRATEGY (default "representative"):
Instead of processing an entire multi-thousand-laptop catalog, pick the
top-N cheapest, top-N closest-to-mean-price, top-N closest-to-median-price,
and top-N most expensive laptops (4 tiers). This gives a price-stratified
eval set in a fraction of the time/compute, while still covering the full
spectrum of what a shopper might ask about.

Results are stored in a SQLite database (ground_truth.db by default) instead
of a JSON file, so re-running the app / re-importing this module never
regenerates anything — the DB is the source of truth, and generation is
resumable (already-processed laptops are skipped unless --no-resume).

Each row is grounded in a laptop's real specs and the same 5-dimension
requirement schema used across the app (GPU intensity / Display quality /
Portability / Multitasking / Processing speed / Budget), which makes the
output directly usable later as a RAGAS / llm-as-judge eval set
(question, ground_truth_answer, context).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sqlite3
import statistics
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

from deepseek_wrapper import DeepSeekLLM, clean_deepseek_response

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

GROUND_TRUTH_MODEL = "deepseek-r1:latest"   # reasoning model — separate from the app's llama3.1
MAX_GROUND_TRUTH = 1500                     # generous ceiling; think tokens get stripped after
GROUND_TRUTH_DB_PATH = "ground_truth.db"
DEFAULT_TOP_N_PER_TIER = 10                 # cheapest / mid-range / most-expensive, each

_ground_truth_llm: Optional[DeepSeekLLM] = None


def _get_ground_truth_llm() -> DeepSeekLLM:
    """Lazily build the DeepSeek-R1 generator LLM."""
    global _ground_truth_llm
    if _ground_truth_llm is None:
        _ground_truth_llm = DeepSeekLLM(
            model=GROUND_TRUTH_MODEL,
            temperature=0.7,          # want variety across synthetic questions
            num_predict=MAX_GROUND_TRUTH,
            num_ctx=CTX_WINDOW,
            system_prompt=(
                "You are a synthetic dataset generator for evaluating a laptop "
                "shopping RAG assistant. You always respond with ONLY a valid "
                "JSON array — no prose, no markdown fences."
            ),
            think=True,  # deepseek-r1 emits <think>...</think>; stripped by clean_deepseek_response
        )
    return _ground_truth_llm


# =============================================================================
# PRICE-TIER SAMPLING
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

    Returns a de-duplicated list of (laptop, price_tier) tuples. Laptops with
    no usable price are dropped from sampling (falls back to first `top_n`
    of those if nothing has a valid price at all).
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
    """Create the ground_truth_qa table if it doesn't exist yet."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ground_truth_qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                laptop_name TEXT NOT NULL,
                laptop_price INTEGER,
                price_tier TEXT,
                question TEXT NOT NULL,
                ground_truth_answer TEXT NOT NULL,
                requirements_json TEXT,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_gt_laptop_name ON ground_truth_qa(laptop_name)"
        )
        conn.commit()


def _laptop_already_generated(conn: sqlite3.Connection, laptop_name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM ground_truth_qa WHERE laptop_name = ? LIMIT 1", (laptop_name,)
    )
    return cur.fetchone() is not None


def _delete_laptop_rows(conn: sqlite3.Connection, laptop_name: str) -> None:
    conn.execute("DELETE FROM ground_truth_qa WHERE laptop_name = ?", (laptop_name,))


def _insert_qa_items(
    conn: sqlite3.Connection,
    laptop_name: str,
    laptop_price: int,
    price_tier: str,
    context: str,
    items: List[dict],
) -> None:
    rows = [
        (
            laptop_name,
            laptop_price,
            price_tier,
            item.get("question", ""),
            item.get("ground_truth_answer", ""),
            json.dumps(item.get("requirements", {})),
            context,
        )
        for item in items
        if item.get("question") and item.get("ground_truth_answer")
    ]
    conn.executemany(
        """
        INSERT INTO ground_truth_qa
            (laptop_name, laptop_price, price_tier, question, ground_truth_answer, requirements_json, context)
        VALUES (?, ?, ?, ?, ?, ?, ?)
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
            "SELECT laptop_name, laptop_price, price_tier, question, ground_truth_answer, "
            "requirements_json, context, created_at FROM ground_truth_qa ORDER BY id"
        )
        rows = cur.fetchall()

    results = []
    for r in rows:
        try:
            requirements = json.loads(r["requirements_json"]) if r["requirements_json"] else {}
        except Exception:
            requirements = {}
        results.append({
            "laptop_name": r["laptop_name"],
            "laptop_price": r["laptop_price"],
            "price_tier": r["price_tier"],
            "question": r["question"],
            "ground_truth_answer": r["ground_truth_answer"],
            "requirements": requirements,
            "context": r["context"],
            "created_at": r["created_at"],
        })
    return results


def dataset_stats(db_path: str = GROUND_TRUTH_DB_PATH) -> dict:
    if not os.path.exists(db_path):
        return {"laptops_covered": 0, "qa_count": 0, "by_tier": {}}
    with _connect(db_path) as conn:
        qa_count = conn.execute("SELECT COUNT(*) FROM ground_truth_qa").fetchone()[0]
        laptops_covered = conn.execute(
            "SELECT COUNT(DISTINCT laptop_name) FROM ground_truth_qa"
        ).fetchone()[0]
        by_tier = dict(conn.execute(
            "SELECT price_tier, COUNT(DISTINCT laptop_name) FROM ground_truth_qa GROUP BY price_tier"
        ).fetchall())
    return {"laptops_covered": laptops_covered, "qa_count": qa_count, "by_tier": by_tier}


# =============================================================================
# PROMPT BUILDING / PARSING
# =============================================================================

def _build_ground_truth_prompt(laptop: dict, features: dict, n: int) -> str:
    """Prompt DeepSeek to write n hypothetical Q/A pairs this laptop should ground-truth answer."""
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
        f'{{"question": "...", "ground_truth_answer": "...", '
        f'"requirements": {{"GPU intensity": "low|medium|high", "Display quality": "low|medium|high", '
        f'"Portability": "low|medium|high", "Multitasking": "low|medium|high", '
        f'"Processing speed": "low|medium|high", "Budget": <integer>}}}}'
    )


def _parse_ground_truth_response(raw: str) -> List[dict]:
    """Strip <think> blocks and pull the JSON array out of a DeepSeek response."""
    cleaned = clean_deepseek_response(raw)
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
    except Exception as e:
        logger.warning(f"   Ground truth JSON parse failed: {e}")
        return []
    return items if isinstance(items, list) else []


# =============================================================================
# MAIN GENERATION ENTRY POINT
# =============================================================================

def generate_ground_truth_dataset(
    pkl_path: str = "data/index.pkl",
    db_path: str = GROUND_TRUTH_DB_PATH,
    n_per_laptop: int = 2,
    strategy: str = "representative",   # "representative" | "all" | "first_n"
    top_n_per_tier: int = DEFAULT_TOP_N_PER_TIER,
    limit: Optional[int] = None,        # only used when strategy == "first_n"
    resume: bool = True,
) -> List[dict]:
    """
    Use DeepSeek-R1 to generate hypothetical (question, ground_truth_answer) pairs,
    grounded in each laptop's real specs and the GPU/Display/Portability/
    Multitasking/Processing-speed/Budget requirement schema.

    strategy:
      "representative" (default) — top_n_per_tier cheapest + mean-range +
          median-range + most expensive laptops (4 tiers). Recommended for a
          catalog this large (e.g. 11k laptops) instead of processing
          everything.
      "all"      — every laptop in pkl_path. Slow on a large catalog.
      "first_n"  — the first `limit` laptops in file order (old behaviour).

    Writes each laptop's rows to SQLite immediately after generating them, so
    a crash mid-run doesn't lose earlier work. With resume=True (default),
    laptops that already have rows in the DB are skipped — so re-running this
    (or importing this module, or restarting the app) never regenerates
    anything unless you explicitly ask it to. Set resume=False to wipe and
    regenerate the selected laptops from scratch.

    Returns the full flattened list of QA records now in the DB.
    """
    init_db(db_path)

    try:
        with open(pkl_path, "rb") as f:
            laptops = pickle.load(f)
    except Exception as e:
        logger.error(f"Could not load laptops from {pkl_path}: {e}")
        return load_ground_truth_dataset(db_path)

    if strategy == "representative":
        selected = select_representative_laptops(laptops, top_n=top_n_per_tier)
    elif strategy == "first_n":
        selected = [(l, "n/a") for l in (laptops[:limit] if limit else laptops)]
    elif strategy == "all":
        selected = [(l, "n/a") for l in laptops]
    else:
        raise ValueError(f"Unknown strategy: {strategy!r} (expected representative/all/first_n)")

    # Reuse the app's feature cache (keyword-tier classification) if present,
    # so we don't reclassify laptops that build_vector_store already did.
    feature_cache: Dict[str, dict] = {}
    if os.path.exists(_FEATURE_CACHE_PATH):
        try:
            with open(_FEATURE_CACHE_PATH) as f:
                feature_cache = json.load(f)
        except Exception:
            feature_cache = {}

    llm = _get_ground_truth_llm()

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
                raw = llm.invoke(prompt)
                items = _parse_ground_truth_response(raw)
            except Exception as e:
                logger.warning(f"   [{i}/{len(selected)}] ❌ generation failed for {name}: {e}")
                items = []

            _insert_qa_items(conn, name, laptop.get("price", 0), price_tier, desc, items)
            logger.info(f"   [{i}/{len(selected)}] ✅ {name} [{price_tier}]: {len(items)} QA pairs")

    results = load_ground_truth_dataset(db_path)
    logger.info(f"📝 Ground truth generation done: {len(results)} QA pairs total in {db_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate hypothetical ground-truth QA pairs via DeepSeek-R1")
    parser.add_argument("--n", type=int, default=2, help="QA pairs per laptop (default 2)")
    parser.add_argument(
        "--strategy", choices=["representative", "all", "first_n"], default="representative",
        help="representative = cheapest/mid/expensive top-N sample (default); all = whole catalog; first_n = first --limit laptops",
    )
    parser.add_argument(
        "--top-n-per-tier", type=int, default=DEFAULT_TOP_N_PER_TIER,
        help="Cheapest/mean-range/median-range/expensive laptops each (only for --strategy representative, default 10)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max laptops (only for --strategy first_n)")
    parser.add_argument("--pkl-path", type=str, default="data/index.pkl")
    parser.add_argument("--db-path", type=str, default=GROUND_TRUTH_DB_PATH)
    parser.add_argument("--no-resume", action="store_true", help="Wipe & regenerate selected laptops")
    args = parser.parse_args()

    results = generate_ground_truth_dataset(
        pkl_path=args.pkl_path,
        db_path=args.db_path,
        n_per_laptop=args.n,
        strategy=args.strategy,
        top_n_per_tier=args.top_n_per_tier,
        limit=args.limit,
        resume=not args.no_resume,
    )
    print(f"\nGenerated/loaded {len(results)} ground-truth QA pairs -> {args.db_path}")