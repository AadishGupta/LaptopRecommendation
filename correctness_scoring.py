"""
correctness_scoring.py

Fuzzy correctness scoring for laptop recommendations against ground_truth.db.

--------------------------------------------------------------------------
WHY FUZZY, NOT EXACT-NAME-MATCH
--------------------------------------------------------------------------
ground_truth.db (see ground_truth_generator.py) is keyed by a `combo_key`
built from a fixed requirement grid: 5 feature dimensions (GPU intensity /
Display quality / Portability / Multitasking / Processing speed, each
low/medium/high) x 3 budget tiers (low/medium/high, split by the catalog's
own price distribution: low = min..median, medium = median..cap,
high = cap..max — see compute_budget_ranges below, mirrors
ground_truth_generator.compute_budget_ranges). For each combo, ground truth
picked ONE representative real laptop — but many laptops in a 1,000-laptop
catalog can equally satisfy the same combo. Scoring the live app's pick
against ground truth by exact name would mark those equally-valid laptops
as wrong. So we score two things instead, same as ground_truth_generator's
own match_score logic:

    correctness = FIELD_WEIGHT * field_agreement_frac + PRICE_WEIGHT * price_closeness

- field_agreement_frac: classify the RECOMMENDED laptop's own description
  with the same keyword rules the rest of the app uses (_KW_RULES /
  _classify_one) and count how many of the 5 dimensions match what the user
  actually asked for. This asks "is the recommended laptop the right KIND
  of laptop", independent of which specific laptop ground truth happened to
  pick. Directly mirrors ground_truth_generator._match_score's 0-5 count.
- price_closeness: how close the recommended laptop's price is to the
  ground-truth laptop's price, normalized by that budget tier's price-range
  width, so a miss in the "low" tier (narrow range) is penalized more per-
  rupee than the same rupee miss in the "high" tier (wide range).

--------------------------------------------------------------------------
NOTE ON THE _KW_RULES/_classify_one DUPLICATION BELOW
--------------------------------------------------------------------------
These are copied from agent_functions.py rather than imported, so this
module stays import-light (no qdrant_client / langchain_ollama / langgraph /
reportlab needed just to score a recommendation or run the standalone demo
at the bottom of this file). When wiring this into run_grounding_eval.py —
where agent_functions is already imported — pass agent_functions._classify_one
as `classify_fn` to score_recommendation()/evaluate_recommendation() instead,
so there is exactly one live copy of the keyword rules in production and
this file's copy only backs standalone tests/tools. If you change
_KW_RULES in agent_functions.py, mirror the change here too (or better:
always pass classify_fn explicitly from run_grounding_eval.py and treat
this copy as test-only).
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from typing import Callable, Dict, List, Optional, Tuple

DIMENSIONS = [
    "GPU intensity",
    "Display quality",
    "Portability",
    "Multitasking",
    "Processing speed",
]

FIELD_WEIGHT = 0.7
PRICE_WEIGHT = 0.3

# --- copied from agent_functions.py — see NOTE above ------------------------
_KW_RULES = {
    "GPU intensity": {
        "high": ["rtx 4090", "rtx 4080", "rtx 4070", "rtx 4060", "rx 7900", "8gb vram", "16gb vram"],
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


def _classify_one_local(description: str) -> dict:
    """Local mirror of agent_functions._classify_one — see NOTE above."""
    lower = (description or "").lower()
    features = {}
    for feature, tiers in _KW_RULES.items():
        matched = "medium"
        for tier in ("high", "low"):
            if any(kw in lower for kw in tiers[tier]):
                matched = tier
                break
        features[feature] = matched
    return features


# =============================================================================
# BUDGET TIERS
# =============================================================================

def compute_budget_ranges(prices: List[float], medium_cap: float = 90_000) -> Dict[str, Tuple[float, float]]:
    """Mirrors ground_truth_generator.compute_budget_ranges exactly:
    low = min..median, medium = median..medium_cap, high = medium_cap..max.
    Call this with the CURRENT live catalog's prices at eval time — if the
    catalog has changed size (e.g. the 1,000-laptop reduction mentioned in
    prior work), tier boundaries may have shifted from whatever ground_truth.db
    was generated against, and matching combo_keys depends on using the same
    boundaries ground truth used. See budget_ranges_from_db() below for a
    fallback when only ground_truth.db itself is available (as in this file's
    standalone demo).
    """
    prices = [p for p in prices if isinstance(p, (int, float)) and p > 0]
    if not prices:
        return {"low": (0, 0), "medium": (0, 0), "high": (0, 0)}
    min_p, max_p = min(prices), max(prices)
    median_p = statistics.median(prices)
    low_medium_split = min(median_p, medium_cap)
    return {
        "low": (min_p, low_medium_split),
        "medium": (low_medium_split, medium_cap),
        "high": (medium_cap, max_p),
    }


def budget_ranges_from_db(db_path: str = "ground_truth.db") -> Dict[str, Tuple[float, float]]:
    """Fallback tier-boundary source when the live catalog isn't available:
    reconstruct approximate ranges from the price_tier/laptop_price columns
    already baked into ground_truth.db. Prefer compute_budget_ranges() against
    the live catalog in production — this is for offline testing/demoing
    against ground_truth.db alone (see __main__ below).
    """
    conn = sqlite3.connect(db_path)
    ranges = {}
    for tier in ("low", "medium", "high"):
        row = conn.execute(
            "SELECT MIN(laptop_price), MAX(laptop_price) FROM ground_truth_qa WHERE price_tier = ?",
            (tier,),
        ).fetchone()
        ranges[tier] = (row[0] or 0, row[1] or 0)
    conn.close()
    return ranges


def budget_tier_from_price(budget: float, ranges: Dict[str, Tuple[float, float]]) -> str:
    """Given a rupee budget number (e.g. state["requirements"]["Budget"]),
    return which tier it falls in using the same boundaries ground truth
    combos were generated with."""
    lo, hi = ranges.get("low", (0, 0))
    if budget <= hi:
        return "low"
    lo, hi = ranges.get("medium", (0, 0))
    if budget <= hi:
        return "medium"
    return "high"


# =============================================================================
# COMBO KEY (must match ground_truth_generator._combo_key exactly)
# =============================================================================

def combo_key_from_requirements(requirements: dict, budget_ranges: Dict[str, Tuple[float, float]]) -> Tuple[str, dict]:
    """Build the same combo_key string ground_truth_generator._combo_key
    produces, from the live app's extracted requirements
    (agent_functions._extract_requirements output — same 5 dimension names,
    plus a numeric "Budget"). Returns (combo_key, combo_dict) so callers can
    also inspect the resolved tier combo directly.
    """
    combo = {d: str(requirements.get(d, "medium")).lower() for d in DIMENSIONS}
    budget_tier = budget_tier_from_price(requirements.get("Budget", 0), budget_ranges)
    combo["Budget tier"] = budget_tier
    key = "|".join(f"{d}:{combo[d]}" for d in DIMENSIONS) + f"|Budget:{budget_tier}"
    return key, combo


# =============================================================================
# GROUND TRUTH INDEX
# =============================================================================

class GroundTruthIndex:
    """Loads ground_truth.db once and indexes rows by combo_key for O(1)
    lookup during a 200-question eval run (instead of a query per question).
    """

    def __init__(self, db_path: str = "ground_truth.db"):
        self.db_path = db_path
        self._by_combo: Dict[str, List[dict]] = {}
        self._load()

    def _load(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT combo_key, laptop_name, laptop_price, price_tier, match_score, "
            "question, ground_truth_answer, requirements_json "
            "FROM ground_truth_qa WHERE combo_key IS NOT NULL"
        )
        for r in cur.fetchall():
            row = dict(r)
            row["requirements"] = json.loads(row["requirements_json"]) if row["requirements_json"] else {}
            self._by_combo.setdefault(row["combo_key"], []).append(row)
        conn.close()

    def lookup(self, combo_key: str) -> List[dict]:
        return self._by_combo.get(combo_key, [])

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_combo.values())

    @property
    def n_combos(self) -> int:
        return len(self._by_combo)


# =============================================================================
# SCORING
# =============================================================================

def field_agreement(recommended_features: dict, combo: dict) -> int:
    """0-5: how many dimensions the recommended laptop's real features match
    the requested combo. Mirrors ground_truth_generator._match_score exactly,
    just applied to a live recommendation instead of a catalog laptop being
    picked as ground truth."""
    return sum(1 for d in DIMENSIONS if recommended_features.get(d) == combo.get(d))


def price_closeness(recommended_price: float, gt_price: float, tier_range: Tuple[float, float]) -> float:
    """1.0 = same price as ground truth's laptop; decays to 0 as the gap
    approaches the full width of that budget tier's price range."""
    lo, hi = tier_range
    width = max(hi - lo, 1.0)
    diff = abs((recommended_price or 0) - (gt_price or 0))
    return max(0.0, 1.0 - diff / width)


def score_recommendation(
    recommended_laptop: dict,
    combo: dict,
    gt_row: dict,
    budget_ranges: Dict[str, Tuple[float, float]],
    classify_fn: Optional[Callable[[str], dict]] = None,
) -> dict:
    """Score one recommended laptop against one matched ground-truth row.
    `classify_fn` defaults to the local keyword-rule copy; pass
    agent_functions._classify_one from run_grounding_eval.py in production
    (see module docstring NOTE)."""
    classify_fn = classify_fn or _classify_one_local
    rec_features = classify_fn(recommended_laptop.get("description", ""))

    fa = field_agreement(rec_features, combo)
    # Score relative to the ceiling ground truth itself achieved for this
    # combo, not a theoretical 5/5 — some requirement combos have no clean
    # match anywhere in the catalog, and ground_truth.db's own match_score
    # column (2-5 across all 729 combos, see correctness_scoring.py demo
    # output) proves ground truth itself didn't always hit 5/5 either. A
    # live pick that reaches GT's own ceiling should score 1.0, not be
    # penalized for a gap that isn't the retrieval pipeline's fault.
    gt_ceiling = gt_row.get("match_score") or len(DIMENSIONS)
    fa_frac = min(1.0, fa / gt_ceiling)

    tier_range = budget_ranges.get(combo.get("Budget tier", "medium"), (0, 0))
    pc = price_closeness(recommended_laptop.get("price", 0), gt_row.get("laptop_price", 0), tier_range)

    correctness = FIELD_WEIGHT * fa_frac + PRICE_WEIGHT * pc

    return {
        "correctness": round(correctness, 4),
        "field_agreement": fa,
        "field_agreement_ceiling": gt_ceiling,
        "field_agreement_frac": round(fa_frac, 4),
        "price_closeness": round(pc, 4),
        "recommended_features": rec_features,
        "combo": combo,
        "gt_laptop_name": gt_row.get("laptop_name"),
        "gt_laptop_price": gt_row.get("laptop_price"),
        "recommended_name": recommended_laptop.get("name"),
        "recommended_price": recommended_laptop.get("price"),
    }


def evaluate_recommendation(
    recommended_laptop: dict,
    requirements: dict,
    gt_index: GroundTruthIndex,
    budget_ranges: Dict[str, Tuple[float, float]],
    classify_fn: Optional[Callable[[str], dict]] = None,
) -> Optional[dict]:
    """End-to-end: requirements -> combo_key -> ground truth lookup -> score.
    Returns None if no ground-truth row exists for this combo (grid is 729
    combos; every combo should have one, but this guards a mismatched/stale db).
    """
    combo_key, combo = combo_key_from_requirements(requirements, budget_ranges)
    rows = gt_index.lookup(combo_key)
    if not rows:
        return None
    gt_row = rows[0]  # all rows sharing a combo_key point at the same laptop
    result = score_recommendation(recommended_laptop, combo, gt_row, budget_ranges, classify_fn)
    result["combo_key"] = combo_key
    return result


# =============================================================================
# STANDALONE DEMO — proves the matcher + scorer against the real
# ground_truth.db, without needing Qdrant/Ollama/agent_functions running.
# =============================================================================

if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "ground_truth.db"
    gt_index = GroundTruthIndex(db_path)
    budget_ranges = budget_ranges_from_db(db_path)

    print(f"Loaded {len(gt_index)} QA rows across {gt_index.n_combos} combos from {db_path}")
    print(f"Budget ranges (derived from db's own price_tier column): {budget_ranges}\n")

    # Pull a handful of real combos to test against, spread across tiers.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    sample_rows = conn.execute(
        "SELECT * FROM ground_truth_qa WHERE combo_key IS NOT NULL "
        "GROUP BY price_tier ORDER BY RANDOM() LIMIT 3"
    ).fetchall()
    # also grab one arbitrary "wrong" laptop to use as a deliberately bad recommendation
    wrong_row = conn.execute(
        "SELECT * FROM ground_truth_qa WHERE combo_key IS NOT NULL ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    conn.close()

    wrong_laptop = {
        "name": wrong_row["laptop_name"],
        "price": wrong_row["laptop_price"],
        "description": wrong_row["laptop_name"],  # no full description column here; name only for this smoke test
    }

    for row in sample_rows:
        requirements = json.loads(row["requirements_json"])
        # requirements_json stores tiers under "Budget tier", but
        # combo_key_from_requirements expects a numeric "Budget" (mirrors what
        # agent_functions._extract_requirements actually produces at runtime).
        # Reconstruct a representative numeric budget from the tier's range
        # midpoint so the demo exercises the same numeric->tier path production
        # code will use.
        tier = requirements.get("Budget tier", row["price_tier"])
        lo, hi = budget_ranges.get(tier, (0, 0))
        req_for_lookup = {**requirements, "Budget": (lo + hi) / 2 if hi else lo}

        print("=" * 78)
        print(f"Combo: {row['combo_key']}")
        print(f"Ground truth laptop: {row['laptop_name']!r} (₹{row['laptop_price']:.0f})")

        # Case 1: recommend the CORRECT laptop -> expect a high score.
        correct_laptop = {
            "name": row["laptop_name"],
            "price": row["laptop_price"],
            "description": row["context"] or row["laptop_name"],
        }
        result = evaluate_recommendation(correct_laptop, req_for_lookup, gt_index, budget_ranges)
        print(f"  [correct pick]  correctness={result['correctness']:.3f}  "
              f"field_agreement={result['field_agreement']}/5 (GT ceiling={result['field_agreement_ceiling']})  "
              f"price_closeness={result['price_closeness']:.3f}")

        # Case 2: recommend an unrelated laptop -> expect a low score.
        result_wrong = evaluate_recommendation(wrong_laptop, req_for_lookup, gt_index, budget_ranges)
        print(f"  [unrelated pick: {wrong_laptop['name'][:40]!r}]  "
              f"correctness={result_wrong['correctness']:.3f}  "
              f"field_agreement={result_wrong['field_agreement']}/5 (GT ceiling={result_wrong['field_agreement_ceiling']})  "
              f"price_closeness={result_wrong['price_closeness']:.3f}")
