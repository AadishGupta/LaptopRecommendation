"""
classifier_stats.py — distribution + mean tier for the 5 classifier
dimensions (GPU intensity, Display quality, Portability, Multitasking,
Processing speed) across the reduced 1000-laptop dataset.

Reads the existing feature cache JSON (built by _classify_one and saved at
_FEATURE_CACHE_PATH in agent_functions.py) rather than re-classifying
anything or importing the app itself.

Check agent_functions.py for the exact _FEATURE_CACHE_PATH value if the
default below doesn't match, and pass --cache-path.

Usage:
    python classifier_stats.py
    python classifier_stats.py --pkl data/index_reduced.pkl --cache-path feature_cache.json
"""
import pickle
import json
import argparse
from collections import Counter

DIMENSIONS = ["GPU intensity", "Display quality", "Portability", "Multitasking", "Processing speed"]
TIERS = ["low", "medium", "high"]
TIER_VALUE = {"low": 1, "medium": 2, "high": 3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="data/index_reduced.pkl")
    parser.add_argument(
        "--cache-path", default="feature_cache.json",
        help="Path to feature cache JSON (see _FEATURE_CACHE_PATH in agent_functions.py)"
    )
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        laptops = pickle.load(f)

    with open(args.cache_path, encoding="utf-8") as f:
        feature_cache = json.load(f)

    counts = {d: Counter() for d in DIMENSIONS}
    missing = 0

    for lap in laptops:
        desc = lap.get("description", "")
        features = feature_cache.get(desc)
        if not features:
            missing += 1
            continue
        for d in DIMENSIONS:
            tier = features.get(d)
            if tier in TIERS:
                counts[d][tier] += 1

    print(f"Loaded {len(laptops)} laptops, {missing} not found in feature cache\n")

    for d in DIMENSIONS:
        total = sum(counts[d].values())
        print(d)
        if total == 0:
            print("  (no cached data for this dimension)\n")
            continue
        for t in TIERS:
            c = counts[d].get(t, 0)
            pct = c / total * 100
            print(f"  {t:8s}: {c:4d} ({pct:.1f}%)")
        avg = sum(TIER_VALUE[t] * c for t, c in counts[d].items()) / total
        print(f"  mean tier (1=low, 2=medium, 3=high): {avg:.2f}\n")


if __name__ == "__main__":
    main()
