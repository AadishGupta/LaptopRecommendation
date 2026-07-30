"""
price_stats.py — compute price statistics from data/index_reduced.pkl
so budget-related parameters in generate_eval_questions.py and
ground_truth_generator.py can be updated to match the new 1000-laptop set.

Usage:
    python price_stats.py
    python price_stats.py --pkl data/index_reduced.pkl
"""
import pickle
import argparse
import statistics

DEFAULT_PKL = "data/index_reduced.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default=DEFAULT_PKL)
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        records = pickle.load(f)

    prices = sorted(r["price"] for r in records)
    n = len(prices)

    mean = statistics.mean(prices)
    median = statistics.median(prices)
    stdev = statistics.stdev(prices)
    lowest = prices[0]
    highest = prices[-1]

    # Terciles (low/medium/high budget split points) — same idea as the
    # 3-way price split used in ground_truth_generator.py's "grid" strategy.
    tercile_1 = prices[n // 3]
    tercile_2 = prices[(2 * n) // 3]

    # Quartiles too, in case you want finer bucketing
    q1 = prices[n // 4]
    q3 = prices[(3 * n) // 4]

    print(f"Loaded {n} laptops from '{args.pkl}'\n")
    print(f"  Lowest price   : {lowest:,}")
    print(f"  Highest price  : {highest:,}")
    print(f"  Mean price     : {mean:,.0f}")
    print(f"  Median price   : {median:,.0f}")
    print(f"  Std deviation  : {stdev:,.0f}")
    print()
    print(f"  Tercile boundaries (low/medium/high budget split):")
    print(f"    low    : < {tercile_1:,}")
    print(f"    medium : {tercile_1:,} - {tercile_2:,}")
    print(f"    high   : > {tercile_2:,}")
    print()
    print(f"  Quartile boundaries:")
    print(f"    Q1 (25%) : {q1:,}")
    print(f"    Q3 (75%) : {q3:,}")


if __name__ == "__main__":
    main()
