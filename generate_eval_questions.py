"""
generate_eval_questions.py

Generates 200 hypothetical user questions for grounding/evaluation, using
DeepSeek-R1 directly (NOT through the app — the app now runs on llama3.1,
but deepseek-r1 is still installed and better suited to generating varied,
realistic questions since it can reason about what a real user would ask).

Output: eval_questions.json — a flat list of 200 question strings, tagged
by category, ready to feed into run_grounding_eval.py.

Usage:
    python generate_eval_questions.py
    python generate_eval_questions.py --count 200 --out eval_questions.json
"""

import argparse
import json
import re
import time

from deepseek_wrapper import DeepSeekLLM, clean_deepseek_response, ERROR_PREFIX

# Categories mirror the app's actual use cases (kg_rag UseCase nodes were:
# 5 types) — adjust this list if your domain differs.
#
# Budget numbers below reflect the reduced 1000-laptop catalog (2026 sample):
#   min=9,990  median=52,994  mean=68,935  Q1=34,990  Q3=81,990  max=500,000
CATEGORIES = [
    "gaming laptop shopping (GPU, refresh rate, budget tradeoffs, typically budgets from ₹60,000-₹150,000+)",
    "student/budget laptop shopping (portability, battery, low budget, typically under ₹35,000)",
    "business/professional laptop shopping (multitasking, build quality, typically ₹50,000-₹90,000)",
    "creative work laptop shopping (display quality, GPU for editing, typically ₹80,000-₹200,000+)",
    "laptop upgrade advice (comparing current laptop to new options)",
    "side-by-side laptop comparison requests (naming two specific models)",
]

QUESTIONS_PER_CATEGORY = 200 // len(CATEGORIES)  # ~33 each, adjust with --count


def generate_batch(llm: DeepSeekLLM, category: str, n: int, max_retries: int = 2) -> list[str]:
    """
    Ask DeepSeek for n realistic user questions in one category.

    Retries on transport-level failures (timeout, connection error, non-200
    status) instead of banking the wrapper's "Error: ..." string as if it
    were a generated question — previously that string (56+ chars) slipped
    past the `len(l) > 8` filter below and got saved into eval_questions.json
    as a bogus entry.
    """
    prompt = (
        f"You are simulating real users of a laptop shopping assistant chatbot. "
        f"Generate {n} realistic, DIVERSE questions/messages a user might type "
        f"for this category: {category}.\n\n"
        f"Vary phrasing, vocabulary, and specificity (some vague, some very "
        f"detailed with exact budgets/specs). Write ONLY the questions, one "
        f"per line, no numbering, no extra commentary."
    )

    for attempt in range(1, max_retries + 2):  # e.g. max_retries=2 -> 3 total attempts
        raw = llm.invoke(prompt)

        if raw.startswith(ERROR_PREFIX):
            print(f"    [attempt {attempt}] {raw}")
            if attempt <= max_retries:
                print(f"    retrying...")
                continue
            else:
                print(f"    giving up after {attempt} attempts for this category")
                return []

        cleaned = clean_deepseek_response(raw)
        lines = [
            re.sub(r"^[\d\.\)\-\s]+", "", line).strip()
            for line in cleaned.splitlines()
            if line.strip()
        ]
        questions = [l for l in lines if len(l) > 8 and not l.startswith(ERROR_PREFIX)][:n]

        if not questions and attempt <= max_retries:
            print(f"    [attempt {attempt}] got 0 usable lines, retrying...")
            continue

        return questions

    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--out", type=str, default="eval_questions.json")
    parser.add_argument("--model", type=str, default="deepseek-r1:7b")
    args = parser.parse_args()

    per_category = max(1, args.count // len(CATEGORIES))

    llm = DeepSeekLLM(
        model=args.model,
        temperature=0.9,      # higher temperature = more variety across questions
        num_predict=3000,     # raised from 1500 — thinking was eating the budget before
                              # any of the 33 questions got written (e.g. only 13/33 came
                              # through). More headroom so both the <think> pass and the
                              # full answer fit.
        num_ctx=4096,
        think=True,           # reasoning helps generate more realistic variety here
        keep_alive="15m",
        system_prompt=None,
        timeout=360,          # longer generations at num_predict=3000 need more time
    )

    all_questions = []
    for category in CATEGORIES:
        print(f"Generating {per_category} questions for: {category}")
        batch = generate_batch(llm, category, per_category)
        print(f"  -> got {len(batch)} questions")
        if not batch:
            print(f"  WARNING: 0 questions for this category after retries — check Ollama/GPU status")
        for q in batch:
            all_questions.append({"category": category, "question": q})
        time.sleep(0.5)  # small pause between calls, not strictly necessary

    # Top up to the exact target count if categories under-produced
    while len(all_questions) < args.count and all_questions:
        all_questions.append(all_questions[len(all_questions) % len(all_questions)])

    all_questions = all_questions[: args.count]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_questions)} questions to {args.out}")


if __name__ == "__main__":
    main()