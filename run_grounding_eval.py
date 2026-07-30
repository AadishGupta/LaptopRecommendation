"""
run_grounding_eval.py

Feeds each question from eval_questions.json through the app's actual
LangGraph pipeline (run_turn), scores the recommendation it produces against
ground_truth.db using correctness_scoring.py, and aggregates the results into
a summary report.

CHANGED FROM THE ORIGINAL VERSION OF THIS SCRIPT: it used to read
state.get("ragas_scores", {}) / state.get("judge_scores", {}) — those keys
never existed anywhere in agent_functions.py or kg_rag.py (RAGAS lives in a
separate offline ragas_eval.py script that reads a JSONL log; there was never
a live judge_scores computation), so every prior run of this script silently
logged {} for both metrics without erroring. This version replaces that with
real reference-based scoring against ground_truth.db.

ADAPTIVE RETRIEVAL TUNING: each question also drives a small epsilon-greedy
bandit (retrieval_memory.py, backed by its own retrieval_memory.db — separate
from ground_truth.db and from the app's Qdrant case_memory collection) that
picks agent_functions.py's kg_weight (the KG/vector blend in search_node) for
the NEXT question based on which values have scored well so far. Over a
200-question run this should visibly converge toward whichever kg_weight
value best fits your catalog/classifier, instead of the previous hardcoded
0.3 for every question. The bandit's learned state persists across separate
runs of this script (it's just a SQLite file), so accuracy should keep
improving run over run, not just within one run.

Requires: Ollama running, Qdrant running, vector store already built — same
prerequisites as running agent_app.py itself.

Usage:
    python run_grounding_eval.py --questions eval_questions.json --out grounding_report.json
    python run_grounding_eval.py --limit 20          # quick smoke test
"""

import argparse
import json
import math
import time
import datetime

import agent_functions
from agent_functions import (
    build_vector_store,
    make_initial_state,
    run_turn,
    write_case,          # quality-gated case bank write — used below after scoring
)

import correctness_scoring
import retrieval_memory


def _retrieval_metrics(ranked: list[dict], eval_result: dict, budget_ranges: dict, k: int = 10) -> dict:
    """Evaluate retrieval against the GT *requirement combination*.

    ground_truth.db contains one representative laptop per tier combination,
    not a unique document label. A different laptop is relevant when it meets
    at least the same feature ceiling within the same generated budget tier.
    Exact model-name rank is retained only as a diagnostic.
    """
    combo = eval_result["combo"]
    ceiling = eval_result["field_agreement_ceiling"]
    lo, hi = budget_ranges.get(combo["Budget tier"], (0, float("inf")))
    relevance = []
    exact_rank = None
    target = (eval_result.get("gt_laptop_name") or "").strip().lower()
    for rank, item in enumerate(ranked[:k], 1):
        features = agent_functions._classify_one(item.get("description", ""))
        agreement = sum(features.get(d) == combo.get(d) for d in correctness_scoring.DIMENSIONS)
        in_budget_tier = lo <= float(item.get("price", 0) or 0) <= hi
        relevance.append(min(1.0, agreement / max(ceiling, 1)) if in_budget_tier else 0.0)
        if item.get("name", "").strip().lower() == target:
            exact_rank = rank
    relevant_ranks = [i + 1 for i, score in enumerate(relevance) if score >= 1.0]
    rank = relevant_ranks[0] if relevant_ranks else None
    dcg = sum(score / math.log2(i + 2) for i, score in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal))
    return {
        f"recall_at_{k}": float(rank is not None),
        f"precision_at_{k}": round(sum(score >= 1.0 for score in relevance) / k, 4),
        "mrr": round(1.0 / rank, 4) if rank else 0.0,
        f"ndcg_at_{k}": round(dcg / idcg, 4) if idcg else 0.0,
        "first_relevant_rank": rank,
        "exact_ground_truth_rank": exact_rank,
    }


def _stratified_sample(questions: list[dict], limit: int) -> list[dict]:
    """
    Round-robin across categories instead of slicing the front of the file.
    eval_questions.json lists all of one category's ~33 questions before the
    next category starts, so questions[:limit] on a small --limit silently
    tests only the first category. This keeps every category represented
    (as evenly as `limit` allows) even for a quick smoke test.
    """
    by_category: dict[str, list[dict]] = {}
    for q in questions:
        by_category.setdefault(q.get("category", "unknown"), []).append(q)

    categories = list(by_category.keys())
    sampled: list[dict] = []
    i = 0
    while len(sampled) < limit and any(by_category.values()):
        cat = categories[i % len(categories)]
        if by_category[cat]:
            sampled.append(by_category[cat].pop(0))
        i += 1
    return sampled[:limit]


def _resolve_budget_ranges(gt_db_path: str) -> dict:
    """Prefer computing tier boundaries from the CURRENT live catalog (what
    the app is actually indexing right now) so combo_key matching uses the
    same boundaries the running app would use. Falls back to reconstructing
    approximate ranges from ground_truth.db's own price_tier column if the
    catalog can't be loaded (e.g. Qdrant reachable but collection empty).
    """
    catalog = agent_functions.load_catalog_from_qdrant()
    prices = [l.get("price", 0) for l in catalog if l.get("price")]
    if prices:
        ranges = correctness_scoring.compute_budget_ranges(prices)
        print(f"Budget tier ranges (from live catalog, n={len(prices)} priced laptops): {ranges}")
        return ranges

    print("WARNING: could not load a priced catalog from Qdrant — falling back to "
          "budget ranges reconstructed from ground_truth.db's own price_tier column. "
          "These may not match the CURRENT catalog's boundaries if it has changed size "
          "since ground_truth.db was generated.")
    ranges = correctness_scoring.budget_ranges_from_db(gt_db_path)
    print(f"Budget tier ranges (from ground_truth.db): {ranges}")
    return ranges


OFFLINE_OPTIMIZATION_STAGES = (
    ("kg_weight", tuple(round(i / 10, 1) for i in range(11))),
    ("hyde_enabled", (True, False)),
    ("top_k", (5, 10, 20, 30, 50)),
    ("reranker_enabled", (True, False)),
)


def _action_key(action: dict) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def _evaluate_retrieval_action(question: str, action: dict | None, gt_index, budget_ranges: dict) -> dict:
    """Run one complete, offline-only retrieval episode for a fixed action."""
    state = make_initial_state()
    state = run_turn(state, "Hello")
    state["offline_evaluation"] = True
    if action is not None:
        state["retrieval_action_override"] = dict(action)
    try:
        state = run_turn(state, question)
    except Exception as exc:
        return {"action": action or {}, "state": state, "error": str(exc), "evaluation": None}

    requirements = state.get("requirements", {}) or {}
    top_k = state.get("top_k_laptops") or []
    recommended = top_k[0] if top_k else None
    evaluation = None
    if recommended is not None:
        evaluation = correctness_scoring.evaluate_recommendation(
            recommended, requirements, gt_index, budget_ranges,
            classify_fn=agent_functions._classify_one,
        )
    return {
        "action": state.get("retrieval_action", action or {}),
        "state": state,
        "recommended": recommended,
        "evaluation": evaluation,
        "error": None,
    }


def _select_winner(candidates: list[dict]) -> dict | None:
    """Choose maximum correctness; break ties by closeness to default KG blend."""
    valid = [candidate for candidate in candidates if candidate.get("evaluation") is not None]
    if not valid:
        return None
    return max(
        valid,
        key=lambda candidate: (
            candidate["evaluation"]["correctness"],
            -abs(float(candidate["action"].get("kg_weight", 0.3)) - 0.3),
        ),
    )


def _stage_variants(base_action: dict, stage_name: str, values: tuple) -> list[dict]:
    """Create candidate actions for one parameter stage; easy to extend."""
    variants = []
    for value in values:
        action = dict(base_action)
        if stage_name == "top_k":
            action["dense_top_k"] = value
            action["sparse_top_k"] = value
        else:
            action[stage_name] = value
        variants.append(action)
    return variants


def _staged_optimize(question: str, baseline: dict, gt_index, budget_ranges: dict) -> tuple[dict | None, list[dict], float | None]:
    """Coordinate descent over retrieval parameters, caching duplicate trials."""
    trial_cache = {_action_key(baseline["action"]): baseline}
    trials = [baseline]
    default_correctness = (baseline.get("evaluation") or {}).get("correctness")
    best = baseline if baseline.get("evaluation") is not None else None
    base_action = dict(baseline["action"])

    for stage_name, values in OFFLINE_OPTIMIZATION_STAGES:
        stage_trials = []
        for action in _stage_variants(base_action, stage_name, values):
            key = _action_key(action)
            candidate = trial_cache.get(key)
            if candidate is None:
                candidate = _evaluate_retrieval_action(question, action, gt_index, budget_ranges)
                trial_cache[key] = candidate
                trials.append(candidate)
            stage_trials.append(candidate)
            evaluation = candidate.get("evaluation")
            correctness = evaluation["correctness"] if evaluation else None
            print(f"    {stage_name}={action.get(stage_name, action.get('dense_top_k'))} action={action} correctness={correctness}")
        stage_winner = _select_winner(stage_trials)
        if stage_winner is not None:
            best = stage_winner
            base_action = dict(stage_winner["action"])
    return best, trials, default_correctness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, default="eval_questions.json")
    parser.add_argument("--out", type=str, default="grounding_report.json")
    parser.add_argument("--ground-truth-db", type=str, default="ground_truth.db")
    parser.add_argument("--limit", type=int, default=None, help="Only run N questions, sampled evenly across categories (for a quick smoke test)")
    parser.add_argument("--label", type=str, default="",
                        help="Short label written into the report, e.g. test-run-1")
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if args.limit:
        questions = _stratified_sample(questions, args.limit)

    print(f"Loaded {len(questions)} questions from {args.questions}")

    print("Building vector store / KG (same as agent_app.py startup)...")
    print(f"  Qdrant collection: '{agent_functions.QDRANT_COLLECTION}' "
          f"(must match COLLECTION_NAME in chunk_qdrant_pytorch.py)")
    build_vector_store()

    gt_index = correctness_scoring.GroundTruthIndex(args.ground_truth_db)
    print(f"Loaded {len(gt_index)} ground-truth QA rows across {gt_index.n_combos} combos "
          f"from {args.ground_truth_db}")
    budget_ranges = _resolve_budget_ranges(args.ground_truth_db)

    results = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        category = item.get("category", "unknown")

        print(f"[{i}/{len(questions)}] ({category}) staged offline retrieval optimization  {q[:70]}")

        # First run uses the normal contextual planner. A trustworthy case
        # means reuse-only; otherwise staged evaluation-only exploration starts.
        baseline = _evaluate_retrieval_action(q, None, gt_index, budget_ranges)
        baseline_policy = (baseline.get("state") or {}).get("retrieval_planner_policy", {})
        if baseline_policy.get("policy") == "similarity_weighted_reward":
            winner = baseline
            candidate_runs = [baseline]
            default_correctness = (baseline.get("evaluation") or {}).get("correctness")
            print("    Reusing trusted similar-case action; exploration skipped.")
        else:
            print("    No trusted similar case; starting staged action optimization.")
            winner, candidate_runs, default_correctness = _staged_optimize(q, baseline, gt_index, budget_ranges)
        if winner is None:
            print("    -> no candidate produced a scored recommendation; no case retained")
            results.append({
                "category": category,
                "question": q,
                "error": "no candidate produced a scored recommendation",
                "offline_kg_trials": [
                    {"action": item.get("action", {}), "correctness": (item.get("evaluation") or {}).get("correctness"), "error": item.get("error")}
                    for item in candidate_runs
                ],
            })
            continue

        state = winner["state"]
        eval_result = winner["evaluation"]
        recommended = winner["recommended"]
        requirements = state.get("requirements", {}) or {}
        kg_weight_used = winner["action"].get("kg_weight")
        kg_changed_top = state.get("kg_changed_top", False)
        retrieval = dict(state.get("retrieval_metrics", {}))
        retrieval.update(_retrieval_metrics(state.get("ranked_laptops", []) or [], eval_result, budget_ranges))
        attribution = dict(state.get("retrieval_attribution", {}))

        # Retain only the winner. The other trial actions are report-only.
        retrieval_memory.store_case(state, eval_result["correctness"])
        req_string_for_case = state.get("requirement_string", "") or q
        write_case(
            pipeline="recommend",
            query_text=req_string_for_case,
            requirements=requirements,
            summary={
                "best_overall": recommended.get("name", "") if recommended else "",
                "top_3": [lap.get("name", "") for lap in (state.get("top_k_laptops") or [])[:3]],
                "combo_key": eval_result["combo_key"],
            },
            correctness=eval_result["correctness"],
        )
        improvement = (eval_result["correctness"] - default_correctness) if default_correctness is not None else None
        print(f"    WINNER action={winner['action']} correctness={eval_result['correctness']:.3f} improvement_over_default={improvement}")
        results.append({
            "category": category,
            "question": q,
            "phase": state.get("phase"),
            "kg_weight_used": kg_weight_used,
            "kg_changed_top": kg_changed_top,
            "combo_key": eval_result["combo_key"],
            "correctness": eval_result["correctness"],
            "field_agreement": eval_result["field_agreement"],
            "field_agreement_ceiling": eval_result["field_agreement_ceiling"],
            "price_closeness": eval_result["price_closeness"],
            "recommended_name": recommended.get("name") if recommended else None,
            "gt_laptop_name": eval_result["gt_laptop_name"],
            "last_response": (state.get("last_response") or "")[:500],
            "retrieval_action": state.get("retrieval_action", {}),
            "retrieval_planner_policy": {"policy": "staged_offline_optimization" if len(candidate_runs) > 1 else baseline_policy.get("policy")},
            "retrieval_planner_case_count": 0,
            "retrieval_metrics": retrieval,
            "retrieval_attribution": attribution,
            "offline_kg_trials": [
                {"action": item.get("action", {}), "correctness": (item.get("evaluation") or {}).get("correctness"), "error": item.get("error")}
                for item in candidate_runs
            ],
            "improvement_over_default": improvement,
        })
        continue

        # Fresh state per question so turns don't bleed into each other —
        # each question is evaluated as if it were a new user's first message.
        state = make_initial_state()
        # Warm-up turn (welcome message), matching how the real app initializes.
        state = run_turn(state, "Hello")

        try:
            state = run_turn(state, q)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "category": category,
                "question": q,
                "kg_weight_used": None,
                "error": str(e),
            })
            continue

        requirements = state.get("requirements", {}) or {}
        top_k = state.get("top_k_laptops") or []
        recommended = top_k[0] if top_k else None

        eval_result = None
        if recommended is not None:
            eval_result = correctness_scoring.evaluate_recommendation(
                recommended, requirements, gt_index, budget_ranges,
                classify_fn=agent_functions._classify_one,
            )

        kg_weight_used = state.get("kg_weight_used")
        kg_changed_top = state.get("kg_changed_top", False)
        retrieval = dict(state.get("retrieval_metrics", {}))
        attribution = dict(state.get("retrieval_attribution", {}))
        if eval_result is not None:
            retrieval.update(_retrieval_metrics(
                state.get("ranked_laptops", []) or [], eval_result, budget_ranges,
            ))

        if eval_result is not None:
            # Memento-style Retain: persist the evaluated state-action-result-
            # reward episode. Future queries select an action from similar cases.
            retrieval_memory.store_case(state, eval_result["correctness"])
            print(f"    -> correctness={eval_result['correctness']:.3f}  "
                  f"field_agreement={eval_result['field_agreement']}/{eval_result['field_agreement_ceiling']}  "
                  f"price_closeness={eval_result['price_closeness']:.3f}  "
                  f"kg_changed_top={kg_changed_top}")

            # ── Write a quality-scored case to the case bank ──────────────────
            # compare_node already wrote this case without a correctness score
            # (live app has no ground truth). Now that we have the verified
            # score we write it AGAIN with correctness attached. write_case is
            # quality-gated: cases below CASE_MEMORY_MIN_WRITE_SCORE are
            # silently dropped, so only demonstrably good retrievals enter the
            # bank as positive examples. Over a 200-question run the bank fills
            # up with verified cases; future runs retrieve those first (quality-
            # aware ranking in retrieve_similar_cases outranks the unscored live
            # copies from compare_node). The unscored duplicates are harmless —
            # they rank below scored ones and get outnumbered quickly.
            req_string_for_case = state.get("requirement_string", "") or q
            write_case(
                pipeline="recommend",
                query_text=req_string_for_case,
                requirements=requirements,
                summary={
                    "best_overall": recommended.get("name", "") if recommended else "",
                    "top_3": [l.get("name", "") for l in (state.get("top_k_laptops") or [])[:3]],
                    "combo_key": eval_result["combo_key"],
                },
                correctness=eval_result["correctness"],   # ← quality gate applied inside write_case
            )
        else:
            print("    -> no recommendation / no matching ground-truth combo, skipped scoring")

        results.append({
            "category": category,
            "question": q,
            "phase": state.get("phase"),
            "kg_weight_used": kg_weight_used,
            "kg_changed_top": kg_changed_top,
            "combo_key": eval_result["combo_key"] if eval_result else None,
            "correctness": eval_result["correctness"] if eval_result else None,
            "field_agreement": eval_result["field_agreement"] if eval_result else None,
            "field_agreement_ceiling": eval_result["field_agreement_ceiling"] if eval_result else None,
            "price_closeness": eval_result["price_closeness"] if eval_result else None,
            "recommended_name": recommended.get("name") if recommended else None,
            "gt_laptop_name": eval_result["gt_laptop_name"] if eval_result else None,
            "last_response": (state.get("last_response") or "")[:500],
            "retrieval_action": state.get("retrieval_action", {}),
            "retrieval_planner_policy": state.get("retrieval_planner_policy", {}),
            "retrieval_planner_case_count": len(state.get("retrieval_planner_cases", []) or []),
            "retrieval_metrics": retrieval,
            "retrieval_attribution": attribution,
        })

        time.sleep(0.2)

    # ── Aggregate ──────────────────────────────────────────────────────────
    def _avg(subset, field):
        vals = [r[field] for r in subset if r.get(field) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    scored = [r for r in results if r.get("correctness") is not None]

    summary = {
        "total_questions": len(results),
        "errors": sum(1 for r in results if "error" in r),
        "scored": len(scored),
        "unscored_no_gt_match": sum(1 for r in results if "error" not in r and r.get("correctness") is None),
        "avg_correctness": _avg(results, "correctness"),
        "avg_field_agreement": _avg(results, "field_agreement"),
        "avg_price_closeness": _avg(results, "price_closeness"),
        "kg_changed_top_rate": (
            round(sum(1 for r in scored if r.get("kg_changed_top")) / len(scored), 3) if scored else None
        ),
        "by_category": {},
        "retrieval_policy": "case_based_similarity_weighted_reward",
        "avg_retrieval_latency_ms": round(sum((r.get("retrieval_metrics", {}).get("latency_ms") or 0) for r in results) / max(len(results), 1), 2),
        "avg_recall_at_10": _avg([{"value": r.get("retrieval_metrics", {}).get("recall_at_10")} for r in results], "value"),
        "avg_precision_at_10": _avg([{"value": r.get("retrieval_metrics", {}).get("precision_at_10")} for r in results], "value"),
        "avg_mrr": _avg([{"value": r.get("retrieval_metrics", {}).get("mrr")} for r in results], "value"),
        "avg_ndcg_at_10": _avg([{"value": r.get("retrieval_metrics", {}).get("ndcg_at_10")} for r in results], "value"),
        "source_influence": {source: sum(1 for r in results if r.get("retrieval_attribution", {}).get(source)) for source in ("dense", "sparse", "kg", "memory", "reranker")},
    }

    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        summary["by_category"][cat] = {
            "count": len(cat_results),
            "avg_correctness": _avg(cat_results, "correctness"),
            "avg_field_agreement": _avg(cat_results, "field_agreement"),
            "avg_price_closeness": _avg(cat_results, "price_closeness"),
        }

    report = {
        "run_metadata": {
            "label": args.label or "unlabeled",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "question_count": len(questions),
            "retrieval_policy": "case_based_similarity_weighted_reward",
            "retrieval_configuration": {
                "dense_top_k": 30,
                "sparse_top_k": 30,
                "reranker_model": getattr(agent_functions, "RERANK_MODEL", "disabled"),
                "reranker_top_k": getattr(agent_functions, "RERANK_TOP_K", 10),
                "hyde_temperature": 0.0,
                "ground_truth_budget_cap": 90_000,
                "relevance_definition": "ground-truth requirement combination and budget tier",
            },
        },
        "summary": summary,
        "results": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("GROUNDING EVAL SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nFull report saved to {args.out}")


if __name__ == "__main__":
    main()
