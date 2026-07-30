"""Contextual retrieval policy backed by Memento-style Qdrant case memory.

This module deliberately has no SQLite database, global arm statistics, or
epsilon-greedy policy. A retrieval decision is conditioned on the current
requirement state and on evaluated cases that are semantically similar to it.

Each retained case is:
    state (requirement profile + embedding), action (retrieval parameters),
    result (retrieved IDs + answer), reward (correctness), timestamp.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple


MIN_CASE_SIMILARITY = 0.65
MIN_CASE_CORRECTNESS = 0.70


def _planner_module():
    """Late import avoids a module-import cycle with agent_functions."""
    import agent_functions
    return agent_functions


def retrieve_similar_cases(query_text: str, requirements: dict) -> List[dict]:
    """Read only trustworthy, analogous evaluated retrieval episodes."""
    af = _planner_module()
    cases = af.retrieve_similar_cases(
        query_text,
        top_k=af.CASE_MEMORY_TOP_K,
        pipeline=af.RETRIEVAL_PLANNER_PIPELINE,
        requirements=requirements,
    )
    return [
        case for case in cases
        if float(case.get("score", 0.0) or 0.0) >= MIN_CASE_SIMILARITY
        and float(case.get("correctness", 0.0) or 0.0) >= MIN_CASE_CORRECTNESS
    ]


def estimate_action_rewards(cases: List[dict]) -> List[dict]:
    """Score candidate actions by similarity-weighted correctness.

    For every candidate action, expected reward is:
        sum(case_similarity * case_correctness) / sum(case_similarity)
    """
    af = _planner_module()
    grouped: Dict[str, dict] = {}
    for case in cases:
        action = case.get("action") or {}
        reward = case.get("correctness")
        similarity = float(case.get("score", 0.0) or 0.0)
        if not action or reward is None or similarity <= 0:
            continue
        action = af._normalise_retrieval_action(action)
        key = json.dumps(action, sort_keys=True, separators=(",", ":"))
        bucket = grouped.setdefault(key, {"action": action, "numerator": 0.0, "denominator": 0.0, "case_count": 0})
        bucket["numerator"] += similarity * float(reward)
        bucket["denominator"] += similarity
        bucket["case_count"] += 1
    candidates = [
        {
            "action": bucket["action"],
            "weighted_reward": round(bucket["numerator"] / bucket["denominator"], 6),
            "case_count": bucket["case_count"],
            "similarity_mass": round(bucket["denominator"], 6),
        }
        for bucket in grouped.values() if bucket["denominator"] > 0
    ]
    return sorted(candidates, key=lambda item: (item["weighted_reward"], item["similarity_mass"]), reverse=True)


def select_action(query_text: str, requirements: dict) -> Tuple[dict, dict, List[dict]]:
    """Choose an action for this state, not for the entire query population.

    Returns (action, policy_audit, similar_cases). When no similar evaluated
    case exists, action is the safe default retrieval configuration.
    """
    af = _planner_module()
    cases = retrieve_similar_cases(query_text, requirements)
    candidates = estimate_action_rewards(cases)
    if candidates:
        action = candidates[0]["action"]
        policy = {
            "policy": "similarity_weighted_reward",
            "selected_weighted_reward": candidates[0]["weighted_reward"],
            "candidate_actions": candidates,
        }
    else:
        action = dict(af.DEFAULT_RETRIEVAL_ACTION)
        policy = {"policy": "default_no_similar_evaluated_cases", "candidate_actions": []}
    return action, policy, cases


def store_case(state: dict, reward: float) -> None:
    """Retain a fully evaluated state-action-result-reward retrieval episode."""
    af = _planner_module()
    af.store_retrieval_case(state, reward)


def default_action() -> Dict:
    """Expose the fallback configuration without mutable global state."""
    return dict(_planner_module().DEFAULT_RETRIEVAL_ACTION)


def policy_summary(query_text: str, requirements: dict) -> dict:
    """Return an auditable per-query contextual-policy decision."""
    action, policy, cases = select_action(query_text, requirements)
    return {
        "policy": policy.get("policy"),
        "selected_action": action,
        "similar_case_count": len(cases),
        "candidate_actions": policy.get("candidate_actions", []),
    }
