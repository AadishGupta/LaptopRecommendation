"""
llm_judge.py - LLM-as-a-judge evaluation for the Laptop Shopping Assistant
=============================================================================

Scores a completed RAG turn on four GraphRAG-style global-quality criteria —
comprehensiveness, diversity, empowerment, directness — via a single LLM
call. This is the only automated RAG-quality scoring in the app; RAGAS has
been removed entirely (no `ragas`/`datasets` dependency, no DeepSeek-R1
judge model, nothing gated behind RAGAS_EVAL).

`llm_judge_evaluate()` is called from agent_functions.py's `_record_judge`,
passing in this app's own LLM (LLMFactory.get_llm(...)) — this module has
no opinion on which model does the judging.
"""

from __future__ import annotations

import json
import logging
from typing import List

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are grading a retrieval-augmented answer against the context it was built from.
Score the ANSWER on these four criteria, each from 1 (poor) to 10 (excellent):

- comprehensiveness: does it cover the relevant facts present in the context?
- diversity: does it draw on a variety of the distinct facts/entities in the context, not just one?
- empowerment: does it give the reader enough to make their own informed decision (reasoning, not just a verdict)?
- directness: does it answer the QUESTION specifically, without padding or drifting off-topic?

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object, no other text:
{{"comprehensiveness": <int>, "diversity": <int>, "empowerment": <int>, "directness": <int>, "rationale": "<one sentence>"}}
"""


def llm_judge_evaluate(question: str, context: List[str] | str, answer: str,
                        llm=None) -> dict:
    """
    Automated LLM-as-a-judge evaluation, complementary to the crude
    lexical `evaluate_kg_rag` metrics in kg_rag.py. Scores the four
    GraphRAG global-quality criteria — comprehensiveness, diversity,
    empowerment, directness — each 1-10, via a single LLM call.

    `llm` should be a LangChain chat model exposing `.invoke(str) -> AIMessage`
    (e.g. `LLMFactory.get_llm(...)` from agent_functions.py). It's a required
    parameter rather than something this module constructs itself, so this
    file stays free of any dependency on a specific judge model or on
    agent_functions.py (avoids a circular import, since agent_functions.py
    is what calls in here).

    Fails soft: returns {} (and logs a warning) on any error, so a flaky
    judge call never breaks the user-facing turn.
    """
    ctx_text = "\n".join(context) if isinstance(context, list) else str(context)
    logger.info(f"⚖️  [LLM JUDGE] scoring answer for question: '{question[:80]}'")

    if llm is None:
        logger.warning("   llm_judge_evaluate: no judge LLM provided")
        return {}

    prompt = _JUDGE_PROMPT.format(question=question, context=ctx_text, answer=answer)
    try:
        raw = llm.invoke(prompt)
        raw_text = raw.content if hasattr(raw, "content") else str(raw)
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").lstrip("json").strip()
        parsed = json.loads(raw_text)
        scores = {
            "comprehensiveness": int(parsed.get("comprehensiveness", 0)),
            "diversity":          int(parsed.get("diversity", 0)),
            "empowerment":        int(parsed.get("empowerment", 0)),
            "directness":         int(parsed.get("directness", 0)),
            "rationale":          parsed.get("rationale", ""),
        }
        logger.info(f"⚖️  [LLM JUDGE] comprehensiveness={scores['comprehensiveness']} "
                    f"diversity={scores['diversity']} empowerment={scores['empowerment']} "
                    f"directness={scores['directness']}")
        return scores
    except Exception as e:
        logger.warning(f"   llm_judge_evaluate failed: {e}")
        return {}
