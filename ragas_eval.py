"""
ragas_eval.py - Standalone offline RAGAS evaluation for the Laptop Shopping Assistant
=========================================================================================

Run this manually whenever you want RAGAS scores. It is completely decoupled
from the live app: agent_app.py and agent_functions.py never import `ragas`
or `datasets`, and nothing here is gated behind an env var like the old
RAGAS_EVAL flag — every run scores every turn currently in the log.

Usage:
    python ragas_eval.py
    python ragas_eval.py --log rag_eval_log.jsonl --out ragas_results.json

Input: a JSONL log of completed RAG turns, one JSON object per line:
    {"pipeline": "...", "question": "...", "contexts": [...], "answer": "...", "timestamp": "..."}

agent_functions.py appends to this log after every turn via its
`_log_rag_turn()` helper (pure stdlib json + file append) — that's the ONLY
thing the live app does towards RAGAS; the actual RAGAS/DeepSeek-R1 code all
lives here, in this standalone script.

Judge model: DeepSeek-R1 (via Ollama). RAGAS's faithfulness metric needs
multi-step, strictly-formatted output (statement extraction + NLI-style
verdicts) that a non-reasoning model reliably fails (faithfulness=nan).
DeepSeek-R1's reasoning pass handles that far more reliably. This is an
offline batch job, so the extra latency/VRAM cost is fine here.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

DEFAULT_LOG_PATH = "rag_eval_log.jsonl"
DEFAULT_OUT_PATH = "ragas_results.json"

JUDGE_MODEL = "deepseek-r1:7b"
EMBEDDING_MODEL = "nomic-embed-text"
CTX_WINDOW = 4096


def _stub_vertexai() -> None:
    """
    ragas pulls in langchain_community's VertexAI integration at import
    time even when it's never used. Stub it out so a missing/broken
    google-cloud-aiplatform install doesn't block importing ragas for a
    purely-Ollama setup.
    """
    if "langchain_community.chat_models.vertexai" in sys.modules:
        return
    stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAIStub:
        def __init__(self, *a, **k):
            raise RuntimeError("ChatVertexAI stub — Vertex AI is not installed/used by this app")

    stub.ChatVertexAI = _ChatVertexAIStub
    sys.modules["langchain_community.chat_models.vertexai"] = stub


def load_turns(log_path: str) -> list[dict]:
    """Read every line of the JSONL turn log. Skips malformed lines with a warning."""
    path = Path(log_path)
    if not path.exists():
        print(f"❌ No log file found at {log_path}. Nothing to evaluate.")
        print("   (agent_functions.py's _log_rag_turn() writes to this file as the live app runs —")
        print("    run the app and complete a few turns first.)")
        return []

    turns = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed line {i}: {e}")
    return turns


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RAGAS evaluation for logged RAG turns")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to the JSONL turn log")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Where to write the results JSON")
    args = parser.parse_args()

    turns = load_turns(args.log)
    if not turns:
        return
    print(f"📄 Loaded {len(turns)} turn(s) from {args.log}")

    _stub_vertexai()

    # All RAGAS/DeepSeek imports are local to this function — they only
    # happen when you actually run this script, never on import of any
    # other module in the app.
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from datasets import Dataset

    judge = OllamaLLM(
        model=JUDGE_MODEL,
        temperature=0.0,
        num_predict=2500,  # headroom: reasoning + strict JSON needs more than a single-shot answer
        num_ctx=CTX_WINDOW,
        system="You are a helpful assistant. Respond with valid JSON.",
        format="json",
        keep_alive="15m",
    )
    llm_wrapper = LangchainLLMWrapper(judge)
    embeddings_wrapper = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBEDDING_MODEL))
    run_config = RunConfig(timeout=900, max_workers=1, max_retries=4)

    results = []
    for i, turn in enumerate(turns, 1):
        pipeline = turn.get("pipeline", "unknown")
        question = turn.get("question", "")
        contexts = [c for c in turn.get("contexts", []) if c and c.strip()]
        answer = turn.get("answer", "")

        if not contexts:
            print(f"  [{i}/{len(turns)}] ⏭️  skipped ({pipeline}) — no non-empty contexts")
            continue
        if not answer.strip():
            print(f"  [{i}/{len(turns)}] ⏭️  skipped ({pipeline}) — empty answer")
            continue

        try:
            dataset = Dataset.from_dict({
                "question": [question],
                "contexts": [contexts],
                "answer": [answer],
            })
            result = ragas_evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=llm_wrapper,
                embeddings=embeddings_wrapper,
                run_config=run_config,
            )
            row = result.to_pandas().iloc[0]
            scores = {
                "faithfulness": round(float(row["faithfulness"]), 3),
                "answer_relevancy": round(float(row["answer_relevancy"]), 3),
            }
            print(f"  [{i}/{len(turns)}] ✅ ({pipeline}) {scores}")
            results.append({
                "pipeline": pipeline,
                "question": question,
                "scores": scores,
                "timestamp": turn.get("timestamp"),
            })
        except Exception as e:
            print(f"  [{i}/{len(turns)}] ⚠️  ({pipeline}) evaluation failed: {e}")

    if not results:
        print("\nNo turns were successfully scored.")
        return

    avg = {}
    for metric in ("faithfulness", "answer_relevancy"):
        vals = [r["scores"][metric] for r in results if metric in r["scores"]]
        if vals:
            avg[metric] = round(sum(vals) / len(vals), 3)

    output = {"turns": results, "average": avg, "count": len(results)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n📊 Average scores across {len(results)} scored turn(s): {avg}")
    print(f"✅ Full results written to {args.out}")


if __name__ == "__main__":
    main()
