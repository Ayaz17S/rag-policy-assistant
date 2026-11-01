"""Measure retrieval quality on a labelled question set.

Run:  python -m eval.evaluate

Metrics:
- Hit@k: share of questions where the correct document is in the top k chunks.
- MRR (Mean Reciprocal Rank): average of 1/rank of the first correct chunk;
  1.0 means the right document is always ranked first.
- Off-topic refusal rate: unrelated questions must be refused, not answered.
"""
import json
from pathlib import Path

from app.assistant import PolicyAssistant

QUESTIONS = Path(__file__).with_name("questions.json")
OFF_TOPIC = [
    "Who won the cricket world cup in 2011?",
    "What is the capital of France?",
    "Write me a poem about the ocean",
]


def evaluate(k: int = 3) -> dict:
    bot = PolicyAssistant(use_llm=False)
    data = json.loads(QUESTIONS.read_text())
    hit1 = hitk = rr = 0.0
    misses = []
    for item in data:
        sources = [c.source for c, _ in bot.retriever.search(item["question"], k=k)]
        rank = sources.index(item["source"]) + 1 if item["source"] in sources else None
        hit1 += rank == 1
        hitk += rank is not None
        rr += 1 / rank if rank else 0
        if rank != 1:
            misses.append((item["question"], sources))

    refused = sum(bot.ask(q).mode == "refused" for q in OFF_TOPIC)
    n = len(data)
    result = {"questions": n, "hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n,
              "off_topic_refusal_rate": refused / len(OFF_TOPIC)}
    for key, val in result.items():
        print(f"{key:24s} {val:.3f}" if isinstance(val, float) else f"{key:24s} {val}")
    for q, s in misses:
        print(f"  not ranked first: {q!r} -> {s}")
    return result


if __name__ == "__main__":
    evaluate()
