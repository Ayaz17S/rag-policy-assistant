"""Answer generation.

Two modes:
- "llm": send the retrieved chunks + question to Claude, which writes a
  grounded answer with citations. Used when ANTHROPIC_API_KEY is set.
- "extractive": no LLM. Pick the sentences from the retrieved chunks that
  share the most words with the question. Free, offline, fully deterministic,
  and the fallback if the API call fails.
"""
import os
import re

from app.ingest import Chunk
from app.retriever import tokenize

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """You are an HR and IT policy assistant for employees.
Answer ONLY using the policy excerpts provided in the user message.
- Cite the excerpt you used in square brackets, e.g. [leave_policy.md].
- If the excerpts do not contain the answer, say you could not find it in the
  policies and suggest contacting HR or the IT helpdesk. Never guess numbers.
- Keep answers short: 1-4 sentences."""


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[{c.source}] {c.title}\n{c.text}" for c in chunks)
    return f"Policy excerpts:\n{context}\n\nQuestion: {question}"


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def generate_llm(question: str, chunks: list[Chunk]) -> str:
    import anthropic  # imported lazily so offline mode needs no API package

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},  # short factual Q&A needs little reasoning
        # If the main model declines a request, the API retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    if response.stop_reason == "refusal":
        return "I can't help with that request. Please contact HR or the IT helpdesk."
    return "".join(b.text for b in response.content if b.type == "text").strip()


def generate_extractive(question: str, chunks: list[Chunk], idf: dict[str, float],
                        max_sentences: int = 2) -> str:
    q_terms = set(tokenize(question))
    candidates = []
    for rank, chunk in enumerate(chunks):
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            # Weight shared words by IDF so a rare word like "carried" counts
            # more than a common one like "leave".
            overlap = sum(idf.get(t, 0.0) for t in q_terms & set(tokenize(sentence)))
            # Prefer sentences from higher-ranked chunks when overlap ties.
            candidates.append((overlap, -rank, sentence, chunk.source))
    candidates.sort(reverse=True)
    picked = [c for c in candidates[:max_sentences] if c[0] > 0]
    return " ".join(f"{s} [{src}]" for _, _, s, src in picked)
