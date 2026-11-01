"""The RAG pipeline: guardrails -> retrieve -> generate -> cite."""
import re
import time
from dataclasses import dataclass, field

from app import generator
from app.ingest import load_chunks
from app.retriever import BM25Retriever

# Below this BM25 score the best match is too weak to trust, so we refuse
# instead of letting the model answer from irrelevant text (hallucination).
MIN_SCORE = 2.0
TOP_K = 3

NOT_FOUND = ("I could not find this in the company policies. "
             "Please contact HR (hr-help@nimbus.example) or the IT helpdesk.")

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[AADHAAR]"),
]


def redact_pii(text: str) -> str:
    """Mask personal data before the question leaves our system (sent to an
    external LLM or written to logs) - a basic responsible-AI control."""
    for pattern, label in PII_PATTERNS:
        text = pattern.sub(label, text)
    return text


@dataclass
class Answer:
    answer: str
    sources: list[str] = field(default_factory=list)
    mode: str = "extractive"          # "llm", "extractive" or "refused"
    top_score: float = 0.0
    latency_ms: float = 0.0


class PolicyAssistant:
    def __init__(self, use_llm: bool | None = None):
        self.chunks = load_chunks()
        self.retriever = BM25Retriever(self.chunks)
        self.use_llm = generator.llm_available() if use_llm is None else use_llm

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        question = redact_pii(question.strip())
        hits = self.retriever.search(question, k=TOP_K)
        top_score = hits[0][1] if hits else 0.0

        if top_score < MIN_SCORE:
            ans = Answer(NOT_FOUND, mode="refused", top_score=top_score)
        else:
            chunks = [c for c, _ in hits]
            sources = sorted({c.source for c in chunks})
            text, mode = None, "extractive"
            if self.use_llm:
                try:
                    text, mode = generator.generate_llm(question, chunks), "llm"
                except Exception as exc:  # network/auth error: degrade gracefully
                    print(f"LLM call failed ({type(exc).__name__}); using extractive answer")
            if not text:
                text = generator.generate_extractive(
                    question, chunks, self.retriever.idf) or NOT_FOUND
            ans = Answer(text, sources, mode, top_score)

        ans.top_score = round(ans.top_score, 3)
        ans.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return ans
