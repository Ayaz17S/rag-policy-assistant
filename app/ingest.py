"""Document ingestion: read markdown files and split them into chunks.

Why chunk? An LLM (and a retriever) works better on small focused passages
than on whole documents: retrieval is more precise, and we send fewer tokens
to the model, which cuts cost and latency.

Strategy: split on "## " section headings (natural topic boundaries). If a
section is very long, split it further into overlapping word windows so that
a sentence cut at a boundary still appears whole in the next chunk.
"""
import re
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
MAX_WORDS = 150
OVERLAP_WORDS = 30


@dataclass
class Chunk:
    chunk_id: str      # e.g. "leave_policy.md#2"
    source: str        # file name, used for citations
    title: str         # "Leave Policy > Sick Leave"
    text: str


def _split_long(text: str) -> list[str]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    step = MAX_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words), step)
            if words[i:i + MAX_WORDS]]


def chunk_markdown(source: str, markdown: str) -> list[Chunk]:
    doc_title = re.search(r"^# (.+)$", markdown, re.MULTILINE)
    doc_title = doc_title.group(1).split(" — ")[0] if doc_title else source
    sections = re.split(r"^## ", markdown, flags=re.MULTILINE)[1:]

    chunks = []
    for section in sections:
        heading, _, body = section.partition("\n")
        for piece in _split_long(body.strip()):
            chunks.append(Chunk(
                chunk_id=f"{source}#{len(chunks)}",
                source=source,
                title=f"{doc_title} > {heading.strip()}",
                text=piece,
            ))
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8")))
    return chunks
