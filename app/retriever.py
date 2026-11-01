"""BM25 retriever, implemented from scratch with NumPy.

BM25 is the ranking function behind search engines like Elasticsearch.
For a query q and chunk d:

    score(d, q) = sum over terms t in q of
        IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

- tf(t,d): how often the term appears in the chunk (more = more relevant,
  but with diminishing returns controlled by k1)
- IDF: rare words ("paternity") matter more than common ones ("leave")
- b: penalises long chunks so they don't win just by containing more words
"""
import math
import re
from collections import Counter

import numpy as np

from app.ingest import Chunk

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "and", "or", "in",
    "on", "for", "with", "at", "by", "from", "it", "its", "this", "that", "i",
    "my", "me", "we", "our", "you", "your", "do", "does", "can", "how", "what",
    "when", "which", "who", "will", "should", "if", "as", "any", "there", "per",
    "much", "many", "get", "need", "have", "has",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # Light stemming: "passwords" -> "password", "devices" -> "device".
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
            for t in tokens if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Index the heading too: "Sick Leave" in a title is a strong signal.
        self.doc_tokens = [tokenize(f"{c.title} {c.text}") for c in chunks]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=float)
        self.avgdl = self.doc_len.mean()
        self.tf = [Counter(t) for t in self.doc_tokens]

        df = Counter(term for tokens in self.doc_tokens for term in set(tokens))
        n = len(chunks)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.chunks))
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in set(tokenize(query)):
            if term not in self.idf:
                continue  # word never appears in the corpus
            tf = np.array([c[term] for c in self.tf], dtype=float)
            result += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return result

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in top if s[i] > 0]
