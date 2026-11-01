# RAG Policy Assistant (Generative AI)

A Retrieval-Augmented Generation (RAG) assistant that answers employee questions about
HR and IT policies, with citations, an "I don't know" guardrail and PII redaction.
Works fully offline, and uses Claude as the LLM when an API key is set.

**Stack:** Python, NumPy (BM25 written from scratch), Anthropic Claude API, FastAPI, pytest, Docker

## Problem
Employees ask the helpdesk the same questions again and again ("how many sick days do I
get?"). The answers are in policy documents nobody reads. An LLM alone would make up
answers (hallucinate); RAG makes it answer *only* from the company's own documents.

The `docs/` folder holds five policy documents of a fictional company (leave, IT
security, travel, hybrid work, onboarding).

## How RAG works here
```
question -> redact PII -> BM25 retriever -> top 3 chunks -> score < 2.0 ? -> "I could not find this"
                                                 |
                                                 v
                      LLM mode: Claude answers using ONLY these chunks, cites [file]
                      Offline mode: pick the best-matching sentences (extractive)
```
1. **Ingest** (`ingest.py`): split documents on section headings into 27 chunks. Long
   sections would be split into 150-word windows with 30-word overlap.
2. **Retrieve** (`retriever.py`): BM25 ranking implemented from scratch with NumPy
   (term frequency with saturation, inverse document frequency, length normalisation).
3. **Guardrail**: if the best BM25 score is below 2.0 the question is off-topic, so the
   bot refuses instead of guessing.
4. **Generate** (`generator.py`): Claude gets a strict system prompt ("answer only from
   these excerpts, cite the file, never guess numbers"). If the API call fails, it falls
   back to the extractive answer, so the service never goes down because of the LLM.
5. **PII redaction**: emails, phone numbers, PAN and Aadhaar numbers are masked before
   the question is sent to an external LLM.

## Evaluation (`python -m eval.evaluate`)
26 labelled questions plus 3 off-topic questions:

| Metric | Value |
|---|---|
| Hit@1 (correct document ranked first) | 1.00 |
| Hit@3 | 1.00 |
| MRR | 1.00 |
| Off-topic refusal rate | 1.00 |

Honest caveat: the corpus is small (27 chunks), so retrieval is easy. Lowest on-topic
score was 3.3 and off-topic scores were 0, which is how the 2.0 threshold was chosen.
The offline extractive answers are sometimes clumsy, which is exactly the gap the LLM
step closes.

## Run it
```bash
pip install -r requirements.txt
python -m pytest -q              # 7 tests
python -m eval.evaluate          # retrieval metrics
python -m app.cli                # chat in the terminal
uvicorn app.api:app --reload     # http://127.0.0.1:8000/docs
```
LLM mode: set `ANTHROPIC_API_KEY` before starting (PowerShell: `$env:ANTHROPIC_API_KEY="..."`).
The model is `claude-opus-5-5` by default and can be changed with `CLAUDE_MODEL`. The request
enables the API's server-side fallback, so if the main model declines a request another model
answers instead.

## Why BM25 and not embeddings?
BM25 needs no model download or GPU, is fast and easy to explain, and works very well
for policy text where users type the same keywords as the document ("sick leave").
Its weakness is synonyms ("pen drive" vs "USB storage"). Next step: hybrid search, i.e.
BM25 + sentence-embedding similarity (for example with `sentence-transformers` and a vector
database such as FAISS or Chroma), then merge the two rankings.

## Next steps
- Hybrid retrieval (keyword + embeddings) and a re-ranker.
- Measure answer quality, not just retrieval (faithfulness: is every claim in the sources?).
- Conversation memory for follow-up questions; a simple web UI with Streamlit.
