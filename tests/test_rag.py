from fastapi.testclient import TestClient

from app.assistant import PolicyAssistant, redact_pii
from app.ingest import chunk_markdown
from app.retriever import BM25Retriever, tokenize

bot = PolicyAssistant(use_llm=False)


def test_chunking_splits_on_sections_and_long_text():
    md = "# Doc\n\n## A\nshort text\n\n## B\n" + "word " * 400
    chunks = chunk_markdown("x.md", md)
    assert chunks[0].title == "Doc > A"
    assert len(chunks) > 2  # section B was split into overlapping windows


def test_tokenize_removes_stopwords_and_plurals():
    assert tokenize("What are the Passwords?") == ["password"]


def test_rare_terms_rank_higher():
    md = "# D\n## One\nleave leave leave\n## Two\nleave paternity\n"
    top_chunk, _ = BM25Retriever(chunk_markdown("x.md", md)).search("paternity leave", k=1)[0]
    assert top_chunk.title == "D > Two"


def test_answers_with_correct_source():
    ans = bot.ask("How many days of sick leave do I get?")
    assert ans.mode == "extractive"
    assert "leave_policy.md" in ans.sources
    assert "12 days" in ans.answer


def test_refuses_off_topic_question():
    assert bot.ask("Who won the cricket world cup in 2011?").mode == "refused"


def test_pii_is_redacted():
    text = redact_pii("I am ravi@gmail.com, phone 9876543210, PAN ABCDE1234F")
    assert "ravi@gmail.com" not in text and "9876543210" not in text
    assert "[EMAIL]" in text and "[PHONE]" in text and "[PAN]" in text


def test_api_ask():
    from app.api import app
    client = TestClient(app)
    body = client.post("/ask", json={"question": "What is the hotel limit in metro cities?"}).json()
    assert "6,000" in body["answer"]
    assert client.post("/ask", json={"question": ""}).status_code == 422
