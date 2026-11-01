"""REST API for the policy assistant.

Run:  uvicorn app.api:app --reload    ->  http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.assistant import Answer, PolicyAssistant

app = FastAPI(title="Policy RAG Assistant", version="1.0.0")
assistant = PolicyAssistant()  # index is built once at startup


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(assistant.chunks),
            "mode": "llm" if assistant.use_llm else "extractive"}


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    return assistant.ask(q.question)
