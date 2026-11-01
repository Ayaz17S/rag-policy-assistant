"""Streamlit chat UI for the policy assistant.

Run locally:  streamlit run streamlit_app.py
On Streamlit Cloud, add ANTHROPIC_API_KEY under the app's Settings > Secrets to turn on LLM mode;
without it the app runs in free offline (extractive) mode.
"""
import os

import streamlit as st

st.set_page_config(page_title="Policy RAG Assistant", page_icon="📚", layout="wide")

# Streamlit Cloud stores keys in st.secrets; the generator reads the environment variable.
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except FileNotFoundError:  # no secrets file when running locally
    pass

from app.assistant import MIN_SCORE, PolicyAssistant  # noqa: E402
from app.ingest import DOCS_DIR  # noqa: E402

EXAMPLES = [
    "How many sick leave days do I get?",
    "My laptop was stolen, what should I do?",
    "What is the hotel limit in metro cities?",
    "Can I work remotely from outside India?",
    "Who won the cricket world cup in 2011?",
]


@st.cache_resource
def get_assistant() -> PolicyAssistant:
    return PolicyAssistant()


bot = get_assistant()

with st.sidebar:
    st.header("📚 Knowledge base")
    st.write("Policies of **Nimbus Telecom** (a fictional company):")
    for path in sorted(DOCS_DIR.glob("*.md")):
        with st.expander(path.stem.replace("_", " ").title()):
            st.markdown(path.read_text(encoding="utf-8"))
    st.divider()
    mode = "🤖 LLM (Claude)" if bot.use_llm else "📄 Offline extractive"
    st.write(f"**Answer mode:** {mode}")
    st.write(f"**Chunks indexed:** {len(bot.chunks)}")
    st.write(f"**Refusal threshold:** BM25 score < {MIN_SCORE}")

st.title("📚 HR & IT Policy Assistant (RAG)")
st.write("Ask about leave, IT security, travel, hybrid work or onboarding. The assistant **searches "
         "the policy documents first** and answers only from what it finds, with the source cited. "
         "Off-topic questions are refused instead of guessed, and personal data (email, phone, PAN, "
         "Aadhaar) is masked before processing.")

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
clicked = None
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        clicked = example

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Type your question...") or clicked
if question:
    st.session_state.history.append((question, bot.ask(question)))

for q, ans in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        if ans.mode == "refused":
            st.warning(ans.answer)
        else:
            st.write(ans.answer)
        details = f"mode: {ans.mode} · best match score: {ans.top_score} · {ans.latency_ms} ms"
        if ans.sources:
            details = "sources: " + ", ".join(ans.sources) + " · " + details
        st.caption(details)

st.divider()
st.caption("Retrieval: BM25 written from scratch · Generation: Claude (when an API key is set) · "
           "[Source code](https://github.com/Ayaz17S/rag-policy-assistant)")
