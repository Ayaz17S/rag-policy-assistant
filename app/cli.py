"""Interactive command-line chat.  Run:  python -m app.cli"""
from app.assistant import PolicyAssistant


def main() -> None:
    bot = PolicyAssistant()
    mode = "llm" if bot.use_llm else "extractive"
    print(f"Policy assistant ready ({len(bot.chunks)} chunks, mode={mode}). "
          "Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        a = bot.ask(q)
        print(f"Bot: {a.answer}")
        print(f"     sources={a.sources} score={a.top_score} {a.latency_ms} ms\n")


if __name__ == "__main__":
    main()
