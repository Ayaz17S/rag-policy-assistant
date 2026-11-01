FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY docs/ docs/
EXPOSE 8000
# Pass ANTHROPIC_API_KEY at runtime (docker run -e ANTHROPIC_API_KEY=...) to enable LLM mode.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
