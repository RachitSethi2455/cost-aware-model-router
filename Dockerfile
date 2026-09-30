FROM python:3.11-slim

WORKDIR /app

# OPENBLAS_NUM_THREADS=1: the router is a 7-feature logistic regression;
# per-thread BLAS buffers only waste memory on small hosting plans.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    OPENBLAS_NUM_THREADS=1 \
    PORT=8000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY evals/ ./evals/

# The trained router is a build artifact (data/ is gitignored), so train it
# here. Offline and deterministic: a fresh clone builds a working image.
RUN python evals/train_router.py

# Non-root; Hugging Face Spaces runs containers as UID 1000.
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
# Hosting platforms set $PORT; add -e PUBLIC_DEMO=1 for a public deployment.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --app-dir src"]
