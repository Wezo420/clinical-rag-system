# ─────────────────────────────────────────────────────────────
# ML Service Dockerfile — CLIP + SentenceTransformers
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS deps
COPY ml-service/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('SentenceTransformer cached')"

FROM deps AS runtime
COPY ml-service/ ./ml-service/

RUN adduser --disabled-password --gecos "" mluser && \
    chown -R mluser:mluser /app
USER mluser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "ml-service.main:app", \
     "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
