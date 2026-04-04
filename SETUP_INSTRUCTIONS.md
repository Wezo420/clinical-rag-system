# Clinical Intelligence RAG System — Setup Instructions

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | Required for backend & ML service |
| Node.js | 20+ | Required for frontend |
| Docker | 24+ | Required for containerized setup |
| Docker Compose | 2.x | Included with Docker Desktop |
| Git | any | For cloning |

---

## 1. Clone & Configure Environment

```bash
git clone https://github.com/your-org/clinical-rag.git
cd clinical-rag

# Copy environment template
cp .env.example .env
```

Edit `.env` and set **at minimum**:

```
GROQ_API_KEY=your_real_groq_api_key_here
POSTGRES_PASSWORD=your_secure_password
JWT_SECRET_KEY=a_random_string_at_least_32_characters_long
```

> **Get your Groq API key**: https://console.groq.com

---

## 2. Option A — Docker Compose (Recommended)

This starts all services (backend, frontend, ML service, PostgreSQL, MongoDB, Redis, Nginx):

```bash
docker-compose up --build
```

First run downloads model weights (~400MB). Subsequent starts are fast.

**Services available at:**
| Service | URL |
|---------|-----|
| Frontend (Next.js) | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (dev) | http://localhost:8000/api/docs |
| ML Service | http://localhost:8001 |
| Nginx proxy | http://localhost:80 |

---

## 3. Option B — Local Development Setup

### 3a. Backend

```bash
cd clinical-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Start PostgreSQL and Redis (using Docker)
docker run -d --name postgres-dev \
  -e POSTGRES_DB=clinical_rag \
  -e POSTGRES_USER=clinical_user \
  -e POSTGRES_PASSWORD=devpassword \
  -p 5432:5432 postgres:16-alpine

docker run -d --name redis-dev -p 6379:6379 redis:7-alpine

# Update .env for local services
# DATABASE_URL=postgresql+asyncpg://clinical_user:devpassword@localhost:5432/clinical_rag
# REDIS_URL=redis://localhost:6379/0

# Run database migrations
alembic upgrade head

# Start backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3b. Frontend

```bash
cd frontend
npm install
npm run dev
# Frontend: http://localhost:3000
```

### 3c. ML Service (optional — needed for image embeddings)

```bash
# In a separate terminal, with venv activated
uvicorn ml-service.main:app --host 0.0.0.0 --port 8001
```

---

## 4. Ingest Medical Literature (RAG Data)

The system comes with an empty FAISS index. You need to populate it with medical literature.

### Option 1 — PubMed (real data, requires internet)

```bash
# With venv activated from project root
python -m data.scripts.ingest_pubmed \
  --query "differential diagnosis internal medicine" \
  --max 200 \
  --save

# Ingest multiple topics
python -m data.scripts.ingest_pubmed --query "pneumonia chest X-ray diagnosis" --max 100
python -m data.scripts.ingest_pubmed --query "cardiac arrhythmia ECG" --max 100
python -m data.scripts.ingest_pubmed --query "diabetes mellitus diagnosis" --max 100
```

### Option 2 — From JSONL file

```bash
python -m data.scripts.ingest_pubmed --file ./data/medical_corpus.jsonl
```

Each line in the JSONL should have: `id`, `title`, `text`, `authors`, `journal`, `year`, `pmid`, `url`

### Option 3 — Seed data script (quick start)

```bash
python -m data.scripts.seed_demo_data
```

---

## 5. Run Tests

```bash
# With venv activated
pytest backend/tests/ -v

# With coverage
pytest backend/tests/ --cov=backend --cov-report=html

# Security/red-team tests only
pytest backend/tests/test_security.py -v

# RAG pipeline tests only
pytest backend/tests/test_rag.py -v
```

---

## 6. Using the Application

1. Open **http://localhost:3000**
2. Enter clinical notes in the text area (minimum 10 characters)
3. Optionally upload medical images (JPEG/PNG/TIFF)
4. Optionally expand "Structured Data" and enter age, sex, labs, medications
5. Click **"Analyze Clinical Case"**
6. Review:
   - **Summary** — overall clinical picture
   - **Possible Hypotheses** — ranked conditions with confidence, supporting/against factors, workup
   - **Differential Reasoning** — evidence-grounded chain of reasoning (with citations)
   - **Evidence** — retrieved PubMed sources with relevance scores and links
   - **Safety Flags** — any emergency or urgent warnings

---

## 7. API Usage (Direct)

### Analyze a case

```bash
curl -X POST http://localhost:8000/api/v1/analyze-case \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_text": "65-year-old male, former smoker. 3-week productive cough, fever 38.8C, dyspnea. Right lower lobe dullness on percussion. WBC 14.5.",
    "image_ids": [],
    "structured_data": {
      "age": 65,
      "sex": "male",
      "lab_values": [{"name": "WBC", "value": 14.5, "unit": "10^9/L"}]
    }
  }'
```

### Upload an image

```bash
curl -X POST http://localhost:8000/api/v1/upload-image \
  -F "file=@./chest_xray.jpg" \
  -F "modality=xray"
```

### Get stored result

```bash
curl http://localhost:8000/api/v1/results/{result_id}
```

### Register + Login

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "researcher1", "email": "r@example.com", "password": "SecurePass123!"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "researcher1", "password": "SecurePass123!"}'
```

---

## 8. Production Deployment (AWS/GCP)

### AWS ECS with RDS + ElastiCache

```bash
# 1. Push images to ECR
aws ecr create-repository --repository-name clinical-rag-backend
docker tag clinical-rag-backend:latest <account>.dkr.ecr.<region>.amazonaws.com/clinical-rag-backend:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/clinical-rag-backend:latest

# 2. Use RDS PostgreSQL (update DATABASE_URL in SSM Parameter Store)
# 3. Use ElastiCache Redis (update REDIS_URL)
# 4. Use S3 for storage (set STORAGE_BACKEND=s3, configure AWS_* vars)
# 5. Deploy via ECS Fargate with Application Load Balancer
```

### GCP Cloud Run

```bash
gcloud run deploy clinical-rag-backend \
  --image gcr.io/PROJECT/clinical-rag-backend \
  --platform managed \
  --region us-central1 \
  --set-env-vars GROQ_API_KEY=... \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

### Scaling Considerations

- **Backend**: Scale horizontally behind a load balancer; FAISS index should be on shared NFS or replaced with Pinecone for multi-instance
- **ML Service**: Scale vertically (more RAM/CPU) or use GPU instances; cache model weights in persistent volume
- **PostgreSQL**: Use managed RDS/Cloud SQL with read replicas
- **Redis**: Use managed ElastiCache/Memorystore with cluster mode

---

## 9. Environment Variables Reference

See `.env.example` for full documentation of all variables.

Key variables:
| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key |
| `POSTGRES_PASSWORD` | ✅ Yes | PostgreSQL password |
| `JWT_SECRET_KEY` | ✅ Yes | Min 32-char random string |
| `GROQ_DEFAULT_MODEL` | No | Default: `mixtral-8x7b-32768` |
| `FAISS_INDEX_PATH` | No | Default: `./data/faiss_index` |
| `STORAGE_BACKEND` | No | `local` or `s3` |

---

## 10. Architecture Decision Records

| Decision | Choice | Reason |
|----------|--------|--------|
| LLM Inference | Groq API | Ultra-low latency (<500ms), mixtral-8x7b quality |
| Vector DB | FAISS | Zero-ops, in-process, production-proven |
| Sparse Retrieval | BM25Okapi | Exact term matching complements dense retrieval |
| Fusion | RRF | Parameter-free, robust, outperforms score normalization |
| Re-ranking | cross-encoder/ms-marco | Strong cross-encoder quality, ~80ms on CPU |
| Image Encoding | OpenAI CLIP ViT-B/32 | Open-source, 400M param, strong zero-shot medical |
| Backend | FastAPI + asyncio | Async I/O critical for RAG + LLM latency |
| Auth | JWT (HS256) | Stateless, horizontally scalable |
| ORM | SQLAlchemy async | Type-safe, async-native, alembic migrations |

---

## Troubleshooting

**"FAISS index not found"** — Normal on first run. Ingest data with `ingest_pubmed.py`.

**"Groq API error"** — Check `GROQ_API_KEY` in `.env`. Verify at https://console.groq.com.

**"Cannot connect to PostgreSQL"** — Ensure postgres container is healthy: `docker-compose ps`

**"Model download stuck"** — First run downloads ~400MB of model weights. Use a stable connection or pre-pull with `docker-compose pull`.

**Tests fail with import errors** — Ensure you're in the project root and venv is activated.
