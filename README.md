# Clinical Intelligence & Multimodal Medical Diagnostics (RAG)

A production-grade, end-to-end clinical intelligence system powered by Retrieval-Augmented Generation, multimodal AI (CLIP/MedCLIP), and Groq LLM inference.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (Next.js)                         │
│   Input Form (text + image) │ Dashboard │ Results UI                │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ HTTPS/REST
┌──────────────────▼──────────────────────────────────────────────────┐
│                       BACKEND (FastAPI)                             │
│  /analyze-case │ /upload-image │ /results/{id}                      │
│  Auth (JWT) │ Rate Limiting │ Input Validation                      │
└──────┬────────────────┬──────────────────┬──────────────────────────┘
       │                │                  │
┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼───────────────────┐
│  RAG Service│  │  ML Service │  │       Groq LLM API        │
│  FAISS+BM25 │  │  CLIP/Embed │  │       mixtral-8x7b        │
│  Re-ranking │  │  Multimodal │  │  llama3.3-70b-versatile   │
└──────┬──────┘  └──────┬──────┘  └───────────────────────────┘
       │                │
┌──────▼────────────────▼──────────────────────────┐
│              Data Layer                          │
│  PostgreSQL │ MongoDB │ FAISS Index │ S3/Local   │
│  PubMed │ MIMIC Data │ Medical Images            │
└──────────────────────────────────────────────────┘
```

## Services

| Service     | Port  | Description                            |
|-------------|-------|----------------------------------------|
| Frontend    | 3000  | Next.js React application              |
| Backend     | 8000  | FastAPI REST API                       |
| ML Service  | 8001  | CLIP embeddings + multimodal pipeline  |
| PostgreSQL  | 5432  | Structured data, users, results        |
| MongoDB     | 27017 | Document store, medical literature     |
| Redis       | 6379  | Caching, rate limiting                 |

## Quick Start

See `SETUP_INSTRUCTIONS.md` for full setup guide.

```bash
cp .env.example .env
# Add your GROQ_API_KEY
docker-compose up --build
```
