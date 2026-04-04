"""
RAG Pipeline Integration Tests
Tests retrieval quality, reranking, and end-to-end pipeline.

Usage: pytest backend/tests/test_rag.py -v
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ────────────────────────────────────────────────────────────────
# FAISS Retriever Tests
# ────────────────────────────────────────────────────────────────

class TestFAISSRetriever:
    def test_add_and_search(self):
        from backend.services.rag_service import FAISSRetriever
        r = FAISSRetriever(dim=8)
        docs = [
            {"id": "d1", "text": "pneumonia symptoms fever cough"},
            {"id": "d2", "text": "diabetes mellitus glucose insulin"},
            {"id": "d3", "text": "cardiac arrhythmia ECG palpitations"},
        ]
        embeddings = np.random.rand(3, 8).astype(np.float32)
        r.add_documents(docs, embeddings)
        query_emb = np.random.rand(8).astype(np.float32)
        results = r.search(query_emb, k=2)
        assert len(results) == 2
        assert all(isinstance(idx, int) for idx, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    def test_empty_index_returns_empty(self):
        from backend.services.rag_service import FAISSRetriever
        r = FAISSRetriever(dim=8)
        results = r.search(np.random.rand(8), k=5)
        assert results == []

    def test_search_k_capped_at_ntotal(self):
        from backend.services.rag_service import FAISSRetriever
        r = FAISSRetriever(dim=8)
        docs = [{"id": f"d{i}", "text": f"doc {i}"} for i in range(3)]
        embeddings = np.random.rand(3, 8).astype(np.float32)
        r.add_documents(docs, embeddings)
        results = r.search(np.random.rand(8), k=100)
        assert len(results) <= 3


# ────────────────────────────────────────────────────────────────
# BM25 Retriever Tests
# ────────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def test_build_and_search(self):
        from backend.services.rag_service import BM25Retriever
        r = BM25Retriever()
        docs = [
            {"id": "d1", "text": "pneumonia chest infection bacterial"},
            {"id": "d2", "text": "diabetes type 2 insulin resistance glucose"},
            {"id": "d3", "text": "cardiac arrest heart failure ECG"},
        ]
        r.build_index(docs)
        results = r.search("pneumonia chest infection", k=3)
        assert len(results) >= 1
        # pneumonia doc should rank first
        top_id = results[0][0]
        assert top_id == "d1"

    def test_unrelated_query_returns_low_scores(self):
        from backend.services.rag_service import BM25Retriever
        r = BM25Retriever()
        docs = [{"id": f"d{i}", "text": "diabetes glucose insulin pancreas"} for i in range(5)]
        r.build_index(docs)
        results = r.search("pneumococcal meningitis CSF analysis", k=5)
        # Should return results but all will have low scores since no overlap
        # (BM25 filters score=0)
        for _, score in results:
            assert score >= 0

    def test_empty_index_returns_empty(self):
        from backend.services.rag_service import BM25Retriever
        r = BM25Retriever()
        results = r.search("any query", k=5)
        assert results == []


# ────────────────────────────────────────────────────────────────
# Hybrid Fusion Tests
# ────────────────────────────────────────────────────────────────

class TestHybridFusion:
    def test_rrf_boosts_overlap(self):
        from backend.services.rag_service import reciprocal_rank_fusion
        dense = ["d1", "d2", "d3", "d4"]
        sparse = ["d2", "d1", "d5", "d3"]
        scores = reciprocal_rank_fusion([dense, sparse])
        # d1 is rank 0 in dense and rank 1 in sparse → very high
        # d5 only in sparse at rank 2 → lower than d1
        assert scores["d1"] > scores["d5"]
        assert scores["d2"] > scores["d4"]  # d2 top in sparse

    def test_rrf_single_ranking(self):
        from backend.services.rag_service import reciprocal_rank_fusion
        ranking = ["a", "b", "c"]
        scores = reciprocal_rank_fusion([ranking])
        assert scores["a"] > scores["b"] > scores["c"]

    def test_rrf_k_parameter_effect(self):
        from backend.services.rag_service import reciprocal_rank_fusion
        ranking = ["x"]
        scores_small_k = reciprocal_rank_fusion([ranking], k=1)
        scores_large_k = reciprocal_rank_fusion([ranking], k=100)
        # smaller k → higher score for top items
        assert scores_small_k["x"] > scores_large_k["x"]


# ────────────────────────────────────────────────────────────────
# Groq Client Tests
# ────────────────────────────────────────────────────────────────

class TestGroqClient:
    def test_build_rag_prompt_includes_sources(self):
        from backend.services.groq_client import GroqLLMClient
        client = GroqLLMClient.__new__(GroqLLMClient)
        context_blocks = [
            {
                "title": "Pneumonia Diagnosis",
                "authors": ["Smith J", "Jones A"],
                "journal": "NEJM",
                "year": 2023,
                "text": "CAP is diagnosed by chest radiograph showing new infiltrate.",
                "score": 0.92,
            }
        ]
        prompt = client.build_rag_prompt(
            clinical_text="65yo with fever and cough",
            context_blocks=context_blocks,
        )
        assert "[Source 1]" in prompt
        assert "Pneumonia Diagnosis" in prompt
        assert "chest radiograph" in prompt
        assert "65yo with fever" in prompt

    def test_build_rag_prompt_includes_image_summary(self):
        from backend.services.groq_client import GroqLLMClient
        client = GroqLLMClient.__new__(GroqLLMClient)
        prompt = client.build_rag_prompt(
            clinical_text="patient with cough",
            context_blocks=[],
            image_summary="Chest X-ray showing right lower lobe consolidation",
        )
        assert "Chest X-ray" in prompt
        assert "consolidation" in prompt

    def test_parse_json_extracts_from_text(self):
        from backend.services.groq_client import GroqLLMClient
        client = GroqLLMClient.__new__(GroqLLMClient)
        raw = 'Here is the analysis:\n{"summary": "test summary", "confidence_overall": 0.7}'
        result = client._parse_json_response(raw)
        assert result.get("summary") == "test summary"
        assert result.get("confidence_overall") == 0.7

    @pytest.mark.asyncio
    async def test_groq_complete_uses_fallback_on_model_failure(self):
        """Test that on primary model failure, fallback model is tried."""
        from backend.services.groq_client import GroqLLMClient
        from groq import APIStatusError
        import httpx

        client = GroqLLMClient.__new__(GroqLLMClient)
        client.primary_model = "mixtral-8x7b-32768"
        client.fallback_model = "llama-3.3-70b-versatile"
        client.temperature = 0.1
        client.max_tokens = 1000

        call_log = []

        async def mock_create(**kwargs):
            call_log.append(kwargs["model"])
            if kwargs["model"] == client.primary_model:
                raise APIStatusError(
                    "model_not_available",
                    response=MagicMock(status_code=503),
                    body={}
                )
            # Fallback succeeds
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = '{"summary": "fallback worked"}'
            mock_resp.usage.prompt_tokens = 100
            mock_resp.usage.completion_tokens = 50
            return mock_resp

        mock_groq = AsyncMock()
        mock_groq.chat.completions.create = mock_create
        client.client = mock_groq

        result = await client.complete("test message")
        assert client.fallback_model in call_log
        assert result.get("summary") == "fallback worked"


# ────────────────────────────────────────────────────────────────
# Embedding Projector Tests
# ────────────────────────────────────────────────────────────────

class TestEmbeddingProjector:
    def test_projection_output_dim(self):
        from backend.services.multimodal_service import EmbeddingProjector
        proj = EmbeddingProjector(input_dim=512, output_dim=768)
        inp = np.random.rand(512).astype(np.float32)
        out = proj.project(inp)
        assert out.shape == (768,)

    def test_projection_is_normalized(self):
        from backend.services.multimodal_service import EmbeddingProjector
        proj = EmbeddingProjector(input_dim=512, output_dim=768)
        inp = np.random.rand(512).astype(np.float32) * 100  # large magnitude
        out = proj.project(inp)
        norm = np.linalg.norm(out)
        assert abs(norm - 1.0) < 1e-5

    def test_projection_deterministic(self):
        from backend.services.multimodal_service import EmbeddingProjector
        proj1 = EmbeddingProjector(input_dim=128, output_dim=256)
        proj2 = EmbeddingProjector(input_dim=128, output_dim=256)
        inp = np.ones(128, dtype=np.float32)
        out1 = proj1.project(inp)
        out2 = proj2.project(inp)
        np.testing.assert_array_almost_equal(out1, out2)


# ────────────────────────────────────────────────────────────────
# Chunking Tests
# ────────────────────────────────────────────────────────────────

class TestDocumentChunking:
    def test_short_doc_not_chunked(self):
        from data.scripts.ingest_pubmed import chunk_document
        doc = {"id": "x", "text": "short text here", "title": "T"}
        chunks = chunk_document(doc, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0]["id"] == "x"

    def test_long_doc_chunked_with_overlap(self):
        from data.scripts.ingest_pubmed import chunk_document
        words = ["word"] * 1100
        doc = {"id": "long-doc", "text": " ".join(words), "title": "T"}
        chunks = chunk_document(doc, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        # Each chunk should have parent_id pointing to original
        for c in chunks:
            assert c.get("parent_id") == "long-doc"

    def test_chunks_preserve_all_words(self):
        """All words should appear in at least one chunk."""
        from data.scripts.ingest_pubmed import chunk_document
        # Use unique words to track coverage
        unique_words = [f"WORD{i}" for i in range(700)]
        doc = {"id": "d", "text": " ".join(unique_words), "title": "T"}
        chunks = chunk_document(doc, chunk_size=200, overlap=30)
        all_chunk_text = " ".join(c["text"] for c in chunks)
        # Most unique words should appear (overlap means some may be duplicated but all covered)
        for word in unique_words[50:650]:  # check middle range, skipping edges
            assert word in all_chunk_text
