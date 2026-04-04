"""
RAG Service — Full Pipeline Implementation.

Steps:
1. Query preprocessing + expansion
2. Dense retrieval (FAISS)
3. Sparse retrieval (BM25)
4. Hybrid fusion (RRF)
5. Cross-encoder re-ranking
6. Context compression
7. Context building
8. Groq inference
"""

import asyncio
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from backend.core.config import settings
from backend.services.groq_client import get_groq_client

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------
# Document store in-memory (backed by MongoDB for persistence)
# -----------------------------------------------------------------------

class DocumentStore:
    """In-memory document store for retrieved chunks."""

    def __init__(self):
        self._documents: Dict[str, dict] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}

    def add(self, doc_id: str, doc: dict):
        idx = len(self._documents)
        self._documents[doc_id] = doc
        self._id_to_idx[doc_id] = idx
        self._idx_to_id[idx] = doc_id

    def get(self, doc_id: str) -> Optional[dict]:
        return self._documents.get(doc_id)

    def get_by_idx(self, idx: int) -> Optional[dict]:
        doc_id = self._idx_to_id.get(idx)
        return self._documents.get(doc_id) if doc_id else None

    def all_texts(self) -> List[str]:
        return [d.get("text", "") for d in self._documents.values()]

    def __len__(self):
        return len(self._documents)


# -----------------------------------------------------------------------
# FAISS Dense Retriever
# -----------------------------------------------------------------------

class FAISSRetriever:
    """Dense retrieval using FAISS."""

    def __init__(self, dim: int = 384):
        import faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after L2 norm)
        self.doc_store = DocumentStore()
        self._initialized = False

    def add_documents(self, documents: List[dict], embeddings: np.ndarray):
        """Add documents with their pre-computed embeddings."""
        import faiss
        # L2 normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        for doc in documents:
            self.doc_store.add(doc["id"], doc)
        self._initialized = True
        logger.info(f"FAISS index: {self.index.ntotal} vectors added")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Returns list of (doc_idx, score) tuples."""
        import faiss
        if not self._initialized or self.index.ntotal == 0:
            return []
        q = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(k, self.index.ntotal))
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    def save(self, path: str):
        import faiss
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs.pkl", "wb") as f:
            pickle.dump(self.doc_store, f)
        logger.info("FAISS index saved", path=path)

    def load(self, path: str) -> bool:
        import faiss
        index_path = f"{path}.index"
        docs_path = f"{path}.docs.pkl"
        if not os.path.exists(index_path):
            return False
        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.doc_store = pickle.load(f)
        self._initialized = True
        logger.info(f"FAISS index loaded", ntotal=self.index.ntotal)
        return True


# -----------------------------------------------------------------------
# BM25 Sparse Retriever
# -----------------------------------------------------------------------

class BM25Retriever:
    """Sparse retrieval using BM25Okapi."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []
        self._initialized = False

    def build_index(self, documents: List[dict]):
        """Build BM25 index from document list."""
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)

        tokenized_corpus = []
        self.doc_ids = []
        
        for doc in documents:
            text = doc.get("text", "")
            tokens = self._tokenize(text)
            tokenized_corpus.append(tokens)
            self.doc_ids.append(doc["id"])

        self.bm25 = BM25Okapi(tokenized_corpus)
        self._initialized = True
        logger.info(f"BM25 index built with {len(documents)} documents")

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Returns list of (doc_id, score) tuples."""
        if not self._initialized:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        return results

    def _tokenize(self, text: str) -> List[str]:
        import nltk
        tokens = nltk.word_tokenize(text.lower())
        stopwords = set(nltk.corpus.stopwords.words("english")) if hasattr(nltk.corpus, "stopwords") else set()
        return [t for t in tokens if t.isalnum() and t not in stopwords and len(t) > 2]


# -----------------------------------------------------------------------
# Reciprocal Rank Fusion
# -----------------------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: List[List[str]], k: int = 60
) -> Dict[str, float]:
    """
    Fuse multiple ranked lists using RRF.
    Returns {doc_id: fused_score}.
    """
    scores: Dict[str, float] = {}
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


# -----------------------------------------------------------------------
# Cross-Encoder Re-ranker
# -----------------------------------------------------------------------

class CrossEncoderReranker:
    """Re-rank retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = None
        self._model_name = model_name

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name, max_length=512)
                logger.info("Cross-encoder loaded", model=self._model_name)
            except Exception as e:
                logger.warning("Cross-encoder load failed, skipping reranking", error=str(e))

    def rerank(
        self, query: str, documents: List[dict], top_k: int = 5
    ) -> List[dict]:
        """Re-rank documents and return top_k."""
        self._load_model()
        if self._model is None or not documents:
            return documents[:top_k]

        pairs = [(query, doc.get("text", "")[:512]) for doc in documents]
        try:
            scores = self._model.predict(pairs)
            scored = list(zip(scores, documents))
            scored.sort(key=lambda x: x[0], reverse=True)
            reranked = []
            for score, doc in scored[:top_k]:
                doc_copy = dict(doc)
                doc_copy["rerank_score"] = float(score)
                reranked.append(doc_copy)
            return reranked
        except Exception as e:
            logger.warning("Re-ranking failed, returning original order", error=str(e))
            return documents[:top_k]


# -----------------------------------------------------------------------
# Main RAG Service
# -----------------------------------------------------------------------

class RAGService:
    """
    Full RAG pipeline: retrieval → reranking → context fusion → Groq inference.
    """

    def __init__(self):
        self.faiss_retriever = FAISSRetriever(dim=384)
        self.bm25_retriever = BM25Retriever()
        self.reranker = CrossEncoderReranker()
        self.groq = get_groq_client()
        self._embedding_model = None
        self._initialized = False

    async def initialize(self):
        """Load FAISS index and BM25 from disk on startup."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_indices)
        self._initialized = True
        logger.info("RAG service initialized")

    def _load_indices(self):
        index_path = settings.FAISS_INDEX_PATH
        loaded = self.faiss_retriever.load(index_path)
        if not loaded:
            logger.warning("No FAISS index found — RAG will use empty index until data is ingested")

    def _get_embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded", model=settings.EMBEDDING_MODEL)
        return self._embedding_model

    def _embed_query(self, text: str) -> np.ndarray:
        model = self._get_embedding_model()
        embedding = model.encode([text], show_progress_bar=False, normalize_embeddings=True)
        return embedding[0]

    async def retrieve(
        self,
        query: str,
        k: int = 20,
        use_query_expansion: bool = True,
    ) -> List[dict]:
        """
        Hybrid retrieval: dense (FAISS) + sparse (BM25) + RRF fusion.
        """
        queries = [query]
        
        # Query expansion via LLM rewriting
        if use_query_expansion:
            try:
                expanded = await self.groq.rewrite_queries(query)
                queries = list(set([query] + expanded[:2]))  # deduplicate
            except Exception as e:
                logger.warning("Query expansion failed", error=str(e))

        loop = asyncio.get_event_loop()
        
        all_dense_ids = []
        all_sparse_ids = []

        for q in queries:
            # Dense
            q_emb = await loop.run_in_executor(None, self._embed_query, q)
            dense_results = self.faiss_retriever.search(q_emb, k=k)
            for idx, score in dense_results:
                doc = self.faiss_retriever.doc_store.get_by_idx(idx)
                if doc:
                    all_dense_ids.append(doc["id"])

            # Sparse
            sparse_results = self.bm25_retriever.search(q, k=k)
            all_sparse_ids.extend([doc_id for doc_id, _ in sparse_results])

        # Hybrid fusion
        fused_scores = reciprocal_rank_fusion(
            [all_dense_ids, all_sparse_ids], k=60
        )

        # Collect unique docs sorted by fused score
        sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        retrieved_docs = []
        seen = set()
        for doc_id, score in sorted_ids[:k]:
            if doc_id not in seen:
                doc = self.faiss_retriever.doc_store.get(doc_id)
                if doc:
                    doc_copy = dict(doc)
                    doc_copy["score"] = score
                    retrieved_docs.append(doc_copy)
                    seen.add(doc_id)

        logger.info(
            "Retrieval complete",
            queries=len(queries),
            retrieved=len(retrieved_docs),
        )
        return retrieved_docs

    async def retrieve_and_rerank(
        self,
        query: str,
        k_retrieve: int = 20,
        k_rerank: int = 8,
    ) -> List[dict]:
        """Full retrieval + re-ranking pipeline."""
        docs = await self.retrieve(query, k=k_retrieve)
        if not docs:
            return []

        loop = asyncio.get_event_loop()
        reranked = await loop.run_in_executor(
            None, self.reranker.rerank, query, docs, k_rerank
        )
        return reranked

    async def run_rag_pipeline(
        self,
        clinical_text: str,
        image_summary: Optional[str] = None,
        structured_data: Optional[dict] = None,
        stream: bool = False,
    ) -> dict:
        """
        Complete RAG pipeline:
        retrieve → rerank → build context → call Groq → safety check
        """
        start_time = time.perf_counter()

        # Build composite query
        composite_query = clinical_text
        if image_summary:
            composite_query += f" {image_summary}"

        # Retrieve & rerank
        reranked_docs = await self.retrieve_and_rerank(
            query=composite_query,
            k_retrieve=20,
            k_rerank=8,
        )

        # Build RAG prompt
        prompt = self.groq.build_rag_prompt(
            clinical_text=clinical_text,
            context_blocks=reranked_docs,
            structured_data=structured_data,
            image_summary=image_summary,
        )

        # Groq inference
        if stream:
            # Return generator directly for streaming endpoint
            return {
                "stream_generator": self.groq.stream_complete(
                    user_message=prompt,
                    system_message=self.groq.__class__.__module__,  # uses SYSTEM_PROMPT_CLINICAL
                ),
                "docs": reranked_docs,
            }

        result = await self.groq.complete(user_message=prompt)

        # Safety check
        safety = await self.groq.safety_check(result)

        # Merge safety flags into result
        existing_flags = result.get("safety_flags", [])
        groq_flags = safety.get("flags", [])
        result["safety_flags"] = existing_flags + groq_flags

        processing_ms = int((time.perf_counter() - start_time) * 1000)
        result["_meta"] = {
            "processing_ms": processing_ms,
            "retrieval_count": len(reranked_docs),
            "model": settings.GROQ_DEFAULT_MODEL,
            "retrieved_sources": [
                {
                    "source_id": d.get("id"),
                    "title": d.get("title", ""),
                    "authors": d.get("authors", []),
                    "journal": d.get("journal", ""),
                    "year": d.get("year"),
                    "pmid": d.get("pmid"),
                    "excerpt": d.get("text", "")[:300],
                    "relevance_score": d.get("rerank_score", d.get("score", 0)),
                    "url": d.get("url"),
                }
                for d in reranked_docs
            ],
        }

        logger.info(
            "RAG pipeline complete",
            processing_ms=processing_ms,
            retrieval_count=len(reranked_docs),
            has_error="error" in result,
        )

        return result

    async def ingest_documents(self, documents: List[dict]):
        """
        Ingest a batch of documents into FAISS + BM25 indices.
        Each doc: {id, text, title, authors, journal, year, pmid, url}
        """
        from sentence_transformers import SentenceTransformer
        
        texts = [d["text"] for d in documents]
        model = self._get_embedding_model()
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        )

        self.faiss_retriever.add_documents(documents, embeddings)
        self.bm25_retriever.build_index(documents)
        self.faiss_retriever.save(settings.FAISS_INDEX_PATH)
        
        logger.info(f"Ingested {len(documents)} documents into RAG indices")
