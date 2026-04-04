"""
Data ingestion pipeline for medical literature.

Sources:
- PubMed (via Entrez API)
- Local JSONL files (MIMIC, curated datasets)

Usage:
    python -m data.scripts.ingest_pubmed --query "pneumonia diagnosis" --max 500
    python -m data.scripts.ingest_pubmed --file ./data/medical_corpus.jsonl
"""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Generator, List

import structlog
from Bio import Entrez

logger = structlog.get_logger(__name__)

# PubMed fields to extract
PUBMED_FIELDS = ["TI", "AB", "AU", "TA", "DP", "PMID", "AID"]

# Medical search queries for building the initial corpus
DEFAULT_QUERIES = [
    "differential diagnosis internal medicine",
    "chest pain differential diagnosis",
    "dyspnea diagnosis workup",
    "fever etiology differential",
    "headache diagnosis clinical",
    "abdominal pain differential diagnosis",
    "skin rash diagnosis dermatology",
    "neurological symptoms diagnosis",
    "cardiac arrhythmia diagnosis",
    "diabetes mellitus diagnosis management",
    "hypertension clinical presentation",
    "pneumonia diagnosis treatment",
    "acute kidney injury diagnosis",
    "liver disease diagnosis",
    "thyroid disease diagnosis",
    "anemia diagnosis workup",
    "sepsis diagnosis criteria",
    "stroke diagnosis imaging",
    "pulmonary embolism diagnosis",
    "deep vein thrombosis diagnosis",
]


def fetch_pubmed_articles(
    query: str,
    max_results: int = 100,
    email: str = "research@example.com",
) -> List[dict]:
    """Fetch articles from PubMed using Entrez."""
    Entrez.email = email
    documents = []

    try:
        # Search
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, usehistory="y")
        search_results = Entrez.read(handle)
        handle.close()

        if not search_results["IdList"]:
            logger.info("No results for query", query=query)
            return []

        ids = search_results["IdList"]
        logger.info(f"Found {len(ids)} articles for: {query}")

        # Fetch in batches
        batch_size = 50
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch_ids),
                rettype="medline",
                retmode="text",
            )
            records_text = fetch_handle.read()
            fetch_handle.close()

            # Parse MEDLINE format
            parsed = _parse_medline(records_text)
            documents.extend(parsed)

            # Rate limiting
            time.sleep(0.35)

    except Exception as e:
        logger.error("PubMed fetch failed", query=query, error=str(e))

    return documents


def _parse_medline(text: str) -> List[dict]:
    """Parse MEDLINE format text into document dicts."""
    documents = []
    current = {}

    for line in text.split("\n"):
        if line.strip() == "":
            if current.get("AB") and current.get("TI"):
                doc = _build_document(current)
                if doc:
                    documents.append(doc)
            current = {}
            continue

        if len(line) > 6 and line[4] == "-":
            field = line[:4].strip()
            value = line[6:].strip()
            if field in current:
                if isinstance(current[field], list):
                    current[field].append(value)
                else:
                    current[field] = [current[field], value]
            else:
                current[field] = value
        elif line.startswith("      ") and current:
            # Continuation line
            last_key = list(current.keys())[-1] if current else None
            if last_key:
                if isinstance(current[last_key], list):
                    current[last_key][-1] += " " + line.strip()
                else:
                    current[last_key] += " " + line.strip()

    return documents


def _build_document(record: dict) -> dict | None:
    """Convert parsed MEDLINE record to document format."""
    title = record.get("TI", "")
    abstract = record.get("AB", "")
    
    if not title or not abstract:
        return None

    # Normalize authors
    authors = record.get("AU", [])
    if isinstance(authors, str):
        authors = [authors]

    # Journal
    journal = record.get("TA", record.get("JT", ""))

    # Year
    date = record.get("DP", "")
    year = None
    if date:
        try:
            year = int(date[:4])
        except (ValueError, IndexError):
            pass

    # PMID
    pmid = record.get("PMID", "")

    # DOI
    doi = None
    aids = record.get("AID", [])
    if isinstance(aids, str):
        aids = [aids]
    for aid in aids:
        if "[doi]" in aid.lower():
            doi = aid.replace("[doi]", "").strip()
            break

    # Build full text for embedding
    full_text = f"{title}\n\n{abstract}"

    return {
        "id": str(uuid.uuid4()),
        "pmid": str(pmid),
        "title": title,
        "text": full_text,
        "abstract": abstract,
        "authors": authors[:5],  # cap at 5
        "journal": journal,
        "year": year,
        "doi": doi,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
        "source": "pubmed",
    }


def load_jsonl_documents(filepath: str) -> List[dict]:
    """Load documents from a JSONL file."""
    docs = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    doc = json.loads(line)
                    # Ensure required fields
                    if not doc.get("id"):
                        doc["id"] = str(uuid.uuid4())
                    if doc.get("text") or doc.get("abstract"):
                        if not doc.get("text"):
                            doc["text"] = f"{doc.get('title', '')}\n\n{doc.get('abstract', '')}"
                        docs.append(doc)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse JSONL line", error=str(e))
    return docs


def chunk_document(doc: dict, chunk_size: int = 512, overlap: int = 50) -> List[dict]:
    """
    Split long documents into overlapping chunks.
    Preserves metadata in each chunk.
    """
    text = doc.get("text", "")
    words = text.split()
    
    if len(words) <= chunk_size:
        return [doc]

    chunks = []
    for start in range(0, len(words), chunk_size - overlap):
        chunk_words = words[start : start + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunk = dict(doc)
        chunk["id"] = f"{doc['id']}_chunk_{start}"
        chunk["text"] = chunk_text
        chunk["parent_id"] = doc["id"]
        chunks.append(chunk)

    return chunks


async def ingest_to_rag(documents: List[dict], batch_size: int = 100):
    """Ingest documents into the RAG service (FAISS + BM25)."""
    from backend.core.config import settings
    from backend.services.rag_service import RAGService

    rag = RAGService()
    await rag.initialize()

    # Chunk documents
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size=400, overlap=50)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks to ingest: {len(all_chunks)}")

    # Ingest in batches
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        await rag.ingest_documents(batch)
        logger.info(f"Ingested batch {i // batch_size + 1}, total so far: {i + len(batch)}")

    logger.info(f"Ingestion complete. Total: {len(all_chunks)} chunks")


async def main(args):
    documents = []

    if args.file:
        logger.info("Loading from file", path=args.file)
        documents = load_jsonl_documents(args.file)
    else:
        queries = [args.query] if args.query else DEFAULT_QUERIES[:5]
        for q in queries:
            logger.info("Fetching PubMed", query=q)
            docs = fetch_pubmed_articles(q, max_results=args.max)
            documents.extend(docs)
            logger.info(f"Collected {len(documents)} documents so far")

    logger.info(f"Total documents collected: {len(documents)}")

    # Optionally save raw corpus
    if args.save:
        out_path = Path("./data/raw_corpus.jsonl")
        with open(out_path, "w") as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")
        logger.info(f"Corpus saved to {out_path}")

    # Ingest
    await ingest_to_rag(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest medical literature into RAG")
    parser.add_argument("--query", type=str, help="PubMed search query")
    parser.add_argument("--file", type=str, help="Path to JSONL file")
    parser.add_argument("--max", type=int, default=100, help="Max results per query")
    parser.add_argument("--save", action="store_true", help="Save raw corpus to disk")
    args = parser.parse_args()

    asyncio.run(main(args))
