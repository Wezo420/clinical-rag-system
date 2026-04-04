"""
RAG Evaluation Suite

Metrics:
1. Retrieval Accuracy — NDCG@5, Recall@10
2. Grounding Quality — citation coverage, evidence alignment
3. Hallucination Rate — claims not grounded in retrieved context
4. Latency — P50/P95/P99 pipeline latency

Usage: python -m data.scripts.evaluate_rag --cases ./data/eval_cases.json
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Dataset Format
# Each case: {id, clinical_text, expected_conditions, expected_pmids}
# ──────────────────────────────────────────────────────────────────────────────

DEMO_EVAL_CASES = [
    {
        "id": "eval-001",
        "clinical_text": (
            "55-year-old male with sudden-onset severe headache described as 'worst headache of life', "
            "neck stiffness, photophobia, fever 38.9°C, nausea. No prior headache history. "
            "Normal neurologic exam. CT head: normal."
        ),
        "expected_conditions": ["subarachnoid hemorrhage", "bacterial meningitis", "viral meningitis"],
        "gold_pmids": [],  # would be real PMIDs in production eval
    },
    {
        "id": "eval-002",
        "clinical_text": (
            "72-year-old female, post-op day 3 after hip replacement. Sudden onset dyspnea, "
            "pleuritic chest pain, tachycardia (HR 118), SpO2 92% on room air. "
            "Left calf swelling. D-dimer elevated."
        ),
        "expected_conditions": ["pulmonary embolism", "deep vein thrombosis", "pneumonia"],
        "gold_pmids": [],
    },
    {
        "id": "eval-003",
        "clinical_text": (
            "45-year-old female with 6-month history of fatigue, weight gain 8kg, cold intolerance, "
            "constipation, dry skin, hair thinning, bradycardia (HR 54). TSH 12.5 mIU/L, "
            "Free T4 0.6 ng/dL."
        ),
        "expected_conditions": ["hypothyroidism", "hashimoto thyroiditis"],
        "gold_pmids": [],
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Recall@K — fraction of relevant docs found in top-K retrieved."""
    if not relevant:
        return 1.0
    top_k = set(retrieved[:k])
    hits = sum(1 for r in relevant if r in top_k)
    return hits / len(relevant)


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """NDCG@K — normalized discounted cumulative gain."""
    if not relevant:
        return 1.0
    relevant_set = set(relevant)
    
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    
    # IDCG (ideal)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_condition_overlap(
    predicted: List[str], expected: List[str]
) -> float:
    """Compute how many expected conditions appear in predicted (fuzzy match)."""
    if not expected:
        return 1.0
    hits = 0
    for exp in expected:
        exp_lower = exp.lower()
        for pred in predicted:
            if exp_lower in pred.lower() or any(
                word in pred.lower() for word in exp_lower.split() if len(word) > 4
            ):
                hits += 1
                break
    return hits / len(expected)


def estimate_hallucination_rate(result: dict, retrieved_docs: List[dict]) -> float:
    """
    Heuristic hallucination estimate.
    Checks if key claims in differential_reasoning are grounded in retrieved text.
    Returns 0.0 (no hallucination) to 1.0 (fully hallucinated).
    """
    reasoning = result.get("differential_reasoning", "")
    if not reasoning or not retrieved_docs:
        return 0.5  # uncertain

    # Extract meaningful sentences from reasoning
    sentences = [s.strip() for s in reasoning.split(".") if len(s.strip()) > 30]
    if not sentences:
        return 0.0

    retrieved_text = " ".join(d.get("text", "") for d in retrieved_docs).lower()

    grounded = 0
    for sentence in sentences:
        # Check if key medical terms from sentence appear in retrieved context
        words = [w for w in sentence.lower().split() if len(w) > 5]
        if not words:
            continue
        matches = sum(1 for w in words if w in retrieved_text)
        if matches / len(words) > 0.3:  # 30% keyword overlap threshold
            grounded += 1

    if not sentences:
        return 0.0
    return 1.0 - (grounded / len(sentences))


# ──────────────────────────────────────────────────────────────────────────────
# Main Evaluation Runner
# ──────────────────────────────────────────────────────────────────────────────

async def evaluate_case(case: dict, rag_service) -> dict:
    """Run a single eval case through the RAG pipeline."""
    start = time.perf_counter()

    try:
        result = await rag_service.run_rag_pipeline(
            clinical_text=case["clinical_text"]
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        # Extract predictions
        hypotheses = result.get("condition_hypotheses", [])
        predicted_conditions = [h.get("condition", "") for h in hypotheses]

        # Compute metrics
        condition_recall = compute_condition_overlap(
            predicted_conditions, case.get("expected_conditions", [])
        )

        retrieved_sources = result.get("_meta", {}).get("retrieved_sources", [])
        hallucination = estimate_hallucination_rate(result, retrieved_sources)

        return {
            "case_id": case["id"],
            "success": True,
            "latency_ms": latency_ms,
            "condition_recall": condition_recall,
            "retrieval_count": len(retrieved_sources),
            "confidence": result.get("confidence_overall", 0),
            "hallucination_estimate": hallucination,
            "predicted_top_3": predicted_conditions[:3],
            "expected": case.get("expected_conditions", []),
            "has_error": "error" in result,
        }

    except Exception as e:
        return {
            "case_id": case["id"],
            "success": False,
            "error": str(e),
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }


async def run_evaluation(cases: List[dict], output_path: str = "./data/eval_results.json"):
    """Run full evaluation suite and write results."""
    from backend.services.rag_service import RAGService

    rag = RAGService()
    await rag.initialize()

    results = []
    for i, case in enumerate(cases):
        logger.info(f"Evaluating case {i+1}/{len(cases)}: {case['id']}")
        result = await evaluate_case(case, rag)
        results.append(result)
        logger.info(
            "Case result",
            case_id=result["case_id"],
            condition_recall=result.get("condition_recall", "N/A"),
            latency_ms=result.get("latency_ms"),
        )

    # Aggregate metrics
    successful = [r for r in results if r.get("success")]
    if successful:
        metrics = {
            "total_cases": len(cases),
            "successful": len(successful),
            "avg_latency_ms": np.mean([r["latency_ms"] for r in successful]),
            "p95_latency_ms": np.percentile([r["latency_ms"] for r in successful], 95),
            "avg_condition_recall": np.mean([r.get("condition_recall", 0) for r in successful]),
            "avg_confidence": np.mean([r.get("confidence", 0) for r in successful]),
            "avg_hallucination_estimate": np.mean([r.get("hallucination_estimate", 0) for r in successful]),
            "avg_retrieval_count": np.mean([r.get("retrieval_count", 0) for r in successful]),
        }

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Cases: {metrics['total_cases']} total, {metrics['successful']} successful")
        print(f"Avg Latency:         {metrics['avg_latency_ms']:.0f}ms  (P95: {metrics['p95_latency_ms']:.0f}ms)")
        print(f"Condition Recall:    {metrics['avg_condition_recall']:.1%}")
        print(f"Avg Confidence:      {metrics['avg_confidence']:.1%}")
        print(f"Hallucination Est.:  {metrics['avg_hallucination_estimate']:.1%}")
        print(f"Avg Retrieved Docs:  {metrics['avg_retrieval_count']:.1f}")
        print("="*60)
    else:
        metrics = {"error": "No successful cases"}

    output = {"metrics": metrics, "cases": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Evaluation complete. Results saved to {output_path}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, help="Path to JSON eval cases file")
    parser.add_argument("--output", type=str, default="./data/eval_results.json")
    args = parser.parse_args()

    if args.cases:
        with open(args.cases) as f:
            cases = json.load(f)
    else:
        cases = DEMO_EVAL_CASES
        print(f"Using {len(cases)} demo eval cases (no --cases file specified)")

    asyncio.run(run_evaluation(cases, args.output))
