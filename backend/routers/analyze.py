"""
POST /api/v1/analyze-case — Core clinical analysis endpoint.
Orchestrates: input validation → image retrieval → RAG pipeline → Groq → response.
"""

import json
import time
import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import get_optional_user
from backend.core.database import get_db
from backend.middleware.rate_limit import rate_limit
from backend.models.schemas import (
    AnalysisResult,
    AnalysisStatus,
    ClinicalCaseRequest,
    ConditionHypothesis,
    ConfidenceLevel,
    MedicalEvidence,
    SafetyFlag,
)
from backend.services.groq_client import get_groq_client
from backend.services.multimodal_service import MultimodalFusionService

logger = structlog.get_logger(__name__)

router = APIRouter()
multimodal_service = MultimodalFusionService()


def _confidence_level(score: float) -> ConfidenceLevel:
    if score >= 0.7:
        return ConfidenceLevel.HIGH
    if score >= 0.4:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _build_analysis_result(
    raw: dict,
    case_id: str,
    model: str,
    processing_ms: int,
    image_summary: Optional[str],
    structured_data_used: bool,
) -> AnalysisResult:
    """Convert raw LLM output dict to validated AnalysisResult."""

    # Handle LLM error
    if "error" in raw:
        return AnalysisResult(
            case_id=case_id,
            status=AnalysisStatus.FAILED,
            summary=raw.get("message", "Analysis failed"),
            condition_hypotheses=[],
            differential_reasoning="Insufficient evidence to generate hypotheses.",
            evidence=[],
            confidence_overall=0.0,
            confidence_level=ConfidenceLevel.LOW,
            safety_flags=[SafetyFlag(
                flag_type="error",
                message=raw.get("message", "Analysis failed"),
                severity="warning"
            )],
            model_used=model,
            retrieval_count=0,
            processing_time_ms=processing_ms,
            image_analysis_summary=image_summary,
            structured_data_used=structured_data_used,
        )

    meta = raw.get("_meta", {})
    sources = meta.get("retrieved_sources", [])

    # Parse hypotheses
    hypotheses = []
    for h in raw.get("condition_hypotheses", []):
        try:
            confidence = float(h.get("confidence", 0.3))
            hypotheses.append(ConditionHypothesis(
                condition=h.get("condition", "Unknown"),
                icd10_code=h.get("icd10_code"),
                confidence=min(1.0, max(0.0, confidence)),
                confidence_level=_confidence_level(confidence),
                supporting_factors=h.get("supporting_factors", []),
                against_factors=h.get("against_factors", []),
                recommended_workup=h.get("recommended_workup", []),
            ))
        except Exception as e:
            logger.warning("Failed to parse hypothesis", error=str(e), hypothesis=h)

    # Parse evidence
    evidence = []
    for s in sources:
        try:
            evidence.append(MedicalEvidence(
                source_id=s.get("source_id", str(uuid.uuid4())),
                title=s.get("title", "Unknown"),
                authors=s.get("authors", []),
                journal=s.get("journal"),
                year=s.get("year"),
                pmid=s.get("pmid"),
                excerpt=s.get("excerpt", ""),
                relevance_score=min(1.0, max(0.0, float(s.get("relevance_score", 0)))),
                url=s.get("url"),
            ))
        except Exception as e:
            logger.warning("Failed to parse evidence", error=str(e))

    # Parse safety flags
    flags = []
    for f in raw.get("safety_flags", []):
        try:
            flags.append(SafetyFlag(
                flag_type=f.get("flag_type", "info"),
                message=f.get("message", ""),
                severity=f.get("severity", "info"),
            ))
        except Exception:
            pass

    overall_confidence = float(raw.get("confidence_overall", 0.3))

    return AnalysisResult(
        case_id=case_id,
        status=AnalysisStatus.COMPLETED,
        summary=raw.get("summary", "Analysis complete."),
        condition_hypotheses=hypotheses,
        differential_reasoning=raw.get("differential_reasoning", ""),
        evidence=evidence,
        confidence_overall=min(1.0, max(0.0, overall_confidence)),
        confidence_level=_confidence_level(overall_confidence),
        safety_flags=flags,
        model_used=meta.get("model", model),
        retrieval_count=meta.get("retrieval_count", len(sources)),
        processing_time_ms=processing_ms,
        image_analysis_summary=image_summary,
        structured_data_used=structured_data_used,
    )


async def _persist_result(
    db: AsyncSession,
    case_id: str,
    user_id: Optional[str],
    request: ClinicalCaseRequest,
    result: AnalysisResult,
):
    """Background task: persist case and result to PostgreSQL."""
    try:
        from backend.models.sql_models import ClinicalCase, AnalysisResult as DBResult
        from sqlalchemy import select

        case = ClinicalCase(
            id=case_id,
            user_id=user_id,
            clinical_text=request.clinical_text[:5000],
            structured_data=request.structured_data.model_dump() if request.structured_data else None,
            status="completed",
        )
        db.add(case)

        db_result = DBResult(
            case_id=case_id,
            status="completed",
            result_json=result.model_dump(mode="json"),
            model_used=result.model_used,
            confidence_overall=result.confidence_overall,
            retrieval_count=result.retrieval_count,
            processing_time_ms=result.processing_time_ms,
        )
        db.add(db_result)
        await db.commit()
        logger.info("Case persisted", case_id=case_id)
    except Exception as e:
        logger.error("Failed to persist case", case_id=case_id, error=str(e))


@router.post(
    "/analyze-case",
    response_model=AnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze a clinical case using RAG + Groq LLM",
    description=(
        "Submit clinical text, optional images, and structured data. "
        "Returns AI-generated hypotheses grounded in medical literature. "
        "⚠️ FOR RESEARCH USE ONLY."
    ),
)
@rate_limit("10/minute")
async def analyze_case(
    request_body: ClinicalCaseRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    case_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    user_id = current_user.get("sub") if current_user else None

    logger.info(
        "Analyzing clinical case",
        case_id=case_id,
        user_id=user_id,
        has_images=bool(request_body.image_ids),
        has_structured=bool(request_body.structured_data),
        stream=request_body.stream,
    )

    # --- Load images (if any) ---
    image_summary = None
    image_bytes_list = []

    if request_body.image_ids:
        from backend.services.storage_service import StorageService
        storage = StorageService()
        for img_id in request_body.image_ids[:3]:  # max 3 images
            try:
                img_bytes = await storage.get_image_bytes(img_id)
                if img_bytes:
                    image_bytes_list.append(img_bytes)
            except Exception as e:
                logger.warning("Failed to load image", image_id=img_id, error=str(e))

    # --- Multimodal processing ---
    if image_bytes_list:
        try:
            descriptions = []
            for img_bytes in image_bytes_list:
                _, desc = await multimodal_service.get_multimodal_query_embedding(
                    clinical_text=request_body.clinical_text,
                    image_bytes=img_bytes,
                )
                if desc:
                    descriptions.append(desc)
            if descriptions:
                image_summary = " | ".join(descriptions)
        except Exception as e:
            logger.warning("Multimodal processing failed", error=str(e))

    # --- Streaming response ---
    if request_body.stream:
        rag_service = request.app.state.rag_service

        async def stream_generator():
            async for chunk in await rag_service.stream_clinical_analysis(
                clinical_text=request_body.clinical_text,
                image_summary=image_summary,
                structured_data=(
                    request_body.structured_data.model_dump()
                    if request_body.structured_data else None
                ),
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- Non-streaming RAG pipeline ---
    rag_service = request.app.state.rag_service
    try:
        raw_result = await rag_service.run_rag_pipeline(
            clinical_text=request_body.clinical_text,
            image_summary=image_summary,
            structured_data=(
                request_body.structured_data.model_dump()
                if request_body.structured_data else None
            ),
        )
    except Exception as e:
        logger.error("RAG pipeline failed", case_id=case_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis pipeline encountered an error. Please try again.",
        )

    processing_ms = int((time.perf_counter() - start_time) * 1000)

    # Build response
    result = _build_analysis_result(
        raw=raw_result,
        case_id=case_id,
        model=settings.GROQ_DEFAULT_MODEL,
        processing_ms=processing_ms,
        image_summary=image_summary,
        structured_data_used=bool(request_body.structured_data),
    )

    # Persist async
    background_tasks.add_task(
        _persist_result,
        db=db,
        case_id=case_id,
        user_id=user_id,
        request=request_body,
        result=result,
    )

    logger.info(
        "Analysis complete",
        case_id=case_id,
        processing_ms=processing_ms,
        hypotheses=len(result.condition_hypotheses),
        evidence=len(result.evidence),
        confidence=result.confidence_overall,
    )

    return result


# Import settings at module level after it's defined
from backend.core.config import settings
