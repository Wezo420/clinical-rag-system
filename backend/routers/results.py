"""
GET /api/v1/results/{result_id} — Retrieve a stored analysis result.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import get_optional_user
from backend.core.database import get_db
from backend.middleware.rate_limit import rate_limit
from backend.models.schemas import AnalysisResult, AnalysisStatusResponse, AnalysisStatus

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/results/{result_id}",
    response_model=AnalysisResult,
    summary="Retrieve a stored analysis result",
)
@rate_limit("60/minute")
async def get_result(
    result_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_optional_user),
):
    from backend.models.sql_models import AnalysisResult as DBResult

    stmt = select(DBResult).where(DBResult.id == result_id)
    row = (await db.execute(stmt)).scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {result_id} not found",
        )

    try:
        return AnalysisResult(**row.result_json)
    except Exception as e:
        logger.error("Failed to deserialize result", result_id=result_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve result",
        )


@router.get(
    "/results/{result_id}/status",
    response_model=AnalysisStatusResponse,
    summary="Check analysis status",
)
async def get_result_status(
    result_id: str,
    db: AsyncSession = Depends(get_db),
):
    from backend.models.sql_models import AnalysisResult as DBResult

    stmt = select(DBResult.status).where(DBResult.id == result_id)
    row = (await db.execute(stmt)).scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail="Result not found")

    return AnalysisStatusResponse(
        result_id=result_id,
        status=AnalysisStatus(row),
    )
