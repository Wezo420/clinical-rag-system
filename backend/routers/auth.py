"""
Auth endpoints: register, login, token refresh.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import create_access_token, hash_password, verify_password
from backend.core.config import settings
from backend.core.database import get_db
from backend.middleware.rate_limit import rate_limit
from backend.models.schemas import TokenResponse, UserCreate, UserLogin, UserResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@rate_limit("5/minute")
async def register(
    request: Request,
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    from backend.models.sql_models import User

    # Check username
    existing = (await db.execute(
        select(User).where(User.username == body.username)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")

    # Check email
    existing_email = (await db.execute(
        select(User).where(User.email == body.email)
    )).scalar_one_or_none()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info("User registered", username=body.username)
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.post("/login", response_model=TokenResponse)
@rate_limit("10/minute")
async def login(
    request: Request,
    body: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    from backend.models.sql_models import User

    user = (await db.execute(
        select(User).where(User.username == body.username)
    )).scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")

    token = create_access_token(
        subject=str(user.id),
        extra={"username": user.username},
    )

    logger.info("User logged in", username=user.username)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
