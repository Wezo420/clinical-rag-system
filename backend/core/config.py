"""
Core configuration using Pydantic Settings.
Loads from environment variables / .env file.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # App
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # Groq
    GROQ_API_KEY: str
    GROQ_DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_MODEL: str = "mixtral-8x7b-32768"
    GROQ_MAX_TOKENS: int = 4096
    GROQ_TEMPERATURE: float = 0.1

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres@localhost:5432/postgres"
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "clinical_rag_docs"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # JWT
    JWT_SECRET_KEY: str = "change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Storage
    STORAGE_BACKEND: str = "local"
    LOCAL_STORAGE_PATH: str = "./data/uploads"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None

    # ML
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MEDICAL_EMBEDDING_MODEL: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    FAISS_DIM: int = 384

    # ML Service
    ML_SERVICE_URL: str = "http://localhost:8001"
    ML_SERVICE_TIMEOUT: int = 30

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 20
    RATE_LIMIT_BURST: int = 5

    # Security
    MAX_TEXT_INPUT_LENGTH: int = 5000
    MAX_IMAGE_SIZE_MB: int = 10
    ENABLE_PROMPT_INJECTION_DETECTION: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
