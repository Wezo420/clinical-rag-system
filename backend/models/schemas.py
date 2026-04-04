"""
Pydantic v2 request/response schemas for the Clinical RAG API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -----------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImageModality(str, Enum):
    XRAY = "xray"
    MRI = "mri"
    CT = "ct"
    DERMATOLOGY = "dermatology"
    PATHOLOGY = "pathology"
    ULTRASOUND = "ultrasound"
    OTHER = "other"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# -----------------------------------------------------------------------
# Request Schemas
# -----------------------------------------------------------------------

class LabValue(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    value: float
    unit: str = Field(..., max_length=50)
    reference_range: Optional[str] = Field(None, max_length=100)
    is_abnormal: Optional[bool] = None


class VitalSign(BaseModel):
    parameter: str = Field(..., max_length=100)
    value: float
    unit: str = Field(..., max_length=50)


class StructuredData(BaseModel):
    lab_values: List[LabValue] = Field(default_factory=list, max_length=50)
    vitals: List[VitalSign] = Field(default_factory=list, max_length=20)
    age: Optional[int] = Field(None, ge=0, le=150)
    sex: Optional[str] = Field(None, pattern=r"^(male|female|other|unknown)$")
    medications: List[str] = Field(default_factory=list, max_length=30)
    allergies: List[str] = Field(default_factory=list, max_length=20)


class ClinicalCaseRequest(BaseModel):
    """Main request for a clinical case analysis."""
    clinical_text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Clinical notes, symptoms, or doctor observations"
    )
    image_ids: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="IDs from previously uploaded medical images"
    )
    structured_data: Optional[StructuredData] = None
    modality: Optional[ImageModality] = None
    patient_id: Optional[str] = Field(None, max_length=100)
    stream: bool = Field(False, description="Enable streaming response")

    @field_validator("clinical_text")
    @classmethod
    def sanitize_clinical_text(cls, v: str) -> str:
        from backend.middleware.security import sanitize_text_input, detect_prompt_injection
        
        v = sanitize_text_input(v)
        is_malicious, pattern = detect_prompt_injection(v)
        if is_malicious:
            raise ValueError(
                f"Input contains potentially malicious content. "
                f"Please provide legitimate clinical information only."
            )
        return v

    @model_validator(mode="after")
    def validate_has_content(self) -> "ClinicalCaseRequest":
        if not self.clinical_text and not self.image_ids:
            raise ValueError("Either clinical_text or image_ids must be provided")
        return self


class ImageUploadMetadata(BaseModel):
    modality: ImageModality = ImageModality.OTHER
    description: Optional[str] = Field(None, max_length=500)
    patient_id: Optional[str] = Field(None, max_length=100)


# -----------------------------------------------------------------------
# Response Schemas
# -----------------------------------------------------------------------

class MedicalEvidence(BaseModel):
    """A single retrieved evidence item."""
    source_id: str
    title: str
    authors: Optional[List[str]] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None
    excerpt: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    url: Optional[str] = None


class ConditionHypothesis(BaseModel):
    """A possible condition hypothesis (NOT a diagnosis)."""
    condition: str
    icd10_code: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    supporting_factors: List[str]
    against_factors: List[str]
    recommended_workup: List[str]


class SafetyFlag(BaseModel):
    flag_type: str
    message: str
    severity: str  # "info" | "warning" | "critical"


class AnalysisResult(BaseModel):
    """Full analysis result returned to client."""
    model_config = ConfigDict(protected_namespaces=())
    
    result_id: UUID = Field(default_factory=uuid4)
    case_id: Optional[str] = None
    status: AnalysisStatus = AnalysisStatus.COMPLETED

    # Core output
    disclaimer: str = Field(
        default=(
            "⚠️ IMPORTANT: This is a research tool, NOT a medical diagnostic system. "
            "Results are AI-generated hypotheses for research purposes only. "
            "ALWAYS consult a qualified healthcare professional for medical decisions."
        )
    )
    summary: str
    condition_hypotheses: List[ConditionHypothesis]
    differential_reasoning: str
    evidence: List[MedicalEvidence]
    confidence_overall: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    safety_flags: List[SafetyFlag] = Field(default_factory=list)

    # Metadata
    model_used: str
    retrieval_count: int
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_analysis_summary: Optional[str] = None
    structured_data_used: bool = False


class ImageUploadResponse(BaseModel):
    image_id: str
    filename: str
    modality: ImageModality
    size_bytes: int
    embedding_status: str  # "pending" | "completed" | "failed"
    message: str


class AnalysisStatusResponse(BaseModel):
    result_id: str
    status: AnalysisStatus
    progress: Optional[int] = None  # 0-100
    message: Optional[str] = None


# -----------------------------------------------------------------------
# Auth Schemas
# -----------------------------------------------------------------------

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=200)


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
