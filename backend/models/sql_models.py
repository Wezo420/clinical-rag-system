"""
SQLAlchemy ORM models for PostgreSQL.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    cases = relationship("ClinicalCase", back_populates="user")
    images = relationship("MedicalImage", back_populates="user")


class ClinicalCase(Base):
    __tablename__ = "clinical_cases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    patient_id = Column(String(100), nullable=True)
    clinical_text = Column(Text, nullable=False)
    structured_data = Column(JSON, nullable=True)
    status = Column(String(20), default="pending", nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="cases")
    result = relationship("AnalysisResult", back_populates="case", uselist=False)
    images = relationship("CaseMedicalImage", back_populates="case")


class MedicalImage(Base):
    __tablename__ = "medical_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    storage_path = Column(String(512), nullable=False)
    modality = Column(String(50), default="other")
    size_bytes = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    embedding_status = Column(String(20), default="pending")
    embedding_path = Column(String(512), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="images")


class CaseMedicalImage(Base):
    __tablename__ = "case_medical_images"

    case_id = Column(UUID(as_uuid=True), ForeignKey("clinical_cases.id"), primary_key=True)
    image_id = Column(UUID(as_uuid=True), ForeignKey("medical_images.id"), primary_key=True)

    case = relationship("ClinicalCase", back_populates="images")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("clinical_cases.id"), nullable=False, unique=True)
    status = Column(String(20), default="completed", nullable=False)
    result_json = Column(JSON, nullable=False)
    model_used = Column(String(100), nullable=False)
    confidence_overall = Column(Float, nullable=True)
    retrieval_count = Column(Integer, default=0)
    processing_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("ClinicalCase", back_populates="result")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(100), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    ip_address = Column(String(45), nullable=True)
    resource_id = Column(String(100), nullable=True)
    event_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
