from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from app.schemas.enrichment import (
    ArticleEnrichmentRequest,
    ArticleEnrichmentResponse,
    DirectTextEnrichmentRequest,
    SchemaModel,
)
from app.schemas.storage import AnalysisOutcome, AnalysisStatus, EnrichmentStoragePayload


class EnrichmentJobStatus(str, Enum):
    QUEUED = "queued"
    RETRY_PENDING = "retry_pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RawNewsIngestionRequest(ArticleEnrichmentRequest):
    """Inbound raw-news payload sent by the upstream backend/news ingestion system."""


class DirectTextIngestionRequest(DirectTextEnrichmentRequest):
    """Inbound raw-news payload with licensed text supplied directly by the upstream backend."""


class EnrichmentJobRecord(SchemaModel):
    job_id: str = Field(..., min_length=1)
    news_id: str = Field(..., min_length=1)
    status: EnrichmentJobStatus = Field(..., description="Queue processing status.")
    attempts: int = Field(..., ge=0)
    max_attempts: int = Field(..., ge=1)
    last_error: str | None = Field(default=None)
    last_analysis_status: AnalysisStatus | None = Field(default=None)
    created_at: datetime = Field(...)
    updated_at: datetime = Field(...)
    next_retry_at: datetime | None = Field(default=None)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)


class IngestionAcceptedResponse(SchemaModel):
    news_id: str = Field(..., min_length=1)
    queued: bool = Field(..., description="Whether a new job was enqueued.")
    message: str = Field(..., min_length=1)
    job: EnrichmentJobRecord = Field(..., description="Active or newly created job record.")


class NewsProcessingStatusResponse(SchemaModel):
    news_id: str = Field(..., min_length=1)
    raw_news: ArticleEnrichmentRequest | None = Field(default=None)
    latest_job: EnrichmentJobRecord | None = Field(default=None)
    enrichment: EnrichmentStoragePayload | None = Field(default=None)


class NewsResultResponse(SchemaModel):
    news_id: str = Field(..., min_length=1)
    raw_news: ArticleEnrichmentRequest | None = Field(default=None)
    latest_job: EnrichmentJobRecord | None = Field(default=None)
    result: ArticleEnrichmentResponse | None = Field(default=None)


class WorkerProcessResponse(SchemaModel):
    processed: bool = Field(..., description="Whether the worker processed a queued job.")
    retry_scheduled: bool = Field(
        default=False,
        description="Whether the processed job was put back into the queue for retry.",
    )
    message: str = Field(..., min_length=1)
    news_id: str | None = Field(default=None)
    job: EnrichmentJobRecord | None = Field(default=None)
    analysis_status: AnalysisStatus | None = Field(default=None)
    analysis_outcome: AnalysisOutcome | None = Field(default=None)
    enrichment: EnrichmentStoragePayload | None = Field(default=None)
