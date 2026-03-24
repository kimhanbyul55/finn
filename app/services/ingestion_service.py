from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.repositories import EnrichmentRepository, create_repository
from app.schemas.ingestion import EnrichmentJobRecord
from app.schemas.ingestion import (
    DirectTextIngestionRequest,
    IngestionAcceptedResponse,
    NewsResultResponse,
    NewsProcessingStatusResponse,
    RawNewsIngestionRequest,
    WorkerProcessResponse,
)
from app.schemas.operations import OperationalStatsResponse
from app.schemas.storage import AnalysisOutcome, AnalysisStatus
from app.services.article_fetcher import FetchRetryPolicy
from app.services.enrichment_service import build_api_enrichment_response
from app.services.orchestrator import EnrichmentOrchestrator


class IngestionService:
    """Coordinate raw-news intake, queue management, and worker processing."""

    def __init__(
        self,
        repository: EnrichmentRepository | None = None,
        fetch_retry_policy: FetchRetryPolicy | None = None,
    ) -> None:
        self._repository = repository or create_repository()
        self._fetch_retry_policy = fetch_retry_policy or FetchRetryPolicy()
        self._orchestrator = EnrichmentOrchestrator(repository=self._repository)

    async def ingest_article(
        self,
        payload: RawNewsIngestionRequest,
    ) -> IngestionAcceptedResponse:
        return await self._ingest_payload(payload)

    async def ingest_article_text(
        self,
        payload: DirectTextIngestionRequest,
    ) -> IngestionAcceptedResponse:
        return await self._ingest_payload(payload)

    async def _ingest_payload(
        self,
        payload: RawNewsIngestionRequest,
    ) -> IngestionAcceptedResponse:
        self._repository.upsert_raw_news(payload)

        active_job = self._repository.get_active_job(payload.news_id)
        if active_job is not None:
            return IngestionAcceptedResponse(
                news_id=payload.news_id,
                queued=False,
                message="An enrichment job is already queued or processing for this article.",
                job=active_job,
            )

        job = self._repository.create_enrichment_job(payload.news_id)
        return IngestionAcceptedResponse(
            news_id=payload.news_id,
            queued=True,
                message="Raw news metadata saved and enrichment job queued.",
                job=job,
            )

    async def get_news_status(self, news_id: str) -> NewsProcessingStatusResponse | None:
        raw_news = self._repository.get_raw_news(news_id)
        latest_job = self._repository.get_latest_job(news_id)
        enrichment = self._repository.get_enrichment_result(news_id)

        if raw_news is None and latest_job is None and enrichment is None:
            return None

        return NewsProcessingStatusResponse(
            news_id=news_id,
            raw_news=raw_news,
            latest_job=latest_job,
            enrichment=enrichment,
        )

    async def get_news_result(self, news_id: str) -> NewsResultResponse | None:
        raw_news = self._repository.get_raw_news(news_id)
        latest_job = self._repository.get_latest_job(news_id)
        enrichment = self._repository.get_enrichment_result(news_id)

        if raw_news is None and latest_job is None and enrichment is None:
            return None

        return NewsResultResponse(
            news_id=news_id,
            raw_news=raw_news,
            latest_job=latest_job,
            result=build_api_enrichment_response(enrichment) if enrichment is not None else None,
        )

    async def get_operational_stats(self) -> OperationalStatsResponse:
        return self._repository.get_operational_stats()

    def process_next_job(self) -> WorkerProcessResponse:
        job = self._repository.claim_next_enrichment_job()
        if job is None:
            return WorkerProcessResponse(
                processed=False,
                retry_scheduled=False,
                message="No queued enrichment job was available.",
            )

        raw_news = self._repository.get_raw_news(job.news_id)
        if raw_news is None:
            failed_job = self._repository.mark_job_failed(
                job.job_id,
                error_message="Raw news metadata was missing for the claimed job.",
            )
            return WorkerProcessResponse(
                processed=True,
                retry_scheduled=False,
                message="Claimed job failed because raw news metadata was missing.",
                news_id=job.news_id,
                job=failed_job,
            )

        if isinstance(raw_news, DirectTextIngestionRequest):
            enrichment = self._orchestrator.run_with_text(
                raw_news,
                article_text=raw_news.article_text,
                summary_text=raw_news.summary_text,
            )
        else:
            enrichment = self._orchestrator.run(raw_news)

        if enrichment.analysis_outcome == AnalysisOutcome.FATAL_FAILURE:
            if self._should_retry_job(job=job, analysis_status=enrichment.analysis_status, enrichment=enrichment):
                updated_job = self._repository.requeue_job(
                    job.job_id,
                    error_message=(
                        f"Retry scheduled after transient failure: {enrichment.analysis_status.value}"
                    ),
                    next_retry_at=self._next_retry_at(job),
                    analysis_status=enrichment.analysis_status,
                )
                return WorkerProcessResponse(
                    processed=True,
                    retry_scheduled=True,
                    message="Processed one enrichment job; retry scheduled.",
                    news_id=job.news_id,
                    job=updated_job,
                    analysis_status=enrichment.analysis_status,
                    analysis_outcome=enrichment.analysis_outcome,
                    enrichment=enrichment,
                )

            updated_job = self._repository.mark_job_failed(
                job.job_id,
                error_message=(
                    f"Enrichment ended with fatal outcome: {enrichment.analysis_status.value}"
                ),
                analysis_status=enrichment.analysis_status,
            )
        else:
            updated_job = self._repository.mark_job_completed(
                job.job_id,
                analysis_status=enrichment.analysis_status,
            )

        if isinstance(raw_news, DirectTextIngestionRequest):
            self._repository.clear_raw_news_text_inputs(raw_news.news_id)

        return WorkerProcessResponse(
            processed=True,
            retry_scheduled=False,
            message="Processed one enrichment job.",
            news_id=job.news_id,
            job=updated_job,
            analysis_status=enrichment.analysis_status,
            analysis_outcome=enrichment.analysis_outcome,
            enrichment=enrichment,
        )

    def _should_retry_job(
        self,
        *,
        job: EnrichmentJobRecord,
        analysis_status: AnalysisStatus,
        enrichment,
    ) -> bool:
        if job.attempts >= job.max_attempts:
            return False
        if analysis_status != AnalysisStatus.FETCH_FAILED:
            return False
        if enrichment.fetch_result is None:
            return False
        return bool(enrichment.fetch_result.retryable)

    def _next_retry_at(self, job: EnrichmentJobRecord) -> datetime:
        attempt_index = max(job.attempts - 1, 0)
        delay_seconds = self._fetch_retry_policy.backoff_seconds(attempt_index)
        return datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
