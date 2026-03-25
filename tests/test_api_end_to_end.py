from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace

from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.api.routes.enrichment as enrichment_route_module
import app.api.routes.ingestion as ingestion_route_module
from app.main import app
from app.repositories import InMemoryEnrichmentRepository, SaveEnrichmentRequest
from app.schemas.article_fetch import ArticleFetchResult, ArticleFetchStatus, ArticleTextSource
from app.schemas.enrichment import (
    ArticleEnrichmentRequest,
    DirectTextEnrichmentRequest,
    FlexibleTextEnrichmentRequest,
)
from app.schemas.sentiment import (
    AggregationStrategy,
    ChunkSentimentResult,
    FinBERTSentimentLabel,
    SentimentChunkSource,
    SentimentProbabilities,
    SentimentResult,
)
from app.schemas.storage import (
    AnalysisOutcome,
    AnalysisStatus,
    EnrichmentStoragePayload,
    build_stored_sentiment_payload,
)
from app.schemas.xai import XAIContributionDirection, XAIHighlight, XAIResult
from app.services.enrichment_service import build_api_enrichment_response
from app.services.ingestion_service import IngestionService


def _build_completed_payload(request: ArticleEnrichmentRequest) -> EnrichmentStoragePayload:
    now = datetime.now(timezone.utc)
    return EnrichmentStoragePayload(
        news_id=request.news_id,
        title=request.title,
        link=str(request.link),
        summary_3lines=[
            "Revenue growth stayed ahead of expectations.",
            "Management highlighted stable demand and improved margins.",
            "Investors are watching whether guidance remains intact.",
        ],
        sentiment=build_stored_sentiment_payload(
            SentimentResult(
                label=FinBERTSentimentLabel.POSITIVE,
                score=61.0,
                confidence=0.86,
                probabilities=SentimentProbabilities(
                    positive=0.81,
                    neutral=0.14,
                    negative=0.05,
                ),
                aggregation_strategy=AggregationStrategy.WEIGHTED_MEAN,
                chunk_results=[
                    ChunkSentimentResult(
                        chunk_index=0,
                        source=SentimentChunkSource.BODY,
                        text="Revenue growth stayed ahead of expectations.",
                        token_count=12,
                        weight=1.0,
                        label=FinBERTSentimentLabel.POSITIVE,
                        score=61.0,
                        confidence=0.86,
                        probabilities=SentimentProbabilities(
                            positive=0.81,
                            neutral=0.14,
                            negative=0.05,
                        ),
                    )
                ],
                disagreement_ratio=0.0,
                chunk_count=1,
            )
        ),
        xai=XAIResult(
            target_label=FinBERTSentimentLabel.POSITIVE,
            highlights=[
                XAIHighlight(
                    text_snippet="Revenue growth stayed ahead of expectations.",
                    weight=0.125,
                    importance_score=0.125,
                    contribution_direction=XAIContributionDirection.POSITIVE,
                    sentence_index=0,
                    start_char=0,
                    end_char=41,
                    keyword_spans=[],
                )
            ],
            limitations=[
                "Sentence-level explanation only.",
            ],
            sentence_count=1,
            truncated=False,
        ),
        article_mixed=None,
        ticker_mixed=None,
        analysis_status=AnalysisStatus.COMPLETED,
        analysis_outcome=AnalysisOutcome.SUCCESS,
        analyzed_at=now,
        cleaned_text_available=True,
        fetch_result=ArticleFetchResult(
            link=str(request.link),
            publisher_domain="example.com",
            final_url=str(request.link),
            content_type="text/html; charset=utf-8",
            extraction_source=ArticleTextSource.PARAGRAPH_BLOCKS,
            attempt_count=1,
            raw_text="Revenue growth stayed ahead of expectations.",
            cleaned_text="Revenue growth stayed ahead of expectations.",
            fetch_status=ArticleFetchStatus.SUCCESS,
            retryable=False,
            failure_category=None,
            error_message=None,
        ),
        stage_statuses=[],
        errors=[],
    )


def test_news_intake_worker_and_status_flow(monkeypatch) -> None:
    repository = InMemoryEnrichmentRepository()
    service = IngestionService(repository=repository)

    def _run_and_persist(raw_news: ArticleEnrichmentRequest) -> EnrichmentStoragePayload:
        payload = _build_completed_payload(raw_news)
        repository.save_enrichment_result(
            SaveEnrichmentRequest(raw_news=raw_news, enrichment=payload)
        )
        return payload

    monkeypatch.setattr(ingestion_route_module, "service", service)
    monkeypatch.setattr(service._orchestrator, "run", _run_and_persist)

    client = TestClient(app)

    intake_response = client.post(
        "/api/v1/news/intake",
        json={
            "news_id": "e2e-news-1",
            "title": "Company beats earnings estimates",
            "link": "https://example.com/articles/e2e-news-1",
            "ticker": ["AAPL"],
            "source": "Reuters",
        },
    )

    assert intake_response.status_code == 200
    intake_payload = intake_response.json()
    assert intake_payload["queued"] is True
    assert intake_payload["job"]["status"] == "queued"

    worker_response = client.post("/api/v1/jobs/process-next")

    assert worker_response.status_code == 200
    worker_payload = worker_response.json()
    assert worker_payload["processed"] is True
    assert worker_payload["retry_scheduled"] is False
    assert worker_payload["analysis_status"] == "completed"
    assert worker_payload["analysis_outcome"] == "success"
    assert worker_payload["job"]["status"] == "completed"

    status_response = client.get("/api/v1/news/e2e-news-1")
    result_response = client.get("/api/v1/news/e2e-news-1/result")

    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["raw_news"]["news_id"] == "e2e-news-1"
    assert status_payload["latest_job"]["status"] == "completed"
    assert status_payload["enrichment"]["analysis_status"] == "completed"
    assert len(status_payload["enrichment"]["summary_3lines"]) == 3
    assert status_payload["enrichment"]["sentiment"]["label"] == "positive"
    assert (
        status_payload["enrichment"]["fetch_result"]["extraction_source"]
        == "paragraph_blocks"
    )
    assert status_payload["enrichment"]["xai"]["highlights"][0]["start_char"] == 0
    assert status_payload["enrichment"]["xai"]["highlights"][0]["end_char"] == 41
    assert "keyword_spans" in status_payload["enrichment"]["xai"]["highlights"][0]

    assert result_response.status_code == 200
    result_payload = result_response.json()
    assert result_payload["raw_news"]["news_id"] == "e2e-news-1"
    assert result_payload["latest_job"]["status"] == "completed"
    assert result_payload["result"]["status"] == "completed"
    assert result_payload["result"]["outcome"] == "success"
    assert result_payload["result"]["sentiment"]["label"] == "bullish"
    assert result_payload["result"]["xai"]["highlights"][0]["excerpt"] == (
        "Revenue growth stayed ahead of expectations."
    )


def test_enrich_endpoint_returns_external_api_shape(monkeypatch) -> None:
    repository = InMemoryEnrichmentRepository()
    request = ArticleEnrichmentRequest(
        news_id="enrich-news-1",
        title="Company beats earnings estimates",
        link="https://example.com/articles/enrich-news-1",
        ticker=["AAPL"],
        source="Reuters",
    )
    payload = _build_completed_payload(request)

    async def _enrich_article(_: ArticleEnrichmentRequest):
        return build_api_enrichment_response(payload)

    monkeypatch.setattr(
        enrichment_route_module.service,
        "enrich_article",
        _enrich_article,
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/articles/enrich",
        json={
            "news_id": "enrich-news-1",
            "title": "Company beats earnings estimates",
            "link": "https://example.com/articles/enrich-news-1",
            "ticker": ["AAPL"],
            "source": "Reuters",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["outcome"] == "success"
    assert len(body["summary_3lines"]) == 3
    assert body["summary_3lines"][0]["line_number"] == 1
    assert body["sentiment"]["label"] == "bullish"
    assert body["sentiment"]["score"] == 0.61
    assert body["xai"]["highlights"][0]["excerpt"] == "Revenue growth stayed ahead of expectations."
    assert body["xai"]["highlights"][0]["relevance_score"] == 0.125
    assert body["xai"]["highlights"][0]["start_char"] == 0
    assert body["xai"]["highlights"][0]["end_char"] == 41
    assert body["error"] is None


def test_enrich_endpoint_accepts_text_alias_and_uses_direct_text(monkeypatch) -> None:
    request = FlexibleTextEnrichmentRequest(
        news_id="enrich-news-text-1",
        title="Company beats earnings estimates",
        link="https://example.com/articles/enrich-news-text-1",
        ticker=["AAPL"],
        source="Licensed Provider",
        summary_text="Revenue growth stayed ahead of expectations.",
    )
    payload = _build_completed_payload(request)
    payload.fetch_result.extraction_source = ArticleTextSource.PROVIDED_SUMMARY_TEXT

    async def _enrich_article(_: FlexibleTextEnrichmentRequest):
        return build_api_enrichment_response(payload)

    monkeypatch.setattr(
        enrichment_route_module.service,
        "enrich_article",
        _enrich_article,
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/articles/enrich",
        json={
            "news_id": "enrich-news-text-1",
            "title": "Company beats earnings estimates",
            "link": "https://example.com/articles/enrich-news-text-1",
            "ticker": ["AAPL"],
            "source": "Licensed Provider",
            "text": "Revenue growth stayed ahead of expectations.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["outcome"] == "success"
    assert body["sentiment"]["label"] == "bullish"


def test_news_intake_text_worker_and_status_flow(monkeypatch) -> None:
    repository = InMemoryEnrichmentRepository()
    service = IngestionService(repository=repository)

    def _run_with_text_and_persist(
        raw_news: ArticleEnrichmentRequest,
        *,
        article_text: str | None = None,
        summary_text: str | None = None,
    ) -> EnrichmentStoragePayload:
        payload = _build_completed_payload(raw_news)
        payload.fetch_result.extraction_source = ArticleTextSource.PROVIDED_SUMMARY_TEXT
        repository.save_enrichment_result(
            SaveEnrichmentRequest(raw_news=raw_news, enrichment=payload)
        )
        return payload

    monkeypatch.setattr(ingestion_route_module, "service", service)
    monkeypatch.setattr(service._orchestrator, "run_with_text", _run_with_text_and_persist)

    client = TestClient(app)

    intake_response = client.post(
        "/api/v1/news/intake-text",
        json={
            "news_id": "e2e-news-text-1",
            "title": "Company beats earnings estimates",
            "link": "https://example.com/articles/e2e-news-text-1",
            "ticker": ["AAPL"],
            "source": "Licensed Provider",
            "summary_text": (
                "Revenue growth stayed ahead of expectations. "
                "Management highlighted stable demand and improved margins."
            ),
        },
    )

    assert intake_response.status_code == 200
    intake_payload = intake_response.json()
    assert intake_payload["queued"] is True
    assert intake_payload["job"]["status"] == "queued"

    worker_response = client.post("/api/v1/jobs/process-next")

    assert worker_response.status_code == 200
    worker_payload = worker_response.json()
    assert worker_payload["processed"] is True
    assert worker_payload["retry_scheduled"] is False
    assert worker_payload["analysis_status"] == "completed"
    assert worker_payload["analysis_outcome"] == "success"
    assert (
        worker_payload["enrichment"]["fetch_result"]["extraction_source"]
        == "provided_summary_text"
    )

    status_response = client.get("/api/v1/news/e2e-news-text-1")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["raw_news"]["news_id"] == "e2e-news-text-1"
    assert "summary_text" not in status_payload["raw_news"]

    stored_raw_news = repository.get_raw_news("e2e-news-text-1")
    assert stored_raw_news is not None
    assert not hasattr(stored_raw_news, "summary_text")


def test_news_intake_accepts_text_alias_and_queues_direct_text(monkeypatch) -> None:
    repository = InMemoryEnrichmentRepository()
    service = IngestionService(repository=repository)

    def _run_with_text_and_persist(
        raw_news: ArticleEnrichmentRequest,
        *,
        article_text: str | None = None,
        summary_text: str | None = None,
    ) -> EnrichmentStoragePayload:
        payload = _build_completed_payload(raw_news)
        payload.fetch_result.extraction_source = ArticleTextSource.PROVIDED_SUMMARY_TEXT
        repository.save_enrichment_result(
            SaveEnrichmentRequest(raw_news=raw_news, enrichment=payload)
        )
        return payload

    monkeypatch.setattr(ingestion_route_module, "service", service)
    monkeypatch.setattr(service._orchestrator, "run_with_text", _run_with_text_and_persist)

    client = TestClient(app)
    intake_response = client.post(
        "/api/v1/news/intake",
        json={
            "news_id": "e2e-news-text-legacy-1",
            "title": "Company beats earnings estimates",
            "link": "https://example.com/articles/e2e-news-text-legacy-1",
            "ticker": ["AAPL"],
            "source": "Licensed Provider",
            "text": "Revenue growth stayed ahead of expectations.",
        },
    )

    assert intake_response.status_code == 200
    assert intake_response.json()["queued"] is True

    worker_response = client.post("/api/v1/jobs/process-next")
    assert worker_response.status_code == 200
    assert (
        worker_response.json()["enrichment"]["fetch_result"]["extraction_source"]
        == "provided_summary_text"
    )


def test_direct_text_request_rejects_empty_placeholder_summary() -> None:
    try:
        DirectTextEnrichmentRequest(
            news_id="direct-text-empty-1",
            title="Placeholder summary",
            link="https://example.com/articles/direct-text-empty-1",
            ticker=["AAPL"],
            source="Licensed Provider",
            summary_text="EMPTY",
        )
    except ValidationError as exc:
        assert "Either article_text or summary_text must be provided." in str(exc)
    else:
        raise AssertionError("Expected EMPTY summary placeholder to be rejected.")


def test_news_intake_treats_empty_placeholder_as_missing_text(monkeypatch) -> None:
    repository = InMemoryEnrichmentRepository()
    service = IngestionService(repository=repository)

    def _run_and_persist(raw_news: ArticleEnrichmentRequest) -> EnrichmentStoragePayload:
        payload = _build_completed_payload(raw_news)
        repository.save_enrichment_result(
            SaveEnrichmentRequest(raw_news=raw_news, enrichment=payload)
        )
        return payload

    def _run_with_text_should_not_execute(*args, **kwargs):
        raise AssertionError("summary_text='EMPTY' should not be treated as direct text.")

    monkeypatch.setattr(ingestion_route_module, "service", service)
    monkeypatch.setattr(service._orchestrator, "run", _run_and_persist)
    monkeypatch.setattr(service._orchestrator, "run_with_text", _run_with_text_should_not_execute)

    client = TestClient(app)
    intake_response = client.post(
        "/api/v1/news/intake",
        json={
            "news_id": "e2e-news-empty-placeholder-1",
            "title": "Placeholder summary should crawl",
            "link": "https://example.com/articles/e2e-news-empty-placeholder-1",
            "ticker": ["AAPL"],
            "source": "Licensed Provider",
            "summary_text": "EMPTY",
        },
    )

    assert intake_response.status_code == 200
    assert intake_response.json()["queued"] is True

    worker_response = client.post("/api/v1/jobs/process-next")
    assert worker_response.status_code == 200
    worker_payload = worker_response.json()
    assert worker_payload["processed"] is True
    assert worker_payload["analysis_status"] == "completed"
    assert (
        worker_payload["enrichment"]["fetch_result"]["extraction_source"]
        == "paragraph_blocks"
    )


def test_enrich_text_endpoint_skips_remote_fetch(monkeypatch) -> None:
    repository = InMemoryEnrichmentRepository()
    request = DirectTextEnrichmentRequest(
        news_id="direct-text-news-1",
        title="Company beats earnings estimates",
        link="https://example.com/articles/direct-text-news-1",
        ticker=["AAPL"],
        source="Reuters",
        article_text=(
            "Revenue growth stayed ahead of expectations. "
            "Management highlighted stable demand and improved margins. "
            "Investors are watching whether guidance remains intact."
        ),
    )
    payload = _build_completed_payload(request)
    payload.fetch_result.extraction_source = ArticleTextSource.PROVIDED_ARTICLE_TEXT

    async def _enrich_article_text(_: DirectTextEnrichmentRequest):
        return build_api_enrichment_response(payload)

    monkeypatch.setattr(
        enrichment_route_module.service,
        "enrich_article_text",
        _enrich_article_text,
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/articles/enrich-text",
        json={
            "news_id": "direct-text-news-1",
            "title": "Company beats earnings estimates",
            "link": "https://example.com/articles/direct-text-news-1",
            "ticker": ["AAPL"],
            "source": "Reuters",
            "article_text": (
                "Revenue growth stayed ahead of expectations. "
                "Management highlighted stable demand and improved margins. "
                "Investors are watching whether guidance remains intact."
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["outcome"] == "success"
    assert body["sentiment"]["label"] == "bullish"


def test_direct_enrichment_endpoints_are_disabled_on_render_web(monkeypatch) -> None:
    monkeypatch.setattr(
        enrichment_route_module,
        "settings",
        replace(enrichment_route_module.settings, enable_direct_enrichment_api=False),
    )

    client = TestClient(app)

    enrich_response = client.post(
        "/api/v1/articles/enrich",
        json={
            "news_id": "render-disabled-enrich-1",
            "title": "Direct enrich disabled on render",
            "link": "https://example.com/articles/render-disabled-enrich-1",
            "ticker": ["AAPL"],
            "source": "Reuters",
        },
    )
    enrich_text_response = client.post(
        "/api/v1/articles/enrich-text",
        json={
            "news_id": "render-disabled-enrich-text-1",
            "title": "Direct enrich-text disabled on render",
            "link": "https://example.com/articles/render-disabled-enrich-text-1",
            "ticker": ["AAPL"],
            "source": "Reuters",
            "article_text": "Revenue growth stayed ahead of expectations.",
        },
    )

    assert enrich_response.status_code == 503
    assert "Direct enrichment APIs are disabled" in enrich_response.json()["detail"]
    assert enrich_text_response.status_code == 503
    assert "Direct enrichment APIs are disabled" in enrich_text_response.json()["detail"]


def test_process_next_endpoint_is_disabled_on_render_web(monkeypatch) -> None:
    monkeypatch.setattr(
        ingestion_route_module,
        "settings",
        replace(ingestion_route_module.settings, enable_job_process_api=False),
    )

    client = TestClient(app)
    response = client.post("/api/v1/jobs/process-next")

    assert response.status_code == 503
    assert "Job processing API is disabled" in response.json()["detail"]
