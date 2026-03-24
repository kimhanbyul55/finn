from app.repositories import EnrichmentRepository, create_repository
from app.schemas.enrichment import (
    ArticleEnrichmentRequest,
    ArticleEnrichmentResponse,
    DirectTextEnrichmentRequest,
    EnrichmentStatus,
    ErrorDetail,
    InternalStageStatus,
    MixedConflictPayload,
    SentimentLabel,
    SentimentResult,
    StageName,
    StageStatus,
    SummaryLine,
    XAIHighlightItem,
    XAIPayload,
)
from app.schemas.mixed import ArticleMixedDetectionResult
from app.schemas.storage import AnalysisOutcome, AnalysisStatus, EnrichmentStoragePayload
from app.schemas.xai import XAIContributionDirection, XAIResult
from app.services.orchestrator import EnrichmentOrchestrator


class EnrichmentService:
    def __init__(
        self,
        repository: EnrichmentRepository | None = None,
    ) -> None:
        self._repository = repository or create_repository()
        self._orchestrator = EnrichmentOrchestrator(repository=self._repository)

    async def enrich_article(
        self,
        payload: ArticleEnrichmentRequest,
    ) -> ArticleEnrichmentResponse:
        storage_payload = self._orchestrator.run(payload)
        return build_api_enrichment_response(storage_payload)

    async def enrich_article_text(
        self,
        payload: DirectTextEnrichmentRequest,
    ) -> ArticleEnrichmentResponse:
        storage_payload = self._orchestrator.run_with_text(
            payload,
            article_text=payload.article_text,
            summary_text=payload.summary_text,
        )
        return build_api_enrichment_response(storage_payload)


def build_api_enrichment_response(
    payload: EnrichmentStoragePayload,
) -> ArticleEnrichmentResponse:
    mixed_result = payload.article_mixed
    api_sentiment = _build_sentiment_payload(payload, mixed_result)

    return ArticleEnrichmentResponse(
        news_id=payload.news_id,
        title=payload.title,
        link=payload.link,
        summary_3lines=[
            SummaryLine(line_number=index, text=text)
            for index, text in enumerate(payload.summary_3lines, start=1)
        ],
        sentiment=api_sentiment,
        xai=_build_xai_payload(payload.xai, api_sentiment),
        mixed_flags=_build_mixed_flags(mixed_result),
        status=_map_overall_status(payload.analysis_status, payload.analysis_outcome),
        outcome=payload.analysis_outcome.value,
        analyzed_at=payload.analyzed_at,
        error=_build_error_detail(payload),
        stage_statuses=[_map_stage_status(stage) for stage in payload.stage_statuses],
    )


def _build_sentiment_payload(
    payload: EnrichmentStoragePayload,
    mixed_result: ArticleMixedDetectionResult | None,
) -> SentimentResult | None:
    if payload.sentiment is None:
        return None

    label = _map_sentiment_label(
        payload.sentiment.label,
        is_mixed=bool(mixed_result and mixed_result.is_mixed),
    )
    normalized_score = max(-1.0, min(1.0, round(payload.sentiment.score / 100.0, 4)))

    return SentimentResult(
        label=label,
        score=normalized_score,
        confidence=payload.sentiment.confidence,
    )


def _build_xai_payload(
    payload: XAIResult | None,
    sentiment: SentimentResult | None,
) -> XAIPayload | None:
    if payload is None:
        return None

    target_label = sentiment.label if sentiment is not None else None
    explanation = "Top article snippets influencing the sentiment result."
    if target_label is not None:
        explanation = (
            f"Top article snippets influencing the {target_label.value} sentiment result."
        )

    return XAIPayload(
        explanation=explanation,
        highlights=[
            XAIHighlightItem(
                excerpt=highlight.text_snippet,
                relevance_score=min(1.0, max(0.0, highlight.importance_score)),
                explanation=None,
                sentiment_signal=_map_highlight_signal(
                    highlight.contribution_direction,
                    target_label,
                ),
                start_char=highlight.start_char,
                end_char=highlight.end_char,
            )
            for highlight in payload.highlights
        ],
    )


def _build_mixed_flags(
    payload: ArticleMixedDetectionResult | None,
) -> MixedConflictPayload | None:
    if payload is None:
        return None

    return MixedConflictPayload(
        is_mixed=payload.is_mixed,
        has_conflicting_signals=payload.has_conflicting_signals,
        dominant_sentiment=_map_sentiment_label(
            payload.dominant_sentiment.value if payload.dominant_sentiment is not None else None,
            is_mixed=payload.is_mixed,
        ),
        conflict_reasons=[reason.message for reason in payload.reasons if reason.triggered],
    )


def _build_error_detail(
    payload: EnrichmentStoragePayload,
) -> ErrorDetail | None:
    if not payload.errors:
        return None

    first_error = payload.errors[0]
    retryable = False
    if payload.fetch_result is not None and payload.analysis_status == AnalysisStatus.FETCH_FAILED:
        retryable = payload.fetch_result.retryable

    return ErrorDetail(
        code=payload.analysis_status.value,
        message=first_error.message,
        retryable=retryable,
        details={
            "stage": first_error.stage.value,
            "analysis_outcome": payload.analysis_outcome.value,
        },
    )


def _map_stage_status(stage: object) -> InternalStageStatus:
    stage_name_map = {
        "fetch": StageName.FETCH,
        "clean": StageName.CLEAN,
        "validate": StageName.VALIDATE,
        "summarize": StageName.SUMMARY_GENERATION,
        "sentiment": StageName.SENTIMENT_ANALYSIS,
        "xai": StageName.XAI_EXTRACTION,
        "mixed_detection": StageName.MIXED_SIGNAL_DETECTION,
        "build_payload": StageName.BUILD_PAYLOAD,
        "persist": StageName.PERSIST,
    }
    stage_status_map = {
        "pending": StageStatus.NOT_STARTED,
        "completed": StageStatus.COMPLETED,
        "failed": StageStatus.FAILED,
        "skipped": StageStatus.SKIPPED,
    }

    stage_value = getattr(stage, "stage").value
    status_value = getattr(stage, "status").value

    return InternalStageStatus(
        stage=stage_name_map[stage_value],
        status=stage_status_map[status_value],
        started_at=getattr(stage, "started_at"),
        completed_at=getattr(stage, "completed_at"),
        error=None,
    )


def _map_overall_status(
    analysis_status: AnalysisStatus,
    analysis_outcome: AnalysisOutcome,
) -> EnrichmentStatus:
    if analysis_status == AnalysisStatus.PENDING:
        return EnrichmentStatus.PENDING
    if analysis_outcome == AnalysisOutcome.SUCCESS:
        return EnrichmentStatus.COMPLETED
    if analysis_outcome == AnalysisOutcome.PARTIAL_SUCCESS:
        return EnrichmentStatus.PARTIAL
    return EnrichmentStatus.FAILED


def _map_sentiment_label(
    label: str | None,
    *,
    is_mixed: bool,
) -> SentimentLabel | None:
    if label is None:
        return None
    if is_mixed:
        return SentimentLabel.MIXED
    label_map = {
        "positive": SentimentLabel.BULLISH,
        "negative": SentimentLabel.BEARISH,
        "neutral": SentimentLabel.NEUTRAL,
    }
    return label_map.get(label, SentimentLabel.NEUTRAL)


def _map_highlight_signal(
    direction: XAIContributionDirection,
    target_label: SentimentLabel | None,
) -> SentimentLabel | None:
    if target_label is None:
        return None
    if direction == XAIContributionDirection.POSITIVE:
        return target_label
    opposite_map = {
        SentimentLabel.BULLISH: SentimentLabel.BEARISH,
        SentimentLabel.BEARISH: SentimentLabel.BULLISH,
        SentimentLabel.NEUTRAL: SentimentLabel.NEUTRAL,
        SentimentLabel.MIXED: SentimentLabel.MIXED,
    }
    return opposite_map[target_label]
