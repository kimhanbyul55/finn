from __future__ import annotations

from datetime import datetime, timezone
import logging

from app.core.logging import log_event
from app.schemas.article_fetch import ArticleFetchResult
from app.schemas.enrichment import (
    LocalizedArticleContent,
    SentimentLabel,
    SummaryLine,
    XAIHighlightItem,
    XAIPayload,
)
from app.schemas.mixed import (
    ArticleMixedDetectionResult,
    TickerMixedDetectionResult,
)
from app.schemas.sentiment import SentimentResult as FinBERTSentimentResult
from app.schemas.storage import (
    AnalysisOutcome,
    AnalysisStatus,
    EnrichmentStoragePayload,
    PipelineStageName,
    PipelineStageResult,
    StoragePayloadError,
    build_stored_sentiment_payload,
)
from app.schemas.xai import XAIContributionDirection
from app.schemas.xai import XAIResult
from app.core import get_settings
from app.services.translation import build_localized_content

logger = logging.getLogger(__name__)


def build_enrichment_storage_payload(
    *,
    news_id: str,
    title: str,
    link: str,
    analysis_status: AnalysisStatus,
    analysis_outcome: AnalysisOutcome,
    stage_statuses: list[PipelineStageResult],
    fetch_result: ArticleFetchResult | None = None,
    cleaned_text: str | None = None,
    summary_3lines: list[str] | None = None,
    sentiment_result: FinBERTSentimentResult | None = None,
    xai_result: XAIResult | None = None,
    article_mixed: ArticleMixedDetectionResult | None = None,
    ticker_mixed: TickerMixedDetectionResult | None = None,
    tickers: list[str] | None = None,
    analyzed_at: datetime | None = None,
    errors: list[StoragePayloadError] | None = None,
) -> EnrichmentStoragePayload:
    """Assemble a database-ready storage payload from enrichment stage outputs."""
    normalized_summary = _normalize_summary_lines(summary_3lines)
    stored_sentiment = (
        build_stored_sentiment_payload(sentiment_result)
        if sentiment_result is not None
        else None
    )
    localized = _build_stored_localized_content(
        title=title,
        summary_3lines=normalized_summary,
        sentiment_label=stored_sentiment.label if stored_sentiment is not None else None,
        xai_result=xai_result,
        is_mixed=bool(article_mixed and article_mixed.is_mixed),
        tickers=tickers,
        analysis_outcome=analysis_outcome,
    )

    aggregated_errors = list(errors or [])
    _append_payload_warnings(
        errors=aggregated_errors,
        analysis_outcome=analysis_outcome,
        normalized_summary=normalized_summary,
        localized=localized,
        sentiment_available=stored_sentiment is not None,
    )
    _log_localization_status(
        news_id=news_id,
        analysis_outcome=analysis_outcome,
        summary_line_count=len(normalized_summary),
        localized=localized,
    )

    return EnrichmentStoragePayload(
        news_id=news_id,
        title=title,
        link=link,
        summary_3lines=normalized_summary,
        sentiment=stored_sentiment,
        xai=xai_result,
        localized=localized,
        article_mixed=article_mixed,
        ticker_mixed=ticker_mixed,
        analysis_status=analysis_status,
        analysis_outcome=analysis_outcome,
        analyzed_at=_normalize_timestamp(analyzed_at),
        cleaned_text_available=bool((cleaned_text or "").strip()),
        fetch_result=fetch_result,
        stage_statuses=stage_statuses,
        errors=aggregated_errors,
    )


def _normalize_summary_lines(summary_3lines: list[str] | None) -> list[str]:
    if not summary_3lines:
        return []
    return [line.strip() for line in summary_3lines if line and line.strip()][:3]


def _build_stored_localized_content(
    *,
    title: str,
    summary_3lines: list[str],
    sentiment_label: str | None,
    xai_result: XAIResult | None,
    is_mixed: bool,
    tickers: list[str] | None,
    analysis_outcome: AnalysisOutcome,
) -> LocalizedArticleContent | None:
    if analysis_outcome == AnalysisOutcome.FILTERED:
        return None

    summary_lines = [
        SummaryLine(line_number=index, text=text)
        for index, text in enumerate(summary_3lines, start=1)
    ]
    localized_sentiment = _map_sentiment_label(sentiment_label, is_mixed=is_mixed)
    return build_localized_content(
        title=title,
        summary_3lines=summary_lines,
        xai=_build_localized_xai_payload(xai_result, localized_sentiment),
        sentiment_label=localized_sentiment,
        tickers=tickers,
        xai_highlight_limit=get_settings().localized_xai_highlight_limit,
        allow_gemini=True,
    )


def _build_localized_xai_payload(
    payload: XAIResult | None,
    sentiment_label: SentimentLabel | None,
) -> XAIPayload | None:
    if payload is None:
        return None

    explanation = "Top article snippets influencing the sentiment result."
    if sentiment_label is not None:
        explanation = f"Top article snippets influencing the {sentiment_label.value} sentiment result."

    return XAIPayload(
        explanation=explanation,
        highlights=[
            XAIHighlightItem(
                excerpt=highlight.text_snippet,
                relevance_score=min(1.0, max(0.0, highlight.importance_score)),
                explanation=None,
                sentiment_signal=_map_highlight_signal(
                    highlight.contribution_direction,
                    sentiment_label,
                ),
                start_char=highlight.start_char,
                end_char=highlight.end_char,
            )
            for highlight in payload.highlights
        ],
    )


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
    if direction == XAIContributionDirection.POSITIVE:
        return SentimentLabel.BULLISH
    if direction == XAIContributionDirection.NEGATIVE:
        return SentimentLabel.BEARISH
    return target_label


def _normalize_timestamp(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _append_payload_warnings(
    *,
    errors: list[StoragePayloadError],
    analysis_outcome: AnalysisOutcome,
    normalized_summary: list[str],
    localized: LocalizedArticleContent | None,
    sentiment_available: bool,
) -> None:
    if analysis_outcome == AnalysisOutcome.FILTERED:
        return

    if not normalized_summary:
        errors.append(
            StoragePayloadError(
                stage=PipelineStageName.SUMMARIZE,
                message="Summary generation returned no usable lines.",
                fatal=False,
            )
        )
    if localized is None:
        errors.append(
            StoragePayloadError(
                stage=PipelineStageName.BUILD_PAYLOAD,
                message="Localized Korean payload is empty.",
                fatal=False,
            )
        )
    elif not localized.summary_3lines and normalized_summary:
        errors.append(
            StoragePayloadError(
                stage=PipelineStageName.BUILD_PAYLOAD,
                message="Localized payload has title but no translated summary lines.",
                fatal=False,
            )
        )
    if not sentiment_available:
        errors.append(
            StoragePayloadError(
                stage=PipelineStageName.SENTIMENT,
                message="Sentiment result is missing.",
                fatal=False,
            )
        )


def _log_localization_status(
    *,
    news_id: str,
    analysis_outcome: AnalysisOutcome,
    summary_line_count: int,
    localized: LocalizedArticleContent | None,
) -> None:
    log_event(
        logger,
        logging.INFO,
        "payload_localization_status",
        news_id=news_id,
        analysis_outcome=analysis_outcome.value,
        summary_line_count=summary_line_count,
        localized_present=localized is not None,
        localized_summary_line_count=(
            len(localized.summary_3lines) if localized is not None else 0
        ),
    )
