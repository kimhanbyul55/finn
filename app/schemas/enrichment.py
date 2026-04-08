from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


JSONScalar = str | int | float | bool | None
JSONObject = dict[str, JSONScalar]
_DIRECT_TEXT_SENTINELS = frozenset(
    {
        "EMPTY",
        "N/A",
        "NA",
        "NONE",
        "NULL",
        "-",
    }
)


class SchemaModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class SentimentLabel(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class EnrichmentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class StageName(str, Enum):
    FETCH = "fetch"
    CLEAN = "clean"
    VALIDATE = "validate"
    SUMMARY_GENERATION = "summary_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    XAI_EXTRACTION = "xai_extraction"
    MIXED_SIGNAL_DETECTION = "mixed_signal_detection"
    BUILD_PAYLOAD = "build_payload"
    PERSIST = "persist"


class StageStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorDetail(SchemaModel):
    code: str = Field(..., description="Stable machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")
    retryable: bool = Field(
        default=False,
        description="Whether the operation may succeed on retry.",
    )
    details: JSONObject | None = Field(
        default=None,
        description="Optional structured metadata for debugging.",
    )


class ErrorCode(str, Enum):
    ARTICLE_FETCH_FAILED = "article_fetch_failed"
    ARTICLE_FETCH_RETRYABLE = "article_fetch_retryable"
    TEXT_CLEAN_FAILED = "text_clean_failed"
    ARTICLE_TEXT_INVALID = "article_text_invalid"
    SUMMARY_GENERATION_FAILED = "summary_generation_failed"
    SENTIMENT_ANALYSIS_FAILED = "sentiment_analysis_failed"
    XAI_EXTRACTION_FAILED = "xai_extraction_failed"
    MIXED_SIGNAL_DETECTION_FAILED = "mixed_signal_detection_failed"
    PAYLOAD_BUILD_FAILED = "payload_build_failed"
    RESULT_PERSIST_FAILED = "result_persist_failed"
    UNKNOWN_FAILURE = "unknown_failure"


class ArticleEnrichmentRequest(SchemaModel):
    news_id: str = Field(..., min_length=1, description="Unique news identifier.")
    title: str = Field(..., min_length=1, description="Original article title.")
    link: HttpUrl = Field(..., description="Canonical article URL.")
    ticker: list[str] | None = Field(
        default=None,
        description="Optional list of related ticker symbols.",
    )
    source: str | None = Field(default=None, description="Publisher or source name.")
    published_at: datetime | None = Field(
        default=None,
        description="Original publication timestamp.",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_tickers(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            symbol = item.strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                normalized.append(symbol)
        return normalized or None


def normalize_optional_text_input(value: object) -> object:
    """Convert placeholder direct-text values into missing values."""
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    normalized = value.strip()
    if not normalized:
        return None
    if normalized.upper() in _DIRECT_TEXT_SENTINELS:
        return None
    return normalized


class FlexibleTextEnrichmentRequest(ArticleEnrichmentRequest):
    article_text: str | None = Field(
        default=None,
        min_length=1,
        description="Licensed full article text supplied directly by the upstream provider.",
    )
    summary_text: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("summary_text", "text"),
        description="Licensed summary/snippet text supplied directly by the upstream provider.",
    )

    @field_validator("article_text", "summary_text", mode="before")
    @classmethod
    def normalize_direct_text_fields(cls, value: object) -> object:
        return normalize_optional_text_input(value)

    @property
    def has_direct_text(self) -> bool:
        return bool((self.article_text or "").strip() or (self.summary_text or "").strip())

    @property
    def resolved_direct_text(self) -> str | None:
        return (self.article_text or "").strip() or (self.summary_text or "").strip() or None


class DirectTextEnrichmentRequest(FlexibleTextEnrichmentRequest):

    @model_validator(mode="after")
    def validate_text_input(self) -> DirectTextEnrichmentRequest:
        if not self.has_direct_text:
            raise ValueError("Either article_text or summary_text must be provided.")
        return self


class SummaryLine(SchemaModel):
    line_number: Literal[1, 2, 3] = Field(..., description="1-based line index.")
    text: str = Field(..., min_length=1, description="Single summary line.")


class SentimentResult(SchemaModel):
    label: SentimentLabel = Field(..., description="Discrete sentiment label.")
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Signed sentiment score between -1 and 1.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence between 0 and 1.",
    )


class XAIHighlightItem(SchemaModel):
    excerpt: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("excerpt", "text"),
        description="Article excerpt used as evidence.",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("relevance_score", "relevance"),
        description="Relative importance of this highlight.",
    )
    explanation: str | None = Field(
        default=None,
        description="Why this excerpt matters for the result.",
    )
    sentiment_signal: SentimentLabel | None = Field(
        default=None,
        description="Optional sentiment signal tied to the highlight.",
    )
    start_char: int | None = Field(
        default=None,
        ge=0,
        description="Optional start offset within the article text.",
    )
    end_char: int | None = Field(
        default=None,
        ge=0,
        description="Optional end offset within the article text.",
    )

    @model_validator(mode="after")
    def validate_offsets(self) -> XAIHighlightItem:
        if self.start_char is not None and self.end_char is not None:
            if self.end_char < self.start_char:
                raise ValueError("end_char must be greater than or equal to start_char")
        return self


class XAIPayload(SchemaModel):
    explanation: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("explanation", "rationale"),
        description="High-level explanation of the enrichment result.",
    )
    highlights: list[XAIHighlightItem] = Field(
        default_factory=list,
        validation_alias=AliasChoices("highlights", "evidence"),
        description="Evidence items grounded in the article text.",
    )


class LocalizedArticleContent(SchemaModel):
    language: str = Field(..., min_length=2, description="Localized display language.")
    title: str = Field(..., min_length=1, description="Localized article title.")
    summary_3lines: list[SummaryLine] = Field(
        default_factory=list,
        max_length=3,
        description="Localized summary lines for display.",
    )
    xai: XAIPayload | None = Field(
        default=None,
        description="Localized XAI payload for display.",
    )
    sentiment_label: str | None = Field(
        default=None,
        description="Localized display label for the sentiment result.",
    )
    ticker_box_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Localized display labels for ticker-box style financial metrics.",
    )


class MixedConflictPayload(SchemaModel):
    is_mixed: bool = Field(..., description="Whether the article has mixed sentiment.")
    has_conflicting_signals: bool = Field(
        ...,
        description="Whether positive and negative evidence conflict materially.",
    )
    dominant_sentiment: SentimentLabel | None = Field(
        default=None,
        description="Dominant sentiment if one still exists.",
    )
    conflict_reasons: list[str] = Field(
        default_factory=list,
        description="Short explanations for mixed or conflicting signals.",
    )


class InternalStageStatus(SchemaModel):
    stage: StageName = Field(..., description="Pipeline stage name.")
    status: StageStatus = Field(..., description="Current stage execution status.")
    started_at: datetime | None = Field(
        default=None,
        description="When stage execution started.",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When stage execution completed.",
    )
    latency_ms: int | None = Field(
        default=None,
        ge=0,
        description="Execution time for the stage in milliseconds.",
    )
    error: ErrorDetail | None = Field(
        default=None,
        description="Stage-level failure details, if any.",
    )


class ArticleEnrichmentResponse(SchemaModel):
    news_id: str = Field(..., description="Unique news identifier.")
    title: str = Field(..., description="Original article title.")
    link: HttpUrl = Field(..., description="Canonical article URL.")
    summary_3lines: list[SummaryLine] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three grounded summary lines when available.",
    )
    sentiment: SentimentResult | None = Field(
        default=None,
        description="Article sentiment output when analysis succeeded.",
    )
    xai: XAIPayload | None = Field(
        default=None,
        description="Explainability payload when analysis succeeded.",
    )
    localized: LocalizedArticleContent | None = Field(
        default=None,
        description="Localized display payload for UI consumers.",
    )
    mixed_flags: MixedConflictPayload | None = Field(
        default=None,
        description="Mixed/conflict analysis output.",
    )
    status: EnrichmentStatus = Field(..., description="Overall pipeline status.")
    outcome: Literal["success", "partial_success", "fatal_failure"] = Field(
        ...,
        description="Top-level outcome separating success, partial success, and fatal failure.",
    )
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When analysis was completed or last updated.",
    )
    error: ErrorDetail | None = Field(
        default=None,
        description="Top-level pipeline error, if any.",
    )
    stage_statuses: list[InternalStageStatus] = Field(
        default_factory=list,
        description="Optional internal execution status per pipeline stage.",
    )

    @field_validator("summary_3lines", mode="before")
    @classmethod
    def normalize_summary_lines(
        cls,
        value: list[SummaryLine] | list[str] | list[dict[str, object]],
    ) -> list[SummaryLine] | list[dict[str, object]]:
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return [
                {"line_number": index, "text": text}
                for index, text in enumerate(value, start=1)
            ]
        return value

    @model_validator(mode="after")
    def validate_summary_sequence(self) -> ArticleEnrichmentResponse:
        if not self.summary_3lines:
            return self
        expected = [1, 2, 3]
        actual = [line.line_number for line in self.summary_3lines]
        if actual != expected:
            raise ValueError("summary_3lines must contain line_number values 1, 2, 3 in order")
        return self


# Backward-compatible aliases for existing imports.
SentimentPayload = SentimentResult
XAIEvidence = XAIHighlightItem
MixedFlags = MixedConflictPayload
