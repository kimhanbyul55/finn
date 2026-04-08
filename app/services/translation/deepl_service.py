from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import requests

from app.core import get_settings
from app.schemas.enrichment import (
    LocalizedArticleContent,
    SentimentLabel,
    SummaryLine,
    XAIHighlightItem,
    XAIPayload,
)

logger = logging.getLogger(__name__)

_FINANCE_TOKEN_PATTERN = re.compile(
    r"\b(?:EPS|YoY|QoQ|P/E|EBITDA|ROI|ROE|CAGR|FCF|AI|IPO)\b"
)
_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:[$€£¥]?\d[\d,]*(?:\.\d+)?%?|\d+(?:\.\d+)?x)(?![A-Za-z])"
)

_SENTIMENT_LABELS_KO = {
    SentimentLabel.BULLISH: "강세",
    SentimentLabel.BEARISH: "약세",
    SentimentLabel.NEUTRAL: "중립",
    SentimentLabel.MIXED: "혼합",
}

_TICKER_BOX_LABELS_KO = {
    "revenue": "매출",
    "net_income": "순이익",
    "operating_income": "영업이익",
    "guidance": "가이던스",
    "target_price": "목표주가",
    "dividend": "배당",
    "eps": "EPS",
    "yoy": "YoY",
    "qoq": "QoQ",
    "market_cap": "시가총액",
    "pe_ratio": "PER",
}


@dataclass(frozen=True, slots=True)
class _MaskedText:
    text: str
    replacements: dict[str, str]


def build_localized_content(
    *,
    title: str,
    summary_3lines: list[SummaryLine],
    xai: XAIPayload | None,
    sentiment_label: SentimentLabel | None,
    tickers: list[str] | None = None,
) -> LocalizedArticleContent:
    translated_title = _translate_with_fallback(title, tickers=tickers)
    translated_summary = [
        SummaryLine(
            line_number=line.line_number,
            text=_translate_with_fallback(line.text, tickers=tickers),
        )
        for line in summary_3lines
    ]
    translated_xai = _translate_xai_payload(xai, tickers=tickers)

    return LocalizedArticleContent(
        language="ko",
        title=translated_title,
        summary_3lines=translated_summary,
        xai=translated_xai,
        sentiment_label=_SENTIMENT_LABELS_KO.get(sentiment_label),
        ticker_box_labels=dict(_TICKER_BOX_LABELS_KO),
    )


def _translate_xai_payload(
    payload: XAIPayload | None,
    *,
    tickers: list[str] | None,
) -> XAIPayload | None:
    if payload is None:
        return None

    return XAIPayload(
        explanation=_translate_with_fallback(payload.explanation, tickers=tickers),
        highlights=[
            XAIHighlightItem(
                excerpt=_translate_with_fallback(item.excerpt, tickers=tickers),
                relevance_score=item.relevance_score,
                explanation=(
                    _translate_with_fallback(item.explanation, tickers=tickers)
                    if item.explanation
                    else None
                ),
                sentiment_signal=item.sentiment_signal,
                start_char=item.start_char,
                end_char=item.end_char,
            )
            for item in payload.highlights
        ],
    )


def _translate_with_fallback(text: str, *, tickers: list[str] | None) -> str:
    normalized = text.strip()
    if not normalized:
        return normalized

    settings = get_settings()
    if not settings.deepl_api_key:
        return normalized

    try:
        return _translate_text(normalized, tickers=tickers, settings=settings)
    except Exception:
        logger.exception("DeepL translation failed; falling back to source text.")
        return normalized


def _translate_text(text: str, *, tickers: list[str] | None, settings) -> str:
    masked = _mask_text(text, tickers=tickers)
    response = requests.post(
        f"{settings.deepl_api_base_url}/v2/translate",
        headers={
            "Authorization": f"DeepL-Auth-Key {settings.deepl_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "text": [masked.text],
            "target_lang": settings.deepl_target_lang,
            "preserve_formatting": True,
        },
        timeout=settings.deepl_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    translations = payload.get("translations") or []
    if not translations or not translations[0].get("text"):
        return text
    translated = translations[0]["text"].strip()
    return _unmask_text(translated, masked.replacements)


def _mask_text(text: str, *, tickers: list[str] | None) -> _MaskedText:
    replacements: dict[str, str] = {}
    masked = text
    protected_tokens = sorted(
        {
            *(ticker.strip() for ticker in (tickers or []) if ticker and ticker.strip()),
            *(_FINANCE_TOKEN_PATTERN.findall(text)),
            *(_NUMBER_PATTERN.findall(text)),
        },
        key=len,
        reverse=True,
    )

    for index, token in enumerate(protected_tokens):
        placeholder = f"ZXQKEEP{index}ZXQ"
        replacements[placeholder] = token
        masked = masked.replace(token, placeholder)

    return _MaskedText(text=masked, replacements=replacements)


def _unmask_text(text: str, replacements: dict[str, str]) -> str:
    unmasked = text
    for placeholder, token in replacements.items():
        unmasked = unmasked.replace(placeholder, token)
    return unmasked
