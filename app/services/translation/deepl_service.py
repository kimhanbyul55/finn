from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from app.core import get_settings
from app.schemas.enrichment import (
    LocalizedArticleContent,
    SentimentLabel,
    SummaryLine,
    XAIHighlightItem,
    XAIPayload,
)
from app.services.groq import groq_chat_completion, groq_is_enabled

logger = logging.getLogger(__name__)

_FINANCE_TOKEN_PATTERN = re.compile(
    r"\b(?:EPS|YoY|QoQ|P/E|EBITDA|ROI|ROE|CAGR|FCF|AI|IPO)\b"
)
_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:[$€£¥]?\d[\d,]*(?:\.\d+)?%?|\d+(?:\.\d+)?x)(?![A-Za-z])"
)
_HANGUL_PATTERN = re.compile(r"[가-힣]")
_LETTER_PATTERN = re.compile(r"[A-Za-z가-힣]")
_DISALLOWED_TRANSLATION_SCRIPT_PATTERN = re.compile(r"[\u0900-\u097F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]")

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


@dataclass(frozen=True, slots=True)
class _TranslationTask:
    key: str
    text: str


def build_localized_content(
    *,
    title: str,
    summary_3lines: list[SummaryLine],
    xai: XAIPayload | None,
    sentiment_label: SentimentLabel | None,
    tickers: list[str] | None = None,
    xai_highlight_limit: int | None = None,
    allow_groq: bool = True,
) -> LocalizedArticleContent:
    limited_xai = _limit_xai_payload(xai, highlight_limit=xai_highlight_limit)
    translations = _translate_localized_payload(
        title=title,
        summary_3lines=summary_3lines,
        xai=limited_xai,
        tickers=tickers,
        allow_groq=allow_groq,
    )
    translated_title = translations["title"]
    translated_summary = [
        SummaryLine(
            line_number=line.line_number,
            text=translations[f"summary_{line.line_number}"],
        )
        for line in summary_3lines
    ]
    translated_xai = _translate_xai_payload(limited_xai, translations=translations)

    return LocalizedArticleContent(
        language="ko",
        title=translated_title,
        summary_3lines=translated_summary,
        xai=translated_xai,
        sentiment_label=_SENTIMENT_LABELS_KO.get(sentiment_label),
        ticker_box_labels=dict(_TICKER_BOX_LABELS_KO),
    )


def _limit_xai_payload(
    payload: XAIPayload | None,
    *,
    highlight_limit: int | None,
) -> XAIPayload | None:
    if payload is None or highlight_limit is None or highlight_limit < 0:
        return payload
    return XAIPayload(
        explanation=payload.explanation,
        highlights=payload.highlights[:highlight_limit],
    )


def _translate_xai_payload(payload: XAIPayload | None, *, translations: dict[str, str]) -> XAIPayload | None:
    if payload is None:
        return None

    return XAIPayload(
        explanation=translations["xai_explanation"],
        highlights=[
            XAIHighlightItem(
                excerpt=translations[f"xai_highlight_{index + 1}"],
                relevance_score=item.relevance_score,
                explanation=(
                    translations.get(f"xai_detail_{index + 1}") if item.explanation else None
                ),
                sentiment_signal=item.sentiment_signal,
                start_char=item.start_char,
                end_char=item.end_char,
            )
            for index, item in enumerate(payload.highlights)
        ],
    )


def _translate_localized_payload(
    *,
    title: str,
    summary_3lines: list[SummaryLine],
    xai: XAIPayload | None,
    tickers: list[str] | None,
    allow_groq: bool,
) -> dict[str, str]:
    tasks = _build_translation_tasks(title=title, summary_3lines=summary_3lines, xai=xai)
    original_values = {task.key: task.text.strip() for task in tasks}
    if not allow_groq or not groq_is_enabled():
        return original_values

    try:
        return _translate_tasks(tasks, tickers=tickers)
    except Exception:
        logger.exception("Groq translation failed; falling back to source text.")
        return original_values


def _build_translation_tasks(
    *,
    title: str,
    summary_3lines: list[SummaryLine],
    xai: XAIPayload | None,
) -> list[_TranslationTask]:
    tasks: list[_TranslationTask] = [_TranslationTask(key="title", text=title)]
    tasks.extend(
        _TranslationTask(key=f"summary_{line.line_number}", text=line.text)
        for line in summary_3lines
    )
    if xai is not None:
        tasks.append(_TranslationTask(key="xai_explanation", text=xai.explanation))
        for index, item in enumerate(xai.highlights, start=1):
            tasks.append(_TranslationTask(key=f"xai_highlight_{index}", text=item.excerpt))
            if item.explanation:
                tasks.append(_TranslationTask(key=f"xai_detail_{index}", text=item.explanation))
    return tasks


def _translate_tasks(
    tasks: list[_TranslationTask],
    *,
    tickers: list[str] | None,
) -> dict[str, str]:
    results = {task.key: task.text.strip() for task in tasks}
    prepared_tasks = [
        _TranslationTask(
            key=task.key,
            text=_prepare_translation_input(task.text.strip(), char_limit=get_settings().groq_translation_char_limit),
        )
        for task in tasks
        if task.text.strip() and not _looks_already_korean(task.text)
    ]
    if not prepared_tasks:
        return results

    batch_payload = _build_translation_batch_payload(prepared_tasks)
    masked = _mask_text(batch_payload, tickers=tickers)
    translated = _cached_translation_batch_completion(
        get_settings().groq_api_base_url,
        get_settings().groq_translation_model,
        masked.text,
        "translate_localized_payload",
    )
    unmasked = _unmask_text(translated, masked.replacements)
    parsed = _parse_translation_batch_output(unmasked, prepared_tasks)
    invalid_tasks: list[_TranslationTask] = []

    for task in prepared_tasks:
        original = task.text.strip()
        translated_text = parsed.get(task.key) or original
        polished = _polish_korean_financial_text(translated_text)
        if _is_usable_korean_translation(polished):
            results[task.key] = polished
        else:
            invalid_tasks.append(task)

    if invalid_tasks:
        repaired = _repair_invalid_translations(invalid_tasks, tickers=tickers)
        results.update(repaired)
    return results


def _build_translation_batch_payload(tasks: Iterable[_TranslationTask]) -> str:
    return "\n".join(f"{task.key}|||{task.text}" for task in tasks)


def _parse_translation_batch_output(
    output: str,
    tasks: list[_TranslationTask],
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    valid_keys = {task.key for task in tasks}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or "|||" not in line:
            continue
        key, text = line.split("|||", 1)
        normalized_key = key.strip()
        normalized_text = text.strip()
        if normalized_key in valid_keys and normalized_text:
            parsed[normalized_key] = normalized_text
    return parsed


def _looks_already_korean(text: str) -> bool:
    letters = _LETTER_PATTERN.findall(text)
    if not letters:
        return False

    hangul_count = len(_HANGUL_PATTERN.findall(text))
    return hangul_count / len(letters) >= 0.35


def _is_usable_korean_translation(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if _DISALLOWED_TRANSLATION_SCRIPT_PATTERN.search(normalized):
        return False
    return _looks_already_korean(normalized)


def _repair_invalid_translations(
    tasks: list[_TranslationTask],
    *,
    tickers: list[str] | None,
) -> dict[str, str]:
    if not tasks:
        return {}

    batch_payload = _build_translation_batch_payload(tasks)
    masked = _mask_text(batch_payload, tickers=tickers)
    try:
        translated = _cached_translation_repair_completion(
            get_settings().groq_api_base_url,
            get_settings().groq_translation_model,
            masked.text,
            "translate_localized_payload_repair",
        )
    except Exception:
        logger.exception("Groq translation repair failed; falling back to source text.")
        return {task.key: task.text.strip() for task in tasks}

    unmasked = _unmask_text(translated, masked.replacements)
    parsed = _parse_translation_batch_output(unmasked, tasks)
    repaired: dict[str, str] = {}
    for task in tasks:
        original = task.text.strip()
        translated_text = parsed.get(task.key) or original
        polished = _polish_korean_financial_text(translated_text)
        if _is_usable_korean_translation(polished):
            repaired[task.key] = polished
        else:
            logger.warning(
                "Groq translation failed Korean validation; falling back to source text.",
                extra={"translation_key": task.key},
            )
            repaired[task.key] = original
    return repaired


def _translate_with_fallback(
    text: str,
    *,
    tickers: list[str] | None,
    request_label: str,
) -> str:
    normalized = text.strip()
    if not normalized:
        return normalized
    if not groq_is_enabled():
        return normalized
    try:
        return _translate_text(normalized, tickers=tickers, request_label=request_label)
    except Exception:
        logger.exception("Groq translation failed; falling back to source text.")
        return normalized


def _translate_text(text: str, *, tickers: list[str] | None, request_label: str) -> str:
    settings = get_settings()
    prepared = _prepare_translation_input(text, char_limit=settings.groq_translation_char_limit)
    masked = _mask_text(prepared, tickers=tickers)
    translated = _cached_translation_completion(
        settings.groq_api_base_url,
        settings.groq_translation_model,
        masked.text,
        request_label,
    )
    return _polish_korean_financial_text(_unmask_text(translated, masked.replacements))


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


@lru_cache(maxsize=512)
def _cached_translation_completion(base_url: str, model: str, masked_text: str, request_label: str) -> str:
    del base_url
    return groq_chat_completion(
        model=model,
        system_prompt=(
            "You are a Korean financial news translator. "
            "Translate the input into natural Korean financial news style. "
            "Return Korean only. Do not use Hindi, Chinese, Japanese, or any non-Korean language. "
            "Use concise declarative 기사체 and avoid literal translation. "
            "Prefer established financial terminology such as '가이던스', '전년 대비', and '경영진' when appropriate. "
            "Keep placeholders unchanged. "
            "Keep numbers, percentages, dates, currencies, ticker symbols, and finance abbreviations exactly as written. "
            "Do not add commentary, quotation marks, bullets, or explanations."
        ),
        user_prompt=masked_text,
        temperature=0.0,
        request_label=request_label,
    )


@lru_cache(maxsize=256)
def _cached_translation_batch_completion(
    base_url: str,
    model: str,
    masked_payload: str,
    request_label: str,
) -> str:
    del base_url
    return groq_chat_completion(
        model=model,
        system_prompt=(
            "You are a Korean financial news translator. "
            "Translate each tagged record into natural Korean financial news style. "
            "Return Korean only. Do not use Hindi, Chinese, Japanese, or any non-Korean language. "
            "Input lines are formatted as KEY|||TEXT. "
            "Return the same number of lines in the exact same KEY|||TEXT format. "
            "Do not reorder, merge, drop, or rename any keys. "
            "Translate only the TEXT portion. "
            "Use concise declarative 기사체 and avoid literal translation. "
            "Prefer established financial terminology such as '가이던스', '전년 대비', and '경영진' when appropriate. "
            "Keep placeholders unchanged. "
            "Keep numbers, percentages, dates, currencies, ticker symbols, and finance abbreviations exactly as written. "
            "Do not add commentary, quotation marks, bullets, explanations, or extra lines."
        ),
        user_prompt=masked_payload,
        temperature=0.0,
        request_label=request_label,
    )


@lru_cache(maxsize=128)
def _cached_translation_repair_completion(
    base_url: str,
    model: str,
    masked_payload: str,
    request_label: str,
) -> str:
    del base_url
    return groq_chat_completion(
        model=model,
        system_prompt=(
            "You are a strict Korean-only financial news translation validator and repairer. "
            "Input lines are formatted as KEY|||TEXT. "
            "Return the same keys in the exact same KEY|||TEXT format. "
            "Translate every TEXT into natural Korean financial news style. "
            "The output must be Korean only, except protected placeholders, ticker symbols, company names, "
            "numbers, percentages, dates, currencies, and standard finance abbreviations. "
            "Never output Hindi, Chinese, Japanese, romanized Hindi, or mixed-language sentences. "
            "Do not add commentary, quotation marks, bullets, explanations, or extra lines."
        ),
        user_prompt=masked_payload,
        temperature=0.0,
        request_label=request_label,
    )


def _prepare_translation_input(text: str, *, char_limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= char_limit:
        return normalized

    truncated = normalized[:char_limit].rsplit(" ", 1)[0].rstrip(",;:-")
    return truncated or normalized[:char_limit]


def _polish_korean_financial_text(text: str) -> str:
    polished = text.strip()
    replacements = (
        ("매니저들", "경영진"),
        ("매니저들은", "경영진은"),
        ("매니저는", "경영진은"),
        ("관리진", "경영진"),
        ("전망을 높였다고", "가이던스를 상향했다고"),
        ("했다고 합니다.", "했다고 밝혔다."),
        ("라고 합니다.", "라고 밝혔다."),
        ("라고 말했다.", "라고 밝혔다."),
        ("말했습니다.", "밝혔다."),
        ("말했다.", "밝혔다."),
        ("전망을 높였다", "가이던스를 상향했다"),
        ("강력하게 유지", "견조했다"),
        ("향상되었다고 합니다.", "개선됐다."),
        ("향상되었다.", "개선됐다."),
    )
    for source, target in replacements:
        polished = polished.replace(source, target)
    return polished
