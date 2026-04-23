from __future__ import annotations

import html
import re
from dataclasses import dataclass
from enum import Enum


MINIMUM_WORD_COUNT = 1
MINIMUM_CHARACTER_COUNT = 10
ARTICLE_MINIMUM_WORD_COUNT = 20
ARTICLE_MINIMUM_CHARACTER_COUNT = 120
SUMMARY_MINIMUM_WORD_COUNT = 8
SUMMARY_MINIMUM_CHARACTER_COUNT = 40

_INTERNAL_WHITESPACE_PATTERN = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
_SEPARATOR_LINE_PATTERN = re.compile(r"^[\W_]{2,}$")
_URL_ONLY_LINE_PATTERN = re.compile(r"^https?://\S+$", re.IGNORECASE)
_SENTENCE_END_PATTERN = re.compile(r"[.!?](?:\s|$)")
_ALPHA_TOKEN_PATTERN = re.compile(r"[A-Za-z]{3,}")
_TRANSCRIPT_SPEAKER_PATTERN = re.compile(
    r"^(?:[A-Z][A-Za-z.'-]{1,24}|Operator|Moderator|Unknown Speaker|Q|A)\s*:\s*",
    re.IGNORECASE,
)
_TRANSCRIPT_CUE_PATTERN = re.compile(
    r"^(?:question-and-answer session|q&a|prepared remarks|earnings call transcript|conference call)$",
    re.IGNORECASE,
)
_HTML_SCRIPT_STYLE_PATTERN = re.compile(
    r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^advertisement$", re.IGNORECASE),
    re.compile(r"^advertisement:?$", re.IGNORECASE),
    re.compile(r"^ad$", re.IGNORECASE),
    re.compile(r"^ads$", re.IGNORECASE),
    re.compile(r"^sponsored content$", re.IGNORECASE),
    re.compile(r"^sponsored$", re.IGNORECASE),
    re.compile(r"^promoted content$", re.IGNORECASE),
    re.compile(r"^paid content$", re.IGNORECASE),
    re.compile(r"^partner content$", re.IGNORECASE),
    re.compile(r"^read more$", re.IGNORECASE),
    re.compile(r"^click here$", re.IGNORECASE),
    re.compile(r"^follow us on .+$", re.IGNORECASE),
    re.compile(r"^sign up for (?:our )?newsletter$", re.IGNORECASE),
    re.compile(r"^subscribe(?: now)?$", re.IGNORECASE),
    re.compile(r"^continue reading$", re.IGNORECASE),
    re.compile(r"^related (?:articles|stories|content|reads)$", re.IGNORECASE),
    re.compile(r"^recommended (?:for you|stories)$", re.IGNORECASE),
    re.compile(r"^image source: .+$", re.IGNORECASE),
    re.compile(r"^source: .+getty images.*$", re.IGNORECASE),
    re.compile(r"^this article was originally published on .+$", re.IGNORECASE),
    re.compile(r"^story continues$", re.IGNORECASE),
    re.compile(r"^watch live$", re.IGNORECASE),
    re.compile(r"^watch now$", re.IGNORECASE),
    re.compile(r"^(?:privacy policy|cookie policy|terms of service)$", re.IGNORECASE),
    re.compile(r"^all rights reserved\.?$", re.IGNORECASE),
    re.compile(r"^condensed consolidated statements? of .+$", re.IGNORECASE),
    re.compile(r"^reconciliation of gaap to non-gaap .+$", re.IGNORECASE),
    re.compile(r"^\(?in millions(?:, except per share data)?\)?$", re.IGNORECASE),
    re.compile(r"^\(?unaudited\)?$", re.IGNORECASE),
    re.compile(r"^table of contents$", re.IGNORECASE),
]
_DATELINE_PREFIX_PATTERN = re.compile(
    r"^[A-Z][A-Z .'-]{1,30},\s+[A-Z][a-z]{2,9}\.?\s+\d{1,2}\s*(?:\([^)]*\))?\s*[-:]\s*"
)
_AD_TECH_PATTERN = re.compile(
    r"(?:doubleclick|googletag|adslot|adserver|taboola|outbrain|sponsor)",
    re.IGNORECASE,
)
_PROMO_CTA_KEYWORDS = (
    "subscribe",
    "sign up",
    "join now",
    "click here",
    "read more",
    "continue reading",
    "learn more",
    "download app",
    "watch now",
    "watch live",
    "get started",
)
_PROMO_OFFER_KEYWORDS = (
    "newsletter",
    "premium",
    "membership",
    "special offer",
    "trial",
    "paid",
    "sponsored",
    "partner content",
    "stock advisor",
)


class ArticleTextValidationStatus(str, Enum):
    VALID = "valid"
    TOO_SHORT = "too_short"
    UNUSABLE = "unusable"


@dataclass(frozen=True, slots=True)
class ArticleTextValidationResult:
    """Validation result for cleaned article text."""

    is_valid: bool
    status: ArticleTextValidationStatus
    reason: str | None = None
    word_count: int = 0
    character_count: int = 0


def clean_article_text(raw_text: str) -> str:
    """Clean article text conservatively without removing likely financial content."""
    if not raw_text:
        return ""

    text = html.unescape(raw_text)
    text = _HTML_SCRIPT_STYLE_PATTERN.sub("\n", text)
    text = _HTML_TAG_PATTERN.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u000c", "\n")
    lines = text.split("\n")

    cleaned_lines: list[str] = []
    for line in lines:
        normalized_line = _normalize_line_whitespace(line)
        normalized_line = _strip_transcript_speaker_prefix(normalized_line)
        if not normalized_line:
            continue
        if _is_safe_boilerplate_line(normalized_line):
            continue
        if cleaned_lines and normalized_line == cleaned_lines[-1]:
            continue
        cleaned_lines.append(normalized_line)

    cleaned_lines = _trim_noise_edges(cleaned_lines)
    text = "\n".join(cleaned_lines)
    text = _MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    return text.strip()


def validate_article_text(
    text: str,
    *,
    allow_brief: bool = False,
) -> ArticleTextValidationResult:
    """Validate whether cleaned article text is usable for downstream enrichment."""
    cleaned = clean_article_text(text)
    character_count = len(cleaned)
    word_count = len(cleaned.split())
    minimum_word_count = SUMMARY_MINIMUM_WORD_COUNT if allow_brief else ARTICLE_MINIMUM_WORD_COUNT
    minimum_character_count = (
        SUMMARY_MINIMUM_CHARACTER_COUNT if allow_brief else ARTICLE_MINIMUM_CHARACTER_COUNT
    )

    if not cleaned:
        return ArticleTextValidationResult(
            is_valid=False,
            status=ArticleTextValidationStatus.UNUSABLE,
            reason="No usable article text after cleaning.",
            word_count=0,
            character_count=0,
        )

    if word_count < minimum_word_count or character_count < minimum_character_count:
        return ArticleTextValidationResult(
            is_valid=False,
            status=ArticleTextValidationStatus.TOO_SHORT,
            reason="Article text is too short for reliable downstream analysis.",
            word_count=word_count,
            character_count=character_count,
        )

    return ArticleTextValidationResult(
        is_valid=True,
        status=ArticleTextValidationStatus.VALID,
        reason=None,
        word_count=word_count,
        character_count=character_count,
    )


def is_article_text_usable(text: str) -> bool:
    """Return a simple boolean signal for downstream pipeline checks."""
    return validate_article_text(text).is_valid


def _normalize_line_whitespace(line: str) -> str:
    normalized = _INTERNAL_WHITESPACE_PATTERN.sub(" ", line.strip())
    normalized = _DATELINE_PREFIX_PATTERN.sub("", normalized)
    return normalized


def _is_safe_boilerplate_line(line: str) -> bool:
    if _SEPARATOR_LINE_PATTERN.match(line):
        return True
    if _URL_ONLY_LINE_PATTERN.match(line):
        return True
    if _TRANSCRIPT_CUE_PATTERN.match(line.strip()):
        return True
    if _is_transcript_speaker_marker_line(line):
        return True
    if _looks_like_table_header(line):
        return True
    if _looks_like_promotional_cta_line(line):
        return True
    return any(pattern.match(line) for pattern in _BOILERPLATE_LINE_PATTERNS)


def _strip_transcript_speaker_prefix(line: str) -> str:
    if not line:
        return ""
    if not _TRANSCRIPT_SPEAKER_PATTERN.match(line):
        return line
    content = _TRANSCRIPT_SPEAKER_PATTERN.sub("", line).strip()
    return content


def _is_transcript_speaker_marker_line(line: str) -> bool:
    if not _TRANSCRIPT_SPEAKER_PATTERN.match(line):
        return False
    content = _TRANSCRIPT_SPEAKER_PATTERN.sub("", line).strip()
    return not content


def _looks_like_promotional_cta_line(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return False
    lowered = compact.lower()
    if _AD_TECH_PATTERN.search(lowered):
        return True

    cta_hits = sum(1 for keyword in _PROMO_CTA_KEYWORDS if keyword in lowered)
    offer_hits = sum(1 for keyword in _PROMO_OFFER_KEYWORDS if keyword in lowered)
    if cta_hits >= 2 and len(compact) <= 200:
        return True
    if cta_hits >= 1 and offer_hits >= 1 and len(compact) <= 220:
        return True
    if offer_hits >= 2 and len(compact) <= 180:
        return True
    if _looks_like_narrative_line(compact):
        return False
    return False


def _looks_like_table_header(line: str) -> bool:
    lowered = line.lower()
    compact_line = line.strip()
    if _looks_like_narrative_line(compact_line):
        return False
    # Guard against long single-line article bodies that merely mention
    # financial terms such as GAAP, non-GAAP, or "in millions".
    if len(compact_line) > 160:
        return len(compact_line) > 45 and compact_line.upper() == compact_line and sum(
            ch.isalpha() for ch in compact_line
        ) >= 25
    if "gaap" in lowered and "non-gaap" in lowered:
        return _looks_like_financial_table_label(compact_line)
    if "condensed consolidated" in lowered:
        return _looks_like_financial_table_label(compact_line)
    if "statements of income" in lowered or "balance sheets" in lowered:
        return _looks_like_financial_table_label(compact_line)
    if "except per share data" in lowered or "in millions" in lowered:
        return _looks_like_financial_table_caption(compact_line)
    if len(line) > 45 and line.upper() == line and sum(ch.isalpha() for ch in line) >= 25:
        return True
    return False


def _looks_like_financial_table_label(line: str) -> bool:
    """Only drop financial phrases when they look like a heading, not article prose."""
    alpha_tokens = _ALPHA_TOKEN_PATTERN.findall(line)
    if _SENTENCE_END_PATTERN.search(line) and len(alpha_tokens) >= 8:
        return False
    if len(line) <= 120:
        return True
    return line.upper() == line


def _looks_like_financial_table_caption(line: str) -> bool:
    """Drop short table units/captions while keeping narrative sentences."""
    alpha_tokens = _ALPHA_TOKEN_PATTERN.findall(line)
    if _SENTENCE_END_PATTERN.search(line) and len(alpha_tokens) >= 8:
        return False
    if len(line) <= 90 and (line.startswith("(") or line.endswith(")") or "," in line):
        return True
    return len(alpha_tokens) <= 6


def _looks_like_narrative_line(line: str) -> bool:
    if not _SENTENCE_END_PATTERN.search(line):
        return False

    alpha_tokens = _ALPHA_TOKEN_PATTERN.findall(line)
    if len(alpha_tokens) < 8:
        return False

    lowercase_like_tokens = [token for token in alpha_tokens if token != token.upper()]
    return len(lowercase_like_tokens) >= 5


def _trim_noise_edges(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)

    while start < end and _is_edge_noise_line(lines[start]):
        start += 1

    while end > start and _is_edge_noise_line(lines[end - 1]):
        end -= 1

    return lines[start:end]


def _is_edge_noise_line(line: str) -> bool:
    if _is_safe_boilerplate_line(line):
        return True
    if len(line) <= 2 and not any(character.isalnum() for character in line):
        return True
    return False
