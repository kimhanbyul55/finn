from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


MINIMUM_WORD_COUNT = 1
MINIMUM_CHARACTER_COUNT = 10

_INTERNAL_WHITESPACE_PATTERN = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
_SEPARATOR_LINE_PATTERN = re.compile(r"^[\W_]{2,}$")
_URL_ONLY_LINE_PATTERN = re.compile(r"^https?://\S+$", re.IGNORECASE)
_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^advertisement$", re.IGNORECASE),
    re.compile(r"^sponsored content$", re.IGNORECASE),
    re.compile(r"^read more$", re.IGNORECASE),
    re.compile(r"^click here$", re.IGNORECASE),
    re.compile(r"^follow us on .+$", re.IGNORECASE),
    re.compile(r"^sign up for (?:our )?newsletter$", re.IGNORECASE),
    re.compile(r"^subscribe(?: now)?$", re.IGNORECASE),
    re.compile(r"^(?:privacy policy|cookie policy|terms of service)$", re.IGNORECASE),
    re.compile(r"^all rights reserved\.?$", re.IGNORECASE),
]


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

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\u000c", "\n")
    lines = text.split("\n")

    cleaned_lines: list[str] = []
    for line in lines:
        normalized_line = _normalize_line_whitespace(line)
        if not normalized_line:
            continue
        if _is_safe_boilerplate_line(normalized_line):
            continue
        cleaned_lines.append(normalized_line)

    cleaned_lines = _trim_noise_edges(cleaned_lines)
    text = "\n".join(cleaned_lines)
    text = _MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    return text.strip()


def validate_article_text(text: str) -> ArticleTextValidationResult:
    """Validate whether cleaned article text is usable for downstream enrichment."""
    cleaned = clean_article_text(text)
    character_count = len(cleaned)
    word_count = len(cleaned.split())

    if not cleaned:
        return ArticleTextValidationResult(
            is_valid=False,
            status=ArticleTextValidationStatus.UNUSABLE,
            reason="No usable article text after cleaning.",
            word_count=0,
            character_count=0,
        )

    if word_count < MINIMUM_WORD_COUNT or character_count < MINIMUM_CHARACTER_COUNT:
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
    return normalized


def _is_safe_boilerplate_line(line: str) -> bool:
    if _SEPARATOR_LINE_PATTERN.match(line):
        return True
    if _URL_ONLY_LINE_PATTERN.match(line):
        return True
    return any(pattern.match(line) for pattern in _BOILERPLATE_LINE_PATTERNS)


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
