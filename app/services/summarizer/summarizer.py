from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.core import get_settings
from app.services.groq import groq_chat_completion, groq_is_enabled
from app.services.text_cleaner import clean_article_text

logger = logging.getLogger(__name__)

SUMMARY_LINE_COUNT = 3
MAX_LINE_LENGTH = 120
MIN_SENTENCE_CHARACTERS = 24
MIN_CLAUSE_CHARACTERS = 20

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_GENERATED_LINE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_PATTERN = re.compile(r";|:| -- | - |, (?=(?:but|while|as|after|before|despite)\b)", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|\d+(?:\.\d+)?%?")
_NUMERIC_TOKEN_PATTERN = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")
_GENERIC_PREFIX_PATTERN = re.compile(
    r"^(?:the article|the report|this article|in the article)\s+(?:says|reports|notes|highlights)\s+that\s+",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "with",
}

_FINANCIAL_KEYWORDS = {
    "acquisition",
    "analyst",
    "bank",
    "bond",
    "buyback",
    "cash",
    "cost",
    "credit",
    "debt",
    "dividend",
    "earnings",
    "estimate",
    "forecast",
    "guidance",
    "inflation",
    "loss",
    "margin",
    "market",
    "merger",
    "outlook",
    "profit",
    "quarter",
    "rate",
    "revenue",
    "risk",
    "sales",
    "shares",
    "stock",
    "tariff",
    "valuation",
}


@dataclass(frozen=True, slots=True)
class _CandidateLine:
    text: str
    score: float
    position: int
    source_position: int
    kind: str


def summarize_to_three_lines(title: str, article_text: str) -> list[str]:
    """Return exactly three concise summary lines derived from article text."""
    cleaned_text = clean_article_text(article_text)
    if not cleaned_text:
        return ["", "", ""]

    groq_summary = _summarize_with_groq(title=title, article_text=cleaned_text)
    if groq_summary is not None:
        return groq_summary

    sentences = _extract_sentences(cleaned_text)
    if not sentences:
        fallback = _truncate_for_card(cleaned_text)
        return [fallback, "", ""]

    title_tokens = _tokenize(title)
    candidates = _build_candidates(sentences, title_tokens)
    selected = _select_distinct_candidates(candidates)

    lines = [_truncate_for_card(candidate.text) for candidate in selected[:SUMMARY_LINE_COUNT]]
    if len(lines) < SUMMARY_LINE_COUNT:
        lines.extend(_build_fallback_lines(cleaned_text, lines))

    while len(lines) < SUMMARY_LINE_COUNT:
        lines.append("")

    return lines[:SUMMARY_LINE_COUNT]


def _summarize_with_groq(*, title: str, article_text: str) -> list[str] | None:
    if not groq_is_enabled():
        return None

    settings = get_settings()
    try:
        content = groq_chat_completion(
            model=settings.groq_summary_model,
            system_prompt=(
                "You are a financial news summarizer. "
                "Write exactly three Korean summary lines. "
                "Each line must be a single sentence. "
                "Keep each line short, self-contained, and usually under 90 Korean characters. "
                "Preserve numbers, percentages, and ticker symbols exactly. "
                "Never invent or alter any number, percentage, ticker, or factual detail. "
                "If a detail is not explicitly stated in the article, omit it. "
                "Do not exaggerate. "
                "Do not repeat the title. "
                "Do not use ellipses. "
                "Ignore table headers, reconciliation labels, boilerplate, and footers. "
                "Return only the three lines with no title, bullets, or commentary."
            ),
            user_prompt=(
                f"Title: {title}\n\n"
                "Article:\n"
                f"{article_text}\n\n"
                "Return exactly three Korean lines."
            ),
        )
    except Exception:
        logger.exception("Groq summary generation failed; falling back to heuristic summarizer.")
        return None

    lines = _parse_summary_lines(content)
    if len(lines) != SUMMARY_LINE_COUNT or not all(line.strip() for line in lines):
        logger.warning(
            "Groq summary generation returned unusable output; falling back to heuristic summarizer."
        )
        return None
    if not _summary_preserves_numeric_facts(lines, title=title, article_text=article_text):
        logger.warning(
            "Groq summary generation introduced unsupported numeric facts; falling back to heuristic summarizer."
        )
        return None
    return lines


def _parse_summary_lines(content: str) -> list[str]:
    lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*•]\s*|\d+[.)]\s*)", "", line).strip()
        if not line:
            continue

        line_fragments = _split_generated_line(line)
        for fragment in line_fragments:
            normalized = _normalize_generated_line(fragment)
            if normalized:
                lines.append(normalized)
            if len(lines) == SUMMARY_LINE_COUNT:
                break
        if len(lines) == SUMMARY_LINE_COUNT:
            break
    return lines


def _split_generated_line(line: str) -> list[str]:
    fragments = [fragment.strip() for fragment in _GENERATED_LINE_SPLIT_PATTERN.split(line)]
    return [fragment for fragment in fragments if fragment]


def _normalize_generated_line(line: str) -> str:
    return _normalize_text(line).removesuffix("...").strip()


def _summary_preserves_numeric_facts(lines: list[str], *, title: str, article_text: str) -> bool:
    source_numeric_tokens = _extract_numeric_tokens(f"{title} {article_text}")
    if not source_numeric_tokens:
        return True

    summary_numeric_tokens = _extract_numeric_tokens(" ".join(lines))
    return summary_numeric_tokens.issubset(source_numeric_tokens)


def _extract_numeric_tokens(text: str) -> set[str]:
    return {match.group(0) for match in _NUMERIC_TOKEN_PATTERN.finditer(text)}


def _extract_sentences(text: str) -> list[str]:
    protected_text = _protect_abbreviations(text)
    parts = _SENTENCE_SPLIT_PATTERN.split(protected_text)
    sentences: list[str] = []
    for part in parts:
        sentence = _normalize_text(_restore_abbreviations(part))
        if len(sentence) < MIN_SENTENCE_CHARACTERS:
            continue
        if sentence not in sentences:
            sentences.append(sentence)
    return sentences


def _build_candidates(sentences: list[str], title_tokens: set[str]) -> list[_CandidateLine]:
    candidates: list[_CandidateLine] = []
    for index, sentence in enumerate(sentences):
        score = _score_sentence(sentence, index, title_tokens)
        candidates.append(
            _CandidateLine(
                text=sentence,
                score=score + 0.3,
                position=index,
                source_position=index,
                kind="sentence",
            )
        )

        for clause in _extract_clauses(sentence):
            clause_score = score - 0.35
            candidates.append(
                _CandidateLine(
                    text=clause,
                    score=clause_score,
                    position=index,
                    source_position=index,
                    kind="clause",
                )
            )

    return candidates


def _score_sentence(sentence: str, index: int, title_tokens: set[str]) -> float:
    tokens = _tokenize(sentence)
    overlap = len(tokens & title_tokens)
    finance_hits = len(tokens & _FINANCIAL_KEYWORDS)
    numeric_hits = sentence.count("%") + sentence.count("$")
    position_bonus = max(0.0, 2.5 - (index * 0.25))
    length_bonus = min(len(tokens) / 12, 1.5)

    score = position_bonus + (overlap * 1.7) + (finance_hits * 0.8) + (numeric_hits * 0.5) + length_bonus

    if _looks_like_title_echo(sentence, title_tokens):
        score -= 1.5
    if _is_generic_sentence(sentence):
        score -= 1.0

    return score


def _extract_clauses(sentence: str) -> list[str]:
    clauses: list[str] = []
    for part in _CLAUSE_SPLIT_PATTERN.split(sentence):
        clause = _normalize_text(part)
        if len(clause) >= MIN_CLAUSE_CHARACTERS and clause != sentence and clause not in clauses:
            clauses.append(clause)
    return clauses


def _select_distinct_candidates(candidates: list[_CandidateLine]) -> list[_CandidateLine]:
    ranked = sorted(
        candidates,
        key=lambda item: (-item.score, item.position, item.kind != "sentence", len(item.text)),
    )
    selected: list[_CandidateLine] = []
    used_sources: set[int] = set()

    for candidate in ranked:
        if candidate.source_position in used_sources:
            continue
        if _is_too_similar_to_any(candidate.text, [item.text for item in selected]):
            continue
        selected.append(candidate)
        used_sources.add(candidate.source_position)
        if len(selected) >= SUMMARY_LINE_COUNT:
            break

    if len(selected) < SUMMARY_LINE_COUNT:
        for candidate in ranked:
            if candidate in selected:
                continue
            if _is_too_similar_to_any(candidate.text, [item.text for item in selected]):
                continue
            selected.append(candidate)
            if len(selected) >= SUMMARY_LINE_COUNT:
                break

    return sorted(selected, key=lambda item: item.position)


def _build_fallback_lines(cleaned_text: str, existing_lines: list[str]) -> list[str]:
    fallback_lines: list[str] = []
    for sentence in _extract_sentences(cleaned_text):
        candidate = _truncate_for_card(sentence)
        if not candidate:
            continue
        if candidate in existing_lines or candidate in fallback_lines:
            continue
        if _is_too_similar_to_any(candidate, existing_lines + fallback_lines):
            continue
        fallback_lines.append(candidate)
        if len(existing_lines) + len(fallback_lines) >= SUMMARY_LINE_COUNT:
            break
    return fallback_lines


def _tokenize(text: str) -> set[str]:
    tokens = {match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)}
    return {token for token in tokens if token not in _STOPWORDS}


def _normalize_text(text: str) -> str:
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip(" \t\n-")
    text = _GENERIC_PREFIX_PATTERN.sub("", text)
    return text.strip()


def _looks_like_title_echo(sentence: str, title_tokens: set[str]) -> bool:
    sentence_tokens = _tokenize(sentence)
    if not sentence_tokens or not title_tokens:
        return False
    overlap_ratio = len(sentence_tokens & title_tokens) / max(len(title_tokens), 1)
    return overlap_ratio >= 0.8


def _is_generic_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return lowered.startswith(("the article ", "the report ", "this article "))


def _is_too_similar_to_any(candidate: str, existing_lines: list[str]) -> bool:
    candidate_tokens = _tokenize(candidate)
    if not candidate_tokens:
        return True

    for line in existing_lines:
        line_tokens = _tokenize(line)
        if not line_tokens:
            continue
        overlap = len(candidate_tokens & line_tokens) / max(len(candidate_tokens | line_tokens), 1)
        if overlap >= 0.7:
            return True
    return False


def _truncate_for_card(text: str) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= MAX_LINE_LENGTH:
        return normalized

    truncated = normalized[:MAX_LINE_LENGTH].rsplit(" ", 1)[0].rstrip(",;:-")
    if not truncated:
        truncated = normalized[:MAX_LINE_LENGTH].rstrip(",;:-")
    return f"{truncated}..."


def _protect_abbreviations(text: str) -> str:
    return re.sub(r"\b(?:[A-Z]\.){2,}", lambda match: match.group(0).replace(".", "<prd>"), text)


def _restore_abbreviations(text: str) -> str:
    return text.replace("<prd>", ".")
