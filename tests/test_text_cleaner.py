from __future__ import annotations

from app.services.text_cleaner import (
    clean_article_text,
    explain_cleaning_decisions,
    validate_article_text,
)


def test_validate_article_text_allows_ten_character_input() -> None:
    validation = validate_article_text(
        "Revenue rose 12% and management raised guidance for the year.",
        allow_brief=True,
    )

    assert validation.is_valid is True
    assert validation.status.value == "valid"


def test_clean_article_text_removes_common_financial_table_headers() -> None:
    raw_text = (
        "CONDENSED CONSOLIDATED STATEMENTS OF INCOME\n"
        "(In millions, except per share data)\n"
        "(Unaudited)\n"
        "Revenue rose 12% year over year.\n"
        "Management raised guidance.\n"
    )

    cleaned = clean_article_text(raw_text)

    assert "CONDENSED CONSOLIDATED" not in cleaned
    assert "In millions" not in cleaned
    assert "Unaudited" not in cleaned
    assert "Revenue rose 12% year over year." in cleaned


def test_clean_article_text_keeps_long_single_line_articles_with_financial_terms() -> None:
    raw_text = (
        "Microsoft posted a solid quarterly earnings and revenue beat, while non-GAAP earnings "
        "rose to $4.14 per share and GAAP remained part of the company discussion in the report. "
        "Revenue reached $81.27 billion and management said Azure growth was 39% year over year, "
        "with analysts continuing to debate whether the outlook remains strong."
    )

    cleaned = clean_article_text(raw_text)

    assert cleaned == raw_text


def test_clean_article_text_keeps_narrative_earnings_lines_with_gaap_terms() -> None:
    raw_text = (
        "WD-40 reported second-quarter results and said revenue rose 5% year over year. "
        "Management said non-GAAP gross margin improved while GAAP results reflected higher "
        "operating costs, and the company reaffirmed full-year guidance.\n"
        "(In millions, except per share data)\n"
        "CONDENSED CONSOLIDATED STATEMENTS OF INCOME\n"
    )

    cleaned = clean_article_text(raw_text)

    assert "WD-40 reported second-quarter results" in cleaned
    assert "non-GAAP gross margin improved while GAAP results reflected higher operating costs" in cleaned
    assert "In millions" not in cleaned
    assert "CONDENSED CONSOLIDATED" not in cleaned


def test_clean_article_text_keeps_sentence_with_table_caption_terms() -> None:
    raw_text = (
        "The company said revenue, reported in millions, increased year over year as demand "
        "improved and management pointed to stronger bookings in the current quarter.\n"
        "In millions\n"
        "Net sales\n"
    )

    cleaned = clean_article_text(raw_text)

    assert "reported in millions, increased year over year" in cleaned
    assert "In millions" not in cleaned


def test_clean_article_text_keeps_sentence_describing_gaap_reconciliation() -> None:
    raw_text = (
        "Management said GAAP and non-GAAP results both improved in the quarter, while the "
        "company noted that cost controls helped offset softer demand in one segment.\n"
        "RECONCILIATION OF GAAP TO NON-GAAP FINANCIAL MEASURES\n"
    )

    cleaned = clean_article_text(raw_text)

    assert "GAAP and non-GAAP results both improved" in cleaned
    assert "RECONCILIATION OF GAAP TO NON-GAAP" not in cleaned


def test_validate_article_text_requires_richer_article_bodies_by_default() -> None:
    validation = validate_article_text("Revenue rose and outlook improved.")

    assert validation.is_valid is False
    assert validation.status.value == "too_short"


def test_clean_article_text_removes_html_tags_and_script_noise() -> None:
    raw_text = (
        "<html><body><script>var x = 1;</script>"
        "<p>Revenue rose 12% year over year.</p>"
        "<p>Management raised guidance.</p>"
        "</body></html>"
    )

    cleaned = clean_article_text(raw_text)

    assert "var x = 1" not in cleaned
    assert "<p>" not in cleaned
    assert "Revenue rose 12% year over year." in cleaned
    assert "Management raised guidance." in cleaned


def test_clean_article_text_keeps_transcript_content_while_dropping_speaker_prefix() -> None:
    raw_text = (
        "Operator: Good day, and welcome to the earnings call.\n"
        "CEO: Revenue rose 12% year over year and demand remained strong.\n"
        "CFO: We raised full-year guidance."
    )

    cleaned = clean_article_text(raw_text)

    assert "Operator:" not in cleaned
    assert "CEO:" not in cleaned
    assert "CFO:" not in cleaned
    assert "Good day, and welcome to the earnings call." in cleaned
    assert "Revenue rose 12% year over year and demand remained strong." in cleaned
    assert "We raised full-year guidance." in cleaned


def test_clean_article_text_deduplicates_consecutive_identical_lines() -> None:
    raw_text = (
        "Revenue rose 12% year over year.\n"
        "Revenue rose 12% year over year.\n"
        "Management raised guidance."
    )

    cleaned = clean_article_text(raw_text)

    assert cleaned.count("Revenue rose 12% year over year.") == 1


def test_clean_article_text_removes_promotional_cta_lines() -> None:
    raw_text = (
        "Revenue rose 14% year over year as enterprise demand remained strong.\n"
        "Sign up for our premium newsletter and read more market insights.\n"
        "Management raised full-year guidance."
    )

    cleaned = clean_article_text(raw_text)

    assert "premium newsletter" not in cleaned.lower()
    assert "Revenue rose 14% year over year" in cleaned
    assert "Management raised full-year guidance." in cleaned


def test_clean_article_text_removes_image_source_and_story_continues_noise() -> None:
    raw_text = (
        "Image source: Getty Images.\n"
        "Story continues\n"
        "Operating margin improved while revenue beat expectations."
    )

    cleaned = clean_article_text(raw_text)

    assert "Image source: Getty Images." not in cleaned
    assert "Story continues" not in cleaned
    assert "Operating margin improved while revenue beat expectations." in cleaned


def test_explain_cleaning_decisions_returns_drop_reasons_for_promotional_line() -> None:
    raw_text = (
        "Revenue rose 10% year over year.\n"
        "Sign up for our premium newsletter and read more market insights."
    )
    decisions = explain_cleaning_decisions(raw_text)
    promo = [item for item in decisions if "premium newsletter" in item.line.lower()][0]

    assert promo.drop is True
    assert promo.score >= 3
    assert any(
        reason in promo.reasons
        for reason in ("multi_cta_keywords", "cta_plus_offer", "known_boilerplate")
    )


def test_clean_article_text_removes_known_yahoo_advertisement_blocks() -> None:
    raw_text = (
        "Revenue rose 12% year over year.\n"
        "Will AI create the world's first trillionaire? Our team just released a report. Continue »\n"
        "Management raised full-year guidance."
    )
    cleaned = clean_article_text(raw_text)

    assert "trillionaire" not in cleaned.lower()
    assert "Continue »" not in cleaned
    assert "Revenue rose 12% year over year." in cleaned
    assert "Management raised full-year guidance." in cleaned


def test_clean_article_text_removes_inline_buy_time_prompt_block() -> None:
    raw_text = (
        "Nvidia beat earnings expectations for the quarter. "
        "Is now the time to buy Nvidia? Access our full analysis report here, it's free. "
        "The company also raised guidance."
    )
    cleaned = clean_article_text(raw_text)

    assert "Access our full analysis report here" not in cleaned
    assert "Nvidia beat earnings expectations for the quarter." in cleaned
    assert "The company also raised guidance." in cleaned


def test_clean_article_text_removes_ui_control_lines() -> None:
    raw_text = (
        "Revenue rose 8% and operating margin improved.\n"
        "Show more\n"
        "Sort by: Most recent results\n"
        "Add to watchlist\n"
        "Management reiterated full-year guidance."
    )
    cleaned = clean_article_text(raw_text)

    assert "Show more" not in cleaned
    assert "Sort by: Most recent results" not in cleaned
    assert "Add to watchlist" not in cleaned
    assert "Revenue rose 8% and operating margin improved." in cleaned
    assert "Management reiterated full-year guidance." in cleaned
