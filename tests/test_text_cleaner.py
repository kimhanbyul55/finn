from __future__ import annotations

from app.services.text_cleaner import validate_article_text


def test_validate_article_text_allows_ten_character_input() -> None:
    validation = validate_article_text("1234567890")

    assert validation.is_valid is True
    assert validation.status.value == "valid"
