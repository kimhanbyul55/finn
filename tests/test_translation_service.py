from __future__ import annotations

from app.schemas.enrichment import SummaryLine, XAIHighlightItem, XAIPayload
from app.schemas.enrichment import SentimentLabel
from app.services.translation.gemini_translation_service import _cached_translation_batch_completion
from app.services.translation.gemini_translation_service import _cached_translation_repair_completion
from app.services.translation.gemini_translation_service import _polish_korean_financial_text
from app.services.translation.gemini_translation_service import build_localized_content


def test_build_localized_content_returns_none_without_api_key(monkeypatch) -> None:
    _cached_translation_batch_completion.cache_clear()
    _cached_translation_repair_completion.cache_clear()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    localized = build_localized_content(
        title="Apple raises guidance",
        summary_3lines=[
            SummaryLine(line_number=1, text="Revenue grew 12%."),
            SummaryLine(line_number=2, text="Margins improved."),
            SummaryLine(line_number=3, text="Guidance was raised."),
        ],
        xai=XAIPayload(
            explanation="Top article snippets influencing the bullish sentiment result.",
            highlights=[
                XAIHighlightItem(
                    excerpt="Guidance was raised.",
                    relevance_score=0.8,
                    explanation=None,
                    sentiment_signal=SentimentLabel.BULLISH,
                    start_char=0,
                    end_char=19,
                )
            ],
        ),
        sentiment_label=SentimentLabel.BULLISH,
        tickers=["AAPL"],
    )

    assert localized is None


def test_build_localized_content_uses_Gemini_when_api_key_present(monkeypatch) -> None:
    _cached_translation_batch_completion.cache_clear()
    _cached_translation_repair_completion.cache_clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    monkeypatch.setenv("GEMINI_TRANSLATION_MODEL", "gemini-2.5-flash-lite")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": (
                                "title|||애플이 가이던스를 상향했습니다\n"
                                "summary_1|||매출은 12% 증가했습니다.\n"
                                "summary_2|||마진이 개선되었습니다.\n"
                                "summary_3|||가이던스가 상향되었습니다.\n"
                                "xai_explanation|||강세 판단에 영향을 준 핵심 문장입니다.\n"
                                "xai_highlight_1|||가이던스가 상향되었습니다."
                            )}]
                        }
                    }
                ]
            }

    def _fake_post(*args, **kwargs):
        return _Response()

    monkeypatch.setattr("app.services.gemini.client.requests.post", _fake_post)

    localized = build_localized_content(
        title="Apple raises guidance",
        summary_3lines=[
            SummaryLine(line_number=1, text="Revenue grew 12%."),
            SummaryLine(line_number=2, text="Margins improved."),
            SummaryLine(line_number=3, text="Guidance was raised."),
        ],
        xai=XAIPayload(
            explanation="Top article snippets influencing the bullish sentiment result.",
            highlights=[
                XAIHighlightItem(
                    excerpt="Guidance was raised.",
                    relevance_score=0.8,
                    explanation=None,
                    sentiment_signal=SentimentLabel.BULLISH,
                    start_char=0,
                    end_char=19,
                )
            ],
        ),
        sentiment_label=SentimentLabel.BULLISH,
        tickers=["AAPL"],
    )

    assert localized.title == "애플이 가이던스를 상향했습니다"
    assert localized.summary_3lines[0].text == "매출은 12% 증가했습니다."
    assert localized.xai is not None
    assert localized.xai.explanation == "강세 판단에 영향을 준 핵심 문장입니다."
    assert localized.xai.highlights[0].excerpt == "가이던스가 상향되었습니다."


def test_build_localized_content_returns_none_when_gemini_is_disabled(monkeypatch) -> None:
    _cached_translation_batch_completion.cache_clear()
    _cached_translation_repair_completion.cache_clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def _fail_post(*args, **kwargs):
        raise AssertionError("response fallback should not spend Gemini tokens")

    monkeypatch.setattr("app.services.gemini.client.requests.post", _fail_post)

    localized = build_localized_content(
        title="Apple raises guidance",
        summary_3lines=[
            SummaryLine(line_number=1, text="Revenue grew 12%."),
            SummaryLine(line_number=2, text="Margins improved."),
            SummaryLine(line_number=3, text="Guidance was raised."),
        ],
        xai=None,
        sentiment_label=SentimentLabel.BULLISH,
        tickers=["AAPL"],
        allow_gemini=False,
    )

    assert localized is None


def test_build_localized_content_reuses_cached_Gemini_translations(monkeypatch) -> None:
    _cached_translation_batch_completion.cache_clear()
    _cached_translation_repair_completion.cache_clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    monkeypatch.setenv("GEMINI_TRANSLATION_MODEL", "gemini-2.5-flash-lite")

    calls = {"count": 0}
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            calls["count"] += 1
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": (
                                "title|||애플이 가이던스를 상향했습니다\n"
                                "summary_1|||매출은 12% 증가했습니다.\n"
                                "summary_2|||마진이 개선되었습니다.\n"
                                "summary_3|||가이던스가 상향되었습니다."
                            )}]
                        }
                    }
                ]
            }

    def _fake_post(*args, **kwargs):
        return _Response()

    monkeypatch.setattr("app.services.gemini.client.requests.post", _fake_post)

    kwargs = dict(
        title="Apple raises guidance",
        summary_3lines=[
            SummaryLine(line_number=1, text="Revenue grew 12%."),
            SummaryLine(line_number=2, text="Margins improved."),
            SummaryLine(line_number=3, text="Guidance was raised."),
        ],
        xai=None,
        sentiment_label=SentimentLabel.BULLISH,
        tickers=["AAPL"],
    )

    first = build_localized_content(**kwargs)
    second = build_localized_content(**kwargs)

    assert first.title == second.title
    assert first.summary_3lines[0].text == second.summary_3lines[0].text
    assert calls["count"] == 1


def test_build_localized_content_skips_already_korean_summary_lines(monkeypatch) -> None:
    _cached_translation_batch_completion.cache_clear()
    _cached_translation_repair_completion.cache_clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    monkeypatch.setenv("GEMINI_TRANSLATION_MODEL", "gemini-2.5-flash-lite")

    captured_payloads: list[str] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "title|||애플이 가이던스를 상향했습니다"}]
                        }
                    }
                ]
            }

    def _fake_post(*args, **kwargs):
        captured_payloads.append(kwargs["json"]["contents"][0]["parts"][0]["text"])
        return _Response()

    monkeypatch.setattr("app.services.gemini.client.requests.post", _fake_post)

    localized = build_localized_content(
        title="Apple raises guidance",
        summary_3lines=[
            SummaryLine(line_number=1, text="매출은 12% 증가했다."),
            SummaryLine(line_number=2, text="마진은 개선됐다."),
            SummaryLine(line_number=3, text="경영진은 가이던스를 상향했다."),
        ],
        xai=None,
        sentiment_label=SentimentLabel.BULLISH,
        tickers=["AAPL"],
    )

    assert captured_payloads == ["title|||Apple raises guidance"]
    assert localized.title == "애플이 가이던스를 상향했습니다"
    assert localized.summary_3lines[0].text == "매출은 12% 증가했다."
    assert localized.summary_3lines[2].text == "경영진은 가이던스를 상향했다."


def test_build_localized_content_repairs_mixed_language_translation(monkeypatch) -> None:
    _cached_translation_batch_completion.cache_clear()
    _cached_translation_repair_completion.cache_clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    monkeypatch.setenv("GEMINI_TRANSLATION_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("GENAI_ENABLE_GEMINI_TRANSLATION_REPAIR", "true")

    calls = {"count": 0}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            calls["count"] += 1
            if calls["count"] == 1:
                content = "title|||VAL खरीदना 지금이 좋은가, 还是等待해야 하는가?"
            else:
                content = "title|||VAL을 지금 매수할 만한지 신중히 검토해야 한다"
            return {"candidates": [{"content": {"parts": [{"text": content}]}}]}

    def _fake_post(*args, **kwargs):
        return _Response()

    monkeypatch.setattr("app.services.gemini.client.requests.post", _fake_post)

    localized = build_localized_content(
        title="Is it time to buy VAL or wait?",
        summary_3lines=[
            SummaryLine(line_number=1, text="매출은 안정적으로 증가했다."),
            SummaryLine(line_number=2, text="마진은 압박을 받았다."),
            SummaryLine(line_number=3, text="투자자는 가이던스를 지켜보고 있다."),
        ],
        xai=None,
        sentiment_label=SentimentLabel.NEUTRAL,
        tickers=["VAL"],
    )

    assert calls["count"] == 2
    assert localized.title == "VAL을 지금 매수할 만한지 신중히 검토해야 한다"


def test_polish_korean_financial_text_normalizes_literal_finance_phrases() -> None:
    polished = _polish_korean_financial_text(
        "매니저들은 올해 전망을 높였다고 합니다. 운영 마진은 향상되었다."
    )

    assert polished == "경영진은 올해 가이던스를 상향했다고 밝혔다. 운영 마진은 개선됐다."
