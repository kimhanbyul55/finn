from __future__ import annotations

from app.schemas.enrichment import SummaryLine, XAIHighlightItem, XAIPayload
from app.schemas.enrichment import SentimentLabel
from app.services.translation.deepl_service import build_localized_content


def test_build_localized_content_falls_back_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)

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

    assert localized.language == "ko"
    assert localized.title == "Apple raises guidance"
    assert localized.summary_3lines[0].text == "Revenue grew 12%."
    assert localized.xai is not None
    assert localized.xai.highlights[0].excerpt == "Guidance was raised."
    assert localized.sentiment_label == "강세"
    assert localized.ticker_box_labels["revenue"] == "매출"


def test_build_localized_content_uses_deepl_when_api_key_present(monkeypatch) -> None:
    monkeypatch.setenv("DEEPL_API_KEY", "test-key")
    monkeypatch.setenv("DEEPL_API_BASE_URL", "https://api-free.deepl.com")

    translated_values = iter(
        [
            "애플이 가이던스를 상향했습니다",
            "매출은 12% 증가했습니다.",
            "마진이 개선되었습니다.",
            "가이던스가 상향되었습니다.",
            "강세 판단에 영향을 준 핵심 문장입니다.",
            "가이던스가 상향되었습니다.",
        ]
    )

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"translations": [{"text": next(translated_values)}]}

    def _fake_post(*args, **kwargs):
        return _Response()

    monkeypatch.setattr("app.services.translation.deepl_service.requests.post", _fake_post)

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
