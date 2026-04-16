from __future__ import annotations

from app.services.summarizer import summarize_to_three_lines


def test_summarizer_uses_groq_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_SUMMARY_MODEL", "llama-3.1-8b-instant")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "매출이 시장 기대를 웃돌았습니다.\n"
                                "회사는 수요 강세와 마진 개선을 강조했습니다.\n"
                                "투자자들은 향후 가이던스 유지 여부를 주목하고 있습니다."
                            )
                        }
                    }
                ]
            }

    def _fake_post(*args, **kwargs):
        return _Response()

    monkeypatch.setattr("app.services.groq.client.requests.post", _fake_post)

    summary = summarize_to_three_lines(
        title="Company raises outlook after quarterly results",
        article_text=(
            "The company reported quarterly revenue of $12.4 billion, up 8% from a year earlier. "
            "Management said cloud demand remained strong and raised its full-year outlook. "
            "Operating margin narrowed as the company increased AI infrastructure spending."
        ),
    )

    assert summary == [
        "매출이 시장 기대를 웃돌았습니다.",
        "회사는 수요 강세와 마진 개선을 강조했습니다.",
        "투자자들은 향후 가이던스 유지 여부를 주목하고 있습니다.",
    ]


def test_summarizer_splits_single_line_groq_output_into_three_sentences(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_SUMMARY_MODEL", "llama-3.1-8b-instant")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "매출은 전년 대비 12% 증가했다. "
                                "영업 마진은 개선됐다. "
                                "회사는 연간 가이던스를 상향 조정했다."
                            )
                        }
                    }
                ]
            }

    def _fake_post(*args, **kwargs):
        return _Response()

    monkeypatch.setattr("app.services.groq.client.requests.post", _fake_post)

    summary = summarize_to_three_lines(
        title="Company raises outlook after quarterly results",
        article_text=(
            "Revenue rose 12% year over year. "
            "Operating margin improved. "
            "Management raised full-year guidance."
        ),
    )

    assert summary == [
        "매출은 전년 대비 12% 증가했다.",
        "영업 마진은 개선됐다.",
        "회사는 연간 가이던스를 상향 조정했다.",
    ]


def test_summarizer_rejects_groq_output_with_invented_numbers(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_SUMMARY_MODEL", "llama-3.1-8b-instant")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "매출은 전년 대비 12% 증가했다.\n"
                                "영업 마진은 개선됐다.\n"
                                "매출은 강한 수요로 전년 대비 15% 증가했다."
                            )
                        }
                    }
                ]
            }

    def _fake_post(*args, **kwargs):
        return _Response()

    monkeypatch.setattr("app.services.groq.client.requests.post", _fake_post)

    summary = summarize_to_three_lines(
        title="Company raises outlook after quarterly results",
        article_text=(
            "Revenue rose 12% year over year. "
            "Operating margin improved. "
            "Management raised full-year guidance."
        ),
    )

    assert summary == [
        "Revenue rose 12% year over year.",
        "Operating margin improved.",
        "Management raised full-year guidance.",
    ]
