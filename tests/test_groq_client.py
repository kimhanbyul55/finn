from __future__ import annotations

import requests

from app.services.groq.client import (
    _build_chat_completions_url,
    _extract_retry_after_seconds,
    groq_chat_completion,
    groq_log_context,
)


def test_build_chat_completions_url_appends_v1_when_missing() -> None:
    assert (
        _build_chat_completions_url("https://api.groq.com/openai")
        == "https://api.groq.com/openai/v1/chat/completions"
    )


def test_build_chat_completions_url_avoids_double_v1() -> None:
    assert (
        _build_chat_completions_url("https://api.groq.com/openai/v1")
        == "https://api.groq.com/openai/v1/chat/completions"
    )


def test_extract_retry_after_seconds_from_rate_limit_message() -> None:
    class _Response:
        headers = {}

        def json(self) -> dict[str, object]:
            return {
                "error": {
                    "message": "Rate limit reached. Please try again in 3.76s."
                }
            }

    assert _extract_retry_after_seconds(_Response()) == 3.76


def test_groq_chat_completion_retries_once_after_429(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1")

    calls = {"count": 0}
    sleep_calls: list[float] = []

    class _Response:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload
            self.headers = {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError("boom", response=self)

        def json(self) -> dict[str, object]:
            return self._payload

    def _fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _Response(
                429,
                {
                    "error": {
                        "message": (
                            "Rate limit reached for model llama-3.1-8b-instant. "
                            "Please try again in 1.25s."
                        )
                    }
                },
            )
        return _Response(
            200,
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    monkeypatch.setattr("app.services.groq.client.requests.post", _fake_post)
    monkeypatch.setattr("app.services.groq.client.time.sleep", lambda seconds: sleep_calls.append(seconds))

    result = groq_chat_completion(
        model="llama-3.1-8b-instant",
        system_prompt="system",
        user_prompt="user",
        request_label="test_retry",
    )

    assert result == "ok"
    assert calls["count"] == 2
    assert sleep_calls == [1.25]


def test_groq_logs_include_news_context_and_token_usage(monkeypatch, caplog) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1")

    class _Response:
        status_code = 200
        headers = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }

    monkeypatch.setattr("app.services.groq.client.requests.post", lambda *args, **kwargs: _Response())

    with caplog.at_level("INFO"), groq_log_context(
        news_id="news-123",
        link="https://example.com/article",
        groq_context="unit_test",
    ):
        result = groq_chat_completion(
            model="llama-3.1-8b-instant",
            system_prompt="system",
            user_prompt="user",
            request_label="summary_generation",
        )

    assert result == "ok"
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=groq_request_started" in messages
    assert "event=groq_request_completed" in messages
    assert 'news_id="news-123"' in messages
    assert 'request_label="summary_generation"' in messages
    assert "total_tokens=12" in messages
