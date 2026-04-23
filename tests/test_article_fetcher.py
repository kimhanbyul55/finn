from __future__ import annotations

import gzip
from types import SimpleNamespace
from urllib.error import URLError

import requests
from app.schemas.article_fetch import ArticleTextSource
from app.services.article_fetcher.fetcher import (
    _build_headers,
    _format_http_error_message,
    _response_to_html,
    _tls_verify_value,
    fetch_article_text,
)
from app.schemas.article_fetch import ArticleFetchFailureCategory
from app.services.article_fetcher.policy import FetchRetryPolicy


class _DummyResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "<html></html>",
        headers: dict[str, str] | None = None,
        apparent_encoding: str = "utf-8",
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.apparent_encoding = apparent_encoding
        self.encoding = None
        self.url = "https://example.com/final"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = SimpleNamespace(status_code=self.status_code)
            raise error


def test_build_headers_uses_browser_like_defaults() -> None:
    headers = _build_headers(
        link="https://www.reuters.com/world/example-article",
        user_agent="TestBrowser/1.0",
    )

    assert headers["User-Agent"] == "TestBrowser/1.0"
    assert "text/html" in headers["Accept"]
    assert headers["Referer"] == "https://www.reuters.com"
    assert headers["Accept-Encoding"] == "gzip, deflate"


def test_retry_policy_retries_rate_limit_and_stops_after_max_retries() -> None:
    policy = FetchRetryPolicy(max_retries=2)
    error = requests.HTTPError("429 error")
    error.response = SimpleNamespace(status_code=429)

    assert policy.should_retry(error, attempt_index=0) is True
    assert policy.should_retry(error, attempt_index=1) is True
    assert policy.should_retry(error, attempt_index=2) is False


def test_retry_policy_marks_403_as_access_block_and_non_retryable() -> None:
    policy = FetchRetryPolicy()
    error = requests.HTTPError("403 error")
    error.response = SimpleNamespace(status_code=403)

    assert policy.is_access_block(error) is True
    assert policy.should_retry(error, attempt_index=0) is False


def test_retry_policy_retries_retryable_network_error() -> None:
    policy = FetchRetryPolicy()
    error = URLError("temporary failure in name resolution")

    assert policy.should_retry(error, attempt_index=0) is True


def test_http_error_message_marks_retryable_server_error() -> None:
    message = _format_http_error_message(503, retry_policy=FetchRetryPolicy())

    assert "retryable" in message.lower()


def test_response_to_html_accepts_html_content_type() -> None:
    response = _DummyResponse(text="<html><body>ok</body></html>")

    html, content_type = _response_to_html(response)

    assert "<body>ok</body>" in html
    assert content_type == "text/html; charset=utf-8"


def test_response_to_html_rejects_non_html_content_type() -> None:
    response = _DummyResponse(headers={"Content-Type": "application/json"})

    try:
        _response_to_html(response)
    except requests.RequestException as exc:
        assert "Unsupported content type" in str(exc)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("Expected RequestException for non-HTML response")


def test_tls_verify_value_returns_cert_bundle_or_true() -> None:
    value = _tls_verify_value()

    assert isinstance(value, (str, bool))


def test_fetch_article_text_marks_429_as_retryable(monkeypatch) -> None:
    error = requests.HTTPError("429 error")
    error.response = SimpleNamespace(
        status_code=429,
        url="https://example.com/final",
        headers={"Content-Type": "text/html; charset=utf-8"},
    )
    monkeypatch.setattr(
        "app.services.article_fetcher.fetcher.fetch_html",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )

    result = fetch_article_text("https://example.com/article")

    assert result.fetch_status.value == "fetch_failed"
    assert result.retryable is True
    assert result.failure_category == ArticleFetchFailureCategory.RATE_LIMITED
    assert result.publisher_domain == "example.com"
    assert result.http_status_code == 429


def test_fetch_article_text_reports_parse_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.article_fetcher.fetcher.fetch_html",
        lambda **kwargs: SimpleNamespace(
            html="<html><body></body></html>",
            final_url="https://www.reuters.com/final",
            content_type="text/html; charset=utf-8",
            attempt_count=2,
        ),
    )
    monkeypatch.setattr(
        "app.services.article_fetcher.fetcher.parse_article_text",
        lambda **kwargs: SimpleNamespace(text="", source=ArticleTextSource.META_DESCRIPTION),
    )

    result = fetch_article_text("https://www.reuters.com/world/article")

    assert result.fetch_status.value == "parse_failed"
    assert result.failure_category == ArticleFetchFailureCategory.EMPTY_EXTRACT
    assert result.publisher_domain == "www.reuters.com"
    assert result.final_url == "https://www.reuters.com/final"
    assert result.content_type == "text/html; charset=utf-8"
    assert result.extraction_source == ArticleTextSource.META_DESCRIPTION
    assert result.attempt_count == 2


def test_fetch_article_text_records_extraction_source_on_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.article_fetcher.fetcher.fetch_html",
        lambda **kwargs: SimpleNamespace(
            html="<html><body></body></html>",
            final_url="https://example.com/final",
            content_type="text/html; charset=utf-8",
            attempt_count=1,
        ),
    )
    monkeypatch.setattr(
        "app.services.article_fetcher.fetcher.parse_article_text",
        lambda **kwargs: SimpleNamespace(
            text="Stocks rose after earnings beat expectations.",
            source=ArticleTextSource.GENERIC_JSON,
        ),
    )

    result = fetch_article_text("https://example.com/article")

    assert result.fetch_status.value == "success"
    assert result.extraction_source == ArticleTextSource.GENERIC_JSON
    assert result.raw_text == "Stocks rose after earnings beat expectations."
