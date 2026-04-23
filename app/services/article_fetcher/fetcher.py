from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Final
from urllib.parse import urlparse

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

from app.schemas.article_fetch import (
    ArticleFetchFailureCategory,
    ArticleFetchResult,
    ArticleFetchStatus,
)
from app.services.article_fetcher.parser import (
    ArticleHTMLParser,
    ArticleParseResult,
    SimpleHTMLArticleParser,
)
from app.services.article_fetcher.policy import FetchRetryPolicy


DEFAULT_TIMEOUT_SECONDS: Final[int] = 10
BROWSER_USER_AGENTS: Final[tuple[str, ...]] = (
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/133.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/133.0.0.0 Safari/537.36"
    ),
)
HTML_CONTENT_TYPES: Final[tuple[str, ...]] = (
    "text/html",
    "application/xhtml+xml",
)

try:
    import certifi
except ImportError:  # pragma: no cover - optional dependency path
    certifi = None


@dataclass(frozen=True, slots=True)
class _FetchedHTML:
    html: str
    final_url: str
    content_type: str | None
    attempt_count: int


@dataclass(frozen=True, slots=True)
class _FetchDiagnosticError(RequestException):
    message: str
    retryable: bool
    failure_category: ArticleFetchFailureCategory
    attempt_count: int
    http_status_code: int | None = None
    content_type: str | None = None
    final_url: str | None = None

    def __str__(self) -> str:
        return self.message


def fetch_html(
    link: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    retry_policy: FetchRetryPolicy | None = None,
) -> _FetchedHTML:
    """Fetch raw article HTML with a shared session and explicit retry policy."""
    active_retry_policy = retry_policy or FetchRetryPolicy()
    session = _get_http_session()
    last_error: Exception | None = None

    for attempt_index in range(active_retry_policy.max_retries + 1):
        user_agent = BROWSER_USER_AGENTS[min(attempt_index, len(BROWSER_USER_AGENTS) - 1)]
        try:
            response = _fetch_html_once(
                session=session,
                link=link,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
                verify=_tls_verify_value(),
            )
            _raise_for_response_status(response)
            html, content_type = _response_to_html(response)
            return _FetchedHTML(
                html=html,
                final_url=str(response.url),
                content_type=content_type,
                attempt_count=attempt_index + 1,
            )
        except RequestException as exc:
            last_error = exc
            if not active_retry_policy.should_retry(exc, attempt_index=attempt_index):
                raise _request_exception_to_diagnostic_error(
                    exc,
                    retry_policy=active_retry_policy,
                    attempt_index=attempt_index,
                ) from exc

        if last_error is not None and active_retry_policy.should_retry(
            last_error,
            attempt_index=attempt_index,
        ):
            time.sleep(active_retry_policy.backoff_seconds(attempt_index))
            continue
        if attempt_index == active_retry_policy.max_retries and last_error is not None:
            if isinstance(last_error, _FetchDiagnosticError):
                raise last_error
            if isinstance(last_error, RequestException):
                raise _request_exception_to_diagnostic_error(
                    last_error,
                    retry_policy=active_retry_policy,
                    attempt_index=attempt_index,
                ) from last_error
            raise _FetchDiagnosticError(
                message=str(last_error),
                retryable=False,
                failure_category=ArticleFetchFailureCategory.UNEXPECTED_ERROR,
                attempt_count=attempt_index + 1,
            ) from last_error

    raise RuntimeError("Article fetch exhausted all retries without a final error.")


def parse_article_text(
    html: str,
    parser: ArticleHTMLParser | None = None,
) -> ArticleParseResult:
    """Parse fetched HTML into article text plus extraction diagnostics."""
    active_parser = parser or SimpleHTMLArticleParser()
    return active_parser.extract_result(html)


def fetch_article_text(
    link: str,
    parser: ArticleHTMLParser | None = None,
    retry_policy: FetchRetryPolicy | None = None,
) -> ArticleFetchResult:
    """Fetch an article page and return structured text extraction output."""
    parsed_url = urlparse(link)
    publisher_domain = parsed_url.netloc.lower()
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return ArticleFetchResult(
            link=link,
            publisher_domain=publisher_domain,
            attempt_count=1,
            raw_text="",
            cleaned_text="",
            fetch_status=ArticleFetchStatus.FETCH_FAILED,
            failure_category=ArticleFetchFailureCategory.INVALID_URL,
            error_message="Invalid article URL. Expected an absolute http or https URL.",
        )

    try:
        fetched = fetch_html(link=link, retry_policy=retry_policy)
    except _FetchDiagnosticError as exc:
        return ArticleFetchResult(
            link=link,
            publisher_domain=publisher_domain,
            final_url=exc.final_url,
            http_status_code=exc.http_status_code,
            content_type=exc.content_type,
            attempt_count=exc.attempt_count,
            raw_text="",
            cleaned_text="",
            fetch_status=ArticleFetchStatus.FETCH_FAILED,
            retryable=exc.retryable,
            failure_category=exc.failure_category,
            error_message=exc.message,
        )
    except RequestException as exc:
        diagnostic_error = _request_exception_to_diagnostic_error(
            exc,
            retry_policy=retry_policy or FetchRetryPolicy(),
            attempt_index=0,
        )
        return ArticleFetchResult(
            link=link,
            publisher_domain=publisher_domain,
            final_url=diagnostic_error.final_url,
            http_status_code=diagnostic_error.http_status_code,
            content_type=diagnostic_error.content_type,
            attempt_count=diagnostic_error.attempt_count,
            raw_text="",
            cleaned_text="",
            fetch_status=ArticleFetchStatus.FETCH_FAILED,
            retryable=diagnostic_error.retryable,
            failure_category=diagnostic_error.failure_category,
            error_message=diagnostic_error.message,
        )
    except Exception as exc:
        return ArticleFetchResult(
            link=link,
            publisher_domain=publisher_domain,
            attempt_count=1,
            raw_text="",
            cleaned_text="",
            fetch_status=ArticleFetchStatus.FETCH_FAILED,
            retryable=False,
            failure_category=ArticleFetchFailureCategory.UNEXPECTED_ERROR,
            error_message=f"Unexpected fetch error: {exc}",
        )

    try:
        parsed_result = parse_article_text(html=fetched.html, parser=parser)
    except Exception as exc:
        return ArticleFetchResult(
            link=link,
            publisher_domain=publisher_domain,
            final_url=fetched.final_url,
            content_type=fetched.content_type,
            attempt_count=fetched.attempt_count,
            raw_text="",
            cleaned_text="",
            fetch_status=ArticleFetchStatus.PARSE_FAILED,
            retryable=False,
            failure_category=ArticleFetchFailureCategory.PARSE_ERROR,
            error_message=f"Failed to parse article HTML: {exc}",
        )

    if not parsed_result.text.strip():
        return ArticleFetchResult(
            link=link,
            publisher_domain=publisher_domain,
            final_url=fetched.final_url,
            content_type=fetched.content_type,
            attempt_count=fetched.attempt_count,
            extraction_source=parsed_result.source,
            raw_text="",
            cleaned_text="",
            fetch_status=ArticleFetchStatus.PARSE_FAILED,
            retryable=False,
            failure_category=ArticleFetchFailureCategory.EMPTY_EXTRACT,
            error_message=(
                "Fetched article HTML successfully, but could not extract usable article text. "
                "The page may be heavily scripted, paywalled, or structured in an unsupported way."
            ),
        )

    return ArticleFetchResult(
        link=link,
        publisher_domain=publisher_domain,
        final_url=fetched.final_url,
        content_type=fetched.content_type,
        extraction_source=parsed_result.source,
        attempt_count=fetched.attempt_count,
        raw_text=parsed_result.text,
        cleaned_text=parsed_result.text,
        fetch_status=ArticleFetchStatus.SUCCESS,
        retryable=False,
        failure_category=None,
        error_message=None,
    )


def _fetch_html_once(
    *,
    session: Session,
    link: str,
    timeout_seconds: int,
    user_agent: str,
    verify: str | bool,
) -> Response:
    return session.get(
        link,
        headers=_build_headers(link=link, user_agent=user_agent),
        timeout=(5, timeout_seconds),
        allow_redirects=True,
        verify=verify,
    )


def _build_headers(*, link: str, user_agent: str) -> dict[str, str]:
    parsed = urlparse(link)
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else link
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": origin,
    }


@lru_cache(maxsize=1)
def _get_http_session() -> Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        max_retries=Retry(
            total=0,
            connect=0,
            read=0,
            status=0,
            redirect=3,
            allowed_methods=frozenset({"GET", "HEAD"}),
            raise_on_status=False,
        )
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _tls_verify_value() -> str | bool:
    if certifi is not None:
        return certifi.where()
    return True


def _raise_for_response_status(response: Response) -> None:
    response.raise_for_status()


def _response_to_html(response: Response) -> tuple[str, str | None]:
    content_type = response.headers.get("Content-Type", "")
    if content_type and not any(
        allowed in content_type.lower() for allowed in HTML_CONTENT_TYPES
    ):
        raise _FetchDiagnosticError(
            message=f"Unsupported content type for article fetch: {content_type}",
            retryable=False,
            failure_category=ArticleFetchFailureCategory.UNSUPPORTED_CONTENT_TYPE,
            attempt_count=1,
            content_type=content_type,
            final_url=str(response.url),
        )

    if not response.encoding:
        response.encoding = response.apparent_encoding or "utf-8"
    return response.text, content_type or None


def _format_http_error_message(
    status_code: int,
    *,
    retry_policy: FetchRetryPolicy | None = None,
) -> str:
    active_retry_policy = retry_policy or FetchRetryPolicy()

    if status_code == 401:
        return (
            "HTTP error while fetching article: 401. "
            "The publisher likely requires authentication, a session, or a paid API."
        )
    if status_code == 403:
        return (
            "HTTP error while fetching article: 403. "
            "The publisher likely blocked the request because of anti-bot or access rules."
        )
    if status_code == 429:
        return (
            "HTTP error while fetching article: 429. "
            "The publisher rate-limited the fetch request. This error is retryable."
        )
    if status_code in active_retry_policy.retryable_http_status_codes:
        return (
            f"HTTP error while fetching article: {status_code}. "
            "The upstream publisher or network returned a retryable server-side failure."
        )
    return f"HTTP error while fetching article: {status_code}"


def _request_exception_to_diagnostic_error(
    error: RequestException,
    *,
    retry_policy: FetchRetryPolicy,
    attempt_index: int,
) -> _FetchDiagnosticError:
    if isinstance(error, _FetchDiagnosticError):
        return error

    response = getattr(error, "response", None)
    status_code = response.status_code if response is not None else None
    final_url = str(response.url) if response is not None and getattr(response, "url", None) else None
    content_type = response.headers.get("Content-Type") if response is not None else None
    retryable = retry_policy.should_retry(error, attempt_index=attempt_index)

    if status_code is not None:
        if retry_policy.is_access_block(error):
            category = ArticleFetchFailureCategory.ACCESS_BLOCKED
        elif retry_policy.is_rate_limited(error):
            category = ArticleFetchFailureCategory.RATE_LIMITED
        else:
            category = ArticleFetchFailureCategory.NETWORK_ERROR
        message = _format_http_error_message(status_code, retry_policy=retry_policy)
    else:
        reason_text = str(error).lower()
        if "ssl" in reason_text or "certificate" in reason_text or "tls" in reason_text:
            category = ArticleFetchFailureCategory.SSL_ERROR
        else:
            category = ArticleFetchFailureCategory.NETWORK_ERROR
        message = f"Network error while fetching article: {error}"

    return _FetchDiagnosticError(
        message=message,
        retryable=retryable,
        failure_category=category,
        attempt_count=attempt_index + 1,
        http_status_code=status_code,
        content_type=content_type,
        final_url=final_url,
    )
