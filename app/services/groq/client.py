from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

import requests

from app.core import get_settings
from app.core.logging import log_event

logger = logging.getLogger(__name__)

_RETRY_AFTER_PATTERN = re.compile(r"Please try again in ([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
_groq_log_context: ContextVar[dict[str, object]] = ContextVar(
    "groq_log_context",
    default={},
)


def groq_is_enabled() -> bool:
    return bool(get_settings().groq_api_key)


@contextmanager
def groq_log_context(**fields: object) -> Iterator[None]:
    """Attach request/job metadata to Groq logs emitted in this execution context."""
    current = _groq_log_context.get()
    merged = {**current, **{key: value for key, value in fields.items() if value is not None}}
    token = _groq_log_context.set(merged)
    try:
        yield
    finally:
        _groq_log_context.reset(token)


def groq_chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.2,
    request_label: str | None = None,
) -> str:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("Groq API key is not configured.")

    label = request_label or "unknown"
    url = _build_chat_completions_url(settings.groq_api_base_url)
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    log_event(
        logger,
        logging.INFO,
        "groq_request_started",
        **_groq_context_fields(),
        request_label=label,
        model=model,
        system_prompt_chars=len(system_prompt),
        user_prompt_chars=len(user_prompt),
    )

    attempt = 0
    max_attempts = 2
    while True:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.groq_timeout_seconds,
        )
        try:
            response.raise_for_status()
            break
        except requests.HTTPError as exc:
            retry_after_seconds = _extract_retry_after_seconds(response)
            error_payload = _safe_json(response)
            error_message = (
                ((error_payload.get("error") or {}).get("message"))
                if isinstance(error_payload, dict)
                else None
            )
            log_event(
                logger,
                logging.WARNING,
                "groq_request_failed",
                **_groq_context_fields(),
                request_label=label,
                model=model,
                status_code=response.status_code,
                retry_after_seconds=retry_after_seconds,
                error_message=error_message,
                attempt=attempt + 1,
            )
            can_retry_rate_limit = (
                response.status_code == 429
                and retry_after_seconds is not None
                and retry_after_seconds <= settings.groq_retry_after_max_seconds
                and not _is_daily_token_limit_error(error_message)
                and attempt + 1 < max_attempts
            )
            if can_retry_rate_limit:
                time.sleep(retry_after_seconds)
                attempt += 1
                continue
            raise exc

    payload = response.json()
    usage = payload.get("usage") or {}
    log_event(
        logger,
        logging.INFO,
        "groq_request_completed",
        **_groq_context_fields(),
        request_label=label,
        model=model,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("Groq response contained no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Groq response contained no message content.")
    return content.strip()


def _build_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _groq_context_fields() -> dict[str, object]:
    return dict(_groq_log_context.get())


def _extract_retry_after_seconds(response: requests.Response) -> float | None:
    retry_after_header = response.headers.get("Retry-After")
    if retry_after_header:
        try:
            return float(retry_after_header)
        except ValueError:
            pass

    payload = _safe_json(response)
    if isinstance(payload, dict):
        error = payload.get("error") or {}
        message = error.get("message")
        if isinstance(message, str):
            match = _RETRY_AFTER_PATTERN.search(message)
            if match:
                return float(match.group(1))
    return None


def _is_daily_token_limit_error(error_message: str | None) -> bool:
    if not error_message:
        return False
    normalized = error_message.lower()
    return "tokens per day" in normalized or "tpd" in normalized


def _safe_json(response: requests.Response) -> dict[str, object] | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None
