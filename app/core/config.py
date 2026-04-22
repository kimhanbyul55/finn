from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AppSettings:
    database_backend: str
    postgres_dsn: str | None
    sqlite_path: str | None
    app_host: str
    app_port: int
    worker_poll_interval_seconds: float
    worker_idle_log_interval_seconds: float
    enable_job_process_api: bool
    use_worker_backed_direct_enrichment: bool
    enable_inline_xai: bool
    xai_backend: str
    direct_enrichment_wait_timeout_seconds: float
    direct_enrichment_poll_interval_seconds: float
    basic_auth_user: str | None
    basic_auth_password: str | None
    deepl_api_key: str | None
    deepl_api_base_url: str
    deepl_target_lang: str
    deepl_timeout_seconds: float
    gemini_api_key: str | None
    gemini_api_base_url: str
    enable_gemini_summary: bool
    enable_gemini_translation: bool
    enable_gemini_translation_repair: bool
    gemini_summary_model: str
    gemini_translation_model: str
    gemini_timeout_seconds: float
    gemini_retry_after_max_seconds: float
    gemini_summary_soft_char_limit: int
    gemini_summary_hard_char_limit: int
    gemini_translation_char_limit: int
    localized_xai_highlight_limit: int
    fetch_blocked_domains: tuple[str, ...]

    @property
    def basic_auth_enabled(self) -> bool:
        return bool(self.basic_auth_user and self.basic_auth_password)


def get_settings() -> AppSettings:
    """Load application settings from environment variables."""
    running_on_render = _env_flag("RENDER", default=False)
    return AppSettings(
        database_backend=(
            os.getenv("GENAI_DATABASE_BACKEND")
            or os.getenv("DATABASE_BACKEND")
            or "sqlite"
        ).lower(),
        postgres_dsn=os.getenv("GENAI_POSTGRES_DSN") or os.getenv("DATABASE_URL"),
        sqlite_path=os.getenv("GENAI_SQLITE_DB_PATH"),
        app_host=os.getenv("GENAI_APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("GENAI_APP_PORT", "8000")),
        worker_poll_interval_seconds=float(os.getenv("GENAI_WORKER_POLL_INTERVAL", "5")),
        worker_idle_log_interval_seconds=float(
            os.getenv("GENAI_WORKER_IDLE_LOG_INTERVAL", "60")
        ),
        enable_job_process_api=_env_flag(
            "GENAI_ENABLE_JOB_PROCESS_API",
            default=not running_on_render,
        ),
        use_worker_backed_direct_enrichment=_env_flag(
            "GENAI_USE_WORKER_FOR_DIRECT_ENRICHMENT",
            default=running_on_render,
        ),
        enable_inline_xai=_env_flag(
            "GENAI_ENABLE_INLINE_XAI",
            default=False,
        ),
        xai_backend=(os.getenv("GENAI_XAI_BACKEND") or "attention").strip().lower(),
        direct_enrichment_wait_timeout_seconds=float(
            os.getenv("GENAI_DIRECT_ENRICHMENT_WAIT_TIMEOUT", "30")
        ),
        direct_enrichment_poll_interval_seconds=float(
            os.getenv("GENAI_DIRECT_ENRICHMENT_POLL_INTERVAL", "0.5")
        ),
        basic_auth_user=os.getenv("BASIC_AUTH_USER"),
        basic_auth_password=os.getenv("BASIC_AUTH_PASSWORD"),
        deepl_api_key=os.getenv("DEEPL_API_KEY"),
        deepl_api_base_url=(
            os.getenv("DEEPL_API_BASE_URL") or "https://api-free.deepl.com"
        ).rstrip("/"),
        deepl_target_lang=(os.getenv("DEEPL_TARGET_LANG") or "KO").strip().upper(),
        deepl_timeout_seconds=float(os.getenv("DEEPL_TIMEOUT_SECONDS", "8")),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        gemini_api_base_url=(
            os.getenv("GEMINI_API_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/"),
        enable_gemini_summary=_env_flag("GENAI_ENABLE_GEMINI_SUMMARY", default=True),
        enable_gemini_translation=_env_flag("GENAI_ENABLE_GEMINI_TRANSLATION", default=False),
        enable_gemini_translation_repair=_env_flag(
            "GENAI_ENABLE_GEMINI_TRANSLATION_REPAIR",
            default=False,
        ),
        gemini_summary_model=(
            os.getenv("GEMINI_SUMMARY_MODEL") or "gemini-1.5-flash"
        ).strip(),
        gemini_translation_model=(
            os.getenv("GEMINI_TRANSLATION_MODEL") or "gemini-1.5-flash"
        ).strip(),
        gemini_timeout_seconds=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "20")),
        gemini_retry_after_max_seconds=float(os.getenv("GEMINI_RETRY_AFTER_MAX_SECONDS", "0")),
        gemini_summary_soft_char_limit=int(os.getenv("GEMINI_SUMMARY_SOFT_CHAR_LIMIT", "3500")),
        gemini_summary_hard_char_limit=int(os.getenv("GEMINI_SUMMARY_HARD_CHAR_LIMIT", "6500")),
        gemini_translation_char_limit=int(os.getenv("GEMINI_TRANSLATION_CHAR_LIMIT", "1200")),
        localized_xai_highlight_limit=int(os.getenv("GENAI_LOCALIZED_XAI_HIGHLIGHT_LIMIT", "2")),
        fetch_blocked_domains=_parse_csv_env(
            "GENAI_FETCH_BLOCKED_DOMAINS",
            default=(
                "finance.yahoo.com",
                "www.finance.yahoo.com",
                "news.yahoo.co.jp",
            ),
        ),
    )


def _parse_csv_env(name: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    return items or default
