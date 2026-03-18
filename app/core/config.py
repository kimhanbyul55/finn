from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppSettings:
    database_backend: str
    postgres_dsn: str | None
    sqlite_path: str | None
    app_host: str
    app_port: int
    worker_poll_interval_seconds: float
    basic_auth_user: str | None
    basic_auth_password: str | None

    @property
    def basic_auth_enabled(self) -> bool:
        return bool(self.basic_auth_user and self.basic_auth_password)


def get_settings() -> AppSettings:
    """Load application settings from environment variables."""
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
        basic_auth_user=os.getenv("BASIC_AUTH_USER"),
        basic_auth_password=os.getenv("BASIC_AUTH_PASSWORD"),
    )
