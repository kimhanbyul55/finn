from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.enrichment import router as enrichment_router
from app.api.routes.health import router as health_router
from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.web import router as web_router
from app.core import get_settings
from app.core.auth import (
    basic_auth_required,
    is_basic_auth_authorized,
    unauthorized_basic_auth_response,
)
from app.core.logging import configure_logging
from app.db import initialize_database_backend


configure_logging()
initialize_database_backend()
settings = get_settings()

WEB_DIR = Path(__file__).resolve().parent / "web"


app = FastAPI(
    title="Financial News Gen AI Service",
    version="0.1.0",
    description="Enrichment API for financial news articles.",
)


@app.middleware("http")
async def basic_auth_middleware(request, call_next):
    if basic_auth_required(request) and not is_basic_auth_authorized(request):
        return unauthorized_basic_auth_response()
    return await call_next(request)

app.include_router(health_router)
app.include_router(web_router)
app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(enrichment_router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
