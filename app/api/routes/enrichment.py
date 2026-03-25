from fastapi import APIRouter, HTTPException

from app.core import get_settings
from app.schemas.enrichment import (
    ArticleEnrichmentResponse,
    DirectTextEnrichmentRequest,
    FlexibleTextEnrichmentRequest,
)
from app.services.enrichment_service import EnrichmentService


router = APIRouter(tags=["enrichment"])
service = EnrichmentService()
settings = get_settings()


def _ensure_direct_enrichment_enabled() -> None:
    if settings.enable_direct_enrichment_api:
        return
    raise HTTPException(
        status_code=503,
        detail=(
            "Direct enrichment APIs are disabled on this web service instance. "
            "Submit work through /api/v1/news/intake or /api/v1/news/intake-text "
            "and let the dedicated worker service process the job."
        ),
    )


@router.post(
    "/articles/enrich",
    response_model=ArticleEnrichmentResponse,
    summary="Enrich a financial news article",
)
async def enrich_article(
    payload: FlexibleTextEnrichmentRequest,
) -> ArticleEnrichmentResponse:
    _ensure_direct_enrichment_enabled()
    return await service.enrich_article(payload)


@router.post(
    "/articles/enrich-text",
    response_model=ArticleEnrichmentResponse,
    summary="Enrich licensed article or summary text without crawling the URL",
)
async def enrich_article_text(
    payload: DirectTextEnrichmentRequest,
) -> ArticleEnrichmentResponse:
    _ensure_direct_enrichment_enabled()
    return await service.enrich_article_text(payload)
