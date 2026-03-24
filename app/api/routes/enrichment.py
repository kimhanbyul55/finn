from fastapi import APIRouter

from app.schemas.enrichment import (
    ArticleEnrichmentRequest,
    ArticleEnrichmentResponse,
    DirectTextEnrichmentRequest,
)
from app.services.enrichment_service import EnrichmentService


router = APIRouter(tags=["enrichment"])
service = EnrichmentService()


@router.post(
    "/articles/enrich",
    response_model=ArticleEnrichmentResponse,
    summary="Enrich a financial news article",
)
async def enrich_article(
    payload: ArticleEnrichmentRequest,
) -> ArticleEnrichmentResponse:
    return await service.enrich_article(payload)


@router.post(
    "/articles/enrich-text",
    response_model=ArticleEnrichmentResponse,
    summary="Enrich licensed article or summary text without crawling the URL",
)
async def enrich_article_text(
    payload: DirectTextEnrichmentRequest,
) -> ArticleEnrichmentResponse:
    return await service.enrich_article_text(payload)
