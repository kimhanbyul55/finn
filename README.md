# Financial News Gen AI Service

FastAPI-based enrichment service for financial news articles. The service accepts article metadata from a backend system, fetches article content from the original link, and produces AI-enriched analysis such as summary, sentiment, XAI, and mixed/conflict flags.

## Local Postgres Setup

1. Copy `.env.example` to `.env`
2. Start Postgres

```bash
docker compose up -d postgres
```

3. Install dependencies

```bash
pip install -e .
```

4. Run the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

5. Run the worker in a separate terminal

```bash
python3 -m app.workers.enrichment_worker --poll-interval 5
```

## Database Check

You can validate the current DB backend setup without starting the full API:

```bash
python3 -m app.db.check
```

Example output:

```json
{
  "database_backend": "postgres",
  "database_ok": false,
  "database_error": "connection refused",
  "postgres_dsn_configured": true,
  "sqlite_path": null
}
```

If you do not have Docker locally, you can still use the same command after starting Postgres with Homebrew or Postgres.app.

## Smoke Test

You can run a quick BE-style smoke test against a running API instance:

```bash
python3 scripts/smoke_test_enrichment.py \
  --title "Apple shares rise after earnings" \
  --link "https://example.com/article" \
  --ticker AAPL \
  --source Reuters
```

What it does:
- calls `POST /api/v1/news/intake`
- optionally calls `POST /api/v1/jobs/process-next`
- calls `GET /api/v1/news/{news_id}`
- prints the full JSON result for quick debugging

Useful flags:
- `--base-url http://127.0.0.1:8000`
- `--skip-worker`
- `--poll-seconds 2`
- `--news-id custom-id-123`

Detailed validation checklist:
- [`docs/smoke_test_checklist.md`](/Users/sta/Documents/개발1/docs/smoke_test_checklist.md)

## Direct Text Enrichment

If your upstream news provider can send licensed `article_text` or `summary_text`
directly, you can skip URL crawling entirely and call:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/articles/enrich-text" \
  -H "Content-Type: application/json" \
  -d '{
    "news_id": "provider-123",
    "title": "Apple shares rise after earnings",
    "link": "https://provider.example.com/article/123",
    "ticker": ["AAPL"],
    "source": "Licensed Provider",
    "article_text": "Full licensed article text goes here."
  }'
```

You may also send `summary_text` when only a licensed summary/snippet is available.
The service will analyze the provided text directly without fetching the original URL.

If your backend prefers the existing async queue flow, use the intake endpoint below
and then process the queued job with the worker just like URL-based ingestion:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/news/intake-text" \
  -H "Content-Type: application/json" \
  -d '{
    "news_id": "provider-124",
    "title": "Apple shares rise after earnings",
    "link": "https://provider.example.com/article/124",
    "ticker": ["AAPL"],
    "source": "Licensed Provider",
    "summary_text": "Apple shares rose after better-than-expected earnings."
  }'
```

The worker will use the supplied text instead of crawling the URL, and the raw input
text is cleared from the queued raw-news record after processing finishes so that the
service retains metadata and derived analysis rather than provider-supplied body text.

## Domain Matrix

You can save smoke test outputs and aggregate them into a domain-level matrix:

```bash
python3 scripts/smoke_test_enrichment.py \
  --title "Sample title" \
  --link "https://example.com/article" \
  --ticker AAPL \
  --source Reuters > results/apnews-1.json
```

```bash
python3 scripts/build_domain_matrix.py results/*.json --format table
```

This helps you see, by domain:
- success vs fatal failure rates
- which extraction paths are being used
- which failure categories are recurring
- recommended support tier: `primary`, `secondary`, `blocked`, or `investigate`

## Smoke Suite

You can run multiple links from a JSON config and save all results automatically:

```json
[
  {
    "title": "AP sample article",
    "link": "https://apnews.com/article/example",
    "ticker": ["SPY"],
    "source": "AP News"
  }
]
```

```bash
python3 scripts/run_smoke_suite.py \
  --config smoke_suite.json \
  --output-dir results
```

This writes:
- one JSON file per article
- `results/suite_summary.json`
- `results/suite_summary.md`

It also prints the aggregated domain matrix immediately.

## Required Environment Variables

- `GENAI_DATABASE_BACKEND`
  - `sqlite` or `postgres`
- `GENAI_POSTGRES_DSN`
  - required when `GENAI_DATABASE_BACKEND=postgres`
- `GENAI_SQLITE_DB_PATH`
  - optional when using SQLite
- `BASIC_AUTH_USER`
  - optional, enable HTTP Basic Auth when set together with `BASIC_AUTH_PASSWORD`
- `BASIC_AUTH_PASSWORD`
  - optional, enable HTTP Basic Auth when set together with `BASIC_AUTH_USER`

When Basic Auth is enabled:
- `/health` stays open for Render health checks
- `/`, `/docs`, `/openapi.json`, `/api/v1/*`, and static assets require credentials

This is useful for short-lived private sharing on Render without building a full login system.

## Health Check

`GET /health`

Returns:
- app status
- active database backend
- database connectivity result

Example response:

```json
{
  "status": "ok",
  "database_backend": "postgres",
  "database_ok": true,
  "database_error": null
}
```
