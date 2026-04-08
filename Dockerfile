FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Runtime-only packages. Keep this list small so the image remains lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the application package first so secrets are never baked into the image.
COPY pyproject.toml README.md ./
COPY app ./app
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install .

# Create a non-root runtime user.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# Web service default. Worker service should override this command with:
# python -m app.workers.enrichment_worker --poll-interval 5
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
