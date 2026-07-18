# =============================================================================
# Stage 1: Builder - Install dependencies with uv
# =============================================================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

# --no-dev + --no-install-project: install deps only, skip dev/test packages
RUN uv sync --frozen --no-dev --no-install-project

# =============================================================================
# Stage 2: Runtime - Minimal final image
# =============================================================================
FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/sh --create-home appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/
COPY start.sh ./
RUN chmod +x start.sh

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
