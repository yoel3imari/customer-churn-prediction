# =============================================================================
# Stage 1: Builder - Install all dependencies with uv (including dev/test)
# =============================================================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install full dependencies (including dev tools for dev/test containers)
RUN uv sync --frozen --no-install-project

# =============================================================================
# Stage 1b: Production Builder - Install runtime-only dependencies
# =============================================================================
FROM builder AS production-builder

RUN uv sync --frozen --no-install-project --no-dev

# =============================================================================
# Stage 2: Runtime - Full dev/test/pipeline runtime image
# =============================================================================
FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/sh --create-home appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/data /app/models /app/reports /app/src /mlflow/artifacts \
    && chown -R appuser:appuser /app /mlflow/artifacts

COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY start.sh ./
RUN chmod +x start.sh && chown appuser:appuser start.sh

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Stage 3: Production - Lean final runtime image (no dev dependencies)
# =============================================================================
FROM runtime AS production

COPY --from=production-builder /app/.venv /app/.venv

