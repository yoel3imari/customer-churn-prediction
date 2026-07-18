"""FastAPI application factory for the Churn Prediction API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import _get_predictor, router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the model on startup so the first request isn't slow."""
    try:
        _get_predictor()
        logger.info("ChurnPredictor loaded and cached on startup")
    except Exception:
        logger.warning(
            "Could not load ChurnPredictor on startup — "
            "will lazy-load on first request",
            exc_info=True,
        )
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Churn Prediction API",
        description="Real-time customer churn inference with optional SHAP explainability",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
