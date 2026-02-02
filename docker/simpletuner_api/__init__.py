"""
SimpleTuner API Package.

Modular structure for the SimpleTuner FLUX.2 Training and Inference API.

This package decomposes the monolithic simpletuner_api.py (2,314 lines) into:
- config.py: Path configuration and global state
- training.py: SimpleTuner configuration and training logic
- inference.py: FLUX.2-dev pipeline loading and generation logic
- vertex.py: Vertex AI batch mode functions
- endpoints/training.py: Training REST API endpoints
- endpoints/inference.py: Image generation REST API endpoints

Usage:
    from simpletuner_api import create_app
    app = create_app()

Or for direct import:
    from simpletuner_api.config import setup_paths
    from simpletuner_api.training import create_simpletuner_config
"""

from fastapi import FastAPI

from .config import setup_paths, get_runtime_paths
from .training import create_simpletuner_config, create_multidatabackend_config, run_training
from .inference import (
    get_inference_pipeline,
    get_inference_pipeline_sync,
    is_quantized_model_cached,
    is_pipeline_ready,
    preload_inference_pipeline,
)
from .endpoints.training import router as training_router
from .endpoints.inference import router as inference_router

__all__ = [
    "create_app",
    "setup_paths",
    "get_runtime_paths",
    "create_simpletuner_config",
    "create_multidatabackend_config",
    "run_training",
    "get_inference_pipeline",
    "get_inference_pipeline_sync",
    "is_quantized_model_cached",
    "is_pipeline_ready",
    "preload_inference_pipeline",
]


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app with all routes registered.
    """
    app = FastAPI(title="SimpleTuner FLUX.2 Training API")

    # Include training endpoints
    app.include_router(training_router)

    # Include inference endpoints
    app.include_router(inference_router)

    return app
