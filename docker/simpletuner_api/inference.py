"""
SimpleTuner Inference Pipeline.

Handles FLUX.2-dev model loading, quantization, and image generation.
"""

import logging
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DEFAULT, HTTP_TIMEOUT_DOWNLOAD, HTTP_TIMEOUT_MODEL_PULL
from . import config as app_config

logger = logging.getLogger(__name__)

# Global inference pipeline (lazy loaded, kept in memory)
_inference_pipeline = None
_inference_lock = None  # Created lazily to avoid event loop binding issues
_pipeline_loading = False  # Flag to track if background loading is in progress
_threading_lock = threading.Lock()  # For sync background tasks

# Quantized model cache paths (on network volume for persistence)
QUANTIZED_CACHE_DIR = f"{app_config.WORKSPACE_PATH}/quantized_flux"
QUANTIZED_TRANSFORMER_PATH = f"{QUANTIZED_CACHE_DIR}/transformer_int8"
QUANTIZED_TEXT_ENCODER_PATH = f"{QUANTIZED_CACHE_DIR}/text_encoder_int8"
QUANTIZED_MARKER_PATH = f"{QUANTIZED_CACHE_DIR}/.quantized_complete"

# GCS cache paths for Vertex AI (ephemeral containers need cloud storage for persistence)
GCS_QUANTIZED_CACHE_PREFIX = "model-cache/flux2-dev-int8-quanto"

# Background generation jobs
generation_jobs: dict = {}

# Preload status
preload_status = {"status": "idle", "progress": "", "error": None}


def _get_gcs_quantized_paths() -> dict:
    """Get GCS paths for quantized model cache (Vertex AI mode only)."""
    bucket = os.environ.get("GCS_TRAINING_BUCKET", "kestrel-training")
    return {
        "bucket": bucket,
        "transformer": f"gs://{bucket}/{GCS_QUANTIZED_CACHE_PREFIX}/transformer_int8.tar.gz",
        "text_encoder": f"gs://{bucket}/{GCS_QUANTIZED_CACHE_PREFIX}/text_encoder_int8.tar.gz",
        "marker": f"gs://{bucket}/{GCS_QUANTIZED_CACHE_PREFIX}/.quantized_complete",
    }


async def _gcs_cache_exists() -> bool:
    """Check if quantized model cache exists in GCS (Vertex AI mode)."""
    import subprocess

    paths = _get_gcs_quantized_paths()
    marker_path = paths["marker"]

    try:
        result = subprocess.run(
            ["gsutil", "stat", marker_path],
            capture_output=True,
            text=True,
            timeout=HTTP_TIMEOUT_DEFAULT,
        )
        exists = result.returncode == 0
        if exists:
            logger.info(f"✅ GCS quantized cache found: {marker_path}")
        return exists
    except Exception as e:
        logger.warning(f"Failed to check GCS cache: {e}")
        return False


async def _download_gcs_quantized_cache() -> bool:
    """
    Download pre-quantized model from GCS to local disk (Vertex AI mode).

    Downloads tarball and extracts to QUANTIZED_CACHE_DIR.
    Returns True if successful, False otherwise.
    """
    import subprocess
    import tarfile

    paths = _get_gcs_quantized_paths()

    # Ensure local cache dir exists
    os.makedirs(QUANTIZED_CACHE_DIR, exist_ok=True)

    try:
        # Download transformer tarball
        logger.info(f"⬇️ Downloading quantized transformer from GCS...")
        transformer_tar = f"{QUANTIZED_CACHE_DIR}/transformer_int8.tar.gz"
        result = subprocess.run(
            ["gsutil", "-q", "cp", paths["transformer"], transformer_tar],
            capture_output=True,
            text=True,
            timeout=HTTP_TIMEOUT_DOWNLOAD,
        )
        if result.returncode != 0:
            logger.error(f"Failed to download transformer: {result.stderr}")
            return False

        # Extract transformer
        logger.info("Extracting transformer...")
        with tarfile.open(transformer_tar, "r:gz") as tar:
            tar.extractall(QUANTIZED_CACHE_DIR)
        os.remove(transformer_tar)

        # Download text encoder tarball
        logger.info(f"⬇️ Downloading quantized text encoder from GCS...")
        text_encoder_tar = f"{QUANTIZED_CACHE_DIR}/text_encoder_int8.tar.gz"
        result = subprocess.run(
            ["gsutil", "-q", "cp", paths["text_encoder"], text_encoder_tar],
            capture_output=True,
            text=True,
            timeout=HTTP_TIMEOUT_DOWNLOAD,
        )
        if result.returncode != 0:
            logger.error(f"Failed to download text encoder: {result.stderr}")
            return False

        # Extract text encoder
        logger.info("Extracting text encoder...")
        with tarfile.open(text_encoder_tar, "r:gz") as tar:
            tar.extractall(QUANTIZED_CACHE_DIR)
        os.remove(text_encoder_tar)

        # Download marker file
        subprocess.run(
            ["gsutil", "-q", "cp", paths["marker"], QUANTIZED_MARKER_PATH],
            capture_output=True,
            timeout=HTTP_TIMEOUT_DEFAULT,
        )

        logger.info(f"✅ GCS quantized cache downloaded to {QUANTIZED_CACHE_DIR}")
        return True

    except Exception as e:
        logger.error(f"Failed to download GCS quantized cache: {e}")
        # Clean up partial download
        if os.path.exists(QUANTIZED_CACHE_DIR):
            shutil.rmtree(QUANTIZED_CACHE_DIR, ignore_errors=True)
        return False


async def _upload_gcs_quantized_cache() -> bool:
    """
    Upload quantized model to GCS for future Vertex AI jobs.

    Creates tarballs and uploads to GCS bucket.
    Returns True if successful, False otherwise.
    """
    import subprocess
    import tarfile

    paths = _get_gcs_quantized_paths()

    if not is_quantized_model_cached():
        logger.warning("No local quantized cache to upload")
        return False

    try:
        # Create transformer tarball
        logger.info("📦 Creating transformer tarball...")
        transformer_tar = f"{QUANTIZED_CACHE_DIR}/transformer_int8.tar.gz"
        with tarfile.open(transformer_tar, "w:gz") as tar:
            tar.add(QUANTIZED_TRANSFORMER_PATH, arcname="transformer_int8")

        # Upload transformer
        logger.info(f"⬆️ Uploading quantized transformer to GCS...")
        result = subprocess.run(
            ["gsutil", "-q", "cp", transformer_tar, paths["transformer"]],
            capture_output=True,
            text=True,
            timeout=HTTP_TIMEOUT_MODEL_PULL,
        )
        os.remove(transformer_tar)
        if result.returncode != 0:
            logger.error(f"Failed to upload transformer: {result.stderr}")
            return False

        # Create text encoder tarball
        logger.info("📦 Creating text encoder tarball...")
        text_encoder_tar = f"{QUANTIZED_CACHE_DIR}/text_encoder_int8.tar.gz"
        with tarfile.open(text_encoder_tar, "w:gz") as tar:
            tar.add(QUANTIZED_TEXT_ENCODER_PATH, arcname="text_encoder_int8")

        # Upload text encoder
        logger.info(f"⬆️ Uploading quantized text encoder to GCS...")
        result = subprocess.run(
            ["gsutil", "-q", "cp", text_encoder_tar, paths["text_encoder"]],
            capture_output=True,
            text=True,
            timeout=HTTP_TIMEOUT_MODEL_PULL,
        )
        os.remove(text_encoder_tar)
        if result.returncode != 0:
            logger.error(f"Failed to upload text encoder: {result.stderr}")
            return False

        # Upload marker file
        result = subprocess.run(
            ["gsutil", "-q", "cp", QUANTIZED_MARKER_PATH, paths["marker"]],
            capture_output=True,
            timeout=HTTP_TIMEOUT_DEFAULT,
        )

        logger.info(f"✅ Quantized cache uploaded to GCS: gs://{paths['bucket']}/{GCS_QUANTIZED_CACHE_PREFIX}/")
        return True

    except Exception as e:
        logger.error(f"Failed to upload GCS quantized cache: {e}")
        return False


def is_quantized_model_cached() -> bool:
    """Check if quantized model components are cached on disk."""
    return (
        os.path.isdir(QUANTIZED_TRANSFORMER_PATH) and
        os.path.isdir(QUANTIZED_TEXT_ENCODER_PATH) and
        os.path.exists(QUANTIZED_MARKER_PATH)
    )


def _load_inference_pipeline_impl(save_quantized: bool = True):
    """
    Internal synchronous implementation of pipeline loading.

    MUST be called while holding _threading_lock.
    This is for RunPod mode only - no GCS/Vertex AI support.
    """
    global _inference_pipeline

    import torch
    from diffusers import Flux2Pipeline  # REQUIRES diffusers from git, not stable!
    from optimum.quanto import freeze, qint8, quantize

    # Ensure directories exist
    paths = app_config.get_runtime_paths()
    tmp_path = paths["tmp_path"]
    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(QUANTIZED_CACHE_DIR, exist_ok=True)

    # Check if we have cached quantized model
    if is_quantized_model_cached():
        logger.info("🚀 Loading CACHED quantized FLUX.2-dev from disk...")
        logger.info(f"   Cache location: {QUANTIZED_CACHE_DIR}")

        # Load base pipeline first (for VAE, scheduler, etc.)
        _inference_pipeline = Flux2Pipeline.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            torch_dtype=torch.bfloat16,
        )

        # Load pre-quantized transformer
        logger.info("Loading cached quantized transformer...")
        from diffusers import FluxTransformer2DModel
        _inference_pipeline.transformer = FluxTransformer2DModel.from_pretrained(
            QUANTIZED_TRANSFORMER_PATH,
            torch_dtype=torch.bfloat16,
        )

        # Load pre-quantized text encoder
        logger.info("Loading cached quantized text encoder...")
        from transformers import AutoModel
        _inference_pipeline.text_encoder = AutoModel.from_pretrained(
            QUANTIZED_TEXT_ENCODER_PATH,
            torch_dtype=torch.bfloat16,
        )

        # Move to GPU
        _inference_pipeline.to("cuda")

        logger.info("✅ FLUX.2-dev loaded from CACHE (skip quantization)")
        return _inference_pipeline

    # No cache - need to quantize from scratch
    logger.info("Loading FLUX.2-dev pipeline for inference (int8-quanto)...")
    logger.info("⚠️ First-time quantization takes 10-15 minutes.")
    logger.info("   Model will be cached to disk for faster future loads.")

    # Load FLUX.2-dev in BF16 first
    _inference_pipeline = Flux2Pipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        torch_dtype=torch.bfloat16,
    )

    # Quantize transformer and text encoder to int8 (same as training)
    logger.info("Quantizing transformer to int8...")
    quantize(_inference_pipeline.transformer, weights=qint8)
    freeze(_inference_pipeline.transformer)

    logger.info("Quantizing text encoder to int8...")
    quantize(_inference_pipeline.text_encoder, weights=qint8)
    freeze(_inference_pipeline.text_encoder)

    # Save quantized model to disk for future loads
    if save_quantized:
        logger.info(f"💾 Saving quantized model to {QUANTIZED_CACHE_DIR}...")
        try:
            # Save transformer
            logger.info("Saving quantized transformer...")
            _inference_pipeline.transformer.save_pretrained(QUANTIZED_TRANSFORMER_PATH)

            # Save text encoder
            logger.info("Saving quantized text encoder...")
            _inference_pipeline.text_encoder.save_pretrained(QUANTIZED_TEXT_ENCODER_PATH)

            # Write marker file to indicate complete save
            with open(QUANTIZED_MARKER_PATH, "w") as f:
                f.write(f"quantized_at={datetime.utcnow().isoformat()}\n")
                f.write(f"model=black-forest-labs/FLUX.2-dev\n")
                f.write(f"quantization=int8-quanto\n")

            # Get cache size
            cache_size_gb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, filenames in os.walk(QUANTIZED_CACHE_DIR)
                for f in filenames
            ) / (1024**3)

            logger.info(f"✅ Quantized model saved to disk ({cache_size_gb:.1f} GB)")
            logger.info(f"   Future loads will skip quantization (~10 min saved)")

        except Exception as e:
            logger.warning(f"Failed to save quantized model to disk: {e}")
            logger.warning("Will quantize again on next load")
            # Clean up partial save
            if os.path.exists(QUANTIZED_MARKER_PATH):
                os.remove(QUANTIZED_MARKER_PATH)

    # Move to GPU - no CPU offload needed with quantization
    _inference_pipeline.to("cuda")

    logger.info("✅ FLUX.2-dev pipeline loaded (int8-quanto, ~50GB VRAM)")
    return _inference_pipeline


def get_inference_pipeline_sync(save_quantized: bool = True):
    """
    Synchronous version for background tasks.
    Uses threading lock to avoid asyncio event loop issues.

    This is the preferred method for background threads (like async generation jobs).
    Does NOT support Vertex AI GCS caching - use get_inference_pipeline() for that.
    """
    global _inference_pipeline

    with _threading_lock:
        if _inference_pipeline is not None:
            return _inference_pipeline
        return _load_inference_pipeline_impl(save_quantized)


async def get_inference_pipeline(save_quantized: bool = True):
    """
    Lazy-load FLUX.2-dev pipeline for inference with int8-quanto quantization.

    CRITICAL: Requires diffusers installed from git (not stable release).
    Flux2Pipeline is NOT in diffusers 0.36.0 - see docs/architecture/RUNPOD_LORA_TRAINING.md

    Uses same quantization as training (~50GB) to avoid CPU offload (faster).

    CACHING BEHAVIOR:
    RunPod Mode (persistent /workspace volume):
    - First load: Quantizes model (~10-15 min), saves to /workspace/quantized_flux/
    - Subsequent loads: Loads pre-quantized model from disk (~2-3 min)
    - Memory: Pipeline kept in memory for fastest inference

    Vertex AI Mode (ephemeral containers, VERTEX_AI_MODE=true):
    - First job: Quantizes model (~10-15 min), saves locally, uploads to GCS
    - Subsequent jobs: Downloads from GCS (~2-3 min), loads from disk
    - GCS path: gs://{GCS_TRAINING_BUCKET}/model-cache/flux2-dev-int8-quanto/
    - Memory: Pipeline kept in memory for fastest inference during job

    Args:
        save_quantized: If True, save quantized model to disk for future loads
    """
    global _inference_pipeline

    # Use threading lock for async context too (simpler than managing async lock lifecycle)
    with _threading_lock:
        if _inference_pipeline is not None:
            return _inference_pipeline

        try:
            import torch
            from diffusers import Flux2Pipeline  # REQUIRES diffusers from git, not stable!
            from optimum.quanto import freeze, qint8, quantize

            # Ensure directories exist
            paths = app_config.get_runtime_paths()
            tmp_path = paths["tmp_path"]
            os.makedirs(tmp_path, exist_ok=True)
            os.makedirs(QUANTIZED_CACHE_DIR, exist_ok=True)

            # Vertex AI mode: Check GCS cache if local cache doesn't exist
            is_vertex_mode = os.environ.get("VERTEX_AI_MODE", "").lower() == "true"
            if is_vertex_mode and not is_quantized_model_cached():
                logger.info("🔍 Vertex AI mode: Checking GCS for quantized cache...")
                # Need to run async check synchronously in this context
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, use run_in_executor
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        gcs_exists = pool.submit(
                            lambda: asyncio.run(_gcs_cache_exists())
                        ).result()
                else:
                    gcs_exists = asyncio.run(_gcs_cache_exists())

                if gcs_exists:
                    logger.info("📥 Found GCS cache, downloading...")
                    if loop.is_running():
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            downloaded = pool.submit(
                                lambda: asyncio.run(_download_gcs_quantized_cache())
                            ).result()
                    else:
                        downloaded = asyncio.run(_download_gcs_quantized_cache())
                    if downloaded:
                        logger.info("✅ GCS cache downloaded successfully")
                    else:
                        logger.warning("⚠️ GCS cache download failed, will quantize from scratch")
                else:
                    logger.info("📭 No GCS cache found, will quantize and upload")

            # Check if we have cached quantized model (local or just downloaded from GCS)
            if is_quantized_model_cached():
                logger.info("🚀 Loading CACHED quantized FLUX.2-dev from disk...")
                logger.info(f"   Cache location: {QUANTIZED_CACHE_DIR}")

                # Load base pipeline first (for VAE, scheduler, etc.)
                _inference_pipeline = Flux2Pipeline.from_pretrained(
                    "black-forest-labs/FLUX.2-dev",
                    torch_dtype=torch.bfloat16,
                )

                # Load pre-quantized transformer
                logger.info("Loading cached quantized transformer...")
                from diffusers import FluxTransformer2DModel
                _inference_pipeline.transformer = FluxTransformer2DModel.from_pretrained(
                    QUANTIZED_TRANSFORMER_PATH,
                    torch_dtype=torch.bfloat16,
                )

                # Load pre-quantized text encoder
                logger.info("Loading cached quantized text encoder...")
                from transformers import AutoModel
                _inference_pipeline.text_encoder = AutoModel.from_pretrained(
                    QUANTIZED_TEXT_ENCODER_PATH,
                    torch_dtype=torch.bfloat16,
                )

                # Move to GPU
                _inference_pipeline.to("cuda")

                logger.info("✅ FLUX.2-dev loaded from CACHE (skip quantization)")
                return _inference_pipeline

            # No cache - need to quantize from scratch
            logger.info("Loading FLUX.2-dev pipeline for inference (int8-quanto)...")
            logger.info("⚠️ First-time quantization takes 10-15 minutes.")
            logger.info("   Model will be cached to disk for faster future loads.")

            # Load FLUX.2-dev in BF16 first
            _inference_pipeline = Flux2Pipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=torch.bfloat16,
            )

            # Quantize transformer and text encoder to int8 (same as training)
            logger.info("Quantizing transformer to int8...")
            quantize(_inference_pipeline.transformer, weights=qint8)
            freeze(_inference_pipeline.transformer)

            logger.info("Quantizing text encoder to int8...")
            quantize(_inference_pipeline.text_encoder, weights=qint8)
            freeze(_inference_pipeline.text_encoder)

            # Save quantized model to disk for future loads
            if save_quantized:
                logger.info(f"💾 Saving quantized model to {QUANTIZED_CACHE_DIR}...")
                try:
                    # Save transformer
                    logger.info("Saving quantized transformer...")
                    _inference_pipeline.transformer.save_pretrained(QUANTIZED_TRANSFORMER_PATH)

                    # Save text encoder
                    logger.info("Saving quantized text encoder...")
                    _inference_pipeline.text_encoder.save_pretrained(QUANTIZED_TEXT_ENCODER_PATH)

                    # Write marker file to indicate complete save
                    with open(QUANTIZED_MARKER_PATH, "w") as f:
                        f.write(f"quantized_at={datetime.utcnow().isoformat()}\n")
                        f.write(f"model=black-forest-labs/FLUX.2-dev\n")
                        f.write(f"quantization=int8-quanto\n")

                    # Get cache size
                    cache_size_gb = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, dn, filenames in os.walk(QUANTIZED_CACHE_DIR)
                        for f in filenames
                    ) / (1024**3)

                    logger.info(f"✅ Quantized model saved to disk ({cache_size_gb:.1f} GB)")
                    logger.info(f"   Future loads will skip quantization (~10 min saved)")

                    # Vertex AI mode: Upload to GCS for future jobs
                    if is_vertex_mode:
                        logger.info("☁️ Vertex AI mode: Uploading quantized cache to GCS...")
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as pool:
                                    uploaded = pool.submit(
                                        lambda: asyncio.run(_upload_gcs_quantized_cache())
                                    ).result()
                            else:
                                uploaded = asyncio.run(_upload_gcs_quantized_cache())
                            if uploaded:
                                logger.info("✅ GCS cache uploaded - future Vertex AI jobs will be faster!")
                            else:
                                logger.warning("⚠️ GCS upload failed - next job will re-quantize")
                        except Exception as upload_err:
                            logger.warning(f"GCS upload failed: {upload_err}")

                except Exception as e:
                    logger.warning(f"Failed to save quantized model to disk: {e}")
                    logger.warning("Will quantize again on next load")
                    # Clean up partial save
                    if os.path.exists(QUANTIZED_MARKER_PATH):
                        os.remove(QUANTIZED_MARKER_PATH)

            # Move to GPU - no CPU offload needed with quantization
            _inference_pipeline.to("cuda")

            logger.info("✅ FLUX.2-dev pipeline loaded (int8-quanto, ~50GB VRAM)")
            return _inference_pipeline

        except Exception as e:
            logger.error(f"Failed to load inference pipeline: {e}")
            raise


async def preload_inference_pipeline():
    """
    Preload the inference pipeline in the background at startup.

    Call this at startup to avoid 10-15 minute delay on first generation request.
    The pipeline will be ready when the first /generate request comes in.
    """
    global _pipeline_loading

    if _pipeline_loading:
        logger.info("Pipeline preload already in progress")
        return

    _pipeline_loading = True
    logger.info("🚀 Starting background pipeline preload...")

    try:
        await get_inference_pipeline()
        logger.info("✅ Pipeline preloaded and ready for generation requests")
    except Exception as e:
        logger.error(f"Pipeline preload failed: {e}")
    finally:
        _pipeline_loading = False


def is_pipeline_ready() -> bool:
    """Check if the inference pipeline is loaded and ready."""
    return _inference_pipeline is not None


def check_flux2_cached() -> bool:
    """Check if FLUX.2-dev model is already fully cached."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # Check for a key file that indicates the model is complete
        # FLUX.2-dev has flux1-dev.safetensors as the main weights file
        # Don't pass cache_dir - let HuggingFace use HF_HOME env var
        result = try_to_load_from_cache(
            "black-forest-labs/FLUX.2-dev",
            "flux1-dev.safetensors",
        )
        return result is not None
    except Exception:
        return False
