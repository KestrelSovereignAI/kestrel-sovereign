#!/usr/bin/env python3
"""
SimpleTuner API Wrapper for FLUX.2-dev Training

FLUX.2-specific implementation of the SimpleTuner API.
Handles FLUX.2-dev model configuration, quantization, and inference.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Form, HTTPException
from google.cloud import storage as gcs
from io import BytesIO
import requests
import sys

from base_simpletuner_api import BaseSimpleTunerAPI, setup_paths, _runtime_paths, WORKSPACE_PATH
from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_DEFAULT,
    HTTP_TIMEOUT_DOWNLOAD,
    TRAINING_TIMEOUT,
)

logger = logging.getLogger(__name__)

class FLUX2API(BaseSimpleTunerAPI):
    """FLUX.2-dev specific SimpleTuner API implementation."""

    def __init__(self):
        super().__init__(
            app_title="SimpleTuner FLUX.2 Training API",
            service_name="simpletuner-flux2"
        )

        # FLUX.2-specific paths
        self.quantized_cache_dir = f"{WORKSPACE_PATH}/quantized_flux"
        self.quantized_transformer_path = f"{self.quantized_cache_dir}/transformer_int8"
        self.quantized_text_encoder_path = f"{self.quantized_cache_dir}/text_encoder_int8"
        self.quantized_marker_path = f"{self.quantized_cache_dir}/.quantized_complete"

        # Register FLUX.2 specific routes
        self._register_flux2_routes()

    def get_model_family(self) -> str:
        return "flux2"

    def get_model_name(self) -> str:
        return "black-forest-labs/FLUX.2-dev"

    def get_pipeline_class(self):
        from diffusers import Flux2Pipeline
        return Flux2Pipeline

    def get_transformer_class(self):
        from diffusers import FluxTransformer2DModel
        return FluxTransformer2DModel

    def get_text_encoder_class(self):
        from transformers import AutoModel
        return AutoModel

    def get_quantized_cache_dir(self) -> str:
        return self.quantized_cache_dir

    def get_gcs_cache_prefix(self) -> str:
        return "model-cache/flux2-dev-int8-quanto"

    def get_training_timeout(self) -> int:
        return TRAINING_TIMEOUT

    def get_cached_model_filename(self) -> str:
        return "flux1-dev.safetensors"  # FLUX.2-dev uses same weights filename

    def create_model_specific_config(self, base_config: dict) -> dict:
        """Create FLUX.2-specific configuration modifications."""
        # FLUX.2 specific quantization settings
        base_config.update({
            # Quantize both transformer and text encoder to int8
            "base_model_precision": "int8-quanto",  # Quantize transformer
            "text_encoder_1_precision": "int8-quanto",  # Quantize Mistral-24B text encoder
            "quantize_via": "accelerator",  # GPU quantization to avoid OOM
        })
        return base_config

    def _get_gcs_quantized_paths(self) -> dict:
        """Get GCS paths for quantized model cache (Vertex AI mode only)."""
        bucket = os.environ.get("GCS_TRAINING_BUCKET", "kestrel-training")
        prefix = self.get_gcs_cache_prefix()
        return {
            "bucket": bucket,
            "transformer": f"gs://{bucket}/{prefix}/transformer_int8.tar.gz",
            "text_encoder": f"gs://{bucket}/{prefix}/text_encoder_int8.tar.gz",
            "marker": f"gs://{bucket}/{prefix}/.quantized_complete",
        }

    async def _gcs_cache_exists(self) -> bool:
        """Check if quantized model cache exists in GCS (Vertex AI mode)."""
        paths = self._get_gcs_quantized_paths()
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

    async def _download_gcs_quantized_cache(self) -> bool:
        """Download pre-quantized model from GCS to local disk (Vertex AI mode)."""
        paths = self._get_gcs_quantized_paths()

        # Ensure local cache dir exists
        os.makedirs(self.quantized_cache_dir, exist_ok=True)

        try:
            # Download transformer tarball
            logger.info(f"⬇️ Downloading quantized transformer from GCS...")
            transformer_tar = f"{self.quantized_cache_dir}/transformer_int8.tar.gz"
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
                tar.extractall(self.quantized_cache_dir)
            os.remove(transformer_tar)

            # Download text encoder tarball
            logger.info(f"⬇️ Downloading quantized text encoder from GCS...")
            text_encoder_tar = f"{self.quantized_cache_dir}/text_encoder_int8.tar.gz"
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
                tar.extractall(self.quantized_cache_dir)
            os.remove(text_encoder_tar)

            # Download marker file
            subprocess.run(
                ["gsutil", "-q", "cp", paths["marker"], self.quantized_marker_path],
                capture_output=True,
                timeout=HTTP_TIMEOUT_DEFAULT,
            )

            logger.info(f"✅ GCS quantized cache downloaded to {self.quantized_cache_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to download GCS quantized cache: {e}")
            # Clean up partial download
            if os.path.exists(self.quantized_cache_dir):
                shutil.rmtree(self.quantized_cache_dir, ignore_errors=True)
            return False

    async def _upload_gcs_quantized_cache(self) -> bool:
        """Upload quantized model to GCS for future Vertex AI jobs."""
        paths = self._get_gcs_quantized_paths()

        if not self.is_quantized_model_cached():
            logger.warning("No local quantized cache to upload")
            return False

        try:
            # Create transformer tarball
            logger.info("📦 Creating transformer tarball...")
            transformer_tar = f"{self.quantized_cache_dir}/transformer_int8.tar.gz"
            with tarfile.open(transformer_tar, "w:gz") as tar:
                tar.add(self.quantized_transformer_path, arcname="transformer_int8")

            # Upload transformer
            logger.info(f"⬆️ Uploading quantized transformer to GCS...")
            result = subprocess.run(
                ["gsutil", "-q", "cp", transformer_tar, paths["transformer"]],
                capture_output=True,
                text=True,
                timeout=TRAINING_TIMEOUT,
            )
            os.remove(transformer_tar)
            if result.returncode != 0:
                logger.error(f"Failed to upload transformer: {result.stderr}")
                return False

            # Create text encoder tarball
            logger.info("📦 Creating text encoder tarball...")
            text_encoder_tar = f"{self.quantized_cache_dir}/text_encoder_int8.tar.gz"
            with tarfile.open(text_encoder_tar, "w:gz") as tar:
                tar.add(self.quantized_text_encoder_path, arcname="text_encoder_int8")

            # Upload text encoder
            logger.info(f"⬆️ Uploading quantized text encoder to GCS...")
            result = subprocess.run(
                ["gsutil", "-q", "cp", text_encoder_tar, paths["text_encoder"]],
                capture_output=True,
                text=True,
                timeout=TRAINING_TIMEOUT,
            )
            os.remove(text_encoder_tar)
            if result.returncode != 0:
                logger.error(f"Failed to upload text encoder: {result.stderr}")
                return False

            # Upload marker file
            result = subprocess.run(
                ["gsutil", "-q", "cp", self.quantized_marker_path, paths["marker"]],
                capture_output=True,
                timeout=HTTP_TIMEOUT_DEFAULT,
            )

            logger.info(f"✅ Quantized cache uploaded to GCS: gs://{paths['bucket']}/{self.get_gcs_cache_prefix()}/")
            return True

        except Exception as e:
            logger.error(f"Failed to upload GCS quantized cache: {e}")
            return False

    def is_quantized_model_cached(self) -> bool:
        """Check if quantized model components are cached on disk."""
        return (
            os.path.isdir(self.quantized_transformer_path) and
            os.path.isdir(self.quantized_text_encoder_path) and
            os.path.exists(self.quantized_marker_path)
        )

    def _load_inference_pipeline_impl(self, save_quantized: bool = True):
        """Internal synchronous implementation of pipeline loading."""
        import torch
        from optimum.quanto import freeze, qint8, quantize

        pipeline_class = self.get_pipeline_class()
        transformer_class = self.get_transformer_class()
        text_encoder_class = self.get_text_encoder_class()

        # Ensure directories exist
        tmp_path = _runtime_paths["tmp_path"]
        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(self.quantized_cache_dir, exist_ok=True)

        # Check if we have cached quantized model
        if self.is_quantized_model_cached():
            logger.info("🚀 Loading CACHED quantized FLUX.2-dev from disk...")
            logger.info(f"   Cache location: {self.quantized_cache_dir}")

            # Load base pipeline first (for VAE, scheduler, etc.)
            self._inference_pipeline = pipeline_class.from_pretrained(
                self.get_model_name(),
                torch_dtype=torch.bfloat16,
            )

            # Load pre-quantized transformer
            logger.info("Loading cached quantized transformer...")
            self._inference_pipeline.transformer = transformer_class.from_pretrained(
                self.quantized_transformer_path,
                torch_dtype=torch.bfloat16,
            )

            # Load pre-quantized text encoder
            logger.info("Loading cached quantized text encoder...")
            self._inference_pipeline.text_encoder = text_encoder_class.from_pretrained(
                self.quantized_text_encoder_path,
                torch_dtype=torch.bfloat16,
            )

            # Move to GPU
            self._inference_pipeline.to("cuda")

            logger.info("✅ FLUX.2-dev loaded from CACHE (skip quantization)")
            return self._inference_pipeline

        # No cache - need to quantize from scratch
        logger.info("Loading FLUX.2-dev pipeline for inference (int8-quanto)...")
        logger.info("⚠️ First-time quantization takes 10-15 minutes.")
        logger.info("   Model will be cached to disk for faster future loads.")

        # Load FLUX.2-dev in BF16 first
        self._inference_pipeline = pipeline_class.from_pretrained(
            self.get_model_name(),
            torch_dtype=torch.bfloat16,
        )

        # Quantize transformer and text encoder to int8 (same as training)
        logger.info("Quantizing transformer to int8...")
        quantize(self._inference_pipeline.transformer, weights=qint8)
        freeze(self._inference_pipeline.transformer)

        logger.info("Quantizing text encoder to int8...")
        quantize(self._inference_pipeline.text_encoder, weights=qint8)
        freeze(self._inference_pipeline.text_encoder)

        # Save quantized model to disk for future loads
        if save_quantized:
            logger.info(f"💾 Saving quantized model to {self.quantized_cache_dir}...")
            try:
                # Save transformer
                logger.info("Saving quantized transformer...")
                self._inference_pipeline.transformer.save_pretrained(self.quantized_transformer_path)

                # Save text encoder
                logger.info("Saving quantized text encoder...")
                self._inference_pipeline.text_encoder.save_pretrained(self.quantized_text_encoder_path)

                # Write marker file to indicate complete save
                with open(self.quantized_marker_path, "w") as f:
                    f.write(f"quantized_at={datetime.utcnow().isoformat()}\n")
                    f.write(f"model={self.get_model_name()}\n")
                    f.write(f"quantization=int8-quanto\n")

                # Get cache size
                cache_size_gb = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, dn, filenames in os.walk(self.quantized_cache_dir)
                    for f in filenames
                ) / (1024**3)

                logger.info(f"✅ Quantized model saved to disk ({cache_size_gb:.1f} GB)")
                logger.info(f"   Future loads will skip quantization (~10 min saved)")

            except Exception as e:
                logger.warning(f"Failed to save quantized model to disk: {e}")
                logger.warning("Will quantize again on next load")
                # Clean up partial save
                if os.path.exists(self.quantized_marker_path):
                    os.remove(self.quantized_marker_path)

        # Move to GPU - no CPU offload needed with quantization
        self._inference_pipeline.to("cuda")

        logger.info("✅ FLUX.2-dev pipeline loaded (int8-quanto, ~50GB VRAM)")
        return self._inference_pipeline

    def get_inference_pipeline_sync(self, save_quantized: bool = True):
        """Synchronous version for background tasks."""
        with self._threading_lock:
            if self._inference_pipeline is not None:
                return self._inference_pipeline
            return self._load_inference_pipeline_impl(save_quantized)

    async def get_inference_pipeline(self, save_quantized: bool = True):
        """Lazy-load FLUX.2-dev pipeline for inference with int8-quanto quantization."""
        # Use threading lock for async context too
        with self._threading_lock:
            if self._inference_pipeline is not None:
                return self._inference_pipeline

            try:
                import torch
                from optimum.quanto import freeze, qint8, quantize

                pipeline_class = self.get_pipeline_class()
                transformer_class = self.get_transformer_class()
                text_encoder_class = self.get_text_encoder_class()

                # Ensure directories exist
                tmp_path = _runtime_paths["tmp_path"]
                os.makedirs(tmp_path, exist_ok=True)
                os.makedirs(self.quantized_cache_dir, exist_ok=True)

                # Vertex AI mode: Check GCS cache if local cache doesn't exist
                is_vertex_mode = os.environ.get("VERTEX_AI_MODE", "").lower() == "true"
                if is_vertex_mode and not self.is_quantized_model_cached():
                    logger.info("🔍 Vertex AI mode: Checking GCS for quantized cache...")
                    # Need to run async check synchronously in this context
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, use run_in_executor
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            gcs_exists = pool.submit(
                                lambda: asyncio.run(self._gcs_cache_exists())
                            ).result()
                    else:
                        gcs_exists = asyncio.run(self._gcs_cache_exists())

                    if gcs_exists:
                        logger.info("📥 Found GCS cache, downloading...")
                        if loop.is_running():
                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                downloaded = pool.submit(
                                    lambda: asyncio.run(self._download_gcs_quantized_cache())
                                ).result()
                        else:
                            downloaded = asyncio.run(self._download_gcs_quantized_cache())
                        if downloaded:
                            logger.info("✅ GCS cache downloaded successfully")
                        else:
                            logger.warning("⚠️ GCS cache download failed, will quantize from scratch")
                    else:
                        logger.info("📭 No GCS cache found, will quantize and upload")

                # Check if we have cached quantized model (local or just downloaded from GCS)
                if self.is_quantized_model_cached():
                    logger.info("🚀 Loading CACHED quantized FLUX.2-dev from disk...")
                    logger.info(f"   Cache location: {self.quantized_cache_dir}")

                    # Load base pipeline first (for VAE, scheduler, etc.)
                    self._inference_pipeline = pipeline_class.from_pretrained(
                        self.get_model_name(),
                        torch_dtype=torch.bfloat16,
                    )

                    # Load pre-quantized transformer
                    logger.info("Loading cached quantized transformer...")
                    self._inference_pipeline.transformer = transformer_class.from_pretrained(
                        self.quantized_transformer_path,
                        torch_dtype=torch.bfloat16,
                    )

                    # Load pre-quantized text encoder
                    logger.info("Loading cached quantized text encoder...")
                    self._inference_pipeline.text_encoder = text_encoder_class.from_pretrained(
                        self.quantized_text_encoder_path,
                        torch_dtype=torch.bfloat16,
                    )

                    # Move to GPU
                    self._inference_pipeline.to("cuda")

                    logger.info("✅ FLUX.2-dev loaded from CACHE (skip quantization)")
                    return self._inference_pipeline

                # No cache - need to quantize from scratch
                logger.info("Loading FLUX.2-dev pipeline for inference (int8-quanto)...")
                logger.info("⚠️ First-time quantization takes 10-15 minutes.")
                logger.info("   Model will be cached to disk for faster future loads.")

                # Load FLUX.2-dev in BF16 first
                self._inference_pipeline = pipeline_class.from_pretrained(
                    self.get_model_name(),
                    torch_dtype=torch.bfloat16,
                )

                # Quantize transformer and text encoder to int8 (same as training)
                logger.info("Quantizing transformer to int8...")
                quantize(self._inference_pipeline.transformer, weights=qint8)
                freeze(self._inference_pipeline.transformer)

                logger.info("Quantizing text encoder to int8...")
                quantize(self._inference_pipeline.text_encoder, weights=qint8)
                freeze(self._inference_pipeline.text_encoder)

                # Save quantized model to disk for future loads
                if save_quantized:
                    logger.info(f"💾 Saving quantized model to {self.quantized_cache_dir}...")
                    try:
                        # Save transformer
                        logger.info("Saving quantized transformer...")
                        self._inference_pipeline.transformer.save_pretrained(self.quantized_transformer_path)

                        # Save text encoder
                        logger.info("Saving quantized text encoder...")
                        self._inference_pipeline.text_encoder.save_pretrained(self.quantized_text_encoder_path)

                        # Write marker file to indicate complete save
                        with open(self.quantized_marker_path, "w") as f:
                            f.write(f"quantized_at={datetime.utcnow().isoformat()}\n")
                            f.write(f"model={self.get_model_name()}\n")
                            f.write(f"quantization=int8-quanto\n")

                        # Get cache size
                        cache_size_gb = sum(
                            os.path.getsize(os.path.join(dp, f))
                            for dp, dn, filenames in os.walk(self.quantized_cache_dir)
                            for f in filenames
                        ) / (1024**3)

                        logger.info(f"✅ Quantized model saved to disk ({cache_size_gb:.1f} GB)")
                        logger.info(f"   Future loads will skip quantization (~10 min saved)")

                        # Vertex AI mode: Upload to GCS for future jobs
                        if is_vertex_mode:
                            logger.info("☁️ Vertex AI mode: Uploading quantized cache to GCS...")
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    import concurrent.futures
                                    with concurrent.futures.ThreadPoolExecutor() as pool:
                                        uploaded = pool.submit(
                                            lambda: asyncio.run(self._upload_gcs_quantized_cache())
                                        ).result()
                                else:
                                    uploaded = asyncio.run(self._upload_gcs_quantized_cache())
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
                        if os.path.exists(self.quantized_marker_path):
                            os.remove(self.quantized_marker_path)

                # Move to GPU - no CPU offload needed with quantization
                self._inference_pipeline.to("cuda")

                logger.info("✅ FLUX.2-dev pipeline loaded (int8-quanto, ~50GB VRAM)")
                return self._inference_pipeline

            except Exception as e:
                logger.error(f"Failed to load inference pipeline: {e}")
                raise

    def _check_flux2_cached(self) -> bool:
        """Check if FLUX.2-dev model is already fully cached."""
        try:
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(
                self.get_model_name(),
                self.get_cached_model_filename(),
            )
            return result is not None
        except Exception:
            return False

    def _register_flux2_routes(self):
        """Register FLUX.2 specific routes."""

        @self.app.post("/generate")
        async def generate_image(
            prompt: str = Form(...),
            lora_path: str = Form(...),
            trigger_word: str = Form("TOK"),
            num_outputs: int = Form(1),
            width: int = Form(1024),
            height: int = Form(1024),
            num_inference_steps: int = Form(20),
            guidance_scale: float = Form(3.5),
        ):
            """Generate images using FLUX.2-dev with a trained LoRA."""
            import base64

            # Validate LoRA path
            lora_file = Path(lora_path)
            if not lora_file.exists():
                # Try runtime lora path
                lora_file = Path(f"{_runtime_paths['lora_path']}/{lora_path}/pytorch_lora_weights.safetensors")
                if not lora_file.exists():
                    raise HTTPException(404, f"LoRA not found: {lora_path}")

            try:
                # Get pipeline
                pipe = await self.get_inference_pipeline()

                # Load LoRA weights
                logger.info(f"Loading LoRA: {lora_file}")
                pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)

                # Ensure trigger word is in prompt
                if trigger_word and trigger_word not in prompt:
                    prompt = f"a photo of {trigger_word}, {prompt}"

                logger.info(f"Generating {num_outputs} images: {prompt[:50]}...")

                # Generate
                images = []
                for i in range(min(num_outputs, 4)):
                    result = pipe(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    )
                    image = result.images[0]

                    # Convert to base64
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    b64 = base64.b64encode(buffer.getvalue()).decode()
                    images.append(f"data:image/png;base64,{b64}")

                # Unload LoRA to free memory for next request
                pipe.unload_lora_weights()

                logger.info(f"Generated {len(images)} images")

                return {
                    "success": True,
                    "images": images,
                    "prompt": prompt,
                    "lora_path": str(lora_file),
                }

            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(500, f"Generation failed: {str(e)}")

        @self.app.post("/generate/async")
        async def generate_image_async(
            background_tasks: BackgroundTasks,
            prompt: str = Form(...),
            lora_path: str = Form(...),
            trigger_word: str = Form("TOK"),
            num_outputs: int = Form(1),
            width: int = Form(1024),
            height: int = Form(1024),
            num_inference_steps: int = Form(28),
            guidance_scale: float = Form(4.0),
            lora_ipfs_cid: Optional[str] = Form(None),
            ipfs_gateway: str = Form("https://gateway.lighthouse.storage/ipfs"),
        ):
            """Start async image generation with FLUX.2-dev."""
            import uuid

            job_id = str(uuid.uuid4())

            self._generation_jobs[job_id] = {
                "status": "pending",
                "prompt": prompt,
                "lora_path": lora_path,
                "lora_ipfs_cid": lora_ipfs_cid,
                "images": [],
                "error": None,
            }

            def run_generation():
                import base64
                from io import BytesIO

                try:
                    self._generation_jobs[job_id]["status"] = "loading_model"

                    # Resolve LoRA path - IPFS takes priority
                    if lora_ipfs_cid:
                        # Download from IPFS gateway
                        self._generation_jobs[job_id]["status"] = "downloading_lora"
                        local_lora_dir = f"{_runtime_paths['tmp_path']}/ipfs_lora_{job_id}"
                        os.makedirs(local_lora_dir, exist_ok=True)
                        local_lora_file = f"{local_lora_dir}/pytorch_lora_weights.safetensors"

                        ipfs_url = f"{ipfs_gateway}/{lora_ipfs_cid}"
                        logger.info(f"Downloading LoRA from IPFS: {ipfs_url}")

                        response = requests.get(ipfs_url, timeout=HTTP_TIMEOUT_DOWNLOAD, stream=True)
                        response.raise_for_status()

                        with open(local_lora_file, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        file_size = os.path.getsize(local_lora_file)
                        logger.info(f"Downloaded LoRA from IPFS ({file_size / 1024 / 1024:.1f} MB)")
                        lora_file = Path(local_lora_file)
                    else:
                        # Use local path
                        lora_file = Path(lora_path)
                        if not lora_file.exists():
                            lora_file = Path(f"{_runtime_paths['lora_path']}/{lora_path}/pytorch_lora_weights.safetensors")
                            if not lora_file.exists():
                                self._generation_jobs[job_id]["status"] = "failed"
                                self._generation_jobs[job_id]["error"] = f"LoRA not found: {lora_path}"
                                return

                    # Get pipeline using sync version
                    pipe = self.get_inference_pipeline_sync()

                    self._generation_jobs[job_id]["status"] = "loading_lora"
                    pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)

                    # Ensure trigger word in prompt
                    gen_prompt = prompt
                    if trigger_word and trigger_word not in prompt:
                        gen_prompt = f"a photo of {trigger_word}, {prompt}"

                    self._generation_jobs[job_id]["status"] = "generating"

                    images = []
                    for i in range(min(num_outputs, 4)):
                        result = pipe(
                            prompt=gen_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                        )
                        image = result.images[0]

                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        b64 = base64.b64encode(buffer.getvalue()).decode()
                        images.append(f"data:image/png;base64,{b64}")

                    pipe.unload_lora_weights()

                    self._generation_jobs[job_id]["status"] = "completed"
                    self._generation_jobs[job_id]["images"] = images
                    logger.info(f"Async generation {job_id} completed: {len(images)} images")

                except Exception as e:
                    logger.error(f"Async generation {job_id} failed: {e}")
                    self._generation_jobs[job_id]["status"] = "failed"
                    self._generation_jobs[job_id]["error"] = str(e)

            background_tasks.add_task(run_generation)

            return {
                "job_id": job_id,
                "status": "pending",
                "message": "Generation started. Poll /generate/status/{job_id} for results.",
            }

        @self.app.get("/generate/status/{job_id}")
        async def get_generation_status(job_id: str):
            """Get status of async generation job."""
            if job_id not in self._generation_jobs:
                raise HTTPException(404, "Job not found")

            job = self._generation_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job["status"],
                "images": job["images"] if job["status"] == "completed" else [],
                "error": job["error"],
            }

        @self.app.post("/preload")
        async def preload_model(background_tasks: BackgroundTasks, quantize: bool = True):
            """Pre-download and optionally pre-quantize FLUX.2-dev model."""
            if self._preload_status["status"] == "running":
                return {"status": "already_running", "progress": self._preload_status["progress"]}

            # Check if quantized cache exists (best case)
            if quantize and self.is_quantized_model_cached():
                self._preload_status = {"status": "complete", "progress": "Quantized model already cached!", "error": None}
                logger.info("Quantized FLUX.2-dev already cached, skipping")
                return {
                    "status": "already_cached",
                    "message": "Quantized FLUX.2-dev model is already cached on disk",
                    "cache_path": self.quantized_cache_dir,
                }

            # Check if base model is cached (partial case)
            if self._check_flux2_cached() and not quantize:
                self._preload_status = {"status": "complete", "progress": "Model already cached!", "error": None}
                logger.info("FLUX.2-dev already cached, skipping download")
                return {"status": "already_cached", "message": "FLUX.2-dev model is already cached"}

            def download_and_quantize():
                try:
                    # Step 1: Download base model if needed
                    if not self._check_flux2_cached():
                        self._preload_status = {"status": "running", "progress": "Downloading FLUX.2-dev (~60GB)...", "error": None}
                        from huggingface_hub import snapshot_download
                        logger.info("Starting FLUX.2-dev download...")
                        snapshot_download(
                            self.get_model_name(),
                            local_dir_use_symlinks=True
                        )
                        logger.info("FLUX.2-dev download complete!")

                    # Step 2: Quantize and cache if requested
                    if quantize and not self.is_quantized_model_cached():
                        self._preload_status = {"status": "running", "progress": "Quantizing model (10-15 min)...", "error": None}
                        logger.info("Starting quantization...")

                        # Use asyncio to call the async function
                        import asyncio
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(self.get_inference_pipeline(save_quantized=True))
                        loop.close()

                        self._preload_status = {"status": "complete", "progress": "Quantized model cached!", "error": None}
                        logger.info("Quantization and caching complete!")
                    else:
                        self._preload_status = {"status": "complete", "progress": "Download complete!", "error": None}

                except Exception as e:
                    self._preload_status = {"status": "failed", "progress": "", "error": str(e)}
                    logger.error(f"Preload failed: {e}")
                    import traceback
                    traceback.print_exc()

            background_tasks.add_task(download_and_quantize)

            return {
                "status": "started",
                "message": f"Model {'download + quantization' if quantize else 'download'} started in background",
                "quantize": quantize,
            }

        @self.app.get("/preload/status")
        async def preload_status():
            """Check status of model preload and quantized model cache."""
            result = dict(self._preload_status)

            # Add cache status
            result["cached"] = self.is_quantized_model_cached()
            result["cache_path"] = self.quantized_cache_dir
            result["pipeline_loaded"] = self._inference_pipeline is not None

            # Add cache size if cached
            if result["cached"]:
                try:
                    cache_size_bytes = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, dn, filenames in os.walk(self.quantized_cache_dir)
                        for f in filenames
                    )
                    result["cache_size_gb"] = round(cache_size_bytes / (1024**3), 2)

                    # Read marker file for metadata
                    if os.path.exists(self.quantized_marker_path):
                        with open(self.quantized_marker_path, "r") as f:
                            result["cache_metadata"] = f.read().strip()
                except Exception as e:
                    result["cache_size_error"] = str(e)

            return result

    async def run_vertex_batch_generation(self, args):
        """Run generation in batch mode for Vertex AI Custom Jobs."""
        from google.cloud import storage as gcs
        from io import BytesIO
        import sys
        import requests

        logger.info("=" * 60)
        logger.info("Vertex AI Batch Generation Mode - FLUX.2")
        logger.info("=" * 60)

        try:
            local_lora_path = "/tmp/lora/pytorch_lora_weights.safetensors"
            os.makedirs("/tmp/lora", exist_ok=True)

            # Step 1: Download LoRA from IPFS or GCS
            if args.lora_ipfs:
                # Download from IPFS via gateway
                ipfs_url = f"{args.ipfs_gateway}/{args.lora_ipfs}"
                logger.info(f"Downloading LoRA from IPFS: {ipfs_url}")

                response = requests.get(ipfs_url, timeout=HTTP_TIMEOUT_DOWNLOAD, stream=True)
                response.raise_for_status()

                with open(local_lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = os.path.getsize(local_lora_path)
                logger.info(f"Downloaded LoRA from IPFS to {local_lora_path} ({file_size / 1024 / 1024:.1f} MB)")

            else:
                # Download from GCS
                logger.info("Downloading LoRA from GCS...")
                client = gcs.Client()

                # Parse GCS URI
                if not args.lora_gcs.startswith("gs://"):
                    raise ValueError(f"Invalid GCS URI: {args.lora_gcs}")
                gcs_parts = args.lora_gcs[5:].split("/", 1)
                bucket_name = gcs_parts[0]
                blob_path = gcs_parts[1]

                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.download_to_filename(local_lora_path)
                logger.info(f"Downloaded LoRA from GCS to {local_lora_path}")

            # Step 2: Load pipeline and generate
            logger.info("Loading inference pipeline...")
            pipe = await self.get_inference_pipeline()

            logger.info(f"Loading LoRA from {local_lora_path}")
            pipe.load_lora_weights("/tmp/lora", weight_name="pytorch_lora_weights.safetensors")

            # Ensure trigger word in prompt
            prompt = args.prompt
            if args.trigger_word and args.trigger_word not in prompt:
                prompt = f"a photo of {args.trigger_word}, {prompt}"

            logger.info(f"Generating {args.num_outputs} images: {prompt[:80]}...")

            images = []
            for i in range(args.num_outputs):
                logger.info(f"Generating image {i+1}/{args.num_outputs}...")
                result = pipe(
                    prompt=prompt,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=28,
                    guidance_scale=4.0,
                )
                images.append(result.images[0])
                logger.info(f"Generated image {i+1}")

            pipe.unload_lora_weights()

            # Step 3: Upload images to GCS
            logger.info("Uploading images to GCS...")

            # Parse output GCS URI
            if not args.output_gcs.startswith("gs://"):
                raise ValueError(f"Invalid output GCS URI: {args.output_gcs}")
            output_parts = args.output_gcs[5:].split("/", 1)
            output_bucket_name = output_parts[0]
            output_prefix = output_parts[1] if len(output_parts) > 1 else ""

            # Create GCS client for output upload
            output_client = gcs.Client()
            output_bucket = output_client.bucket(output_bucket_name)

            image_urls = []
            for i, image in enumerate(images):
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)

                blob_name = f"{output_prefix}/image_{i}.png"
                output_blob = output_bucket.blob(blob_name)
                output_blob.upload_from_file(buffer, content_type="image/png")
                image_urls.append(f"gs://{output_bucket_name}/{blob_name}")
                logger.info(f"Uploaded: gs://{output_bucket_name}/{blob_name}")

            # Write metadata
            metadata = {
                "prompt": prompt,
                "trigger_word": args.trigger_word,
                "lora_source": args.lora_ipfs if args.lora_ipfs else args.lora_gcs,
                "num_outputs": len(images),
                "image_urls": image_urls,
                "width": args.width,
                "height": args.height,
                "completed_at": datetime.utcnow().isoformat(),
                "model": "FLUX.2-dev",
            }
            metadata_blob = output_bucket.blob(f"{output_prefix}/metadata.json")
            metadata_blob.upload_from_string(json.dumps(metadata, indent=2))

            logger.info("=" * 60)
            logger.info(f"FLUX.2 Generation completed! {len(images)} images")
            logger.info("=" * 60)
            sys.exit(0)

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    async def run_vertex_batch_training(self, args):
        """Run training in batch mode for Vertex AI Custom Jobs."""
        from google.cloud import storage as gcs
        import sys

        logger.info("=" * 60)
        logger.info("Vertex AI Batch Training Mode - FLUX.2")
        logger.info("=" * 60)

        job_id = args.companion_id

        try:
            # Step 1: Download avatar from GCS
            logger.info("Downloading training image from GCS...")

            # Parse GCS URI: gs://bucket/path/to/file
            if not args.avatar_gcs.startswith("gs://"):
                raise ValueError(f"Invalid GCS URI: {args.avatar_gcs}")

            gcs_parts = args.avatar_gcs[5:].split("/", 1)
            bucket_name = gcs_parts[0]
            blob_path = gcs_parts[1] if len(gcs_parts) > 1 else ""

            client = gcs.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            # Create dataset directory - use configured paths
            dataset_path = f"{_runtime_paths['datasets_path']}/{job_id}"
            os.makedirs(dataset_path, exist_ok=True)

            # Download image
            local_image_path = f"{dataset_path}/image_001.png"
            blob.download_to_filename(local_image_path)
            logger.info(f"Downloaded training image to {local_image_path}")

            # Step 2: Create training config
            output_path = f"{_runtime_paths['output_path']}/{job_id}"
            cache_path = f"{output_path}/cache"

            config = self.create_simpletuner_config(
                job_id=job_id,
                trigger_word=args.trigger_word,
                dataset_path=dataset_path,
                output_path=output_path,
                cache_path=cache_path,
                steps=args.steps,
                lora_rank=args.lora_rank,
            )

            # Create job record for monitoring
            self._training_jobs[job_id] = {
                "job_id": job_id,
                "companion_id": args.companion_id,
                "trigger_word": args.trigger_word,
                "status": "preparing",
                "progress": 0.0,
                "current_step": 0,
                "total_steps": args.steps,
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": None,
                "lora_path": None,
                "error": None,
            }

            # Step 3: Run training (synchronously in batch mode)
            logger.info("Starting SimpleTuner training...")
            await self.run_training(job_id, config, dataset_path)

            job = self._training_jobs[job_id]

            if job["status"] == "failed":
                raise RuntimeError(f"Training failed: {job.get('error', 'Unknown error')}")

            # Step 4: Upload LoRA to GCS
            logger.info("Uploading trained LoRA to GCS...")

            lora_path = job.get("lora_path")
            if not lora_path or not os.path.exists(lora_path):
                raise RuntimeError(f"LoRA file not found: {lora_path}")

            # Parse output GCS URI
            if not args.output_gcs.startswith("gs://"):
                raise ValueError(f"Invalid output GCS URI: {args.output_gcs}")

            output_parts = args.output_gcs[5:].split("/", 1)
            output_bucket_name = output_parts[0]
            output_prefix = output_parts[1] if len(output_parts) > 1 else ""

            output_bucket = client.bucket(output_bucket_name)

            # Upload LoRA file
            lora_blob_path = f"{output_prefix}/{job_id}/pytorch_lora_weights.safetensors"
            lora_blob = output_bucket.blob(lora_blob_path)
            lora_blob.upload_from_filename(lora_path)
            logger.info(f"Uploaded LoRA to gs://{output_bucket_name}/{lora_blob_path}")

            # Upload metadata
            metadata = {
                "job_id": job_id,
                "companion_id": args.companion_id,
                "trigger_word": args.trigger_word,
                "steps": args.steps,
                "lora_rank": args.lora_rank,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "lora_gcs_path": f"gs://{output_bucket_name}/{lora_blob_path}",
                "model": "FLUX.2-dev",
            }
            metadata_blob_path = f"{output_prefix}/{job_id}/metadata.json"
            metadata_blob = output_bucket.blob(metadata_blob_path)
            metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
            logger.info(f"Uploaded metadata to gs://{output_bucket_name}/{metadata_blob_path}")

            logger.info("=" * 60)
            logger.info("FLUX.2 Training completed successfully!")
            logger.info(f"LoRA: gs://{output_bucket_name}/{lora_blob_path}")
            logger.info("=" * 60)

            sys.exit(0)

        except Exception as e:
            logger.error(f"Batch training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point for the FLUX.2 API."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SimpleTuner FLUX.2 Training - API or Batch Mode"
    )
    # Training mode (Vertex AI batch)
    parser.add_argument(
        "--vertex-mode",
        action="store_true",
        help="Run in Vertex AI training batch mode (download from GCS, train, upload to GCS, exit)"
    )
    parser.add_argument(
        "--avatar-gcs",
        type=str,
        help="GCS URI for training image (gs://bucket/path/to/avatar.png)"
    )
    parser.add_argument(
        "--companion-id",
        type=str,
        help="Companion UUID being trained"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Training steps (default: 1000)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )

    # Generation mode (Vertex AI batch)
    parser.add_argument(
        "--generate-mode",
        action="store_true",
        help="Run in Vertex AI generation batch mode (download LoRA, generate, upload to GCS, exit)"
    )
    parser.add_argument(
        "--lora-gcs",
        type=str,
        help="GCS URI for LoRA weights (gs://bucket/path/to/pytorch_lora_weights.safetensors)"
    )
    parser.add_argument(
        "--lora-ipfs",
        type=str,
        help="IPFS CID for LoRA weights (e.g., QmXxx...)"
    )
    parser.add_argument(
        "--ipfs-gateway",
        type=str,
        default="https://gateway.lighthouse.storage/ipfs",
        help="IPFS gateway URL (default: https://gateway.lighthouse.storage/ipfs)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Generation prompt"
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)"
    )

    # Shared arguments
    parser.add_argument(
        "--output-gcs",
        type=str,
        help="GCS URI prefix for output (gs://bucket/path/for/output/)"
    )
    parser.add_argument(
        "--trigger-word",
        type=str,
        help="LoRA trigger word (default: TOK)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000, only for API mode)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SimpleTuner FLUX.2 Training")
    logger.info("=" * 60)

    # Log GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.warning("CUDA not available!")
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")

    # Create API instance
    api = FLUX2API()

    if args.generate_mode:
        # Vertex AI generation batch mode
        if not args.lora_gcs and not args.lora_ipfs:
            parser.error("--lora-gcs or --lora-ipfs is required for --generate-mode")
        if args.lora_gcs and args.lora_ipfs:
            parser.error("Specify either --lora-gcs or --lora-ipfs, not both")
        if not args.output_gcs:
            parser.error("--output-gcs is required for --generate-mode")
        if not args.prompt:
            parser.error("--prompt is required for --generate-mode")

        # Set default trigger word if not provided
        if not args.trigger_word:
            args.trigger_word = "TOK"

        # Configure paths for Vertex AI environment
        setup_paths(is_vertex_mode=True)

        logger.info("Running in VERTEX AI GENERATION MODE")
        asyncio.run(api.run_vertex_batch_generation(args))

    elif args.vertex_mode:
        # Vertex AI training batch mode
        if not args.avatar_gcs:
            parser.error("--avatar-gcs is required for --vertex-mode")
        if not args.output_gcs:
            parser.error("--output-gcs is required for --vertex-mode")
        if not args.companion_id:
            parser.error("--companion-id is required for --vertex-mode")

        # Set default trigger word if not provided
        if not args.trigger_word:
            clean_id = args.companion_id.replace("-", "")[:8]
            args.trigger_word = f"TOK{clean_id}"

        # Configure paths for Vertex AI environment
        setup_paths(is_vertex_mode=True)

        logger.info("Running in VERTEX AI TRAINING MODE")
        asyncio.run(api.run_vertex_batch_training(args))

    else:
        # API server mode (RunPod)
        # Configure paths for RunPod environment - WILL FAIL if /workspace not mounted
        setup_paths(is_vertex_mode=False)

        logger.info("Running in API SERVER MODE")
        api.run_server(host="0.0.0.0", port=args.port)