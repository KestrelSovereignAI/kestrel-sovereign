#!/usr/bin/env python3
"""
Base SimpleTuner API Wrapper for Kestrel LoRA Training

Provides shared functionality for both FLUX.1 and FLUX.2 training APIs.
Contains all common endpoints, path management, training logic, inference
pipeline management, GCS caching, and Vertex AI batch operations.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import aiofiles
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_DEFAULT,
    HTTP_TIMEOUT_DOWNLOAD,
    HTTP_TIMEOUT_MODEL_PULL,
)
from kestrel_sovereign.kestrel_config.defaults import get_lighthouse_gateway_url

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RunPod network volume mount point - REQUIRED, no alternatives
WORKSPACE_PATH = "/workspace"

# Global path configuration (set by setup_paths at startup)
_runtime_paths = {
    "base_path": WORKSPACE_PATH,
    "hf_path": f"{WORKSPACE_PATH}/huggingface",
    "tmp_path": f"{WORKSPACE_PATH}/tmp",
    "lora_path": f"{WORKSPACE_PATH}/trained_loras",
    "output_path": f"{WORKSPACE_PATH}/output",
    "datasets_path": f"{WORKSPACE_PATH}/datasets",
}


def setup_paths(is_vertex_mode: bool = False):
    """
    Configure paths based on runtime environment.

    Vertex AI: Uses standard container paths (/app, /tmp)
    RunPod: REQUIRES network volume at /workspace - FAILS if not mounted
    """
    global _runtime_paths

    if is_vertex_mode:
        # Vertex AI: use standard Linux paths within the container
        base_path = "/app"
        tmp_path = "/tmp"
        logger.info("Configuring paths for VERTEX AI mode")

        _runtime_paths = {
            "base_path": base_path,
            "hf_path": f"{base_path}/huggingface",
            "tmp_path": tmp_path,
            "lora_path": f"{base_path}/trained_loras",
            "output_path": f"{base_path}/output",
            "datasets_path": f"{base_path}/datasets",
        }
    else:
        # RunPod: REQUIRE network volume at /workspace
        # NO FALLBACKS - fail fast if not properly configured
        if not os.path.isdir(WORKSPACE_PATH):
            raise RuntimeError(
                f"FATAL: {WORKSPACE_PATH} does not exist. "
                f"RunPod network volume MUST be mounted at {WORKSPACE_PATH}. "
                f"Check pod configuration."
            )

        if not os.path.ismount(WORKSPACE_PATH):
            raise RuntimeError(
                f"FATAL: {WORKSPACE_PATH} is not a mount point. "
                f"RunPod network volume MUST be mounted at {WORKSPACE_PATH}. "
                f"This is NOT a directory on the container disk. "
                f"Check pod configuration - volume_mount_path must be {WORKSPACE_PATH}."
            )

        logger.info(f"Network volume verified at {WORKSPACE_PATH}")

        # All paths under /workspace for persistence
        _runtime_paths = {
            "base_path": WORKSPACE_PATH,
            "hf_path": f"{WORKSPACE_PATH}/huggingface",
            "tmp_path": f"{WORKSPACE_PATH}/tmp",
            "lora_path": f"{WORKSPACE_PATH}/trained_loras",
            "output_path": f"{WORKSPACE_PATH}/output",
            "datasets_path": f"{WORKSPACE_PATH}/datasets",
        }

        logger.info(f"Configuring paths for RUNPOD mode with base: {WORKSPACE_PATH}")

    # Create all directories
    for key, path in _runtime_paths.items():
        if key != "base_path":  # Don't try to create the base mount point
            os.makedirs(path, exist_ok=True)

    # Set environment variables for HuggingFace/PyTorch
    os.environ["HF_HOME"] = _runtime_paths["hf_path"]
    os.environ["TRANSFORMERS_CACHE"] = _runtime_paths["hf_path"]
    os.environ["TORCH_HOME"] = _runtime_paths["hf_path"]
    os.environ["TMPDIR"] = _runtime_paths["tmp_path"]
    os.environ["TEMP"] = _runtime_paths["tmp_path"]
    os.environ["TMP"] = _runtime_paths["tmp_path"]

    logger.info(f"Paths configured: {_runtime_paths}")
    return _runtime_paths


class BaseSimpleTunerAPI:
    """
    Base class for SimpleTuner API implementations.

    Contains all shared functionality between FLUX.1 and FLUX.2 variants.
    Subclasses must implement model-specific configurations and pipeline loading.
    """

    def __init__(self, app_title: str, service_name: str):
        self.app = FastAPI(title=app_title)
        self.service_name = service_name

        # Training state
        self._training_jobs: dict = {}
        self._current_job: Optional[str] = None

        # Inference state
        self._inference_pipeline = None
        self._inference_lock = None
        self._pipeline_loading = False
        self._threading_lock = __import__('threading').Lock()

        # Preload status
        self._preload_status = {"status": "idle", "progress": "", "error": None}

        # Background generation jobs
        self._generation_jobs: dict[str, dict] = {}

        # Quantized model paths (set from WORKSPACE_PATH, subclass can override)
        self.quantized_cache_dir = f"{WORKSPACE_PATH}/quantized_flux"
        self.quantized_transformer_path = f"{self.quantized_cache_dir}/transformer_int8"
        self.quantized_text_encoder_path = f"{self.quantized_cache_dir}/text_encoder_int8"
        self.quantized_marker_path = f"{self.quantized_cache_dir}/.quantized_complete"

        # Register routes
        self._register_routes()
        self._register_inference_routes()

    # =========================================================================
    # Abstract methods that subclasses must implement
    # =========================================================================

    def get_model_family(self) -> str:
        """Return the model family string for SimpleTuner config."""
        raise NotImplementedError

    def get_model_name(self) -> str:
        """Return the HuggingFace model name."""
        raise NotImplementedError

    def get_display_name(self) -> str:
        """Return a human-readable model name for log messages."""
        raise NotImplementedError

    def get_pipeline_class(self):
        """Return the diffusers pipeline class."""
        raise NotImplementedError

    def get_transformer_class(self):
        """Return the transformer model class."""
        raise NotImplementedError

    def get_text_encoder_class(self):
        """Return the text encoder class."""
        raise NotImplementedError

    def get_quantized_cache_dir(self) -> str:
        """Return the quantized model cache directory."""
        raise NotImplementedError

    def get_gcs_cache_prefix(self) -> str:
        """Return the GCS cache prefix for this model."""
        raise NotImplementedError

    def get_training_timeout(self) -> int:
        """Return the training timeout in seconds."""
        raise NotImplementedError

    def create_model_specific_config(self, base_config: dict) -> dict:
        """Create model-specific configuration modifications."""
        raise NotImplementedError

    def get_cached_model_filename(self) -> str:
        """Return the filename to check if model is cached."""
        raise NotImplementedError

    # =========================================================================
    # Hook methods (subclasses can override for model-specific behavior)
    # =========================================================================

    def load_generation_loras(self, pipe, local_lora_path: str):
        """
        Load LoRA adapters for batch generation.

        Default: loads a single LoRA. FLUX.1 overrides this to add
        uncensored LoRA support with multi-adapter composition.
        """
        pipe.load_lora_weights(
            os.path.dirname(local_lora_path),
            weight_name=os.path.basename(local_lora_path),
        )

    def get_generation_metadata_extras(self) -> dict:
        """Return extra metadata fields for batch generation output."""
        return {}

    # =========================================================================
    # SimpleTuner config generation
    # =========================================================================

    def create_multidatabackend_config(
        self,
        trigger_word: str,
        dataset_path: str,
        cache_path: str,
        steps: int = 500,
    ) -> list:
        """
        Create SimpleTuner multidatabackend.json config.

        Repeats is set to match training steps so training completes at max_train_steps.
        For single-image training, repeats should be >= steps to ensure enough samples.
        """
        # Calculate repeats to match training steps
        # For 1 image, repeats should be >= steps to have enough training samples
        # Add 10% buffer to avoid running out of samples before max_train_steps
        repeats = int(steps * 1.1) + 10

        return [
            {
                "id": "main_dataset",
                "type": "local",
                "instance_data_dir": dataset_path,
                "caption_strategy": "instanceprompt",
                "instance_prompt": f"a photo of {trigger_word}",
                "resolution": 1024,
                "minimum_image_size": 512,
                "maximum_image_size": 2048,
                "target_downsample_size": 1024,
                "resolution_type": "pixel_area",
                "prepend_instance_prompt": False,
                "cache_dir_vae": f"{cache_path}/vae",
                "cache_dir_text": f"{cache_path}/text",
                "disabled": False,
                "skip_file_discovery": "",  # Empty string = don't skip (SimpleTuner expects string, not bool)
                "preserve_data_backend_cache": True,
                # Repeat image to have enough samples for max_train_steps
                "repeats": repeats,
            }
        ]

    def create_simpletuner_config(
        self,
        job_id: str,
        trigger_word: str,
        dataset_path: str,
        output_path: str,
        cache_path: str,
        steps: int = 1000,
        lora_rank: int = 16,
    ) -> dict:
        """
        Create base SimpleTuner config.
        Subclasses can modify via create_model_specific_config().
        """
        base_config = {
            # Model config - will be modified by subclass
            "model_family": self.get_model_family(),
            "model_flavour": "dev",
            "pretrained_model_name_or_path": self.get_model_name(),

            # Training type
            "model_type": "lora",
            "lora_rank": lora_rank,
            "lora_alpha": lora_rank,  # Typically same as rank

            # Training params - REQUIRED
            "optimizer": "adamw_bf16",
            "num_train_epochs": 0,  # Required when using max_train_steps
            "max_train_steps": steps,
            "learning_rate": 1e-4,
            "lr_scheduler": "constant",
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,  # Reduced from 4 for faster iteration

            # Data backend config path - REQUIRED
            "data_backend_config": f"{output_path}/config/multidatabackend.json",

            # Memory optimization for A100 80GB
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,

            # Guidance settings
            "flux_guidance_mode": "constant",
            "flux_guidance_value": 1.0,

            # Output
            "output_dir": output_path,

            # Validation - disabled during training for speed, final only
            "validation_prompt": f"a portrait photo of {trigger_word}, high quality",
            "validation_steps": 0,  # Disable mid-training validation for speed
            "validation_resolution": 1024,

            # Checkpointing - save at end of training
            # Set to steps value so checkpoint saves once when max_train_steps is reached
            "checkpoint_step_interval": steps,  # Save final checkpoint
            "checkpoints_total_limit": 1,  # Keep only final checkpoint

            # Logging - TensorBoard for local monitoring
            "logging_dir": f"{output_path}/logs",
            "report_to": "tensorboard",

            # Debug options - enable verbose logging
            "debug_dataset_loader": True,
            "print_filenames": True,

            # Cache paths
            "cache_dir_vae": f"{cache_path}/vae",
        }

        # Let subclass modify the config for model-specific settings
        return self.create_model_specific_config(base_config)

    # =========================================================================
    # Training
    # =========================================================================

    async def run_training(self, job_id: str, config: dict, dataset_path: str):
        """
        Run SimpleTuner training in background.
        """
        self._current_job = job_id

        job = self._training_jobs[job_id]
        job["status"] = "preparing"
        job["progress"] = 0.05

        try:
            output_path = config["output_dir"]
            cache_path = f"{output_path}/cache"

            # Create directories
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(cache_path, exist_ok=True)
            os.makedirs(f"{cache_path}/vae", exist_ok=True)
            os.makedirs(f"{cache_path}/text", exist_ok=True)

            # Write config files
            config_dir = f"{output_path}/config"
            os.makedirs(config_dir, exist_ok=True)

            # Main config as JSON (SimpleTuner accepts JSON or env)
            config_path = f"{config_dir}/config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            # Multidatabackend config - pass steps so repeats matches max_train_steps
            multidata_config = self.create_multidatabackend_config(
                trigger_word=job["trigger_word"],
                dataset_path=dataset_path,
                cache_path=cache_path,
                steps=config["max_train_steps"],
            )
            multidata_path = f"{config_dir}/multidatabackend.json"
            with open(multidata_path, "w") as f:
                json.dump(multidata_config, f, indent=2)

            logger.info(f"Config written to {config_path}")
            logger.info(f"Multidatabackend written to {multidata_path}")

            job["status"] = "training"
            job["progress"] = 0.1

            env = os.environ.copy()

            # SimpleTuner CLI: `simpletuner train --env <config_dir>`
            # This runs accelerate launch internally and captures output
            cmd = [
                "simpletuner", "train",
                "--env", config_dir,
            ]

            logger.info(f"Starting SimpleTuner: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )

            # Monitor progress by reading output
            steps_total = config["max_train_steps"]
            current_step = 0
            buffer = ""
            training_timeout = self.get_training_timeout()

            while True:
                try:
                    # Read chunks instead of lines to handle tqdm progress bars
                    chunk = await asyncio.wait_for(
                        process.stdout.read(4096),  # Read up to 4KB at a time
                        timeout=training_timeout
                    )
                    if not chunk:
                        # EOF - process finished
                        break

                    # Decode and add to buffer
                    buffer += chunk.decode(errors='replace')

                    # Process complete lines from buffer
                    while '\n' in buffer:
                        line_str, buffer = buffer.split('\n', 1)
                        line_str = line_str.strip()

                        # Skip empty lines and carriage returns (tqdm updates)
                        if not line_str or line_str == '\r':
                            continue

                        # Clean up tqdm carriage returns within line
                        if '\r' in line_str:
                            # Take the last segment after carriage return (final tqdm state)
                            line_str = line_str.split('\r')[-1].strip()
                            if not line_str:
                                continue

                        logger.info(f"[SimpleTuner] {line_str}")

                        # Parse step progress from output
                        # SimpleTuner outputs: "Step 100/1000 - loss: 0.xxx"
                        if "Step " in line_str and "/" in line_str:
                            try:
                                step_part = line_str.split("Step ")[1].split(" ")[0]
                                current_step = int(step_part.split("/")[0])
                                progress = 0.1 + (current_step / steps_total) * 0.8
                                job["progress"] = min(progress, 0.9)
                                job["current_step"] = current_step
                            except (IndexError, ValueError):
                                pass

                        # Also check for percentage-based progress (e.g., "50%|████")
                        if "%" in line_str and "|" in line_str:
                            try:
                                pct_str = line_str.split("%")[0].strip()
                                # Get the last number before %
                                pct = int(pct_str.split()[-1])
                                if pct > 0:
                                    progress = 0.1 + (pct / 100) * 0.8
                                    job["progress"] = min(progress, 0.9)
                                    job["current_step"] = int(pct * steps_total / 100)
                            except (IndexError, ValueError):
                                pass

                except asyncio.TimeoutError:
                    # Check if process is still running
                    if process.returncode is not None:
                        break
                    logger.warning("No output for timeout period, continuing to wait...")
                    continue
                except Exception as e:
                    logger.warning(f"Error reading output: {e}")
                    # Check if process is still alive before breaking
                    if process.returncode is not None:
                        break
                    continue

            await process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"SimpleTuner exited with code {process.returncode}")

            # Find the output LoRA file
            job["status"] = "finalizing"
            job["progress"] = 0.95

            # SimpleTuner outputs to output_dir/pytorch_lora_weights.safetensors
            lora_file = Path(output_path) / "pytorch_lora_weights.safetensors"
            if not lora_file.exists():
                # Check for checkpoint directories
                for checkpoint_dir in Path(output_path).glob("checkpoint-*"):
                    potential_lora = checkpoint_dir / "pytorch_lora_weights.safetensors"
                    if potential_lora.exists():
                        lora_file = potential_lora
                        break

            if not lora_file.exists():
                raise RuntimeError(f"LoRA file not found in {output_path}")

            job["lora_path"] = str(lora_file)
            job["status"] = "completed"
            job["progress"] = 1.0
            job["completed_at"] = datetime.utcnow().isoformat()

            logger.info(f"Training completed! LoRA at: {lora_file}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            job["status"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = datetime.utcnow().isoformat()

        finally:
            self._current_job = None

    # =========================================================================
    # GCS quantized model cache operations
    # =========================================================================

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
                logger.info(f"GCS quantized cache found: {marker_path}")
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
            logger.info("Downloading quantized transformer from GCS...")
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
            logger.info("Downloading quantized text encoder from GCS...")
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

            logger.info(f"GCS quantized cache downloaded to {self.quantized_cache_dir}")
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
            logger.info("Creating transformer tarball...")
            transformer_tar = f"{self.quantized_cache_dir}/transformer_int8.tar.gz"
            with tarfile.open(transformer_tar, "w:gz") as tar:
                tar.add(self.quantized_transformer_path, arcname="transformer_int8")

            # Upload transformer
            logger.info("Uploading quantized transformer to GCS...")
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
            logger.info("Creating text encoder tarball...")
            text_encoder_tar = f"{self.quantized_cache_dir}/text_encoder_int8.tar.gz"
            with tarfile.open(text_encoder_tar, "w:gz") as tar:
                tar.add(self.quantized_text_encoder_path, arcname="text_encoder_int8")

            # Upload text encoder
            logger.info("Uploading quantized text encoder to GCS...")
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
                ["gsutil", "-q", "cp", self.quantized_marker_path, paths["marker"]],
                capture_output=True,
                timeout=HTTP_TIMEOUT_DEFAULT,
            )

            logger.info(f"Quantized cache uploaded to GCS: gs://{paths['bucket']}/{self.get_gcs_cache_prefix()}/")
            return True

        except Exception as e:
            logger.error(f"Failed to upload GCS quantized cache: {e}")
            return False

    # =========================================================================
    # Quantized model cache checks
    # =========================================================================

    def is_quantized_model_cached(self) -> bool:
        """Check if quantized model components are cached on disk."""
        return (
            os.path.isdir(self.quantized_transformer_path) and
            os.path.isdir(self.quantized_text_encoder_path) and
            os.path.exists(self.quantized_marker_path)
        )

    def _check_model_cached(self) -> bool:
        """Check if the base model is already fully cached in HuggingFace cache."""
        try:
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(
                self.get_model_name(),
                self.get_cached_model_filename(),
            )
            return result is not None
        except Exception:
            return False

    # =========================================================================
    # Inference pipeline loading
    # =========================================================================

    def _load_inference_pipeline_impl(self, save_quantized: bool = True):
        """Internal synchronous implementation of pipeline loading."""
        import torch
        from optimum.quanto import freeze, qint8, quantize

        pipeline_class = self.get_pipeline_class()
        transformer_class = self.get_transformer_class()
        text_encoder_class = self.get_text_encoder_class()
        display_name = self.get_display_name()

        # Ensure directories exist
        tmp_path = _runtime_paths["tmp_path"]
        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(self.quantized_cache_dir, exist_ok=True)

        # Check if we have cached quantized model
        if self.is_quantized_model_cached():
            logger.info(f"Loading CACHED quantized {display_name} from disk...")
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

            logger.info(f"{display_name} loaded from CACHE (skip quantization)")
            return self._inference_pipeline

        # No cache - need to quantize from scratch
        logger.info(f"Loading {display_name} pipeline for inference (int8-quanto)...")
        logger.info("First-time quantization takes 10-15 minutes.")
        logger.info("   Model will be cached to disk for faster future loads.")

        # Load model in BF16 first
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
            logger.info(f"Saving quantized model to {self.quantized_cache_dir}...")
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

                logger.info(f"Quantized model saved to disk ({cache_size_gb:.1f} GB)")
                logger.info(f"   Future loads will skip quantization (~10 min saved)")

            except Exception as e:
                logger.warning(f"Failed to save quantized model to disk: {e}")
                logger.warning("Will quantize again on next load")
                # Clean up partial save
                if os.path.exists(self.quantized_marker_path):
                    os.remove(self.quantized_marker_path)

        # Move to GPU - no CPU offload needed with quantization
        self._inference_pipeline.to("cuda")

        logger.info(f"{display_name} pipeline loaded (int8-quanto, ~50GB VRAM)")
        return self._inference_pipeline

    def get_inference_pipeline_sync(self, save_quantized: bool = True):
        """Synchronous version for background tasks."""
        with self._threading_lock:
            if self._inference_pipeline is not None:
                return self._inference_pipeline
            return self._load_inference_pipeline_impl(save_quantized)

    async def get_inference_pipeline(self, save_quantized: bool = True):
        """Lazy-load pipeline for inference with int8-quanto quantization."""
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
                display_name = self.get_display_name()

                # Ensure directories exist
                tmp_path = _runtime_paths["tmp_path"]
                os.makedirs(tmp_path, exist_ok=True)
                os.makedirs(self.quantized_cache_dir, exist_ok=True)

                # Vertex AI mode: Check GCS cache if local cache doesn't exist
                is_vertex_mode = os.environ.get("VERTEX_AI_MODE", "").lower() == "true"
                if is_vertex_mode and not self.is_quantized_model_cached():
                    logger.info("Vertex AI mode: Checking GCS for quantized cache...")
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
                        logger.info("Found GCS cache, downloading...")
                        if loop.is_running():
                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                downloaded = pool.submit(
                                    lambda: asyncio.run(self._download_gcs_quantized_cache())
                                ).result()
                        else:
                            downloaded = asyncio.run(self._download_gcs_quantized_cache())
                        if downloaded:
                            logger.info("GCS cache downloaded successfully")
                        else:
                            logger.warning("GCS cache download failed, will quantize from scratch")
                    else:
                        logger.info("No GCS cache found, will quantize and upload")

                # Check if we have cached quantized model (local or just downloaded from GCS)
                if self.is_quantized_model_cached():
                    logger.info(f"Loading CACHED quantized {display_name} from disk...")
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

                    logger.info(f"{display_name} loaded from CACHE (skip quantization)")
                    return self._inference_pipeline

                # No cache - need to quantize from scratch
                logger.info(f"Loading {display_name} pipeline for inference (int8-quanto)...")
                logger.info("First-time quantization takes 10-15 minutes.")
                logger.info("   Model will be cached to disk for faster future loads.")

                # Load model in BF16 first
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
                    logger.info(f"Saving quantized model to {self.quantized_cache_dir}...")
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

                        logger.info(f"Quantized model saved to disk ({cache_size_gb:.1f} GB)")
                        logger.info(f"   Future loads will skip quantization (~10 min saved)")

                        # Vertex AI mode: Upload to GCS for future jobs
                        if is_vertex_mode:
                            logger.info("Vertex AI mode: Uploading quantized cache to GCS...")
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
                                    logger.info("GCS cache uploaded - future Vertex AI jobs will be faster!")
                                else:
                                    logger.warning("GCS upload failed - next job will re-quantize")
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

                logger.info(f"{display_name} pipeline loaded (int8-quanto, ~50GB VRAM)")
                return self._inference_pipeline

            except Exception as e:
                logger.error(f"Failed to load inference pipeline: {e}")
                raise

    # =========================================================================
    # Vertex AI batch operations
    # =========================================================================

    async def run_vertex_batch_generation(self, args):
        """Run generation in batch mode for Vertex AI Custom Jobs."""
        from google.cloud import storage as gcs
        from io import BytesIO
        import sys
        import requests

        display_name = self.get_display_name()

        logger.info("=" * 60)
        logger.info(f"Vertex AI Batch Generation Mode - {display_name}")
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

            # Load LoRA(s) via hook method (FLUX.1 overrides for uncensored support)
            self.load_generation_loras(pipe, local_lora_path)

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
                "model": display_name,
            }
            # Add any model-specific metadata
            metadata.update(self.get_generation_metadata_extras())

            metadata_blob = output_bucket.blob(f"{output_prefix}/metadata.json")
            metadata_blob.upload_from_string(json.dumps(metadata, indent=2))

            logger.info("=" * 60)
            logger.info(f"{display_name} Generation completed! {len(images)} images")
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

        display_name = self.get_display_name()

        logger.info("=" * 60)
        logger.info(f"Vertex AI Batch Training Mode - {display_name}")
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
                "model": display_name,
            }
            metadata_blob_path = f"{output_prefix}/{job_id}/metadata.json"
            metadata_blob = output_bucket.blob(metadata_blob_path)
            metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
            logger.info(f"Uploaded metadata to gs://{output_bucket_name}/{metadata_blob_path}")

            logger.info("=" * 60)
            logger.info(f"{display_name} Training completed successfully!")
            logger.info(f"LoRA: gs://{output_bucket_name}/{lora_blob_path}")
            logger.info("=" * 60)

            sys.exit(0)

        except Exception as e:
            logger.error(f"Batch training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # =========================================================================
    # Route registration - training and management endpoints
    # =========================================================================

    def _register_routes(self):
        """Register training and management API endpoints."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": self.service_name,
                "current_job": self._current_job,
                "jobs_count": len(self._training_jobs),
                "workspace_path": _runtime_paths["base_path"],
            }

        @self.app.get("/debug/exec")
        async def debug_exec(cmd: str = "which simpletuner"):
            """Debug endpoint to execute commands and see what's available."""
            import subprocess
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=HTTP_TIMEOUT_DEFAULT
                )
                return {
                    "command": cmd,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }
            except Exception as e:
                return {"command": cmd, "error": str(e)}

        @self.app.get("/debug/logs/{job_id}")
        async def get_training_logs(job_id: str, lines: int = 100):
            """Get training logs for a job."""
            if job_id not in self._training_jobs:
                raise HTTPException(status_code=404, detail="Job not found")

            job = self._training_jobs[job_id]
            output_path = f"{_runtime_paths['output_path']}/{job_id}"

            logs = {
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "debug_log": None,
                "tensorboard_events": [],
                "checkpoints": [],
            }

            # Read SimpleTuner's debug.log (may be in /app or output dir)
            for debug_log_path in [Path("/app/debug.log"), Path(output_path) / "logs" / "debug.log"]:
                if debug_log_path.exists():
                    try:
                        with open(debug_log_path, "r") as f:
                            all_lines = f.readlines()
                            logs["debug_log"] = "".join(all_lines[-lines:])
                        break
                    except Exception as e:
                        logs["debug_log_error"] = str(e)

            # List TensorBoard event files
            logs_dir = Path(output_path) / "logs"
            if logs_dir.exists():
                for f in logs_dir.glob("events.out.tfevents.*"):
                    logs["tensorboard_events"].append({
                        "name": f.name,
                        "size_kb": f.stat().st_size / 1024,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    })

            # List checkpoints
            output_dir = Path(output_path)
            if output_dir.exists():
                for checkpoint in output_dir.glob("checkpoint-*"):
                    if checkpoint.is_dir():
                        lora_file = checkpoint / "pytorch_lora_weights.safetensors"
                        logs["checkpoints"].append({
                            "name": checkpoint.name,
                            "has_lora": lora_file.exists(),
                            "lora_size_mb": lora_file.stat().st_size / (1024*1024) if lora_file.exists() else None,
                        })

            return logs

        @self.app.get("/ready")
        async def ready_check():
            """Readiness check endpoint."""
            if self._current_job is not None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Training in progress: {self._current_job}"
                )

            return {
                "status": "ready",
                "service": self.service_name,
                "can_accept_job": True,
            }

        @self.app.post("/train")
        async def start_training(
            background_tasks: BackgroundTasks,
            image: UploadFile = File(...),
            companion_id: str = Form(...),
            trigger_word: Optional[str] = Form(None),
            steps: int = Form(1000),
            lora_rank: int = Form(16),
            callback_url: Optional[str] = Form(None),
        ):
            """Start LoRA training for a companion."""
            if self._current_job is not None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Training already in progress: {self._current_job}"
                )

            job_id = str(uuid.uuid4())

            # Generate trigger word if not provided
            if not trigger_word:
                clean_id = companion_id.replace("-", "")[:8]
                trigger_word = f"TOK{clean_id}"

            # Create dataset directory with the image - use configured paths
            dataset_path = f"{_runtime_paths['datasets_path']}/{job_id}"
            output_path = f"{_runtime_paths['output_path']}/{job_id}"
            os.makedirs(dataset_path, exist_ok=True)

            # Save uploaded image
            image_path = f"{dataset_path}/image_001.jpg"
            async with aiofiles.open(image_path, "wb") as f:
                content = await image.read()
                await f.write(content)

            logger.info(f"Saved training image: {image_path} ({len(content)} bytes)")

            # Create job record
            self._training_jobs[job_id] = {
                "job_id": job_id,
                "companion_id": companion_id,
                "trigger_word": trigger_word,
                "status": "queued",
                "progress": 0.0,
                "current_step": 0,
                "total_steps": steps,
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": None,
                "lora_path": None,
                "error": None,
                "callback_url": callback_url,
            }

            # Create SimpleTuner config
            cache_path = f"{output_path}/cache"
            config = self.create_simpletuner_config(
                job_id=job_id,
                trigger_word=trigger_word,
                dataset_path=dataset_path,
                output_path=output_path,
                cache_path=cache_path,
                steps=steps,
                lora_rank=lora_rank,
            )

            # Start training in background
            background_tasks.add_task(self.run_training, job_id, config, dataset_path)

            return {
                "job_id": job_id,
                "status": "queued",
                "companion_id": companion_id,
                "trigger_word": trigger_word,
                "estimated_minutes": steps // 60 + 10,  # Rough estimate
            }

        @self.app.get("/status/{job_id}")
        async def get_status(job_id: str):
            """Get training job status."""
            if job_id not in self._training_jobs:
                raise HTTPException(status_code=404, detail="Job not found")

            job = self._training_jobs[job_id]
            return {
                "job_id": job["job_id"],
                "companion_id": job["companion_id"],
                "trigger_word": job["trigger_word"],
                "status": job["status"],
                "progress": job["progress"],
                "current_step": job["current_step"],
                "total_steps": job["total_steps"],
                "error": job["error"],
                "started_at": job["started_at"],
                "completed_at": job["completed_at"],
                "lora_path": job.get("lora_path"),
            }

        @self.app.get("/download/{job_id}")
        async def download_lora(job_id: str):
            """Download trained LoRA weights."""
            if job_id not in self._training_jobs:
                raise HTTPException(status_code=404, detail="Job not found")

            job = self._training_jobs[job_id]

            if job["status"] != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Training not complete. Status: {job['status']}"
                )

            if not job["lora_path"] or not os.path.exists(job["lora_path"]):
                raise HTTPException(status_code=404, detail="LoRA file not found")

            return FileResponse(
                job["lora_path"],
                media_type="application/octet-stream",
                filename=f"{job['companion_id']}.safetensors"
            )

        @self.app.get("/current-job")
        async def get_current_job():
            """Get information about the currently running job."""
            if self._current_job is None:
                return {"current_job": None, "status": "idle"}

            job = self._training_jobs.get(self._current_job, {})
            return {
                "current_job": self._current_job,
                "status": job.get("status", "unknown"),
                "progress": job.get("progress", 0.0),
                "current_step": job.get("current_step", 0),
                "total_steps": job.get("total_steps", 0),
                "companion_id": job.get("companion_id"),
                "started_at": job.get("started_at"),
            }

        @self.app.post("/cancel/{job_id}")
        async def cancel_training(job_id: str):
            """Cancel a running training job."""
            if job_id not in self._training_jobs:
                raise HTTPException(status_code=404, detail="Job not found")

            job = self._training_jobs[job_id]

            if job["status"] in ("completed", "failed", "cancelled"):
                return {
                    "job_id": job_id,
                    "status": job["status"],
                    "message": f"Job already {job['status']}, cannot cancel"
                }

            # Mark as cancelled
            job["status"] = "cancelled"
            job["error"] = "Cancelled by user"
            job["completed_at"] = datetime.utcnow().isoformat()

            # Clear current job if this was it
            if self._current_job == job_id:
                self._current_job = None

            logger.info(f"Training job {job_id} marked as cancelled")

            return {
                "job_id": job_id,
                "status": "cancelled",
                "message": "Job marked as cancelled. Note: The training process may still be running. Restart pod if needed."
            }

        @self.app.post("/clear-current-job")
        async def clear_current_job():
            """Force-clear the current job lock."""
            old_job = self._current_job
            self._current_job = None

            if old_job:
                if old_job in self._training_jobs:
                    job = self._training_jobs[old_job]
                    if job["status"] not in ("completed", "failed", "cancelled"):
                        job["status"] = "failed"
                        job["error"] = "Force-cleared by admin"
                        job["completed_at"] = datetime.utcnow().isoformat()

                logger.warning(f"Force-cleared current job lock: {old_job}")
                return {
                    "cleared_job": old_job,
                    "message": "Current job lock cleared. New training can be submitted."
                }

            return {
                "cleared_job": None,
                "message": "No job was running."
            }

        @self.app.get("/loras")
        async def list_loras():
            """List available trained LoRAs."""
            loras = []
            lora_dir = Path(_runtime_paths["lora_path"])

            if lora_dir.exists():
                for entry in lora_dir.iterdir():
                    if entry.is_dir():
                        lora_file = entry / "pytorch_lora_weights.safetensors"
                        if lora_file.exists():
                            loras.append({
                                "id": entry.name,
                                "path": str(lora_file),
                                "size_mb": lora_file.stat().st_size / (1024 * 1024),
                            })

            return {"loras": loras, "count": len(loras)}

        @self.app.delete("/job/{job_id}")
        async def delete_job(job_id: str):
            """Delete a completed job and its files."""
            if job_id not in self._training_jobs:
                raise HTTPException(status_code=404, detail="Job not found")

            job = self._training_jobs[job_id]

            if job["status"] not in ("completed", "failed"):
                raise HTTPException(
                    status_code=400,
                    detail="Can only delete completed or failed jobs"
                )

            # Clean up files - use configured paths
            dataset_path = f"{_runtime_paths['datasets_path']}/{job_id}"
            output_path = f"{_runtime_paths['output_path']}/{job_id}"

            if os.path.exists(dataset_path):
                shutil.rmtree(dataset_path)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

            del self._training_jobs[job_id]

            return {"status": "deleted", "job_id": job_id}

    # =========================================================================
    # Route registration - inference and generation endpoints
    # =========================================================================

    def _register_inference_routes(self):
        """Register inference/generation API endpoints."""
        import requests as req_lib

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
            """Generate images with a trained LoRA."""
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
                    from io import BytesIO
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
            ipfs_gateway: Optional[str] = Form(None),
        ):
            """Start async image generation."""
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

                    # Use canonical gateway URL if not provided
                    gateway_url = ipfs_gateway or get_lighthouse_gateway_url()

                    # Resolve LoRA path - IPFS takes priority
                    if lora_ipfs_cid:
                        # Download from IPFS gateway
                        self._generation_jobs[job_id]["status"] = "downloading_lora"
                        local_lora_dir = f"{_runtime_paths['tmp_path']}/ipfs_lora_{job_id}"
                        os.makedirs(local_lora_dir, exist_ok=True)
                        local_lora_file = f"{local_lora_dir}/pytorch_lora_weights.safetensors"

                        ipfs_url = f"{gateway_url}/{lora_ipfs_cid}"
                        logger.info(f"Downloading LoRA from IPFS: {ipfs_url}")

                        response = req_lib.get(ipfs_url, timeout=HTTP_TIMEOUT_DOWNLOAD, stream=True)
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
            """Pre-download and optionally pre-quantize the model."""
            display_name = self.get_display_name()

            if self._preload_status["status"] == "running":
                return {"status": "already_running", "progress": self._preload_status["progress"]}

            # Check if quantized cache exists (best case)
            if quantize and self.is_quantized_model_cached():
                self._preload_status = {"status": "complete", "progress": "Quantized model already cached!", "error": None}
                logger.info(f"Quantized {display_name} already cached, skipping")
                return {
                    "status": "already_cached",
                    "message": f"Quantized {display_name} model is already cached on disk",
                    "cache_path": self.quantized_cache_dir,
                }

            # Check if base model is cached (partial case)
            if self._check_model_cached() and not quantize:
                self._preload_status = {"status": "complete", "progress": "Model already cached!", "error": None}
                logger.info(f"{display_name} already cached, skipping download")
                return {"status": "already_cached", "message": f"{display_name} model is already cached"}

            def download_and_quantize():
                try:
                    # Step 1: Download base model if needed
                    if not self._check_model_cached():
                        self._preload_status = {"status": "running", "progress": f"Downloading {display_name} (~60GB)...", "error": None}
                        from huggingface_hub import snapshot_download
                        logger.info(f"Starting {display_name} download...")
                        snapshot_download(
                            self.get_model_name(),
                            local_dir_use_symlinks=True
                        )
                        logger.info(f"{display_name} download complete!")

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

    def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server."""
        uvicorn.run(self.app, host=host, port=port)


# =========================================================================
# Shared CLI entry point
# =========================================================================

def run_main(api_class, description: str):
    """
    Shared main() entry point for FLUX API implementations.

    Handles argument parsing, GPU logging, and mode dispatch
    (API server, Vertex AI training, Vertex AI generation).
    """
    import argparse

    parser = argparse.ArgumentParser(description=description)

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
        default=None,
        help=f"IPFS gateway URL (default: {get_lighthouse_gateway_url()})"
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

    # Create API instance
    api = api_class()
    display_name = api.get_display_name()

    logger.info("=" * 60)
    logger.info(f"SimpleTuner {display_name} Training")
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
