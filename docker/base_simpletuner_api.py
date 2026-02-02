#!/usr/bin/env python3
"""
Base SimpleTuner API Wrapper for Kestrel LoRA Training

Provides shared functionality for both FLUX.1 and FLUX.2 training APIs.
Contains all common endpoints, path management, and training logic.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
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
)

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

        # Register routes
        self._register_routes()

    # Abstract methods that subclasses must implement
    def get_model_family(self) -> str:
        """Return the model family string for SimpleTuner config."""
        raise NotImplementedError

    def get_model_name(self) -> str:
        """Return the HuggingFace model name."""
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

    def _register_routes(self):
        """Register all API endpoints."""

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

    def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server."""
        uvicorn.run(self.app, host=host, port=port)