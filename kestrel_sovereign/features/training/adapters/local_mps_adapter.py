"""
Local MPS Training Adapter.

Training provider for Apple Silicon Macs using Metal Performance Shaders (MPS).
Uses HuggingFace diffusers for LoRA training on SDXL with text encoder training.

Key feature: --train_text_encoder flag enables trigger word to encode character identity.
This is what makes "TOKEMMA" alone generate the character without description.

Environment Variables:
    LOCAL_MPS_MODEL_PATH: Path to SDXL model in diffusers format (required)
    LOCAL_MPS_WORKING_DIR: Working directory for training (default: ~/models/local-training)
    DIFFUSERS_PATH: Path to diffusers installation (required for training)
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from ..protocol import TrainingProvider, TrainingSubmissionError
from ..types import (
    TrainingConfig,
    TrainingJob,
    TrainingState,
    TrainingStatus,
    ProviderType,
    GenerationConfig,
    GenerationResult,
    GenerationState,
)

logger = logging.getLogger(__name__)

# Default paths - configure via environment variables
# RealVisXL V5.0 in diffusers format - photorealistic, top-rated SDXL model
# Used for both training (with text encoder) and inference
DEFAULT_MODEL_PATH = os.environ.get("LOCAL_MPS_MODEL_PATH", "")
DEFAULT_WORKING_DIR = os.environ.get(
    "LOCAL_MPS_WORKING_DIR",
    os.path.join(os.environ.get("KESTREL_DATA_DIR", os.path.expanduser("~")), "kestrel-training"),
)
DEFAULT_DIFFUSERS_PATH = os.environ.get("DIFFUSERS_PATH", "")


class LocalMPSTrainingAdapter(TrainingProvider):
    """
    Local training provider for Apple Silicon using MPS backend.

    Uses HuggingFace diffusers train_text_to_image_lora_sdxl.py with --train_text_encoder.
    This enables trigger words to encode character identity in the text encoder.
    """

    provider_name = "local_mps"
    provider_type = ProviderType.LOCAL

    def __init__(
        self,
        model_path: str | None = None,
        working_dir: str | None = None,
        diffusers_path: str | None = None,
    ):
        """
        Initialize local MPS training provider.

        Args:
            model_path: Path to SDXL model in diffusers format
            working_dir: Working directory for training jobs
            diffusers_path: Path to diffusers installation with training scripts
        """
        self.model_path = Path(
            model_path or os.getenv("LOCAL_MPS_MODEL_PATH", DEFAULT_MODEL_PATH)
        )
        self.working_dir = Path(
            working_dir or os.getenv("LOCAL_MPS_WORKING_DIR", DEFAULT_WORKING_DIR)
        )
        self.diffusers_path = Path(
            diffusers_path or os.getenv("DIFFUSERS_PATH", DEFAULT_DIFFUSERS_PATH)
        )

        # Create directories
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.working_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir = self.working_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.working_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.working_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Active training jobs (job_id -> job_info)
        self._active_jobs: dict[str, dict[str, Any]] = {}
        self._training_processes: dict[str, subprocess.Popen] = {}

        # Lazy-loaded diffusers pipeline for generation
        self._pipeline = None
        self._pipeline_loaded = False

    def is_available(self) -> bool:
        """
        Check if local MPS training is available.

        Returns True if:
        - SDXL model directory exists (diffusers format)
        - Diffusers training scripts installed
        - PyTorch MPS backend is available
        """
        # Check model path (diffusers format directory)
        if not self.model_path.exists():
            logger.warning(f"SDXL model not found at {self.model_path}")
            return False

        # Check for diffusers format (should have model_index.json)
        model_index = self.model_path / "model_index.json"
        if not model_index.exists():
            logger.warning(f"Model at {self.model_path} is not in diffusers format (missing model_index.json)")
            return False

        # Check diffusers training script
        training_script = self.diffusers_path / "examples/text_to_image/train_text_to_image_lora_sdxl.py"
        if not training_script.exists():
            logger.warning(f"Diffusers training script not found at {training_script}")
            return False

        # Check PyTorch MPS
        try:
            import torch
            if not torch.backends.mps.is_available():
                logger.warning("PyTorch MPS backend not available")
                return False
        except ImportError:
            logger.warning("PyTorch not installed")
            return False

        return True

    async def start_training(
        self,
        companion_id: str,
        avatar_data: bytes,
        config: TrainingConfig | None = None,
    ) -> TrainingJob:
        """
        Start LoRA training using HuggingFace diffusers with text encoder training.

        Uses train_text_to_image_lora_sdxl.py with --train_text_encoder flag.
        This enables the trigger word to encode character identity in the text encoder,
        so "TOKEMMA" alone generates the character without needing descriptions.

        Args:
            companion_id: UUID of companion being trained
            avatar_data: Avatar image bytes (JPEG/PNG)
            config: Training configuration

        Returns:
            TrainingJob with job details
        """
        if not self.is_available():
            raise TrainingSubmissionError(
                "Local MPS training not available",
                provider=self.provider_name,
                details={"check": "Run is_available() for details"}
            )

        config = config or TrainingConfig()
        job_id = str(uuid.uuid4())
        trigger_word = config.trigger_word or f"TOK{companion_id[:8]}"

        # Create directories for this job
        job_dir = self.configs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = self.datasets_dir / job_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        output_dir = self.output_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save avatar image
        avatar_filename = f"{trigger_word}_portrait.png"
        avatar_path = dataset_dir / avatar_filename
        avatar_path.write_bytes(avatar_data)

        # Create metadata.jsonl for HuggingFace datasets format
        # Caption uses trigger word to bind identity to the token
        metadata_path = dataset_dir / "metadata.jsonl"
        metadata_content = json.dumps({
            "file_name": avatar_filename,
            "text": f"{trigger_word} portrait photo, professional headshot"
        })
        metadata_path.write_text(metadata_content + "\n")

        started_at = datetime.now(timezone.utc)
        log_file = job_dir / "training.log"

        # Start diffusers training in background
        try:
            env = os.environ.copy()
            env["TOKENIZERS_PARALLELISM"] = "false"

            # Training parameters
            steps = config.steps or 500
            learning_rate = config.learning_rate or 1e-4
            lora_rank = config.lora_rank or 128  # High rank for strong identity encoding

            # Diffusers script needs a single int resolution, not multi-res string
            raw_resolution = config.resolution or "512"
            resolution = int(str(raw_resolution).split(",")[0].strip())

            # Build training command using diffusers script
            training_script = self.diffusers_path / "examples/text_to_image/train_text_to_image_lora_sdxl.py"

            from kestrel_sovereign.kestrel_config.constants import DEFAULT_TRAINING_BATCH_SIZE

            cmd = [
                str(self.diffusers_path / ".venv/bin/python3"),
                str(training_script),
                f"--pretrained_model_name_or_path={self.model_path}",
                f"--train_data_dir={dataset_dir}",
                "--caption_column=text",
                f"--resolution={resolution}",
                f"--train_batch_size={DEFAULT_TRAINING_BATCH_SIZE}",
                "--num_train_epochs=1",
                f"--max_train_steps={steps}",
                "--gradient_checkpointing",
                f"--learning_rate={learning_rate}",
                "--lr_scheduler=constant",
                "--lr_warmup_steps=0",
                "--seed=42",
                f"--output_dir={output_dir}",
                "--train_text_encoder",  # CRITICAL: Enable text encoder LoRA
                f"--rank={lora_rank}",
            ]

            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.diffusers_path),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

            self._training_processes[job_id] = process
            logger.info(f"Training process started (PID: {process.pid}), logging to {log_file}")

        except Exception as e:
            raise TrainingSubmissionError(
                f"Failed to start training: {e}",
                provider=self.provider_name,
                details={"error": str(e)}
            )

        # Track job
        self._active_jobs[job_id] = {
            "companion_id": companion_id,
            "job_dir": job_dir,
            "dataset_dir": dataset_dir,
            "output_dir": output_dir,
            "log_file": log_file,
            "started_at": started_at,
            "trigger_word": trigger_word,
            "config": config,
            "steps": config.steps or 500,
        }

        job = TrainingJob(
            job_id=job_id,
            companion_id=companion_id,
            provider=self.provider_name,
            state=TrainingState.TRAINING,
            trigger_word=trigger_word,
            created_at=started_at,
            started_at=started_at,
            config=config,
        )

        logger.info(
            f"Started local MPS training for {companion_id[:8]} "
            f"(job: {job_id[:8]}, trigger: {trigger_word})"
        )

        return job

    async def get_status(self, job_id: str) -> TrainingStatus:
        """
        Check training progress.

        Args:
            job_id: Training job ID

        Returns:
            TrainingStatus with current state and progress
        """
        if job_id not in self._active_jobs:
            raise ValueError(f"Unknown job: {job_id}")

        job_info = self._active_jobs[job_id]
        elapsed = (datetime.now(timezone.utc) - job_info["started_at"]).total_seconds()
        output_dir = Path(job_info["output_dir"])

        # Check if process is still running
        process = self._training_processes.get(job_id)
        if process:
            poll_result = process.poll()

            if poll_result is None:
                # Still running - estimate progress from time
                # Rough estimate: ~1 step/second on M3 Ultra for SDXL
                estimated_steps = min(int(elapsed), job_info["steps"])
                progress = estimated_steps / job_info["steps"]

                return TrainingStatus(
                    job_id=job_id,
                    state=TrainingState.TRAINING,
                    progress=progress,
                    message=f"Training ~{estimated_steps}/{job_info['steps']} steps",
                    elapsed_seconds=elapsed,
                    provider_details={
                        "current_step": estimated_steps,
                        "total_steps": job_info["steps"],
                    },
                )

            elif poll_result == 0:
                # Completed successfully
                return TrainingStatus(
                    job_id=job_id,
                    state=TrainingState.COMPLETED,
                    progress=1.0,
                    message="Training completed",
                    elapsed_seconds=elapsed,
                )

            else:
                # Failed - read error from log file
                log_file = job_info.get("log_file")
                error_msg = f"Training failed (exit {poll_result})"
                if log_file and Path(log_file).exists():
                    log_content = Path(log_file).read_text()
                    # Get last 500 chars of log
                    error_msg = f"{error_msg}: {log_content[-500:]}"
                return TrainingStatus(
                    job_id=job_id,
                    state=TrainingState.FAILED,
                    progress=0.0,
                    error=error_msg,
                    elapsed_seconds=elapsed,
                )

        # No process - check for output files
        lora_files = list(output_dir.glob("**/*.safetensors"))
        if lora_files:
            return TrainingStatus(
                job_id=job_id,
                state=TrainingState.COMPLETED,
                progress=1.0,
                message="Training completed",
                elapsed_seconds=elapsed,
            )

        return TrainingStatus(
            job_id=job_id,
            state=TrainingState.PENDING,
            progress=0.0,
            message="Training status unknown",
            elapsed_seconds=elapsed,
        )

    async def download_weights(self, job_id: str) -> bytes | None:
        """
        Get trained LoRA weights.

        Args:
            job_id: Completed training job ID

        Returns:
            LoRA weights as bytes (.safetensors format)
        """
        if job_id not in self._active_jobs:
            return None

        job_info = self._active_jobs[job_id]
        output_dir = Path(job_info["output_dir"])

        # Find LoRA files
        lora_files = list(output_dir.glob("*.safetensors"))
        if not lora_files:
            # Check checkpoints
            lora_files = list(output_dir.glob("checkpoint-*/*.safetensors"))

        if not lora_files:
            logger.warning(f"No LoRA weights found in {output_dir}")
            return None

        # Get most recent
        lora_path = sorted(lora_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        logger.info(f"Loading LoRA from {lora_path}")

        return lora_path.read_bytes()

    async def cancel(self, job_id: str) -> bool:
        """
        Cancel training job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled successfully
        """
        process = self._training_processes.get(job_id)
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            del self._training_processes[job_id]
            logger.info(f"Cancelled training job {job_id[:8]}")
            return True
        return False

    async def cleanup(self, job_id: str) -> None:
        """
        Cleanup training artifacts.

        Args:
            job_id: Job ID to cleanup
        """
        if job_id in self._training_processes:
            await self.cancel(job_id)

        if job_id in self._active_jobs:
            job_info = self._active_jobs[job_id]

            # Clean dataset images (keep LoRA output)
            dataset_dir = Path(job_info["dataset_dir"])
            for img in dataset_dir.glob("*"):
                img.unlink()
            try:
                dataset_dir.rmdir()
            except OSError:
                pass

            del self._active_jobs[job_id]

    async def generate_image(
        self,
        config: GenerationConfig,
        session=None,
        lora_ipfs_cid: str | None = None,
        ipfs_gateway: str | None = None,
        flux_version: str | None = None,
        lora_bytes: bytes | None = None,
    ) -> GenerationResult:
        """
        Generate image with trained LoRA using SDXL on MPS.

        Runs generation as a subprocess using the diffusers venv's Python,
        same pattern as training. This avoids version conflicts between the
        server's diffusers and the training/inference diffusers.

        Args:
            config: Generation configuration (prompt, lora_path, dimensions, etc.)
            session: Ignored (RunPod compatibility)
            lora_ipfs_cid: Ignored (no IPFS fetch on local)
            ipfs_gateway: Ignored
            flux_version: Ignored (always SDXL on MPS)
            lora_bytes: Direct LoRA bytes (optional, overrides config.lora_path)

        Returns:
            GenerationResult with base64 images
        """
        prompt = config.prompt if config else ""
        num_inference_steps = config.num_inference_steps if config else 40
        guidance_scale = config.guidance_scale if config else 7.5
        width = config.width if config else 1024
        height = config.height if config else 1024

        # Resolve LoRA path
        lora_path = None
        if config and config.lora_path:
            lora_path = config.lora_path
            if lora_path.startswith("file://"):
                lora_path = lora_path[7:]
        elif lora_bytes:
            # Save bytes to temp file
            temp_lora = self.working_dir / "temp_lora.safetensors"
            temp_lora.write_bytes(lora_bytes)
            lora_path = str(temp_lora)

        if not lora_path or not Path(lora_path).exists():
            return GenerationResult(
                job_id="local-mps", images=[], state=GenerationState.FAILED,
                error=f"LoRA file not found: {lora_path}",
            )

        # Use diffusers venv Python for generation (avoids version conflicts)
        diffusers_python = str(self.diffusers_path / ".venv/bin/python3")
        if not Path(diffusers_python).exists():
            return GenerationResult(
                job_id="local-mps", images=[], state=GenerationState.FAILED,
                error=f"Diffusers Python not found: {diffusers_python}",
            )

        output_path = self.working_dir / "generated_selfie.png"

        # Inline generation script — runs in the diffusers venv
        script = f"""
import torch, base64, sys
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "{self.model_path}",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.load_lora_weights("{lora_path}")
pipe.to("mps")

image = pipe(
    prompt="{prompt.replace('"', '\\"')}",
    negative_prompt="deformed, bad anatomy, missing limbs, extra limbs, missing arms, extra arms, missing fingers, extra fingers, mutated hands, blurry, low quality",
    num_inference_steps={num_inference_steps},
    guidance_scale={guidance_scale},
    width={width},
    height={height},
    generator=torch.Generator(device="mps").manual_seed(42),
).images[0]

image.save("{output_path}")
print("OK")
"""
        logger.info(f"Generating selfie via subprocess: {prompt[:60]}...")
        start_time = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                diffusers_python, "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=300,
            )

            elapsed = asyncio.get_event_loop().time() - start_time

            if process.returncode != 0:
                error_msg = stderr.decode()[-500:] if stderr else "Unknown error"
                logger.error(f"Generation subprocess failed: {error_msg}")
                return GenerationResult(
                    job_id="local-mps", images=[], state=GenerationState.FAILED,
                    error=f"Generation failed: {error_msg}",
                    elapsed_seconds=elapsed,
                )

            # Read output image and convert to base64
            if not output_path.exists():
                return GenerationResult(
                    job_id="local-mps", images=[], state=GenerationState.FAILED,
                    error="Generation produced no output file",
                    elapsed_seconds=elapsed,
                )

            img_bytes = output_path.read_bytes()
            img_base64 = base64.b64encode(img_bytes).decode()
            data_url = f"data:image/png;base64,{img_base64}"

            logger.info(f"Selfie generated in {elapsed:.1f}s ({len(img_bytes)} bytes)")

            return GenerationResult(
                job_id="local-mps",
                state=GenerationState.COMPLETED,
                images=[data_url],
                elapsed_seconds=elapsed,
            )

        except asyncio.TimeoutError:
            return GenerationResult(
                job_id="local-mps", images=[], state=GenerationState.FAILED,
                error="Generation timed out (300s)",
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                job_id="local-mps", images=[], state=GenerationState.FAILED,
                error=str(e),
            )
        finally:
            if output_path.exists():
                output_path.unlink()

    async def close(self) -> None:
        """Close resources."""
        # Cancel any active training
        for job_id in list(self._training_processes.keys()):
            await self.cancel(job_id)

        # Unload pipeline
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._pipeline_loaded = False

            import gc
            gc.collect()

            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

        logger.info("LocalMPSTrainingAdapter closed")
