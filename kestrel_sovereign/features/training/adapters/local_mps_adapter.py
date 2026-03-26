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
DEFAULT_WORKING_DIR = os.environ.get("LOCAL_MPS_WORKING_DIR", "/Volumes/data2/models/local-training")
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

        Unified interface matching RunPod/VertexAI adapters.
        Loads LoRA from config.lora_path (local file) or lora_bytes.

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
        num_inference_steps = config.num_inference_steps if config else 30
        guidance_scale = config.guidance_scale if config else 7.5
        width = config.width if config else 1024
        height = config.height if config else 1024

        # Load LoRA bytes from file path if not provided directly
        if not lora_bytes and config and config.lora_path:
            lora_file = config.lora_path
            # Strip file:// prefix if present
            if lora_file.startswith("file://"):
                lora_file = lora_file[7:]
            if Path(lora_file).exists():
                lora_bytes = Path(lora_file).read_bytes()
                logger.info(f"Loaded LoRA from {lora_file} ({len(lora_bytes)} bytes)")
            else:
                return GenerationResult(
                    job_id="local-mps", images=[], state=GenerationState.FAILED,
                    error=f"LoRA file not found: {lora_file}",
                )

        if not lora_bytes:
            return GenerationResult(
                job_id="local-mps", images=[], state=GenerationState.FAILED,
                error="No LoRA weights provided (no lora_path or lora_bytes)",
            )

        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as e:
            return GenerationResult(
                job_id="local-mps",
                images=[],
                state=GenerationState.FAILED,
                error=f"Missing dependencies: {e}. Install torch and diffusers in the server venv.",
            )

        # Load pipeline lazily
        if not self._pipeline_loaded:
            logger.info(f"Loading SDXL pipeline from {self.model_path}")
            model_path_str = str(self.model_path)

            # Handle single safetensors file (like Illustrious XL) vs diffusers directory
            if model_path_str.endswith(".safetensors"):
                self._pipeline = StableDiffusionXLPipeline.from_single_file(
                    model_path_str,
                    torch_dtype=torch.float16,
                )
            else:
                try:
                    self._pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_path_str,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True,
                    )
                except (OSError, ValueError):
                    # fp16 variant not available, load without variant
                    logger.info("fp16 variant not available, loading full precision")
                    self._pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_path_str,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                    )
            self._pipeline.to("mps")
            self._pipeline_loaded = True

        # Save LoRA temporarily
        temp_lora = self.working_dir / "temp_lora.safetensors"
        temp_lora.write_bytes(lora_bytes)

        try:
            self._pipeline.load_lora_weights(str(temp_lora))

            generator = torch.Generator(device="mps").manual_seed(42)

            def generate():
                return self._pipeline(
                    prompt=prompt,
                    negative_prompt="",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=width,
                    height=height,
                ).images[0]

            image = await asyncio.to_thread(generate)

            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            data_url = f"data:image/png;base64,{img_base64}"

            self._pipeline.unload_lora_weights()

            return GenerationResult(
                job_id="inline-generation",
                state=GenerationState.COMPLETED,
                images=[data_url],
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            try:
                self._pipeline.unload_lora_weights()
            except Exception:
                pass
            return GenerationResult(
                job_id="inline-generation",
                state=GenerationState.FAILED,
                images=[],
                error=str(e),
            )

        finally:
            if temp_lora.exists():
                temp_lora.unlink()

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
