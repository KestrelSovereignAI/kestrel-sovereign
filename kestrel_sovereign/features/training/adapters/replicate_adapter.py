"""
Replicate adapter for TrainingProvider protocol.

Provides unified LoRA training and image generation using Replicate's serverless infrastructure.
Training and generation run on managed infrastructure with no instance lifecycle management needed.

Uses the Replicate API directly for FLUX.1 LoRA training:
- ~$2-5 per training run
- ~15-20 minutes training time
- Weights downloaded and stored locally after training

Image generation with trained LoRA:
- ~$0.003-0.03 per image depending on model
- Multiple FLUX models supported (schnell, dev, pro)
- NOTE: Replicate applies content safety filters.
"""

import asyncio
import base64
import hashlib
import io
import logging
import os
import tarfile
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

from kestrel_sovereign.kestrel_config.constants import REPLICATE_POLL_INTERVAL_SECONDS

from ..types import (
    TrainingJob,
    TrainingStatus,
    TrainingConfig,
    TrainingState,
    ProviderType,
    GenerationConfig,
    GenerationResult,
    GenerationState,
    ProviderCapabilities,
)
from ..protocol import (
    TrainingProviderError,
    ProviderNotAvailableError,
    TrainingSubmissionError,
    DownloadError,
)
from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_DOWNLOAD,
)
from kestrel_sovereign.kestrel_config.defaults import get_lighthouse_gateway_url

logger = logging.getLogger(__name__)


class ReplicateTrainingAdapter:
    """
    Adapts Replicate API to TrainingProvider protocol.

    Replicate is serverless - training and generation run on managed infrastructure.
    - Training: ~$2-5 per run, ~15-20 minutes
    - Generation: ~$0.003-0.03 per image
    - FLUX.1-dev based (not FLUX.2)
    - Replicate applies content safety filters.
    """

    TRAINER_VERSION = "ostris/flux-dev-lora-trainer:b6af14222e6bd9be257cbc1ea4afda3cd0503e1133083b9d1de0364d8568e6ef"

    # Available inference models - schnell is fastest and cheapest
    # Note: Some models require version hash for replicate.run() to work
    INFERENCE_MODELS = {
        "schnell": "lucataco/flux-schnell-lora:2a6b576af31790b470f0a8442e1e9791213fa13799cbb65a9fc1436e96389574",  # LoRA-enabled
        "dev": "black-forest-labs/flux-dev",           # Higher quality, non-commercial license
        "pro": "black-forest-labs/flux-1.1-pro",       # Best quality, commercial license
    }

    # Provider capabilities (used by factory for intelligent routing)
    CAPABILITIES = ProviderCapabilities(
        training=True,
        generation=True,
        unfiltered_generation=False,  # Replicate applies content safety filters
        flux_version="1.x",  # Uses FLUX.1, not FLUX.2
        supports_lora_download=True,  # Weights can be downloaded for use elsewhere
    )

    def __init__(self, replicate_username: str = "kestrel"):
        """
        Initialize Replicate adapter.

        Args:
            replicate_username: Replicate username for model destination
        """
        self.replicate_username = replicate_username
        self._jobs: Dict[str, TrainingJob] = {}
        self._training_data: Dict[str, Dict] = {}  # Store Replicate-specific data

    @property
    def provider_name(self) -> str:
        return "replicate"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.SERVERLESS

    def is_available(self) -> bool:
        """Check if Replicate API token is available."""
        return bool(os.getenv("REPLICATE_API_TOKEN"))

    async def start_training(
        self,
        companion_id: str,
        avatar_data: bytes,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingJob:
        """Start LoRA training on Replicate."""
        if not self.is_available():
            raise ProviderNotAvailableError(
                "REPLICATE_API_TOKEN not set",
                provider=self.provider_name
            )

        try:
            import replicate
        except ImportError:
            raise ProviderNotAvailableError(
                "replicate package not installed",
                provider=self.provider_name
            )

        config = config or TrainingConfig()

        # Generate trigger word
        trigger_word = config.trigger_word or f"TOK{companion_id[:8]}"

        # Create training job
        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            companion_id=companion_id,
            provider=self.provider_name,
            state=TrainingState.PREPARING,
            trigger_word=trigger_word,
            created_at=datetime.now(timezone.utc),
            config=config,
        )

        self._jobs[job.job_id] = job
        self._training_data[job.job_id] = {
            "replicate_training_id": None,
            "replicate_model_version": None,
            "weights_url": None,
            "avatar_data": avatar_data,
        }

        # Start training in background
        asyncio.create_task(self._run_training(job, avatar_data, trigger_word))

        logger.info(f"Started Replicate training job: {job.job_id}")
        return job

    async def _run_training(
        self,
        job: TrainingJob,
        avatar_data: bytes,
        trigger_word: str
    ):
        """Background task that runs Replicate training."""
        try:
            import replicate

            # Step 1: Create training zip. The Replicate Python client accepts
            # local file objects as inputs and uploads them through Replicate's
            # own file path, avoiding ad hoc public temporary hosting.
            job.state = TrainingState.PREPARING
            zip_data = self._build_training_zip(avatar_data)

            # Step 2: Create model destination
            model_name = f"lora-{job.companion_id[:8]}"
            destination = f"{self.replicate_username}/{model_name}"

            # Step 3: Start Replicate training
            job.state = TrainingState.PROVISIONING
            with tempfile.NamedTemporaryFile(suffix=".zip") as training_zip:
                training_zip.write(zip_data)
                training_zip.flush()
                training_zip.seek(0)
                training = replicate.trainings.create(
                    destination=destination,
                    version=self.TRAINER_VERSION,
                    input={
                        "input_images": training_zip,
                        "trigger_word": trigger_word,
                        "steps": job.config.steps,
                        "lora_rank": job.config.lora_rank,
                        "optimizer": "adamw8bit",
                        "batch_size": job.config.batch_size,
                        "resolution": job.config.resolution,
                        "autocaption": True,
                        "autocaption_prefix": f"a photo of {trigger_word}, "
                    }
                )

            self._training_data[job.job_id]["replicate_training_id"] = training.id
            job.provider_job_id = training.id
            job.state = TrainingState.TRAINING
            job.started_at = datetime.now(timezone.utc)

            logger.info(f"Replicate training started: {training.id}")

            # Step 4: Poll for completion
            while True:
                training = replicate.trainings.get(training.id)

                if training.status == "succeeded":
                    break
                elif training.status == "failed":
                    raise RuntimeError(f"Training failed: {training.error}")
                elif training.status == "canceled":
                    raise RuntimeError("Training was canceled")

                await asyncio.sleep(REPLICATE_POLL_INTERVAL_SECONDS)

            # Step 5: Get trained model version
            model_version = training.output.get("version") if training.output else None
            weights_url = training.output.get("weights") if training.output else None

            self._training_data[job.job_id]["replicate_model_version"] = model_version
            self._training_data[job.job_id]["weights_url"] = weights_url
            job.output_path = weights_url

            # Success!
            job.state = TrainingState.COMPLETED
            job.completed_at = datetime.now(timezone.utc)

            logger.info(f"Training completed for {job.companion_id}, model: {destination}")

        except Exception as e:
            logger.error(f"Training failed for {job.companion_id}: {e}")
            job.state = TrainingState.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)

    def _build_training_zip(self, avatar_data: bytes) -> bytes:
        """Create the Replicate training zip in memory."""
        # Create in-memory zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("avatar_01.jpg", avatar_data)

        zip_buffer.seek(0)
        return zip_buffer.read()

    async def get_status(self, job_id: str) -> TrainingStatus:
        """Get job status."""
        job = self._jobs.get(job_id)
        if not job:
            return TrainingStatus(
                job_id=job_id,
                state=TrainingState.FAILED,
                progress=0.0,
                error="Job not found",
            )

        # Calculate progress based on state
        progress = 0.0
        if job.state == TrainingState.PREPARING:
            progress = 0.05
        elif job.state == TrainingState.PROVISIONING:
            progress = 0.1
        elif job.state == TrainingState.TRAINING:
            # Estimate progress based on elapsed time (~15 min total)
            if job.started_at:
                elapsed = (datetime.now(timezone.utc) - job.started_at).total_seconds()
                progress = min(0.15 + (elapsed / 900) * 0.7, 0.85)
            else:
                progress = 0.15
        elif job.state == TrainingState.COMPLETED:
            progress = 1.0

        elapsed = None
        if job.started_at:
            elapsed = (datetime.now(timezone.utc) - job.started_at).total_seconds()

        return TrainingStatus(
            job_id=job_id,
            state=job.state,
            progress=progress,
            error=job.error_message,
            elapsed_seconds=elapsed,
        )

    async def download_weights(self, job_id: str) -> Optional[bytes]:
        """Download trained LoRA weights."""
        job = self._jobs.get(job_id)
        data = self._training_data.get(job_id, {})

        weights_url = data.get("weights_url")
        if not weights_url:
            logger.warning(f"No weights URL for job {job_id}")
            return None

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DOWNLOAD) as client:
                response = await client.get(weights_url)
                if response.status_code != 200:
                    raise DownloadError(
                        f"Failed to download weights: {response.status_code}",
                        provider=self.provider_name
                    )

                # Weights come as tar.gz, extract the .safetensors file
                return self._extract_safetensors(response.content)

        except Exception as e:
            raise DownloadError(
                f"Failed to download weights: {e}",
                provider=self.provider_name,
                details={"job_id": job_id}
            )

    def _extract_safetensors(self, tar_data: bytes) -> Optional[bytes]:
        """Extract .safetensors file from tar.gz archive."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tar_path = os.path.join(tmpdir, "weights.tar.gz")
                with open(tar_path, "wb") as f:
                    f.write(tar_data)

                with tarfile.open(tar_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.endswith(".safetensors"):
                            tar.extract(member, tmpdir)
                            extracted_path = os.path.join(tmpdir, member.name)
                            with open(extracted_path, "rb") as f:
                                return f.read()

            logger.warning("No .safetensors file found in weights archive")
            return None
        except Exception as e:
            logger.error(f"Failed to extract safetensors: {e}")
            return None

    async def cancel(self, job_id: str) -> bool:
        """Cancel Replicate training."""
        job = self._jobs.get(job_id)
        data = self._training_data.get(job_id, {})

        replicate_id = data.get("replicate_training_id")
        if not replicate_id:
            return False

        try:
            import replicate
            replicate.trainings.cancel(replicate_id)

            job.state = TrainingState.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
            logger.info(f"Cancelled Replicate training: {replicate_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel Replicate training: {e}")
            return False

    async def cleanup(self, job_id: str) -> None:
        """
        No cleanup needed for serverless provider.

        Could optionally delete the model from Replicate if not needed.
        """
        pass

    # =========================================================================
    # Image Generation Methods
    # =========================================================================

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities for factory routing decisions."""
        return self.CAPABILITIES

    def get_trained_model_name(self, job_id: str) -> Optional[str]:
        """
        Get the Replicate model name for a completed training job.

        Returns model in format: username/model-name (e.g., kestrel/lora-abc12345)
        """
        job = self._jobs.get(job_id)
        if not job:
            return None
        return f"{self.replicate_username}/lora-{job.companion_id[:8]}"

    def _resolve_lora_url(self, lora_path: str) -> Optional[str]:
        """
        Resolve a LoRA path to a URL format that Replicate accepts.

        Replicate's hf_lora parameter accepts:
        1. HuggingFace paths: "username/model-name" or full HF URLs
        2. Direct URLs ending in .safetensors
        3. Replicate delivery URLs (.tar files from training)
        4. CivitAI download URLs

        LIMITATION: Plain IPFS CIDs (Qm...) are NOT supported because Replicate
        validates that URLs end in .safetensors. IPFS gateway URLs like
        https://gateway.lighthouse.storage/ipfs/Qm... don't work.

        SOLUTION: Upload LoRA to HuggingFace and use HF path instead.

        Args:
            lora_path: Path/URL to LoRA weights. Can be:
                - HuggingFace path: "username/model"
                - HuggingFace URL: "https://huggingface.co/.../file.safetensors"
                - Replicate URL: "https://replicate.delivery/.../trained_model.tar"
                - CivitAI URL: "https://civitai.com/api/download/..."
                - Direct .safetensors URL: "https://example.com/model.safetensors"

        Returns:
            URL string for hf_lora parameter, or None if format not supported
        """
        if not lora_path:
            return None

        # HuggingFace path (no URL scheme)
        if "/" in lora_path and not lora_path.startswith("http"):
            # Looks like a HF path: username/model-name
            return lora_path

        # Already a URL
        if lora_path.startswith("http"):
            # Check if it's a supported format
            if any(domain in lora_path for domain in [
                "huggingface.co",
                "replicate.delivery",
                "civitai.com",
            ]):
                return lora_path

            # Direct URL - must end in .safetensors
            if lora_path.endswith(".safetensors"):
                return lora_path

            # IPFS gateway URL - NOT SUPPORTED by Replicate
            if "ipfs" in lora_path.lower() or "lighthouse" in lora_path.lower():
                logger.warning(
                    f"IPFS URLs are not supported by Replicate's hf_lora parameter. "
                    f"URL: {lora_path}. "
                    f"Replicate requires URLs ending in .safetensors. "
                    f"SOLUTION: Upload LoRA to HuggingFace and use HF path instead."
                )
                return None

            logger.warning(f"URL format may not be supported by Replicate: {lora_path}")
            return lora_path

        # Plain IPFS CID (Qm... or bafy...)
        if lora_path.startswith("Qm") or lora_path.startswith("bafy"):
            logger.warning(
                f"Plain IPFS CIDs are not supported by Replicate. "
                f"CID: {lora_path}. "
                f"SOLUTION: Upload LoRA to HuggingFace and use HF path instead."
            )
            return None

        # Unknown format - try anyway
        logger.warning(f"Unknown LoRA path format: {lora_path}")
        return lora_path

    async def generate_image(
        self,
        config: GenerationConfig,
        session=None,  # Ignored for serverless - kept for interface compatibility
        lora_ipfs_cid: Optional[str] = None,  # DEPRECATED: Use config.lora_path instead
        ipfs_gateway: Optional[str] = None,  # DEPRECATED
        flux_version: Optional[str] = None,  # Not used - Replicate manages container
    ) -> GenerationResult:
        """
        Generate image using trained LoRA on Replicate.

        NOTE: Output is censored by Replicate's content safety filters.
        For providers with different policy needs, download weights and use
        another configured backend.

        IMPORTANT: Replicate does NOT support IPFS URLs for LoRA weights.
        Use HuggingFace paths/URLs instead. See _resolve_lora_url() for details.

        Args:
            config: GenerationConfig with prompt, lora_path, trigger_word, etc.
                   config.lora_path should be a HuggingFace path like "username/model"
                   or a direct URL ending in .safetensors
            session: Ignored for serverless provider (kept for interface compatibility)
            lora_ipfs_cid: DEPRECATED - IPFS URLs are not supported by Replicate.
                          Use config.lora_path with a HuggingFace path instead.
            ipfs_gateway: DEPRECATED - Not used.

        Returns:
            GenerationResult with image URLs or data URIs
        """
        if not self.is_available():
            return GenerationResult(
                job_id=str(uuid.uuid4()),
                state=GenerationState.FAILED,
                error="REPLICATE_API_TOKEN not set",
            )

        try:
            import replicate
        except ImportError:
            return GenerationResult(
                job_id=str(uuid.uuid4()),
                state=GenerationState.FAILED,
                error="replicate package not installed",
            )

        # Extract parameters from config
        prompt = config.prompt
        trigger_word = config.trigger_word
        num_outputs = config.num_outputs
        width = config.width
        height = config.height
        model = "schnell"  # Default model for Replicate

        gen_job_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        try:
            # Determine aspect ratio from dimensions
            if width == height:
                aspect_ratio = "1:1"
            elif width > height:
                aspect_ratio = "16:9" if width / height > 1.5 else "4:3"
            else:
                aspect_ratio = "9:16" if height / width > 1.5 else "3:4"

            # Resolve LoRA path to a URL that Replicate accepts
            # Replicate supports: HuggingFace paths, direct .safetensors URLs,
            # Replicate delivery URLs, CivitAI URLs
            # NOT supported: IPFS URLs, plain CIDs
            lora_url = self._resolve_lora_url(config.lora_path)
            if config.lora_path and not lora_url:
                # LoRA was requested but couldn't be resolved
                return GenerationResult(
                    job_id=gen_job_id,
                    state=GenerationState.FAILED,
                    error=(
                        f"LoRA path format not supported by Replicate: {config.lora_path}. "
                        f"Replicate requires HuggingFace paths (username/model), "
                        f"direct URLs ending in .safetensors, or Replicate delivery URLs. "
                        f"IPFS URLs are NOT supported. "
                        f"Workaround: Upload LoRA to HuggingFace."
                    ),
                    elapsed_seconds=0,
                )

            if lora_url:
                logger.info(f"Using LoRA URL for Replicate: {lora_url}")

            # Run generation with lucataco/flux-schnell-lora model
            base_model = self.INFERENCE_MODELS.get(model, self.INFERENCE_MODELS["schnell"])

            if lora_url:
                # Use base model with external LoRA
                # Supported: HuggingFace paths, .safetensors URLs, Replicate delivery URLs
                logger.info(f"Running Replicate with LoRA: model={base_model}, lora={lora_url}")
                output = replicate.run(
                    base_model,
                    input={
                        "prompt": prompt,
                        "num_outputs": num_outputs,
                        "aspect_ratio": aspect_ratio,
                        "output_format": "jpg",
                        "output_quality": 80,
                        "hf_lora": lora_url,
                        "lora_scale": 0.8,
                    }
                )
            else:
                # Use base model without LoRA
                logger.info(f"Running Replicate without LoRA: model={base_model}")
                output = replicate.run(
                    base_model,
                    input={
                        "prompt": prompt,
                        "num_outputs": num_outputs,
                        "aspect_ratio": aspect_ratio,
                        "output_format": "jpg",
                        "output_quality": 80,
                    }
                )

            # Process output - convert to URLs
            image_urls: List[str] = []
            if output:
                for item in output:
                    if hasattr(item, 'url'):
                        image_urls.append(item.url)
                    elif isinstance(item, str):
                        image_urls.append(item)
                    else:
                        image_urls.append(str(item))

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

            if image_urls:
                logger.info(f"Generated {len(image_urls)} image(s) via Replicate in {elapsed:.1f}s")
                return GenerationResult(
                    job_id=gen_job_id,
                    state=GenerationState.COMPLETED,
                    images=image_urls,
                    elapsed_seconds=elapsed,
                )
            else:
                return GenerationResult(
                    job_id=gen_job_id,
                    state=GenerationState.FAILED,
                    error="No images returned from Replicate",
                    elapsed_seconds=elapsed,
                )

        except Exception as e:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.error(f"Replicate generation failed: {e}")
            return GenerationResult(
                job_id=gen_job_id,
                state=GenerationState.FAILED,
                error=str(e),
                elapsed_seconds=elapsed,
            )

    async def generate_image_with_config(
        self,
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Generate image using GenerationConfig.

        This is a compatibility alias - calls generate_image() with config.
        The LoRA path should be set in config.lora_path as a HuggingFace path
        or direct .safetensors URL.

        IMPORTANT: IPFS URLs are NOT supported by Replicate.
        Use HuggingFace paths like "username/model-name" instead.

        Args:
            config: GenerationConfig with all parameters including lora_path

        Returns:
            GenerationResult with image URLs
        """
        return await self.generate_image(config=config)
