"""
Vertex AI adapter for TrainingProvider protocol.

Wraps VertexAIManager to provide a unified training interface.
Vertex AI is serverless - jobs run to completion without session management.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

import asyncio
import base64

from ..types import (
    TrainingJob,
    TrainingStatus,
    TrainingConfig,
    TrainingState,
    ProviderType,
    GenerationConfig,
    GenerationResult,
    GenerationState,
)
from ..protocol import (
    TrainingProviderError,
    ProviderNotAvailableError,
    TrainingSubmissionError,
    DownloadError,
    GenerationError,
)

logger = logging.getLogger(__name__)


class VertexAITrainingAdapter:
    """
    Adapts VertexAIManager to TrainingProvider protocol.

    Vertex AI is serverless - jobs run to completion without session management.
    This is the most reliable provider as jobs don't suffer from TTL issues.
    """

    def __init__(self):
        self._manager = None
        self._jobs: Dict[str, TrainingJob] = {}

    @property
    def provider_name(self) -> str:
        return "vertex_ai"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.SERVERLESS

    def is_available(self) -> bool:
        """Check if GCP credentials are available."""
        return bool(os.getenv("GCP_PROJECT_ID"))

    def _get_manager(self):
        """Lazy-load the VertexAIManager."""
        if self._manager is None:
            try:
                from kestrel_sovereign.features.vertex_ai.vertex_ai_manager import VertexAIManager
                self._manager = VertexAIManager()
            except ImportError as e:
                raise ProviderNotAvailableError(
                    f"VertexAIManager not available: {e}",
                    provider=self.provider_name
                )
        return self._manager

    async def start_training(
        self,
        companion_id: str,
        avatar_data: bytes,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingJob:
        """Submit training job to Vertex AI."""
        if not self.is_available():
            raise ProviderNotAvailableError(
                "GCP_PROJECT_ID not set",
                provider=self.provider_name
            )

        config = config or TrainingConfig()
        manager = self._get_manager()

        # Generate trigger word if not provided
        trigger_word = config.trigger_word or f"TOK{companion_id[:8]}"

        try:
            # Submit to Vertex AI
            vertex_job = await manager.submit_training_job(
                companion_id=companion_id,
                avatar_data=avatar_data,
                trigger_word=trigger_word,
                steps=config.steps,
                lora_rank=config.lora_rank,
            )

            # Create unified TrainingJob
            job = TrainingJob(
                job_id=str(uuid.uuid4()),
                companion_id=companion_id,
                provider=self.provider_name,
                state=TrainingState.PENDING,
                trigger_word=trigger_word,
                created_at=datetime.now(timezone.utc),
                config=config,
                provider_job_id=vertex_job.job_id,
                output_path=vertex_job.gcs_output_path,
            )

            self._jobs[job.job_id] = job
            logger.info(f"Started Vertex AI training job: {job.job_id} (vertex: {vertex_job.job_id})")
            return job

        except Exception as e:
            raise TrainingSubmissionError(
                f"Failed to submit Vertex AI job: {e}",
                provider=self.provider_name,
                details={"companion_id": companion_id}
            )

    async def get_status(self, job_id: str) -> TrainingStatus:
        """Get job status from Vertex AI."""
        job = self._jobs.get(job_id)
        if not job:
            return TrainingStatus(
                job_id=job_id,
                state=TrainingState.FAILED,
                progress=0.0,
                error="Job not found",
            )

        try:
            manager = self._get_manager()
            status = await manager.get_job_status(job.provider_job_id)

            # Map Vertex state to unified state
            state = TrainingState.from_vertex_state(status["state"])

            # Update job state
            job.state = state
            if state == TrainingState.TRAINING and not job.started_at:
                job.started_at = datetime.now(timezone.utc)
            if state.is_terminal():
                job.completed_at = datetime.now(timezone.utc)
            if status.get("error"):
                job.error_message = status["error"]

            # Calculate elapsed time
            elapsed = None
            if job.started_at:
                elapsed = (datetime.now(timezone.utc) - job.started_at).total_seconds()

            return TrainingStatus(
                job_id=job_id,
                state=state,
                progress=status.get("progress", 0.0),
                error=status.get("error"),
                elapsed_seconds=elapsed,
                provider_details=status,
            )

        except Exception as e:
            logger.error(f"Failed to get Vertex AI job status: {e}")
            return TrainingStatus(
                job_id=job_id,
                state=job.state,
                progress=0.0,
                error=str(e),
            )

    async def download_weights(self, job_id: str) -> Optional[bytes]:
        """Download LoRA from Vertex AI/GCS."""
        job = self._jobs.get(job_id)
        if not job or not job.provider_job_id:
            logger.error(f"Job {job_id} not found")
            return None

        try:
            manager = self._get_manager()
            weights = await manager.download_lora(job.provider_job_id)

            if weights:
                logger.info(f"Downloaded LoRA weights: {len(weights)} bytes")
            else:
                logger.warning(f"No LoRA weights found for job {job_id}")

            return weights

        except Exception as e:
            raise DownloadError(
                f"Failed to download LoRA: {e}",
                provider=self.provider_name,
                details={"job_id": job_id}
            )

    async def cancel(self, job_id: str) -> bool:
        """Cancel Vertex AI job."""
        job = self._jobs.get(job_id)
        if not job or not job.provider_job_id:
            return False

        try:
            manager = self._get_manager()
            result = await manager.cancel_job(job.provider_job_id)

            if result:
                job.state = TrainingState.CANCELLED
                job.completed_at = datetime.now(timezone.utc)
                logger.info(f"Cancelled Vertex AI job: {job_id}")

            return result

        except Exception as e:
            logger.error(f"Failed to cancel Vertex AI job: {e}")
            return False

    async def cleanup(self, job_id: str) -> None:
        """
        No cleanup needed for serverless provider.

        Vertex AI jobs are automatically cleaned up after completion.
        GCS outputs may be retained based on bucket lifecycle policy.
        """
        pass

    # Container images for different FLUX versions (mirrors VertexAIManager)
    CONTAINER_IMAGES = {
        "flux2": "gcr.io/YOUR_PROJECT_ID/kestrel-lora:v8",
        "flux1": "gcr.io/YOUR_PROJECT_ID/kestrel-lora-flux1:v1",
    }

    async def generate_image(
        self,
        config: GenerationConfig,
        session=None,  # Ignored for serverless - kept for interface compatibility
        lora_ipfs_cid: Optional[str] = None,
        ipfs_gateway: str = "https://gateway.lighthouse.storage/ipfs",
        flux_version: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate images using Vertex AI Custom Job.

        Unlike RunPod (persistent pod), Vertex AI is serverless so we:
        1. Submit a Custom Job with --generate-mode
        2. Poll until complete
        3. Download images from GCS
        4. Return as base64

        Args:
            config: GenerationConfig with prompt, lora_path (GCS URI or empty if using IPFS), etc.
            session: Ignored (interface compatibility with RunPod)
            lora_ipfs_cid: Optional IPFS CID for LoRA model (from Lighthouse).
                          If provided, container will fetch from IPFS gateway.
            ipfs_gateway: IPFS gateway URL (default: Lighthouse gateway).
                         Full URL will be: {gateway}/{cid}

        Returns:
            GenerationResult with base64 images

        Expected timing (A100 80GB with int8-quanto):
            First job (no GCS cache):
                - Job submission: ~10s
                - Container startup: ~30s
                - Model quantization: ~10-15 min (uploaded to GCS after)
                - LoRA loading: ~10s
                - Image generation: ~30-60s per image
                - Total: ~12-18 min

            Subsequent jobs (GCS cache available):
                - Job submission: ~10s
                - Container startup: ~30s
                - GCS cache download: ~2-3 min
                - Model loading: ~30s
                - LoRA loading: ~10s
                - Image generation: ~30-60s per image
                - Total: ~4-5 min
        """
        start_time = datetime.now(timezone.utc)

        try:
            manager = self._get_manager()

            # Validate lora source: need either GCS path or IPFS CID
            if not lora_ipfs_cid and not config.lora_path.startswith("gs://"):
                raise GenerationError(
                    f"Vertex AI requires GCS path or IPFS CID for LoRA, got: {config.lora_path}",
                    provider=self.provider_name,
                )

            # Generate output path
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            job_uuid = str(uuid.uuid4())[:8]
            output_gcs = f"gs://{manager.gcs_bucket}/generation/{job_uuid}/{timestamp}"

            # Select container image based on FLUX version
            # flux1 = FLUX.1-dev with uncensored LoRA stacking support
            # flux2 = FLUX.2-dev with standard content filtering (default)
            image_uri = None
            if flux_version and flux_version in self.CONTAINER_IMAGES:
                image_uri = self.CONTAINER_IMAGES[flux_version]
                logger.info(f"Using FLUX version '{flux_version}' container: {image_uri}")
            else:
                logger.info(f"Using default FLUX.2-dev container (flux_version={flux_version})")

            # Submit generation job
            # IPFS CID takes priority - don't pass GCS path if IPFS CID is provided
            logger.info(f"Submitting Vertex AI generation job: {config.prompt[:50]}...")
            job_info = await manager.submit_generation_job(
                lora_gcs_path=None if lora_ipfs_cid else (config.lora_path if config.lora_path.startswith("gs://") else None),
                lora_ipfs_cid=lora_ipfs_cid,
                ipfs_gateway=ipfs_gateway,
                prompt=config.prompt,
                trigger_word=config.trigger_word,
                output_gcs_prefix=output_gcs,
                num_outputs=config.num_outputs,
                width=config.width,
                height=config.height,
                image_uri=image_uri,
            )

            job_id = job_info["job_id"]
            logger.info(f"Generation job submitted: {job_id}")

            # Poll for completion (max 30 minutes)
            max_wait = 30 * 60
            poll_interval = 15
            elapsed = 0

            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                status = await manager.get_job_status(job_id)
                state = status.get("state", "pending")

                logger.info(f"[{elapsed}s] Generation job {job_id}: {state}")

                if state == "completed":
                    break
                elif state == "failed":
                    error = status.get("error", "Unknown error")
                    return GenerationResult(
                        job_id=job_id,
                        state=GenerationState.FAILED,
                        error=error,
                    )
            else:
                return GenerationResult(
                    job_id=job_id,
                    state=GenerationState.FAILED,
                    error=f"Generation timed out after {max_wait}s",
                )

            # Download images from GCS
            logger.info(f"Downloading images from {output_gcs}")
            images = []

            for i in range(config.num_outputs):
                image_gcs = f"{output_gcs}/image_{i}.png"
                try:
                    image_bytes = await manager._download_from_gcs(image_gcs)
                    b64 = base64.b64encode(image_bytes).decode()
                    images.append(f"data:image/png;base64,{b64}")
                    logger.info(f"Downloaded image {i}: {len(image_bytes)} bytes")
                except Exception as e:
                    logger.warning(f"Failed to download image {i}: {e}")

            if not images:
                return GenerationResult(
                    job_id=job_id,
                    state=GenerationState.FAILED,
                    error="No images downloaded from GCS",
                )

            total_elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"✅ Generation completed in {total_elapsed:.1f}s: {len(images)} images")

            return GenerationResult(
                job_id=job_id,
                state=GenerationState.COMPLETED,
                images=images,
                elapsed_seconds=total_elapsed,
            )

        except GenerationError:
            raise
        except Exception as e:
            logger.error(f"Vertex AI generation failed: {e}")
            raise GenerationError(
                f"Generation failed: {e}",
                provider=self.provider_name,
            )

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID (for internal use)."""
        return self._jobs.get(job_id)
