"""
Vast.ai Training Adapter.

Wraps VastAIManager (session-based) to implement the TrainingProvider protocol.
This is a session-based provider that requires instance lifecycle management.

Uses HTTP API endpoints from the shared SimpleTuner Docker image
(same as RunPod/Vertex AI) for training and generation.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx

from ..protocol import (
    TrainingProvider,
    TrainingProviderError,
    ProviderNotAvailableError,
    TrainingSubmissionError,
    TrainingStatusError,
    DownloadError,
    GenerationError,
)
from ..types import (
    GenerationConfig,
    GenerationResult,
    GenerationState,
    ProviderType,
    TrainingConfig,
    TrainingJob,
    TrainingState,
    TrainingStatus,
)
from kestrel_sovereign.kestrel_config.defaults import get_lighthouse_gateway_url

logger = logging.getLogger(__name__)


class VastAITrainingAdapter:
    """
    Adapter wrapping VastAIManager for TrainingProvider protocol.

    This is a SESSION-BASED provider:
    - Requires renting an instance before training
    - Instances are billed hourly from a marketplace
    - Training jobs run via HTTP API to SimpleTuner container
    - Must track both session (instance_id) and job_id
    - Supports image generation with trained LoRAs
    """

    provider_name = "vastai"
    provider_type = ProviderType.SESSION_BASED

    def __init__(self, manager=None):
        """
        Initialize with optional pre-configured manager.

        Args:
            manager: VastAIManager instance (lazy loaded if not provided)
        """
        self._manager = manager
        self._active_jobs: dict[str, dict] = {}  # job_id -> {session, companion_id, ...}

    def _get_manager(self):
        """Lazy load the Vast.ai manager."""
        if self._manager is None:
            try:
                from kestrel_sovereign.features.vastai.manager import VastAIManager
                self._manager = VastAIManager()
            except ImportError as e:
                raise ProviderNotAvailableError(
                    f"Vast.ai manager not available: {e}"
                )
            except Exception as e:
                raise ProviderNotAvailableError(
                    f"Failed to initialize Vast.ai manager: {e}"
                )
        return self._manager

    def is_available(self) -> bool:
        """Check if Vast.ai is available (API key configured)."""
        try:
            manager = self._get_manager()
            # VastAI requires VASTAI_API_KEY
            return manager is not None and manager.api_key is not None
        except (ProviderNotAvailableError, ImportError):
            return False
        except Exception:
            logger.debug("Vast.ai availability check failed", exc_info=True)
            return False

    async def start_training(
        self,
        companion_id: str,
        avatar_data: bytes,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingJob:
        """
        Start a LoRA training job on Vast.ai.

        This is a multi-step process for session-based providers:
        1. Rent an instance (if not already running)
        2. Wait for SimpleTuner API to be ready
        3. Submit training job via HTTP API

        Args:
            companion_id: Companion being trained
            avatar_data: Training image bytes
            config: Training configuration

        Returns:
            TrainingJob with session tracking info
        """
        config = config or TrainingConfig()
        job_id = str(uuid.uuid4())
        trigger_word = config.trigger_word or f"TOK{companion_id[:8]}"
        now = datetime.now(timezone.utc)

        try:
            manager = self._get_manager()

            # Step 1: Start a session (rent an instance)
            logger.info(f"Starting Vast.ai training instance for companion {companion_id}")
            session_result = await manager.start_session(
                task_profile=config.profile,
                ttl_seconds=config.ttl_seconds,
                metadata={"companion_id": companion_id, "job_id": job_id},
            )

            session = manager._session
            if session is None:
                raise TrainingSubmissionError("Failed to get active session")

            if not session.backend_base_url:
                raise TrainingSubmissionError(
                    f"Training instance {session.instance_id} started but has no backend URL"
                )

            logger.info(f"Training instance ready: {session.instance_id}, URL: {session.backend_base_url}")

            # Step 2: Submit training job in background
            # DON'T wait for model ready here - that blocks the HTTP response
            # The background task will handle waiting and submission

            # Track this job with PENDING state - training not yet started
            self._active_jobs[job_id] = {
                "session": session,
                "companion_id": companion_id,
                "trigger_word": trigger_word,
                "started_at": now,
                "config": config,
                "avatar_data": avatar_data,  # Store for background submission
                "training_job_id": None,  # Will be set when /train is called
                "state": "pending",  # Model loading, not yet submitted
            }

            # Launch background task to wait for model and submit training
            asyncio.create_task(
                self._submit_training_when_ready(
                    job_id=job_id,
                    manager=manager,
                    session=session,
                    avatar_data=avatar_data,
                    companion_id=companion_id,
                    trigger_word=trigger_word,
                    config=config,
                )
            )

            return TrainingJob(
                job_id=job_id,
                companion_id=companion_id,
                provider=self.provider_name,
                state=TrainingState.PENDING,  # PENDING until model ready
                trigger_word=trigger_word,
                created_at=now,
                started_at=None,  # Not started yet
                config=config,
                provider_job_id=None,  # Will be set when submitted
                provider_session_id=str(session.instance_id),
            )

        except TrainingProviderError:
            raise
        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"Vast.ai training connection error: {e}")
            raise TrainingSubmissionError(f"Failed to start Vast.ai training: {e}")
        except Exception as e:
            logger.error(f"Vast.ai training submission failed: {e}", exc_info=True)
            raise TrainingSubmissionError(f"Failed to start Vast.ai training: {e}")

    async def _submit_training_when_ready(
        self,
        job_id: str,
        manager,
        session,
        avatar_data: bytes,
        companion_id: str,
        trigger_word: str,
        config: TrainingConfig,
    ) -> None:
        """
        Background task: Wait for model to load, then submit training.

        This runs asynchronously so the HTTP endpoint can return immediately.
        Updates job state from PENDING -> TRAINING when submission succeeds.
        """
        try:
            logger.info(f"[{job_id}] Background: waiting for model ready...")

            # Wait for FLUX model to load and submit training
            training_job_id = await manager.submit_training_job_http(
                session=session,
                avatar_data=avatar_data,
                companion_id=companion_id,
                trigger_word=trigger_word,
                steps=config.steps,
                lora_rank=config.lora_rank,
                callback_url=config.callback_url,
                wait_for_ready=True,  # This does the waiting
            )

            # Update job with training_job_id
            if job_id in self._active_jobs:
                self._active_jobs[job_id]["training_job_id"] = training_job_id
                self._active_jobs[job_id]["state"] = "training"
                self._active_jobs[job_id]["started_at"] = datetime.now(timezone.utc)
                logger.info(f"[{job_id}] Training submitted: {training_job_id}")

        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"[{job_id}] Background training network error: {e}")
            if job_id in self._active_jobs:
                self._active_jobs[job_id]["state"] = "failed"
                self._active_jobs[job_id]["error"] = str(e)
        except Exception as e:
            logger.error(f"[{job_id}] Background training submission failed: {e}", exc_info=True)
            if job_id in self._active_jobs:
                self._active_jobs[job_id]["state"] = "failed"
                self._active_jobs[job_id]["error"] = str(e)

    async def get_status(self, job_id: str) -> TrainingStatus:
        """
        Get status of a training job.

        Args:
            job_id: Job ID to check

        Returns:
            TrainingStatus with current progress
        """
        try:
            if job_id not in self._active_jobs:
                raise TrainingStatusError(f"Unknown job: {job_id}")

            job_info = self._active_jobs[job_id]
            session = job_info["session"]
            training_job_id = job_info.get("training_job_id")
            manager = self._get_manager()

            # Check if still waiting for model to load (background submission)
            if training_job_id is None:
                job_state = job_info.get("state", "pending")
                if job_state == "failed":
                    return TrainingStatus(
                        job_id=job_id,
                        state=TrainingState.FAILED,
                        progress=0.0,
                        error=job_info.get("error", "Background submission failed"),
                    )
                # Still waiting for model to load
                return TrainingStatus(
                    job_id=job_id,
                    state=TrainingState.PREPARING,
                    progress=0.0,
                    message="Waiting for FLUX model to load (may take 5-10 min)...",
                )

            # Poll the training status via HTTP API
            status_result = await manager.poll_training_status_http(
                session=session,
                job_id=training_job_id,
            )

            # Map SimpleTuner status to unified status
            api_status = status_result.get("status", "unknown").lower()
            progress = status_result.get("progress", 0.0)
            error = status_result.get("error")

            # Map status strings to TrainingState
            if api_status == "completed":
                state = TrainingState.COMPLETED
                # Store output path for download
                job_info["lora_path"] = status_result.get("lora_path")
            elif api_status == "failed":
                state = TrainingState.FAILED
            elif api_status in ("running", "training"):
                state = TrainingState.TRAINING
            elif api_status in ("queued", "pending"):
                state = TrainingState.PENDING
            elif api_status in ("preparing", "finalizing"):
                state = TrainingState.PREPARING
            else:
                state = TrainingState.PENDING

            elapsed = None
            if job_info.get("started_at"):
                elapsed = (datetime.now(timezone.utc) - job_info["started_at"]).total_seconds()

            return TrainingStatus(
                job_id=job_id,
                state=state,
                progress=progress,
                message=status_result.get("message"),
                error=error,
                elapsed_seconds=elapsed,
                provider_details=status_result,
            )

        except TrainingProviderError:
            raise
        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"Vast.ai training status network error: {e}")
            raise TrainingStatusError(f"Status check failed: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"Vast.ai training status parse error: {e}")
            raise TrainingStatusError(f"Status check failed: {e}")
        except Exception as e:
            logger.error(f"Failed to get Vast.ai training status: {e}", exc_info=True)
            raise TrainingStatusError(f"Status check failed: {e}")

    async def download_weights(self, job_id: str) -> Optional[bytes]:
        """
        Download trained LoRA weights.

        Args:
            job_id: Job ID to download weights for

        Returns:
            LoRA weights as bytes, or None if not ready
        """
        try:
            if job_id not in self._active_jobs:
                raise DownloadError(f"Unknown job: {job_id}")

            job_info = self._active_jobs[job_id]
            session = job_info["session"]
            training_job_id = job_info["training_job_id"]
            manager = self._get_manager()

            # Download via HTTP API
            logger.info(f"Downloading LoRA weights for job {training_job_id}")
            lora_bytes = await manager.download_lora_http(
                session=session,
                job_id=training_job_id,
            )

            logger.info(f"Downloaded {len(lora_bytes)} bytes of LoRA weights")
            return lora_bytes

        except TrainingProviderError:
            raise
        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"Vast.ai LoRA download network error: {e}")
            raise DownloadError(f"Download failed: {e}")
        except Exception as e:
            logger.error(f"Failed to download Vast.ai LoRA weights: {e}", exc_info=True)
            raise DownloadError(f"Download failed: {e}")

    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a running training job.

        For session-based providers, this terminates the session.

        Args:
            job_id: Job to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            if job_id not in self._active_jobs:
                return False

            job_info = self._active_jobs[job_id]
            session = job_info["session"]
            manager = self._get_manager()

            # Terminate the session (which cancels the job)
            await manager.terminate_session(session)

            # Clean up tracking
            del self._active_jobs[job_id]

            logger.info(f"Cancelled Vast.ai training job {job_id}")
            return True

        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to cancel Vast.ai training: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel Vast.ai training: {e}", exc_info=True)
            return False

    async def cleanup(self, job_id: str) -> None:
        """
        Clean up resources for a completed job.

        For session-based providers, this terminates the instance to stop
        billing. Vast.ai bills by the hour so early termination saves money.

        Args:
            job_id: Job to clean up
        """
        try:
            if job_id not in self._active_jobs:
                return

            job_info = self._active_jobs[job_id]
            session = job_info["session"]
            manager = self._get_manager()

            # Terminate the session to stop billing
            logger.info(f"Terminating Vast.ai instance {session.instance_id} to stop billing")
            await manager.terminate_session(session)

            # Clean up tracking
            del self._active_jobs[job_id]

        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to cleanup Vast.ai session: {e}")
        except Exception as e:
            logger.warning(f"Failed to cleanup Vast.ai session: {e}", exc_info=True)

    async def generate_image(
        self,
        config: GenerationConfig,
        session=None,
        lora_ipfs_cid: Optional[str] = None,
        ipfs_gateway: Optional[str] = None,
        flux_version: Optional[str] = None,  # Reserved for future container selection
    ) -> GenerationResult:
        """
        Generate images using FLUX.2-dev with a trained LoRA on Vast.ai.

        Uses async generation to avoid timeouts:
        1. Call /generate/async to start generation (returns immediately)
        2. Poll /generate/status/{job_id} until completed (~5-6 min with int8-quanto)
        3. Return base64 images

        Args:
            config: Generation configuration (prompt, lora_path, etc.)
            session: Optional existing Vast.ai session. If None, will start a new instance.
            lora_ipfs_cid: Optional IPFS CID for LoRA model (from Lighthouse).
                          If provided, container will fetch from IPFS gateway.
            ipfs_gateway: IPFS gateway URL (default: Lighthouse gateway).
                         Full URL will be: {gateway}/{cid}

        Returns:
            GenerationResult with base64 images or error

        Timing (A100 80GB with int8-quanto):
            - Model load: ~20-60 seconds
            - Generation per image: ~60-120 seconds
        """
        start_time = datetime.now(timezone.utc)

        try:
            manager = self._get_manager()

            # Get or create session
            if session is None:
                logger.info("Getting Vast.ai session for image generation...")
                # Use existing active session if we have one
                for job_info in self._active_jobs.values():
                    if job_info.get("session"):
                        session = job_info["session"]
                        logger.info(f"Reusing existing session: {session.instance_id}")
                        break

                if session is None:
                    # Start a new training instance (which has generation capability)
                    logger.info("Starting new Vast.ai instance for generation...")
                    await manager.start_session(
                        task_profile="training",  # Training profile has generation too
                        ttl_seconds=3600,
                        metadata={"purpose": "generation"},
                    )
                    session = manager._session
                    if session is None:
                        raise GenerationError("Failed to start Vast.ai instance for generation")

            if not session.backend_base_url:
                raise GenerationError(
                    f"Instance {session.instance_id} has no backend URL"
                )

            logger.info(f"Using Vast.ai backend: {session.backend_base_url}")

            # Use canonical gateway URL if not provided
            gateway_url = ipfs_gateway or get_lighthouse_gateway_url()

            # Generate via HTTP API
            result = await manager.generate_image_http(
                session=session,
                prompt=config.prompt,
                lora_path=config.lora_path,
                trigger_word=config.trigger_word,
                num_outputs=config.num_outputs,
                width=config.width,
                height=config.height,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                lora_ipfs_cid=lora_ipfs_cid,
                ipfs_gateway=gateway_url,
            )

            total_elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"Generation completed in {total_elapsed:.1f}s: {len(result['images'])} images")

            return GenerationResult(
                job_id=result.get("job_id", str(uuid.uuid4())),
                state=GenerationState.COMPLETED,
                images=result["images"],
                elapsed_seconds=total_elapsed,
            )

        except GenerationError:
            raise
        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"Vast.ai generation connection error: {e}")
            raise GenerationError(f"Generation failed: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"Vast.ai generation response error: {e}")
            raise GenerationError(f"Generation failed: {e}")
        except Exception as e:
            logger.error(f"Vast.ai generation failed: {e}", exc_info=True)
            raise GenerationError(f"Generation failed: {e}")

    async def generate_image_simple(
        self,
        prompt: str,
        lora_path: str,
        trigger_word: str = "TOK",
        session=None,
        lora_ipfs_cid: Optional[str] = None,
        ipfs_gateway: Optional[str] = None,
    ) -> list[str]:
        """
        Simplified generation interface - returns list of base64 images.

        This is a convenience wrapper around generate_image() for common use cases.

        Args:
            prompt: Generation prompt (trigger word added automatically if missing)
            lora_path: Path to LoRA on the instance (can be empty if using IPFS)
            trigger_word: LoRA trigger word
            session: Optional existing session
            lora_ipfs_cid: Optional IPFS CID for LoRA (from Lighthouse)
            ipfs_gateway: IPFS gateway URL

        Returns:
            List of base64 data URLs (data:image/png;base64,...)

        Raises:
            GenerationError: If generation fails
        """
        config = GenerationConfig(
            prompt=prompt,
            lora_path=lora_path,
            trigger_word=trigger_word,
        )
        result = await self.generate_image(
            config, session,
            lora_ipfs_cid=lora_ipfs_cid,
            ipfs_gateway=ipfs_gateway,
        )
        return result.images
