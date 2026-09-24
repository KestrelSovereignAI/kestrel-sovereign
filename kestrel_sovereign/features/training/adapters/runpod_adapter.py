"""
RunPod Training Adapter.

Wraps RunPodManager (session-based) to implement the TrainingProvider
protocol. This is a session-based provider that supports both persistent
pods (fast resume) and on-demand pods.

Key features:
- Persistent pod support (resume ~10-30s vs create ~2-5min)
- Network volume caching (models cached across sessions)
- SimpleTuner training API integration
- Direct RunPod API access via RUNPOD_API_KEY
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx

from ._session_training_lifecycle import (
    ReleaseIntent,
    SessionLifecycleTimeouts,
    SessionProviderHooks,
    SessionTrainingLifecycle,
    SessionTrainingRecord,
)

from ..protocol import (
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
from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_DEFAULT,
    TRAINING_GENERATION_TIMEOUT,
    TRAINING_POLL_INTERVAL_FAST,
)
from kestrel_sovereign.kestrel_config.defaults import get_lighthouse_gateway_url

logger = logging.getLogger(__name__)


class RunPodTrainingAdapter:
    """
    Adapter wrapping RunPodManager for TrainingProvider protocol.

    This is a SESSION-BASED provider:
    - Uses persistent pods when configured (fastest)
    - Falls back to resuming stopped pods (~10-30s)
    - Creates new pods as last resort (~2-5min)
    - Training jobs run via HTTP API to SimpleTuner container
    - Network volumes provide model caching across sessions

    The RunPodManager handles:
    - Pod lifecycle (create, resume, stop)
    - SSH/HTTP communication with pods
    - Training job submission via /train endpoint
    - Status polling via /status/{job_id} endpoint
    - LoRA download via /download/{job_id} endpoint
    """

    provider_name = "runpod"
    provider_type = ProviderType.SESSION_BASED

    def __init__(
        self,
        manager=None,
        *,
        lifecycle_timeouts: Optional[SessionLifecycleTimeouts] = None,
    ):
        """
        Initialize with optional pre-configured manager.

        Args:
            manager: RunPodManager instance (lazy loaded if not provided)
            lifecycle_timeouts: Teardown bounds (defaults from kestrel_config)
        """
        self._manager = manager
        self._lifecycle = SessionTrainingLifecycle(
            SessionProviderHooks(
                provider_name=self.provider_name,
                display_name="RunPod",
                acquire_session=self._acquire_session,
                session_id=lambda session: session.pod_id,
                validate_session=self._validate_session,
                submit_job=self._submit_job,
                release_session=self._release_session,
                cancel_provider_job=self._cancel_provider_job,
            ),
            lifecycle_timeouts,
        )

    @property
    def _active_jobs(self) -> dict[str, SessionTrainingRecord]:
        """Live custody registry: a job stays here until its pod is released."""
        return self._lifecycle.records

    def _get_manager(self):
        """Lazy load the RunPod manager.

        RunPod support lives in the kestrel-cloud-runpod feature package
        now (extracted in #462). If it's not installed, this adapter
        cleanly reports the provider as unavailable.
        """
        if self._manager is None:
            try:
                from kestrel_cloud_runpod.manager import RunPodManager
                self._manager = RunPodManager()
            except ImportError as e:
                raise ProviderNotAvailableError(
                    f"RunPod manager not available — install kestrel-cloud-runpod "
                    f"to enable RunPod-as-training-provider: {e}"
                )
            except Exception as e:
                raise ProviderNotAvailableError(
                    f"Failed to initialize RunPod manager: {e}"
                )
        return self._manager

    def is_available(self) -> bool:
        """
        Check if RunPod is available.

        Returns True if RUNPOD_API_KEY is set and manager can be initialized.
        """
        try:
            # First check environment variable
            api_key = os.getenv("RUNPOD_API_KEY")
            if not api_key:
                return False

            # Try to get manager (validates configuration)
            manager = self._get_manager()
            return manager is not None
        except (ProviderNotAvailableError, ImportError):
            return False
        except Exception:
            logger.debug("RunPod availability check failed", exc_info=True)
            return False

    async def start_training(
        self,
        companion_id: str,
        avatar_data: bytes,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingJob:
        """
        Start a LoRA training job on RunPod.

        This is a multi-step process for session-based providers:
        1. Start a training pod (resume persistent, resume stopped, or create new)
        2. Wait for model to load (may take 5-10 min on first run)
        3. Submit training job via HTTP API

        Steps 2-3 run in a background submission task owned by the shared
        session lifecycle, so this returns as soon as the pod is up.

        Args:
            companion_id: Companion being trained
            avatar_data: Training image bytes (JPEG/PNG)
            config: Training configuration

        Returns:
            TrainingJob with session tracking info
        """
        config = config or TrainingConfig()
        trigger_word = config.trigger_word or f"TOK{companion_id[:8]}"

        try:
            record = await self._lifecycle.start(
                job_id=str(uuid.uuid4()),
                companion_id=companion_id,
                trigger_word=trigger_word,
                config=config,
                avatar_data=avatar_data,
            )
        except TrainingProviderError:
            raise
        except httpx.HTTPError as e:
            logger.error(f"RunPod training HTTP error: {e}")
            raise TrainingSubmissionError(f"Failed to start RunPod training: {e}")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"RunPod training connection error: {e}")
            raise TrainingSubmissionError(f"Failed to start RunPod training: {e}")
        except Exception as e:
            logger.error(f"RunPod training submission failed: {e}", exc_info=True)
            raise TrainingSubmissionError(f"Failed to start RunPod training: {e}")

        logger.info(
            f"Training pod ready: {record.session.pod_id}, "
            f"URL: {record.session.backend_base_url}"
        )
        return self._lifecycle.training_job(record)

    # -- Provider hooks for the shared session lifecycle -------------------

    async def _acquire_session(
        self, job_id: str, companion_id: str, config: TrainingConfig
    ):
        """Start or resume the training pod for one job."""
        manager = self._get_manager()
        logger.info(f"Starting RunPod training pod for companion {companion_id}")
        session = await manager.start_training_pod(companion_id)
        if session is None:
            raise TrainingSubmissionError(
                "Failed to start RunPod training pod - no GPUs available or all profiles failed"
            )
        return session

    @staticmethod
    def _validate_session(session) -> None:
        if not session.backend_base_url:
            raise TrainingSubmissionError(
                f"Training pod {session.pod_id} started but has no backend URL"
            )

    async def _submit_job(self, record: SessionTrainingRecord, avatar_data: bytes) -> str:
        """Wait for the FLUX model (5-10 min on cold start), then POST /train."""
        return await self._get_manager().submit_training_job(
            session=record.session,
            avatar_data=avatar_data,
            companion_id=record.companion_id,
            callback_url=record.config.callback_url,
            wait_for_model_ready=True,
        )

    async def _cancel_provider_job(
        self, record: SessionTrainingRecord, provider_job_id: str
    ) -> dict:
        return await self._get_manager().cancel_training_job(
            record.session, provider_job_id
        )

    async def _release_session(self, session) -> None:
        """Stop exactly this job's pod.

        ``stop_session()`` takes no session argument: it stops whatever pod
        the manager currently holds. It is only correct when that is this
        job's pod; any other pod is stopped by identity.
        """
        manager = self._get_manager()
        if getattr(manager, "_session", None) is session:
            await manager.stop_session()
        else:
            await manager.terminate_session(session)

    # -- TrainingProvider --------------------------------------------------

    async def get_status(self, job_id: str) -> TrainingStatus:
        """
        Get status of a training job.

        Polls the pod's /status/{job_id} endpoint to get current progress.

        Args:
            job_id: Job ID to check

        Returns:
            TrainingStatus with current progress
        """
        try:
            record = self._lifecycle.get(job_id)
            if record is None:
                raise TrainingStatusError(f"Unknown job: {job_id}")

            local = self._lifecycle.local_status(
                record,
                preparing_message="Waiting for FLUX model to load (may take 5-10 min)...",
            )
            if local is not None:
                return local

            # Poll the training status via manager
            manager = self._get_manager()
            status_result = await manager.poll_training_status(
                session=record.session,
                job_id=record.provider_job_id,
            )

            # Map pod status to unified status
            pod_status = status_result.get("status", "unknown").lower()
            progress = status_result.get("progress", 0.0)
            error = status_result.get("error")

            # Map RunPod training status to unified TrainingState
            if pod_status == "completed":
                state = TrainingState.COMPLETED
            elif pod_status == "failed":
                state = TrainingState.FAILED
            elif pod_status in ("running", "training"):
                state = TrainingState.TRAINING
            elif pod_status in ("pending", "queued"):
                state = TrainingState.PENDING
            elif pod_status in ("preparing", "loading"):
                state = TrainingState.PREPARING
            else:
                # Map from RunPod pod state if no training status
                state = TrainingState.from_runpod_state(pod_status)

            return TrainingStatus(
                job_id=job_id,
                state=state,
                progress=progress,
                message=status_result.get("message"),
                error=error,
                elapsed_seconds=record.elapsed_seconds(),
                provider_details=status_result,
            )

        except TrainingProviderError:
            raise
        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"RunPod training status network error: {e}")
            raise TrainingStatusError(f"Status check failed: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"RunPod training status parse error: {e}")
            raise TrainingStatusError(f"Status check failed: {e}")
        except Exception as e:
            logger.error(f"Failed to get RunPod training status: {e}", exc_info=True)
            raise TrainingStatusError(f"Status check failed: {e}")

    async def download_weights(self, job_id: str) -> Optional[bytes]:
        """
        Download trained LoRA weights.

        Downloads the .safetensors file from the pod's /download/{job_id} endpoint.

        Args:
            job_id: Job ID to download weights for

        Returns:
            LoRA weights as bytes, or None if not ready
        """
        try:
            record = self._lifecycle.get(job_id)
            if record is None:
                raise DownloadError(f"Unknown job: {job_id}")
            training_job_id = record.provider_job_id
            if training_job_id is None:
                raise DownloadError(f"Job {job_id} was never submitted to RunPod")

            manager = self._get_manager()

            # Download via manager
            logger.info(f"Downloading LoRA weights for job {training_job_id}")
            lora_bytes = await manager.download_lora(
                session=record.session,
                job_id=training_job_id,
            )

            logger.info(f"Downloaded {len(lora_bytes)} bytes of LoRA weights")
            return lora_bytes

        except TrainingProviderError:
            raise
        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"RunPod LoRA download network error: {e}")
            raise DownloadError(f"Download failed: {e}")
        except Exception as e:
            logger.error(f"Failed to download RunPod LoRA weights: {e}", exc_info=True)
            raise DownloadError(f"Download failed: {e}")

    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a training job and release its pod.

        Stops the background submission, cancels a published RunPod job, then
        stops the pod. Concurrent and repeated calls share one teardown.

        Args:
            job_id: Job to cancel

        Returns:
            True once no pod or task of the job remains; False if the job is
            unknown or its pod could not be released (custody is retained).
        """
        released = await self._lifecycle.release(job_id, ReleaseIntent.CANCEL)
        if released:
            logger.info(f"Cancelled RunPod training job {job_id}")
        return released

    async def cleanup(self, job_id: str) -> None:
        """
        Clean up resources for a completed job.

        Stops the job's pod (a persistent pod is paused, cost-free while
        paused). A failed release keeps the job tracked so it can be retried.

        IMPORTANT: Always call this after download_weights() to stop billing!

        Args:
            job_id: Job to clean up
        """
        if self._lifecycle.get(job_id) is None:
            return
        if not await self._lifecycle.release(job_id, ReleaseIntent.CLEANUP):
            logger.warning(f"Failed to cleanup RunPod session for job {job_id}; custody retained")

    async def close(self) -> None:
        """Drain background submissions and release every pod this adapter holds."""
        await self._lifecycle.close()

    async def generate_image(
        self,
        config: GenerationConfig,
        session=None,
        lora_ipfs_cid: Optional[str] = None,
        ipfs_gateway: Optional[str] = None,
        flux_version: Optional[str] = None,  # Reserved for future container selection
    ) -> GenerationResult:
        """
        Generate images using FLUX.2-dev with a trained LoRA on RunPod.

        Uses async generation to avoid Cloudflare's 100s timeout:
        1. Call /generate/async to start generation (returns immediately)
        2. Poll /generate/status/{job_id} until completed (~5-6 min with CPU offload)
        3. Return base64 images

        Args:
            config: Generation configuration (prompt, lora_path, etc.)
            session: Optional existing RunPod session. If None, will start/resume a pod.
            lora_ipfs_cid: Optional IPFS CID for LoRA model (from Lighthouse).
                          If provided, container will fetch from IPFS gateway.
            ipfs_gateway: IPFS gateway URL (default: Lighthouse gateway).
                         Full URL will be: {gateway}/{cid}

        Returns:
            GenerationResult with base64 images or error

        Timing (A100 80GB with CPU offload):
            - Model load: ~20 seconds
            - Generation per image: ~330 seconds (~5.5 min)
        """
        start_time = datetime.now(timezone.utc)

        try:
            manager = self._get_manager()

            # Get or create session
            if session is None:
                logger.info("Starting RunPod pod for image generation...")
                # Use existing active session if we have one
                session = self._lifecycle.any_live_session()
                if session is not None:
                    logger.info(f"Reusing existing session: {session.pod_id}")

                if session is None:
                    # Start a new pod
                    session = await manager.start_training_pod("generation")
                    if session is None:
                        raise GenerationError("Failed to start RunPod pod for generation")

            if not session.backend_base_url:
                raise GenerationError(
                    f"Pod {session.pod_id} has no backend URL"
                )

            base_url = session.backend_base_url.rstrip("/")
            logger.info(f"Using RunPod backend: {base_url}")

            # Use canonical gateway URL if not provided
            gateway_url = ipfs_gateway or get_lighthouse_gateway_url()

            # Step 1: Start async generation
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
                form_data = {
                    "prompt": config.prompt,
                    "lora_path": config.lora_path,
                    "trigger_word": config.trigger_word,
                    "num_outputs": str(config.num_outputs),
                    "width": str(config.width),
                    "height": str(config.height),
                    "num_inference_steps": str(config.num_inference_steps),
                    "guidance_scale": str(config.guidance_scale),
                }

                # Add IPFS parameters if CID provided
                if lora_ipfs_cid:
                    form_data["lora_ipfs_cid"] = lora_ipfs_cid
                    form_data["ipfs_gateway"] = gateway_url

                logger.info(f"Starting async generation: {config.prompt[:50]}...")
                response = await client.post(
                    f"{base_url}/generate/async",
                    data=form_data,
                )

                if response.status_code != 200:
                    error_detail = response.text
                    raise GenerationError(
                        f"Failed to start generation: {response.status_code} - {error_detail}"
                    )

                result = response.json()
                gen_job_id = result["job_id"]
                logger.info(f"Generation job started: {gen_job_id}")

            # Step 2: Poll for completion
            # With CPU offload, generation takes ~5.5 min per image
            max_wait = TRAINING_GENERATION_TIMEOUT  # 15 minutes max
            poll_interval = TRAINING_POLL_INTERVAL_FAST  # Poll every 10 seconds
            elapsed = 0

            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
                while elapsed < max_wait:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

                    response = await client.get(
                        f"{base_url}/generate/status/{gen_job_id}"
                    )

                    if response.status_code != 200:
                        logger.warning(f"Status check failed: {response.status_code}")
                        continue

                    status = response.json()
                    pod_status = status.get("status", "unknown")

                    logger.info(f"[{elapsed}s] Generation status: {pod_status}")

                    if pod_status == "completed":
                        images = status.get("images", [])
                        total_elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                        logger.info(f"✅ Generation completed in {total_elapsed:.1f}s: {len(images)} images")
                        return GenerationResult(
                            job_id=gen_job_id,
                            state=GenerationState.COMPLETED,
                            images=images,
                            elapsed_seconds=total_elapsed,
                        )

                    if pod_status == "failed":
                        error = status.get("error", "Unknown error")
                        raise GenerationError(f"Generation failed: {error}")

                    # Map pod status to generation state for progress
                    if pod_status == "loading_model":
                        state = GenerationState.LOADING_MODEL
                    elif pod_status == "loading_lora":
                        state = GenerationState.LOADING_LORA
                    elif pod_status == "generating":
                        state = GenerationState.GENERATING
                    else:
                        state = GenerationState.PENDING

                # Timeout
                raise GenerationError(f"Generation timed out after {max_wait}s")

        except GenerationError:
            raise
        except httpx.HTTPError as e:
            logger.error(f"RunPod generation HTTP error: {e}")
            raise GenerationError(f"Generation failed: {e}")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"RunPod generation connection error: {e}")
            raise GenerationError(f"Generation failed: {e}")
        except Exception as e:
            logger.error(f"RunPod generation failed: {e}", exc_info=True)
            raise GenerationError(f"Generation failed: {e}")

    async def is_training_in_progress(self, session=None) -> Optional[dict]:
        """
        Check if a training job is currently running on the pod.

        Args:
            session: Optional session to check. If None, uses any active session.

        Returns:
            Job info dict if training in progress, None if idle
        """
        try:
            manager = self._get_manager()

            if session is None:
                # Use any active session we're tracking
                session = self._lifecycle.any_live_session()

            if session is None:
                # No active session, so no training in progress (on our pod)
                return None

            return await manager.get_current_job(session)
        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to check training status: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to check training status: {e}", exc_info=True)
            return None

    async def cancel_training(self, job_id: str) -> dict:
        """
        Cancel a training job on the pod while keeping the pod.

        Stops the background submission first, so a submission cannot land
        after this returns. The job becomes FAILED; call cleanup() to release
        the pod. The pod's /cancel may not stop the actual training process;
        for stuck jobs, use clear_training_lock().

        Args:
            job_id: Job ID to cancel

        Returns:
            Cancellation result
        """
        record = await self._lifecycle.stop_submission(job_id, reason="Cancelled")
        if record.session_released:
            return {"status": "cancelled", "message": "Job pod already released"}
        if record.provider_job_id is None:
            return {"status": "cancelled", "message": "Job cancelled before submission"}

        manager = self._get_manager()
        return await manager.cancel_training_job(record.session, record.provider_job_id)

    async def clear_training_lock(self, session=None) -> dict:
        """
        Force-clear the training lock on the pod.

        USE WITH CAUTION: Only use when a job is stuck and unresponsive.
        This clears the lock but does NOT kill any running processes.
        For truly stuck training, you may need to restart the pod.

        Args:
            session: Optional session. If None, uses any active session.

        Returns:
            Result with cleared_job info
        """
        manager = self._get_manager()

        if session is None:
            session = self._lifecycle.any_live_session()

        if session is None:
            return {"cleared_job": None, "message": "No active session"}

        return await manager.clear_current_job(session)

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
            prompt: Generation prompt (trigger word added automatically)
            lora_path: Path to LoRA on the pod (can be empty if using IPFS)
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
