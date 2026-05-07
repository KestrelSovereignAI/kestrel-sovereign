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

    def __init__(self, manager=None):
        """
        Initialize with optional pre-configured manager.

        Args:
            manager: RunPodManager instance (lazy loaded if not provided)
        """
        self._manager = manager
        self._active_jobs: dict[str, dict] = {}  # job_id -> {session, companion_id, ...}

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

        Args:
            companion_id: Companion being trained
            avatar_data: Training image bytes (JPEG/PNG)
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

            # Step 1: Start or resume training pod
            logger.info(f"Starting RunPod training pod for companion {companion_id}")
            session = await manager.start_training_pod(companion_id)

            if session is None:
                raise TrainingSubmissionError(
                    "Failed to start RunPod training pod - no GPUs available or all profiles failed"
                )

            if not session.backend_base_url:
                raise TrainingSubmissionError(
                    f"Training pod {session.pod_id} started but has no backend URL"
                )

            logger.info(f"Training pod ready: {session.pod_id}, URL: {session.backend_base_url}")

            # Step 2: Submit training job in background
            # DON'T wait for model ready here - that blocks the HTTP response for 5-10 min
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
            import asyncio
            asyncio.create_task(
                self._submit_training_when_ready(
                    job_id=job_id,
                    manager=manager,
                    session=session,
                    avatar_data=avatar_data,
                    companion_id=companion_id,
                    callback_url=config.callback_url,
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
                provider_session_id=session.pod_id,  # RunPod pod ID
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

    async def _submit_training_when_ready(
        self,
        job_id: str,
        manager,
        session,
        avatar_data: bytes,
        companion_id: str,
        callback_url: Optional[str] = None,
    ) -> None:
        """
        Background task: Wait for model to load, then submit training.

        This runs asynchronously so the HTTP endpoint can return immediately.
        Updates job state from PENDING -> TRAINING when submission succeeds.
        """
        try:
            logger.info(f"[{job_id}] Background: waiting for model ready...")

            # Wait for FLUX model to load (5-10 min on cold start)
            training_job_id = await manager.submit_training_job(
                session=session,
                avatar_data=avatar_data,
                companion_id=companion_id,
                callback_url=callback_url,
                wait_for_model_ready=True,  # This does the waiting
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

        Polls the pod's /status/{job_id} endpoint to get current progress.

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

            # Poll the training status via manager
            status_result = await manager.poll_training_status(
                session=session,
                job_id=training_job_id,
            )

            # Map pod status to unified status
            pod_status = status_result.get("status", "unknown").lower()
            progress = status_result.get("progress", 0.0)
            error = status_result.get("error")
            output_path = status_result.get("output_path")

            # Map RunPod training status to unified TrainingState
            if pod_status == "completed":
                state = TrainingState.COMPLETED
                # Store output path for download
                job_info["output_path"] = output_path
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
            if job_id not in self._active_jobs:
                raise DownloadError(f"Unknown job: {job_id}")

            job_info = self._active_jobs[job_id]
            session = job_info["session"]
            training_job_id = job_info["training_job_id"]
            manager = self._get_manager()

            # Download via manager
            logger.info(f"Downloading LoRA weights for job {training_job_id}")
            lora_bytes = await manager.download_lora(
                session=session,
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
        Cancel a running training job.

        For RunPod persistent pods, this pauses the pod (can be resumed).
        For on-demand pods, this terminates the pod.

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

            # Stop the session (pauses pod for persistent, terminates for on-demand)
            await manager.stop_session()

            # Clean up tracking
            del self._active_jobs[job_id]

            logger.info(f"Cancelled RunPod training job {job_id}")
            return True

        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to cancel RunPod training: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel RunPod training: {e}", exc_info=True)
            return False

    async def cleanup(self, job_id: str) -> None:
        """
        Clean up resources for a completed job.

        For RunPod persistent pods, this pauses the pod (cost-free while paused).
        For on-demand pods, this terminates the pod.

        IMPORTANT: Always call this after download_weights() to stop billing!

        Args:
            job_id: Job to clean up
        """
        try:
            if job_id not in self._active_jobs:
                return

            job_info = self._active_jobs[job_id]
            session = job_info["session"]
            manager = self._get_manager()

            # Check if this is a persistent pod (configured in profile)
            # Expand env var at runtime to handle dynamic pod ID changes
            profile = session.profile
            resolved_pod_id = manager._expand_single_env_var(profile.persistent_pod_id)
            is_persistent = resolved_pod_id is not None

            if is_persistent:
                logger.info(f"Pausing persistent pod {session.pod_id} after training cleanup")
                await manager.stop_session()
            else:
                # Terminate on-demand pod
                logger.info(f"Terminating on-demand pod {session.pod_id}")
                await manager.terminate_session(session)

            # Clean up tracking
            del self._active_jobs[job_id]

        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to cleanup RunPod session: {e}")
        except Exception as e:
            logger.warning(f"Failed to cleanup RunPod session: {e}", exc_info=True)

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
                for job_info in self._active_jobs.values():
                    if job_info.get("session"):
                        session = job_info["session"]
                        logger.info(f"Reusing existing session: {session.pod_id}")
                        break

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
                for job_info in self._active_jobs.values():
                    if job_info.get("session"):
                        session = job_info["session"]
                        break

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
        Cancel a training job on the pod.

        Note: This marks the job as cancelled but may not stop the actual
        training process. For stuck jobs, use clear_training_lock().

        Args:
            job_id: Job ID to cancel

        Returns:
            Cancellation result
        """
        if job_id not in self._active_jobs:
            raise TrainingStatusError(f"Unknown job: {job_id}")

        job_info = self._active_jobs[job_id]
        session = job_info["session"]
        training_job_id = job_info.get("training_job_id")

        if not training_job_id:
            # Job hasn't been submitted yet, just mark as failed
            job_info["state"] = "failed"
            job_info["error"] = "Cancelled before submission"
            return {"status": "cancelled", "message": "Job cancelled before submission"}

        manager = self._get_manager()
        result = await manager.cancel_training_job(session, training_job_id)

        # Update local tracking
        job_info["state"] = "failed"
        job_info["error"] = "Cancelled"

        return result

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
            for job_info in self._active_jobs.values():
                if job_info.get("session"):
                    session = job_info["session"]
                    break

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
