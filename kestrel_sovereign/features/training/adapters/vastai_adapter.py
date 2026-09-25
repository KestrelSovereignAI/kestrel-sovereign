"""
Vast.ai Training Adapter.

Wraps VastAIManager (session-based) to implement the TrainingProvider protocol.
This is a session-based provider that requires instance lifecycle management.

Uses HTTP API endpoints from the shared SimpleTuner Docker image
(same as RunPod/Vertex AI) for training and generation.
"""

import logging
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
from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DEFAULT
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

    def __init__(
        self,
        manager=None,
        *,
        lifecycle_timeouts: Optional[SessionLifecycleTimeouts] = None,
        http_transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        """
        Initialize with optional pre-configured manager.

        Args:
            manager: VastAIManager instance (lazy loaded if not provided)
            lifecycle_timeouts: Teardown bounds (defaults from kestrel_config)
            http_transport: Transport for direct Vast.ai API calls (default:
                the network)
        """
        self._manager = manager
        self._http_transport = http_transport
        self._lifecycle = SessionTrainingLifecycle(
            SessionProviderHooks(
                provider_name=self.provider_name,
                display_name="Vast.ai",
                acquire_session=self._acquire_session,
                session_id=lambda session: str(session.instance_id),
                validate_session=self._validate_session,
                submit_job=self._submit_job,
                release_session=self._release_session,
            ),
            lifecycle_timeouts,
        )

    @property
    def _active_jobs(self) -> dict[str, SessionTrainingRecord]:
        """Live custody registry: a job stays here until its instance is released."""
        return self._lifecycle.records

    def _get_manager(self):
        """Lazy load the Vast.ai manager.

        Vast.ai support lives in the kestrel-cloud-vastai feature package
        now (extracted in #462). If it's not installed, this adapter
        cleanly reports the provider as unavailable.
        """
        if self._manager is None:
            try:
                from kestrel_cloud_vastai.manager import VastAIManager
                self._manager = VastAIManager()
            except ImportError as e:
                raise ProviderNotAvailableError(
                    f"Vast.ai manager not available — install kestrel-cloud-vastai "
                    f"to enable Vast.ai-as-training-provider: {e}"
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

        Steps 2-3 run in a background submission task owned by the shared
        session lifecycle, so this returns as soon as the instance is up.

        Args:
            companion_id: Companion being trained
            avatar_data: Training image bytes
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
        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"Vast.ai training connection error: {e}")
            raise TrainingSubmissionError(f"Failed to start Vast.ai training: {e}")
        except Exception as e:
            logger.error(f"Vast.ai training submission failed: {e}", exc_info=True)
            raise TrainingSubmissionError(f"Failed to start Vast.ai training: {e}")

        logger.info(
            f"Training instance ready: {record.session.instance_id}, "
            f"URL: {record.session.backend_base_url}"
        )
        return self._lifecycle.training_job(record)

    # -- Provider hooks for the shared session lifecycle -------------------

    async def _acquire_session(
        self, job_id: str, companion_id: str, config: TrainingConfig
    ):
        """Rent (or reuse) the instance for one job."""
        manager = self._get_manager()
        logger.info(f"Starting Vast.ai training instance for companion {companion_id}")
        await manager.start_session(
            task_profile=config.profile,
            ttl_seconds=config.ttl_seconds,
            metadata={"companion_id": companion_id, "job_id": job_id},
        )
        session = manager._session
        if session is None:
            raise TrainingSubmissionError("Failed to get active session")
        return session

    @staticmethod
    def _validate_session(session) -> None:
        if not session.backend_base_url:
            raise TrainingSubmissionError(
                f"Training instance {session.instance_id} started but has no backend URL"
            )

    async def _submit_job(self, record: SessionTrainingRecord, avatar_data: bytes) -> str:
        """Wait for the SimpleTuner API, then POST /train."""
        config = record.config
        return await self._get_manager().submit_training_job_http(
            session=record.session,
            avatar_data=avatar_data,
            companion_id=record.companion_id,
            trigger_word=record.trigger_word,
            steps=config.steps,
            lora_rank=config.lora_rank,
            callback_url=config.callback_url,
            wait_for_ready=True,
        )

    async def _release_session(self, session) -> None:
        """Destroy the instance; Vast.ai bills hourly, and this also stops its job.

        Raises unless Vast.ai confirms the destroy. Neither the SDK nor the
        manager can prove it: the SDK's ``destroy_instance`` swallows request
        errors and discards the response, and the manager's
        ``terminate_session`` logs a failure and returns. The destroy is
        therefore issued against the Vast.ai API directly (at the SDK's
        configured server) and its response checked. An instance the API no
        longer knows (a retry after an earlier destroy completed) counts as
        released.
        """
        manager = self._get_manager()
        server_url = manager._get_sdk().server_url.rstrip("/")
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_DEFAULT, transport=self._http_transport
        ) as client:
            response = await client.request(
                "DELETE",
                f"{server_url}/api/v0/instances/{session.instance_id}/",
                headers={"Authorization": f"Bearer {manager.api_key}"},
                json={},
            )
        if response.status_code == 404:
            logger.info(f"Vast.ai instance {session.instance_id} is already destroyed")
        else:
            response.raise_for_status()
            result = response.json()
            if not (isinstance(result, dict) and result.get("success") is True):
                raise TrainingProviderError(
                    f"Vast.ai did not confirm destroying instance "
                    f"{session.instance_id}: {result!r}",
                    provider=self.provider_name,
                )
            logger.info(f"Destroyed Vast.ai instance {session.instance_id}")
        if getattr(manager, "_session", None) is session:
            manager._session = None

    # -- TrainingProvider --------------------------------------------------

    async def get_status(self, job_id: str) -> TrainingStatus:
        """
        Get status of a training job.

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

            # Poll the training status via HTTP API
            manager = self._get_manager()
            status_result = await manager.poll_training_status_http(
                session=record.session,
                job_id=record.provider_job_id,
            )

            # Map SimpleTuner status to unified status
            api_status = status_result.get("status", "unknown").lower()
            progress = status_result.get("progress", 0.0)
            error = status_result.get("error")

            # Map status strings to TrainingState
            if api_status == "completed":
                state = TrainingState.COMPLETED
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
            record = self._lifecycle.get(job_id)
            if record is None:
                raise DownloadError(f"Unknown job: {job_id}")
            training_job_id = record.provider_job_id
            if training_job_id is None:
                raise DownloadError(f"Job {job_id} was never submitted to Vast.ai")

            manager = self._get_manager()

            # Download via HTTP API
            logger.info(f"Downloading LoRA weights for job {training_job_id}")
            lora_bytes = await manager.download_lora_http(
                session=record.session,
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
        Cancel a training job and destroy its instance.

        Stops the background submission, then terminates the instance (which
        also stops any job on it). Concurrent and repeated calls share one
        teardown.

        Args:
            job_id: Job to cancel

        Returns:
            True once no instance or task of the job remains; False if the job
            is unknown or its instance could not be released (custody retained).
        """
        released = await self._lifecycle.release(job_id, ReleaseIntent.CANCEL)
        if released:
            logger.info(f"Cancelled Vast.ai training job {job_id}")
        return released

    async def cleanup(self, job_id: str) -> None:
        """
        Clean up resources for a completed job.

        Terminates the instance to stop billing. Vast.ai bills by the hour so
        early termination saves money. A failed release keeps the job tracked
        so it can be retried.

        Args:
            job_id: Job to clean up
        """
        if self._lifecycle.get(job_id) is None:
            return
        if not await self._lifecycle.release(job_id, ReleaseIntent.CLEANUP):
            logger.warning(f"Failed to cleanup Vast.ai session for job {job_id}; custody retained")

    async def close(self) -> None:
        """Drain background submissions and release every instance this adapter holds."""
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
                session = self._lifecycle.any_live_session()
                if session is not None:
                    logger.info(f"Reusing existing session: {session.instance_id}")

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
