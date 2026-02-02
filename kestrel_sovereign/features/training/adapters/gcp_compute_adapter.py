"""
GCP Compute Engine Training Adapter.

Wraps GCPComputeEngineManager (session-based) to implement the TrainingProvider
protocol. This is a session-based provider that requires instance lifecycle
management.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from ..protocol import (
    TrainingProvider,
    TrainingProviderError,
    ProviderNotAvailableError,
    TrainingSubmissionError,
    TrainingStatusError,
    DownloadError,
)
from ..types import (
    ProviderType,
    TrainingConfig,
    TrainingJob,
    TrainingState,
    TrainingStatus,
)

logger = logging.getLogger(__name__)


class GCPComputeTrainingAdapter:
    """
    Adapter wrapping GCPComputeEngineManager for TrainingProvider protocol.

    This is a SESSION-BASED provider:
    - Requires starting an instance before training
    - Instance remains running until explicitly terminated
    - Training jobs run via SSH commands to the container
    - Must track both session_id and job_id

    Note: The actual manager class is named GCPComputeManager but we reference
    it via feature import to allow for future renaming.
    """

    provider_name = "gcp_compute"
    provider_type = ProviderType.SESSION_BASED

    def __init__(self, manager=None):
        """
        Initialize with optional pre-configured manager.

        Args:
            manager: GCPComputeManager instance (lazy loaded if not provided)
        """
        self._manager = manager
        self._active_jobs: dict[str, dict] = {}  # job_id -> {session, companion_id, ...}

    def _get_manager(self):
        """Lazy load the GCP Compute Engine manager."""
        if self._manager is None:
            try:
                from kestrel_sovereign.features.gcp_compute.manager import GCPComputeEngineManager
                self._manager = GCPComputeEngineManager()
            except ImportError as e:
                raise ProviderNotAvailableError(
                    f"GCP Compute manager not available: {e}"
                )
            except Exception as e:
                raise ProviderNotAvailableError(
                    f"Failed to initialize GCP Compute manager: {e}"
                )
        return self._manager

    def is_available(self) -> bool:
        """Check if GCP Compute Engine is available."""
        try:
            manager = self._get_manager()
            # Check for required credentials
            return manager is not None and manager._credentials is not None
        except Exception:
            return False

    async def start_training(
        self,
        companion_id: str,
        avatar_data: bytes,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingJob:
        """
        Start a LoRA training job on GCP Compute Engine.

        This is a multi-step process for session-based providers:
        1. Start an instance (if not already running)
        2. Upload training image to instance
        3. Submit training job via SSH

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

            # Step 1: Start or reuse a session
            session_result = await manager.start_session(
                task_profile=config.profile,
                ttl_seconds=config.ttl_seconds,
                use_spot=config.use_spot,
                metadata={"companion_id": companion_id, "job_id": job_id},
            )

            session = manager._session
            if session is None:
                raise TrainingSubmissionError("Failed to get active session")

            # Track this job with PENDING state - training not yet started
            self._active_jobs[job_id] = {
                "session": session,
                "companion_id": companion_id,
                "trigger_word": trigger_word,
                "started_at": now,
                "config": config,
                "avatar_data": avatar_data,  # Store for background submission
                "state": "pending",  # Not yet submitted
            }

            # Launch background task to upload and submit training
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
                state=TrainingState.PENDING,  # PENDING until upload+submit
                trigger_word=trigger_word,
                created_at=now,
                started_at=None,  # Not started yet
                config=config,
                provider_job_id=None,  # Will be set when submitted
                provider_session_id=session.instance_name,
            )

        except TrainingProviderError:
            raise
        except Exception as e:
            logger.error(f"GCP Compute training submission failed: {e}")
            raise TrainingSubmissionError(f"Failed to start GCP training: {e}")

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
        Background task: Upload image and submit training.

        This runs asynchronously so the HTTP endpoint can return immediately.
        """
        import tempfile
        import os

        try:
            logger.info(f"[{job_id}] Background: uploading training image...")

            # Save avatar_data to a temp file, then upload via SSH
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(avatar_data)
                temp_path = f.name

            try:
                mount_path = manager.disk_config.get("mount_path", "/workspace")
                remote_path = f"{mount_path}/training_data/{companion_id}/image_001.png"

                # Create directory and upload
                await manager._ssh_command(
                    session,
                    f"mkdir -p {mount_path}/training_data/{companion_id}",
                )
                await manager._scp_upload(session, temp_path, remote_path)
            finally:
                os.unlink(temp_path)

            logger.info(f"[{job_id}] Background: submitting training job...")

            # Submit training job
            await manager.submit_training_job(
                session=session,
                image_url=f"file://{remote_path}",
                companion_id=companion_id,
                trigger_word=trigger_word,
                network_dim=config.lora_rank,
                learning_rate=config.learning_rate,
            )

            # Update job state
            if job_id in self._active_jobs:
                self._active_jobs[job_id]["state"] = "training"
                self._active_jobs[job_id]["started_at"] = datetime.now(timezone.utc)
                logger.info(f"[{job_id}] Training submitted successfully")

        except Exception as e:
            logger.error(f"[{job_id}] Background training submission failed: {e}")
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
            companion_id = job_info["companion_id"]
            manager = self._get_manager()

            # Check if still in background submission phase
            job_state = job_info.get("state", "pending")
            if job_state == "pending":
                return TrainingStatus(
                    job_id=job_id,
                    state=TrainingState.PREPARING,
                    progress=0.0,
                    message="Uploading training image and submitting job...",
                )
            elif job_state == "failed":
                return TrainingStatus(
                    job_id=job_id,
                    state=TrainingState.FAILED,
                    progress=0.0,
                    error=job_info.get("error", "Background submission failed"),
                )

            # Poll the training status via manager
            status_result = await manager.poll_training_status(
                session=session,
                job_id=companion_id,
            )

            # Map GCP status to unified status
            gcp_status = status_result.get("status", "unknown")
            progress = status_result.get("progress", 0.0)

            if gcp_status == "completed":
                state = TrainingState.COMPLETED
            elif gcp_status == "failed":
                state = TrainingState.FAILED
            elif gcp_status == "running":
                state = TrainingState.TRAINING
            else:
                state = TrainingState.PREPARING

            elapsed = None
            if job_info.get("started_at"):
                elapsed = (datetime.now(timezone.utc) - job_info["started_at"]).total_seconds()

            return TrainingStatus(
                job_id=job_id,
                state=state,
                progress=progress,
                message=status_result.get("message"),
                elapsed_seconds=elapsed,
                provider_details=status_result,
            )

        except TrainingProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to get GCP training status: {e}")
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
            companion_id = job_info["companion_id"]
            manager = self._get_manager()

            # Download via manager
            lora_bytes = await manager.download_lora(
                session=session,
                job_id=companion_id,
            )

            return lora_bytes

        except TrainingProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to download GCP LoRA weights: {e}")
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

            return True

        except Exception as e:
            logger.error(f"Failed to cancel GCP training: {e}")
            return False

    async def cleanup(self, job_id: str) -> None:
        """
        Clean up resources for a completed job.

        For session-based providers, this terminates the instance to stop
        billing.

        Args:
            job_id: Job to clean up
        """
        try:
            if job_id not in self._active_jobs:
                return

            job_info = self._active_jobs[job_id]
            session = job_info["session"]
            manager = self._get_manager()

            # Terminate the session
            await manager.terminate_session(session)

            # Clean up tracking
            del self._active_jobs[job_id]

        except Exception as e:
            logger.warning(f"Failed to cleanup GCP session: {e}")
