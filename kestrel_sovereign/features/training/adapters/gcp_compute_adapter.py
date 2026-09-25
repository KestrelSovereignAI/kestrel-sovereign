"""
GCP Compute Engine Training Adapter.

Wraps GCPComputeEngineManager (session-based) to implement the TrainingProvider
protocol. This is a session-based provider that requires instance lifecycle
management.
"""

import asyncio
import logging
import os
import shlex
import subprocess
import tempfile
import uuid
from typing import Optional

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
)
from ..types import (
    ProviderType,
    TrainingConfig,
    TrainingJob,
    TrainingState,
    TrainingStatus,
)

logger = logging.getLogger(__name__)


def _is_not_found(error: Exception) -> bool:
    """Whether a Compute Engine call failed because the resource is gone."""
    try:
        from google.api_core.exceptions import NotFound
    except ImportError:
        return False
    return isinstance(error, NotFound)


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

    def __init__(
        self,
        manager=None,
        *,
        lifecycle_timeouts: Optional[SessionLifecycleTimeouts] = None,
    ):
        """
        Initialize with optional pre-configured manager.

        Args:
            manager: GCPComputeManager instance (lazy loaded if not provided)
            lifecycle_timeouts: Teardown bounds (defaults from kestrel_config)
        """
        self._manager = manager
        self._lifecycle = SessionTrainingLifecycle(
            SessionProviderHooks(
                provider_name=self.provider_name,
                display_name="GCP Compute",
                acquire_session=self._acquire_session,
                session_id=lambda session: session.instance_name,
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
        """Lazy load the GCP Compute Engine manager.

        GCP support lives in the kestrel-cloud-gcp feature package now
        (extracted in #462). If it's not installed, this adapter cleanly
        reports the provider as unavailable.
        """
        if self._manager is None:
            try:
                from kestrel_cloud_gcp.compute.manager import GCPComputeEngineManager
                self._manager = GCPComputeEngineManager()
            except ImportError as e:
                raise ProviderNotAvailableError(
                    f"GCP Compute manager not available — install kestrel-cloud-gcp "
                    f"to enable GCP-as-training-provider: {e}"
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
        except (ProviderNotAvailableError, ImportError):
            return False
        except Exception:
            logger.debug("GCP Compute availability check failed", exc_info=True)
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
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"GCP Compute training system error: {e}")
            raise TrainingSubmissionError(f"Failed to start GCP training: {e}")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"GCP Compute training connection error: {e}")
            raise TrainingSubmissionError(f"Failed to start GCP training: {e}")
        except Exception as e:
            logger.error(f"GCP Compute training submission failed: {e}", exc_info=True)
            raise TrainingSubmissionError(f"Failed to start GCP training: {e}")

        return self._lifecycle.training_job(record)

    # -- Provider hooks for the shared session lifecycle -------------------

    async def _acquire_session(
        self, job_id: str, companion_id: str, config: TrainingConfig
    ):
        """Start or reuse the instance for one job."""
        manager = self._get_manager()
        await manager.start_session(
            task_profile=config.profile,
            ttl_seconds=config.ttl_seconds,
            use_spot=config.use_spot,
            metadata={"companion_id": companion_id, "job_id": job_id},
        )
        session = manager._session
        if session is None:
            raise TrainingSubmissionError("Failed to get active session")
        return session

    async def _submit_job(self, record: SessionTrainingRecord, avatar_data: bytes) -> str:
        """Upload the training image over SSH, then start training on the instance."""
        manager = self._get_manager()
        session = record.session
        companion_id = record.companion_id
        config = record.config

        # Save avatar_data to a temp file, then upload via SSH
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(avatar_data)
            temp_path = f.name

        try:
            mount_path = manager.disk_config.get("mount_path", "/workspace")
            remote_dir = f"{mount_path}/training_data/{companion_id}"
            remote_path = f"{remote_dir}/image_001.png"

            # Create directory and upload
            await manager._ssh_command(session, f"mkdir -p {shlex.quote(remote_dir)}")
            await manager._scp_upload(session, temp_path, remote_path)
        finally:
            os.unlink(temp_path)

        logger.info(f"[{record.job_id}] Background: submitting training job...")

        # The manager keys GCP training jobs by companion ID.
        return await manager.submit_training_job(
            session=session,
            image_url=f"file://{remote_path}",
            companion_id=companion_id,
            trigger_word=record.trigger_word,
            network_dim=config.lora_rank,
            learning_rate=config.learning_rate,
        )

    async def _release_session(self, session) -> None:
        """Delete the instance, which also stops any training job on it.

        Raises unless the delete operation completed without error. The
        manager's ``terminate_session`` logs a failed delete and returns, so it
        cannot prove the instance stopped billing; the delete is issued and
        awaited here instead. An instance that no longer exists (a retry after
        an earlier delete completed) counts as released.
        """
        manager = self._get_manager()
        client = manager._get_instances_client()
        try:
            operation = await asyncio.to_thread(
                client.delete,
                project=manager.project_id,
                zone=session.zone,
                instance=session.instance_name,
            )
        except Exception as error:
            if not _is_not_found(error):
                raise
            logger.info(f"GCP instance {session.instance_name} is already deleted")
        else:
            await manager._wait_for_operation(operation.name, session.zone)
            logger.info(f"Deleted GCP instance {session.instance_name}")
        current = getattr(manager, "_session", None)
        if current is not None and current.instance_name == session.instance_name:
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
                preparing_message="Uploading training image and submitting job...",
            )
            if local is not None:
                return local

            # Poll the training status via manager
            manager = self._get_manager()
            status_result = await manager.poll_training_status(
                session=record.session,
                job_id=record.provider_job_id,
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

            return TrainingStatus(
                job_id=job_id,
                state=state,
                progress=progress,
                message=status_result.get("message"),
                elapsed_seconds=record.elapsed_seconds(),
                provider_details=status_result,
            )

        except TrainingProviderError:
            raise
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"GCP training status system error: {e}")
            raise TrainingStatusError(f"Status check failed: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"GCP training status parse error: {e}")
            raise TrainingStatusError(f"Status check failed: {e}")
        except Exception as e:
            logger.error(f"Failed to get GCP training status: {e}", exc_info=True)
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
            if record.provider_job_id is None:
                raise DownloadError(f"Job {job_id} was never submitted to GCP Compute")

            # Download via manager
            manager = self._get_manager()
            lora_bytes = await manager.download_lora(
                session=record.session,
                job_id=record.provider_job_id,
            )

            return lora_bytes

        except TrainingProviderError:
            raise
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"GCP LoRA download system error: {e}")
            raise DownloadError(f"Download failed: {e}")
        except Exception as e:
            logger.error(f"Failed to download GCP LoRA weights: {e}", exc_info=True)
            raise DownloadError(f"Download failed: {e}")

    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a training job and delete its instance.

        Stops the background upload/submission, then deletes the instance
        (which also stops any job on it). Concurrent and repeated calls share
        one teardown.

        Args:
            job_id: Job to cancel

        Returns:
            True once no instance or task of the job remains; False if the job
            is unknown or its instance could not be released (custody retained).
        """
        return await self._lifecycle.release(job_id, ReleaseIntent.CANCEL)

    async def cleanup(self, job_id: str) -> None:
        """
        Clean up resources for a completed job.

        Deletes the instance to stop billing. A failed release keeps the job
        tracked so it can be retried.

        Args:
            job_id: Job to clean up
        """
        if self._lifecycle.get(job_id) is None:
            return
        if not await self._lifecycle.release(job_id, ReleaseIntent.CLEANUP):
            logger.warning(f"Failed to cleanup GCP session for job {job_id}; custody retained")

    async def close(self) -> None:
        """Drain background submissions and release every instance this adapter holds."""
        await self._lifecycle.close()
