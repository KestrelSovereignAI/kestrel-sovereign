"""
Historical Runpod training lifecycle adapter.

Fresh training/generation is retired in favor of the authorized host catalog
workflow. The SAME shared session lifecycle retains already-admitted work,
provider status, cancellation, weights and exact resource-release custody.
"""

import asyncio
import logging
import os
from collections.abc import Awaitable, Mapping
from typing import NoReturn, Optional

from kestrel_sovereign.execution_custody import is_tool_execution_terminal_error, require_execution_work

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
    ProviderType,
    TrainingConfig,
    TrainingJob,
    TrainingState,
    TrainingStatus,
)

logger = logging.getLogger(__name__)

# Observed pod states that prove a release. A stopped (EXITED) pod no longer
# bills for its GPU, which is all a pause promises; a terminated one no longer
# exists at all.
_PAUSED_POD_STATES = frozenset({"EXITED", "STOPPED", "TERMINATED"})
_TERMINATED_POD_STATES = frozenset({"TERMINATED"})
# Read-back marker for a pod RunPod answers 404 for: it no longer exists.
_POD_GONE = "GONE"
_HISTORICAL_PREPARING_MESSAGE = "Original Runpod training submission is still preparing"

LEGACY_RUNPOD_ADMISSION_MESSAGE = (
    "Legacy Runpod training/generation admission is retired. Use the host's "
    "authorized catalog workflow with its accepted inputs, artifact identity "
    "and consent/price gates. Historical status, cancellation, download and "
    "resource cleanup remain available; no FLUX-to-SDXL mapping or provider "
    "fallback is performed."
)


def reject_legacy_runpod_new_work(error_type: type[TrainingProviderError]) -> NoReturn:
    """Reject fresh work before effects, preserving original native denial."""
    require_execution_work()
    raise error_type(LEGACY_RUNPOD_ADMISSION_MESSAGE, provider="runpod")


def _is_pod_not_found(error: Exception) -> bool:
    """Whether a RunPod call failed because the pod no longer exists.

    ``RunPodAPIError`` carries the HTTP status itself; the managed provider's
    ``raise_for_status()`` errors carry it on their response.
    """
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
    return status_code == 404


async def _pod_action(action: Awaitable[object], pod_id: str) -> None:
    """Await a stop/terminate; a pod already gone is left for confirmation."""
    try:
        await action
    except Exception as error:
        if is_tool_execution_terminal_error(error):
            raise
        if not _is_pod_not_found(error):
            raise
        logger.info(f"RunPod pod {pod_id} no longer exists")


async def _read_pod_status(
    manager, pod_id: str, outcome: str
) -> tuple[object, str | None]:
    """Read the pod back: ``(observed, STATUS)``, with ``STATUS`` ``"GONE"`` on 404.

    Raises if the provider cannot be asked or the read fails for another reason.
    """
    get_status = getattr(manager.provider, "get_status", None)
    if get_status is None:
        raise TrainingProviderError(
            f"{type(manager.provider).__name__} cannot confirm pod {pod_id} was "
            f"{outcome}; it is retained until a release succeeds",
            provider="runpod",
        )
    try:
        observed = await asyncio.to_thread(get_status, pod_id)
    except Exception as error:
        if is_tool_execution_terminal_error(error):
            raise
        if _is_pod_not_found(error):
            return None, _POD_GONE
        raise
    status = None
    if isinstance(observed, Mapping):
        status = observed.get("status") or observed.get("desiredStatus")
    return observed, status.upper() if isinstance(status, str) else None


async def _confirm_pod_state(
    manager, pod_id: str, accepted: frozenset[str], outcome: str
) -> None:
    """Read the pod back and raise unless it is in an ``accepted`` state.

    A pod RunPod no longer knows (404) is released. Anything else - an error
    payload, an empty response, or a pod still RUNNING - raises, so the job
    stays in custody and a later release retries.
    """
    observed, status = await _read_pod_status(manager, pod_id, outcome)
    if status == _POD_GONE:
        return
    if status not in accepted:
        raise TrainingProviderError(
            f"RunPod did not confirm pod {pod_id} was {outcome}: observed "
            f"{observed!r}; it is retained until a release succeeds",
            provider="runpod",
        )


class RunPodTrainingAdapter:
    """
    Recovery-only RunPodManager adapter preserving original session custody.

    Existing lifecycle hooks remain for already-admitted tails; public fresh
    admission fails before manager, task, model or provider effects. This is
    not a mapping from legacy FLUX inputs to catalog SDXL authority.
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
                if is_tool_execution_terminal_error(e):
                    raise
                raise ProviderNotAvailableError(
                    f"RunPod manager not available — install kestrel-cloud-runpod "
                    f"to enable RunPod-as-training-provider: {e}"
                )
            except Exception as e:
                if is_tool_execution_terminal_error(e):
                    raise
                raise ProviderNotAvailableError(
                    f"Failed to initialize RunPod manager: {e}"
                )
        return self._manager

    def is_available(self) -> bool:
        """
        Check configured historical provider access, not fresh-work permission.

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
        except (ProviderNotAvailableError, ImportError) as error:
            if is_tool_execution_terminal_error(error):
                raise
            return False
        except Exception as error:
            if is_tool_execution_terminal_error(error):
                raise
            logger.debug("RunPod availability check failed", exc_info=True)
            return False

    async def start_training(
        self,
        companion_id: str,
        avatar_data: bytes,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingJob:
        """Retired: fresh training belongs to the authorized catalog workflow."""
        reject_legacy_runpod_new_work(TrainingSubmissionError)

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
        """Release exactly this job's pod, raising unless RunPod confirms it.

        A persistent pod (the profile's ``persistent_pod_id`` resolves at
        release time) is paused so it can be resumed; every other pod is
        terminated. Both act on this pod's identity rather than through the
        manager's ``terminate_session``/``terminate_pod``, which log a failed
        call and return (and only stop, never terminate).

        A provider call that returns is not proof either: kestrel-cloud-runpod's
        ``DirectRunPodProvider.terminate_pod``/``stop_pod`` raise on an HTTP
        error but report ``TERMINATED``/``EXITED`` for an empty 204 response
        without observing the pod. The release therefore reads the pod back
        and succeeds only on the confirmed state (see ``_confirm_pod_state``).
        A provider with no terminate call gets the pod stopped and the release
        still raises (see ``_stop_unterminable_pod``).
        """
        manager = self._get_manager()
        persistent_pod_id = manager._expand_single_env_var(
            session.profile.persistent_pod_id
        )
        if persistent_pod_id is not None:
            await self._pause_pod(manager, session)
        else:
            await self._terminate_pod(manager, session)

    @staticmethod
    async def _pause_pod(manager, session) -> None:
        await RunPodTrainingAdapter._stop_pod(manager, session)
        logger.info(f"Paused persistent RunPod pod {session.pod_id}")

    @staticmethod
    async def _stop_pod(manager, session) -> None:
        """Stop this job's pod and confirm it no longer bills for its GPU."""
        # stop_session() takes no session argument: it stops whatever pod the
        # manager holds, so it is only used when that is this job's pod (it
        # also records GPU metering and lets a provider failure propagate).
        if manager._session is session:
            await _pod_action(manager.stop_session(), session.pod_id)
        else:
            await _pod_action(
                asyncio.to_thread(manager.provider.stop_pod, session.pod_id),
                session.pod_id,
            )
        await _confirm_pod_state(manager, session.pod_id, _PAUSED_POD_STATES, "paused")
        await RunPodTrainingAdapter._forget_session(manager, session)

    @staticmethod
    async def _stop_unterminable_pod(manager, session) -> None:
        """Stop an on-demand pod the provider cannot terminate, then raise.

        ``ManagedRunPodProvider`` has ``stop_pod`` but no ``terminate_pod``.
        Stopping ends GPU billing, but the stopped pod's disk still bills, so
        the release is only partial: it raises and the job stays
        ``RELEASE_FAILED`` with its pod ID until the pod is terminated. A pod
        already stopped by an earlier attempt is not stopped again, and one
        RunPod no longer knows (404) is fully released.
        """
        _, status = await _read_pod_status(manager, session.pod_id, "stopped")
        if status == _POD_GONE:
            await RunPodTrainingAdapter._forget_session(manager, session)
            logger.info(f"On-demand RunPod pod {session.pod_id} no longer exists")
            return
        if status in _TERMINATED_POD_STATES:
            await RunPodTrainingAdapter._forget_session(manager, session)
            logger.info(f"Terminated on-demand RunPod pod {session.pod_id}")
            return
        if status not in _PAUSED_POD_STATES:
            await RunPodTrainingAdapter._stop_pod(manager, session)
        else:
            await RunPodTrainingAdapter._forget_session(manager, session)
        raise TrainingProviderError(
            f"{type(manager.provider).__name__} cannot terminate on-demand pod "
            f"{session.pod_id}: it was stopped (GPU billing ended) but its disk "
            "still bills; it is retained until it is terminated",
            provider="runpod",
        )

    @staticmethod
    async def _terminate_pod(manager, session) -> None:
        terminate_pod = getattr(manager.provider, "terminate_pod", None)
        if terminate_pod is None:
            await RunPodTrainingAdapter._stop_unterminable_pod(manager, session)
            return
        await _pod_action(
            asyncio.to_thread(terminate_pod, session.pod_id), session.pod_id
        )
        # A stopped pod still bills for its disk, so only TERMINATED releases it.
        await _confirm_pod_state(
            manager, session.pod_id, _TERMINATED_POD_STATES, "terminated"
        )
        await RunPodTrainingAdapter._forget_session(manager, session)
        logger.info(f"Terminated on-demand RunPod pod {session.pod_id}")

    @staticmethod
    async def _forget_session(manager, session) -> None:
        """The manager must not hand a released pod to its next caller."""
        async with manager._lock:
            if manager._session is session:
                manager._session = None

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
            lookup = self._lifecycle.local_status(
                job_id,
                preparing_message=_HISTORICAL_PREPARING_MESSAGE,
            )
            if lookup.poll is None:
                return lookup.status
            record = lookup.poll

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
            if is_tool_execution_terminal_error(e):
                raise
            logger.error(f"RunPod training status network error: {e}")
            raise TrainingStatusError(f"Status check failed: {e}")
        except (KeyError, ValueError) as e:
            if is_tool_execution_terminal_error(e):
                raise
            logger.error(f"RunPod training status parse error: {e}")
            raise TrainingStatusError(f"Status check failed: {e}")
        except Exception as e:
            if is_tool_execution_terminal_error(e):
                raise
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
            if is_tool_execution_terminal_error(e):
                raise
            logger.error(f"RunPod LoRA download network error: {e}")
            raise DownloadError(f"Download failed: {e}")
        except Exception as e:
            if is_tool_execution_terminal_error(e):
                raise
            logger.error(f"Failed to download RunPod LoRA weights: {e}", exc_info=True)
            raise DownloadError(f"Download failed: {e}")

    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a training job and release its pod.

        Stops the background submission, cancels a published RunPod job, then
        releases the pod: a persistent pod is paused (it can be resumed), an
        on-demand pod is terminated. Concurrent and repeated calls share one
        teardown.

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

        Releases the job's pod: a persistent pod is paused (its disk can still
        bill), an on-demand pod is terminated. A failed release keeps the
        job tracked so it can be retried.

        IMPORTANT: Always call this after download_weights() to stop billing!

        Args:
            job_id: Job to clean up
        """
        if self._lifecycle.get(job_id) is None:
            try:
                # Only the SAME owner's retained release history can make
                # cleanup idempotently complete. A restart/eviction is not
                # evidence that the original provider resource disappeared.
                self._lifecycle.local_status(
                    job_id, preparing_message=_HISTORICAL_PREPARING_MESSAGE
                )
            except TrainingStatusError as error:
                if is_tool_execution_terminal_error(error):
                    raise
                raise TrainingProviderError(
                    "Runpod cleanup was not confirmed for this unknown original "
                    "job. Reconcile its retained cloud ledger/resource identity; "
                    "do not create a replacement pod.", provider="runpod",
                ) from error
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
        flux_version: Optional[str] = None,
    ) -> GenerationResult:
        """Retired even with an old session; that is not fresh input authority."""
        reject_legacy_runpod_new_work(GenerationError)

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
            if is_tool_execution_terminal_error(e):
                raise
            logger.warning(f"Failed to check training status: {e}")
            return None
        except Exception as e:
            if is_tool_execution_terminal_error(e):
                raise
            logger.warning(f"Failed to check training status: {e}", exc_info=True)
            return None

    async def cancel_training(self, job_id: str) -> dict:
        """
        Cancel a training job on the pod while keeping the pod.

        Stops the background submission first and returns only once it has
        stopped, cancelling on the pod any RunPod job it got accepted. The job
        becomes FAILED; call cleanup() to release the pod. The pod's /cancel
        may not stop the actual training process; for stuck jobs, use
        clear_training_lock().

        A job the pod accepted whose response was lost to the cancellation
        (the POST /train was in flight) has no ID to cancel here; it keeps
        running on the retained pod until cleanup() releases the pod.

        Args:
            job_id: Job ID to cancel

        Returns:
            Cancellation result

        Raises:
            SubmissionStopIncomplete: The submission did not stop in time (a
                job it still gets accepted is cancelled when it lands) or the
                RunPod job could not be cancelled. The job was not stopped;
                retry, or call cleanup() to release the pod.
        """
        record = await self._lifecycle.stop_submission(job_id, reason="Cancelled")
        if record.session_released:
            return {"status": "cancelled", "message": "Job pod already released"}
        if record.provider_job_id is None:
            return {"status": "cancelled", "message": "Job cancelled before submission"}
        return record.provider_cancel_result

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
