"""Shared lifecycle owner for session-based training adapters (#2524).

RunPod, Vast.ai, and GCP Compute each acquire a billable provider session,
answer ``start_training()`` immediately, and finish readiness, upload, and job
submission in the background. This module owns that background work so one
state machine governs all three providers; each adapter supplies only the
provider API differences as :class:`SessionProviderHooks`.

Ownership contract
------------------
* Every local job owns exactly one submission task. Its terminal outcome is
  always retrieved: submission failures are recorded on the job, and the task
  only ever ends in success, a recorded failure, or cancellation.
* Job state is a :class:`SessionJobPhase` whose legal transitions are fixed in
  one table. Nothing else mutates it.
* ``cancel``/``cleanup``/``close`` share one release task per job. Concurrent
  or repeated callers join it, and a caller's own cancellation never abandons
  it: the caller is released only after the teardown settles.
* A release first cancels the submission task and waits for it (bounded). The
  provider job ID is published by the submission coroutine in the same step
  the provider returns it, so a submission that wins the race against
  cancellation still leaves its ID on the job. A published ID is then
  compensated with provider job cancellation (where the provider has one)
  before the session itself is terminated. A job accepted remotely whose ID
  never reached us is still stopped by terminating the session it runs on.
* A job leaves the registry only after its session release succeeded and its
  submission task terminated. Anything else leaves it in
  ``RELEASE_FAILED``: custody stays visible and the release can be retried.
* ``close()`` refuses new work, cancels in-flight session acquisitions,
  releases every job, and waits for all of it within one deadline.

Error precedence
----------------
1. Caller cancellation wins: a cancelled caller of ``release``/``close``
   re-raises its ``CancelledError`` once the owned teardown settles, with any
   teardown failure logged and chained as its cause.
2. Within a release the session termination is primary. Its failure makes the
   release fail and retains the job; a failed compensating job cancellation is
   secondary, logged, and kept on the record, and never masks the primary.
3. A submission failure is recorded on the job (``FAILED``); a later release
   still terminates the session it held.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from kestrel_sovereign._async_ownership import (
    OwnedTaskOutcome,
    await_owned_task,
    raise_owned_outcome,
)
from kestrel_sovereign.kestrel_config.timeouts import (
    TRAINING_SESSION_CLOSE_DEADLINE,
    TRAINING_SESSION_RELEASE_TIMEOUT,
    TRAINING_SUBMISSION_DRAIN_TIMEOUT,
)

from ..protocol import TrainingProviderError, TrainingStatusError, TrainingSubmissionError
from ..types import TrainingConfig, TrainingJob, TrainingState, TrainingStatus


logger = logging.getLogger(__name__)

SessionT = TypeVar("SessionT")


class SessionJobPhase(enum.Enum):
    """Where one session-backed training job is in its local lifecycle."""

    SUBMITTING = "submitting"  # background readiness/upload/submission running
    TRAINING = "training"  # provider job ID published
    FAILED = "failed"  # submission failed or job cancelled; session still held
    RELEASING = "releasing"  # the one release task is tearing the job down
    RELEASE_FAILED = "release_failed"  # teardown incomplete; custody retained
    RELEASED = "released"  # session terminated; job left the registry


_LEGAL_TRANSITIONS: dict[SessionJobPhase, frozenset[SessionJobPhase]] = {
    SessionJobPhase.SUBMITTING: frozenset(
        {SessionJobPhase.TRAINING, SessionJobPhase.FAILED, SessionJobPhase.RELEASING}
    ),
    SessionJobPhase.TRAINING: frozenset(
        {SessionJobPhase.FAILED, SessionJobPhase.RELEASING}
    ),
    SessionJobPhase.FAILED: frozenset({SessionJobPhase.RELEASING}),
    SessionJobPhase.RELEASING: frozenset(
        {SessionJobPhase.RELEASED, SessionJobPhase.RELEASE_FAILED}
    ),
    SessionJobPhase.RELEASE_FAILED: frozenset({SessionJobPhase.RELEASING}),
    SessionJobPhase.RELEASED: frozenset(),
}


class ReleaseIntent(enum.Enum):
    """Why a job's provider session is being released."""

    CANCEL = "cancel"
    CLEANUP = "cleanup"
    CLOSE = "close"


class IllegalSessionJobTransition(RuntimeError):
    """A lifecycle transition outside the legal-transition table."""


class SessionReleaseError(TrainingProviderError):
    """``close()`` could not prove every provider session was released."""

    def __init__(self, message: str, *, provider: str, retained_job_ids: list[str]):
        super().__init__(message, provider=provider)
        self.retained_job_ids = retained_job_ids


@dataclass(frozen=True, slots=True)
class SessionLifecycleTimeouts:
    """Bounds on teardown; defaults come from ``kestrel_config.timeouts``."""

    submission_drain: float = TRAINING_SUBMISSION_DRAIN_TIMEOUT
    release: float = TRAINING_SESSION_RELEASE_TIMEOUT
    close: float = TRAINING_SESSION_CLOSE_DEADLINE


@dataclass(eq=False)
class SessionTrainingRecord(Generic[SessionT]):
    """Typed local custody of one session-backed training job."""

    job_id: str
    companion_id: str
    trigger_word: str
    config: TrainingConfig
    session: SessionT
    created_at: datetime
    started_at: datetime
    phase: SessionJobPhase = SessionJobPhase.SUBMITTING
    provider_job_id: str | None = None
    error: str | None = None
    compensation_error: str | None = None
    release_intent: ReleaseIntent | None = None
    provider_job_cancelled: bool = False
    session_released: bool = False
    submission_task: asyncio.Task[None] | None = field(default=None, repr=False)
    release_task: asyncio.Task[bool] | None = field(default=None, repr=False)

    def transition(self, target: SessionJobPhase, *, error: str | None = None) -> None:
        if target not in _LEGAL_TRANSITIONS[self.phase]:
            raise IllegalSessionJobTransition(
                f"training job {self.job_id}: {self.phase.value} -> {target.value}"
            )
        self.phase = target
        if error is not None:
            self.error = error

    def elapsed_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()


@dataclass(frozen=True, slots=True)
class SessionProviderHooks(Generic[SessionT]):
    """The provider-specific operations a session-backed adapter supplies.

    ``acquire_session`` starts or resumes the billable session and returns it
    (``None`` if none could be obtained). ``submit_job`` performs readiness,
    upload, and submission and returns the provider job ID. ``release_session``
    terminates or pauses one specific session and must raise when it cannot.
    ``cancel_provider_job`` is the optional compensating cancellation for a
    published provider job ID; providers without one rely on session release.
    """

    provider_name: str
    display_name: str
    acquire_session: Callable[[str, str, TrainingConfig], Awaitable[SessionT | None]]
    session_id: Callable[[SessionT], str]
    submit_job: Callable[[SessionTrainingRecord[SessionT], bytes], Awaitable[str]]
    release_session: Callable[[SessionT], Awaitable[None]]
    validate_session: Callable[[SessionT], None] | None = None
    cancel_provider_job: (
        Callable[[SessionTrainingRecord[SessionT], str], Awaitable[object]] | None
    ) = None


def _observe_outcome(task: asyncio.Task[Any]) -> None:
    """Retrieve an owned task's outcome so no exception goes unobserved."""

    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "Owned training task %s failed: %s",
            task.get_name(),
            error,
            exc_info=(type(error), error, error.__traceback__),
        )


class SessionTrainingLifecycle(Generic[SessionT]):
    """Owns every background task and billable session of one adapter."""

    def __init__(
        self,
        hooks: SessionProviderHooks[SessionT],
        timeouts: SessionLifecycleTimeouts | None = None,
    ) -> None:
        self._hooks = hooks
        self._timeouts = timeouts or SessionLifecycleTimeouts()
        self._records: dict[str, SessionTrainingRecord[SessionT]] = {}
        self._released_job_ids: set[str] = set()
        self._openings: set[asyncio.Task[SessionTrainingRecord[SessionT]]] = set()
        self._closed = False

    @property
    def records(self) -> dict[str, SessionTrainingRecord[SessionT]]:
        """Live custody registry: a job is present until its session is released."""

        return self._records

    @property
    def closed(self) -> bool:
        return self._closed

    def get(self, job_id: str) -> SessionTrainingRecord[SessionT] | None:
        return self._records.get(job_id)

    def any_live_session(self) -> SessionT | None:
        """A session still held by some job, for provider helpers that reuse one."""

        for record in self._records.values():
            if not record.session_released:
                return record.session
        return None

    def training_job(self, record: SessionTrainingRecord[SessionT]) -> TrainingJob:
        return TrainingJob(
            job_id=record.job_id,
            companion_id=record.companion_id,
            provider=self._hooks.provider_name,
            state=TrainingState.PENDING,
            trigger_word=record.trigger_word,
            created_at=record.created_at,
            started_at=None,
            config=record.config,
            provider_job_id=record.provider_job_id,
            provider_session_id=self._hooks.session_id(record.session),
        )

    def local_status(
        self,
        record: SessionTrainingRecord[SessionT],
        *,
        preparing_message: str,
    ) -> TrainingStatus | None:
        """Status answerable without the provider, or ``None`` to poll it."""

        phase = record.phase
        if phase is SessionJobPhase.TRAINING:
            return None
        if phase is SessionJobPhase.SUBMITTING:
            return TrainingStatus(
                job_id=record.job_id,
                state=TrainingState.PREPARING,
                progress=0.0,
                message=preparing_message,
            )
        if phase is SessionJobPhase.FAILED:
            return TrainingStatus(
                job_id=record.job_id,
                state=TrainingState.FAILED,
                progress=0.0,
                error=record.error or "Background submission failed",
            )
        if record.release_intent is ReleaseIntent.CLEANUP:
            raise TrainingStatusError(
                f"Job {record.job_id} is being cleaned up; provider status is "
                "no longer available",
                provider=self._hooks.provider_name,
            )
        return TrainingStatus(
            job_id=record.job_id,
            state=TrainingState.CANCELLED,
            progress=0.0,
            message="Releasing provider session",
            error=record.error,
        )

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def start(
        self,
        *,
        job_id: str,
        companion_id: str,
        trigger_word: str,
        config: TrainingConfig,
        avatar_data: bytes,
    ) -> SessionTrainingRecord[SessionT]:
        """Acquire a session, register the job, and own its submission task.

        Acquisition runs in an owned task. If the caller is cancelled, the
        acquisition is cancelled too; a job that registered anyway is released
        before the cancellation propagates, because its ID never reached a
        caller who could release it.
        """

        if self._closed:
            raise TrainingSubmissionError(
                f"{self._hooks.display_name} training adapter is closed",
                provider=self._hooks.provider_name,
            )
        opening = asyncio.create_task(
            self._open(job_id, companion_id, trigger_word, config, avatar_data),
            name=f"{self._hooks.provider_name}-training-open:{job_id}",
        )
        self._openings.add(opening)
        opening.add_done_callback(self._openings.discard)
        try:
            # ``wait`` (unlike ``shield``) raises only for this caller's own
            # cancellation, never because close() cancelled the opening.
            await asyncio.wait({opening})
        except asyncio.CancelledError as cancellation:
            opening.cancel()
            outcome = await await_owned_task(opening, cancellation)
            record = outcome.result
            if record is not None:
                release = self._ensure_release(record, ReleaseIntent.CANCEL)
                released = await await_owned_task(release, outcome.cancellation)
                if released.error is not None or released.result is not True:
                    logger.error(
                        "%s job %s registered after its caller was cancelled "
                        "and could not be released; custody retained",
                        self._hooks.display_name,
                        job_id,
                    )
            raise cancellation
        if opening.cancelled():
            raise TrainingSubmissionError(
                f"{self._hooks.display_name} training adapter closed while "
                "starting a session",
                provider=self._hooks.provider_name,
            )
        return opening.result()

    async def _open(
        self,
        job_id: str,
        companion_id: str,
        trigger_word: str,
        config: TrainingConfig,
        avatar_data: bytes,
    ) -> SessionTrainingRecord[SessionT]:
        session = await self._hooks.acquire_session(job_id, companion_id, config)
        if session is None:
            raise TrainingSubmissionError(
                f"Failed to start a {self._hooks.display_name} training session",
                provider=self._hooks.provider_name,
            )
        try:
            if self._hooks.validate_session is not None:
                self._hooks.validate_session(session)
            if self._closed:
                raise TrainingSubmissionError(
                    f"{self._hooks.display_name} training adapter closed while "
                    "its session was starting",
                    provider=self._hooks.provider_name,
                )
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                # The acquisition finished despite a cancellation request
                # (caller or close()): nobody will own this job.
                raise asyncio.CancelledError()
        except BaseException as error:
            await self._release_unregistered(session, job_id, error)
            raise

        now = datetime.now(timezone.utc)
        record: SessionTrainingRecord[SessionT] = SessionTrainingRecord(
            job_id=job_id,
            companion_id=companion_id,
            trigger_word=trigger_word,
            config=config,
            session=session,
            created_at=now,
            started_at=now,
        )
        self._records[job_id] = record
        submission = asyncio.create_task(
            self._run_submission(record, avatar_data),
            name=f"{self._hooks.provider_name}-training-submit:{job_id}",
        )
        submission.add_done_callback(_observe_outcome)
        record.submission_task = submission
        return record

    async def _release_unregistered(
        self, session: SessionT, job_id: str, cause: BaseException
    ) -> None:
        """Terminate a session no job will own; the original error still wins."""

        release = asyncio.create_task(
            self._bounded(self._hooks.release_session(session)),
            name=f"{self._hooks.provider_name}-training-release-unregistered:{job_id}",
        )
        outcome = await await_owned_task(release)
        if outcome.error is not None:
            logger.error(
                "%s session %s for rejected job %s could not be released "
                "(it may still be billing): %s",
                self._hooks.display_name,
                self._hooks.session_id(session),
                job_id,
                outcome.error,
            )
            cause.add_note(f"session release also failed: {outcome.error}")
        if outcome.cancellation is not None:
            raise outcome.cancellation from cause

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def _run_submission(
        self, record: SessionTrainingRecord[SessionT], avatar_data: bytes
    ) -> None:
        name = self._hooks.display_name
        logger.info("[%s] %s: waiting for readiness and submitting", record.job_id, name)
        try:
            provider_job_id = await self._hooks.submit_job(record, avatar_data)
        except asyncio.CancelledError:
            logger.info("[%s] %s submission cancelled", record.job_id, name)
            raise
        except Exception as error:
            logger.error(
                "[%s] %s background submission failed: %s",
                record.job_id,
                name,
                error,
                exc_info=True,
            )
            if record.phase is SessionJobPhase.SUBMITTING:
                record.transition(SessionJobPhase.FAILED, error=str(error))
            return
        # No await between the provider's return and publication: whatever
        # interrupts this task next, the ID is already on the record.
        self._publish_provider_job(record, provider_job_id)

    def _publish_provider_job(
        self, record: SessionTrainingRecord[SessionT], provider_job_id: str
    ) -> None:
        if not provider_job_id:
            if record.phase is SessionJobPhase.SUBMITTING:
                record.transition(
                    SessionJobPhase.FAILED,
                    error="Provider accepted the submission without a job ID",
                )
            return
        record.provider_job_id = provider_job_id
        if record.phase is SessionJobPhase.SUBMITTING:
            record.started_at = datetime.now(timezone.utc)
            record.transition(SessionJobPhase.TRAINING)
            logger.info(
                "[%s] %s training submitted: %s",
                record.job_id,
                self._hooks.display_name,
                provider_job_id,
            )
            return
        logger.warning(
            "[%s] %s accepted job %s after the job left SUBMITTING (%s); the "
            "ID is retained for compensating cancellation",
            record.job_id,
            self._hooks.display_name,
            provider_job_id,
            record.phase.value,
        )

    async def _drain_submission(self, record: SessionTrainingRecord[SessionT]) -> bool:
        """Cancel the submission task and wait for it; ``True`` once terminal."""

        task = record.submission_task
        if task is None or task.done():
            return True
        if not task.cancelling():
            task.cancel()
        done, _pending = await asyncio.wait(
            {task}, timeout=self._timeouts.submission_drain
        )
        if not done:
            logger.error(
                "[%s] %s submission task did not stop within %.1fs",
                record.job_id,
                self._hooks.display_name,
                self._timeouts.submission_drain,
            )
            return False
        return True

    async def stop_submission(
        self, job_id: str, *, reason: str
    ) -> SessionTrainingRecord[SessionT]:
        """Stop local submission but keep the session (provider-side cancel).

        The job becomes ``FAILED`` with ``reason``; a provider ID that was
        published before or during the stop is retained on the record.
        """

        record = self._records.get(job_id)
        if record is None:
            raise TrainingStatusError(
                f"Unknown job: {job_id}", provider=self._hooks.provider_name
            )
        stop = asyncio.create_task(
            self._stop_submission(record, reason),
            name=f"{self._hooks.provider_name}-training-stop:{job_id}",
        )
        outcome = await await_owned_task(stop)
        return raise_owned_outcome(
            outcome, operation=f"{self._hooks.display_name} submission stop of {job_id}"
        )

    async def _stop_submission(
        self, record: SessionTrainingRecord[SessionT], reason: str
    ) -> SessionTrainingRecord[SessionT]:
        await self._drain_submission(record)
        if record.phase in (SessionJobPhase.SUBMITTING, SessionJobPhase.TRAINING):
            record.transition(SessionJobPhase.FAILED, error=reason)
        return record

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    async def release(self, job_id: str, intent: ReleaseIntent) -> bool:
        """Release one job's session; ``True`` once nothing of it remains.

        Idempotent: a job this lifecycle already released reports ``True``;
        an unknown job reports ``False``.
        """

        record = self._records.get(job_id)
        if record is None:
            return job_id in self._released_job_ids
        outcome = await await_owned_task(self._ensure_release(record, intent))
        return self._release_result(outcome, job_id)

    def _release_result(self, outcome: OwnedTaskOutcome[bool], job_id: str) -> bool:
        if outcome.cancellation is None and isinstance(
            outcome.error, asyncio.CancelledError
        ):
            logger.error(
                "%s release of job %s was interrupted; custody retained",
                self._hooks.display_name,
                job_id,
            )
            return False
        return bool(
            raise_owned_outcome(
                outcome, operation=f"{self._hooks.display_name} release of {job_id}"
            )
        )

    def _ensure_release(
        self, record: SessionTrainingRecord[SessionT], intent: ReleaseIntent
    ) -> asyncio.Task[bool]:
        """Return the job's in-flight release task, starting one if needed."""

        task = record.release_task
        if task is not None and not task.done():
            return task
        record.transition(SessionJobPhase.RELEASING)
        record.release_intent = intent
        # Cancel in the same step as the request, so no submission step can
        # run between cancel() being called and the release task starting.
        submission = record.submission_task
        if submission is not None and not submission.done():
            submission.cancel()
        task = asyncio.create_task(
            self._run_release(record),
            name=f"{self._hooks.provider_name}-training-release:{record.job_id}",
        )
        task.add_done_callback(_observe_outcome)
        record.release_task = task
        return task

    async def _run_release(self, record: SessionTrainingRecord[SessionT]) -> bool:
        name = self._hooks.display_name
        try:
            drained = await self._drain_submission(record)
            await self._compensate_provider_job(record)
            if not record.session_released:
                try:
                    await self._bounded(self._hooks.release_session(record.session))
                except Exception as error:
                    record.transition(
                        SessionJobPhase.RELEASE_FAILED,
                        error=f"session release failed: {error}",
                    )
                    logger.error(
                        "[%s] %s session %s could not be released (it may still "
                        "be billing): %s",
                        record.job_id,
                        name,
                        self._hooks.session_id(record.session),
                        error,
                        exc_info=True,
                    )
                    return False
                record.session_released = True
            if not drained:
                record.transition(
                    SessionJobPhase.RELEASE_FAILED,
                    error="submission task did not stop; session released but "
                    "the task is still owned",
                )
                return False
            record.transition(SessionJobPhase.RELEASED)
            self._records.pop(record.job_id, None)
            self._released_job_ids.add(record.job_id)
            logger.info(
                "[%s] %s released session %s (%s)",
                record.job_id,
                name,
                self._hooks.session_id(record.session),
                record.release_intent.value if record.release_intent else "release",
            )
            return True
        except BaseException as error:
            # Never leave a job stranded in RELEASING: a retry must be able to
            # start a new release task from RELEASE_FAILED.
            if record.phase is SessionJobPhase.RELEASING:
                record.transition(
                    SessionJobPhase.RELEASE_FAILED,
                    error=f"release interrupted: {error!r}",
                )
            raise

    async def _compensate_provider_job(
        self, record: SessionTrainingRecord[SessionT]
    ) -> None:
        """Best-effort provider job cancellation; secondary to session release."""

        cancel = self._hooks.cancel_provider_job
        provider_job_id = record.provider_job_id
        if cancel is None or provider_job_id is None or record.provider_job_cancelled:
            return
        try:
            await self._bounded(cancel(record, provider_job_id))
        except Exception as error:
            record.compensation_error = str(error)
            logger.warning(
                "[%s] %s could not cancel provider job %s; terminating its "
                "session instead: %s",
                record.job_id,
                self._hooks.display_name,
                provider_job_id,
                error,
            )
            return
        record.provider_job_cancelled = True

    async def _bounded(self, operation: Awaitable[Any]) -> Any:
        return await asyncio.wait_for(operation, timeout=self._timeouts.release)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Refuse new work, then drain every owned task and release every session.

        Raises :class:`SessionReleaseError` naming the jobs whose sessions could
        not be proven released within the close deadline. Caller cancellation
        takes precedence over that error, after the drain settles.
        """

        self._closed = True
        closing = asyncio.create_task(
            self._close_owned(),
            name=f"{self._hooks.provider_name}-training-close",
        )
        outcome = await await_owned_task(closing)
        raise_owned_outcome(
            outcome, operation=f"{self._hooks.display_name} training adapter close"
        )

    async def _close_owned(self) -> None:
        openings = list(self._openings)
        for opening in openings:
            opening.cancel()
        releases: list[asyncio.Task[Any]] = [
            self._ensure_release(record, ReleaseIntent.CLOSE)
            for record in list(self._records.values())
        ]
        pending: set[asyncio.Task[Any]] = set(openings) | set(releases)
        if pending:
            _done, pending = await asyncio.wait(pending, timeout=self._timeouts.close)
        retained = sorted(self._records)
        if pending:
            logger.error(
                "%s close deadline (%.1fs) passed with %d owned task(s) running",
                self._hooks.display_name,
                self._timeouts.close,
                len(pending),
            )
        if retained or pending:
            raise SessionReleaseError(
                f"{self._hooks.display_name} could not release provider sessions "
                f"for jobs {retained}",
                provider=self._hooks.provider_name,
                retained_job_ids=retained,
            )
