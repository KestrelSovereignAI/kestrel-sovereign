"""
Kestrel Compute Feature - Base Executor.

Abstract base class for script execution environments.
"""

import asyncio
import logging
import os
import signal
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypeVar, cast
from uuid import uuid4

from ..models import ComputeScript, ExecutionRecord
from kestrel_sovereign.security.subprocess_env import SAFE_SUBPROCESS_ENV_VARS

logger = logging.getLogger(__name__)

_OUTPUT_CHUNK_BYTES = 64 * 1024
_DEFAULT_MAX_OUTPUT_BYTES = 1024 * 1024
_OUTPUT_TRUNCATED_SUFFIX = "\n... [output truncated]"
_TERMINATION_REAP_TIMEOUT_SECONDS = 5.0
_EXECUTION_CLEANUP_TIMEOUT_SECONDS = 10.0
_T = TypeVar("_T")


# Allowlist of environment variables safe to pass to executed scripts.
# Never pass API keys, tokens, encryption keys (e.g. KESTREL_DATA_KEY), or
# other secrets. Shared by every executor that builds a subprocess env so the
# host environment is never leaked into script code (F129).
# Preserve the executor's public-by-convention module contract: LocalExecutor
# and UvExecutor re-export this exact mutable ``set`` object.  The process-wide
# source allowlist remains an immutable frozenset; compute gets its historical
# local copy while the new computer-use boundary shares the canonical values.
_SAFE_ENV_VARS = set(SAFE_SUBPROCESS_ENV_VARS)


@dataclass(frozen=True, slots=True)
class _CapturedOutput:
    """A byte-bounded stream capture."""

    content: bytes
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    """State shared by one concrete executor run and its lifecycle."""

    execution_id: str
    started_at: datetime
    workdir: str


@dataclass(frozen=True, slots=True)
class _ExecutionResult:
    """Executor-specific process result consumed by the shared lifecycle."""

    exit_code: Optional[int]
    stdout: _CapturedOutput
    stderr: _CapturedOutput
    container_id: Optional[str] = None


_ExecutionRunner = Callable[[_ExecutionContext], Awaitable[_ExecutionResult]]
_ExecutionCleanup = Callable[[_ExecutionContext], Awaitable[None]]
_ProcessTerminator = Callable[[], Awaitable[None]]


class ExecutionError(Exception):
    """Base exception for execution errors."""

    pass


class ExecutionTimeoutError(ExecutionError):
    """Raised when script execution times out."""

    def __init__(self, script_id: str, timeout_seconds: int):
        self.script_id = script_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Script {script_id[:8]}... timed out after {timeout_seconds}s"
        )


class ExecutionEnvironmentError(ExecutionError):
    """Raised when the execution environment is not available."""

    pass


class BaseExecutor(ABC):
    """
    Abstract base class for script executors.

    Executors provide isolated environments for running scripts safely. The
    base lifecycle owns bounded output capture, execution records, and
    temporary-directory cleanup. Concrete executors own command construction,
    isolation, and their timeout termination strategy.

    Implementations:
    - UvExecutor: Uses `uv run --isolated --no-project` with a pinned base
      interpreter for project-free Python scripts
    - DockerExecutor: Full container isolation for any language
    - LocalExecutor: Direct execution (development only)
    """

    def __init__(
        self,
        *,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        self._max_output_bytes = max_output_bytes

    @property
    @abstractmethod
    def name(self) -> str:
        """Executor name for identification."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this executor is available in the current environment."""
        pass

    @abstractmethod
    async def execute(
        self,
        script: ComputeScript,
        working_dir: Optional[str] = None,
    ) -> ExecutionRecord:
        """
        Execute a script and return the execution record.

        Args:
            script: The ComputeScript to execute
            working_dir: Optional working directory for execution

        Returns:
            ExecutionRecord with stdout, stderr, exit_code, etc.

        Raises:
            ExecutionError: If execution fails
            ExecutionTimeoutError: If execution times out
        """
        pass

    async def _execute_with_lifecycle(
        self,
        script: ComputeScript,
        *,
        temp_dir_prefix: str,
        runner: _ExecutionRunner,
        cleanup: Optional[_ExecutionCleanup] = None,
    ) -> ExecutionRecord:
        """Run executor-specific work inside the canonical record lifecycle."""
        context = _ExecutionContext(
            execution_id=str(uuid4()),
            started_at=datetime.now(),
            workdir=tempfile.mkdtemp(prefix=temp_dir_prefix),
        )

        try:
            result = await runner(context)
            record = self._build_record(
                script=script,
                context=context,
                exit_code=result.exit_code,
                stdout=self._decode_output(result.stdout),
                stderr=self._decode_output(result.stderr),
                container_id=result.container_id,
            )
            logger.info(
                "Script %s... completed with exit code %s in %.2fs",
                script.id[:8],
                result.exit_code,
                record.duration_seconds or 0.0,
            )
            return record
        except ExecutionTimeoutError:
            raise
        except Exception as error:
            logger.error("%s execution failed: %s", self.name, error, exc_info=True)
            return self._build_record(
                script=script,
                context=context,
                exit_code=-1,
                stdout="",
                stderr=str(error),
            )
        finally:
            try:
                if cleanup is not None:
                    await self._run_cleanup_preserving_cancellation(
                        cleanup,
                        context,
                    )
            finally:
                self._remove_temp_dir(context.workdir)

    async def _run_cleanup_preserving_cancellation(
        self,
        cleanup: _ExecutionCleanup,
        context: _ExecutionContext,
    ) -> None:
        """Finish bounded cleanup without consuming caller cancellation.

        Cleanup runs in its own task so cancellation of ``execute()`` cannot
        strand executor-owned resources. Once cleanup reaches a terminal
        state, the original cancellation is re-raised to the caller.
        """
        cleanup_task = asyncio.create_task(self._bounded_cleanup(cleanup, context))
        _, cancellation = await self._await_owned_task(cleanup_task)
        if cancellation is not None:
            raise cancellation

    @staticmethod
    async def _await_owned_task(
        task: asyncio.Task[_T],
    ) -> tuple[_T, Optional[asyncio.CancelledError]]:
        """Finish an owned task despite repeated caller cancellation.

        Once caller cancellation has been observed it remains the primary
        outcome.  A later owned-task failure is retrieved and chained for
        diagnosis, but cannot replace the caller's ``CancelledError`` or leak
        as an unobserved shield-future exception.
        """

        async def settle() -> tuple[bool, object]:
            try:
                return True, await task
            except BaseException as task_error:
                # Return failures as data so cancellation of a Python 3.14
                # ``shield`` cannot install its noisy ``_log_on_exception``
                # callback on a future that later fails.  The outcome is
                # interpreted below after caller cancellation has settled.
                return False, task_error

        settled_task = asyncio.create_task(settle())
        pending_cancellation: Optional[asyncio.CancelledError] = None
        while True:
            try:
                succeeded, outcome = await asyncio.shield(settled_task)
            except asyncio.CancelledError as error:
                if settled_task.cancelled():
                    raise
                if pending_cancellation is None:
                    pending_cancellation = error
                continue
            break

        if succeeded:
            return cast(_T, outcome), pending_cancellation

        task_error = cast(BaseException, outcome)
        if pending_cancellation is None:
            raise task_error
        logger.warning(
            "Executor-owned task failed while caller cancellation was pending: %s",
            task_error,
            exc_info=(
                type(task_error),
                task_error,
                task_error.__traceback__,
            ),
        )
        raise pending_cancellation from task_error

    async def _bounded_cleanup(
        self,
        cleanup: _ExecutionCleanup,
        context: _ExecutionContext,
    ) -> None:
        """Run best-effort executor cleanup without masking the primary result."""
        try:
            await asyncio.wait_for(
                cleanup(context),
                timeout=_EXECUTION_CLEANUP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "%s cleanup exceeded %.1fs",
                self.name,
                _EXECUTION_CLEANUP_TIMEOUT_SECONDS,
            )
        except Exception as error:
            logger.warning("%s cleanup failed: %s", self.name, error, exc_info=True)

    async def _capture_process_output(
        self,
        process: asyncio.subprocess.Process,
        *,
        timeout_seconds: int,
        terminate: _ProcessTerminator,
    ) -> tuple[_CapturedOutput, _CapturedOutput]:
        """Drain both pipes concurrently, bounding retained bytes per stream.

        Timeout and cancellation both invoke the concrete executor's
        termination strategy, then drain and reap the child before propagating.
        """
        try:
            return await asyncio.wait_for(
                self._drain_process(process),
                timeout=timeout_seconds,
            )
        except (TimeoutError, asyncio.CancelledError):
            stop_task = asyncio.create_task(
                self._bounded_terminate_and_reap(process, terminate)
            )
            _, cancellation = await self._await_owned_task(stop_task)
            if cancellation is not None:
                raise cancellation
            raise

    async def _bounded_terminate_and_reap(
        self,
        process: asyncio.subprocess.Process,
        terminate: _ProcessTerminator,
    ) -> None:
        """Attempt complete teardown without allowing it to defeat the timeout."""
        try:
            await asyncio.wait_for(
                self._terminate_and_reap(process, terminate),
                timeout=_TERMINATION_REAP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            await self._kill_process(process)
            logger.warning(
                "%s process teardown exceeded %.1fs; abandoned stream drain "
                "after forcing the direct child to stop",
                self.name,
                _TERMINATION_REAP_TIMEOUT_SECONDS,
            )

    async def _drain_process(
        self,
        process: asyncio.subprocess.Process,
    ) -> tuple[_CapturedOutput, _CapturedOutput]:
        stdout, stderr, _ = await asyncio.gather(
            self._drain_stream(process.stdout),
            self._drain_stream(process.stderr),
            process.wait(),
        )
        return stdout, stderr

    async def _drain_stream(
        self,
        stream: Optional[asyncio.StreamReader],
    ) -> _CapturedOutput:
        if stream is None:
            return _CapturedOutput(b"")

        retained = bytearray()
        truncated = False
        while chunk := await stream.read(_OUTPUT_CHUNK_BYTES):
            remaining = max(0, self._max_output_bytes - len(retained))
            if remaining:
                retained.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True

        return _CapturedOutput(bytes(retained), truncated)

    async def _terminate_and_reap(
        self,
        process: asyncio.subprocess.Process,
        terminate: _ProcessTerminator,
    ) -> None:
        try:
            await terminate()
        except Exception as error:
            logger.warning("Executor termination hook failed: %s", error, exc_info=True)

        if process.returncode is None:
            await self._kill_process(process)

        results = await asyncio.gather(
            self._discard_stream(process.stdout),
            self._discard_stream(process.stderr),
            process.wait(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.debug(
                    "Failed while draining/reaping terminated process: %s", result
                )

    @staticmethod
    async def _kill_process(process: asyncio.subprocess.Process) -> None:
        try:
            process.kill()
        except (ProcessLookupError, OSError) as error:
            logger.debug("Failed to kill process: %s", error)

    @staticmethod
    async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
        """Kill a POSIX child session, falling back to the direct process."""
        process_id = getattr(process, "pid", None)
        if os.name != "posix" or process_id is None:
            await BaseExecutor._kill_process(process)
            return

        try:
            os.killpg(process_id, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError as error:
            logger.debug(
                "Failed to kill process group %s; killing child directly: %s",
                process_id,
                error,
            )
            await BaseExecutor._kill_process(process)

    @staticmethod
    async def _discard_stream(stream: Optional[asyncio.StreamReader]) -> None:
        if stream is None:
            return
        while await stream.read(_OUTPUT_CHUNK_BYTES):
            pass

    def _decode_output(self, output: _CapturedOutput) -> str:
        decoded = output.content.decode(errors="replace")
        if output.truncated:
            decoded += _OUTPUT_TRUNCATED_SUFFIX
        return decoded

    def _build_record(
        self,
        *,
        script: ComputeScript,
        context: _ExecutionContext,
        exit_code: Optional[int],
        stdout: str,
        stderr: str,
        container_id: Optional[str] = None,
    ) -> ExecutionRecord:
        return ExecutionRecord(
            id=context.execution_id,
            script_id=script.id,
            started_at=context.started_at,
            completed_at=datetime.now(),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            executor=self.name,
            container_id=container_id,
            workdir=context.workdir,
        )

    @staticmethod
    def _remove_temp_dir(workdir: str) -> None:
        try:
            shutil.rmtree(workdir)
        except (PermissionError, OSError) as error:
            logger.warning("Failed to clean up temp dir %s: %s", workdir, error)

    def supports_language(self, language: str) -> bool:
        """
        Check if this executor supports a language.

        Args:
            language: "bash" or "python"

        Returns:
            True if supported, False otherwise
        """
        # Default: support both languages
        return language in ("bash", "python")

    async def cleanup(self) -> None:
        """
        Clean up any resources used by the executor.

        Called when the executor is no longer needed.
        """
        pass
