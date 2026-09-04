"""
Kestrel Compute Feature - Docker Executor.

Execute scripts in isolated Docker containers for maximum security.
"""

import asyncio
import logging
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from kestrel_sovereign.kestrel_config.constants import SUBPROCESS_TIMEOUT_SHORT

from .base import (
    BaseExecutor,
    ExecutionError,
    ExecutionEnvironmentError,
    ExecutionTimeoutError,
    _ExecutionContext,
    _ExecutionResult,
)
from ..destructive_policy import DestructiveOperationPolicy
from ..models import ComputeCommand, ComputeScript, ExecutionRecord

logger = logging.getLogger(__name__)


# Default images for script execution
DEFAULT_IMAGES = {
    "bash": "alpine:3.19",
    "python": "python:3.11-slim",
}

# Image for argv execution. Keyed by nothing, because an argv vector has
# no language: element zero names a program, and the image is simply
# where that program has to exist. Named separately from
# ``DEFAULT_IMAGES["bash"]`` even though it is the same image today —
# the script entry means "the image whose shell runs bash scripts", and
# reusing it here would re-attach a shell to a path that has none.
DEFAULT_COMMAND_IMAGE = "alpine:3.19"
_DOCKER_CONTROL_REAP_TIMEOUT_SECONDS = 1.0

_CONTAINER_TRASH_DIR = "/kestrel-trash"


class DockerExecutor(BaseExecutor):
    """
    Execute scripts in Docker containers for maximum isolation.

    Security measures:
    - Read-only root filesystem
    - No network by default
    - Resource limits (CPU, memory)
    - No privilege escalation
    - PID namespace isolation

    Example:
        executor = DockerExecutor()
        if executor.is_available:
            record = await executor.execute(script, network=False)
    """

    def __init__(
        self,
        docker_path: Optional[str] = None,
        images: Optional[Dict[str, str]] = None,
        command_image: Optional[str] = None,
        default_memory_limit: str = "256m",
        default_cpu_quota: int = 50000,  # 50% of one CPU
        default_pids_limit: int = 50,
        max_output_bytes: int = 1024 * 1024,  # 1MB
        current_agent_data_path: Optional[str | Path] = None,
    ):
        """
        Initialize the Docker executor.

        Args:
            docker_path: Path to docker binary (default: auto-detect)
            images: Docker images by language (default: alpine for bash, python:3.11-slim for python)
            command_image: Docker image for argv execution (default: alpine)
            default_memory_limit: Memory limit for containers
            default_cpu_quota: CPU quota (microseconds per 100ms)
            default_pids_limit: Maximum number of processes
            max_output_bytes: Maximum stdout/stderr size
        """
        super().__init__(max_output_bytes=max_output_bytes)
        self._docker_path = docker_path
        self._cached_docker_path: Optional[str] = None
        self._images = images or DEFAULT_IMAGES
        self._command_image = command_image or DEFAULT_COMMAND_IMAGE
        self._memory_limit = default_memory_limit
        self._cpu_quota = default_cpu_quota
        self._pids_limit = default_pids_limit
        self._policy = DestructiveOperationPolicy(
            current_agent_data_path=current_agent_data_path
        )

    @property
    def name(self) -> str:
        return "docker"

    @property
    def is_available(self) -> bool:
        """Check if Docker is installed and the daemon is running."""
        try:
            docker_path = self._get_docker_path()
            if not docker_path:
                return False

            # Check if daemon is running (synchronous check)
            result = subprocess.run(
                [docker_path, "info"],
                capture_output=True,
                timeout=SUBPROCESS_TIMEOUT_SHORT,
            )
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            return False
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False

    def _get_docker_path(self) -> Optional[str]:
        """Find the docker binary path."""
        if self._cached_docker_path:
            return self._cached_docker_path

        if self._docker_path and shutil.which(self._docker_path):
            self._cached_docker_path = self._docker_path
            return self._docker_path

        docker_path = shutil.which("docker")
        if docker_path:
            self._cached_docker_path = docker_path
            return docker_path

        return None

    async def execute(
        self,
        script: ComputeScript,
        working_dir: Optional[str] = None,
        network: bool = False,
        mounts: Optional[List[Dict[str, str]]] = None,
    ) -> ExecutionRecord:
        """
        Execute a script in a Docker container.

        Args:
            script: The ComputeScript to execute
            working_dir: Optional working directory (mounted read-only)
            network: Whether to allow network access (default: False)
            mounts: Additional mounts [{"src": "/host/path", "dst": "/container/path", "ro": True}]

        Returns:
            ExecutionRecord with execution results
        """
        docker_path = self._get_docker_path()
        if not docker_path:
            raise ExecutionEnvironmentError("Docker not found")

        image = self._images.get(script.language)
        if not image:
            raise ExecutionError(
                f"No Docker image configured for language: {script.language}"
            )

        async def run(context: _ExecutionContext) -> _ExecutionResult:
            container_name = self._container_name(context.execution_id)
            return await self._execute_script(
                script,
                working_dir,
                context,
                docker_path=docker_path,
                image=image,
                container_name=container_name,
                network=network,
                mounts=mounts,
            )

        async def cleanup(context: _ExecutionContext) -> None:
            await self._remove_container(
                docker_path,
                self._container_name(context.execution_id),
            )

        return await self._execute_with_lifecycle(
            script,
            temp_dir_prefix="kestrel_compute_docker_",
            runner=run,
            cleanup=cleanup,
        )

    async def _execute_script(
        self,
        script: ComputeScript,
        working_dir: Optional[str],
        context: _ExecutionContext,
        *,
        docker_path: str,
        image: str,
        container_name: str,
        network: bool,
        mounts: Optional[List[Dict[str, str]]],
    ) -> _ExecutionResult:
        host_trash_dir = self._policy.trash_dir.expanduser().resolve(strict=False)
        host_trash_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Mount a PER-EXECUTION staging directory, never the shared trash
        # root: a read/write bind of the root would let any container script
        # read or corrupt entries trashed by previous runs and other agents.
        # Staged entries are promoted into the real trash root host-side
        # after the container exits (same filesystem, atomic renames).
        staging_dir = host_trash_dir / f".staging-{uuid.uuid4().hex[:12]}"
        staging_dir.mkdir(mode=0o700)

        # Container mounts (/scripts, /workspace) are read-only, so no
        # workdir is authorized for direct deletion; every delete moves to
        # the trash bind mount.  The container cwd only resolves relative
        # operands for policy checks — which is why it is bound once and
        # passed to both the rewriter and the container: if the two ever
        # disagreed, the rewriter would vet a different path than the one
        # the script actually names.
        container_cwd = "/workspace" if working_dir else "/scripts"
        safe_content = self._policy.rewrite_script(
            script.content,
            script.language,
            None,
            runtime_trash_dir=_CONTAINER_TRASH_DIR,
            script_cwd=container_cwd,
        )

        script_name = "script.py" if script.language == "python" else "script.sh"
        script_path = Path(context.workdir) / script_name
        script_path.write_text(safe_content)
        script_path.chmod(0o755)

        cmd, log_safe_cmd = self._container_invocation(
            docker_path=docker_path,
            container_name=container_name,
            image=image,
            working_dir=working_dir,
            container_cwd=container_cwd,
            network=network,
            mounts=mounts,
            environment=script.environment,
            binds=[
                f"{context.workdir}:/scripts:ro",
                # Safe deletions must survive the container.  The rewriter
                # uses the container path while this dedicated bind mount
                # anchors it to the host's configured Kestrel trash
                # directory.
                f"{staging_dir}:{_CONTAINER_TRASH_DIR}:rw",
            ],
        )
        if script.language == "python":
            runtime_command = ["python", "/scripts/script.py"]
        else:
            runtime_command = ["sh", "/scripts/script.sh"]
        cmd.extend(runtime_command)
        log_safe_cmd.extend(runtime_command)

        logger.info("Executing script %s... in Docker container", script.id[:8])
        logger.debug("Container command: %s", " ".join(log_safe_cmd))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await self._capture_process_output(
                process,
                timeout_seconds=script.timeout_seconds,
                terminate=lambda: self._kill_container(
                    docker_path,
                    container_name,
                ),
            )
        except TimeoutError:
            raise ExecutionTimeoutError(script.id, script.timeout_seconds) from None
        finally:
            # Promote staged trash entries even on timeout/failure: deletions
            # performed before the interruption already happened, and their
            # trash entries must stay restorable from the real trash root.
            self._promote_staged_trash(staging_dir, host_trash_dir)

        return _ExecutionResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            container_id=container_name,
        )

    def _container_invocation(
        self,
        *,
        docker_path: str,
        container_name: str,
        image: str,
        working_dir: Optional[str],
        container_cwd: str,
        network: bool,
        mounts: Optional[List[Dict[str, str]]],
        environment: Dict[str, str],
        binds: List[str],
    ) -> tuple[List[str], List[str]]:
        """Build ``docker run`` up to and including the image.

        Both execution modes build their container here so the vetted
        isolation flags (``--read-only``, ``--network=none``,
        ``--security-opt=no-new-privileges``, memory and pid limits)
        cannot drift apart between them.

        Returns ``(cmd, log_safe_cmd)``. The second is identical except
        that environment values are redacted, so a debug log of the
        container command cannot leak a secret the caller passed in.

        Everything the caller appends after the returned command lands
        *after the image*, which is where ``docker run`` stops reading
        flags — so no element of a command vector can be read as an
        option to Docker itself.
        """
        cmd = [
            docker_path,
            "run",
            "--name",
            container_name,
            "--rm",
            "--read-only",
            f"--memory={self._memory_limit}",
            f"--cpu-quota={self._cpu_quota}",
            f"--pids-limit={self._pids_limit}",
            "--security-opt=no-new-privileges",
        ]
        if not network:
            cmd.append("--network=none")

        for bind in binds:
            cmd.extend(["-v", bind])

        if working_dir:
            cmd.extend(["-v", f"{working_dir}:/workspace:ro"])
        cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])

        for mount in mounts or []:
            src = mount.get("src")
            dst = mount.get("dst")
            read_only = mount.get("ro", True)
            if src and dst:
                if dst == _CONTAINER_TRASH_DIR or dst.startswith(
                    f"{_CONTAINER_TRASH_DIR}/"
                ):
                    raise ExecutionError(
                        f"Mount destination is reserved: {_CONTAINER_TRASH_DIR}"
                    )
                ro_flag = ":ro" if read_only else ""
                cmd.extend(["-v", f"{src}:{dst}{ro_flag}"])

        cmd.extend(["-w", container_cwd])
        log_safe_cmd = list(cmd)
        for key, value in environment.items():
            cmd.extend(["-e", f"{key}={value}"])
            log_safe_cmd.extend(["-e", f"{key}=<redacted>"])

        cmd.append(image)
        log_safe_cmd.append(image)
        return cmd, log_safe_cmd

    async def execute_command(
        self,
        command: ComputeCommand,
        working_dir: Optional[str] = None,
    ) -> ExecutionRecord:
        """Execute an argv vector in a container. No script, no shell.

        ``docker run`` hands everything after the image straight to
        ``execve`` in the container, so ``command.argv[0]`` is the
        program and every later element is an argument to it. That is
        the whole difference from :meth:`execute`: a script's first word
        is read by a shell's grammar first, which is how a vetted
        ``eval`` ran an unvetted ``printf`` (#3187).

        No trash mount is created. The rewriter that redirects deletions
        into it only rewrites script text, and there is no script text
        here — mounting a writable host directory that nothing can be
        rewritten to use would be a hole with no purpose. Every other
        mount is read-only, so the container has nothing of the host's
        to delete.

        The signature is the base contract exactly: no network, no
        extra mounts. :meth:`execute` takes both because
        ``ComputeFeature`` passes them for a reviewed, signed script;
        nothing asks it of a one-shot vector, and an unused parameter is
        an untested way to widen a container.

        Args:
            command: The :class:`ComputeCommand` to execute
            working_dir: Optional working directory (mounted read-only)

        Returns:
            ExecutionRecord with execution results
        """
        docker_path = self._get_docker_path()
        if not docker_path:
            raise ExecutionEnvironmentError("Docker not found")

        async def run(context: _ExecutionContext) -> _ExecutionResult:
            container_name = self._container_name(context.execution_id)
            return await self._execute_argv(
                command,
                working_dir,
                docker_path=docker_path,
                container_name=container_name,
            )

        async def cleanup(context: _ExecutionContext) -> None:
            await self._remove_container(
                docker_path,
                self._container_name(context.execution_id),
            )

        return await self._execute_with_lifecycle(
            command,
            temp_dir_prefix="kestrel_compute_docker_command_",
            runner=run,
            cleanup=cleanup,
        )

    async def _execute_argv(
        self,
        command: ComputeCommand,
        working_dir: Optional[str],
        *,
        docker_path: str,
        container_name: str,
    ) -> _ExecutionResult:
        cmd, log_safe_cmd = self._container_invocation(
            docker_path=docker_path,
            container_name=container_name,
            image=self._command_image,
            working_dir=working_dir,
            # With no host directory to mount there is no meaningful
            # workspace; the image's read-only root is a defined place
            # to stand rather than an inherited one.
            container_cwd="/workspace" if working_dir else "/",
            network=False,
            mounts=None,
            environment=command.environment,
            binds=[],
        )
        cmd.extend(command.argv)
        log_safe_cmd.extend(command.argv)

        logger.info("Executing command %s... in Docker container", command.id[:8])
        # Quoted, unlike the script path's log line: here the reader is
        # looking at a vector whose word boundaries are the point, and a
        # space-joined rendering of ["printf", "a b"] reads as three
        # arguments. `shlex.join` makes the log line reproduce the run.
        logger.debug("Container command: %s", shlex.join(log_safe_cmd))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await self._capture_process_output(
                process,
                timeout_seconds=command.timeout_seconds,
                terminate=lambda: self._kill_container(
                    docker_path,
                    container_name,
                ),
            )
        except TimeoutError:
            raise ExecutionTimeoutError(
                command.id, command.timeout_seconds
            ) from None

        return _ExecutionResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            container_id=container_name,
        )

    @staticmethod
    def _promote_staged_trash(staging_dir: Path, host_trash_dir: Path) -> None:
        """Move per-execution staged trash entries into the real trash root.

        Renames each staged entry (same filesystem, atomic) with a collision
        suffix, then removes the staging directory. Best-effort: a promotion
        failure must not mask the execution result — but it is logged loudly
        because it strands restorable trash entries in a hidden directory.
        """
        try:
            if not staging_dir.is_dir():
                return
            for entry in staging_dir.iterdir():
                destination = host_trash_dir / entry.name
                suffix = 0
                while destination.exists():
                    suffix += 1
                    destination = host_trash_dir / f"{entry.name}.{suffix}"
                entry.rename(destination)
            staging_dir.rmdir()
        except OSError:
            logger.warning(
                "Failed to promote staged trash entries from %s into %s; "
                "trashed files remain there and are NOT visible to trash "
                "restore/listing until moved manually.",
                staging_dir,
                host_trash_dir,
                exc_info=True,
            )

    @staticmethod
    def _container_name(execution_id: str) -> str:
        return f"kestrel_compute_{execution_id[:8]}"

    @staticmethod
    async def _kill_container(docker_path: str, container_name: str) -> None:
        returncode = await DockerExecutor._run_control_command(
            docker_path,
            "kill",
            container_name,
        )
        if returncode not in (None, 0):
            logger.debug(
                "Docker kill returned exit code %s for container %s",
                returncode,
                container_name,
            )

    @staticmethod
    async def _remove_container(docker_path: str, container_name: str) -> None:
        returncode = await DockerExecutor._run_control_command(
            docker_path,
            "rm",
            "-f",
            container_name,
        )
        if returncode not in (None, 0):
            logger.debug(
                "Docker rm returned exit code %s for container %s",
                returncode,
                container_name,
            )

    @staticmethod
    async def _run_control_command(
        docker_path: str,
        action: str,
        *arguments: str,
    ) -> Optional[int]:
        """Run one bounded Docker lifecycle command.

        The Docker CLI is a child process too: a wedged daemon must not leave
        its client waiting forever, and cancellation must reap that client
        before propagating.
        """
        process: Optional[asyncio.subprocess.Process] = None
        try:
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    docker_path,
                    action,
                    *arguments,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                ),
                timeout=SUBPROCESS_TIMEOUT_SHORT,
            )
            try:
                return await asyncio.wait_for(
                    process.wait(),
                    timeout=SUBPROCESS_TIMEOUT_SHORT,
                )
            except TimeoutError:
                await DockerExecutor._stop_control_process(process)
                logger.warning(
                    "Timed out running Docker %s %s",
                    action,
                    " ".join(arguments),
                )
                return None
        except asyncio.CancelledError:
            if process is not None:
                stop_task = asyncio.create_task(
                    DockerExecutor._stop_control_process(process)
                )
                _, cancellation = await BaseExecutor._await_owned_task(stop_task)
                if cancellation is not None:
                    raise cancellation
            raise
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            OSError,
        ) as error:
            logger.debug(
                "Failed to run Docker %s %s: %s",
                action,
                " ".join(arguments),
                error,
            )
            return None

    @staticmethod
    async def _stop_control_process(process: asyncio.subprocess.Process) -> None:
        try:
            process.kill()
        except (ProcessLookupError, OSError) as error:
            logger.debug("Failed to kill Docker control process: %s", error)

        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=_DOCKER_CONTROL_REAP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning("Docker control process did not exit after kill")

    async def pull_image(self, language: str) -> bool:
        """
        Pull the Docker image for a language.

        Args:
            language: "bash" or "python"

        Returns:
            True if pull succeeded, False otherwise
        """
        docker_path = self._get_docker_path()
        if not docker_path:
            return False

        image = self._images.get(language)
        if not image:
            return False

        try:
            process = await asyncio.create_subprocess_exec(
                docker_path,
                "pull",
                image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()
            return process.returncode == 0
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            OSError,
            asyncio.CancelledError,
        ) as e:
            logger.error(f"Failed to pull image {image}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to pull image {image}: {e}", exc_info=True)
            return False

    async def cleanup(self) -> None:
        """Clean up any orphaned containers."""
        docker_path = self._get_docker_path()
        if not docker_path:
            return

        try:
            # List containers with kestrel_compute prefix
            process = await asyncio.create_subprocess_exec(
                docker_path,
                "ps",
                "-a",
                "-q",
                "--filter",
                "name=kestrel_compute_",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate()

            container_ids = stdout.decode().strip().split("\n")
            container_ids = [c for c in container_ids if c]

            if container_ids:
                rm_process = await asyncio.create_subprocess_exec(
                    docker_path,
                    "rm",
                    "-f",
                    *container_ids,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await rm_process.wait()
                logger.info(f"Cleaned up {len(container_ids)} orphaned containers")

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            OSError,
            asyncio.CancelledError,
        ) as e:
            logger.warning(f"Container cleanup failed: {e}")
        except Exception as e:
            logger.warning(f"Container cleanup failed: {e}", exc_info=True)
