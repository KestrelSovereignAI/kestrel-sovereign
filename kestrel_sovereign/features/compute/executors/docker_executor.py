"""
Kestrel Compute Feature - Docker Executor.

Execute scripts in isolated Docker containers for maximum security.
"""

import asyncio
import logging
import shutil
import subprocess
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
from ..models import ComputeScript, ExecutionRecord

logger = logging.getLogger(__name__)


# Default images for script execution
DEFAULT_IMAGES = {
    "bash": "alpine:3.19",
    "python": "python:3.11-slim",
}
_DOCKER_CONTROL_REAP_TIMEOUT_SECONDS = 1.0


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
            default_memory_limit: Memory limit for containers
            default_cpu_quota: CPU quota (microseconds per 100ms)
            default_pids_limit: Maximum number of processes
            max_output_bytes: Maximum stdout/stderr size
        """
        super().__init__(max_output_bytes=max_output_bytes)
        self._docker_path = docker_path
        self._cached_docker_path: Optional[str] = None
        self._images = images or DEFAULT_IMAGES
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
        safe_content = self._policy.rewrite_script(
            script.content,
            script.language,
            "/workspace",
        )

        script_name = "script.py" if script.language == "python" else "script.sh"
        script_path = Path(context.workdir) / script_name
        script_path.write_text(safe_content)
        script_path.chmod(0o755)

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

        cmd.extend(["-v", f"{context.workdir}:/scripts:ro"])
        if working_dir:
            cmd.extend(["-v", f"{working_dir}:/workspace:ro"])
        cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])

        for mount in mounts or []:
            src = mount.get("src")
            dst = mount.get("dst")
            read_only = mount.get("ro", True)
            if src and dst:
                ro_flag = ":ro" if read_only else ""
                cmd.extend(["-v", f"{src}:{dst}{ro_flag}"])

        cmd.extend(["-w", "/workspace" if working_dir else "/scripts"])
        log_safe_cmd = list(cmd)
        for key, value in script.environment.items():
            cmd.extend(["-e", f"{key}={value}"])
            log_safe_cmd.extend(["-e", f"{key}=<redacted>"])

        cmd.append(image)
        log_safe_cmd.append(image)
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

        return _ExecutionResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            container_id=container_name,
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
