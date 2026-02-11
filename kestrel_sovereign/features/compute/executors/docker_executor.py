"""
Kestrel Compute Feature - Docker Executor.

Execute scripts in isolated Docker containers for maximum security.
"""

import asyncio
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from kestrel_sovereign.kestrel_config.constants import SUBPROCESS_TIMEOUT_SHORT
from .base import BaseExecutor, ExecutionError, ExecutionEnvironmentError, ExecutionTimeoutError
from ..models import ComputeScript, ExecutionRecord
from ..destructive_policy import DestructiveOperationPolicy

logger = logging.getLogger(__name__)


# Default images for script execution
DEFAULT_IMAGES = {
    "bash": "alpine:3.19",
    "python": "python:3.11-slim",
}


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
        self._docker_path = docker_path
        self._cached_docker_path: Optional[str] = None
        self._images = images or DEFAULT_IMAGES
        self._memory_limit = default_memory_limit
        self._cpu_quota = default_cpu_quota
        self._pids_limit = default_pids_limit
        self._max_output_bytes = max_output_bytes
        self._policy = DestructiveOperationPolicy()
    
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
            import subprocess
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
            raise ExecutionError(f"No Docker image configured for language: {script.language}")
        
        execution_id = str(uuid4())
        container_name = f"kestrel_compute_{execution_id[:8]}"
        started_at = datetime.now()
        
        # Create temporary directory for script
        tmpdir = tempfile.mkdtemp(prefix="kestrel_compute_docker_")
        
        try:
            # Rewrite script for safe deletion
            safe_content = self._policy.rewrite_script(
                script.content,
                script.language,
                "/workspace",  # Container workdir
            )
            
            # Write script file
            if script.language == "python":
                script_path = Path(tmpdir) / "script.py"
            else:
                script_path = Path(tmpdir) / "script.sh"
            
            script_path.write_text(safe_content)
            script_path.chmod(0o755)
            
            # Build docker run command
            cmd = [
                docker_path, "run",
                "--name", container_name,
                "--rm",  # Remove container after exit
                "--read-only",  # Read-only root filesystem
                f"--memory={self._memory_limit}",
                f"--cpu-quota={self._cpu_quota}",
                f"--pids-limit={self._pids_limit}",
                "--security-opt=no-new-privileges",
            ]
            
            # Network isolation
            if not network:
                cmd.append("--network=none")
            
            # Mount script directory
            cmd.extend(["-v", f"{tmpdir}:/scripts:ro"])
            
            # Mount working directory if provided
            if working_dir:
                cmd.extend(["-v", f"{working_dir}:/workspace:ro"])
            
            # Create a writable tmp for the container
            cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])
            
            # Additional mounts
            if mounts:
                for mount in mounts:
                    src = mount.get("src")
                    dst = mount.get("dst")
                    ro = mount.get("ro", True)
                    if src and dst:
                        ro_flag = ":ro" if ro else ""
                        cmd.extend(["-v", f"{src}:{dst}{ro_flag}"])
            
            # Working directory
            cmd.extend(["-w", "/workspace" if working_dir else "/scripts"])
            
            # Environment variables
            for key, value in script.environment.items():
                cmd.extend(["-e", f"{key}={value}"])
            
            # Image and command
            cmd.append(image)
            
            if script.language == "python":
                cmd.extend(["python", "/scripts/script.py"])
            else:
                cmd.extend(["sh", "/scripts/script.sh"])
            
            logger.info(f"Executing script {script.id[:8]}... in Docker container")
            logger.debug(f"Container command: {' '.join(cmd)}")
            
            # Execute
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            container_id = container_name
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=script.timeout_seconds,
                )
            except asyncio.TimeoutError:
                # Kill the container on timeout
                try:
                    kill_process = await asyncio.create_subprocess_exec(
                        docker_path, "kill", container_name,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await kill_process.wait()
                except (subprocess.SubprocessError, FileNotFoundError, OSError, asyncio.CancelledError) as e:
                    logger.debug(f"Failed to kill container {container_name} on timeout: {e}")
                raise ExecutionTimeoutError(script.id, script.timeout_seconds)
            
            completed_at = datetime.now()
            
            # Process output
            stdout_str = stdout.decode(errors="replace")
            stderr_str = stderr.decode(errors="replace")
            
            if len(stdout_str) > self._max_output_bytes:
                stdout_str = stdout_str[:self._max_output_bytes] + "\n... [output truncated]"
            if len(stderr_str) > self._max_output_bytes:
                stderr_str = stderr_str[:self._max_output_bytes] + "\n... [output truncated]"
            
            record = ExecutionRecord(
                id=execution_id,
                script_id=script.id,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=process.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                executor="docker",
                container_id=container_id,
                workdir=tmpdir,
            )
            
            logger.info(
                f"Script {script.id[:8]}... completed with exit code {process.returncode} "
                f"in {record.duration_seconds:.2f}s"
            )
            
            return record
            
        except ExecutionTimeoutError:
            raise
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.error(f"Docker execution failed: {e}")
            completed_at = datetime.now()

            return ExecutionRecord(
                id=execution_id,
                script_id=script.id,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                executor="docker",
                workdir=tmpdir,
            )
        except (UnicodeDecodeError, ValueError) as e:
            logger.error(f"Docker execution failed due to encoding/value error: {e}")
            completed_at = datetime.now()

            return ExecutionRecord(
                id=execution_id,
                script_id=script.id,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                executor="docker",
                workdir=tmpdir,
            )
        except Exception as e:
            logger.error(f"Docker execution failed: {e}", exc_info=True)
            completed_at = datetime.now()

            return ExecutionRecord(
                id=execution_id,
                script_id=script.id,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                executor="docker",
                workdir=tmpdir,
            )
            
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(tmpdir)
            except (PermissionError, OSError) as e:
                logger.warning(f"Failed to clean up temp dir {tmpdir}: {e}")
            
            # Ensure container is removed (in case --rm didn't work)
            try:
                rm_process = await asyncio.create_subprocess_exec(
                    docker_path, "rm", "-f", container_name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await rm_process.wait()
            except (subprocess.SubprocessError, FileNotFoundError, OSError, asyncio.CancelledError) as e:
                logger.debug(f"Failed to remove container {container_name}: {e}")
    
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
                docker_path, "pull", image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()
            return process.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, OSError, asyncio.CancelledError) as e:
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
                docker_path, "ps", "-a", "-q",
                "--filter", "name=kestrel_compute_",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate()
            
            container_ids = stdout.decode().strip().split('\n')
            container_ids = [c for c in container_ids if c]
            
            if container_ids:
                rm_process = await asyncio.create_subprocess_exec(
                    docker_path, "rm", "-f", *container_ids,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await rm_process.wait()
                logger.info(f"Cleaned up {len(container_ids)} orphaned containers")

        except (subprocess.SubprocessError, FileNotFoundError, OSError, asyncio.CancelledError) as e:
            logger.warning(f"Container cleanup failed: {e}")
        except Exception as e:
            logger.warning(f"Container cleanup failed: {e}", exc_info=True)
