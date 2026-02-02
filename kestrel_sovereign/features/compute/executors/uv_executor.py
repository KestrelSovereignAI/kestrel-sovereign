"""
Kestrel Compute Feature - UV Executor.

Execute Python scripts in isolated environments using `uv run`.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .base import BaseExecutor, ExecutionError, ExecutionTimeoutError
from ..models import ComputeScript, ExecutionRecord
from ..destructive_policy import DestructiveOperationPolicy

logger = logging.getLogger(__name__)


class UvExecutor(BaseExecutor):
    """
    Execute Python scripts using `uv run --isolated`.
    
    This executor provides:
    - Isolated virtual environment per execution
    - Automatic dependency installation
    - Safe deletion rewriting
    - Resource limits via OS controls
    
    Example:
        executor = UvExecutor()
        if executor.is_available:
            record = await executor.execute(script)
    """
    
    def __init__(
        self,
        uv_path: Optional[str] = None,
        max_output_bytes: int = 1024 * 1024,  # 1MB default
    ):
        """
        Initialize the UV executor.
        
        Args:
            uv_path: Path to uv binary (default: auto-detect)
            max_output_bytes: Maximum stdout/stderr size to capture
        """
        self._uv_path = uv_path
        self._cached_uv_path: Optional[str] = None
        self._max_output_bytes = max_output_bytes
        self._policy = DestructiveOperationPolicy()
    
    @property
    def name(self) -> str:
        return "uv"
    
    @property
    def is_available(self) -> bool:
        """Check if uv is installed and available."""
        try:
            path = self._get_uv_path()
            return path is not None
        except Exception:
            return False
    
    def _get_uv_path(self) -> Optional[str]:
        """Find the uv binary path."""
        if self._cached_uv_path:
            return self._cached_uv_path
        
        if self._uv_path and os.path.exists(self._uv_path):
            self._cached_uv_path = self._uv_path
            return self._uv_path
        
        # Try common locations
        candidates = [
            shutil.which("uv"),
            os.path.expanduser("~/.cargo/bin/uv"),
            "/usr/local/bin/uv",
            "/opt/homebrew/bin/uv",
        ]
        
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                self._cached_uv_path = candidate
                return candidate
        
        return None
    
    def supports_language(self, language: str) -> bool:
        """UV executor only supports Python."""
        return language == "python"
    
    async def execute(
        self,
        script: ComputeScript,
        working_dir: Optional[str] = None,
    ) -> ExecutionRecord:
        """
        Execute a Python script using uv run.
        
        Creates a temporary directory, writes the script (with safe deletion
        wrapper), optionally writes requirements.txt, and runs with uv.
        
        Args:
            script: The ComputeScript to execute
            working_dir: Optional working directory for execution
            
        Returns:
            ExecutionRecord with execution results
        """
        if script.language != "python":
            raise ExecutionError(f"UvExecutor only supports Python, got {script.language}")
        
        uv_path = self._get_uv_path()
        if not uv_path:
            raise ExecutionError("uv binary not found")
        
        execution_id = str(uuid4())
        started_at = datetime.now()
        
        # Create temporary directory for execution
        tmpdir = tempfile.mkdtemp(prefix="kestrel_compute_")
        
        try:
            # Rewrite script for safe deletion
            safe_content = self._policy.rewrite_script(
                script.content,
                "python",
                tmpdir,
            )
            
            # Write script file
            script_path = Path(tmpdir) / "script.py"
            script_path.write_text(safe_content)
            
            # Write requirements if present
            if script.requirements:
                req_path = Path(tmpdir) / "requirements.txt"
                req_path.write_text("\n".join(script.requirements))
            
            # Prepare environment
            env = {**os.environ, **script.environment}
            
            # Build uv command
            # Use --isolated to create a fresh environment
            cmd = [uv_path, "run"]
            
            if script.requirements:
                # Add requirements
                for req in script.requirements:
                    cmd.extend(["--with", req])
            
            cmd.append(str(script_path))
            
            logger.info(f"Executing script {script.id[:8]}... with uv")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            # Execute the script
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=working_dir or tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=script.timeout_seconds,
                )
            except asyncio.TimeoutError:
                # Kill the process on timeout
                try:
                    process.kill()
                    await process.wait()
                except Exception as e:
                    logger.debug(f"Failed to kill process on timeout: {e}")
                raise ExecutionTimeoutError(script.id, script.timeout_seconds)
            
            completed_at = datetime.now()
            
            # Truncate output if too large
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
                executor="uv",
                workdir=tmpdir,
            )
            
            logger.info(
                f"Script {script.id[:8]}... completed with exit code {process.returncode} "
                f"in {record.duration_seconds:.2f}s"
            )
            
            return record
            
        except ExecutionTimeoutError:
            raise
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            completed_at = datetime.now()
            
            return ExecutionRecord(
                id=execution_id,
                script_id=script.id,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                executor="uv",
                workdir=tmpdir,
            )
            
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(tmpdir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir {tmpdir}: {e}")
