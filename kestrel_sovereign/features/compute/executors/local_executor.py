"""
Kestrel Compute Feature - Local Executor.

Execute scripts directly on the host system.

WARNING: Only for trusted development environments!
This executor provides NO isolation and should never be used in production.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .base import BaseExecutor, ExecutionError, ExecutionEnvironmentError, ExecutionTimeoutError
from ..models import ComputeScript, ExecutionRecord
from ..destructive_policy import DestructiveOperationPolicy

logger = logging.getLogger(__name__)

# Allowlist of environment variables safe to pass to subprocesses.
# Never pass API keys, tokens, encryption keys, or other secrets.
_SAFE_ENV_VARS = {
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "LC_CTYPE",
    "TMPDIR", "TERM", "TZ", "PYTHONPATH", "VIRTUAL_ENV",
}


class LocalExecutor(BaseExecutor):
    """
    Execute scripts directly on the host system.
    
    WARNING: This executor provides NO ISOLATION and should only be used
    in trusted development environments where script content is controlled.
    
    Enable with: KESTREL_ALLOW_LOCAL_COMPUTE=true
    
    Example:
        # Only if explicitly enabled
        if os.environ.get("KESTREL_ALLOW_LOCAL_COMPUTE") == "true":
            executor = LocalExecutor()
            record = await executor.execute(script)
    """
    
    def __init__(
        self,
        max_output_bytes: int = 1024 * 1024,
        require_env_flag: bool = True,
    ):
        """
        Initialize the local executor.
        
        Args:
            max_output_bytes: Maximum stdout/stderr size to capture
            require_env_flag: Require KESTREL_ALLOW_LOCAL_COMPUTE=true
        """
        self._max_output_bytes = max_output_bytes
        self._require_env_flag = require_env_flag
        self._policy = DestructiveOperationPolicy()
    
    @property
    def name(self) -> str:
        return "local"
    
    @property
    def is_available(self) -> bool:
        """Check if local execution is enabled."""
        if self._require_env_flag:
            return os.environ.get("KESTREL_ALLOW_LOCAL_COMPUTE") == "true"
        return True
    
    async def execute(
        self,
        script: ComputeScript,
        working_dir: Optional[str] = None,
    ) -> ExecutionRecord:
        """
        Execute a script directly on the host.
        
        Args:
            script: The ComputeScript to execute
            working_dir: Optional working directory
            
        Returns:
            ExecutionRecord with execution results
        """
        if not self.is_available:
            raise ExecutionEnvironmentError(
                "Local executor not enabled. Set KESTREL_ALLOW_LOCAL_COMPUTE=true"
            )
        
        execution_id = str(uuid4())
        started_at = datetime.now()
        
        # Create temporary directory for script
        tmpdir = tempfile.mkdtemp(prefix="kestrel_compute_local_")
        
        try:
            # Rewrite script for safe deletion
            safe_content = self._policy.rewrite_script(
                script.content,
                script.language,
                tmpdir,
            )
            
            # Write script file
            if script.language == "python":
                script_path = Path(tmpdir) / "script.py"
                cmd = ["python", str(script_path)]
            else:
                script_path = Path(tmpdir) / "script.sh"
                cmd = ["bash", str(script_path)]
            
            script_path.write_text(safe_content)
            script_path.chmod(0o755)
            
            # Prepare environment - only pass safe variables, never leak secrets
            env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_VARS}
            env.update(script.environment)
            
            logger.warning(
                f"Executing script {script.id[:8]}... with LOCAL executor (no isolation!)"
            )
            
            # Execute
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
                try:
                    process.kill()
                    await process.wait()
                except (ProcessLookupError, OSError, asyncio.CancelledError) as e:
                    logger.debug(f"Failed to kill process on timeout: {e}")
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
                executor="local",
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
            logger.error(f"Local execution failed: {e}")
            completed_at = datetime.now()

            return ExecutionRecord(
                id=execution_id,
                script_id=script.id,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                executor="local",
                workdir=tmpdir,
            )
        except (UnicodeDecodeError, ValueError) as e:
            logger.error(f"Local execution failed due to encoding/value error: {e}")
            completed_at = datetime.now()

            return ExecutionRecord(
                id=execution_id,
                script_id=script.id,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                executor="local",
                workdir=tmpdir,
            )
        except Exception as e:
            logger.error(f"Local execution failed: {e}", exc_info=True)
            completed_at = datetime.now()

            return ExecutionRecord(
                id=execution_id,
                script_id=script.id,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                executor="local",
                workdir=tmpdir,
            )
            
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(tmpdir)
            except (PermissionError, OSError) as e:
                logger.warning(f"Failed to clean up temp dir {tmpdir}: {e}")
