"""
Kestrel Compute Feature - Local Executor.

Execute scripts directly on the host system.

WARNING: Only for trusted development environments!
This executor provides NO isolation and should never be used in production.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .base import (
    BaseExecutor,
    ExecutionEnvironmentError,
    ExecutionTimeoutError,
    _ExecutionContext,
    _ExecutionResult,
    _SAFE_ENV_VARS,
)
from ..destructive_policy import DestructiveOperationPolicy
from ..models import ComputeScript, ExecutionRecord

logger = logging.getLogger(__name__)


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
        current_agent_data_path: Optional[str | Path] = None,
    ):
        """
        Initialize the local executor.
        
        Args:
            max_output_bytes: Maximum stdout/stderr size to capture
            require_env_flag: Require KESTREL_ALLOW_LOCAL_COMPUTE=true
        """
        super().__init__(max_output_bytes=max_output_bytes)
        self._require_env_flag = require_env_flag
        self._policy = DestructiveOperationPolicy(
            current_agent_data_path=current_agent_data_path
        )
    
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

        async def run(context: _ExecutionContext) -> _ExecutionResult:
            return await self._execute_script(script, working_dir, context)

        return await self._execute_with_lifecycle(
            script,
            temp_dir_prefix="kestrel_compute_local_",
            runner=run,
        )

    async def _execute_script(
        self,
        script: ComputeScript,
        working_dir: Optional[str],
        context: _ExecutionContext,
    ) -> _ExecutionResult:
        # Only the executor-owned temp dir authorizes direct deletion; a
        # caller-supplied working_dir is the resolution cwd for checks only.
        safe_content = self._policy.rewrite_script(
            script.content,
            script.language,
            context.workdir,
            script_cwd=working_dir or context.workdir,
        )

        if script.language == "python":
            script_path = Path(context.workdir) / "script.py"
            cmd = [sys.executable, str(script_path)]
        else:
            script_path = Path(context.workdir) / "script.sh"
            cmd = ["bash", str(script_path)]

        script_path.write_text(safe_content)
        script_path.chmod(0o755)

        # Only pass safe host variables; script-specific values remain explicit.
        env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_VARS}
        env.update(script.environment)

        logger.warning(
            "Executing script %s... with LOCAL executor (no isolation!)",
            script.id[:8],
        )
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=working_dir or context.workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=os.name == "posix",
        )

        try:
            stdout, stderr = await self._capture_process_output(
                process,
                timeout_seconds=script.timeout_seconds,
                terminate=lambda: self._kill_process_group(process),
            )
        except TimeoutError:
            raise ExecutionTimeoutError(script.id, script.timeout_seconds) from None

        return _ExecutionResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
