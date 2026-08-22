"""
Kestrel Compute Feature - UV Executor.

Execute Python scripts in project-free environments using `uv run`.
"""

import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from .base import (
    BaseExecutor,
    ExecutionEnvironmentError,
    ExecutionError,
    ExecutionTimeoutError,
    _ExecutionContext,
    _ExecutionResult,
    _SAFE_ENV_VARS,
)
from ..destructive_policy import DestructiveOperationPolicy
from ..models import ComputeScript, ExecutionRecord

logger = logging.getLogger(__name__)


class UvExecutor(BaseExecutor):
    """
    Execute Python scripts using a project-free ephemeral uv environment.

    In uv 0.9, ``uv run --isolated`` alone still discovers and installs the
    current project. This executor combines ``--isolated``, ``--no-project``,
    and an explicit base interpreter path. The first forces a fresh execution
    environment, the second prevents project/workspace discovery, and the
    third anchors interpreter selection outside Kestrel's runtime as
    defense-in-depth against uv resolver changes.
    
    This executor provides:
    - Project-free ephemeral environment per execution
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
        current_agent_data_path: Optional[str | Path] = None,
    ):
        """
        Initialize the UV executor.
        
        Args:
            uv_path: Path to uv binary (default: auto-detect)
            max_output_bytes: Maximum stdout/stderr size to capture
        """
        super().__init__(max_output_bytes=max_output_bytes)
        self._uv_path = uv_path
        self._cached_uv_path: Optional[str] = None
        self._policy = DestructiveOperationPolicy(
            current_agent_data_path=current_agent_data_path
        )
    
    @property
    def name(self) -> str:
        return "uv"
    
    @property
    def is_available(self) -> bool:
        """Check that uv and a safe base interpreter are available."""
        try:
            path = self._get_uv_path()
            return path is not None and bool(self._get_base_python_path())
        except (
            ExecutionEnvironmentError,
            FileNotFoundError,
            OSError,
            PermissionError,
        ):
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

    @staticmethod
    def _get_base_python_path() -> str:
        """Resolve the executable outside Kestrel's virtual environment.

        ``--isolated --no-project`` provides the currently verified fresh,
        project-free behavior. The concrete executable additionally makes the
        interpreter choice independent of the working directory and fails
        closed if future uv resolution behavior would otherwise select
        Kestrel's runtime. Running Kestrel outside a virtual environment cannot
        provide that independent trust anchor, so it is rejected.
        """
        if sys.prefix == sys.base_prefix:
            raise ExecutionEnvironmentError(
                "UvExecutor requires Kestrel to run inside a Python venv or "
                "virtualenv so compute scripts cannot inherit Kestrel's "
                "site-packages; a Conda environment alone is not sufficient"
            )

        base_prefix = Path(sys.base_prefix)
        candidates: list[Path] = []
        base_executable = getattr(sys, "_base_executable", None)
        if base_executable:
            candidates.append(Path(base_executable))

        if os.name == "nt":
            candidates.extend(
                [
                    base_prefix / "python.exe",
                    base_prefix / "Scripts" / "python.exe",
                ]
            )
        else:
            candidates.extend(
                [
                    base_prefix
                    / "bin"
                    / f"python{sys.version_info.major}.{sys.version_info.minor}",
                    base_prefix / "bin" / "python3",
                    base_prefix / "bin" / "python",
                ]
            )

        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())

        raise ExecutionEnvironmentError(
            "UvExecutor could not resolve an executable base Python interpreter"
        )
    
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
        base_python_path = self._get_base_python_path()

        async def run(context: _ExecutionContext) -> _ExecutionResult:
            return await self._execute_script(
                script,
                working_dir,
                context,
                uv_path,
                base_python_path,
            )

        return await self._execute_with_lifecycle(
            script,
            temp_dir_prefix="kestrel_compute_",
            runner=run,
        )

    async def _execute_script(
        self,
        script: ComputeScript,
        working_dir: Optional[str],
        context: _ExecutionContext,
        uv_path: str,
        base_python_path: str,
    ) -> _ExecutionResult:
        # Only the executor-owned temp dir authorizes direct deletion; the
        # Python runtime resolves relative paths against the child's cwd.
        safe_content = self._policy.rewrite_script(
            script.content,
            "python",
            context.workdir,
        )

        script_path = Path(context.workdir) / "script.py"
        script_path.write_text(safe_content)

        if script.requirements:
            req_path = Path(context.workdir) / "requirements.txt"
            req_path.write_text("\n".join(script.requirements))

        # Only pass safe host variables; never leak host credentials to scripts.
        env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_VARS}
        # Avoid uv falling back to an inaccessible cache beneath host HOME.
        env["UV_CACHE_DIR"] = str(Path(context.workdir) / ".uv-cache")
        # Apply script-supplied overrides, then enforce Python isolation below.
        env.update(script.environment)
        # PYTHONPATH bypasses uv's interpreter/environment boundary entirely.
        env.pop("PYTHONPATH", None)

        cmd = [
            uv_path,
            "run",
            "--isolated",
            "--no-project",
            "--python",
            base_python_path,
        ]
        for requirement in script.requirements:
            cmd.extend(["--with", requirement])
        cmd.append(str(script_path))

        logger.info("Executing script %s... with uv", script.id[:8])
        logger.debug("Command: %s", " ".join(cmd))
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
