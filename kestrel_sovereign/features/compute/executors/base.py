"""
Kestrel Compute Feature - Base Executor.

Abstract base class for script execution environments.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..models import ComputeScript, ExecutionRecord

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Base exception for execution errors."""
    pass


class ExecutionTimeoutError(ExecutionError):
    """Raised when script execution times out."""
    
    def __init__(self, script_id: str, timeout_seconds: int):
        self.script_id = script_id
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Script {script_id[:8]}... timed out after {timeout_seconds}s")


class ExecutionEnvironmentError(ExecutionError):
    """Raised when the execution environment is not available."""
    pass


class BaseExecutor(ABC):
    """
    Abstract base class for script executors.
    
    Executors provide isolated environments for running scripts safely.
    Each executor type has different isolation guarantees and requirements.
    
    Implementations:
    - UvExecutor: Uses `uv run --isolated` for Python scripts
    - DockerExecutor: Full container isolation for any language
    - LocalExecutor: Direct execution (development only)
    """
    
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
