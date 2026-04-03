"""Base class for health checkers."""
import time
import uuid
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import HealthCheck

class HealthChecker(ABC):
    """Base class for layer-specific health checkers."""

    def __init__(self, agent):
        self.agent = agent

    @abstractmethod
    async def run_all(self) -> List["HealthCheck"]:
        """Run all checks for this layer."""
        pass

    def _start_timer(self) -> float:
        return time.time()

    def _elapsed_ms(self, start: float) -> int:
        return int((time.time() - start) * 1000)

    def _gen_id(self, prefix: str) -> str:
        return f"{prefix}.{uuid.uuid4().hex[:8]}"
