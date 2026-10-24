"""Local skill execution contract; native Task values retain lifecycle results."""

from typing import Any, Protocol, runtime_checkable

from kestrel_sdk.native_a2a.types import Task


@runtime_checkable
class TaskHandler(Protocol):
    async def handle_task(
        self, task: Task, *, arguments: dict[str, Any] | None = None
    ) -> Task:
        """Execute a skill with explicit local arguments or native request metadata."""
        ...

    def get_skill_for_command(self, command: str) -> str | None: ...
