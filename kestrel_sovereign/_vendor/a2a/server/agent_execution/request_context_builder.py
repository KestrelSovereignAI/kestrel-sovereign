# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.
from abc import ABC, abstractmethod

from kestrel_sovereign._vendor.a2a.server.agent_execution import RequestContext
from kestrel_sovereign._vendor.a2a.server.context import ServerCallContext
from kestrel_sovereign._vendor.a2a.types.a2a_pb2 import SendMessageRequest, Task


class RequestContextBuilder(ABC):
    """Builds request context to be supplied to agent executor."""

    @abstractmethod
    async def build(
        self,
        context: ServerCallContext,
        params: SendMessageRequest | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Task | None = None,
    ) -> RequestContext:
        pass
