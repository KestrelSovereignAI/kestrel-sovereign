"""Public endpoint contracts for exact conversation purges."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.endpoints import conversations


class _Request:
    def __init__(self, storage: SimpleNamespace) -> None:
        self.state = SimpleNamespace(agent=SimpleNamespace(storage=storage))

    async def json(self) -> dict[str, str]:
        return {"reason": "privacy request"}


@pytest.mark.asyncio
async def test_purge_endpoint_reports_the_complete_uncapped_count() -> None:
    purge = AsyncMock(return_value=10_001)
    storage = SimpleNamespace(
        agent_id="agent-endpoint-count",
        purge_conversation_session=purge,
    )

    result = await conversations.purge_conversation.__wrapped__(
        _Request(storage), "session-over-ten-thousand"
    )

    assert result == {
        "success": True,
        "session_id": "session-over-ten-thousand",
        "purged_count": 10_001,
        "reason": "privacy request",
    }
    purge.assert_awaited_once_with(
        "session-over-ten-thousand",
        "agent-endpoint-count",
        reason="privacy request",
    )
