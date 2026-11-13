"""A live agent turn filed under a session, through the real turn lifecycle.

A producer that names the session a turn is filed under reads it with
``agent.get_turn_bound_session_id()``. :class:`TurnBoundAgent` carries that
accessor unmodified, so a test exercises the same ownership check a real turn
does rather than a stub that answers a session unconditionally: outside
:func:`live_turn` it answers ``None`` even when the agent-global session is set.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from kestrel_sovereign.agent.turn_lifecycle import (
    TurnLifecycleMixin,
    publish_turn_ownership,
)


class TurnBoundAgent(TurnLifecycleMixin):
    """The real turn-lifecycle session accessor, plus the attributes given."""

    def __init__(self, **attributes: Any) -> None:
        self._live_turn_id: Optional[str] = None
        self._active_session_id: Optional[str] = None
        for name, value in attributes.items():
            setattr(self, name, value)


@contextmanager
def live_turn(
    agent: TurnBoundAgent,
    session_id: Optional[str],
    *,
    turn_id: str = "live-turn",
) -> Iterator[None]:
    """Run the body as ``agent``'s live turn, filed under ``session_id``.

    Does what the turn lifecycle does at entry: publishes this task's
    ownership of the turn, marks the turn live, and records its session.
    """
    with publish_turn_ownership(agent, turn_id):
        agent._live_turn_id = turn_id
        agent._active_session_id = session_id
        try:
            yield
        finally:
            agent._live_turn_id = None
            agent._active_session_id = None
