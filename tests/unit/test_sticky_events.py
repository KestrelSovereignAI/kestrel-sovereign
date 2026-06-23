"""Sticky aux events (current-state replay) on EventManagerMixin (#1825).

Unlike ``emit_event``'s drain-once pending buffer, sticky events are replayed to
EVERY new SSE connection until cleared — so a channel pairing QR shows in any
chat session opened while the channel is unlinked, not just the first.
"""

from __future__ import annotations

from kestrel_sovereign.agent.event_manager import EventManagerMixin


class _AgentLike(EventManagerMixin):
    def __init__(self):
        self._event_listeners = []


def test_sticky_event_persists_for_repeated_reads():
    agent = _AgentLike()
    assert agent.get_sticky_events() == []

    agent.set_sticky_event("channel_link_qr:whatsapp", "channel_link_qr", {"path": "/x"})

    # Not drained — readable repeatedly (every new SSE client replays it).
    assert agent.get_sticky_events() == [("channel_link_qr", {"path": "/x"})]
    assert agent.get_sticky_events() == [("channel_link_qr", {"path": "/x"})]


def test_sticky_event_update_and_clear():
    agent = _AgentLike()
    agent.set_sticky_event("k", "channel_link_qr", {"v": 1})
    agent.set_sticky_event("k", "channel_link_qr", {"v": 2})  # same key updates
    assert agent.get_sticky_events() == [("channel_link_qr", {"v": 2})]

    agent.clear_sticky_event("k")
    assert agent.get_sticky_events() == []
    # Clearing an absent key is a no-op.
    agent.clear_sticky_event("missing")
