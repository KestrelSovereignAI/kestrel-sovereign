"""Regression: #1914 typed parts survive the codex reader-task boundary (#2081).

On the openai:plan route, tools execute INLINE via the codex app-server. The
long-lived ``CodexAppServerClient`` spawns its ``_read_loop`` ONCE (inside the
first plan turn's ``part_collector``), so that reader task freezes a COPY of
turn-1's context — including the ``emit_part`` collector. Each server→client
``item/tool/call`` is then dispatched in ITS OWN ``asyncio.create_task`` from
the reader, inheriting the reader's frozen context, not the current turn's.

Pre-fix consequence: ``emit_part`` calls made by an inline tool landed on
turn-1's abandoned collector and were SILENTLY DROPPED on every later turn
(``emit_part`` even returned ``True``, appending to the stale list). Since Emma
runs openai:plan long-lived, the WhatsApp ``channel_link`` part — requested on a
later turn — never rendered, regressing #1918.

The fix re-binds the OWNING turn's collector across the reader-task boundary
inside ``OrchestratorEngineMixin._make_inline_tool_executor`` (captured at
closure-creation time, on the turn task; re-entered via ``bind_part_collector``
around the actual tool execution). This test reproduces the codex task topology
WITHOUT a live app-server, driving the executor through the REAL fixed path, and
asserts the part is drained on the turn task for turns 1, 2 AND 3.
"""
from __future__ import annotations

import pytest

from tests.utils.codex_inline import OwnedCallbackReader

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.parts import (
    drain_parts,
    emit_part,
    part_collector,
)


class _Agent(OrchestratorEngineMixin):
    """Minimal host for ``_make_inline_tool_executor``.

    ``execute_named_tool`` is stubbed to call ``emit_part`` exactly like the
    WhatsApp pairing tool does when it runs inline — that emission is what must
    land on the owning turn's collector.
    """

    async def execute_named_tool(self, name, args, *, session_id, source, _capture):
        # Mirror the inline dispatch contract: record effective args and emit a
        # typed part, as the pairing tool does through ProxyFeature.
        _capture["effective_args"] = args
        emit_part("channel_link", {"tool": name, "args": args}, part_id=name)
        return {"ok": True, "tool": name}


@pytest.mark.asyncio
async def test_inline_emit_part_lands_on_owning_turn_across_reader_boundary():
    agent = _Agent()
    harness = OwnedCallbackReader()

    async def run_turn(turn_no: int):
        drained_on_turn = []
        with part_collector():
            # Executor is built INSIDE the turn task (as orchestrator_engine
            # does at :500/:2152/:2501), so it captures THIS turn's collector.
            executor = agent._make_inline_tool_executor(f"session-{turn_no}")
            # First turn also starts the long-lived reader — freezing turn-1's
            # context, exactly like ensure_started awaited inside _run_turn.
            if turn_no == 1:
                await harness.start()
            # The tool call is dispatched on a reader-spawned task.
            eff_args, result = await harness.dispatch(
                executor, f"whatsapp_link_{turn_no}", {"turn": turn_no},
            )
            assert result["ok"] is True
            # The turn drains its own collector (as streaming.py does on the
            # turn task).
            drained_on_turn = drain_parts()
        return drained_on_turn

    parts1 = await run_turn(1)
    parts2 = await run_turn(2)
    parts3 = await run_turn(3)
    await harness.stop()

    # Pre-fix: only turn 1 saw its part (reader froze turn-1's collector);
    # turns 2 and 3 drained empty. Post-fix: every turn sees its own part.
    assert [p["data"]["args"]["turn"] for p in parts1] == [1]
    assert [p["data"]["args"]["turn"] for p in parts2] == [2], (
        "turn 2's inline part was dropped — collector not re-bound across the "
        "codex reader-task boundary"
    )
    assert [p["data"]["args"]["turn"] for p in parts3] == [3], (
        "turn 3's inline part was dropped — collector not re-bound across the "
        "codex reader-task boundary"
    )


@pytest.mark.asyncio
async def test_anthropic_style_in_turn_tool_still_delivers():
    """No regression on the anthropic path: a tool that runs INSIDE the turn
    task (no reader boundary) still lands its part on the turn's collector."""
    agent = _Agent()
    with part_collector():
        executor = agent._make_inline_tool_executor("session-anthropic")
        # Run the executor directly on the turn task (no cross-task dispatch).
        _eff, result = await executor("whatsapp_link", {"turn": 0})
        assert result["ok"] is True
        drained = drain_parts()
    assert [p["type"] for p in drained] == ["channel_link"]
