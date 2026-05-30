"""Regression tests for inline-executed tool dispatch logging (#1458 follow-up).

The codex app-server adapter (openai:plan, gpt-5.5) executes tools INSIDE
the LLM call via ``item/tool/call`` RPC and reports back through an
``executed_tool_calls`` attribute on ``LLMResponse``. Before this fix
neither the non-streaming path
(``OrchestratorEngineMixin._append_executed_tool_breadcrumbs``) nor the
streaming path (``StreamingMixin._process_input_streaming_traced_locked``
inline branch) wrote anything to ``a2a_tool_dispatches`` — so the
dominant runtime execution path silently bypassed structured
observability and operators had no queryable record of "what tool did
this turn call, did it succeed, what was the error".

These tests pin that:

  - Successful inline-executed tools produce ``result_status='success'``
    rows in ``a2a_tool_dispatches`` (one per executed entry).
  - Failed inline-executed tools produce ``result_status='error'`` rows
    with the error message preserved.
  - Adapter label carries ``inline:`` prefix so an operator can tell
    inline-executed calls from orchestrator-dispatched ones.
  - Session id and turn id are populated.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityStore,
)
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


class _Agent(OrchestratorEngineMixin, TurnLifecycleMixin):
    def __init__(self, store):
        self.did = "did:test:emma"
        self._direct_tools = {}
        self._tool_to_feature = {"search_memory": "MemoryFeature"}
        self.hooks_manager = HooksManager()
        self.observability_store = store

    def _visible_features_by_tool_name(self):
        return {}


@pytest.mark.asyncio
async def test_inline_executed_success_writes_dispatch_row(tmp_path):
    """The codex app-server's ``executed_tool_calls`` payload for a
    successful inline-executed call must produce an
    ``a2a_tool_dispatches`` row with ``result_status='success'`` and
    the ``inline:`` adapter prefix so the dominant runtime execution
    path is queryable."""
    backend = SQLiteBackend(str(tmp_path / "inline-success.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    try:
        agent = _Agent(store)
        messages: list = []
        executed = [
            {
                "id": "tc-A",
                "name": "search_memory",
                "arguments": {"query": "meridian rescue", "limit": 3},
                "result": {
                    "success": True,
                    "data": {"matches": [], "took_ms": 12},
                },
            },
        ]
        async with agent._turn_lifecycle():
            await agent._append_executed_tool_breadcrumbs(
                messages, executed, session_id="sess-inline-1",
            )

        rows = await backend.fetch_all(
            """
            SELECT tool_name, adapter, result_status, session_id, turn_id
            FROM a2a_tool_dispatches
            WHERE agent_did=?
            """,
            ("did:test:emma",),
        )
        assert len(rows) == 1, (
            f"Expected exactly one inline-dispatch row, got {len(rows)}. "
            f"Before the fix this was zero — codex app-server tool calls "
            f"silently bypassed structured logging."
        )
        tool_name, adapter, status, session_id, turn_id = rows[0]
        assert tool_name == "search_memory"
        assert adapter.startswith("inline:"), (
            f"Adapter label must carry 'inline:' prefix so operators can "
            f"separate inline-executed from orchestrator-dispatched calls. "
            f"Got {adapter!r}."
        )
        assert status == "success"
        assert session_id == "sess-inline-1"
        assert str(turn_id).startswith("turn_")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_inline_executed_failure_writes_error_row(tmp_path):
    """A failed inline-executed tool (the codex adapter wraps the
    exception into ``{'success': False, 'error': ...}``) must produce
    an error-status row with the error message preserved. This is the
    specific observability gap that hid Emma's runtime memory_feature
    failures — the orchestrator's structured log had no record."""
    backend = SQLiteBackend(str(tmp_path / "inline-failure.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    try:
        agent = _Agent(store)
        messages: list = []
        executed = [
            {
                "id": "tc-B",
                "name": "search_memory",
                "arguments": {"query": "rescue"},
                "result": {
                    "success": False,
                    "error": "embedding provider unavailable",
                },
            },
        ]
        async with agent._turn_lifecycle():
            await agent._append_executed_tool_breadcrumbs(
                messages, executed, session_id="sess-inline-fail",
            )

        rows = await backend.fetch_all(
            """
            SELECT tool_name, adapter, result_status, error_message
            FROM a2a_tool_dispatches
            WHERE agent_did=?
            """,
            ("did:test:emma",),
        )
        assert len(rows) == 1
        tool_name, adapter, status, err_msg = rows[0]
        assert tool_name == "search_memory"
        assert status == "error", (
            f"Inline tool returned success=False; dispatch row must "
            f"reflect that as result_status='error', got {status!r}."
        )
        assert err_msg and "embedding provider unavailable" in err_msg
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_inline_executed_empty_list_writes_no_rows(tmp_path):
    """No executed entries → no dispatch rows. Without this guard,
    misconfigured adapters that attach an empty ``executed_tool_calls``
    list would spam the table with zero-row writes (defensive)."""
    backend = SQLiteBackend(str(tmp_path / "inline-empty.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    try:
        agent = _Agent(store)
        async with agent._turn_lifecycle():
            await agent._append_executed_tool_breadcrumbs(
                [], [], session_id="sess-empty",
            )
        rows = await backend.fetch_all(
            "SELECT COUNT(*) FROM a2a_tool_dispatches WHERE agent_did=?",
            ("did:test:emma",),
        )
        assert rows[0][0] == 0
    finally:
        await store.close()
