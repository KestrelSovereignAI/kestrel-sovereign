import json

import pytest

from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityStore,
    ToolDispatchEntry,
    redact_tool_args_json,
)
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.llm.adapter import ToolCall
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


class _BrokenTool:
    async def execute(self, **kwargs):
        raise ValueError("broken on purpose")


class _Agent(OrchestratorEngineMixin, TurnLifecycleMixin):
    def __init__(self, store):
        self.did = "did:test:emma"
        self._direct_tools = {"broken_tool": _BrokenTool()}
        self._tool_to_feature = {"broken_tool": "TestFeature"}
        self.hooks_manager = HooksManager()
        self.observability_store = store


class _SchemaCaptureBackend:
    backend_type = "postgres"
    is_connected = True

    def __init__(self):
        self.scripts = []
        self.statements = []

    async def execute_script(self, script):
        self.scripts.append(script)

    async def execute(self, query, params=()):
        self.statements.append(query)
        return 0

    async def close(self):
        self.is_connected = False


@pytest.mark.asyncio
async def test_tool_dispatch_schema_uses_database_dialect_helpers():
    backend = _SchemaCaptureBackend()
    store = ObservabilityStore(backend)

    await store.initialize()

    schema = "\n".join(backend.scripts)
    assert "a2a_tool_dispatches" in schema
    assert "id BIGSERIAL PRIMARY KEY" in schema
    assert "AUTOINCREMENT" not in schema


@pytest.mark.asyncio
async def test_tool_dispatch_write_path_redacts_and_queries(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "tool-dispatches.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    try:
        await store.log_tool_dispatch(
            ToolDispatchEntry(
                agent_did="did:test:emma",
                session_id=None,
                turn_id="turn_1",
                tool_name="github",
                adapter="cli.github",
                args_redacted={
                    "token": "secret-token",
                    "query": "hello",
                    "blob": "x" * 5000,
                },
                result_status="error",
                error_class="RuntimeError",
                error_message="boom",
                latency_ms=12,
                result_size_bytes=42,
            )
        )

        row = await backend.fetch_one(
            "SELECT session_id, args_redacted FROM a2a_tool_dispatches WHERE agent_did=?",
            ("did:test:emma",),
        )
        assert row[0] is None
        args = json.loads(row[1])
        assert args["token"] == "<redacted>"
        assert len(row[1].encode("utf-8")) <= 2048

        summary = await store.tool_failure_rate("did:test:emma", last_n_turns=100)
        assert summary["total_calls"] == 1
        assert summary["failure_calls"] == 1
        assert summary["dominant_failures"][0]["tool_name"] == "github"
        assert summary["dominant_failures"][0]["error_class"] == "RuntimeError"

        failures = await store.recent_failures("did:test:emma")
        assert failures[0]["error_message"] == "boom"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_llm_call_observability_records_and_filters_agent_did(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "llm-agent-did.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    try:
        await store.log_llm_call(
            agent_did="did:test:emma",
            provider="openai",
            model="gpt-5.4",
            duration_ms=17,
            session_id="session-1",
            tool_calls=[{"name": "github_issue_view", "arguments": {}}],
        )
        await store.log_llm_call(
            agent_did="did:test:claw",
            provider="openai",
            model="gpt-5.4",
            duration_ms=23,
            session_id="session-2",
        )

        emma_calls = await store.query_llm_calls(agent_did="did:test:emma")

        assert len(emma_calls) == 1
        assert emma_calls[0].agent_did == "did:test:emma"
        assert emma_calls[0].session_id == "session-1"
        assert emma_calls[0].tool_calls == [
            {"name": "github_issue_view", "arguments": {}}
        ]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_broken_direct_tool_dispatch_writes_failure_row(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "tool-dispatch.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    try:
        agent = _Agent(store)
        messages = []
        async with agent._turn_lifecycle():
            await agent._dispatch_tool_call(
                ToolCall(
                    id="tc-1",
                    name="broken_tool",
                    arguments={"api_key": "secret", "task": "fail"},
                ),
                features_by_tool_name={},
                known_tools={"broken_tool"},
                messages=messages,
                iteration=0,
                user_message="please fail",
                session_id="sess-1",
            )

        row = await backend.fetch_one(
            """
            SELECT agent_did, session_id, turn_id, ts, tool_name, adapter,
                   args_redacted, result_status, error_class, error_message
            FROM a2a_tool_dispatches
            WHERE agent_did=?
            """,
            ("did:test:emma",),
        )
        assert row is not None
        assert row[1] == "sess-1"
        assert str(row[2]).startswith("turn_")
        assert row[3] is not None
        assert row[4] == "broken_tool"
        assert row[5] == "TestFeature.broken_tool"
        assert json.loads(row[6])["api_key"] == "<redacted>"
        assert row[7] == "error"
        assert row[8] == "ValueError"
        assert row[9] == "broken on purpose"
        assert messages and "broken on purpose" in messages[0]["content"]
    finally:
        await store.close()


def test_redact_args_json_caps_large_payloads():
    text = redact_tool_args_json(
        {"password": "pw", "payload": {f"k{i}": "z" * 100 for i in range(100)}}
    )
    assert len(text.encode("utf-8")) <= 2048
    assert json.loads(text)["_truncated"] is True
