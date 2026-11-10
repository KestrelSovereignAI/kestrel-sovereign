"""#4002 acceptance: an agent booted from a released database audits its tool calls.

#3995 added ``a2a_tool_dispatches.dispatch_id`` and ``parent_dispatch_id`` to a
CORE store without moving ``NativeStoreSchema.CORE``. Startup skips the CORE
initializers for a database already at that revision, so no existing database
received the columns, and every audit row failed to land. #3999 then made the
audit required, and every tool call failed. Every test database was born on
the new code, so none of them saw it.

These tests start from the committed, content-free snapshot of a released
Core's fresh agent database (``tests/shared/agent_store_schema.py``), boot a
real agent on it, and run one real tool call through the orchestrator. Only
the model is scripted.

- 0.63.0 is the last release before #3995: ``core=5`` without the columns.
- 0.64.1 is ``core=5`` born with the columns, the shape every database the
  operator repaired by hand also has.
"""

from __future__ import annotations

import sqlite3

import pytest

from kestrel_sovereign.a2a.native_schema import NativeStoreSchema
from kestrel_sovereign.features.health.checks import check_tool_dispatch_audit
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from tests.shared.agent_store_schema import capture, load, snapshot_path
from tests.shared.booted_agent import booted_agent

DONE = "The wait list is checked."

#: release -> whether its fresh database already has the #3995 columns.
RELEASES = {"0.63.0": False, "0.64.1": True}


def _dispatch_columns(connection: sqlite3.Connection) -> set[str]:
    return {
        row[1]
        for row in connection.execute("PRAGMA table_info(a2a_tool_dispatches)")
    } & {"dispatch_id", "parent_dispatch_id"}


def _core_revision(connection: sqlite3.Connection) -> int:
    (revision,) = connection.execute(
        "SELECT revision FROM a2a_native_store_schema WHERE component = 'core'"
    ).fetchone()
    return revision


class _OneToolModel:
    """Calls ``wait_status`` once, then answers."""

    def __init__(self) -> None:
        self.tool_results: list[str] = []

    def _respond(self, kwargs) -> LLMResponse:
        messages = kwargs.get("messages") or [{}]
        if messages[-1].get("role") != "tool":
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="status-1", name="wait_status", arguments={"limit": 5})
                ],
            )
        self.tool_results.append(messages[-1]["content"])
        return LLMResponse(content=DONE)

    async def generate_with_messages(self, **kwargs):
        return self._respond(kwargs)

    async def stream_with_tool_detection(self, **kwargs):
        yield self._respond(kwargs)


@pytest.mark.parametrize("release", sorted(RELEASES))
def test_released_snapshot_is_the_shape_it_claims(tmp_path, release):
    """The fixture still holds the release's real revision rows and table shape."""
    db_path = tmp_path / "kestrel_prime.db"
    load(db_path, snapshot_path(release))
    connection = sqlite3.connect(db_path)
    try:
        assert _core_revision(connection) == 5
        assert _dispatch_columns(connection) == (
            {"dispatch_id", "parent_dispatch_id"} if RELEASES[release] else set()
        )
    finally:
        connection.close()


@pytest.mark.parametrize("release", sorted(RELEASES))
def test_released_snapshot_reloads_to_itself(tmp_path, release):
    """Loading a snapshot and capturing it again changes nothing."""
    db_path = tmp_path / "kestrel_prime.db"
    load(db_path, snapshot_path(release))
    assert capture(db_path, release=release) == snapshot_path(release).read_text(
        encoding="utf-8"
    )


def test_a_snapshot_keeps_only_revision_rows(tmp_path):
    """Content rows are dropped; revision rows and every object are kept."""
    db_path = tmp_path / "source.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            "CREATE TABLE a2a_protocol_schema (singleton INTEGER PRIMARY KEY, revision INTEGER);"
            "INSERT INTO a2a_protocol_schema VALUES (1, 1);"
            "CREATE TABLE graph_nodes (id TEXT PRIMARY KEY, properties TEXT);"
            "INSERT INTO graph_nodes VALUES ('did:secret', '{\"name\": \"private\"}');"
            "CREATE INDEX idx_nodes ON graph_nodes(properties);"
            "CREATE VIRTUAL TABLE notes USING fts5(body);"
            "INSERT INTO notes VALUES ('private note');"
        )
    finally:
        connection.close()
    text = capture(db_path, release="test")
    assert "private" not in text and "did:secret" not in text
    assert 'INSERT INTO a2a_protocol_schema ("singleton", "revision") VALUES (1, 1);' in text

    reloaded = tmp_path / "reloaded.db"
    snapshot = tmp_path / "snapshot.sql"
    snapshot.write_text(text, encoding="utf-8")
    load(reloaded, snapshot)
    connection = sqlite3.connect(reloaded)
    try:
        assert {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master")
        } >= {"a2a_protocol_schema", "graph_nodes", "idx_nodes", "notes"}
        assert connection.execute("SELECT COUNT(*) FROM graph_nodes").fetchone() == (0,)
        assert connection.execute("SELECT * FROM a2a_protocol_schema").fetchall() == [(1, 1)]
    finally:
        connection.close()
    with pytest.raises(FileExistsError):
        load(reloaded, snapshot)


@pytest.mark.parametrize("release", sorted(RELEASES))
async def test_agent_booted_from_a_released_database_audits_a_real_tool_call(
    tmp_path, monkeypatch, release
):
    # The test instance has no interactive approver; approve the ASK-level
    # tool the way a headless test instance is told to, so it executes.
    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    agent_dir = tmp_path / "agent"
    model = _OneToolModel()
    async with booted_agent(
        agent_dir,
        name="Upgraded",
        provenance="test:released_agent_database_upgrade",
        features=set(),
        released_schema=snapshot_path(release),
    ) as agent:
        monkeypatch.setattr(
            agent.llm_service, "generate_with_messages", model.generate_with_messages
        )
        monkeypatch.setattr(
            agent.llm_service,
            "stream_with_tool_detection",
            model.stream_with_tool_detection,
        )
        reply = await agent.process_input("Check my waits.", session_id="upgrade")
        health = await check_tool_dispatch_audit(agent)
        did = agent.did

    assert DONE in reply, reply
    assert len(model.tool_results) == 1, model.tool_results

    connection = sqlite3.connect(agent_dir / "kestrel_prime.db")
    try:
        assert _core_revision(connection) == NativeStoreSchema.CORE.value[1]
        assert _dispatch_columns(connection) == {"dispatch_id", "parent_dispatch_id"}
        rows = connection.execute(
            "SELECT agent_did, tool_name, result_status, dispatch_id "
            "FROM a2a_tool_dispatches"
        ).fetchall()
    finally:
        connection.close()
    assert [(row[0], row[1]) for row in rows] == [(did, "wait_status")], rows
    assert rows[0][2] == "success" and rows[0][3], rows

    assert health["status"] == "pass", health
    assert health["details"]["failures"] == 0
    assert health["details"]["last_success_at"] is not None
