"""Tests for the codex sandbox-approval decline event log (#1581).

Pins:

* Schema creation is idempotent.
* The bridge handler records a typed event on each decline path
  (policy DENY, queue denied, missing queue, exception).
* The app-server records an event for default-decline RPCs
  (_DEFAULT_APPROVAL_REPLIES path).
* The operational state block renders a one-line summary on the
  agent's next turn.
* DID-scoping isolates rows in shared-backend deployments.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.codex_decline_events import (
    ensure_codex_decline_events_table,
    list_recent_declines_for_agent,
    record_decline,
)
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _backend(tmp_path):
    raw = SQLiteBackend(str(tmp_path / "decline-test.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    await ensure_codex_decline_events_table(db)
    return db


def _agent_with_db(db, *, did="did:test:emma"):
    return SimpleNamespace(
        did=did,
        _agent_name="emma",
        _raw_storage=SimpleNamespace(db=db),
        storage=None,
        features={},
    )


# ---------------------------------------------------------------------------
# Pure store contracts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_table_idempotent(tmp_path):
    db = await _backend(tmp_path)
    await ensure_codex_decline_events_table(db)  # second call no-ops


@pytest.mark.asyncio
async def test_record_and_list_round_trip(tmp_path):
    db = await _backend(tmp_path)
    await record_decline(
        db, agent_id="emma",
        request="item/commandExecution/requestApproval",
        tool="gh issue create -R O/R --title x",
        reason="policy_deny:binary:gh",
    )
    rows = await list_recent_declines_for_agent(db, agent_id="emma")
    assert len(rows) == 1
    assert rows[0].request == "item/commandExecution/requestApproval"
    assert "gh issue create" in rows[0].tool
    assert rows[0].reason == "policy_deny:binary:gh"
    assert rows[0].status == "declined"


@pytest.mark.asyncio
async def test_did_scoping_isolates_agents(tmp_path):
    """Shared-backend safety: agent A's declines must not appear in
    agent B's listing."""
    db = await _backend(tmp_path)
    await record_decline(
        db, agent_id="emma", request="r1", tool="t1", reason="policy_deny",
    )
    await record_decline(
        db, agent_id="meridian", request="r2", tool="t2", reason="auto_default",
    )
    emma_rows = await list_recent_declines_for_agent(db, agent_id="emma")
    mer_rows = await list_recent_declines_for_agent(db, agent_id="meridian")
    assert {r.tool for r in emma_rows} == {"t1"}
    assert {r.tool for r in mer_rows} == {"t2"}


@pytest.mark.asyncio
async def test_tool_field_truncated_at_200_chars(tmp_path):
    db = await _backend(tmp_path)
    big = "x" * 1000
    row = await record_decline(
        db, agent_id="emma", request="r", tool=big, reason="x",
    )
    assert len(row.tool) <= 200
    assert row.tool.endswith("...")


# ---------------------------------------------------------------------------
# Bridge handler decline-record wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_policy_deny_records_event(tmp_path):
    """Policy-gate DENY in the codex_adapter bridge must write a
    typed decline event tagged with the policy reason."""
    from kestrel_sovereign.features.computer_use.policy import (
        BinaryPolicy,
    )
    db = await _backend(tmp_path)
    agent = _agent_with_db(db)
    agent.features = {
        "SecurityFeature": SimpleNamespace(
            approval_queue=SimpleNamespace(
                request_approval=AsyncMock(return_value=(True, "auto"))
            )
        ),
        "ComputerUseFeature": SimpleNamespace(
            _binary_policy=BinaryPolicy(allow=["gh"]),
            _path_policy=None,
        ),
    }
    adapter = CodexAdapter()
    handler = adapter._make_codex_approval_handler(agent, "commandExecution")
    reply = await handler({"command": "rm -rf /"})  # rm not allow-listed
    assert reply == {"decision": "decline"}

    rows = await list_recent_declines_for_agent(db, agent_id="did:test:emma")
    assert len(rows) == 1
    assert rows[0].reason.startswith("policy_deny:binary:")
    assert "rm" in rows[0].tool


@pytest.mark.asyncio
async def test_bridge_queue_denial_records_event(tmp_path):
    """Approval queue saying 'denied' must surface as queue_denied."""
    from kestrel_sovereign.features.computer_use.policy import (
        BinaryPolicy,
    )
    db = await _backend(tmp_path)
    agent = _agent_with_db(db)
    agent.features = {
        "SecurityFeature": SimpleNamespace(
            approval_queue=SimpleNamespace(
                request_approval=AsyncMock(return_value=(False, "user_denied"))
            )
        ),
        "ComputerUseFeature": SimpleNamespace(
            _binary_policy=BinaryPolicy(allow=["gh"]),
            _path_policy=None,
        ),
    }
    adapter = CodexAdapter()
    handler = adapter._make_codex_approval_handler(agent, "commandExecution")
    reply = await handler({"command": "gh issue create -R O/R --title x"})
    assert reply == {"decision": "decline"}

    rows = await list_recent_declines_for_agent(db, agent_id="did:test:emma")
    assert len(rows) == 1
    assert rows[0].reason == "queue_denied"


@pytest.mark.asyncio
async def test_bridge_no_queue_records_event(tmp_path):
    """Missing SecurityFeature → no_approval_queue reason."""
    db = await _backend(tmp_path)
    agent = _agent_with_db(db)
    agent.features = {
        # CU present (so policy gate passes) but no SecurityFeature.
        "ComputerUseFeature": SimpleNamespace(
            _binary_policy=None,  # no policy → passes through (round 2 still declines on no_binary_policy)
            _path_policy=None,
        ),
    }
    # Actually with no _binary_policy the gate returns "no_binary_policy"
    # — that's a policy_deny variant. Use a permissive policy instead.
    from kestrel_sovereign.features.computer_use.policy import BinaryPolicy
    agent.features["ComputerUseFeature"] = SimpleNamespace(
        _binary_policy=BinaryPolicy(allow=["gh"]),
        _path_policy=None,
    )

    adapter = CodexAdapter()
    handler = adapter._make_codex_approval_handler(agent, "commandExecution")
    reply = await handler({"command": "gh issue create -R O/R --title x"})
    assert reply == {"decision": "decline"}

    rows = await list_recent_declines_for_agent(db, agent_id="did:test:emma")
    assert len(rows) == 1
    assert rows[0].reason == "no_approval_queue"


# ---------------------------------------------------------------------------
# App-server default-decline wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_server_default_decline_records_event(tmp_path):
    """The codex_app_server's _DEFAULT_APPROVAL_REPLIES path is the
    fallback decline source for RPCs the bridge intentionally doesn't
    cover (elicitation / permissions / userInput). When an audit
    agent is attached, those declines must surface as typed events."""
    db = await _backend(tmp_path)
    agent = _agent_with_db(db)

    client = CodexAppServerClient.__new__(CodexAppServerClient)
    client._server_request_handlers = {}
    client._audit_agent = None
    client.attach_audit_agent(agent)

    # Capture _send calls so we can verify the wire reply was the
    # default decline shape.
    sent = []
    client._send = lambda msg: sent.append(msg)

    await client._handle_server_request(
        7, "item/commandExecution/requestApproval", {"command": "ls /"},
    )
    assert sent[-1]["result"] == {"decision": "decline"}

    rows = await list_recent_declines_for_agent(db, agent_id="did:test:emma")
    assert len(rows) == 1
    assert rows[0].request == "item/commandExecution/requestApproval"
    assert rows[0].reason == "auto_default"


@pytest.mark.asyncio
async def test_app_server_records_action_shape_decline(tmp_path):
    """Codex review #1581 round 1 P1: ``mcpServer/elicitation/request``
    declines with ``{\"action\": \"decline\"}`` (NOT ``decision``).
    The auto-default recorder must catch this shape too."""
    db = await _backend(tmp_path)
    agent = _agent_with_db(db)

    client = CodexAppServerClient.__new__(CodexAppServerClient)
    client._server_request_handlers = {}
    client._audit_agent = None
    client.attach_audit_agent(agent)

    sent = []
    client._send = lambda msg: sent.append(msg)

    await client._handle_server_request(
        9, "mcpServer/elicitation/request", {"hint": "x"},
    )
    assert sent[-1]["result"] == {"action": "decline"}

    rows = await list_recent_declines_for_agent(db, agent_id="did:test:emma")
    assert len(rows) == 1
    assert rows[0].request == "mcpServer/elicitation/request"
    assert rows[0].reason == "auto_default"


@pytest.mark.asyncio
async def test_app_server_no_audit_agent_no_event(tmp_path):
    """Without an attached audit agent (test stubs, headless callers),
    default declines must NOT raise — they just don't get recorded."""
    db = await _backend(tmp_path)

    client = CodexAppServerClient.__new__(CodexAppServerClient)
    client._server_request_handlers = {}
    client._audit_agent = None
    sent = []
    client._send = lambda msg: sent.append(msg)

    await client._handle_server_request(
        8, "item/fileChange/requestApproval", {"fileChanges": []},
    )
    assert sent[-1]["result"] == {"decision": "decline"}

    rows = await list_recent_declines_for_agent(db, agent_id="anyone")
    assert rows == []


# ---------------------------------------------------------------------------
# Operational-state-block rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_operational_block_includes_decline_summary(tmp_path):
    """The always-on operational block must surface a decline summary
    line on the agent's next turn when the table has rows for her."""
    db = await _backend(tmp_path)
    await record_decline(
        db, agent_id="did:test:emma",
        request="item/commandExecution/requestApproval",
        tool="rm -rf /",
        reason="policy_deny:binary:rm",
    )
    agent = _agent_with_db(db)

    from kestrel_sovereign.agent.preturn_state import (
        build_operational_state_block,
    )
    block = await build_operational_state_block(agent)
    assert block is not None
    assert "OPERATIONAL STATE" in block
    assert "Codex declines" in block
    assert "policy_deny" in block


@pytest.mark.asyncio
async def test_operational_block_silent_when_no_declines(tmp_path):
    """A clean agent with no decline rows produces no decline section
    in the block."""
    db = await _backend(tmp_path)
    agent = _agent_with_db(db)

    from kestrel_sovereign.agent.preturn_state import (
        build_operational_state_block,
    )
    block = await build_operational_state_block(agent)
    # Block may be None (no restart events either) or, if restart
    # rows exist, must NOT mention codex declines.
    if block is not None:
        assert "Codex declines" not in block


@pytest.mark.asyncio
async def test_operational_block_did_scoped_declines(tmp_path):
    """Peer agents' declines must not leak into this agent's block."""
    db = await _backend(tmp_path)
    await record_decline(
        db, agent_id="did:test:meridian",
        request="r", tool="t", reason="policy_deny",
    )
    agent = _agent_with_db(db, did="did:test:emma")

    from kestrel_sovereign.agent.preturn_state import (
        build_operational_state_block,
    )
    block = await build_operational_state_block(agent)
    # Emma has zero rows; Meridian's row must not leak through.
    if block is not None:
        assert "Codex declines" not in block
