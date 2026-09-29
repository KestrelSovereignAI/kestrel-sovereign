"""Gate-2 review findings at PR #3112 head 866e457c (2026-09-29).

Two P1s, each reproduced through the real components before being fixed, and
each test written so that reverting its fix turns it red.

1. The causation chain was dropped at the inline-tool task boundary. A
   ``self_followup`` turn on the codex app-server route runs its tools on a
   reader task spawned before the turn, so ``TaskManager.create_task`` read an
   empty chain and the outbound A2A task left without lineage. The peer's
   completion then woke a depth-1 turn with no trace of the follow-up that
   caused it, and that turn could schedule another follow-up -- the
   single-hop bound evaded by one A2A round trip.

2. Stored results were redacted by task NAME. A scheduled ``schedule_list`` /
   ``schedule_self_followups`` / ``schedule_history`` run under durable
   storage writes its output, follow-up intents included, into its own
   ``task_execution_log`` row, and after a switch to a volatile mode
   ``schedule_history`` returned it because only ``self_followup`` rows were
   redacted.
"""

from __future__ import annotations

import json
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from kestrel_sdk.signals import Status
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.features.scheduler.outcome import REDACTED_RESULT_TEXT
from kestrel_sovereign.privacy import PrivacyConfig
from kestrel_sovereign.signals.sources.self_followup import (
    TASK_NAME as SELF_FOLLOWUP,
)

from tests.unit.test_inline_executor_contextvar_invariant import (
    _CodexReaderHarness,
)
from tests.unit.test_self_followup_schedule import SOURCE, _drain, _schedule

SENTINEL = "gate2-intent-XYZZY"


def _bind_inline_executor(agent, execute_named_tool):
    """Give the fixture's agent the REAL parent-turn inline executor.

    ``_make_inline_tool_executor`` and ``_capture_transition_reentry_token``
    are production ``OrchestratorEngineMixin`` code; only the tool dispatch
    they delegate to is supplied by the test.
    """
    agent._capture_transition_reentry_token = types.MethodType(
        OrchestratorEngineMixin._capture_transition_reentry_token, agent
    )
    agent.execute_named_tool = execute_named_tool
    return types.MethodType(
        OrchestratorEngineMixin._make_inline_tool_executor, agent
    )


async def _task_manager(tmp_path, agent):
    """A real TaskManager wired to the agent's REAL chain provider."""
    from kestrel_sovereign.a2a.stores import (
        SQLiteObservabilityStore,
        SQLiteSessionService,
        SQLiteTaskStore,
    )
    from kestrel_sovereign.a2a.task_manager import TaskManager
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    db_path = str(tmp_path / "a2a.db")
    manager = TaskManager(
        task_store=SQLiteTaskStore(db_path),
        session_service=SQLiteSessionService(db_path),
        observability_store=SQLiteObservabilityStore(db_path),
        # Same wiring as KestrelAgent.initialize(): the production provider
        # reads the ContextVar chain of whichever task calls create_task.
        causation_chain_provider=types.MethodType(
            KestrelAgent._provide_causation_chain, agent
        ),
    )
    await manager.initialize()
    return manager


# ---------------------------------------------------------------------------
# P1 #1 -- the causation chain crosses the inline-tool task boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a2a_sent_inline_from_a_follow_up_keeps_the_single_hop_bound(
    followup_env, tmp_path
):
    """self_followup -> inline A2A send -> completion turn -> refused.

    End to end through the real runner, dispatcher, parent inline executor,
    TaskManager, chain provider and A2A completion signal builder. The codex
    reader is spawned before either turn, so every tool runs on a task whose
    frozen context has no chain and no signal: exactly the production
    topology in which the chain was lost.
    """
    from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart
    from kestrel_sovereign.signals.sources.a2a import (
        SOURCE_NAME as A2A_COMPLETE,
        build_a2a_task_complete_registration,
        build_signal_for_completed_task,
    )

    agent, feature, runner, db, _backend = followup_env
    agent.signal_registry.register(build_a2a_task_complete_registration())
    manager = await _task_manager(tmp_path, agent)

    outbound = []
    refusals = []

    async def execute_named_tool(name, args, *, session_id, source, _capture):
        _capture["effective_args"] = args
        if name == "send_a2a_task":
            params = TaskSendParams(
                id=f"outbound-{len(outbound) + 1}",
                sessionId="session-gate2",
                message=Message(role="user", parts=[TextPart(text=args["message"])]),
                metadata={"agent_id": "PeerB"},
            )
            outbound.append(await manager.create_task(params, agent_name="PeerB"))
            return {"ok": True}
        if name == "schedule_add_deadline":
            result = await feature.schedule_add_deadline(**args)
            refusals.append(result)
            return {"ok": result.status is ToolResultStatus.OK}
        raise AssertionError(f"unexpected inline tool {name}")

    make_executor = _bind_inline_executor(agent, execute_named_tool)
    harness = _CodexReaderHarness()
    # Spawned BEFORE either turn: its frozen context carries no chain.
    await harness.ensure_started()

    async def follow_up_turn(prompt, **kwargs):
        agent.turn_prompts.append(prompt)
        async with agent._turn_lifecycle():
            executor = make_executor("")
            await harness.dispatch(
                executor, "send_a2a_task", {"message": "check CI on PR 3096"}
            )
        return "asked the peer to check CI"

    try:
        created = await _schedule(feature)
        assert created.status is ToolResultStatus.OK, created.error

        agent.process_input = follow_up_turn
        await runner._tick()
        await _drain(agent)

        assert len(outbound) == 1, "the follow-up turn did not send its A2A task"
        chain = outbound[0].metadata.get("causation_chain") or []
        assert SOURCE in {frame["source"] for frame in chain}, (
            "the outbound A2A task left without the follow-up turn's causation "
            "chain: the inline executor did not re-present it across the "
            "reader-task boundary (#3112 gate-2 P1)"
        )

        async def completion_turn(prompt, **kwargs):
            agent.turn_prompts.append(prompt)
            async with agent._turn_lifecycle():
                executor = make_executor("")
                run_at = (
                    datetime.now(timezone.utc) + timedelta(minutes=20)
                ).isoformat()
                await harness.dispatch(
                    executor,
                    "schedule_add_deadline",
                    {
                        "run_at": run_at,
                        "task_name": SELF_FOLLOWUP,
                        "args_json": json.dumps({"intent": "and merge it"}),
                    },
                )
            return "handled the completion"

        agent.process_input = completion_turn
        completed = SimpleNamespace(
            id=outbound[0].id,
            status=SimpleNamespace(
                state=SimpleNamespace(value="completed"),
                message=SimpleNamespace(parts=[SimpleNamespace(text="CI green")]),
            ),
            metadata=outbound[0].metadata,
        )
        wake = build_signal_for_completed_task(completed, target_agent=agent.did)
        result = await agent.dispatcher.dispatch_signal(wake)
    finally:
        await harness.stop()
        await manager.close()

    assert wake.source == A2A_COMPLETE
    assert result.status is Status.OK, "the completion turn itself must still run"
    assert len(refusals) == 1, "the completion turn never tried to reschedule"
    assert refusals[0].status is ToolResultStatus.ERROR, (
        "a turn descended from a follow-up via an A2A round trip scheduled "
        "another follow-up: the single-hop bound was evaded"
    )
    assert refusals[0].data.get("refused") == "self_followup_chain_ancestor"
    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?", (SELF_FOLLOWUP,)
    )
    assert len(rows) == 1, "only the original follow-up may exist"


# ---------------------------------------------------------------------------
# P1 #2 -- no stored result echoes a follow-up after a volatile switch
# ---------------------------------------------------------------------------


READ_TOOLS = ("schedule_list", "schedule_self_followups")


async def _schedule_due(feature, task_name):
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    created = await feature.schedule_add_deadline(
        run_at=run_at, task_name=task_name, args_json="{}"
    )
    assert created.status is ToolResultStatus.OK, created.error
    return created.data["task_id"]


async def _persist_read_tool_outputs(agent, feature, runner):
    """Queue a follow-up, then let scheduled read tools store their output.

    Everything runs under durable storage through the real runner, so the
    rows under test hold what production would write -- not hand-inserted
    text. ``schedule_history`` runs last, so its own row copies the earlier
    rows' results, which is the recursive door.
    """
    async with agent._turn_lifecycle():
        queued = await feature.schedule_add_deadline(
            run_at="2099-01-01T00:00:00+00:00",
            task_name=SELF_FOLLOWUP,
            args_json=json.dumps({"intent": SENTINEL}),
        )
    assert queued.status is ToolResultStatus.OK, queued.error

    agent.features = {"SchedulerFeature": feature}
    for name in READ_TOOLS:
        await _schedule_due(feature, name)
    await runner._tick()
    await _drain(agent)
    await _schedule_due(feature, "schedule_history")
    await runner._tick()
    await _drain(agent)


async def _stored_results_by_task(db):
    rows = await db.fetchall(
        """SELECT st.task_name, el.result_text
           FROM task_execution_log el
           JOIN scheduled_tasks st ON st.id = el.task_id"""
    )
    return {row[0]: row[1] or "" for row in rows}


@pytest.mark.asyncio
async def test_the_read_tool_door_is_real_under_durable_storage(followup_env):
    """Positive control: each scheduled read tool really stores the intent.

    Without this, the redaction test below could pass on rows that never held
    the sentinel in the first place.
    """
    agent, feature, runner, db, _backend = followup_env
    await _persist_read_tool_outputs(agent, feature, runner)

    stored = await _stored_results_by_task(db)
    for name in (*READ_TOOLS, "schedule_history"):
        assert SENTINEL in stored.get(name, ""), (
            f"{name} did not store the follow-up intent under durable storage; "
            "the redaction test would be vacuous"
        )

    history = await feature.schedule_history()
    assert SENTINEL in json.dumps(history.data or {}), (
        "under durable storage schedule_history must still return results"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["none", "temp", "deidentified"])
async def test_schedule_history_returns_no_follow_up_text_from_any_row(
    followup_env, storage
):
    """The ticket's test, verbatim: switch to volatile, and no row leaks."""
    agent, feature, runner, _db, _backend = followup_env
    await _persist_read_tool_outputs(agent, feature, runner)

    agent.privacy_config = PrivacyConfig(storage=storage)
    history = await feature.schedule_history()

    assert history.status is ToolResultStatus.OK
    executions = history.data["executions"]
    assert {row["task_name"] for row in executions} >= {
        *READ_TOOLS,
        "schedule_history",
    }, "every read-tool row must still be listed, only its text withheld"
    assert SENTINEL not in json.dumps(history.data), (
        f"{storage}: schedule_history returned follow-up intent text stored "
        "by a scheduled read tool (#3112 gate-2 P1)"
    )
    for row in executions:
        assert row["result_text"] in (None, "", REDACTED_RESULT_TEXT), (
            f"{row['task_name']} result was returned in a volatile mode"
        )
        assert row["status"], "status must stay visible when text is withheld"

    followups = await feature.schedule_self_followups()
    assert SENTINEL not in json.dumps(followups.data or {})


# ---------------------------------------------------------------------------
# P1 #2, reader N+1 -- the reflection status endpoint reads the same column
# ---------------------------------------------------------------------------


def _reflection_status(agent):
    from tests.unit.test_agent_runtime_endpoint_contracts import (
        _api_headers,
        _prepare_app,
        _restore_app,
    )

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    "/api/agent/reflection/status", headers=_api_headers()
                )
    finally:
        _restore_app(app, original)
    assert response.status_code == 200
    return response.json()


def _reflection_agent():
    db = MagicMock()
    db.fetchall = AsyncMock(
        return_value=[
            ("task-1", "reflect", "success", 12, "2026-09-29T12:00:00Z", SENTINEL),
        ]
    )
    agent = MagicMock()
    agent.sleep_hooks = []
    agent.features = {}
    agent._raw_storage = SimpleNamespace(db=db)
    agent.agent_id = "did:test:agent"
    return agent


def test_reflection_status_withholds_stored_results_in_a_volatile_mode():
    agent = _reflection_agent()
    agent.privacy_config = PrivacyConfig(storage="none")

    payload = _reflection_status(agent)

    assert SENTINEL not in json.dumps(payload), (
        "the reflection endpoint returned a stored result in a volatile mode"
    )
    assert payload["recent_executions"][0]["result_preview"] == REDACTED_RESULT_TEXT
    assert payload["recent_executions"][0]["status"] == "success"


def test_reflection_status_still_returns_results_under_durable_storage():
    agent = _reflection_agent()
    agent.privacy_config = PrivacyConfig(storage="full")

    payload = _reflection_status(agent)

    assert payload["recent_executions"][0]["result_preview"] == SENTINEL
