"""Focused contract tests for agent runtime/status endpoints."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from kestrel_sovereign.a2a.types import Artifact, Message, Task, TaskState, TaskStatus, TextPart


def _prepare_app(agent):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


def _api_headers():
    return {"X-API-Key": "test-key"}


class _CounterStub:
    def __init__(self, context_limit=4000):
        self._context_limit = context_limit

    def count(self, text):
        return len(text)

    def get_context_limit(self):
        return self._context_limit


def _breakdown_payload(total_measured, total_budget, response_reserve=1024, *, sections_overrides=None):
    """Build a ``measure_context_breakdown``-shaped dict for endpoint tests.

    The endpoint cares about ``total_measured`` / ``total_budget`` /
    ``utilization_percent`` / ``response_reserve`` and the ``sections``
    block; build a minimal valid shape and let callers override.
    """
    util = min(100.0, (total_measured / total_budget * 100.0) if total_budget else 0.0)
    sections = {
        "system": {"tokens": 0, "budget": 0, "subsections": []},
        "tools": {"tokens": 0, "count": 0, "estimated": True, "estimation_method": ""},
        "history": {
            "tokens": total_measured,
            "budget": total_budget,
            "messages_total": 0,
            "messages_kept_after_pruning": 0,
            "raw_tokens": total_measured,
        },
        "episodes": {"tokens": 0, "budget": 0, "count": 0, "threshold": 20},
        "memories": {"tokens": 0, "budget": 0, "count": 0, "wired": False, "excluded": False},
        "rag": {"tokens": 0, "budget": 0, "chunks": 0, "estimated": True, "estimation_method": "", "skipped": True, "excluded": False},
        "dynamic_context_overhead": {"tokens": 0, "applies_when": "memories or rag included", "applied": False},
    }
    if sections_overrides:
        for k, v in sections_overrides.items():
            sections[k] = {**sections.get(k, {}), **v}
    return {
        "model": "gpt-5",
        "context_limit": total_budget + response_reserve,
        "response_reserve": response_reserve,
        "total_budget": total_budget,
        "total_measured": total_measured,
        "utilization_percent": round(util, 1),
        "budget_summary": {},
        "sections": sections,
        "notes": [],
        "_artifacts": {"system_prompt": "", "formatted_history": [], "dynamic_user_context": ""},
    }


def test_context_status_reports_whole_window_utilization_and_warning_band():
    """#1310: the pill % now reflects whole-window utilization (sum of
    all measured sections / total budget), not history-only. Status
    bands key off the whole-window figure. Codex-reviewed change to
    the endpoint contract (PR #1306, Emma-acked)."""
    history = [
        {"content": "a" * 1000},
        {"content": "b" * 1200},
        {"content": "c" * 700},
    ]
    agent = MagicMock()
    agent.get_current_model = MagicMock(return_value="gpt-5")
    agent.storage.get_conversation_history = AsyncMock(return_value=history)
    agent.get_constitution = lambda: ""
    agent.llm_service = None
    agent.tool_registry = None

    # Endpoint now delegates to ContextBuilder.measure_context_breakdown
    # (#1308 source of truth). Stub it to a payload that produces ~97%
    # utilization so the critical-band logic kicks in.
    ctx_builder = MagicMock()
    ctx_builder.measure_context_breakdown = AsyncMock(
        return_value=_breakdown_payload(2900, 2976)
    )
    agent.context_builder = ctx_builder

    app, original = _prepare_app(agent)
    try:
        with patch(
            "kestrel_sovereign.agent.token_counter.get_token_counter",
            return_value=_CounterStub(context_limit=4000),
        ):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    response = client.get(
                        "/api/agent/context-status?session_id=session-1",
                        headers=_api_headers(),
                    )
        assert response.status_code == 200
        payload = response.json()
        assert payload["model"] == "gpt-5"
        assert payload["message_count"] == 3
        assert payload["total_tokens"] == 2900
        assert payload["context_limit"] == 4000
        assert payload["response_reserve"] == 1024
        assert payload["total_budget"] == 2976
        assert payload["status"] == "critical"
        assert payload["compression_recommended"] is True
        assert "compression strongly recommended" in payload["warnings"][0]
        # New for #1310: full breakdown block surfaced for the popup.
        assert "breakdown" in payload
        assert payload["breakdown"]["total_measured"] == 2900
        assert "sections" in payload["breakdown"]
        # Auto-detection of legacy silent-prune (Emma 2026-05-20
        # hardening): unconditionally True until #1311 ships.
        assert payload["silently_pruned_path_active"] is True
        agent.storage.get_conversation_history.assert_awaited_once_with(
            limit=10000,
            session_id="session-1",
        )
        # ``include_rag=False`` is the cheap-poll default.
        ctx_builder.measure_context_breakdown.assert_awaited_once()
        kw = ctx_builder.measure_context_breakdown.call_args.kwargs
        assert kw["include_rag"] is False
    finally:
        _restore_app(app, original)


def test_context_status_full_query_param_runs_rag():
    """``?full=true`` makes the endpoint pass ``include_rag=True`` to
    the breakdown — the on-demand call the popup makes when it opens."""
    history = [{"content": "x"}]
    agent = MagicMock()
    agent.get_current_model = MagicMock(return_value="gpt-5")
    agent.storage.get_conversation_history = AsyncMock(return_value=history)
    agent.get_constitution = lambda: ""
    agent.llm_service = None
    agent.tool_registry = None

    ctx_builder = MagicMock()
    ctx_builder.measure_context_breakdown = AsyncMock(
        return_value=_breakdown_payload(100, 2976)
    )
    agent.context_builder = ctx_builder

    app, original = _prepare_app(agent)
    try:
        with patch(
            "kestrel_sovereign.agent.token_counter.get_token_counter",
            return_value=_CounterStub(context_limit=4000),
        ):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.get(
                        "/api/agent/context-status?session_id=session-1&full=true",
                        headers=_api_headers(),
                    )
        assert resp.status_code == 200
        kw = ctx_builder.measure_context_breakdown.call_args.kwargs
        assert kw["include_rag"] is True
    finally:
        _restore_app(app, original)


def test_context_status_full_path_uses_last_user_turn_as_rag_query():
    """Codex round 1 P2: ``full=true`` must seed the RAG query from
    the most recent user turn so the figure approximates what the
    next LLM turn would see — otherwise we overstate accuracy.
    """
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "<user_input>\nlatest user question\n</user_input>"},
        {"role": "assistant", "content": "last answer"},
    ]
    agent = MagicMock()
    agent.get_current_model = MagicMock(return_value="gpt-5")
    agent.storage.get_conversation_history = AsyncMock(return_value=history)
    agent.get_constitution = lambda: ""
    agent.llm_service = None
    agent.tool_registry = None

    ctx_builder = MagicMock()
    ctx_builder.measure_context_breakdown = AsyncMock(
        return_value=_breakdown_payload(50, 2976)
    )
    agent.context_builder = ctx_builder

    app, original = _prepare_app(agent)
    try:
        with patch(
            "kestrel_sovereign.agent.token_counter.get_token_counter",
            return_value=_CounterStub(context_limit=4000),
        ):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.get(
                        "/api/agent/context-status?session_id=session-1&full=true",
                        headers=_api_headers(),
                    )
        assert resp.status_code == 200
        kw = ctx_builder.measure_context_breakdown.call_args.kwargs
        # The latest user turn is wrapped in sent-form; the endpoint
        # unwraps it via ``extract_raw_user_content``. Either form
        # (raw or unwrapped "latest user question") is acceptable —
        # the point is it's not empty.
        assert kw["query"], "full=true must seed RAG query from latest user turn"
        assert "latest user question" in kw["query"]
    finally:
        _restore_app(app, original)


def test_context_status_full_path_labels_rag_when_no_user_turn_available():
    """Codex round 1 P2: when ``full=true`` cannot find a recent user
    turn for the query, the response annotates the RAG section so the
    popup can label it ``estimated with no current query`` instead of
    pretending the figure is representative."""
    history = [{"role": "assistant", "content": "only an assistant turn"}]
    agent = MagicMock()
    agent.get_current_model = MagicMock(return_value="gpt-5")
    agent.storage.get_conversation_history = AsyncMock(return_value=history)
    agent.get_constitution = lambda: ""
    agent.llm_service = None
    agent.tool_registry = None

    ctx_builder = MagicMock()
    ctx_builder.measure_context_breakdown = AsyncMock(
        return_value=_breakdown_payload(50, 2976)
    )
    agent.context_builder = ctx_builder

    app, original = _prepare_app(agent)
    try:
        with patch(
            "kestrel_sovereign.agent.token_counter.get_token_counter",
            return_value=_CounterStub(context_limit=4000),
        ):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.get(
                        "/api/agent/context-status?session_id=session-1&full=true",
                        headers=_api_headers(),
                    )
        payload = resp.json()
        rag = payload["breakdown"]["sections"]["rag"]
        assert "query_used_label" in rag
        assert "no recent user turn" in rag["query_used_label"]
    finally:
        _restore_app(app, original)


def test_context_status_idle_shape_includes_silently_pruned_flag():
    """Idle path (no session_id) must still surface the flags the
    popup expects, so a session-less open does not throw on missing
    keys."""
    agent = MagicMock()
    agent.get_current_model = MagicMock(return_value="gpt-5")
    agent.storage.get_conversation_history = AsyncMock(
        side_effect=AssertionError("must not be called for idle path")
    )
    agent.context_builder = MagicMock()
    app, original = _prepare_app(agent)
    try:
        with patch(
            "kestrel_sovereign.agent.token_counter.get_token_counter",
            return_value=_CounterStub(context_limit=4000),
        ):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    resp = client.get(
                        "/api/agent/context-status",
                        headers=_api_headers(),
                    )
        payload = resp.json()
        assert payload["status"] == "idle"
        assert payload["breakdown"] is None
        assert payload["silently_pruned_path_active"] is False
    finally:
        _restore_app(app, original)


def test_context_status_returns_idle_shape_when_no_session_id():
    """Regression for #713.  Without a session_id the endpoint used to fall
    through to ``storage.get_conversation_history(session_id=None)``, which
    returns the agent's aggregate history across ALL sessions.  That made
    the chat-footer indicator show the cross-session total (e.g. "472 msgs
    · 100% Compress") on a fresh empty chat pane where no conversation was
    active.  The fixed endpoint returns an idle/zeroed shape and never
    reads storage in this case.
    """
    agent = MagicMock()
    agent.get_current_model = MagicMock(return_value="gpt-5")
    # Make this very loud if the code path ever tries to load history:
    # the assertion below pins that storage was NOT consulted.
    agent.storage.get_conversation_history = AsyncMock(
        side_effect=AssertionError(
            "storage must not be queried when session_id is absent"
        )
    )
    agent.context_builder = MagicMock()

    app, original = _prepare_app(agent)
    try:
        with patch(
            "kestrel_sovereign.agent.token_counter.get_token_counter",
            return_value=_CounterStub(context_limit=4000),
        ):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    response = client.get(
                        "/api/agent/context-status",
                        headers=_api_headers(),
                    )
        assert response.status_code == 200
        payload = response.json()
        # Idle contract: zero counters, no warnings, status=idle.
        assert payload["message_count"] == 0
        assert payload["total_tokens"] == 0
        assert payload["utilization_percent"] == 0.0
        assert payload["compression_recommended"] is False
        assert payload["status"] == "idle"
        assert payload["warnings"] == []
        # Model / limits should still reflect the agent's configuration so
        # the UI can decide what budget to show when a session eventually
        # IS selected.
        assert payload["model"] == "gpt-5"
        assert payload["context_limit"] == 4000
        assert payload["response_reserve"] == 1024
        assert payload["total_budget"] == 4000 - 1024
        # Double-check: storage wasn't touched.
        agent.storage.get_conversation_history.assert_not_awaited()
    finally:
        _restore_app(app, original)


def test_context_status_returns_idle_shape_for_empty_session_id():
    """Same guard as above, but for an explicitly-empty session_id string
    (some clients send ``?session_id=`` rather than omit the param).  That
    should also take the idle path, not leak aggregate counts.
    """
    agent = MagicMock()
    agent.get_current_model = MagicMock(return_value="gpt-5")
    agent.storage.get_conversation_history = AsyncMock(
        side_effect=AssertionError(
            "storage must not be queried for empty session_id"
        )
    )
    agent.context_builder = MagicMock()

    app, original = _prepare_app(agent)
    try:
        with patch(
            "kestrel_sovereign.agent.token_counter.get_token_counter",
            return_value=_CounterStub(context_limit=4000),
        ):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    response = client.get(
                        "/api/agent/context-status?session_id=",
                        headers=_api_headers(),
                    )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "idle"
        assert payload["message_count"] == 0
        agent.storage.get_conversation_history.assert_not_awaited()
    finally:
        _restore_app(app, original)


def test_reflection_status_filters_scheduler_tasks_and_serializes_execution_history():
    from kestrel_sdk.tools.result import ToolResult

    scheduler = MagicMock()
    # SchedulerFeature.schedule_list returns a ToolResult envelope
    # post-#1061 wave 8; the .data dict carries the legacy
    # {"tasks": [...]} shape the endpoint reads.
    scheduler.schedule_list = AsyncMock(
        return_value=ToolResult.ok(
            confirmation="listed 3 tasks",
            data={
                "tasks": [
                    {"task_name": "reflect", "schedule": "daily"},
                    {"task_name": "training_cycle", "schedule": "hourly"},
                    {"task_name": "backup", "schedule": "daily"},
                ],
            },
        )
    )
    db = MagicMock()
    db.fetchall = AsyncMock(
        return_value=[
            ("task-1", "reflect", "completed", 1234, "2026-03-19T12:00:00Z", "All good"),
            ("task-2", "training_cycle", "failed", 456, "2026-03-18T12:00:00Z", "Needs review"),
        ]
    )
    agent = MagicMock()
    agent.reflection_hook = object()
    agent.features = {"SchedulerFeature": scheduler}
    agent._raw_storage = SimpleNamespace(db=db)
    agent.agent_id = "did:test:agent"

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/agent/reflection/status", headers=_api_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["reflection_hook_active"] is True
        assert [task["task_name"] for task in payload["scheduled_tasks"]] == ["reflect", "training_cycle"]
        assert payload["recent_executions"][0]["task_id"] == "task-1"
        assert payload["recent_executions"][1]["task_name"] == "training_cycle"
    finally:
        _restore_app(app, original)


def test_tasks_endpoint_filters_by_status_and_rejects_invalid_values():
    working_task = Task(
        id="task-1",
        status=TaskStatus(
            state=TaskState.WORKING,
            message=Message(role="agent", parts=[TextPart(text="still working")]),
        ),
        artifacts=[],
        metadata={"agent_id": "did:test:agent", "skill": "reflect"},
    )
    completed_task = Task(
        id="task-2",
        status=TaskStatus(state=TaskState.COMPLETED),
        artifacts=[Artifact(parts=[TextPart(text="done")])],
        metadata={"agent_id": "did:test:agent", "skill": "deliver"},
    )
    task_store = MagicMock()
    task_store.list_tasks = AsyncMock(return_value=[working_task, completed_task])
    task_manager = MagicMock(task_store=task_store)
    agent = MagicMock(task_manager=task_manager)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                filtered_response = client.get("/api/agent/tasks?status=working&limit=25", headers=_api_headers())
                invalid_response = client.get("/api/agent/tasks?status=bogus", headers=_api_headers())
        assert filtered_response.status_code == 200
        filtered = filtered_response.json()
        assert filtered["total"] == 1
        assert filtered["tasks"][0]["id"] == "task-1"
        assert filtered["tasks"][0]["status"] == "working"
        assert filtered["tasks"][0]["message"] == "still working"
        assert filtered["tasks"][0]["agent_id"] == "did:test:agent"
        assert filtered["tasks"][0]["skill"] == "reflect"
        assert filtered["tasks"][0]["artifacts_count"] == 0
        task_store.list_tasks.assert_awaited_with(limit=25)
        assert invalid_response.status_code == 400
        assert "Invalid status" in invalid_response.json()["detail"]
    finally:
        _restore_app(app, original)


def test_task_detail_endpoint_returns_task_with_artifacts():
    task = Task(
        id="task-42",
        status=TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(role="agent", parts=[TextPart(text="all done")]),
        ),
        artifacts=[
            Artifact(name="summary", parts=[TextPart(text="summary body")]),
            Artifact(name="notes", parts=[TextPart(text="notes body")]),
        ],
        metadata={"agent_id": "did:test:agent", "skill": "deliver"},
    )
    task_store = MagicMock()
    task_store.get = AsyncMock(return_value=task)
    task_manager = MagicMock(task_store=task_store)
    agent = MagicMock(task_manager=task_manager)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/agent/tasks/task-42", headers=_api_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["id"] == "task-42"
        assert payload["status"] == "completed"
        assert payload["message"] == "all done"
        assert len(payload["artifacts"]) == 2
        assert payload["artifacts"][0]["name"] == "summary"
        assert payload["metadata"]["skill"] == "deliver"
        task_store.get.assert_awaited_once_with("task-42")
    finally:
        _restore_app(app, original)


def test_task_detail_endpoint_returns_404_when_task_missing():
    task_store = MagicMock()
    task_store.get = AsyncMock(return_value=None)
    task_manager = MagicMock(task_store=task_store)
    agent = MagicMock(task_manager=task_manager)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/agent/tasks/missing", headers=_api_headers())
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    finally:
        _restore_app(app, original)


def test_task_detail_endpoint_returns_404_when_task_manager_absent():
    agent = MagicMock(spec=[])  # no task_manager attribute

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/agent/tasks/anything", headers=_api_headers())
        assert response.status_code == 404
        assert "TaskManager not available" in response.json()["detail"]
    finally:
        _restore_app(app, original)


def test_health_endpoints_return_feature_status_and_run_once():
    """Regression for #753 — /agent/health/* is the liveness-probe surface
    (not to be confused with /agent/heartbeat/* which is the LLM self-check).
    Routes to the HealthFeature instance on the agent."""
    health_feature = MagicMock()
    health_feature.__class__.__name__ = "HealthFeature"
    health_feature.get_status = MagicMock(return_value={"enabled": True, "interval_seconds": 60})
    health_feature.run_once = AsyncMock(return_value={"status": "healthy", "checks": []})

    agent = MagicMock()
    agent.features = {"HealthFeature": health_feature}

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                status_resp = client.get("/api/agent/health/status", headers=_api_headers())
                trigger_resp = client.post("/api/agent/health/trigger", headers=_api_headers())
        assert status_resp.status_code == 200
        assert status_resp.json()["enabled"] is True
        assert status_resp.json()["interval_seconds"] == 60
        assert trigger_resp.status_code == 200
        assert trigger_resp.json()["status"] == "healthy"
        health_feature.run_once.assert_awaited_once()
    finally:
        _restore_app(app, original)


def test_health_endpoints_return_404_when_feature_missing():
    """Agents without HealthFeature loaded should return a 404 from
    /agent/health/trigger and a shaped-disabled payload from /status."""
    agent = MagicMock()
    agent.features = {}  # No HealthFeature

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                status_resp = client.get("/api/agent/health/status", headers=_api_headers())
                trigger_resp = client.post("/api/agent/health/trigger", headers=_api_headers())
        assert status_resp.status_code == 200
        assert status_resp.json()["enabled"] is False
        assert trigger_resp.status_code == 404
    finally:
        _restore_app(app, original)


def test_heartbeat_endpoints_cover_disabled_status_success_and_error_paths():
    disabled_agent = MagicMock(heartbeat_runner=None)
    result = SimpleNamespace(to_dict=lambda: {"ok": True, "checks": 1})
    good_runner = MagicMock()
    good_runner.get_status = MagicMock(return_value={"enabled": True, "last_result": "ok"})
    good_runner.run_once = AsyncMock(return_value=result)
    failing_runner = MagicMock()
    failing_runner.run_once = AsyncMock(side_effect=RuntimeError("boom"))

    app, original = _prepare_app(disabled_agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                disabled_status = client.get("/api/agent/heartbeat/status", headers=_api_headers())
                disabled_trigger = client.post("/api/agent/heartbeat/trigger", headers=_api_headers())
        assert disabled_status.status_code == 200
        assert disabled_status.json() == {"enabled": False, "message": "Heartbeat not configured"}
        assert disabled_trigger.status_code == 404
    finally:
        _restore_app(app, original)

    app, original = _prepare_app(MagicMock(heartbeat_runner=good_runner))
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                status_response = client.get("/api/agent/heartbeat/status", headers=_api_headers())
                trigger_response = client.post("/api/agent/heartbeat/trigger", headers=_api_headers())
        assert status_response.status_code == 200
        assert status_response.json()["enabled"] is True
        assert trigger_response.status_code == 200
        assert trigger_response.json() == {"ok": True, "checks": 1}
    finally:
        _restore_app(app, original)

    app, original = _prepare_app(MagicMock(heartbeat_runner=failing_runner))
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                error_response = client.post("/api/agent/heartbeat/trigger", headers=_api_headers())
        assert error_response.status_code == 500
        assert error_response.json()["detail"] == "Error triggering heartbeat."
    finally:
        _restore_app(app, original)
