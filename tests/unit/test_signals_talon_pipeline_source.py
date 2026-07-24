"""Tests for the ``talon_pipeline_dispatch`` signal source.

Covers: registration shape, fail-closed param validation, and the dispatch
invocation shape against a fake coordinator (claim vs iterate, wait vs
no-wait, terminal failure/timeout fail closed), plus one round-trip through
the real SignalDispatcher mirroring the workflow-rescue source tests.
"""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sdk.signals import (
    Signal,
    SignalMode,
    Status,
    Urgency,
    Visibility,
)
from kestrel_sdk.tools import ToolResult
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.talon_pipeline import (
    DEFAULT_WAIT_TIMEOUT_SECONDS,
    SOURCE_NAME,
    build_talon_pipeline_dispatch_registration,
    make_talon_pipeline_dispatch_handler,
    register_talon_pipeline_source,
    validate_pipeline_params,
)
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeWaitRegistry:
    """Records wait calls and returns a canned engine ToolResult."""

    def __init__(self, result: ToolResult):
        self.result = result
        self.calls: list = []

    async def wait(self, ref: str, *, timeout_seconds: int, **_kw) -> ToolResult:
        self.calls.append((ref, timeout_seconds))
        return self.result


class _FakeAgentWithWait:
    def __init__(self, wait_registry=None):
        self.wait_registry = wait_registry


class _FakeCoordinator:
    """Stands in for TalonCoordinatorFeature at the dispatch seam."""

    def __init__(self, dispatch_result=None, wait_result=None, log_tail=""):
        self.dispatch_result = dispatch_result or {
            "dispatched": True,
            "method": "cli_background",
            "job_id": "job123",
            "log_path": "/tmp/job123.log",
        }
        self.dispatch_calls: list = []
        self._log_tail = log_tail
        self._jobs = {"job123": {"log_path": "/tmp/job123.log"}}
        self.wait_registry = (
            _FakeWaitRegistry(wait_result) if wait_result is not None else None
        )
        self.agent = _FakeAgentWithWait(self.wait_registry)

    async def dispatch_pipeline(self, **kwargs):
        self.dispatch_calls.append(kwargs)
        return self.dispatch_result

    @staticmethod
    def _tail_job_log(path, lines=20):  # noqa: ARG004 - signature parity
        return ""


def _complete_wait(pr_url: str = "") -> ToolResult:
    return ToolResult.ok(
        confirmation="done",
        data={
            "status": "complete",
            "returncode": 0,
            "completed_at": "2026-07-09T00:00:00+00:00",
            "waited_seconds": 12,
            "timed_out": False,
        },
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registration_shape():
    reg = build_talon_pipeline_dispatch_registration(_FakeCoordinator())
    assert reg.name == SOURCE_NAME == "talon_pipeline_dispatch"
    assert reg.default_mode is SignalMode.ACTION
    assert reg.allowed_modes == frozenset({SignalMode.ACTION})
    assert reg.handler is not None
    assert reg.log_redaction is not None
    # Schema rejects non-dict payloads.
    with pytest.raises(ValueError):
        reg.schema(["not", "a", "dict"])
    assert reg.schema({"repo": "o/r"}) == {"repo": "o/r"}


def test_register_is_idempotent():
    registry = SourceRegistry()
    coordinator = _FakeCoordinator()
    assert register_talon_pipeline_source(registry, coordinator) is True
    assert registry.get(SOURCE_NAME) is not None
    # Second call skips the pre-existing source.
    assert register_talon_pipeline_source(registry, coordinator) is False


def test_register_no_registry_is_noop():
    assert register_talon_pipeline_source(None, _FakeCoordinator()) is False


# ---------------------------------------------------------------------------
# Param validation (fail closed)
# ---------------------------------------------------------------------------


def test_validate_requires_repo():
    with pytest.raises(ValueError, match="repo"):
        validate_pipeline_params({"issue": 5})


@pytest.mark.parametrize("payload", [
    {"repo": "o/r"},                                # neither issue nor pr
    {"repo": "o/r", "issue": 1, "pr": 2},           # both, no explicit mode
    {"repo": "o/r", "issue": 1, "mode": "yolo"},    # unknown mode
    {"repo": "o/r", "mode": "claim"},               # claim without issue
    {"repo": "o/r", "mode": "iterate"},             # iterate without pr
    {"repo": "o/r", "issue": 0},                    # non-positive number
    {"repo": "o/r", "issue": "not-a-number"},
    {"repo": "o/r", "issue": True},                 # bool is not an issue no.
    {"repo": "o/r", "issue": 1, "self_review": "maybe"},
    # Repo shape (machine-driven surface — strict owner/name):
    {"repo": "no-slash", "issue": 1},               # missing owner/name split
    {"repo": "o/r/extra", "issue": 1},              # too many segments
    {"repo": "-dash/repo", "issue": 1},             # leading dash (owner)
    {"repo": "owner/-repo", "issue": 1},            # leading dash (name)
    {"repo": "o wner/repo", "issue": 1},            # whitespace inside
    {"repo": "owner/", "issue": 1},                 # empty name segment
    # Wait ceiling: run_wait_loop hard-rejects > MAX_HANDLE_WAIT_SECONDS,
    # so validation must catch it first with an actionable message.
    {"repo": "o/r", "issue": 1, "wait_timeout_seconds": 3601},
])
def test_validate_fails_closed(payload):
    with pytest.raises(ValueError):
        validate_pipeline_params(payload)


def test_validate_wait_timeout_ceiling_message_names_limit():
    with pytest.raises(ValueError, match="3600"):
        validate_pipeline_params(
            {"repo": "o/r", "issue": 1, "wait_timeout_seconds": 7200}
        )


def test_validate_accepts_self_and_ceiling_timeout():
    params = validate_pipeline_params(
        {"repo": "self", "issue": 1, "wait_timeout_seconds": 3600}
    )
    assert params["repo"] == "self"
    assert params["wait_timeout_seconds"] == 3600


# Strict numeric typing (codex P2): {"pr": 12.9} must never silently
# truncate to 12 and dispatch the irreversible pipeline against the wrong
# PR. Only actual ints (bools excluded) or digit-only strings are accepted;
# floats are rejected even when integral.


@pytest.mark.parametrize("key", ["issue", "pr", "wait_timeout_seconds"])
@pytest.mark.parametrize("bad", [12.9, 12.0, True, "12.9", "abc"])
def test_numeric_params_reject_non_integers(key, bad):
    payload = {"repo": "o/r", key: bad}
    if key == "wait_timeout_seconds":
        payload["issue"] = 1  # valid target so only the timeout is at fault
    with pytest.raises(ValueError, match=key):
        validate_pipeline_params(payload)


@pytest.mark.parametrize("key,mode", [("issue", "claim"), ("pr", "iterate")])
@pytest.mark.parametrize("good", [12, "12"])
def test_numeric_params_accept_ints_and_digit_strings(key, mode, good):
    params = validate_pipeline_params({"repo": "o/r", key: good})
    assert params[key] == 12
    assert params["mode"] == mode


@pytest.mark.parametrize("good", [12, "12"])
def test_wait_timeout_accepts_ints_and_digit_strings(good):
    params = validate_pipeline_params(
        {"repo": "o/r", "issue": 1, "wait_timeout_seconds": good}
    )
    assert params["wait_timeout_seconds"] == 12


def test_validate_infers_claim_from_issue():
    params = validate_pipeline_params({"repo": "o/r", "issue": "42"})
    assert params["mode"] == "claim"
    assert params["issue"] == 42
    assert params["pr"] is None
    assert params["wait"] is True
    assert params["wait_timeout_seconds"] == DEFAULT_WAIT_TIMEOUT_SECONDS


def test_validate_infers_iterate_from_pr():
    params = validate_pipeline_params(
        {"repo": "o/r", "pr": 7, "self_review": True,
         "demo_check": "yes", "eye_check": False, "wait": "no",
         "wait_timeout_seconds": 120}
    )
    assert params["mode"] == "iterate"
    assert params["pr"] == 7
    assert params["issue"] is None
    assert params["self_review"] is True
    assert params["demo_check"] is True
    assert params["eye_check"] is False
    assert params["wait"] is False
    assert params["wait_timeout_seconds"] == 120


def test_validate_explicit_mode_disambiguates_both_numbers():
    params = validate_pipeline_params(
        {"repo": "o/r", "issue": 1, "pr": 2, "mode": "iterate"}
    )
    assert params["mode"] == "iterate"
    assert params["pr"] == 2
    # Claim-only target is dropped so the dispatch seam gets one target.
    assert params["issue"] is None


# ---------------------------------------------------------------------------
# Dispatch invocation shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_dispatch_invocation_shape_no_wait():
    coordinator = _FakeCoordinator()
    handler = make_talon_pipeline_dispatch_handler(coordinator)
    result = await handler(
        {"repo": "org/repo", "issue": 42, "self_review": True, "wait": False}
    )
    assert coordinator.dispatch_calls == [{
        "repo": "org/repo",
        "issue": 42,
        "pr": None,
        "mode": "claim",
        "self_review": True,
        "demo_check": False,
        "eye_check": False,
        # Detached: the wait_ref must outlive this stage (and possibly the
        # process), so the dispatch is forced onto the durable CLI path.
        "force_cli": True,
    }]
    assert result["state"] == "dispatched"
    assert result["job_id"] == "job123"
    assert result["wait_ref"] == "talon:job123"
    assert result["method"] == "cli_background"
    # No wait was requested — nothing polled.
    assert coordinator.wait_registry is None


@pytest.mark.asyncio
async def test_iterate_dispatch_invocation_shape():
    coordinator = _FakeCoordinator(wait_result=_complete_wait())
    handler = make_talon_pipeline_dispatch_handler(coordinator)
    result = await handler(
        {"repo": "org/repo", "pr": 9, "demo_check": True,
         "eye_check": True, "wait_timeout_seconds": 300}
    )
    assert coordinator.dispatch_calls == [{
        "repo": "org/repo",
        "issue": None,
        "pr": 9,
        "mode": "iterate",
        "self_review": None,
        "demo_check": True,
        "eye_check": True,
        # Attached (wait: true, the default): the in-process wait keeps the
        # A2A-preferred path available — no CLI forcing.
        "force_cli": False,
    }]
    assert coordinator.wait_registry.calls == [("talon:job123", 300)]
    assert result["state"] == "complete"
    assert result["status"] == "complete"
    assert result["returncode"] == 0


@pytest.mark.asyncio
async def test_claim_with_wait_keeps_a2a_preference():
    """Attached claim (wait: true) must NOT force CLI — the A2A-preferred
    path stays available because the wait is held in-process."""
    coordinator = _FakeCoordinator(wait_result=_complete_wait())
    handler = make_talon_pipeline_dispatch_handler(coordinator)
    await handler({"repo": "org/repo", "issue": 42})
    assert len(coordinator.dispatch_calls) == 1
    assert coordinator.dispatch_calls[0]["force_cli"] is False


@pytest.mark.asyncio
async def test_failed_dispatch_fails_closed():
    coordinator = _FakeCoordinator(
        dispatch_result={"dispatched": False, "error": "no workspace"}
    )
    handler = make_talon_pipeline_dispatch_handler(coordinator)
    with pytest.raises(ValueError, match="no workspace"):
        await handler({"repo": "org/repo", "issue": 1})


@pytest.mark.asyncio
async def test_completed_run_reports_pr_url_from_log():
    coordinator = _FakeCoordinator(wait_result=_complete_wait())
    coordinator._tail_job_log = lambda path, lines=20: (
        "pushing...\nOpened PR: https://github.com/org/repo/pull/77\n"
    )
    handler = make_talon_pipeline_dispatch_handler(coordinator)
    result = await handler({"repo": "org/repo", "issue": 5})
    assert result["pr_url"] == "https://github.com/org/repo/pull/77"
    assert result["state"] == "complete"


@pytest.mark.asyncio
async def test_terminal_failure_fails_closed():
    failed = ToolResult.failed(
        "Talon job job123 ended in 'failed' (rc=1)",
        data={"status": "failed", "returncode": 1, "timed_out": False},
    )
    coordinator = _FakeCoordinator(wait_result=failed)
    handler = make_talon_pipeline_dispatch_handler(coordinator)
    with pytest.raises(ValueError, match="'failed'"):
        await handler({"repo": "org/repo", "issue": 5})


@pytest.mark.asyncio
async def test_wait_timeout_fails_closed_with_wait_ref():
    pending = ToolResult.partial(
        confirmation="still pending",
        error="Timeout",
        data={"status": "running", "timed_out": True},
    )
    coordinator = _FakeCoordinator(wait_result=pending)
    handler = make_talon_pipeline_dispatch_handler(coordinator)
    with pytest.raises(ValueError, match=r"talon:job123"):
        await handler({"repo": "org/repo", "issue": 5})


# ---------------------------------------------------------------------------
# Round-trip through the real dispatcher (workflow-rescue pattern)
# ---------------------------------------------------------------------------


class _FakeAgent:
    # SignalDispatcher's durable consumer contract scopes the ledger to the
    # owning agent DID, just like the production DispatcherAgent protocol.
    did = "did:web:k.example"

    def __init__(self):
        self.background_tasks: list = []

    def _track_background_task(self, coro, *, name):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.mark.asyncio
async def test_dispatcher_round_trip(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "pipeline.db"))
    await backend.connect()
    try:
        store = SignalLogStore(backend)
        await store.initialize()
        registry = SourceRegistry()
        coordinator = _FakeCoordinator()
        assert register_talon_pipeline_source(registry, coordinator) is True
        agent = _FakeAgent()
        dispatcher = SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
        )
        result = await dispatcher.dispatch_signal(Signal(
            source=SOURCE_NAME,
            kind="workflow.stage",
            mode=SignalMode.ACTION,
            payload={"repo": "org/repo", "issue": 3, "wait": False},
            target_agent="did:web:k.example",
            visibility=Visibility.INTERNAL,
            urgency=Urgency.NORMAL,
        ))
        assert result.status is Status.OK
        assert result.action_result["state"] == "dispatched"
        assert result.action_result["job_id"] == "job123"
        pending = [t for t in agent.background_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        await backend.close()
