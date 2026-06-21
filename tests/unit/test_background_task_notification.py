"""Regression tests for background-task completion notification formatting.

Issue #1526: completion toasts showed ``unknown/task (task: 2ea43cc2)``
because the formatter only read ``agent_id`` / ``skill`` metadata and
truncated the task id to an unresolvable 8-char prefix.

These tests pin the label derivation against the metadata that real
producers actually stamp on an A2A ``Task``:

- ``execute_skill`` → ``agent_id`` + ``skill``
- the inbound A2A submit endpoint → ``sender`` (+ optional ``task_type``)
- ``create_task`` → ``causation_chain`` (the only thread that ties a
  scheduler-originated task back to its ``cron.<name>`` schedule, since
  the scheduler dispatches a *signal*, not an A2A task)

The causation-chain test builds its metadata with the production
serializer (``serialize_chain_for_metadata``) so the reader is verified
against the exact wire shape the writer emits — not a hand-built dict.
"""

from __future__ import annotations

from datetime import datetime, timezone

from kestrel_sdk.signals import CausationFrame
from kestrel_sovereign.agent.event_manager import (
    EventManagerMixin,
    background_task_identifiers,
    describe_background_task,
)
from kestrel_sovereign.signals.sources.a2a import serialize_chain_for_metadata
from kestrel_sovereign.a2a.types import (
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)


def _task(metadata, *, task_id="task-1526-full-id-0001",
          state=TaskState.COMPLETED, message=None):
    status = TaskStatus(state=state, message=message)
    return Task(id=task_id, sessionId="s-1", status=status, metadata=metadata)


def _cron_chain_metadata(*sources):
    """Build ``task.metadata`` carrying a causation chain via the real
    production serializer, so the test exercises the exact shape
    ``create_task`` attaches. ``sources`` are signal source names, oldest
    first; the most recent is the proximate cause."""
    frames = [
        CausationFrame(
            agent_id="did:test:1526",
            source=src,
            signal_id=f"sig-{i}",
            turn_id=f"turn-{i}",
            depth=i,
            emitted_at=datetime(2026, 6, 4, 15, 49, i, tzinfo=timezone.utc),
        )
        for i, src in enumerate(sources)
    ]
    return {"causation_chain": serialize_chain_for_metadata(frames)}


class _AgentLike(EventManagerMixin):
    """Minimal agent with no dispatcher — exercises only the SSE path."""

    did = "did:test:1526"

    def __init__(self):
        self._event_listeners = []
        self._pending_task_notifications = []


# ---------------------------------------------------------------------------
# describe_background_task — label derivation priority
# ---------------------------------------------------------------------------


def test_execute_skill_metadata_preserved():
    """The A2A skill path (agent_id + skill, stamped by execute_skill)
    yields ``<agent>/<skill>``."""
    source, name = describe_background_task(
        _task({"agent_id": "PeerAgent", "skill": "do_thing"})
    )
    assert (source, name) == ("PeerAgent", "do_thing")


def test_inbound_sender_and_task_type_used():
    """An inbound peer task (sender + task_type from the A2A submit
    endpoint) renders ``<sender>/<task_type>``."""
    source, name = describe_background_task(
        _task({"sender": "did:peer:9", "task_type": "lora_training"})
    )
    assert (source, name) == ("did:peer:9", "lora_training")


def test_scheduler_origin_via_causation_chain():
    """The issue's headline case. A scheduled task dispatches a signal,
    not an A2A task; a task spawned downstream carries the causation
    chain whose proximate frame is ``cron.restart_coordinator``. That
    must render as ``scheduler``-recognisable ``cron/restart_coordinator``
    — never ``unknown/task``."""
    source, name = describe_background_task(
        _task(_cron_chain_metadata(
            "cron.wait_reconcile", "cron.restart_coordinator",
        ))
    )
    assert (source, name) == ("cron", "restart_coordinator")


def test_causation_chain_single_segment_source():
    """A source with no ``.`` separator degrades to ``<source>/task``."""
    source, name = describe_background_task(
        _task(_cron_chain_metadata("scheduler"))
    )
    assert (source, name) == ("scheduler", "task")


def test_real_producer_keys_win_over_causation_chain():
    """When a direct producer key is present it is preferred over the
    causation-chain fallback (more specific identifies the actual work)."""
    md = _cron_chain_metadata("cron.restart_coordinator")
    md.update({"agent_id": "PeerAgent", "skill": "do_thing"})
    assert describe_background_task(_task(md)) == ("PeerAgent", "do_thing")


def test_empty_metadata_falls_back_to_unknown_task():
    """No usable metadata still degrades gracefully to the historical
    label (but the full task id is exposed separately)."""
    assert describe_background_task(_task({})) == ("unknown", "task")
    assert describe_background_task(_task(None)) == ("unknown", "task")


def test_task_type_only_metadata():
    source, name = describe_background_task(_task({"task_type": "lora_training"}))
    assert (source, name) == ("unknown", "lora_training")


# ---------------------------------------------------------------------------
# background_task_identifiers — full id exposure
# ---------------------------------------------------------------------------


def test_identifiers_expose_full_task_id():
    """The full (untruncated) task id must appear so it is resolvable via
    check_task_status (#1526 acceptance criterion 3)."""
    ids = background_task_identifiers(_task({}, task_id="abcd1234-full-uuid"))
    # Not the truncated 8-char prefix that the old format emitted.
    assert ids == "task: abcd1234-full-uuid"


def test_identifiers_ignore_unproduced_execution_id():
    """No producer stamps ``execution_id`` on an A2A task (the scheduler's
    execution id is born after the run, in task_execution_log), so the
    identifier suffix never surfaces one even if a stray key is present."""
    ids = background_task_identifiers(
        _task({"execution_id": "should-not-appear"}, task_id="45a190ee-task")
    )
    assert ids == "task: 45a190ee-task"


# ---------------------------------------------------------------------------
# End-to-end: the queued notification string
# ---------------------------------------------------------------------------


def test_scheduler_completion_notification_string():
    """Full formatting for the scheduler path: a task spawned during a
    cron-driven flow renders the cron task name and the full task id —
    and never ``unknown/task``. Metadata is built with the production
    serializer to prove the consumer matches the producer's wire shape."""
    agent = _AgentLike()
    task = _task(
        _cron_chain_metadata("cron.restart_coordinator"),
        task_id="45a190ee-cee5-4137-a739-21ae1ea4711b",
    )

    agent._on_background_task_complete(task)

    assert len(agent._pending_task_notifications) == 1
    msg = agent._pending_task_notifications[0]
    assert "Background task completed" in msg
    assert "cron/restart_coordinator" in msg
    assert "unknown/task" not in msg
    assert "task: 45a190ee-cee5-4137-a739-21ae1ea4711b" in msg


def test_failed_notification_includes_label_and_error():
    agent = _AgentLike()
    task = _task(
        _cron_chain_metadata("cron.wait_reconcile"),
        task_id="6314e1ea-41a8-4074-87ba-4f191551de61",
        state=TaskState.FAILED,
        message=Message(role="agent", parts=[TextPart(text="boom")]),
    )

    agent._on_background_task_complete(task)

    msg = agent._pending_task_notifications[0]
    assert "Background task failed" in msg
    assert "cron/wait_reconcile" in msg
    assert "boom" in msg
    assert "task: 6314e1ea-41a8-4074-87ba-4f191551de61" in msg


def test_canceled_notification_uses_full_id():
    agent = _AgentLike()
    task = _task({}, task_id="2ea43cc2-full-id", state=TaskState.CANCELED)

    agent._on_background_task_complete(task)

    msg = agent._pending_task_notifications[0]
    assert "Background task canceled" in msg
    assert "task: 2ea43cc2-full-id" in msg
