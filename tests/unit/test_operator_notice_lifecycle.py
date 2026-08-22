"""Operator-notice lifecycle: audit truth and retry behaviour (#2530).

Before #2530 a notice was audited ``delivered: true`` at COLLECT time — before
injection, before any provider saw it, before a turn completed — while the same
call drained the producer's pending auto-mode queue and advanced its dedupe
state. A turn that died afterwards left a durable lie plus a notice nobody
could ever see again.

These tests pin the two halves of the fix that a mocked test cannot reach:

1. **Audit truth** — no row claims ``delivered`` until that notice's own
   beyond-loss boundary was observed, and the first terminal write wins.
2. **Retry** — exactly what left no durable trace is requeued, proven by
   running a SECOND turn and watching the notice come back (or, for a
   persisted fallback, correctly not come back).

The turn-level cases drive the real ``StreamingMixin`` / ``KestrelAgent``
entry points against a real ``OperatorSignalProducer`` and a real SQLite
``AsyncDatabase``, because the defect lived in the seam between them.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.agent.operator_signals import (
    SOURCE_AUTO_MODE,
    OperatorSignalProducer,
    OperatorTurnInjectionResult,
)
from kestrel_sovereign.agent.streaming import StreamingMixin
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.async_database import (
    CORE_SCHEMA,
    AsyncDatabase,
    normalize_schema,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.operator_notice_store import (
    OperatorNoticeAuditStore,
    OperatorNoticeState,
)

AGENT_DID = "did:test:operator-notice-lifecycle"


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


async def _audit_db(tmp_path):
    """A real AsyncDatabase built by the real production schema path.

    Deliberately ``_init_schema`` rather than hand-applying this table's DDL.
    ``_init_schema`` splits ``CORE_SCHEMA`` on ``;`` with no SQL parser, so a
    table can be perfectly well-formed on its own and still break every agent's
    boot through how it is *spliced in*. A fixture that applied only this
    table's statements would pass while the fleet failed to start.
    """
    backend = SQLiteBackend(str(tmp_path / "operator-notice.db"))
    await backend.connect()
    db = AsyncDatabase(backend)
    await db._init_schema()
    return db


class _InlineRoute:
    """LLM-service double whose route advertises inline system support.

    ``supports_inline`` drives the producer's transport choice, which is what
    selects the delivery boundary under test: inline ``system`` (ephemeral,
    settles at provider-accept) vs fallback ``user`` (durable, settles at
    persistence).
    """

    def __init__(self, *, supports_inline=True, stream_factory=None):
        self.supports_inline = supports_inline
        self.calls = []
        self._stream_factory = stream_factory

    def resolve_provider_routing(self, *, model_override=None, force_local_only=False):
        return (
            [
                {
                    "name": "contract",
                    "vendor": "contract",
                    "model": "test-model",
                    "capabilities": {
                        "supports_inline_system": self.supports_inline
                    },
                    "adapter": None,
                }
            ],
            "test-model",
        )

    async def generate_with_messages(self, **kwargs):
        self.calls.append(kwargs)
        raise _ProviderExploded

    def stream_with_tool_detection(self, **kwargs):
        self.calls.append(kwargs)
        if self._stream_factory is not None:
            return self._stream_factory()
        return self._exploding_stream()

    async def _exploding_stream(self):
        """Default: die before the first chunk, i.e. never delivered."""
        raise _ProviderExploded
        yield  # pragma: no cover - makes this an async generator


class _ProviderExploded(RuntimeError):
    """The provider call died before accepting the request."""


class _PreStreamExploded(RuntimeError):
    """Turn setup died between collection and the provider boundary."""


class _PrivacyAgent:
    def __init__(self):
        self.persist_calls = []
        self.privacy_config = SimpleNamespace(allows_cloud_llm=lambda: True)
        self.privacy_mode = SimpleNamespace(name="NORMAL")

    async def get_conversation_history(self, **_kwargs):
        return []

    async def add_conversation(self, role, content, **kwargs):
        self.persist_calls.append({"role": role, "content": content, **kwargs})

    def operator_rows(self):
        return [
            call
            for call in self.persist_calls
            if (call.get("metadata") or {}).get("operator_signal")
        ]


class _ContextManager:
    """Budget deliberately NOT low and state_of_mind absent.

    Keeps every turn's notice set to exactly the enqueued auto-mode event, so
    "did the notice come back?" is unambiguous.
    """

    async def build_context(self, **_kwargs):
        return SimpleNamespace(
            system_prompt="system prompt",
            messages=[],
            total_tokens=42,
            budget_summary={"total_budget": 100000, "total_used": 42},
            warnings=[],
            dynamic_user_context="retrieved context",
            episode_count=0,
            memory_count=0,
            rag_chunks=0,
            degraded_mode=False,
            state_of_mind=None,
        )


class _ObservabilityStore:
    def __init__(self, *, fail_tool_call=False):
        self.fail_tool_call = fail_tool_call

    async def log_metric(self, **_kwargs):
        return None

    async def log_tool_call(self, **_kwargs):
        if self.fail_tool_call:
            raise _PreStreamExploded("observability unavailable")
        return "llm-event"

    async def log_tool_response(self, **_kwargs):
        return None


def _turn_agent(
    db,
    llm,
    *,
    fail_tool_call=False,
    resolve_eager_images=None,
):
    producer = OperatorSignalProducer(None)
    agent = SimpleNamespace(
        did=AGENT_DID,
        hooks_manager=None,
        llm_service=llm,
        privacy_agent=_PrivacyAgent(),
        context_manager=_ContextManager(),
        features={},
        operator_signal_producer=producer,
        observability_store=_ObservabilityStore(fail_tool_call=fail_tool_call),
        extension=None,
        user_prompt_template="{context}\n{query}",
        _raw_storage=SimpleNamespace(db=db),
        _privacy_mode=SimpleNamespace(value="NORMAL"),
        _session_briefed=False,
        _maybe_compact_codex_thread=AsyncMock(),
        _get_governing_constitution=AsyncMock(return_value="constitution"),
        _assemble_post_build_system_prompt=(
            lambda system_prompt, _context, *, user_prompt: system_prompt
        ),
        _build_all_tools=lambda: [],
        _lazy_attachment_hint=lambda attachments: "",
        _make_inline_tool_executor=lambda _session_id: None,
        _resolve_eager_images=resolve_eager_images,
        is_request_cancelled=lambda _rid=None: False,
    )
    # The producer resolves its audit store off the agent it belongs to.
    producer._agent = agent
    return agent, producer


async def _run_streaming(agent, *, attachments=None):
    async for _ in StreamingMixin._process_input_streaming_traced_locked(
        agent, "current question", "test-model", "session-1", None,
        attachments=attachments,
    ):
        pass


async def _run_non_streaming(agent):
    await KestrelAgent._process_input_traced_locked(
        agent, "current question", "test-model", "session-1", None,
    )


async def _auto_mode_rows(db):
    store = OperatorNoticeAuditStore(db, AGENT_DID)
    return [
        row
        for row in await store.list_recent(limit=50)
        if row.source == SOURCE_AUTO_MODE
    ]


# ---------------------------------------------------------------------------
# Schema: the new table must not break how CORE_SCHEMA is spliced
# ---------------------------------------------------------------------------


def test_core_schema_statements_are_executable():
    """No CORE_SCHEMA fragment may be orphaned by the naive ``;`` split.

    ``_init_schema`` has no SQL parser — it splits the whole schema on ``;``
    and executes the pieces. A semicolon inside a ``--`` comment therefore cuts
    the statement that follows it in half, and because init is one sequential
    loop the failure is not local to the offending table: schema creation dies
    there and every table declared after it silently never exists.

    This is cheap to get wrong from a docstring edit and catastrophic at boot,
    so the constraint is asserted rather than left as folklore.
    """
    for backend_type in ("sqlite", "postgres"):
        schema = normalize_schema(CORE_SCHEMA, backend_type)
        for fragment in schema.split(";"):
            body = "\n".join(
                line
                for line in fragment.splitlines()
                if line.strip() and not line.strip().startswith("--")
            ).strip()
            if not body:
                continue
            assert body.upper().startswith(("CREATE", "ALTER", "DROP", "INSERT")), (
                f"{backend_type}: CORE_SCHEMA fragment does not start with DDL — "
                "a ';' inside a comment almost certainly split a statement:\n"
                f"{body[:200]}"
            )


@pytest.mark.asyncio
async def test_init_schema_creates_the_operator_notice_table(tmp_path):
    """The real boot path, on a fresh database, actually creates the table."""
    db = await _audit_db(tmp_path)
    rows = await db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        ("operator_notice_audit",),
    )
    assert [row[0] for row in rows] == ["operator_notice_audit"]


# ---------------------------------------------------------------------------
# Store: the collect-phase row must not claim delivery, and cannot be rewritten
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_phase_row_claims_nothing_and_is_visible_as_unsettled(
    tmp_path,
):
    """The negative evidence this table exists for."""
    db = await _audit_db(tmp_path)
    store = OperatorNoticeAuditStore(db, AGENT_DID)

    await store.record_collected(
        notice_id="n1",
        session_id="s1",
        delivery_role="system",
        fallback=False,
        route="contract/test-model",
        events=[(SOURCE_AUTO_MODE, {"scope": "session"})],
    )

    (row,) = await store.list_for_notice("n1")
    assert row.state == OperatorNoticeState.COLLECTED.value
    assert row.claims_delivery is False
    assert row.settled_at is None
    assert row.injected_at is None
    assert row.durable_trace is False
    assert row.payload == {"scope": "session"}
    # A notice that was collected and then vanished is findable, not silence.
    assert [r.notice_id for r in await store.list_unsettled()] == ["n1"]


@pytest.mark.asyncio
async def test_first_terminal_settle_wins(tmp_path):
    """A late "the turn failed" must not overwrite an honest delivery.

    This is the fallback-notice case: it settles ``delivered`` the moment it is
    persisted to history, and the surrounding turn dying afterwards does not
    take it back out of the user's history.
    """
    db = await _audit_db(tmp_path)
    store = OperatorNoticeAuditStore(db, AGENT_DID)
    await store.record_collected(
        notice_id="n1", session_id=None, delivery_role="user", fallback=True,
        route="r", events=[(SOURCE_AUTO_MODE, {"scope": "off"})],
    )
    await store.mark_injected("n1")

    assert await store.settle(
        "n1", state=OperatorNoticeState.DELIVERED, durable_trace=True
    ) == 1
    # Second, contradictory settle changes nothing.
    assert await store.settle(
        "n1", state=OperatorNoticeState.FAILED, reason="turn_failed", requeued=True
    ) == 0

    (row,) = await store.list_for_notice("n1")
    assert row.state == OperatorNoticeState.DELIVERED.value
    assert row.durable_trace is True
    assert row.requeued is False
    assert row.state_reason == ""


@pytest.mark.asyncio
async def test_settle_refuses_a_non_terminal_state(tmp_path):
    """``collected``/``injected`` are not outcomes and cannot be settled to."""
    db = await _audit_db(tmp_path)
    store = OperatorNoticeAuditStore(db, AGENT_DID)
    with pytest.raises(ValueError):
        await store.settle("n1", state=OperatorNoticeState.COLLECTED)


@pytest.mark.asyncio
async def test_rows_are_scoped_to_the_owning_agent(tmp_path):
    """A shared backend must not leak one agent's notices into another's."""
    db = await _audit_db(tmp_path)
    mine = OperatorNoticeAuditStore(db, AGENT_DID)
    theirs = OperatorNoticeAuditStore(db, "did:test:someone-else")
    await mine.record_collected(
        notice_id="n1", session_id=None, delivery_role="system", fallback=False,
        route="r", events=[(SOURCE_AUTO_MODE, {"scope": "off"})],
    )

    assert await theirs.list_for_notice("n1") == []
    assert await theirs.list_unsettled() == []
    # And another agent cannot settle it.
    assert await theirs.settle("n1", state=OperatorNoticeState.DELIVERED) == 0
    assert (await mine.list_for_notice("n1"))[0].state == (
        OperatorNoticeState.COLLECTED.value
    )


# ---------------------------------------------------------------------------
# watch_stream: streaming early-close / cancellation semantics, stated + tested
# ---------------------------------------------------------------------------


async def _collected_turn(db, *, supports_inline=True):
    """A real producer that has genuinely consumed a pending auto-mode event."""
    llm = _InlineRoute(supports_inline=supports_inline)
    agent = SimpleNamespace(did=AGENT_DID, _raw_storage=SimpleNamespace(db=db))
    producer = OperatorSignalProducer(agent)
    producer.enqueue_auto_mode("session")

    batch = await producer.collect_for_turn(
        session_id="s1",
        llm_service=llm,
        model_override=None,
        force_local_only=False,
        budget_summary=None,
        state_of_mind=None,
    )
    assert batch.has_events
    assert producer._pending_auto_mode == [], "collect must drain the queue"
    result = OperatorTurnInjectionResult(batch=batch, injected_message={})
    await batch.mark_injected()
    return producer, result


@pytest.mark.asyncio
async def test_first_chunk_settles_delivered_and_does_not_requeue(tmp_path):
    db = await _audit_db(tmp_path)
    producer, turn = await _collected_turn(db)

    async def _stream():
        yield "hello"
        yield " world"

    assert [item async for item in turn.watch_stream(_stream())] == [
        "hello", " world",
    ]

    assert turn.batch.state is OperatorNoticeState.DELIVERED
    assert producer._pending_auto_mode == []
    (row,) = await _auto_mode_rows(db)
    assert row.claims_delivery is True
    assert row.requeued is False
    # Ephemeral inline notice: delivered, but nothing durable was left behind.
    assert row.durable_trace is False


class _RecordingStream:
    """Async iterator that closes ONLY when someone explicitly asks it to.

    A plain async generator is also finalized by the event loop's asyncgen GC
    hook, so asserting "it got closed" against one passes whether or not
    ``watch_stream`` closed it — a vacuous test. This object has no finalizer,
    so ``closed`` is True only if ``aclose()`` was genuinely called.
    """

    def __init__(self, items):
        self._items = list(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_consumer_break_after_first_chunk_stays_delivered(tmp_path):
    """The #1256 stop button does not un-send what the model already read."""
    db = await _audit_db(tmp_path)
    producer, turn = await _collected_turn(db)
    provider_stream = _RecordingStream(["hello", "never reached"])

    stream = turn.watch_stream(provider_stream)
    async for _item in stream:
        break
    await stream.aclose()

    assert turn.batch.state is OperatorNoticeState.DELIVERED
    # Requeuing here would re-report a fact the turn genuinely carried.
    assert producer._pending_auto_mode == []
    (row,) = await _auto_mode_rows(db)
    assert row.claims_delivery is True
    assert row.requeued is False
    # P3: the provider stream — which holds the cancel token and the upstream
    # connection — is released by watch_stream, not left to the GC hook.
    assert provider_stream.closed is True


@pytest.mark.asyncio
async def test_failure_before_first_chunk_requeues(tmp_path):
    db = await _audit_db(tmp_path)
    producer, turn = await _collected_turn(db)

    async def _stream():
        raise _ProviderExploded
        yield  # pragma: no cover - makes this an async generator

    with pytest.raises(_ProviderExploded):
        async for _ in turn.watch_stream(_stream()):
            pass  # pragma: no cover

    assert turn.batch.state is OperatorNoticeState.FAILED
    assert producer._pending_auto_mode == ["session"]
    (row,) = await _auto_mode_rows(db)
    assert row.claims_delivery is False
    assert row.state == OperatorNoticeState.FAILED.value
    assert row.requeued is True
    assert row.state_reason == "stream_failed:_ProviderExploded"


@pytest.mark.asyncio
async def test_cancellation_before_first_chunk_is_cancelled_not_failed(tmp_path):
    """Cancelled and failed answer different questions after the fact."""
    db = await _audit_db(tmp_path)
    producer, turn = await _collected_turn(db)

    async def _stream():
        raise asyncio.CancelledError
        yield  # pragma: no cover - makes this an async generator

    with pytest.raises(asyncio.CancelledError):
        async for _ in turn.watch_stream(_stream()):
            pass  # pragma: no cover

    assert turn.batch.state is OperatorNoticeState.CANCELLED
    assert producer._pending_auto_mode == ["session"]
    (row,) = await _auto_mode_rows(db)
    assert row.state == OperatorNoticeState.CANCELLED.value
    assert row.requeued is True


@pytest.mark.asyncio
async def test_empty_stream_is_not_an_observation_of_delivery(tmp_path):
    db = await _audit_db(tmp_path)
    producer, turn = await _collected_turn(db)

    async def _stream():
        return
        yield  # pragma: no cover - makes this an async generator

    assert [item async for item in turn.watch_stream(_stream())] == []

    assert turn.batch.state is OperatorNoticeState.FAILED
    assert producer._pending_auto_mode == ["session"]
    (row,) = await _auto_mode_rows(db)
    assert row.claims_delivery is False
    assert row.state_reason == "stream_closed_without_output"


# ---------------------------------------------------------------------------
# Turn level: the window between collection and the provider boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["eager_images", "log_tool_call"],
    ids=["eager-image-resolution", "tool-call-logging"],
)
async def test_streaming_pre_provider_failure_never_consumes_the_notice(
    tmp_path, failure
):
    """Turn setup that dies before the provider must not eat the notice.

    ``_resolve_eager_images`` and ``log_tool_call`` both ``await`` during turn
    setup. While the operator notice was collected ABOVE them, either one
    raising drained the pending auto-mode queue and advanced dedupe state for a
    notice that no provider ever saw, and nothing settled the row — the exact
    silent-drop class #2530 closes, reached without touching the LLM at all.

    The fix is structural: nothing between collection and the provider call
    awaits, because collection moved down to the provider boundary. So the
    assertion is the strong one — the notice is not merely requeued, it was
    never consumed, and no audit row was ever written.
    """
    db = await _audit_db(tmp_path)

    async def _boom(_attachments):
        raise _PreStreamExploded("attachment store unavailable")

    llm = _InlineRoute()
    agent, producer = _turn_agent(
        db,
        llm,
        fail_tool_call=(failure == "log_tool_call"),
        resolve_eager_images=_boom,
    )
    producer.enqueue_auto_mode("session")
    attachments = [{"id": "a1", "kind": "image", "inline": True}]

    with pytest.raises(_PreStreamExploded):
        await _run_streaming(
            agent,
            attachments=attachments if failure == "eager_images" else None,
        )

    assert llm.calls == [], "the provider must never have been called"
    # Never consumed => nothing to requeue, and nothing claimed.
    assert producer._pending_auto_mode == ["session"]
    assert await _auto_mode_rows(db) == []

    # And the notice genuinely still reaches the next turn's provider call.
    agent.observability_store.fail_tool_call = False
    with pytest.raises(_ProviderExploded):
        await _run_streaming(agent)

    assert len(llm.calls) == 1
    assert llm.calls[0]["messages"][-1]["role"] == "system"
    assert "auto-mode is now enabled" in llm.calls[0]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_streaming_provider_failure_requeues_for_the_next_turn(tmp_path):
    """Inline notice + dead provider => failed row, and the notice comes back."""
    db = await _audit_db(tmp_path)

    def _stream_factory():
        async def _stream():
            raise _ProviderExploded
            yield  # pragma: no cover - makes this an async generator

        return _stream()

    llm = _InlineRoute(stream_factory=_stream_factory)
    agent, producer = _turn_agent(db, llm)
    producer.enqueue_auto_mode("always")

    with pytest.raises(_ProviderExploded):
        await _run_streaming(agent)

    rows = await _auto_mode_rows(db)
    assert [row.state for row in rows] == [OperatorNoticeState.FAILED.value]
    assert rows[0].claims_delivery is False
    assert rows[0].requeued is True
    assert producer._pending_auto_mode == ["always"]

    # Second turn re-emits it — the retry is real, not just bookkeeping.
    with pytest.raises(_ProviderExploded):
        await _run_streaming(agent)
    assert len(llm.calls) == 2
    assert "standing consent" in llm.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_non_streaming_provider_failure_requeues_for_the_next_turn(tmp_path):
    db = await _audit_db(tmp_path)
    llm = _InlineRoute()
    agent, producer = _turn_agent(db, llm)
    producer.enqueue_auto_mode("session")

    with pytest.raises(_ProviderExploded):
        await _run_non_streaming(agent)

    rows = await _auto_mode_rows(db)
    assert [row.state for row in rows] == [OperatorNoticeState.FAILED.value]
    assert rows[0].requeued is True
    assert producer._pending_auto_mode == ["session"]

    with pytest.raises(_ProviderExploded):
        await _run_non_streaming(agent)
    assert len(llm.calls) == 2
    assert llm.calls[1]["messages"][-1]["role"] == "system"


@pytest.mark.asyncio
async def test_persisted_fallback_notice_is_delivered_and_never_requeued(tmp_path):
    """The durable form settles at persistence and survives the turn dying.

    Requeuing it would put a second copy of the same notice in the user's
    conversation history — the double-delivery the split boundary exists to
    prevent.
    """
    db = await _audit_db(tmp_path)
    llm = _InlineRoute(supports_inline=False)
    agent, producer = _turn_agent(db, llm)
    producer.enqueue_auto_mode("session")

    with pytest.raises(_ProviderExploded):
        await _run_non_streaming(agent)

    rows = await _auto_mode_rows(db)
    assert [row.state for row in rows] == [OperatorNoticeState.DELIVERED.value]
    assert rows[0].durable_trace is True
    assert rows[0].requeued is False
    assert rows[0].delivery_role == "user"
    assert producer._pending_auto_mode == []

    # Exactly one operator row in history, and the next turn adds no second one.
    assert len(agent.privacy_agent.operator_rows()) == 1
    with pytest.raises(_ProviderExploded):
        await _run_non_streaming(agent)
    assert len(agent.privacy_agent.operator_rows()) == 1
    assert llm.calls[1]["messages"][-1]["role"] == "user"
