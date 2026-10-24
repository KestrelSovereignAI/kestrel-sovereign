"""#3655: a live autonomous-wake bubble shows the whole reply, not 2 KB of it.

The ``signal_completed`` event's ``result_summary`` is capped at
``MAX_RESULT_SUMMARY_BYTES`` (and some sources shorten it further), so the chat
console rendered a long wake reply cut off mid-sentence, as if that were the
reply. The cap stays. The event now says whether the summary is the whole
result (``result_summary_complete``) and, for a COGNITION wake, names the
stored row holding the reply (``reply_message_id``); the console loads the
full text from ``GET /api/conversations/messages/{id}``.

The row is named by the wake's own stamp (``signal_wake_reply = <signal id>``
on the reply), never by "the newest assistant row in the session": in an
autonomous chain another wake's reply lands in the same session.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from kestrel_sdk.signals import (
    MAX_RESULT_SUMMARY_BYTES,
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
    Visibility,
)
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.output_ceiling import attach_stop_reason
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.db import SQLiteBackend
import kestrel_sovereign.storage.db.sqlite as sqlite_backend_module
from kestrel_sovereign.storage.session_id_column import new_session_id
from kestrel_sovereign.storage.turn_provenance import (
    SIGNAL_WAKE_REPLY_KEY,
    WakeReplyReceipt,
    wake_reply_metadata,
)
from tests.shared.booted_agent import booted_agent
from tests.utils.aiosqlite_workers import aiosqlite_worker, normal_close_schedule

SOURCE = "test.wake_reply"
LONG_REPLY = (
    "After a time limit expires the restart proceeds without waiting for the "
    "busy turn. " * 50
)
assert len(LONG_REPLY.encode("utf-8")) > MAX_RESULT_SUMMARY_BYTES


def _identity_summary(body):
    """The shape restart.completed / a2a.question_answered register."""
    if body is None:
        return ""
    return body if isinstance(body, str) else str(body)


def _registration(template: Path, *, result_summary=_identity_summary) -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE,
        schema=lambda payload: payload,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=template,
        log_redaction=RedactionPolicy(summarize=lambda payload: "<redacted>"),
        result_summary=result_summary,
        allow_self_loops=True,
    )


def _wake(*, session_id=None, kind="done") -> Signal:
    return Signal(
        source=SOURCE,
        kind=kind,
        mode=SignalMode.COGNITION,
        payload={},
        target_agent="did:test:3655",
        visibility=Visibility.USER_VISIBLE,
        session_id=session_id,
    )


@pytest.fixture
def template(tmp_path) -> Path:
    path = tmp_path / "wake.md"
    path.write_text("wake {kind}")
    return path


# ---------------------------------------------------------------------------
# The event fields, through the production dispatcher with a fake agent body
# ---------------------------------------------------------------------------


class _ReplyingAgent:
    """Stores nothing; records the receipt the dispatcher hands it and names
    a row for it, as the real agent's reply persist does."""

    did = "did:test:3655"

    def __init__(self):
        self.background_tasks: list[asyncio.Task] = []
        self.emitted: list[tuple[str, dict]] = []
        self.receipts: list[WakeReplyReceipt] = []
        self.replies: list[str] = []
        self.row_ids: list = []

    async def emit_event(self, event_type: str, data: dict) -> None:
        self.emitted.append((event_type, data))

    async def process_input(self, prompt, *, wake_reply=None, **_kwargs):
        self.receipts.append(wake_reply)
        if wake_reply is not None:
            wake_reply.message_id = self.row_ids.pop(0)
        return self.replies.pop(0)

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.fixture
async def fake(tmp_path, template):
    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _ReplyingAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=store,
    )
    registry.register(_registration(template))
    yield SimpleNamespace(agent=agent, dispatcher=dispatcher, registry=registry)
    await _drain(dispatcher)
    await backend.close()


async def _drain(dispatcher) -> None:
    """Wait for every outcome writer: each emits its event after the log."""
    await dispatcher._drain_outcome_log_tasks()


@pytest.mark.asyncio
async def test_event_names_the_row_the_turn_recorded(fake):
    fake.agent.replies = [LONG_REPLY]
    fake.agent.row_ids = [41]
    signal = _wake()
    result = await fake.dispatcher.dispatch_signal(signal)
    await _drain(fake.dispatcher)

    assert result.status == Status.OK
    receipt = fake.agent.receipts[0]
    assert isinstance(receipt, WakeReplyReceipt) and receipt.signal_id == signal.id
    (event_type, payload), = fake.agent.emitted
    assert event_type == "signal_completed"
    assert payload["reply_message_id"] == 41
    assert payload["result_summary_complete"] is False
    assert payload["result_summary"].endswith("...(truncated)")
    assert len(payload["result_summary"].encode("utf-8")) <= MAX_RESULT_SUMMARY_BYTES


@pytest.mark.asyncio
async def test_a_whole_summary_is_marked_complete(fake):
    fake.agent.replies = ["Short and whole."]
    fake.agent.row_ids = [42]
    await fake.dispatcher.dispatch_signal(_wake())
    await _drain(fake.dispatcher)

    payload = fake.agent.emitted[0][1]
    assert payload["result_summary"] == "Short and whole."
    assert payload["result_summary_complete"] is True
    assert payload["reply_message_id"] == 42


@pytest.mark.asyncio
async def test_a_source_shortened_summary_is_not_complete(tmp_path, template):
    """``self_followup`` and ``a2a`` cut their summary below the cap
    themselves; the flag compares the summary with the result, so it does
    not depend on which layer shortened it."""
    from kestrel_sovereign.signals.sources.self_followup import (
        self_followup_result_summary,
    )

    backend = SQLiteBackend(str(tmp_path / "shortened.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _ReplyingAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=store,
    )
    registry.register(_registration(template, result_summary=self_followup_result_summary))
    reply = "y" * 2001  # 2001 ASCII bytes: under the store cap, over the source's
    agent.replies = [reply]
    agent.row_ids = [7]
    await dispatcher.dispatch_signal(_wake())
    await _drain(dispatcher)
    await backend.close()

    payload = agent.emitted[0][1]
    assert payload["result_summary"] == "y" * 2000 + "...(truncated)"
    assert payload["result_summary_complete"] is False
    assert payload["reply_message_id"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("turn", ["unstored", "raises"])
async def test_a_redelivered_signal_never_carries_an_earlier_row(fake, turn):
    """The id rides on the signal object between the turn and the outcome
    writer. A later dispatch of a signal an earlier attempt already named a
    row for must not carry that row: not when its own reply was not stored,
    and not when its turn failed before replying."""
    from kestrel_sovereign.signals.dispatcher import _REPLY_MESSAGE_ID_ATTR

    signal = _wake()
    setattr(signal, _REPLY_MESSAGE_ID_ATTR, 10)  # an earlier attempt's row
    if turn == "unstored":
        fake.agent.replies = [LONG_REPLY]
        fake.agent.row_ids = [None]
    else:
        async def failing_turn(prompt, **_kwargs):
            raise RuntimeError("turn failed")

        fake.agent.process_input = failing_turn
    await fake.dispatcher.dispatch_signal(signal)
    await _drain(fake.dispatcher)

    (_, payload), = fake.agent.emitted
    assert payload["reply_message_id"] is None


@pytest.mark.asyncio
async def test_an_agent_without_the_kwarg_still_dispatches(tmp_path, template):
    class _MinimalAgent(_ReplyingAgent):
        async def process_input(self, prompt):
            return LONG_REPLY

    backend = SQLiteBackend(str(tmp_path / "minimal.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _MinimalAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=store,
    )
    registry.register(_registration(template))
    result = await dispatcher.dispatch_signal(_wake())
    await _drain(dispatcher)
    await backend.close()

    assert result.status == Status.OK
    payload = agent.emitted[0][1]
    assert payload["reply_message_id"] is None
    assert payload["result_summary_complete"] is False


# ---------------------------------------------------------------------------
# The store finds the stamped row exactly
# ---------------------------------------------------------------------------


AGENT = "did:test:3655-store"


@pytest.fixture
async def store(tmp_path, sqlite_database_factory):
    db = await sqlite_database_factory(tmp_path / "agent.db")
    return AsyncConversationStore(db, agent_id=AGENT)


async def _newest_id(store) -> int:
    row = await store.db.fetchone(
        "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
    )
    return int(row[0])


@pytest.mark.asyncio
async def test_store_finds_the_row_stamped_with_the_wake(store):
    session = new_session_id()
    await store.add_conversation(
        "assistant", "reply to A", wake_reply_metadata("sig-A"), session_id=session
    )
    a_row = await _newest_id(store)
    # Rows that must not answer for sig-A: another wake's newer reply in the
    # same session, an id sig-A is a prefix of, and a user row with the tag.
    await store.add_conversation(
        "assistant", "reply to B", wake_reply_metadata("sig-A0"), session_id=session
    )
    await store.add_conversation(
        "user", "not a reply", wake_reply_metadata("sig-A"), session_id=session
    )
    await store.add_conversation("assistant", "untagged", None, session_id=session)
    # The LIKE prefilter matches the tag anywhere in the document; only the
    # row's own top-level stamp names it the reply.
    await store.add_conversation(
        "assistant", "quotes the tag", {"quoted": wake_reply_metadata("sig-A")},
        session_id=session,
    )

    assert await store.find_signal_wake_reply_id("sig-A") == a_row
    assert await store.find_signal_wake_reply_id("sig-") is None
    assert await store.find_signal_wake_reply_id("") is None
    rows = await store.get_messages_by_ids([a_row])
    assert rows[0]["content"] == "reply to A"


@pytest.mark.asyncio
async def test_store_reads_minified_metadata_and_skips_deleted_rows(store):
    """``json_set`` updates re-emit metadata minified; a deleted reply is
    gone. Among live replies to one wake the newest answers (a redelivery
    writes its own, later)."""
    await store.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, metadata) "
        "VALUES (?, 'assistant', 'minified', ?)",
        (AGENT, json.dumps({SIGNAL_WAKE_REPLY_KEY: "sig_%"}, separators=(",", ":"))),
    )
    minified = await _newest_id(store)
    assert await store.find_signal_wake_reply_id("sig_%") == minified
    assert await store.find_signal_wake_reply_id("sigx%") is None, (
        "LIKE wildcards in the id must not broaden the match"
    )

    await store.add_conversation("assistant", "first", wake_reply_metadata("sig-R"))
    first = await _newest_id(store)
    await store.add_conversation("assistant", "second", wake_reply_metadata("sig-R"))
    second = await _newest_id(store)
    assert await store.find_signal_wake_reply_id("sig-R") == second
    await store.delete_message(second)
    assert await store.find_signal_wake_reply_id("sig-R") == first


# ---------------------------------------------------------------------------
# A real agent: the id on the event is the row holding that wake's reply
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _waking_agent(tmp_path, monkeypatch, template, replies):
    monkeypatch.delenv("KESTREL_CONTINUATION_CHECK", raising=False)
    async with booted_agent(
        tmp_path, name="Kite", provenance="test:3655", features=frozenset()
    ) as agent:
        queue = list(replies)

        async def scripted_reply(**_kwargs):
            return attach_stop_reason(LLMResponse(content=queue.pop(0)), "stop")

        monkeypatch.setattr(agent.llm_service, "generate_with_messages", scripted_reply)
        agent.signal_registry.register(_registration(template))
        emitted: list[dict] = []

        async def record(event_type, data):
            if event_type == "signal_completed":
                emitted.append(data)

        monkeypatch.setattr(agent, "emit_event", record)
        yield agent, emitted


async def _settle(agent) -> None:
    await _drain(agent.dispatcher)


async def _row(agent, message_id):
    rows = await agent._raw_storage.db.fetchall(
        "SELECT role, content, metadata FROM conversation_history WHERE id = ?",
        (message_id,),
    )
    role, content, metadata = rows[0]
    return role, json.loads(metadata)


@pytest.mark.asyncio
async def test_event_id_is_the_row_holding_that_wakes_reply(tmp_path, monkeypatch, template):
    """Two wakes in one autonomous chain: the second's reply is the newest
    row in the session by the time either event is read, and each event still
    names its own wake's reply."""
    second_reply = "Second wake handled."
    async with _waking_agent(
        tmp_path, monkeypatch, template, [LONG_REPLY, second_reply]
    ) as (agent, emitted):
        backend = agent._raw_storage.db.backend
        assert isinstance(backend, SQLiteBackend)
        original_open = backend._open_snapshot_read_connection
        original_close = sqlite_backend_module._close_aiosqlite_connection
        snapshots = set()
        admitted = set()
        loop = asyncio.get_running_loop()
        original_clock = loop.time
        # This functional test is about durable reply IDs, not native close
        # speed. Only snapshots actually created by THIS backend are scheduled.
        async with normal_close_schedule(prejoined=True) as finish:
            async def opened_snapshot():
                connection = await original_open()
                snapshots.add(connection)
                return connection

            async def close_snapshot(connection, **kwargs):
                if connection not in snapshots:
                    return await original_close(connection, **kwargs)
                assert connection not in admitted
                assert aiosqlite_worker(connection).is_alive()
                assert "close_task" not in kwargs
                assert kwargs.get("deadline") is None
                assert kwargs["retained_closes"] is backend._retired_connection_closes
                admitted.add(connection)
                result = await finish(
                    connection,
                    lambda original_task: original_close(
                        connection, close_task=original_task, **kwargs
                    ),
                    retained_closes=backend._retired_connection_closes,
                )
                assert not aiosqlite_worker(connection).is_alive()
                assert loop.time == original_clock
                return result

            with monkeypatch.context() as snapshot_patch:
                snapshot_patch.setattr(backend, "_open_snapshot_read_connection", opened_snapshot)
                snapshot_patch.setattr(sqlite_backend_module, "_close_aiosqlite_connection", close_snapshot)
                chain = new_session_id()
                first = _wake(session_id=chain, kind="first")
                second = _wake(session_id=chain, kind="second")
                assert (await agent.dispatcher.dispatch_signal(first)).status == Status.OK
                assert (await agent.dispatcher.dispatch_signal(second)).status == Status.OK
                await _settle(agent)

                by_signal = {event["signal_id"]: event for event in emitted}
                first_event, second_event = by_signal[first.id], by_signal[second.id]

                assert first_event["result_summary_complete"] is False
                assert second_event["result_summary_complete"] is True
                first_id = first_event["reply_message_id"]
                second_id = second_event["reply_message_id"]
                assert isinstance(first_id, int) and isinstance(second_id, int)
                assert first_id < second_id

                role, metadata = await _row(agent, first_id)
                assert role == "assistant"
                assert metadata[SIGNAL_WAKE_REPLY_KEY] == first.id
                assert metadata["session_id"] == chain
                stored = await agent.storage.get_conversation_message(first_id)
                assert stored["content"] == LONG_REPLY, "the stored row holds the whole reply"
                assert (await agent.storage.get_conversation_message(second_id))["content"] == second_reply
                assert admitted == snapshots
        assert loop.time == original_clock


@asynccontextmanager
async def _conversation_doors(agent):
    from fastapi import FastAPI

    from kestrel_sovereign.endpoints.conversations import router

    app = FastAPI()
    app.state.agent = agent
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_message_door_serves_the_full_reply(tmp_path, monkeypatch, template):
    async with _waking_agent(tmp_path, monkeypatch, template, [LONG_REPLY]) as (agent, emitted):
        await agent.dispatcher.dispatch_signal(_wake(session_id=new_session_id()))
        await _settle(agent)
        message_id = emitted[0]["reply_message_id"]

        async with _conversation_doors(agent) as client:
            found = await client.get(f"/api/conversations/messages/{message_id}")
            missing = await client.get(f"/api/conversations/messages/{message_id + 999}")
            invalid = await client.get("/api/conversations/messages/not-a-number")

    assert found.status_code == 200, found.text
    body = found.json()
    assert body["id"] == message_id
    assert body["role"] == "assistant"
    assert body["content"] == LONG_REPLY
    assert body["metadata"][SIGNAL_WAKE_REPLY_KEY] == emitted[0]["signal_id"]
    assert missing.status_code == 404
    assert invalid.status_code == 422


@pytest.mark.asyncio
async def test_a_reply_kept_out_of_storage_carries_no_id(tmp_path, monkeypatch, template):
    """ISOLATED keeps the turn in memory only: there is no stored row to
    name, so the event carries none and the console shows its labelled
    preview. The message door has no persistent row to serve either."""
    async with _waking_agent(
        tmp_path, monkeypatch, template, [LONG_REPLY, LONG_REPLY]
    ) as (agent, emitted):
        await agent.dispatcher.dispatch_signal(_wake(session_id=new_session_id()))
        await _settle(agent)
        stored_id = emitted[0]["reply_message_id"]
        assert isinstance(stored_id, int)

        await agent.set_privacy_mode(PrivacyMode.ISOLATED)
        await agent.dispatcher.dispatch_signal(_wake(session_id=new_session_id()))
        await _settle(agent)

        assert emitted[1]["reply_message_id"] is None
        assert emitted[1]["result_summary_complete"] is False
        assert await agent.storage.get_conversation_message(stored_id) is None


@pytest.mark.asyncio
async def test_a_reply_kept_in_memory_never_names_an_earlier_stored_reply(
    tmp_path, monkeypatch, template,
):
    """A redelivered wake whose turn runs where nothing is stored durably
    (DEIDENTIFIED buffers the turn in memory) must not name the row an
    earlier delivery of the same wake stored: that row is not this reply.
    The persistent store still holds it, so only the turn knows."""
    async with _waking_agent(
        tmp_path, monkeypatch, template, [LONG_REPLY, "A different reply."]
    ) as (agent, emitted):
        signal = _wake(session_id=new_session_id())
        await agent.dispatcher.dispatch_signal(signal)
        await _settle(agent)
        earlier = emitted[0]["reply_message_id"]
        assert isinstance(earlier, int)

        await agent.set_privacy_mode(PrivacyMode.DEIDENTIFIED)
        receipt = WakeReplyReceipt(signal_id=signal.id)
        await agent.process_input(
            "wake again",
            session_id=signal.session_id,
            signal_wake={"source": SOURCE, "mode": "cognition"},
            wake_reply=receipt,
        )

        assert receipt.message_id is None
        assert await agent.storage.find_signal_wake_reply_id(signal.id) == earlier
