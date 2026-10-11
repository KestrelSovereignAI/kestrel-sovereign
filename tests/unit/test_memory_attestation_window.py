"""The sleep attestation window and each memory's own turn (#3544).

The pre-sleep hook used to attest only memories retrieved in the 30 minutes
before sleep, and judged all of them against the latest messages. Sleep runs
once a day, so it found nothing to attest. These tests drive the real hook over
real storage and the real ``MemorySystem``; only the model is a double.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from kestrel_sdk.llm.decisions import DecisionResult, DecisionTimeout, NoulAnswer
from kestrel_sdk.storage.database.interface import QueryError

from kestrel_sovereign.agent.sleep import SleepHookStatus, SleepMixin
from kestrel_sovereign.features.memory import reflection_hook as hook_module
from kestrel_sovereign.features.memory.reflection_hook import (
    ATTESTATION_BOOTSTRAP_LOOKBACK,
    ATTESTATION_MARKS_TABLE,
    ATTESTATION_QUESTION,
    ATTESTATION_SETTLE,
    ATTESTATION_STATE_TABLE,
    ReflectionSleepHook,
    RetrievalMarks,
    RetrievedMemoryCandidate,
    render_turn,
    retrieving_turn,
)
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.conversation_created_at import created_at_bind
from kestrel_sovereign.storage.memory_attestation_state import ensure_attestation_tables
from kestrel_sovereign.storage.memory_system import (
    MemoryApplicationNotRecorded,
    MemorySystem,
)
from kestrel_sovereign.storage.privacy_wrapper import (
    PrivacyEnforcingStorage,
    ReentrantTransitionLock,
)
from tests.utils.postgres_schema import disposable_postgres_schema, with_search_path


# --- fixtures and doubles ----------------------------------------------------


@pytest.fixture(params=["sqlite", "postgres"])
async def storage(request, tmp_path):
    """Real storage on both engines, for a DID of this test's own."""
    did = f"did:test:attestation-window-{uuid4()}"
    if request.param == "postgres":
        url = os.environ.get("TEST_POSTGRES_URL")
        if not url:
            pytest.skip("TEST_POSTGRES_URL is not set")
        handle = AsyncStorage(backend="postgres", dsn=url, agent_id=did)
    else:
        handle = AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=did)
    async with handle as opened:
        opened.test_did = did
        try:
            yield opened
        finally:
            # A PostgreSQL database outlives the run: remove what this wrote.
            await ensure_attestation_tables(opened.db)
            for table in (ATTESTATION_STATE_TABLE, ATTESTATION_MARKS_TABLE):
                await opened.db.execute(
                    f"DELETE FROM {table} WHERE agent_id = ?", (did,)
                )
            await opened.db.execute(
                "DELETE FROM conversation_history WHERE agent_id = ?", (did,)
            )


class _ChatJudge:
    """``generate`` that records the session each memory was judged in.

    ``on_judged(memory)`` runs after a memory is judged and before the answer
    returns: what the world does while the model is thinking.
    """

    def __init__(self, *, applied=(), fail_on=(), on_judged=None):
        self.applied = set(applied)
        self.fail_on = set(fail_on)
        self.on_judged = on_judged
        self.sessions = {}

    def _current_force_local_only(self):
        return False

    async def generate(self, *, system_prompt, user_prompt, **kwargs):
        memory = user_prompt.split("Memory content:\n", 1)[1].split(
            "\n\nSession context:", 1
        )[0]
        session = user_prompt.split("Session context:\n", 1)[1].split(
            "\n\nAnswer as JSON only", 1
        )[0]
        if memory in self.fail_on:
            raise RuntimeError("provider down")
        self.sessions[memory] = session
        if self.on_judged is not None:
            await self.on_judged(memory)
        return json.dumps({"applied": memory in self.applied, "reason": "Used it."})


class _DecisionJudge:
    """``decide`` that records the session each memory was judged in."""

    def __init__(self, *, applied=(), fail_on=(), on_judged=None):
        self.applied = set(applied)
        self.fail_on = set(fail_on)
        self.on_judged = on_judged
        self.sessions = {}

    def _current_force_local_only(self):
        return False

    async def decide(self, request, **kwargs):
        memory = request.state["memory"]
        if memory in self.fail_on:
            raise DecisionTimeout("slow")
        self.sessions[memory] = request.state["session"]
        if self.on_judged is not None:
            await self.on_judged(memory)
        return DecisionResult(
            answers=MappingProxyType({
                ATTESTATION_QUESTION: NoulAnswer(
                    p_true=0.9 if memory in self.applied else 0.1
                )
            }),
            vendor="ollama", route="ollama:local", model="tev1:0.8b",
            thresholds=MappingProxyType({ATTESTATION_QUESTION: 0.5}),
            calibrated=False, input_tokens=1, duration_ms=3,
        )


_JUDGES = {"chat": _ChatJudge, "decision": _DecisionJudge}


class _Agent:
    def __init__(self, storage, memory_system, llm_service):
        self.did = storage.test_did
        self.agent_id = storage.test_did
        self._raw_storage = storage
        self.storage = storage
        self.memory_system = memory_system
        self.memory_consolidator = memory_system.consolidator
        self.llm_service = llm_service


async def _memory_system(storage):
    memory_system = MemorySystem(storage, storage.test_did)
    await memory_system.initialize()
    return memory_system


def _second(instant):
    return instant.replace(microsecond=0)


async def _add(storage, role, content, *, at):
    """A row created at ``at``, as the store's own clock would stamp it."""
    await storage.conversation.add_conversation(role, content)
    row_id = (await storage.conversation.get_full_history_with_ids())[-1]["id"]
    await storage.db.execute(
        "UPDATE conversation_history SET created_at = ? WHERE id = ?",
        (created_at_bind(storage.db.backend_type, _second(at)), row_id),
    )
    return row_id


async def _retrieved(storage, row_id, *, at):
    await storage.conversation.update_message_metadata(
        row_id, {"access_count": 1, "last_accessed": at.isoformat()}
    )


async def _memory(storage, content, *, retrieved_at):
    """A memory written days ago and last retrieved at ``retrieved_at``."""
    row_id = await _add(
        storage, "assistant", content,
        at=datetime.now(timezone.utc) - timedelta(days=9),
    )
    await _retrieved(storage, row_id, at=retrieved_at)
    return row_id


async def _turn(storage, user, reply, *, at):
    """A user row at ``at`` and its reply four seconds later."""
    await _add(storage, "user", user, at=at)
    await _add(storage, "assistant", reply, at=at + timedelta(seconds=4))


async def _metadata(storage, row_id):
    row = await storage.db.fetchone(
        "SELECT metadata FROM conversation_history WHERE id = ?", (row_id,)
    )
    return json.loads(row[0] or "{}")


async def _watermark(storage):
    row = await storage.db.fetchone(
        f"SELECT attested_through FROM {ATTESTATION_STATE_TABLE} WHERE agent_id = ?",
        (storage.test_did,),
    )
    return None if row is None else datetime.fromisoformat(row[0])


async def _set_watermark(storage, instant):
    await ensure_attestation_tables(storage.db)
    await ReflectionSleepHook()._advance_watermark(
        storage.db, agent_id=storage.test_did, through=instant
    )


async def _marks(storage):
    return await storage.db.fetchall(
        f"SELECT message_id FROM {ATTESTATION_MARKS_TABLE} WHERE agent_id = ?",
        (storage.test_did,),
    )


def _clock_ahead(monkeypatch, delta):
    """The hook's clock ``delta`` ahead: its next sleep comes later."""

    class _Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + delta

    monkeypatch.setattr(hook_module, "datetime", _Later)


# --- the window --------------------------------------------------------------


@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_a_memory_retrieved_hours_before_sleep_is_attested(storage, backend):
    """The reported case: retrieved at 14:25, attested at the 04:00 sleep."""
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    memory = "The user prefers concise weekly plans."
    retrieved_at = now - timedelta(hours=13, minutes=35)
    row_id = await _memory(storage, memory, retrieved_at=retrieved_at)
    await _turn(
        storage, "Plan my week.", "A concise plan: Monday writing, Tuesday review.",
        at=retrieved_at,
    )

    judge = _JUDGES[backend](applied={memory})
    agent = _Agent(storage, await _memory_system(storage), judge)
    hook = ReflectionSleepHook(backend=backend)
    result = await hook.on_pre_sleep(agent)

    assert result["success"] is True
    assert result["candidates"] == 1
    assert result["attested_message_ids"] == [row_id]
    assert (await _metadata(storage, row_id))["applied_count"] == 1
    assert hook._pre_sleep_status.get() is SleepHookStatus.SUCCESS


@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_each_memory_is_judged_against_its_own_turn(storage, backend):
    """Not against whatever was said last."""
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    plan, dog, tool = (
        "The user prefers concise weekly plans.",
        "The user's dog is named Biscuit.",
        "The user's sister lives in Lisbon.",
    )

    # Context assembly: retrieval just before the turn's user row is written.
    plan_at = _second(now - timedelta(hours=6)) + timedelta(milliseconds=200)
    await _memory(storage, plan, retrieved_at=plan_at)
    await _turn(storage, "Plan my week.", "A concise plan: Monday writing.", at=plan_at)

    # The same, with the user row written in the next second.
    dog_at = _second(now - timedelta(hours=3)) + timedelta(milliseconds=950)
    await _memory(storage, dog, retrieved_at=dog_at)
    await _turn(
        storage, "What should I feed my dog?", "Biscuit does well on kibble.",
        at=dog_at + timedelta(seconds=1),
    )

    # A memory tool: retrieved inside a turn whose user row came first.
    tool_turn = now - timedelta(hours=2)
    await _memory(storage, tool, retrieved_at=tool_turn + timedelta(seconds=2))
    await _turn(
        storage, "Where should I stay in Portugal?",
        "Your sister is in Lisbon; stay near her.", at=tool_turn,
    )

    # What was said last, and has nothing to do with any of them.
    await _turn(
        storage, "Thanks, that is all.", "You're welcome.",
        at=now - timedelta(minutes=10),
    )

    judge = _JUDGES[backend]()
    agent = _Agent(storage, await _memory_system(storage), judge)
    result = await ReflectionSleepHook(backend=backend).on_pre_sleep(agent)

    assert result["success"] is True and result["candidates"] == 3
    assert judge.sessions == {
        plan: "user: Plan my week.\nassistant: A concise plan: Monday writing.",
        dog: (
            "user: What should I feed my dog?\n"
            "assistant: Biscuit does well on kibble."
        ),
        tool: (
            "user: Where should I stay in Portugal?\n"
            "assistant: Your sister is in Lisbon; stay near her."
        ),
    }


async def test_the_window_starts_at_the_watermark_and_ends_at_the_pass(storage):
    """Retrieval time selects candidates, not row id; most recent 20 kept."""
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))

    # Written first, so the lowest id, and retrieved last in the window.
    oldest_row = await _memory(
        storage, "memory written first", retrieved_at=now - timedelta(minutes=30)
    )
    in_window = [oldest_row]
    for hour in range(1, 22):
        in_window.append(await _memory(
            storage, f"memory retrieved {hour}h ago",
            retrieved_at=now - timedelta(hours=hour),
        ))
    # Written later, so higher ids, but retrieved before the watermark.
    for index in range(85):
        await _memory(
            storage, f"retrieved before the window {index}",
            retrieved_at=now - timedelta(days=3),
        )
    # Retrieved within the settle period before the pass, or after it
    # started: the next pass's.
    settling = await _memory(
        storage, "retrieved just before sleep", retrieved_at=now - timedelta(minutes=1)
    )
    late = await _memory(
        storage, "retrieved during the pass", retrieved_at=now + timedelta(hours=1)
    )

    judge = _ChatJudge()
    agent = _Agent(storage, await _memory_system(storage), judge)
    result = await ReflectionSleepHook().on_pre_sleep(agent)

    assert result["candidates"] == 20
    judged = set(judge.sessions)
    assert "memory written first" in judged
    assert judged == {
        "memory written first",
        *(f"memory retrieved {hour}h ago" for hour in range(1, 20)),
    }
    assert {settling, late}.isdisjoint(result.get("attested_message_ids", []))


# --- the watermark -----------------------------------------------------------


@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_a_failed_pass_is_retried_and_only_a_completed_one_advances(
    storage, backend, monkeypatch
):
    now = datetime.now(timezone.utc)
    previous_pass = now - timedelta(hours=24)
    await _set_watermark(storage, previous_pass)
    first, second = "first retrieved memory", "second retrieved memory"
    first_id = await _memory(storage, first, retrieved_at=now - timedelta(hours=9))
    second_id = await _memory(storage, second, retrieved_at=now - timedelta(hours=4))
    memory_system = await _memory_system(storage)

    # The model fails on one memory: the pass fails after marking the other.
    failing = _JUDGES[backend](applied={first, second}, fail_on={second})
    hook = ReflectionSleepHook(backend=backend)
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, failing))
    assert result["success"] is False
    assert result["reason"] == "attestation_failed"
    assert hook._pre_sleep_status.get() is SleepHookStatus.FAILED
    assert await _watermark(storage) == previous_pass
    assert (await _metadata(storage, first_id))["applied_count"] == 1
    assert [row[0] for row in await _marks(storage)] == [first_id]

    # The next sleep covers the same window. The retrieval already attested
    # is not counted again.
    retry = _JUDGES[backend](applied={first, second})
    started = datetime.now(timezone.utc)
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, retry))
    finished = datetime.now(timezone.utc)
    assert result["success"] is True
    assert result["attested_message_ids"] == [second_id]
    assert set(retry.sessions) == {second}
    assert (await _metadata(storage, first_id))["applied_count"] == 1
    assert (await _metadata(storage, second_id))["applied_count"] == 1
    watermark = await _watermark(storage)
    assert started - ATTESTATION_SETTLE <= watermark <= finished - ATTESTATION_SETTLE
    # The watermark has passed every marked retrieval.
    assert await _marks(storage) == []

    # Nothing retrieved since: nothing attested, and the watermark moves on.
    idle = _JUDGES[backend](applied={first, second})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, idle))
    assert result["success"] is True and result["candidates"] == 0
    assert idle.sessions == {}
    assert await _watermark(storage) > watermark

    # Retrieved again after that pass, and attested by the next sleep.
    await _retrieved(storage, first_id, at=datetime.now(timezone.utc))
    _clock_ahead(monkeypatch, timedelta(days=1))
    again = _JUDGES[backend](applied={first})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, again))
    assert result["attested_message_ids"] == [first_id]
    assert (await _metadata(storage, first_id))["applied_count"] == 2


async def test_an_older_mark_does_not_hide_an_unjudged_retrieval(storage):
    """``last_applied`` newer than ``last_accessed`` is not proof of a judgment.

    A pass that judged an older retrieval can mark the memory after a turn
    retrieved it again. That newer retrieval was never judged, and a failed
    pass that did not reach it must not make the retry skip it.
    """
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    failing_memory, memory = "the memory the model fails on", "remarked memory"
    await _memory(storage, failing_memory, retrieved_at=now - timedelta(hours=9))
    row_id = await _memory(storage, memory, retrieved_at=now - timedelta(hours=6))
    await storage.conversation.update_message_metadata(row_id, {
        "applied_count": 1,
        "last_applied": (now - timedelta(hours=5)).isoformat(),
    })
    memory_system = await _memory_system(storage)
    hook = ReflectionSleepHook()

    # The chat backend judges oldest first, so this pass fails before it.
    failing = _ChatJudge(fail_on={failing_memory})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, failing))
    assert result["success"] is False
    assert memory not in failing.sessions

    retry = _ChatJudge(applied={memory})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, retry))
    assert result["success"] is True
    assert memory in retry.sessions
    assert result["attested_message_ids"] == [row_id]
    assert (await _metadata(storage, row_id))["applied_count"] == 2


async def test_a_later_retrieval_of_a_marked_memory_is_judged_again(storage):
    """The mark names the retrieval it judged, not the memory."""
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    memory, failing_memory = "marked memory", "the memory the model fails on"
    row_id = await _memory(storage, memory, retrieved_at=now - timedelta(hours=8))
    await _memory(storage, failing_memory, retrieved_at=now - timedelta(hours=7))
    memory_system = await _memory_system(storage)
    hook = ReflectionSleepHook(backend="decision")

    failing = _DecisionJudge(applied={memory}, fail_on={failing_memory})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, failing))
    assert result["success"] is False
    assert (await _metadata(storage, row_id))["applied_count"] == 1

    # Retrieved again before the retry: a retrieval no pass has judged.
    await _retrieved(storage, row_id, at=now - timedelta(hours=3))
    retry = _DecisionJudge(applied={memory})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, retry))
    assert result["success"] is True
    assert set(retry.sessions) == {memory, failing_memory}
    assert (await _metadata(storage, row_id))["applied_count"] == 2


async def test_a_retrieval_just_before_the_pass_is_left_to_the_next(
    storage, monkeypatch
):
    """Its ``last_accessed`` write may not have committed when the pass read."""
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    row_id = await _memory(
        storage, "retrieved just before sleep", retrieved_at=now - timedelta(minutes=1)
    )
    memory_system = await _memory_system(storage)
    hook = ReflectionSleepHook()

    judge = _ChatJudge(applied={"retrieved just before sleep"})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, judge))
    assert result["success"] is True and result["candidates"] == 0
    assert await _watermark(storage) < now - timedelta(minutes=1)

    _clock_ahead(monkeypatch, timedelta(days=1))
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, judge))
    assert result["attested_message_ids"] == [row_id]


async def test_a_first_pass_looks_back_a_day_and_keeps_that_window_on_failure(
    storage,
):
    now = datetime.now(timezone.utc)
    inside = await _memory(
        storage, "retrieved within a day", retrieved_at=now - timedelta(hours=23)
    )
    await _memory(
        storage, "retrieved before that", retrieved_at=now - timedelta(hours=25)
    )
    memory_system = await _memory_system(storage)
    hook = ReflectionSleepHook()

    started = datetime.now(timezone.utc)
    failing = _ChatJudge(fail_on={"retrieved within a day"})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, failing))
    finished = datetime.now(timezone.utc)
    assert result["success"] is False and result["candidates"] == 1
    bootstrap = await _watermark(storage)
    lookback = ATTESTATION_BOOTSTRAP_LOOKBACK
    assert started - lookback <= bootstrap <= finished - lookback

    retry = _ChatJudge(applied={"retrieved within a day"})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, retry))
    assert result["attested_message_ids"] == [inside]
    assert set(retry.sessions) == {"retrieved within a day"}
    assert await _watermark(storage) > bootstrap


async def test_the_watermark_never_moves_backwards(storage):
    later = datetime.now(timezone.utc)
    earlier = later - timedelta(hours=5)
    await _set_watermark(storage, later)
    await _set_watermark(storage, earlier)
    assert await _watermark(storage) == later


async def test_a_pass_completes_only_when_its_watermark_is_written(
    storage, monkeypatch
):
    """A pass whose watermark write fails did not complete."""
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    await _memory(storage, "a memory", retrieved_at=now - timedelta(hours=2))
    hook = ReflectionSleepHook()

    async def unwritable(db, *, agent_id, through):
        raise ConnectionError("database went away")

    monkeypatch.setattr(hook, "_advance_watermark", unwritable)
    agent = _Agent(storage, await _memory_system(storage), _ChatJudge())
    with pytest.raises(ConnectionError):
        await hook.on_pre_sleep(agent)

    assert hook._pre_sleep_status.get() is None
    assert await _watermark(storage) == now - timedelta(hours=24)
    post = await hook.on_post_consolidation(agent, {})
    assert post["reason"] == "pre_sleep_reflection_not_completed"


def _fail_next_application_write(storage, monkeypatch):
    """The store's next ``applied_count`` increment raises; later ones write.

    ``mark_applied`` reaches the store's ``atomic_increment_metadata_counter``
    through the real ``MemorySystem`` and its retriever. The failure is raised
    there, as the database would raise it, and nothing is written. Returns the
    ids whose write failed.
    """
    conversation = storage.conversation
    write = conversation.atomic_increment_metadata_counter
    failed = []

    async def fail_once(message_id, counter_field, timestamp_field=None):
        if counter_field == "applied_count" and not failed:
            failed.append(message_id)
            raise QueryError("application write failed")
        return await write(
            message_id, counter_field=counter_field, timestamp_field=timestamp_field
        )

    monkeypatch.setattr(conversation, "atomic_increment_metadata_counter", fail_once)
    return failed


@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_an_application_write_that_fails_is_judged_again(
    storage, backend, monkeypatch
):
    """A failed ``applied_count`` write is not recorded as an attestation.

    ``mark_applied`` used to log the storage error and return, and the pass
    then recorded the retrieval as marked and advanced past it: no sleep
    would judge it again.
    """
    now = datetime.now(timezone.utc)
    previous_pass = now - timedelta(hours=24)
    await _set_watermark(storage, previous_pass)
    first, second = "first retrieved memory", "second retrieved memory"
    first_id = await _memory(storage, first, retrieved_at=now - timedelta(hours=9))
    second_id = await _memory(storage, second, retrieved_at=now - timedelta(hours=4))
    memory_system = await _memory_system(storage)
    failed_writes = _fail_next_application_write(storage, monkeypatch)

    hook = ReflectionSleepHook(backend=backend)
    agent = _Agent(storage, memory_system, _JUDGES[backend](applied={first, second}))
    with pytest.raises(QueryError, match="application write failed"):
        await hook.on_pre_sleep(agent)

    assert failed_writes == [first_id]
    assert hook._pre_sleep_status.get() is None
    assert await _watermark(storage) == previous_pass
    assert await _marks(storage) == []
    assert "applied_count" not in await _metadata(storage, first_id)
    assert "applied_count" not in await _metadata(storage, second_id)
    post = await hook.on_post_consolidation(agent, {})
    assert post["reason"] == "pre_sleep_reflection_not_completed"

    # The write works again: the next sleep judges both retrievals.
    retry = _JUDGES[backend](applied={first, second})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, retry))
    assert result["success"] is True
    assert set(retry.sessions) == {first, second}
    assert result["attested_message_ids"] == [first_id, second_id]
    assert (await _metadata(storage, first_id))["applied_count"] == 1
    assert (await _metadata(storage, second_id))["applied_count"] == 1
    assert await _watermark(storage) > previous_pass


@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_an_application_that_matched_no_row_fails_the_pass(storage, backend):
    """The store's ``False`` (no row updated) is not a confirmed write."""
    now = datetime.now(timezone.utc)
    previous_pass = now - timedelta(hours=24)
    await _set_watermark(storage, previous_pass)
    gone, kept = "memory purged while judged", "memory still stored"
    gone_id = await _memory(storage, gone, retrieved_at=now - timedelta(hours=9))
    kept_id = await _memory(storage, kept, retrieved_at=now - timedelta(hours=4))
    memory_system = await _memory_system(storage)

    async def purge(memory):
        if memory == gone:
            await storage.db.execute(
                "DELETE FROM conversation_history WHERE id = ?", (gone_id,)
            )

    hook = ReflectionSleepHook(backend=backend)
    judge = _JUDGES[backend](applied={gone, kept}, on_judged=purge)
    with pytest.raises(MemoryApplicationNotRecorded):
        await hook.on_pre_sleep(_Agent(storage, memory_system, judge))

    assert hook._pre_sleep_status.get() is None
    assert await _watermark(storage) == previous_pass
    assert await _marks(storage) == []
    assert "applied_count" not in await _metadata(storage, kept_id)

    # The purged memory is no longer a candidate; the other is attested once.
    retry = _JUDGES[backend](applied={gone, kept})
    result = await hook.on_pre_sleep(_Agent(storage, memory_system, retry))
    assert result["success"] is True
    assert set(retry.sessions) == {kept}
    assert result["attested_message_ids"] == [kept_id]
    assert (await _metadata(storage, kept_id))["applied_count"] == 1
    assert await _watermark(storage) > previous_pass


# --- overlapping passes ------------------------------------------------------


def _hold_until_both_judging():
    """An ``on_judged`` that holds each pass until both are judging."""
    judging = []
    both = asyncio.Event()

    async def hold(memory):
        judging.append(memory)
        if len(judging) == 2:
            both.set()
        await both.wait()

    return hold


@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_overlapping_passes_count_a_retrieval_once(storage, backend, monkeypatch):
    """Two passes that judged the same retrieval record it once.

    A scheduled and a manual sleep can overlap, sharing the hook. Both select
    the retrieval before either records it, and the second to record must not
    increment ``applied_count`` again.
    """
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    memory = "the memory both passes judge"
    row_id = await _memory(storage, memory, retrieved_at=now - timedelta(hours=5))
    memory_system = await _memory_system(storage)
    hook = ReflectionSleepHook(backend=backend)

    # Neither pass moves its watermark until both have recorded.
    advance = hook._advance_watermark
    both_recorded = asyncio.Event()
    advancing = []
    marks_once_both_recorded = []

    async def advance_once_both_recorded(db, *, agent_id, through):
        advancing.append(through)
        if len(advancing) == 2:
            marks_once_both_recorded.extend(await _marks(storage))
            both_recorded.set()
        await both_recorded.wait()
        await advance(db, agent_id=agent_id, through=through)

    monkeypatch.setattr(hook, "_advance_watermark", advance_once_both_recorded)
    hold = _hold_until_both_judging()
    results = await asyncio.gather(*(
        hook.on_pre_sleep(_Agent(
            storage, memory_system,
            _JUDGES[backend](applied={memory}, on_judged=hold),
        ))
        for _ in range(2)
    ))

    assert [result["success"] for result in results] == [True, True]
    assert sorted(result["applied_count"] for result in results) == [0, 1]
    assert [row[0] for row in marks_once_both_recorded] == [row_id]
    assert (await _metadata(storage, row_id))["applied_count"] == 1


@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_a_pass_that_records_after_another_completed_does_not_recount(
    storage, backend
):
    """The other pass completed, and its watermark took the mark with it.

    The slower pass judged the same retrieval. That retrieval is now behind the
    watermark, so the slower pass's record must not count it again.
    """
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    memory = "the memory both passes judge"
    row_id = await _memory(storage, memory, retrieved_at=now - timedelta(hours=5))
    memory_system = await _memory_system(storage)
    hook = ReflectionSleepHook(backend=backend)

    judging = asyncio.Event()
    other_completed = asyncio.Event()

    async def hold(_memory):
        judging.set()
        await other_completed.wait()

    slower = asyncio.create_task(hook.on_pre_sleep(_Agent(
        storage, memory_system, _JUDGES[backend](applied={memory}, on_judged=hold)
    )))
    await judging.wait()
    completed = await hook.on_pre_sleep(_Agent(
        storage, memory_system, _JUDGES[backend](applied={memory})
    ))
    assert completed["attested_message_ids"] == [row_id]
    assert await _marks(storage) == []
    other_completed.set()
    result = await slower

    assert result["success"] is True
    assert result["applied_count"] == 0
    assert (await _metadata(storage, row_id))["applied_count"] == 1
    assert await _marks(storage) == []


class _CommitOnRelease:
    """The database, with each transaction held open after its last write."""

    def __init__(self, db):
        self._db = db
        self.written = asyncio.Event()
        self.release = asyncio.Event()

    def __getattr__(self, name):
        return getattr(self._db, name)

    @asynccontextmanager
    async def transaction(self, **options):
        async with self._db.transaction(**options):
            yield
            self.written.set()
            await self.release.wait()


async def _until_waiting_on_a_lock(storage):
    """Until a statement on the attestation tables waits on a lock.

    PostgreSQL shows it; SQLite serializes inside the process, through the
    connection's write guard, and shows nothing, so there this only yields.
    """
    if storage.db.backend_type != "postgres":
        await asyncio.sleep(0.2)
        return
    for _ in range(200):
        row = await storage.db.fetchone(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE datname = current_database() AND wait_event_type = 'Lock' "
            "AND query LIKE '%memory_attestation%'"
        )
        if row[0]:
            return
        await asyncio.sleep(0.05)
    pytest.fail("the record never waited on the advance's transaction")


async def test_a_record_racing_a_watermark_advance_does_not_recount(storage):
    """A record that starts while an advance is committing waits for it.

    The advance has moved the watermark past the retrieval and dropped its
    mark, but not committed. On PostgreSQL the record's insert would otherwise
    read the old watermark, wait out the mark's deletion, and count again.
    """
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    retrieved_at = now - timedelta(hours=5)
    row_id = await _memory(storage, "a counted memory", retrieved_at=retrieved_at)
    memory_system = await _memory_system(storage)
    marks = RetrievalMarks(storage.db, storage.test_did)
    candidate = RetrievedMemoryCandidate(
        message_id=row_id, content="a counted memory",
        retrieved_at=retrieved_at.isoformat(), created_at=None, role="assistant",
    )
    assert await marks.record(memory_system, candidate, reason="Used it.") is True

    held = _CommitOnRelease(storage.db)
    advance = asyncio.create_task(ReflectionSleepHook()._advance_watermark(
        held, agent_id=storage.test_did, through=now - timedelta(minutes=5)
    ))
    await held.written.wait()
    racing = asyncio.create_task(
        marks.record(memory_system, candidate, reason="Used it.")
    )
    await _until_waiting_on_a_lock(storage)
    held.release.set()
    await advance

    assert await racing is False
    assert (await _metadata(storage, row_id))["applied_count"] == 1
    assert await _marks(storage) == []


# --- privacy -----------------------------------------------------------------


class _SleepingAgent(_Agent, SleepMixin):
    """The agent's own sleep cycle around the hook, consolidation included."""

    def __init__(self, storage, memory_system, llm_service, hook):
        super().__init__(storage, memory_system, llm_service)
        self.sleep_hooks = [hook]


async def _governed_memory_system(storage, privacy, lock):
    """``MemorySystem`` as the agent builds it, governed by ``privacy``."""
    memory_system = MemorySystem(
        storage, storage.test_did, privacy_storage=privacy, transition_lock=lock
    )
    await memory_system.initialize()
    return memory_system


async def _attestation_rows(storage):
    """This agent's rows in either attestation table, where it exists."""
    rows = []
    for table in (ATTESTATION_STATE_TABLE, ATTESTATION_MARKS_TABLE):
        if await storage.db.table_exists(table):
            rows += await storage.db.fetchall(
                f"SELECT * FROM {table} WHERE agent_id = ?", (storage.test_did,)
            )
    return rows


@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_a_sleep_in_a_no_storage_mode_writes_nothing(storage, backend):
    """EPHEMERAL: no watermark, mark or application, and the purge is clean.

    The pass used to create its tables and store this agent's watermark
    before anything read the mode. ``purge_ephemeral_session`` does not sweep
    those tables, so it reported a clean purge with the row still there.
    """
    now = datetime.now(timezone.utc)
    memory = "a memory retrieved before the agent went off the record"
    row_id = await _memory(storage, memory, retrieved_at=now - timedelta(hours=3))
    private = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
    memory_system = await _governed_memory_system(
        storage, private, ReentrantTransitionLock()
    )
    judge = _JUDGES[backend](applied={memory})
    agent = _SleepingAgent(
        storage, memory_system, judge, ReflectionSleepHook(backend=backend)
    )
    agent.storage = private

    report = await agent.sleep(skip_export=True)

    (pre,) = report.pre_reflection
    assert pre["skipped"] is True
    assert pre["reason"] == "privacy_mode_forbids_persistence"
    assert judge.sessions == {}
    assert await _attestation_rows(storage) == []
    if storage.db.backend_type == "sqlite":
        # This test's own database: not even the tables were created.
        assert not await storage.db.table_exists(ATTESTATION_STATE_TABLE)
    assert "applied_count" not in await _metadata(storage, row_id)

    purge = await private.purge_ephemeral_session(reason="test")
    assert purge.leaked_rows == 0
    assert not purge.any_sweep_failed
    assert await _attestation_rows(storage) == []


async def test_a_memory_that_cannot_say_whether_it_may_write_is_not_attested():
    """Without ``durable_writes`` the pass cannot know the mode allows it."""
    memory = SimpleNamespace(mark_applied=AsyncMock())
    agent = SimpleNamespace(memory_system=memory, did="did:test:no-write-scope")
    hook = ReflectionSleepHook()
    result = await hook.on_pre_sleep(agent)
    assert result["skipped"] is True
    assert result["reason"] == "memory_system_unavailable"
    memory.mark_applied.assert_not_awaited()


@pytest.mark.parametrize("applied", [True, False])
@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_a_transition_during_the_pass_stops_its_writes(storage, backend, applied):
    """Each write reads the mode under the transition lock.

    The transition holds the lock while it changes the mode. The pass's next
    write (the application, or with nothing applied the watermark) waits for
    it, then finds the mode forbids persistence and writes nothing. Read
    before taking the lock, the mode would still say NORMAL.
    """
    now = datetime.now(timezone.utc)
    previous_pass = now - timedelta(hours=24)
    await _set_watermark(storage, previous_pass)
    memory = "a memory judged while the mode changes"
    row_id = await _memory(storage, memory, retrieved_at=now - timedelta(hours=3))
    private = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    lock = ReentrantTransitionLock()
    memory_system = await _governed_memory_system(storage, private, lock)

    transition_holds_the_lock = asyncio.Event()

    async def transition():
        async with lock:
            transition_holds_the_lock.set()
            # Time for a write that did not wait for the lock to land first.
            await asyncio.sleep(0.5)
            private.set_privacy_mode(PrivacyMode.EPHEMERAL)

    transitions = []

    async def while_judging(_memory):
        transitions.append(asyncio.create_task(transition()))
        await transition_holds_the_lock.wait()

    hook = ReflectionSleepHook(backend=backend)
    judge = _JUDGES[backend](
        applied={memory} if applied else (), on_judged=while_judging
    )
    agent = _Agent(storage, memory_system, judge)
    agent.storage = private
    result = await hook.on_pre_sleep(agent)
    await asyncio.gather(*transitions)

    assert set(judge.sessions) == {memory}
    assert result["skipped"] is True
    assert result["reason"] == "privacy_mode_forbids_persistence"
    assert hook._pre_sleep_status.get() is SleepHookStatus.SKIPPED
    assert "applied_count" not in await _metadata(storage, row_id)
    assert await _marks(storage) == []
    assert await _watermark(storage) == previous_pass


# --- archival ----------------------------------------------------------------


async def _decayed_memory(storage, content, *, retrieved_at):
    """240 days old, importance 0.5, retrieved once: decayed below the
    archive threshold unless an attestation applies it."""
    row_id = await _add(
        storage, "assistant", content,
        at=datetime.now(timezone.utc) - timedelta(days=240),
    )
    await storage.conversation.update_message_metadata(
        row_id,
        {
            "importance": 0.5,
            "access_count": 1,
            "last_accessed": retrieved_at.isoformat(),
        },
    )
    return row_id


async def _archived(storage, row_id):
    row = await storage.db.fetchone(
        "SELECT archived_at FROM conversation_history WHERE id = ?", (row_id,)
    )
    return row[0] is not None


async def _require_archival(storage, control):
    """``control``, decayed and judged long ago, was archived by the sleep.

    On PostgreSQL nothing is: ``created_at`` is ``timestamp without time
    zone``, so ``calculate_decay`` subtracts a naive value from an aware now
    and the ``TypeError`` it catches makes every row's strength 0.5. That is
    so before and after #3544, and until it is fixed a PostgreSQL case cannot
    show what archival leaves alone.
    """
    if await _archived(storage, control):
        return
    if storage.db.backend_type == "postgres":
        pytest.xfail(
            "decay archival archives nothing on PostgreSQL: calculate_decay "
            "gets a naive created_at (pre-existing; reported on #3544)"
        )
    pytest.fail("the decayed control memory was not archived")


@pytest.mark.parametrize("applied_next_sleep", [True, False])
@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_a_retrieval_left_to_the_next_pass_is_not_archived_before_it(
    storage, backend, applied_next_sleep, monkeypatch
):
    """Codex's case: retrieved inside the settle margin, then archived.

    The pass left the retrieval to the next one, and the same sleep's
    consolidation archived it. The next pass's application raised its
    ``applied_count`` but never cleared ``archived_at``.
    """
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    deferred_memory = "retrieved a minute before sleep"
    deferred = await _decayed_memory(
        storage, deferred_memory, retrieved_at=now - timedelta(minutes=1)
    )
    # Retrieved before the watermark: judged by an earlier pass, so archival
    # runs over it as before.
    judged = await _decayed_memory(
        storage, "retrieved before the last sleep", retrieved_at=now - timedelta(days=3)
    )
    memory_system = await _memory_system(storage)
    hook = ReflectionSleepHook(backend=backend)

    judge = _JUDGES[backend](applied={deferred_memory})
    report = await _SleepingAgent(storage, memory_system, judge, hook).sleep(
        skip_export=True
    )
    assert judge.sessions == {}
    assert not await _archived(storage, deferred)
    await _require_archival(storage, judged)
    assert report.messages_archived == 1

    # The next sleep judges it. Applied, it is kept; otherwise it decays.
    _clock_ahead(monkeypatch, timedelta(days=1))
    judge = _JUDGES[backend](applied={deferred_memory} if applied_next_sleep else ())
    await _SleepingAgent(storage, memory_system, judge, hook).sleep(skip_export=True)
    assert set(judge.sessions) == {deferred_memory}
    assert await _archived(storage, deferred) is not applied_next_sleep


@pytest.mark.parametrize("backend", ["chat", "decision"])
async def test_a_retrieval_whose_pass_failed_is_not_archived_before_it_is_judged(
    storage, backend
):
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    memory = "a memory whose judgment failed"
    row_id = await _decayed_memory(
        storage, memory, retrieved_at=now - timedelta(hours=5)
    )
    judged = await _decayed_memory(
        storage, "retrieved before the last sleep", retrieved_at=now - timedelta(days=3)
    )
    memory_system = await _memory_system(storage)
    hook = ReflectionSleepHook(backend=backend)

    failing = _JUDGES[backend](fail_on={memory})
    await _SleepingAgent(storage, memory_system, failing, hook).sleep(skip_export=True)
    assert not await _archived(storage, row_id)
    await _require_archival(storage, judged)

    retry = _JUDGES[backend](applied={memory})
    await _SleepingAgent(storage, memory_system, retry, hook).sleep(skip_export=True)
    assert (await _metadata(storage, row_id))["applied_count"] == 1
    assert not await _archived(storage, row_id)


async def test_consolidation_leaves_a_retrieval_at_the_watermark(storage):
    """A retrieval at the watermark itself is in the next pass's window."""
    watermark = datetime.now(timezone.utc) - timedelta(hours=2)
    await _set_watermark(storage, watermark)
    at_watermark = await _decayed_memory(
        storage, "retrieved at the watermark", retrieved_at=watermark
    )
    judged = await _decayed_memory(
        storage, "retrieved just before it",
        retrieved_at=watermark - timedelta(microseconds=1),
    )
    memory_system = await _memory_system(storage)
    await memory_system.consolidate()
    assert not await _archived(storage, at_watermark)
    await _require_archival(storage, judged)


# --- the tables --------------------------------------------------------------


async def _no_schema(_db):
    return None


@pytest.fixture(params=["sqlite", "postgres"])
async def fresh_databases(request, tmp_path):
    """Eight connections to one database the attestation tables are not in."""
    if request.param == "postgres":
        url = os.environ.get("TEST_POSTGRES_URL")
        if not url:
            pytest.skip("TEST_POSTGRES_URL is not set")
        admin = await AsyncDatabase.postgres(url, schema_initializer=_no_schema)
        try:
            async with disposable_postgres_schema(admin, "attestation_tables") as schema:
                handles = [
                    await AsyncDatabase.postgres(
                        with_search_path(url, schema), schema_initializer=_no_schema
                    )
                    for _ in range(8)
                ]
                try:
                    yield handles
                finally:
                    for handle in handles:
                        await handle.close()
        finally:
            await admin.close()
    else:
        path = str(tmp_path / "fresh.db")
        handles = [
            await AsyncDatabase.sqlite(path, schema_initializer=_no_schema)
            for _ in range(8)
        ]
        try:
            yield handles
        finally:
            for handle in handles:
                await handle.close()


async def test_first_passes_on_many_connections_create_the_tables_once(
    fresh_databases,
):
    """Concurrent first passes create the tables without colliding.

    Two PostgreSQL sessions that both ran the bare ``CREATE TABLE IF NOT
    EXISTS`` could both find the table missing, and one then failed on the
    catalogue's unique index.
    """
    outcomes = await asyncio.gather(
        *(ensure_attestation_tables(db) for db in fresh_databases),
        return_exceptions=True,
    )
    assert outcomes == [None] * len(fresh_databases)
    for table in (ATTESTATION_STATE_TABLE, ATTESTATION_MARKS_TABLE):
        assert await fresh_databases[0].table_exists(table)


async def test_a_memory_whose_turn_was_never_written_is_judged_without_one(storage):
    """A retrieval whose turn failed before its user row: no other turn's."""
    now = datetime.now(timezone.utc)
    await _set_watermark(storage, now - timedelta(hours=24))
    await _turn(storage, "An earlier question.", "An earlier answer.",
                at=now - timedelta(hours=3))
    await _memory(storage, "orphaned retrieval", retrieved_at=now - timedelta(hours=2))

    judge = _ChatJudge()
    agent = _Agent(storage, await _memory_system(storage), judge)
    await ReflectionSleepHook().on_pre_sleep(agent)
    assert judge.sessions == {
        "orphaned retrieval": "(no recent session context available)"
    }


# --- the store read ----------------------------------------------------------


async def test_messages_around_split_at_the_instant_in_canonical_order(storage):
    at = _second(datetime.now(timezone.utc) - timedelta(hours=1))
    before_ids = [
        await _add(storage, "user", f"before {index}", at=at - timedelta(minutes=index))
        for index in (3, 2, 1)
    ]
    same_second = await _add(storage, "assistant", "same second", at=at)
    after_id = await _add(storage, "user", "after", at=at + timedelta(minutes=1))
    trashed = await _add(storage, "user", "trashed", at=at + timedelta(minutes=2))
    await storage.conversation.delete_message(trashed)

    earlier, later = await storage.conversation.get_messages_around(
        at, before=2, after=10
    )
    assert [row["id"] for row in earlier] == before_ids[1:]
    assert [row["id"] for row in later] == [same_second, after_id]
    assert later[0]["content"] == "same second"


# --- turn selection ----------------------------------------------------------


_T = datetime(2026, 10, 10, 14, 25, 3, tzinfo=timezone.utc)


def _row(role, content, *, at):
    return {
        "role": role,
        "content": content,
        "metadata": {},
        "created_at": at.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _split(rows, second=_T):
    earlier = [r for r in rows if r["created_at"] < second.strftime("%Y-%m-%d %H:%M:%S")]
    later = [r for r in rows if r not in earlier]
    return earlier, later


def _contents(turn):
    return [row["content"] for row in turn]


def test_turn_of_a_context_retrieval_in_the_user_rows_second():
    rows = [
        _row("user", "u0", at=_T - timedelta(minutes=5)),
        _row("assistant", "r0", at=_T - timedelta(minutes=4)),
        _row("user", "u1", at=_T),
        _row("assistant", "r1", at=_T + timedelta(seconds=6)),
        _row("user", "u2", at=_T + timedelta(minutes=3)),
    ]
    assert _contents(retrieving_turn(*_split(rows), _T)) == ["u1", "r1"]


def test_turn_of_a_context_retrieval_whose_user_row_is_a_second_later():
    rows = [
        _row("user", "u0", at=_T - timedelta(minutes=5)),
        _row("assistant", "r0", at=_T - timedelta(minutes=4)),
        _row("user", "u1", at=_T + timedelta(seconds=1)),
        _row("assistant", "r1", at=_T + timedelta(seconds=6)),
    ]
    assert _contents(retrieving_turn(*_split(rows), _T)) == ["u1", "r1"]


def test_turn_of_a_tool_retrieval_inside_an_open_turn():
    rows = [
        _row("user", "u1", at=_T - timedelta(seconds=8)),
        _row("assistant", "r1", at=_T + timedelta(seconds=5)),
        _row("user", "u2", at=_T + timedelta(minutes=2)),
    ]
    assert _contents(retrieving_turn(*_split(rows), _T)) == ["u1", "r1"]


def test_a_wake_right_after_a_reply_is_its_own_turn():
    rows = [
        _row("user", "u0", at=_T - timedelta(seconds=30)),
        _row("assistant", "r0", at=_T),
        _row("user", "wake", at=_T),
        _row("assistant", "r1", at=_T + timedelta(seconds=9)),
    ]
    assert _contents(retrieving_turn(*_split(rows), _T)) == ["wake", "r1"]


def test_a_wake_whose_row_follows_a_reply_in_the_retrievals_second():
    """The previous reply shares the retrieval's second; the wake does not."""
    rows = [
        _row("user", "u0", at=_T - timedelta(seconds=5)),
        _row("assistant", "r0", at=_T),
        _row("user", "wake", at=_T + timedelta(seconds=1)),
        _row("assistant", "r1", at=_T + timedelta(seconds=9)),
    ]
    assert _contents(retrieving_turn(*_split(rows), _T)) == ["wake", "r1"]


def test_a_tool_retrievals_turn_answered_within_its_second_is_kept():
    rows = [
        _row("user", "u1", at=_T - timedelta(seconds=8)),
        _row("assistant", "r1", at=_T),
    ]
    assert _contents(retrieving_turn(*_split(rows), _T)) == ["u1", "r1"]


def test_an_unanswered_earlier_turn_is_not_the_retrievals():
    rows = [
        _row("user", "failed turn", at=_T - timedelta(minutes=2)),
        _row("user", "u1", at=_T + timedelta(seconds=1)),
        _row("assistant", "r1", at=_T + timedelta(seconds=7)),
    ]
    assert _contents(retrieving_turn(*_split(rows), _T)) == ["u1", "r1"]


def test_no_turn_when_the_retrievals_turn_was_never_written():
    rows = [
        _row("user", "u0", at=_T - timedelta(minutes=5)),
        _row("assistant", "r0", at=_T - timedelta(minutes=4)),
    ]
    assert retrieving_turn(*_split(rows), _T) == []


def test_a_user_row_a_session_gap_later_is_not_the_retrievals_turn():
    """Context assembly failed before writing the user row: no turn."""
    rows = [
        _row("user", "u0", at=_T - timedelta(minutes=5)),
        _row("assistant", "r0", at=_T - timedelta(minutes=4)),
        _row("user", "next session", at=_T + timedelta(minutes=30)),
        _row("assistant", "r1", at=_T + timedelta(minutes=31)),
    ]
    assert retrieving_turn(*_split(rows), _T) == []
    rows[2] = _row("user", "same session", at=_T + timedelta(minutes=29))
    assert _contents(retrieving_turn(*_split(rows), _T)) == ["same session", "r1"]


def test_a_same_second_reply_keeps_its_turn_when_the_next_is_a_session_away():
    rows = [
        _row("user", "u1", at=_T - timedelta(seconds=8)),
        _row("assistant", "r1", at=_T),
        _row("user", "tomorrow", at=_T + timedelta(hours=20)),
    ]
    assert _contents(retrieving_turn(*_split(rows), _T)) == ["u1", "r1"]


def test_an_operator_notice_stored_as_a_user_row_stays_inside_its_turn():
    """A fallback-route notice is written after the turn's user row."""
    notice = {
        **_row("user", "operator notice", at=_T + timedelta(seconds=1)),
        "metadata": {"operator_signal": True},
    }
    # Context assembly: the notice must not end the turn before its reply.
    rows = [
        _row("user", "u1", at=_T),
        notice,
        _row("assistant", "r1", at=_T + timedelta(seconds=6)),
    ]
    assert _contents(retrieving_turn(*_split(rows), _T)) == [
        "u1", "operator notice", "r1",
    ]
    # A memory tool: the open turn is the user row's, not the notice's.
    rows = [
        _row("user", "u1", at=_T - timedelta(seconds=8)),
        {**notice, "created_at": (_T - timedelta(seconds=7)).strftime(
            "%Y-%m-%d %H:%M:%S")},
        _row("assistant", "r1", at=_T + timedelta(seconds=5)),
        _row("user", "u2", at=_T + timedelta(minutes=2)),
    ]
    assert _contents(retrieving_turn(*_split(rows), _T)) == [
        "u1", "operator notice", "r1",
    ]


def test_rendering_omits_excluded_rows_and_keeps_the_reply():
    turn = [
        _row("user", "question", at=_T),
        {**_row("assistant", "excluded", at=_T), "metadata": {"excluded_from_context": True}},
        _row("assistant", "x" * 7000 + " the end", at=_T),
    ]
    rendered = render_turn(turn)
    assert "excluded" not in rendered
    assert rendered.endswith("the end")
    assert len(rendered) == 6000
