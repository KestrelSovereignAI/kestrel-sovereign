"""Actual completed/background owner and lifecycle receipt regressions."""

import asyncio
from types import SimpleNamespace
import pytest
from kestrel_sdk.storage.database import QueryError
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_commit_outcome,
    is_execution_control_error,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["ordinary", "resident"])
@pytest.mark.parametrize("carrier", ["error", "cancel"])
@pytest.mark.parametrize("closer", ["feature", "agent"])
@pytest.mark.parametrize("phase", ["completed", "during_close"])
async def test_completed_feature_worker_receipt_survives_actual_retirement(
    tmp_path, route, carrier, closer, phase
):
    async def nothing():
        pass

    agent = KestrelAgent(
        did="did:test:feature-receipt",
        storage_path=str(tmp_path / "agent.db"),
        llm_service=SimpleNamespace(close=nothing),
    )
    db = await AsyncDatabase.sqlite(str(tmp_path / "native.db"))
    await db.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    storage = AsyncStorage(backend=db.backend)
    storage.db = db
    agent.storage = agent._raw_storage = storage

    class Native(Feature):
        async def initialize(self):
            pass

        @property
        def tool_description(self):
            return "actual native receipt fixture"

    feature = Native(agent)
    agent.features = {"Native": feature}
    marker = ExecutionCommitOutcomeError("unknown")
    completed = asyncio.Event()
    entered = asyncio.Event()
    feature._register_sleep_hook(agent, object())

    async def worker():
        await db.execute("INSERT INTO effects VALUES (1)")
        entered.set()
        if phase == "during_close":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
        wrapper = (
            asyncio.CancelledError("original completed worker cancellation")
            if carrier == "cancel"
            else QueryError("original native effect receipt")
        )
        raise wrapper from marker

    track = (
        feature._track_owned_runtime_task
        if route == "resident"
        else feature._track_owned_background_task
    )
    task = track(worker(), name="actual-native-receipt")
    task.add_done_callback(lambda _done: completed.set())
    agent._runtime_publication_ready.set()
    await asyncio.wait_for(entered.wait(), 10)
    if phase == "completed":
        await asyncio.wait_for(completed.wait(), 10)
        assert (
            task.done()
            and not agent._background_tasks
            and not feature._owned_background_tasks
        )
    assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    try:
        with pytest.raises(BaseException) as caught:
            if closer == "feature":
                await feature.shutdown()
            else:
                await agent.shutdown()
        assert execution_commit_outcome(caught.value) == "unknown"
        assert task.done() and not feature._owned_background_tasks
        assert not feature._owned_sleep_hooks
    finally:
        try:
            await agent.shutdown()
        except BaseException as error:
            assert is_execution_control_error(error)


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_task_receipt_is_shared_by_completion_harvest_and_repeated_joins(carrier):
    from kestrel_sovereign._async_ownership import owned_task_error, await_owned_task

    marker = ExecutionCommitOutcomeError("unknown")

    async def worker():
        wrapper = (
            asyncio.CancelledError("first native cancellation")
            if carrier == "cancel"
            else QueryError("first native receipt")
        )
        raise wrapper from marker

    task = asyncio.create_task(worker())
    completed = asyncio.Event()
    task.add_done_callback(lambda _done: completed.set())
    await completed.wait()
    first = owned_task_error(task)
    assert execution_commit_outcome(first) == "unknown"
    for _ in range(2):
        outcome = await await_owned_task(task)
        assert outcome.error is first
        assert outcome.cancellation is None
        assert execution_commit_outcome(outcome.error) == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel", "group"])
async def test_actual_feature_unregister_keeps_control_ahead_of_ordinary_first_failure(
    tmp_path, carrier
):
    async def nothing():
        pass

    agent = KestrelAgent(
        did="did:test:unregister",
        storage_path=str(tmp_path / "agent.db"),
        llm_service=SimpleNamespace(close=nothing),
    )
    db = await AsyncDatabase.sqlite(str(tmp_path / "native.db"))
    await db.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    storage = AsyncStorage(backend=db.backend)
    storage.db = db
    agent.storage = agent._raw_storage = storage
    marker = ExecutionCommitOutcomeError("unknown")

    class Native(Feature):
        async def initialize(self):
            pass

        @property
        def tool_description(self):
            return "actual unregistration fixture"

        async def on_disable(self):
            raise ValueError("ordinary first cleanup failure")

        async def shutdown(self):
            await super().shutdown()
            await db.execute("INSERT INTO effects VALUES (1)")
            wrapper = (
                asyncio.CancelledError("native cleanup interruption")
                if carrier != "error"
                else QueryError("native cleanup receipt")
            )
            wrapper.__cause__ = marker
            if carrier == "group":
                raise BaseExceptionGroup(
                    "actual cleanup aggregate",
                    [wrapper, ValueError("another cleanup failure")],
                )
            raise wrapper

    feature = Native(agent)
    agent.features = {"Native": feature}
    feature._register_sleep_hook(agent, object())
    try:
        with pytest.raises(BaseException) as caught:
            await agent._unregister_feature_runtime(feature)
        assert execution_commit_outcome(caught.value) == "unknown"
        assert "Native" not in agent.features and not feature.enabled
        assert not feature._owned_sleep_hooks
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        # Prevent repeating this fault-injection effect during the retained
        # feature's later real shutdown. The original failure is still held.
        feature.shutdown = nothing
        try:
            await agent.shutdown()
        except BaseException as error:
            assert is_execution_control_error(error)
