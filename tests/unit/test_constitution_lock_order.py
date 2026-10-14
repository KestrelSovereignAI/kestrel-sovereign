"""Real file-backed SQLite lock ordering, with event barriers and bounded cleanup."""

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files
from tests.unit.test_constitution_audit import _DurableConstitutionHarness

# Only a hang guard. Every step here is ordered by an event or by lock state,
# never by elapsed time, so a loaded host cannot decide the outcome (#3614).
_HANG_GUARD_SECONDS = 30


@pytest.fixture(autouse=True)
def _no_ambient_constitution_authority(monkeypatch):
    """The signed writers pass an explicit per-test trust root. A root or
    source descriptor exported by the host would make it ambiguous, and
    reanchor would refuse before ever reaching the lock under test."""
    monkeypatch.delenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH", raising=False)
    monkeypatch.delenv("KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH", raising=False)


class _ConstitutionLockInversion(Exception):
    """The caller-transaction owner tried to wait on the constitution lock."""


def _task_outcome(task):
    if task.cancelled():
        return "cancellation"
    if task.exception() is not None:
        return repr(task.exception())
    return f"result {task.result()!r}"


async def _wait_for_step(step, event, task):
    """Wait until ``task`` reaches ``step``; fail at once if it ends first."""
    reached = asyncio.ensure_future(event.wait())
    try:
        done, _ = await asyncio.wait(
            {reached, task}, timeout=_HANG_GUARD_SECONDS, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        if not reached.done():
            reached.cancel()
            with suppress(asyncio.CancelledError):
                await reached
    if reached in done:
        return
    if task in done:
        pytest.fail(f"{step}: the task ended first with {_task_outcome(task)}")
    pytest.fail(f"{step}: not reached within the {_HANG_GUARD_SECONDS}s hang guard")


async def _await_task(step, task):
    done, _ = await asyncio.wait({task}, timeout=_HANG_GUARD_SECONDS)
    if task not in done:
        pytest.fail(f"{step}: not finished within the {_HANG_GUARD_SECONDS}s hang guard")
    return task.result()


async def _refuse_without_lock_wait(step, lock, transition):
    """Run ``transition()`` in THIS transaction-owning task.

    Another task holds ``lock`` and is waiting for this task's transaction, so
    this task reaching ``lock.acquire`` IS the deadlock. It is recorded and
    failed when it happens rather than inferred from a clock. A task-switching
    wrapper such as ``wait_for(coroutine)`` would erase the ownership relation
    under test; ``asyncio.timeout`` keeps the same task and is only a hang guard.
    """
    if not lock.locked():
        pytest.fail(f"{step}: precondition: another task must hold the constitution lock")
    owner = asyncio.current_task()
    original_acquire = lock.acquire
    waits = []

    async def refuse_owner_acquire():
        if asyncio.current_task() is owner:
            waits.append(step)
            raise _ConstitutionLockInversion(step)
        return await original_acquire()

    lock.acquire = refuse_owner_acquire
    guard = asyncio.timeout(_HANG_GUARD_SECONDS)
    result = None
    try:
        async with guard:
            result = await transition()
    except _ConstitutionLockInversion:
        pass  # Failed below from ``waits``, which also survives a swallowed raise.
    except TimeoutError:
        if not guard.expired():
            raise
        pytest.fail(
            f"{step}: did not return within the {_HANG_GUARD_SECONDS}s hang guard "
            "while owning the caller transaction"
        )
    finally:
        del lock.acquire
        # Checked on every exit, so an exception raised after the refusal was
        # caught cannot carry the test past a recorded wait (test 1 expects one).
        if waits:
            pytest.fail(
                f"{step}: the caller-transaction owner waited on the constitution lock "
                "held by a task waiting for that transaction: lock inversion deadlock"
            )
    return result


@pytest.mark.asyncio
async def test_implicit_rollback_keeps_caller_scope_visible_before_state_lock(tmp_path):
    """Native rollback must not erase a caller's still-held writer lock."""
    from kestrel_sovereign.storage.db.interface import QueryError, TransactionError

    storage = AsyncStorage(str(tmp_path / "rollback.db"), backend="sqlite", agent_id="did:test:rolled-back:" + uuid4().hex)
    await storage.initialize()
    waiter = None
    waiting = asyncio.Event()
    try:
        agent = _DurableConstitutionHarness(storage, datetime.now(timezone.utc))
        agent.agent_id = storage.agent_id
        await agent._initialize_constitution_runtime_state(is_new_identity=True)
        before = await agent._constitution_state_store.load(agent.agent_id)
        await storage.db.execute_commit("CREATE TABLE rollback_conflict (id INTEGER PRIMARY KEY)")
        await storage.db.execute_commit("INSERT INTO rollback_conflict VALUES (1)")

        async def competing_state_owner():
            async with ConstitutionMixin._constitution_state_guard(agent):
                waiting.set()
                async with storage.transaction():
                    return True

        with pytest.raises(TransactionError, match="rolled back implicitly"):
            async with storage.transaction():
                waiter = asyncio.create_task(competing_state_owner())
                await _wait_for_step("competing owner holds the constitution lock", waiting, waiter)
                with pytest.raises(QueryError, match="UNIQUE"):
                    await storage._backend.execute("INSERT OR ROLLBACK INTO rollback_conflict VALUES (1)")
                assert storage._backend._connection.in_transaction is False
                # SAME task still holds the writer scope after the implicit
                # rollback, so a wait here reopens the state/writer inversion.
                result = await _refuse_without_lock_wait(
                    "safe-mode entry after implicit rollback",
                    agent._constitution_state_lock,
                    lambda: agent.enter_safe_mode("caller caught implicit rollback"),
                )
                assert result is False
                assert agent._safe_mode is True and agent._constitution_state_persistence_pending is True
        assert await _await_task("competing owner commits after the caller scope ends", waiter) is True
        assert await agent._constitution_state_store.load(agent.agent_id) == before
    finally:
        if waiter is not None:
            if not waiter.done():
                waiter.cancel()
            with suppress(asyncio.CancelledError):
                await waiter
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor_writer", ["automatic", "signed_same", "signed_new"])
@pytest.mark.parametrize("transition", ["entry", "exit", "audit_begin", "audit_record", "audit_run", "periodic"])
async def test_outer_transaction_refuses_before_constitution_lock_wait(
    tmp_path, monkeypatch, anchor_writer, transition
):
    storage = AsyncStorage(str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id="did:test:lock-order:" + uuid4().hex)
    await storage.initialize()
    anchor_task = None
    allow_anchor = asyncio.Event()
    try:
        agent = _DurableConstitutionHarness(storage, datetime.now(timezone.utc))
        agent.agent_id = storage.agent_id
        await agent._initialize_constitution_runtime_state(is_new_identity=True)
        agent.extension = None
        agent._anchor_constitution_governance = KestrelAgent._anchor_constitution_governance.__get__(agent)
        agent._agent_signing_dids = ConstitutionMixin._agent_signing_dids.__get__(agent)
        agent._trusted_sovereign_did_document = ConstitutionMixin._trusted_sovereign_did_document.__get__(agent)
        content = resolve_governing_constitution_bytes(None)
        props = {}
        if anchor_writer == "signed_same":
            digest = await storage.store_file(content, "KESTREL_CONSTITUTION.md")
            props["constitution_hash"] = digest
        await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="lock-order", properties=props))
        if anchor_writer == "signed_same":
            await agent._anchor_constitution_governance(digest)
        artifact, root = _write_authority_files(tmp_path, content)
        agent._sovereign_trust_root_path = root
        before = await agent._constitution_state_store.load(agent.agent_id)
        if transition == "periodic":
            # Exercise the mutable periodic branch, not its already-pending
            # startup-audit no-op. Durable bootstrap custody is unchanged.
            agent._constitution_audit_pending = False
        prepared = asyncio.Event()
        anchor_waiting_for_database = asyncio.Event()
        original_guard = ConstitutionMixin._constitution_state_guard
        original_transaction = storage.transaction

        @asynccontextmanager
        async def coordinated_guard(owner):
            # Delay only the native anchor's first lock entry, after its real
            # reads/source/signature checks. No lock or SQL behavior is faked.
            if asyncio.current_task() is anchor_task and not prepared.is_set():
                prepared.set()
                await allow_anchor.wait()
            async with original_guard(owner):
                yield

        @asynccontextmanager
        async def observed_transaction(*args, **kwargs):
            if asyncio.current_task() is anchor_task:
                assert agent._constitution_state_lock.locked()
                anchor_waiting_for_database.set()
            async with original_transaction(*args, **kwargs):
                yield

        monkeypatch.setattr(ConstitutionMixin, "_constitution_state_guard", coordinated_guard)
        monkeypatch.setattr(storage, "transaction", observed_transaction)

        async def anchor():
            if anchor_writer == "automatic":
                return await ConstitutionMixin._get_governing_constitution(agent)
            return await ConstitutionMixin.reanchor_constitution(agent, amendment_artifact_path=str(artifact))

        async def run_transition():
            if transition == "entry":
                return await agent.enter_safe_mode("caller restriction")
            if transition == "exit":
                return await agent.exit_safe_mode(authorization="test sovereign")
            if transition == "audit_begin":
                return await agent._begin_explicit_constitution_audit()
            if transition == "audit_record":
                return await agent._record_successful_constitution_audit(source="caller")
            if transition == "audit_run":
                return await agent._run_explicit_constitution_audit()
            return await agent._maybe_audit()

        anchor_task = asyncio.create_task(anchor())
        await _wait_for_step(f"{anchor_writer} anchor reaches its first constitution-lock entry", prepared, anchor_task)
        async with storage.transaction():
            assert storage.owns_open_transaction is True
            allow_anchor.set()
            await _wait_for_step(
                f"{anchor_writer} anchor holds the constitution lock and opens its transaction",
                anchor_waiting_for_database,
                anchor_task,
            )
            result = await _refuse_without_lock_wait(
                f"{transition} transition inside the caller transaction",
                agent._constitution_state_lock,
                run_transition,
            )
            assert agent._safe_mode is True
            assert agent._constitution_state_persistence_pending is True
            assert (agent._constitution_state_revision, agent._constitution_state_generation) == (before.revision, before.generation)
            if transition in {"entry", "audit_begin", "audit_record"}:
                assert result is False
            elif transition == "exit":
                assert result.startswith("Safe Mode remains active:")
            elif transition == "audit_run":
                assert result[0] is None and result[2] is False
            else:
                assert result is None
        outcome = await _await_task(f"{anchor_writer} anchor completes after the caller transaction", anchor_task)
        assert not outcome.startswith("Error:"), outcome
        durable = await agent._constitution_state_store.load(agent.agent_id)
        # No refused audit/entry/exit was silently included in the anchor's
        # later owned commit. Its original restricted state remains exact.
        assert durable.safe_mode == before.safe_mode
        assert durable.last_successful_audit_at == before.last_successful_audit_at
        assert durable.interaction_count == before.interaction_count
    finally:
        allow_anchor.set()
        if anchor_task is not None and not anchor_task.done():
            anchor_task.cancel()
        if anchor_task is not None:
            with suppress(asyncio.CancelledError):
                await anchor_task
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_at", ["verification", "database"])
async def test_refused_lifecycle_entry_invalidates_inflight_native_exit(
    tmp_path, monkeypatch, blocked_at
):
    """A lock-free failure latch must not be erased by the current lock owner."""
    from kestrel_sovereign.agent.constitution import SafeModeCause

    storage = AsyncStorage(str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id="did:test:refused-exit:" + uuid4().hex)
    await storage.initialize()
    exit_task = None
    proceed = asyncio.Event()
    try:
        agent = _DurableConstitutionHarness(storage, datetime.now(timezone.utc))
        agent.agent_id = storage.agent_id
        await agent._initialize_constitution_runtime_state(is_new_identity=True)
        agent.extension = None
        agent._anchor_constitution_governance = KestrelAgent._anchor_constitution_governance.__get__(agent)
        content = resolve_governing_constitution_bytes(None)
        digest = await storage.store_file(content, "KESTREL_CONSTITUTION.md")
        await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="exit", properties={"constitution_hash": digest}))
        await agent._anchor_constitution_governance(digest)
        await agent.enter_safe_mode("old integrity restriction")
        before = await agent._constitution_state_store.load(agent.agent_id)
        agent.verify_constitution_overlay = ConstitutionMixin.verify_constitution_overlay.__get__(agent)
        agent._verify_spawn_mandate_constraints = ConstitutionMixin._verify_spawn_mandate_constraints.__get__(agent)
        native_verify = ConstitutionMixin._verify_constitution_integrity.__get__(agent)
        verified = asyncio.Event()
        blocked = asyncio.Event()
        database_waiting = asyncio.Event()
        original_transaction = storage._backend.transaction

        async def observed_verify():
            result = await native_verify()
            assert result[0] is True, result
            verified.set()
            if blocked_at == "verification":
                blocked.set()
                await proceed.wait()
            return result

        @asynccontextmanager
        async def observed_transaction(*args, **kwargs):
            if asyncio.current_task() is exit_task and verified.is_set() and not blocked.is_set():
                blocked.set()
                await proceed.wait()
                database_waiting.set()
            async with original_transaction(*args, **kwargs):
                yield

        agent._verify_constitution_integrity = observed_verify
        if blocked_at == "database":
            monkeypatch.setattr(storage._backend, "transaction", observed_transaction)
        exit_task = asyncio.create_task(agent.exit_safe_mode(authorization="test sovereign"))
        await _wait_for_step(f"exit holds the constitution lock, blocked at {blocked_at}", blocked, exit_task)
        async with storage.transaction():
            if blocked_at == "database":
                proceed.set()
                await _wait_for_step("exit opens its transaction behind the caller's", database_waiting, exit_task)
            assert await _refuse_without_lock_wait(
                "lifecycle-uncertain entry inside the caller transaction",
                agent._constitution_state_lock,
                lambda: agent.enter_safe_mode(
                    "feature quarantine could not persist",
                    cause=SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value,
                ),
            ) is False
            assert agent._safe_mode_cause == SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value
            assert agent._feature_lifecycle_integrity_uncertain is True
            assert agent._feature_lifecycle_repair_verified is False
        proceed.set()
        result = await _await_task("exit completes after the caller transaction", exit_task)
        assert result.startswith("Safe Mode remains active:"), result
        assert agent._safe_mode is True
        assert agent._constitution_state_persistence_pending is True
        assert await agent._constitution_state_store.load(agent.agent_id) == before
    finally:
        proceed.set()
        if exit_task is not None and not exit_task.done():
            exit_task.cancel()
        if exit_task is not None:
            with suppress(asyncio.CancelledError):
                await exit_task
        await storage.close()
