"""Review36 native receipts survive privacy, fleet, and isolated lifecycle owners."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.agent.privacy_state import PrivacyState, restore_privacy_state
from kestrel_sovereign.features import isolated_runtime as runtime
from kestrel_sovereign.privacy import PrivacyMode
from tests.unit.test_execution_custody_review29 import native_owner
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_execution_custody_review35_native import traceback_size

KINDS = ["unknown", "cancel_unknown", "group"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", ["record", "mandate", "read", "restore", "restore_recovery"])
async def test_privacy_native_control_is_not_refusal_or_best_effort(monkeypatch, tmp_path, kind, route):
    agent, db = await native_owner(tmp_path)
    await db.execute("CREATE TABLE IF NOT EXISTS agent_metadata(agent_id TEXT, key TEXT, value TEXT, updated_at TEXT, PRIMARY KEY(agent_id,key))")
    failure = receipt(kind)
    original_execute, original_read = db.execute, db.fetchall
    effects = []

    async def write_then_receipt(*args, **kwargs):
        await original_execute(*args, **kwargs)
        effects.append(True)
        raise failure

    async def read_then_receipt(*args, **kwargs):
        if route == "restore_recovery" and not effects:
            effects.append("ordinary-first-read")
            raise OSError("first read unavailable")
        await original_read(*args, **kwargs)
        await original_execute("INSERT INTO effects VALUES (1)")
        effects.append(True)
        raise failure

    try:
        monkeypatch.setattr(db, "execute" if route in {"record", "mandate"} else "fetchall",
                            write_then_receipt if route in {"record", "mandate"} else read_then_receipt)
        async def invoke():
            if route == "record":
                return await agent._record_privacy_state(PrivacyState(PrivacyMode.NORMAL))
            if route == "mandate":
                return await agent._write_privacy_mandate({"model": "offline", "vendor": "local", "route": "local"})
            if route == "read":
                return await agent._recorded_privacy_state()
            return await restore_privacy_state(db, agent.agent_id, initial_mode=PrivacyMode.NORMAL)
        outcome = await await_owned_task(asyncio.create_task(invoke()))
        assert outcome.error is failure, "native control became ordinary privacy refusal/recovery"
        assert effects.count(True) == 1, "native uncertainty authorized another storage attempt"
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_constructed_manager_agent_keeps_initialization_control_after_cleanup(monkeypatch, tmp_path, kind):
    from kestrel_sovereign.multi_agent import agent_manager as module
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig
    agent, db = await native_owner(tmp_path)
    manager = module.AgentManager(base_data_dir=tmp_path)
    failure = receipt(kind)
    source = OriginalFailure.capture(failure)
    closed = []
    async def initialize():
        await db.execute("INSERT INTO effects VALUES (1)")
        raise failure
    async def shutdown():
        await agent.storage.close()
        closed.append(True)
        raise OSError("ordinary rollback receipt after native close")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    monkeypatch.setattr(LocalAgentConfig, "validate_runtime", lambda *_a, **_kw: [])
    monkeypatch.setattr(module, "read_anchor_agent_did", AsyncMock(return_value=agent.did))
    monkeypatch.setattr(module, "LLMService", lambda **_kwargs: agent.llm_service)
    monkeypatch.setattr(module, "KestrelAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(agent, "initialize", initialize)
    monkeypatch.setattr(agent, "shutdown", shutdown)
    try:
        outcome = await await_owned_task(asyncio.create_task(manager._initialize_agent(
            "child", LocalAgentConfig(data_dir=str(tmp_path), port=18888))))
        assert closed == [True] and db.backend._connection is None
        source.assert_retained(outcome.error, resource=agent,
                               task=agent._unpublished_cleanup_owner.admission.task)
        assert agent._shutdown_control is failure
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_batch_fatal_cleanup_includes_all_native_initializer_receipts(monkeypatch, tmp_path, kind):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
    agent, db = await native_owner(tmp_path)
    manager = AgentManager(base_data_dir=tmp_path)
    failure = receipt(kind)
    source = OriginalFailure.capture(failure)
    async def initialize(name, *_args, **_kwargs):
        if name == "Alpha":
            raise asyncio.CancelledError("ordinary first initializer cancellation")
        if name == "Beta":
            await db.execute("INSERT INTO effects VALUES (1)")
            raise failure
        return agent
    async def shutdown():
        await agent.storage.close()
        raise OSError("ordinary unpublished sibling cleanup failure")
    monkeypatch.setattr(manager, "_initialize_agent", initialize)
    monkeypatch.setattr(agent, "shutdown", shutdown)
    config = MultiAgentConfig(agents={name: LocalAgentConfig(data_dir=name.lower(), port=8801+i)
                                    for i, name in enumerate(("Alpha", "Beta", "Gamma"))})
    try:
        outcome = await await_owned_task(asyncio.create_task(manager.load_from_config(config)))
        assert db.backend._connection is None and not manager.list_agents()
        source.assert_retained(outcome.error, resource=agent,
                               task=agent._unpublished_cleanup_owner.admission.task)
        assert manager._agent_operations["gamma"].child is agent
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_write", [False, True])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("boundary", ["wait", "facade", "facade_cancel", "facade_timeout"])
async def test_isolated_lifecycle_join_preserves_first_native_receipt(monkeypatch, tmp_path, kind, boundary, delayed_write):
    agent, db = await native_owner(tmp_path)
    if delayed_write:
        native_execute = db.execute
        async def delayed_execute(*args, **kwargs):
            result = await native_execute(*args, **kwargs)
            if args and args[0].startswith("INSERT INTO effects"):
                # Transparent post-write latency, not CPU stress or a native
                # ACK stub. The real SQLite effect must precede the 20ms test.
                await asyncio.sleep(0.05)
            return result
        monkeypatch.setattr(db, "execute", delayed_execute)
    failure = receipt(kind)
    original = OriginalFailure.capture(failure)
    withdrawals = []
    native_wait = asyncio.wait_for
    async def observed_wait(*args, **kwargs):
        try:
            return await native_wait(*args, **kwargs)
        except asyncio.CancelledError as error:
            withdrawals.append(error)
            raise
    if boundary == "facade_cancel":
        monkeypatch.setattr(asyncio, "wait_for", observed_wait)
    entered, release, admitted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    child = outer = None
    async def native():
        await db.execute("INSERT INTO effects VALUES (1)")
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            pass
        raise failure
    async def join():
        if boundary == "wait":
            return await runtime._await_task_until_complete(child, preserve_cancellation=False)
        return await runtime._await_owned_facade_lifecycle_operation(
            child, name="review36-native", on_late_task=lambda _task: pytest.fail("native child orphaned"),
            on_admitted=lambda _operation: admitted.set())
    try:
        if boundary == "facade_timeout":
            monkeypatch.setattr(runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.02)
        child = asyncio.create_task(native())
        await asyncio.wait_for(entered.wait(), 5)
        # The 20ms facade budget measures an admitted native receipt, NOT the
        # unrelated time SQLite needed to open/write on a busy CI worker.
        outer = asyncio.create_task(join())
        if boundary == "facade_cancel":
            await asyncio.wait_for(admitted.wait(), 5)
            outer.cancel("ordinary owner cancellation")
        if boundary != "facade_timeout":
            release.set()
        outcome = await await_owned_task(outer)
        assert child.done() and await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        original.assert_retained(outcome.error)
        if boundary == "facade_cancel":
            from kestrel_sovereign.execution_custody import _explicit_control_chain
            assert withdrawals and withdrawals[0] in tuple(_explicit_control_chain(outcome.error)), "lifecycle lost actual finite observer withdrawal"
        for _ in range(100):
            try:
                await runtime._await_task_until_complete(child, preserve_cancellation=False)
            except BaseException as error:
                assert error is failure
        assert traceback_size(failure) <= 8, "completed lifecycle reads grow retained receipt"
    finally:
        release.set()
        for task in (child, outer):
            if task is not None:
                await await_owned_task(task)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_repeated_lifecycle_deadlines_keep_one_terminal_observer(tmp_path, kind):
    agent, db = await native_owner(tmp_path)
    failure = receipt(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    async def native():
        await db.execute("INSERT INTO effects VALUES (1)")
        entered.set()
        await release.wait()
        raise failure
    child = asyncio.create_task(native())
    try:
        await entered.wait()
        for _ in range(100):
            waiter = asyncio.create_task(runtime._await_task_until_complete(
                child, preserve_cancellation=False, settle_on_cancellation=False))
            await asyncio.sleep(0)
            waiter.cancel("bounded caller handoff")
            outcome = await await_owned_task(waiter)
            assert isinstance(outcome.error, asyncio.CancelledError)
            assert not child.done()
            assert len(child._callbacks or ()) <= 1, "deadline retained abandoned waiter closures"
        release.set()
        outcome = await await_owned_task(child)
        assert outcome.error is failure
        for _ in range(100):
            try:
                await runtime._await_task_until_complete(child, preserve_cancellation=False, settle_on_cancellation=False)
            except BaseException as error:
                assert error is failure
        assert traceback_size(failure) <= 8
    finally:
        release.set()
        await await_owned_task(child)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_proxy_retained_control_reads_are_bounded(monkeypatch, tmp_path, kind):
    from tests.unit.test_execution_custody_proxy_ingress import owner
    failure = receipt(kind)
    agent, db, feature, _client, _event, _entered, release = await owner(monkeypatch, tmp_path, failure)
    await db.execute("INSERT INTO effects VALUES (1)")
    feature._retain_execution_control(failure)
    try:
        for _ in range(100):
            try:
                feature._require_event_ingress_work()
            except BaseException as error:
                assert error is failure
        assert traceback_size(failure) <= 6, "proxy ingress reads accumulate retained reader frames"
        for _ in range(100):
            try:
                await feature.shutdown()
            except BaseException as error:
                assert error is failure
        assert feature._client is None and traceback_size(failure) <= 10
    finally:
        release.set()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("boundary", ["stage", "promote", "abort", "fenced_recovery"])
async def test_native_config_cas_uncertainty_survives_recovery(monkeypatch, tmp_path, kind, boundary):
    from tests.unit.test_execution_custody_proxy_ingress import owner
    from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_custody_snapshot
    class LocalFence:
        backend_type = "sqlite"
        def require_work(self):
            pass
    failure = receipt(kind)
    source = OriginalFailure.capture(failure)
    agent, db, feature, client, _event, _entered, release = await owner(monkeypatch, tmp_path, failure)
    scope = ExecutionCustody(LocalFence())
    native_cas = agent.storage.compare_and_swap_node
    commits = []
    async def cas_then_receipt(node_id, expected, node):
        result = await native_cas(node_id, expected, node)
        if (boundary == "stage" and "pending_config" in node.properties) or (
            boundary in {"abort", "promote", "fenced_recovery"} and expected and "pending_config" in expected and "pending_config" not in node.properties
        ):
            commits.append(result)
            scope.preserve_commit_uncertainty("unknown")
            raise failure
        return result
    monkeypatch.setattr(agent.storage, "compare_and_swap_node", cas_then_receipt)
    async def apply():
        with bind_execution_custody_snapshot((scope,)):
            if boundary in {"stage", "promote"}:
                return await feature.set_config({"enabled": True})
            transition = await feature._stage_pending_config({"enabled": True})
            if boundary == "fenced_recovery":
                return await feature._recover_fenced_transition(
                    transition, asyncio.CancelledError("original ordinary cancellation"),
                    preserve_cancellation=True,
                )
            await feature._run_owned_transition_cleanup(transition, force=False, preserve_cancellation=True)
    try:
        outcome = await await_owned_task(asyncio.create_task(apply()))
        assert commits == ["swapped"], "uncertain native CAS was retried"
        source.assert_retained(outcome.error)
        assert client.stopped and feature._client is None and feature._terminal_lifecycle_latched
    finally:
        release.set()
        try:
            await feature.shutdown()
        except BaseException:
            pass
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_legacy_authority_reconciliation_exports_original_native_control(monkeypatch, tmp_path, kind):
    from tests.unit.test_execution_custody_proxy_ingress import owner
    failure = receipt(kind)
    agent, db, feature, client, _event, _entered, release = await owner(monkeypatch, tmp_path, failure)
    native_read = agent.storage.get_node
    effects = []
    async def read_then_receipt(*args, **kwargs):
        await native_read(*args, **kwargs)
        await db.execute("INSERT INTO effects VALUES (1)")
        effects.append(True)
        raise failure
    monkeypatch.setattr(agent.storage, "get_node", read_then_receipt)
    try:
        outcome = await await_owned_task(asyncio.create_task(feature._handle_config_authority_change(None)))
        assert effects == [True]
        assert outcome.error is failure, "legacy authority handler masked original native control"
        assert client.stopped and feature._client is None and feature._terminal_lifecycle_latched
    finally:
        release.set()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["lexical", "scope"])
async def test_host_anchor_capability_never_widens_or_reacquires_authority(tmp_path, retirement):
    from kestrel_sovereign.execution_custody import (
        _host_identity_anchor_access, ExecutionCustody, ProcessRuntimeExecutionFence,
        bind_execution_custody_snapshot, ExecutionAuthorityError,
    )
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original-parent"))
    db = None
    try:
        with bind_execution_custody_snapshot((scope,)):
            with _host_identity_anchor_access(tmp_path / "kestrel_prime.db"):
                db = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
                assert db.backend._identity_anchor_access.scopes == (scope,)
                with pytest.raises(ExecutionAuthorityError):
                    await AsyncDatabase.sqlite(str(tmp_path / "ordinary.db"))
                assert not (tmp_path / "ordinary.db").exists()
                if retirement == "scope":
                    scope.revoke("original parent retired during anchor work")
                    with pytest.raises(ExecutionAuthorityError):
                        await db.execute("INSERT INTO graph_nodes VALUES ('escape', 'agent', 'escape', '{}')")
                    return
        # Losing ambient membership is NOT losing the retained file grant's
        # original scope or its lexical expiry, even on an unbound host task.
        with pytest.raises(BaseException) as caught:
            await db.fetchval("SELECT COUNT(*) FROM graph_nodes")
        from kestrel_sovereign.execution_custody import is_execution_control_error
        assert is_execution_control_error(caught.value)
        assert "retired" in str(caught.value)
    finally:
        if db is not None:
            await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo"])
async def test_host_anchor_file_refusal_precedes_capability_without_widening(tmp_path, kind):
    import os
    from kestrel_sovereign.execution_custody import (
        _host_identity_anchor_access, ExecutionCustody, ProcessRuntimeExecutionFence,
        bind_execution_custody_snapshot, ExecutionAuthorityError,
    )
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    target = tmp_path / "original.db"
    original = await AsyncDatabase.sqlite(str(target))
    await original.close()
    before = (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes())
    path = tmp_path / "kestrel_prime.db"
    if kind == "symlink":
        path.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, path)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original-parent"))
    with bind_execution_custody_snapshot((scope,)):
        with pytest.raises(ValueError, match="exclusively retained regular database"):
            with _host_identity_anchor_access(path):
                pytest.fail("unsafe anchor acquired host capability")
        scope.require_work()
        with pytest.raises(ExecutionAuthorityError):
            await AsyncDatabase.sqlite(str(tmp_path / "ordinary.db"))
    assert not (tmp_path / "ordinary.db").exists()
    assert before == (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes())
