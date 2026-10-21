"""Standalone restoration's accepted native owner is not a success on control.

This exercises actual SpawnFeature shutdown, actual PG work, original owned
Task cancellation and an actual retired advisory-lease denial. The accepted
restore handle/Agent view are declared component inputs, not Manager tree
recovery, full spawned Agent READY or retry authority.
"""

import asyncio
import secrets
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.features.spawn.feature import SpawnFeature
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize(
    "kind", ["native-control", "native-cancel", "ordinary", "healthy"]
)
async def test_spawn_shutdown_retains_original_restore_control(
    native_pg, kind, deferred
):
    backend, observer, schema = native_pg
    await backend.execute("CREATE TABLE restore_effects (id INTEGER PRIMARY KEY)")
    async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
        lease.require_live()
    feature = SpawnFeature(
        SimpleNamespace(did="did:test:restore", agent_id="did:test:restore")
    )
    await feature.initialize()
    entered, retired = asyncio.Event(), asyncio.Event()
    originals = []
    ordinary = OSError("ordinary original restoration failure")
    delegated_started, delegated_release = asyncio.Event(), asyncio.Event()

    async def delegated():
        await backend.execute("INSERT INTO restore_effects VALUES (2)")
        delegated_started.set()
        try:
            await delegated_release.wait()
        except asyncio.CancelledError:
            # Actual accepted work resists its first cooperative withdrawal;
            # the existing retained shutdown owner must join it before exit.
            await delegated_release.wait()

    async def restore():
        await backend.execute("INSERT INTO restore_effects VALUES (1)")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            try:
                if kind.startswith("native"):
                    try:
                        lease.require_live()
                    except BaseException as original:
                        originals.append(original)
                        if kind == "native-cancel":
                            raise asyncio.CancelledError(
                                "original native restoration observation"
                            ) from original
                        raise
                if kind == "ordinary":
                    raise ordinary
            finally:
                retired.set()

    task = create_owned_task(
        restore(), name="accepted-standalone-restore", retain_originals=True
    )
    feature._standalone_restore_task = task
    delegated_task = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if deferred:
            delegated_task = create_owned_task(
                delegated(),
                name="accepted-standalone-delegation",
                retain_originals=True,
            )
            feature._delegated_task_owners.add(delegated_task)
            await asyncio.wait_for(delegated_started.wait(), 5)
        caught = None
        try:
            await feature.shutdown()
        except BaseException as error:
            caught = error
        if deferred:
            assert caught is None
            retained_owner = feature._standalone_shutdown_task
            assert retained_owner is not None and not delegated_task.done()
            delegated_release.set()
            await asyncio.wait_for(await_owned_task(retained_owner), 5)
            try:
                await feature.shutdown()
            except BaseException as error:
                caught = error
            assert delegated_task.done()
        assert task.done() and retired.is_set()
        assert feature._standalone_restore_task is None
        assert await observer.fetchval(
            f'SELECT count(*) FROM "{schema}".restore_effects'
        ) == (2 if deferred else 1)
        if kind.startswith("native"):
            assert (
                originals
                and caught is not None
                and originals[0] in tuple(_explicit_control_chain(caught))
            ), "spawn shutdown normalized original native restore control"
            assert is_execution_control_error(caught)
            original_owner = feature._standalone_shutdown_task
            for _ in range(3):
                with pytest.raises(BaseException) as later:
                    await feature.shutdown()
                assert originals[0] in tuple(_explicit_control_chain(later.value))
                assert feature._standalone_shutdown_task is original_owner
            with pytest.raises(BaseException) as reenable:
                await feature.initialize()
            assert originals[0] in tuple(_explicit_control_chain(reenable.value))
        elif kind == "ordinary":
            assert caught is ordinary
        else:
            assert caught is None
    finally:
        delegated_release.set()
        if not task.done():
            task.cancel()
        await await_owned_task(task)
        if delegated_task is not None:
            await await_owned_task(delegated_task)


@pytest.mark.asyncio
@pytest.mark.parametrize("out_of_band", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
@pytest.mark.parametrize(
    "kind", ["native-control", "native-cancel", "ordinary", "healthy"]
)
async def test_ready_restore_preserves_source_and_actual_observer(
    native_pg, tmp_path, monkeypatch, out_of_band, withdraw, kind
):
    """Actual Manager object/Task seam; injected restore body is not tree READY."""
    backend, observer, schema = native_pg
    await backend.execute("CREATE TABLE ready_restore_effects (id INTEGER PRIMARY KEY)")
    async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
        lease.require_live()
    manager = AgentManager(base_data_dir=tmp_path, startup_roster_enabled=False)
    feature = SpawnFeature(
        SimpleNamespace(did="did:test:ready", agent_id="did:test:ready")
    )
    await feature.initialize()
    feature._agent_manager = manager
    feature._standalone_manager_owned = True
    feature._standalone_restore_pending = True
    entered, release = asyncio.Event(), asyncio.Event()
    sources, calls, withdrawals = [], [], []
    ordinary = OSError("ordinary ready restoration failure")

    async def restore(agent):
        calls.append(asyncio.current_task())
        await backend.execute("INSERT INTO ready_restore_effects VALUES (1)")
        entered.set()
        await release.wait()
        error = ordinary if kind == "ordinary" else None
        if kind.startswith("native"):
            try:
                lease.require_live()
            except BaseException as original:
                error = original
                if kind == "native-cancel":
                    error = asyncio.CancelledError(
                        "actual ready restore source cancellation"
                    )
                    error.__cause__ = original
        if error is not None:
            sources.append(error)
            if out_of_band:
                manager._init_failures = [("actual-component-child", error)]
            else:
                raise error

    # Transparent observation of SAME real await outcome, never an ACK stub.
    from kestrel_sovereign.features.spawn import feature as feature_module
    from kestrel_sovereign import _async_ownership

    actual_wait = _async_ownership.await_owned_task

    async def wait(task, *args, **kwargs):
        result = await actual_wait(task, *args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result

    monkeypatch.setattr(manager, "restore_spawn_authority_tree", restore)
    monkeypatch.setattr(_async_ownership, "await_owned_task", wait)
    monkeypatch.setattr(feature_module, "await_owned_task", wait)
    owner = create_owned_task(feature._get_ready_agent_manager(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if withdraw:
            owner.cancel("original ready observer withdrawn")
        release.set()
        outcome = await actual_wait(owner)
        assert all(task.done() for task in calls)
        assert (
            await observer.fetchval(
                f'SELECT count(*) FROM "{schema}".ready_restore_effects'
            )
            == 1
        )
        if kind.startswith("native"):
            assert outcome.error is not None and sources[0] in tuple(
                _explicit_control_chain(outcome.error)
            ), "ready restore normalized the original native outcome"
            for _ in range(3):
                with pytest.raises(BaseException) as later:
                    await feature._get_ready_agent_manager()
                assert sources[0] in tuple(_explicit_control_chain(later.value))
            with pytest.raises(BaseException) as reenable:
                await feature.initialize()
            assert sources[0] in tuple(_explicit_control_chain(reenable.value))
            assert len(calls) == 1, "uncertain restore acquired fresh retry authority"
        elif kind == "ordinary" and not out_of_band:
            assert outcome.error is not None and ordinary in tuple(
                _explicit_control_chain(outcome.error)
            )
        elif not withdraw:
            assert outcome.error is None and outcome.result is manager
        if withdraw:
            assert withdrawals and withdrawals[0] in tuple(
                _explicit_control_chain(outcome.error)
            ), "ready restore replaced actual observer withdrawal"
            control = vars(feature).get("_standalone_restore_control")
            assert withdrawals[0] not in tuple(_explicit_control_chain(control))
    finally:
        release.set()
        await actual_wait(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["native-control", "native-cancel"])
@pytest.mark.parametrize("door", ["ready", "shutdown-owner", "descendant"])
async def test_spawn_native_task_admission_retains_original_custody(
    native_pg, tmp_path, monkeypatch, kind, door
):
    """Real allocator refusal/retired PG lease; no fabricated accepted Task."""
    backend, observer, schema = native_pg
    await backend.execute("CREATE TABLE admitted_spawn_effect (id INTEGER PRIMARY KEY)")
    async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
        lease.require_live()
    manager = AgentManager(base_data_dir=tmp_path, startup_roster_enabled=False)
    feature = SpawnFeature(
        SimpleNamespace(did="did:test:admission", agent_id="did:test:admission")
    )
    await feature.initialize()
    feature._agent_manager = manager
    feature._standalone_manager_owned = True
    feature._standalone_restore_pending = True

    async def native_body(agent):
        await backend.execute("INSERT INTO admitted_spawn_effect VALUES (1)")

    monkeypatch.setattr(manager, "restore_spawn_authority_tree", native_body)
    monkeypatch.setattr(manager, "shutdown_spawn_authority_tree", native_body)
    refused_name = {
        "ready": "standalone_spawn_authority_restore",
        "shutdown-owner": "standalone_spawn_shutdown_owner",
        "descendant": "standalone_spawn_descendant_shutdown",
    }[door]
    actual_create = asyncio.create_task
    errors = []

    def create(operation, *args, **kwargs):
        if kwargs.get("name") == refused_name:
            try:
                lease.require_live()
            except BaseException as original:
                error = original
                if kind == "native-cancel":
                    error = asyncio.CancelledError(
                        "actual native spawn admission withdrawal"
                    )
                    error.__cause__ = original
                errors.append(error)
                raise error
        return actual_create(operation, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", create)
    operation = (
        feature._get_ready_agent_manager if door == "ready" else feature.shutdown
    )
    with pytest.raises(BaseException) as first:
        await operation()
    assert errors[0] in tuple(_explicit_control_chain(first.value))
    with pytest.raises(BaseException) as reenable:
        await feature.initialize()
    assert errors[0] in tuple(_explicit_control_chain(reenable.value))
    assert (
        await observer.fetchval(
            f'SELECT count(*) FROM "{schema}".admitted_spawn_effect'
        )
        == 0
    )
    assert feature._agent_manager is manager
    if door == "ready":
        with pytest.raises(BaseException) as again:
            await feature._get_ready_agent_manager()
        assert errors[0] in tuple(_explicit_control_chain(again.value))
        assert len(errors) == 1
