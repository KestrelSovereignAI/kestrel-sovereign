"""Cold idle cleanup must retain durable historical immutable launch evidence."""
import os
import asyncio
from pathlib import Path

import pytest

from kestrel_sovereign.features import isolated_runtime as runtime
from tests.unit.test_isolated_feature_runtime import _idle_test_runtime


@pytest.mark.skipif(os.name != "posix", reason="native no-follow dirfd custody")
@pytest.mark.parametrize("marker_kind", ["original", "malformed", "symlink"])
def test_cold_feature_reclamation_cannot_erase_namespace_launch_custody(tmp_path, marker_kind):
    root = tmp_path / "runtime"
    scope = runtime.IsolatedRuntimeNamespace(root, Path("tenant-a"), root / "tenant-a")
    owner = "did:test:original-launch"
    component = runtime._hosted_feature_runtime_component(_idle_test_runtime())
    workspace = runtime.prepare_isolated_runtime_namespace(scope, owner,
        relative_directories=(("feature_venvs", component),))
    launch = workspace / "feature_venvs" / component / "operator-bin"
    launch.write_bytes(b"original immutable launch artifact")
    runtime.retain_host_launch_artifact_custody(scope, owner, None, (launch,))
    marker = workspace / runtime._IMMUTABLE_LAUNCH_CUSTODY_MARKER
    if marker_kind == "malformed":
        marker.write_bytes(b"restrictive evidence cannot authorize cleanup")
    elif marker_kind == "symlink":
        marker.unlink()
        marker.symlink_to(tmp_path / "absent-marker-target")
    # Cold cleanup possesses no old proxy, environment selection, or path list.
    # Retargeting launch configuration cannot erase durable restrictive evidence.
    with pytest.raises(runtime.IsolatedRuntimePreparationError, match="immutable"):
        runtime._remove_isolated_feature_runtime(scope, owner, component)
    assert launch.read_bytes() == b"original immutable launch artifact"
    assert os.path.lexists(marker)


@pytest.mark.asyncio
async def test_normal_stop_registry_close_joins_native_poll_commit(tmp_path, monkeypatch):
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
    from kestrel_sovereign.execution_custody import execution_commit_outcome
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.stop import DistributedInvocationRegistry, DistributedInvocationStore
    from tests.unit.test_distributed_stop_invocations import _ReplicaAgent

    db = await AsyncDatabase.sqlite(str(tmp_path / "native-stop-poll.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(store)
    agent = _ReplicaAgent("did:test:original-stop-poll")
    assert await registry.register(agent, "original-turn", 1)
    entered, release = asyncio.Event(), asyncio.Event()
    native_commit = db.backend._connection.commit
    commits = []

    async def held_commit():
        await native_commit()
        task_name = asyncio.current_task().get_name()
        if task_name == "distributed-stop-relay" or task_name.startswith("distributed-stop-owner-poll:"):
            commits.append(True)
            entered.set()
            await release.wait()

    monkeypatch.setattr(db.backend._connection, "commit", held_commit)
    registry.start()
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        stopping = create_owned_task(registry.close())
        done, _ = await asyncio.wait({stopping}, timeout=.05)
        assert not done, "normal registry close cancelled accepted poll COMMIT acknowledgement"
        release.set()
        outcome = await await_owned_task(stopping)
        assert outcome.error is None and execution_commit_outcome(outcome.error) is None
        assert registry._terminal_control is None and registry._relay_task is None
        assert len(commits) == 1
    finally:
        release.set()
        if stopping is not None:
            await await_owned_task(stopping)
        await await_owned_task(create_owned_task(registry.close()))
        await db.close()
