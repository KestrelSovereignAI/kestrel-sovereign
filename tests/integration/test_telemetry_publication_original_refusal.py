"""Actual telemetry coroutine publication honors native Task admission refusal.

The proxy is constructed, not initialized: no child/Agent READY is claimed.
The original lease denial comes from the fixture's actual retired PostgreSQL
advisory owner. Native Task creation is transparently forwarded or refused,
never replaced by a fabricated future, native ACK or successful result.
"""

import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.features.isolated_runtime import ProxyFeature
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_embedding_job_original_control import _job_agent
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_isolated_feature_runtime import (
    _configure_idle_lifecycle,
    _idle_test_runtime,
)
from tests.shared.original_task_admission import assert_unknown_task_admission


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "publisher,kind",
    [
        (publisher, kind)
        for publisher in ("agent", "feature")
        for kind in ("native-control", "ordinary", "healthy")
    ]
    + [("invalid", "healthy")],
)
async def test_actual_telemetry_publication_does_not_retry_retired_coroutine(
    native_memory_owners, tmp_path, monkeypatch, publisher, kind
):
    n = native_memory_owners
    agent = _job_agent(n)
    agent.did, agent.features = n.did, {}
    agent.storage_path = str(tmp_path / "component-agent.db")
    observations, attempts, accepted, submitted, sources = [], [], [], [], []

    async def observe(snapshot):
        observations.append(snapshot)

    _configure_idle_lifecycle(
        agent, tmp_path, idle_timeout_seconds=3600, telemetry_observer=observe
    )
    feature = ProxyFeature(agent, _idle_test_runtime())
    actual_create = asyncio.create_task
    if publisher == "agent":
        actual_publish = agent._track_background_task

        def publish(operation, *, name):
            submitted.append(operation)
            return actual_publish(operation, name=name)

        agent._track_background_task = publish
    elif publisher == "feature":
        del agent._track_background_task
    else:

        def invalid_publication(operation, *, name):
            # Explicit invalid interface input: no native Task was accepted.
            # This is NOT a fake publication acknowledgment.
            submitted.append(operation)
            return None

        agent._track_background_task = invalid_publication

    def refuse(operation, *args, **kwargs):
        name = kwargs.get("name") or ""
        if name.startswith("isolated-runtime-telemetry:"):
            attempts.append(operation)
            if kind != "healthy" and len(attempts) == 1:
                if kind == "native-control":
                    try:
                        n.lease.require_live()
                    except BaseException as original:
                        sources.append(original)
                        raise
                error = OSError("ordinary telemetry publication unavailable")
                sources.append(error)
                raise error
        task = actual_create(operation, *args, **kwargs)
        if name.startswith("isolated-runtime-telemetry:"):
            accepted.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", refuse)
    failure = None
    try:
        try:
            feature._schedule_runtime_telemetry(
                force=True, refresh_disk=True, refresh_environment=True
            )
        except BaseException as error:
            failure = error
        for task in accepted:
            await await_owned_task(task)
        if publisher == "invalid":
            assert isinstance(failure, TypeError), (
                "telemetry accepted an invalid owner instead of failing its interface contract"
            )
            assert not attempts and not accepted and not observations
            assert submitted and all(
                inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
                for operation in submitted
            )
        else:
            assert len(attempts) == 1, (
                "telemetry retried an already-refused/closed original coroutine"
            )
        if publisher == "invalid":
            pass  # No native admission occurred; its invalid contract is above.
        elif kind == "healthy":
            assert failure is None and accepted and observations
        else:
            assert sources
            # Even an ordinary OSError from a substituted constructor leaves
            # native acceptance unknown. Advisory telemetry cannot erase it.
            assert_unknown_task_admission(failure, sources[0])
            assert not accepted and not observations
            assert all(
                inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
                for operation in submitted + attempts
            ), "refused telemetry publication left a fresh coroutine unowned"
            assert feature._telemetry_emit_pending
            assert feature._telemetry_emit_force_pending
            assert feature._telemetry_disk_refresh_pending
            assert feature._telemetry_environment_refresh_pending
        assert not n.requests
    finally:
        for task in accepted:
            await await_owned_task(task)
        # For the pre-fix feature-owned refusal, the test still owns the
        # unstarted operation. Closing it is cleanup, never a native ACK.
        for operation in submitted + attempts:
            operation.close()
        for task in tuple(
            feature._telemetry_emit_tasks | feature._telemetry_observer_tasks
        ):
            if isinstance(task, asyncio.Task):
                await await_owned_task(task)
