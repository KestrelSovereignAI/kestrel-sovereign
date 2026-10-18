"""Many actual native ACK losses cannot become completed-source history.

Refund/reaper bodies are declared component operations with real SQLite effects;
this does not execute a financial refund, genesis or provider inference.
"""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_commit_outcome, execution_native_source_error
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("writer", ["budget", "reaper", "mixed"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_completed_native_sources_remain_bounded_after_metadata_ack_and_eviction(
    native_pg, onboarding_owner, monkeypatch, writer, source_cancel,
):
    from kestrel_sovereign._async_ownership import create_owned_task
    manager, agent, db = onboarding_owner
    backend, observer, schema = native_pg
    first = None
    count = 260 if writer == "mixed" else 130
    # Keep logging/harness capture out of the explicit graph measurement.
    from kestrel_sovereign.multi_agent import agent_manager as manager_module
    monkeypatch.setattr(manager_module.logger, "error", lambda *args, **kwargs: None)
    for index in range(count):
        # Every source is a DISTINCT actual native COMMIT/ACK loss. The shared
        # fixture helper uses one key; resetting only this isolated schema row
        # prevents later sources becoming unrelated primary-key failures.
        await observer.execute('DELETE FROM "' + schema + '".effects WHERE id=231')
        with monkeypatch.context() as local_fault:
            error = await _original_control(backend, local_fault, source_cancel)
        error._completed_source_probe = index
        snapshot = OriginalFailure.capture(error)
        first = first or snapshot
        async def completed_operation(failure=error, index=index):
            await db.execute("INSERT INTO host_rollback_effects VALUES (?)", (10000 + index,))
            raise failure
        actual_writer = ("budget" if index % 2 == 0 else "reaper") if writer == "mixed" else writer
        if actual_writer == "budget":
            task = manager._start_child_budget_release("bounded-" + str(index), release_operation=completed_operation).task
        else:
            task = create_owned_task(completed_operation(), retain_originals=True)
            manager._retain_quarantined_cleanup(name="bounded-" + str(index), agent_id=agent.did, task=task)
        try:
            finite = await await_owned_task(task)
            await asyncio.sleep(0)
            snapshot.assert_retained(finite.error)
            snapshot.assert_retained(error)
        finally:
            await await_owned_task(task)
    summary = manager._quarantined_shutdown_control
    first.assert_retained(summary)
    selected = [value for value in _explicit_control_chain(summary) if hasattr(value, "_completed_source_probe")]
    assert len(selected) == 1 and selected[0] is first.error
    assert execution_commit_outcome(execution_native_source_error(summary)) == "unknown"
    assert not manager._quarantined_shutdown_reapers
    assert not manager._inflight_removal_budget_releases
    assert len(manager._unsafe_removal_budget_release_failures) <= 128
    assert len(manager._unsafe_quarantined_shutdown_failures) <= 128
    assert len(manager._quarantined_shutdown_history) <= 128
    assert await db.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id >= 10000") == count
    for identity in tuple(manager._unsafe_removal_budget_release_failures):
        assert manager.acknowledge_unsafe_removal_budget_release_failure(identity)
    for identity in tuple(manager._unsafe_quarantined_shutdown_failures):
        assert manager.acknowledge_unsafe_quarantined_shutdown_failure(identity)
    manager.acknowledge_unsafe_removal_budget_release_failure_evictions()
    manager.acknowledge_unsafe_quarantined_shutdown_failure_evictions()
    assert not manager._unsafe_removal_budget_release_failures
    assert not manager._unsafe_quarantined_shutdown_failures
    first.assert_retained(manager._quarantined_shutdown_control)
    assert execution_commit_outcome(manager._quarantined_shutdown_control) == "unknown"
    assert len([value for value in _explicit_control_chain(manager._quarantined_shutdown_control) if hasattr(value, "_completed_source_probe")]) == 1
    # Explicit cause/group/provenance retention, not global GC or transient
    # traceback traversal through the manager's legitimately live state.
