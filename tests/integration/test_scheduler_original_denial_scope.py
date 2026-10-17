"""A second native scheduler admission is not this occurrence's authority."""

import asyncio
from uuid import uuid4

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    current_execution_custody,
    execution_denial_scopes,
    is_execution_control_error,
)
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_scheduler_postgres_claims import (
    _activate_protocol_for_test_agent,
)
from tests.unit.test_scheduler_occurrence_liveness import _seed_due
from tests.utils.scheduler_ticks import tick_and_settle


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", ["other-runner", "same-runner-other-admission"])
async def test_actual_foreign_scheduler_advisory_denial_is_not_occurrence_local(
    native_pg, monkeypatch, provenance
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    did = "did:web:original-" + uuid4().hex
    foreign = SchedulerRunner(db, did + "-foreign", lambda *args: None)
    callbacks, original_scopes, entered = [], [], asyncio.Event()

    async def executor(name, args):
        original_scopes.extend(current_execution_custody())
        assert all(scope is not foreign_scope for scope in original_scopes)
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 0
        entered.set()
        # A callback surfaces control from the other original admission.
        raise failure

    runner = SchedulerRunner(db, did, executor, on_protocol_failure=callbacks.append)
    scope_owner = foreign if provenance == "other-runner" else runner
    scope_did = did + "-foreign" if scope_owner is foreign else did
    # Acquire and retire the other runner's ACTUAL original PostgreSQL
    # advisory lease; no fabricated expired-fence carrier.
    async with scope_owner._postgres_rollout_effect_gate(scope_did):
        (foreign_scope,) = current_execution_custody()
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 0
    with pytest.raises(BaseException) as original:
        foreign_scope.require_work()
    failure = original.value
    assert execution_denial_scopes(failure) == (foreign_scope,)
    await _activate_protocol_for_test_agent(db, did, monkeypatch)
    await runner._ensure_tables()
    await _seed_due(db, uuid4().hex, "native-foreign-callback", agent_id=did)
    try:
        await tick_and_settle(runner)
        assert entered.is_set() and original_scopes
        assert (
            runner.readiness_failure is failure
            and callbacks
            and all(callback is failure for callback in callbacks)
        ), "foreign native scheduler denial was localized solely by fence type"
        assert not runner._occurrence_control_receipts
    finally:
        try:
            await runner.stop()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
        await foreign.stop()
