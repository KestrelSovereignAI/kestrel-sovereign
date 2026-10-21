"""Refused/accepted publication failures keep the SAME canonical ingress owner."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.unit.test_isolated_feature_runtime import (
    FakeIsolatedClient, _initialized_host_ingress_proxy, _traceback_locals_reach_any,
)
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("publication", [False, True])
async def test_ingress_retains_original_admission_before_public_redaction(native_pg, tmp_path, monkeypatch, publication):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, FakeIsolatedClient)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    accepted, recipes = [], []
    class PublicationTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.failed and getattr(callback, "__name__", "") == "close_unstarted":
                self.failed = True
                super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname == "ProxyFeature._run_host_ingress":
            recipes.append(operation)
            if not publication:
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    try:
        loop.set_task_factory(factory)
        outcome = await await_owned_task(create_owned_task(feature.call_host_ingress("private-input", {"secret": "ingress-admission-secret"}), retain_originals=True))
        loop.set_task_factory(previous)
        assert is_execution_control_error(outcome.error) and execution_commit_outcome(outcome.error) is None
        (record,) = feature._host_ingress_admissions
        source.assert_retained(record.native_error)
        assert record.admission.task is (accepted[0] if accepted else None)
        assert all(task.done() for task in accepted) and recipes[0].cr_frame is None
        assert not _traceback_locals_reach_any(outcome.error, (feature, record, source.error, *accepted))
        with pytest.raises(BaseException) as denied:
            feature._require_event_ingress_work()
        assert is_execution_control_error(denied.value), "failed original Core ingress admission reopened another producer"
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
