"""Real native control must survive the original approval queue and policy I/O."""
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    is_execution_control_error,
)
from kestrel_sovereign.features.security.approval_queue import (
    ApprovalQueue,
    ApprovalRequest,
    ApprovalStatus,
)
from kestrel_sovereign.features.security.permissions import PermissionStore
from kestrel_sovereign.security.auto_approve import AutoApprovePolicy

pytestmark = pytest.mark.asyncio


def original_control(agent, carrier):
    try:
        agent._get_privacy_transition_lock().require_native_scope(None)
    except ExecutionAuthorityError as authority:
        if carrier == "group":
            error = ExceptionGroup("original native control", [authority])
        elif carrier == "timeout":
            try:
                raise TimeoutError("Original authority timeout carrier") from authority
            except TimeoutError as timeout:
                error = timeout
        else:
            error = authority
        assert is_execution_control_error(error)
        return error
    pytest.fail("Core admitted an absent native authority")


@pytest.mark.parametrize("carrier", ["direct", "group", "timeout"])
@pytest.mark.parametrize("boundary", [
    "permission_read", "dynamic_rules", "request_added", "request_wait",
    "decision_audit", "withdrawn", "sweep",
])
async def test_original_approval_boundaries_preserve_native_control(
    native_tool_effect_agent, tmp_path, monkeypatch, carrier, boundary,
):
    agent = await native_tool_effect_agent(did="did:test:approval-control")
    error = original_control(agent, carrier)
    store = PermissionStore(str(tmp_path / "owned-permissions.db"))
    await store.initialize()
    queue = ApprovalQueue()
    request_args = dict(feature_name="computer_use", tool_name="shell",
                        tool_args={"argv": ["gh", "issue", "view", "1", "-R", "KestrelSovereignAI/kestrel-sovereign"]})
    async def reject(*args, **kwargs):
        raise error
    if boundary == "permission_read":
        original = store.get_permission
        async def original_read_then_fence(*args, **kwargs):
            await original(*args, **kwargs)
            raise error
        monkeypatch.setattr(store, "get_permission", original_read_then_fence)
        queue = ApprovalQueue(permission_store=store)
        operation = queue.request_approval(**request_args, allow_blocking=False)
    elif boundary == "dynamic_rules":
        original = store.list_auto_approve_rules
        async def original_rules_then_fence(*args, **kwargs):
            await original(*args, **kwargs)
            raise error
        monkeypatch.setattr(store, "list_auto_approve_rules", original_rules_then_fence)
        queue = ApprovalQueue(permission_store=store, auto_approve_policy=AutoApprovePolicy([], store))
        operation = queue.request_approval(**request_args, allow_blocking=False)
    elif boundary == "request_added":
        queue = ApprovalQueue(on_request_added=reject)
        operation = queue.request_approval(**request_args, timeout=0)
    elif boundary == "request_wait":
        async def arm_original_request(request):
            monkeypatch.setattr(request.resume_event, "wait", reject)
        queue = ApprovalQueue(on_request_added=arm_original_request)
        operation = queue.request_approval(**request_args, timeout=None)
    elif boundary == "decision_audit":
        original = store.log_decision
        async def original_audit_then_fence(*args, **kwargs):
            await original(*args, **kwargs)
            raise error
        monkeypatch.setattr(store, "log_decision", original_audit_then_fence)
        queue = ApprovalQueue(permission_store=store)
        operation = queue._persist_decision(**request_args, approved=False,
                                            scope="no_approver", audit_action="tool_execution")
    elif boundary == "withdrawn":
        queue = ApprovalQueue(on_request_withdrawn=reject)
        operation = queue.request_approval(**request_args, timeout=0)
    else:
        queue = ApprovalQueue(on_request_withdrawn=reject)
        request = ApprovalRequest(id=uuid4().hex, created_at=datetime.now(timezone.utc), **request_args)
        queue._pending[request.id] = request
        operation = queue.sweep_stale(older_than_seconds=0)
    with pytest.raises(type(error)) as captured:
        await operation
    assert captured.value is error, "Approval replaced original native control"
    if boundary in {"request_added", "request_wait"}:
        assert len(queue.pending_requests) == 1
        assert queue.pending_requests[0].status is ApprovalStatus.AWAITER_GONE, (
            "Approval retained a live-looking request after native control ended its awaiter")
    elif boundary != "decision_audit":
        assert not queue.pending_requests
