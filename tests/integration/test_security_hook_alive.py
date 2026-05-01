"""
Sanity tests for the SecurityHook chain in integration mode.

These tests guard the test-suite assumption that explicit
``grant_permissions(...)`` calls in fixtures are the *reason* tests
don't hang on the SecurityHook — not because the hook itself has
been silently bypassed.

The earlier autouse blanket-grant fixture (commit 9a9a1ec3) made
every integration test stop firing the security path entirely; a
regression that broke the SecurityHook would have looked exactly
like a passing test.  These assertions establish the inverse: a
tool that wasn't pre-granted MUST queue for approval, and the
SecurityHook MUST be present in the agent's hook chain.

Tied to issue #879 follow-up.  See
``tests/integration/conftest.grant_permissions`` for the explicit-
grant API and the production background.
"""

import asyncio

import pytest
import pytest_asyncio

from kestrel_sovereign.hooks import HookEvent, HookInput
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode


@pytest_asyncio.fixture
async def bare_agent(temp_db, monkeypatch):
    """A real KestrelAgent with NO pre-granted permissions.

    Deliberately doesn't call ``grant_permissions`` — every tool
    stays at the default PermissionLevel.ASK so the SecurityHook
    actually exercises its queue path.

    Two cleanup steps are needed to guarantee the bare state:

    1. Unset ``LIGHTHOUSE_API_KEY`` for this fixture so the agent's
       cold-start restore (kestrel_agent.initialize) doesn't pull a
       prior test's snapshot from Lighthouse.  Without this the
       fresh ``temp_db`` gets overwritten with whatever permissions
       a previous test wrote, and "ungranted" stops being true.

    2. After ``initialize()`` runs ``_register_all_tools`` (which
       inserts every tool at the default ASK level via INSERT OR
       IGNORE), no further grants are made — but if any cross-test
       cache leaks through we DELETE every row tagged with a known
       test-grant reason as a belt-and-braces clear.
    """
    monkeypatch.delenv("LIGHTHOUSE_API_KEY", raising=False)

    llm_service = LLMService()
    agent = KestrelAgent(
        did="did:test:security-hook-alive",
        storage_path=str(temp_db),
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()

    from tests.integration.conftest import complete_bootstrap
    await complete_bootstrap(agent)

    # Reset every permission row to the default ASK level — even if a
    # cross-test cache leaked, this guarantees the assertion below
    # (level == ASK on an "ungranted" tool) reflects fixture state,
    # not a stale snapshot.
    security_feature = agent.get_feature("SecurityFeature")
    if security_feature and security_feature.permission_store:
        import aiosqlite
        async with aiosqlite.connect(
            security_feature.permission_store.db_path
        ) as db:
            await db.execute(
                "UPDATE security_permissions SET level='ask', reason=NULL"
            )
            await db.commit()

    yield agent

    await agent.shutdown()
    await llm_service.close()


@pytest.mark.asyncio
async def test_security_guard_hook_is_registered_on_pre_tool_use(bare_agent):
    """SecurityFeature must register ``security_guard`` against PRE_TOOL_USE
    and PRE_SUBAGENT_CALL.  Catches a regression where the hook is
    no longer wired into the manager — without which every tool
    dispatch would skip the security check entirely.
    """
    manager = bare_agent.hooks_manager
    pre_tool_hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
    pre_subagent_hooks = manager.get_hooks(HookEvent.PRE_SUBAGENT_CALL)

    pre_tool_names = [h.name for h in pre_tool_hooks]
    pre_subagent_names = [h.name for h in pre_subagent_hooks]

    assert "security_guard" in pre_tool_names, (
        f"SecurityHook missing from PRE_TOOL_USE chain: {pre_tool_names}"
    )
    assert "security_guard" in pre_subagent_names, (
        f"SecurityHook missing from PRE_SUBAGENT_CALL chain: {pre_subagent_names}"
    )


@pytest.mark.asyncio
async def test_ungranted_tool_queues_for_approval(bare_agent):
    """An ungranted tool dispatch must enqueue an approval request.

    Fires the SecurityHook for ``("ModelAgent", "list_models")`` —
    a tool that is registered with default ASK permission but NOT
    pre-granted by this fixture.  The hook should call into
    ``approval_queue.request_approval`` which then sits waiting on
    its resume_event.  We don't await the hook (it would block
    forever); we just verify the queue grew by one entry, proving
    the hook chain dispatched into the security path end-to-end.
    """
    from kestrel_sovereign.features.security.permissions import PermissionLevel

    security_feature = bare_agent.get_feature("SecurityFeature")
    assert security_feature is not None
    store = security_feature.permission_store
    queue = security_feature.approval_queue
    assert queue.pending_count == 0

    # Sanity: the bare-agent fixture installs no grants, so the
    # registered ``("ModelAgent", "list_models")`` row should be at
    # the default ASK level.
    level = await store.get_permission("ModelAgent", "list_models")
    assert level == PermissionLevel.ASK, (
        f"Expected default ASK on ungranted tool, got {level!r}.  "
        f"If a fixture above is silently granting permissions, the "
        f"sanity test below is meaningless."
    )

    hook_input = HookInput(
        session_id="security-alive-test",
        hook_event_name=HookEvent.PRE_TOOL_USE.value,
        tool_name="list_models",
        feature_name="ModelAgent",
        tool_input={"use_cache": False},
    )

    # Fire the hook chain in a background task so we don't block on
    # request_approval's event.wait().  The task will sit pending
    # until we cancel or submit a decision.
    hook_task = asyncio.create_task(
        bare_agent.hooks_manager.execute_hooks(
            HookEvent.PRE_TOOL_USE, hook_input
        )
    )
    try:
        # Poll for the approval to appear.  500ms is generous — the
        # hook only has to traverse the manager and reach
        # request_approval.  The first iteration's permission DB
        # read is the slow step; subsequent calls are warm.
        for _ in range(50):
            if queue.pending_count >= 1:
                break
            await asyncio.sleep(0.01)

        assert queue.pending_count == 1, (
            "Expected SecurityHook to enqueue an approval request for an "
            "ungranted tool dispatch, but the queue is empty.  Either the "
            "hook is no longer wired, or the dispatch path bypasses it.\n"
            f"hook_task done={hook_task.done()}, "
            f"exception={hook_task.exception() if hook_task.done() else 'pending'}"
        )

        pending = queue.pending_requests[0]
        assert pending.feature_name == "ModelAgent"
        assert pending.tool_name == "list_models"
    finally:
        # Clean up: deny the request so the hook task unblocks, then
        # await to satisfy asyncio task hygiene.  Without this, the
        # queued request leaks into the agent's shutdown which logs
        # a benign "cancelled_all" line.
        if queue.pending_count > 0:
            queue.submit_decision(
                queue.pending_requests[0].id,
                approved=False,
                scope="once",
            )
        try:
            await asyncio.wait_for(hook_task, timeout=2.0)
        except asyncio.TimeoutError:
            hook_task.cancel()


@pytest.mark.asyncio
async def test_explicit_grant_lets_hook_short_circuit(bare_agent):
    """Pre-granting ALLOW for a specific tool must skip the queue.

    Inverse of the previous test: after ``grant_permissions``, the
    SecurityHook returns ALLOW immediately without queueing.  This
    proves the explicit grants in other test fixtures work the way
    we claim (and not that those tests pass for some unrelated
    reason).
    """
    from tests.integration.conftest import grant_permissions
    await grant_permissions(
        bare_agent,
        ("ModelAgent", "list_models"),
        reason="security-hook-alive sanity",
    )

    security_feature = bare_agent.get_feature("SecurityFeature")
    queue = security_feature.approval_queue
    assert queue.pending_count == 0

    hook_input = HookInput(
        session_id="security-alive-test",
        hook_event_name=HookEvent.PRE_TOOL_USE.value,
        tool_name="list_models",
        feature_name="ModelAgent",
        tool_input={"use_cache": False},
    )

    # This should return synchronously (no queue traffic) because
    # the permission is now ALLOW.
    output = await asyncio.wait_for(
        bare_agent.hooks_manager.execute_hooks(
            HookEvent.PRE_TOOL_USE, hook_input
        ),
        timeout=2.0,
    )

    assert queue.pending_count == 0, (
        "Pre-granted tool should not have queued for approval"
    )
    # The hook output should be ALLOW; HooksManager surfaces this
    # via permission_decision on the aggregated output.
    from kestrel_sovereign.hooks.base import PermissionDecision
    assert output.permission_decision == PermissionDecision.ALLOW
