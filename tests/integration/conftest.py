"""
Pytest configuration for integration tests.

Provides fixtures and utilities specific to integration testing,
including handling of bootstrap state for test agents.
"""
import os
import pytest
import pytest_asyncio


@pytest.fixture(autouse=True)
def _force_single_agent_mode(monkeypatch, tmp_path):
    """Prevent rookery.toml detection in integration tests.

    When rookery.toml exists in the project root, server.py starts in
    multi-agent mode, which breaks TestClient-based tests that expect
    a single agent on app.state.agent.
    """
    monkeypatch.setenv(
        "KESTREL_ROOKERY_CONFIG",
        str(tmp_path / "nonexistent_rookery.toml"),
    )


@pytest.fixture(autouse=True)
def _auto_grant_security_permissions(monkeypatch):
    """Grant ALLOW for every (feature, tool) pair on every test KestrelAgent.

    Background: commit 61b431a4 ("take the watchdog clock off
    human-input hooks") removed the hook manager's 5-second wait_for
    around hooks marked awaits_user_input=True.  That's correct in
    production — the approval modal stays open until a human clicks.
    But integration tests that exercise the orchestrator hook chain
    (PRE_TOOL_USE / PRE_SUBAGENT_CALL) have no UI to click, so the
    SecurityHook's request_approval call blocks indefinitely until
    pytest-timeout fires.  The integration-tests CI job has been
    skipped on every recent main commit, so this regression went
    unnoticed until issue #879 re-enabled the suite for a PR.

    The semantically correct fix in tests that aren't *about* the
    approval flow is to pre-grant ALLOW so the SecurityHook short-
    circuits at the permission check and never queues.  Mirrors a
    "trust-all" deployment.

    Patch ``KestrelAgent.initialize`` to invoke the bulk grant after
    a successful initialize().  Approval-flow tests
    (test_approval_*.py) build their own ApprovalQueue against a
    shim — they don't instantiate KestrelAgent, so this hook doesn't
    touch them and they keep driving the queue with their own
    HTTP/SSE clients.

    Production code paths are untouched.
    """
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    original_initialize = KestrelAgent.initialize

    async def patched_initialize(self, *args, **kwargs):
        result = await original_initialize(self, *args, **kwargs)
        try:
            await grant_all_permissions(self, reason="autouse-test-grant (#879)")
        except Exception:
            # Never let a permission-grant failure mask the agent's
            # own initialization success.  Tests that hit the security
            # hook will fail loudly; tests that don't will be unaffected.
            pass
        return result

    monkeypatch.setattr(KestrelAgent, "initialize", patched_initialize)


async def complete_bootstrap(agent):
    """
    Mark an agent's bootstrap as complete for testing.

    Integration tests typically want to test specific features
    without going through the bootstrap discovery flow.
    """
    if hasattr(agent, 'bootstrap_service') and agent.bootstrap_service:
        from kestrel_sovereign.bootstrap import BootstrapState
        await agent.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)


async def grant_all_permissions(agent, reason: str = "integration-test"):
    """
    Grant PermissionLevel.ALLOW for every (feature, tool) pair on this agent.

    Background: commit 61b431a4 ("take the watchdog clock off
    human-input hooks") removed the hook manager's 5-second wait_for
    around hooks marked awaits_user_input=True.  That's the correct
    production behaviour — the approval modal stays open until the
    human actually clicks.  But integration tests that exercise the
    full orchestrator hook chain (PRE_SUBAGENT_CALL / PRE_TOOL_USE)
    have no UI to click; the SecurityHook's request_approval call
    blocks indefinitely until pytest-timeout fires.

    The semantically correct fix in tests that aren't *about* the
    approval flow is to pre-grant ALLOW so the SecurityHook short-
    circuits at the permission check and never queues for human
    input.  This mirrors how an operator running an agent in a
    "trust-all" deployment would configure it: every tool's
    PermissionLevel is set to ALLOW up front.

    Production code paths are untouched.  Tests that *do* exercise
    the approval flow itself (test_approval_*.py) leave permissions
    at the default ASK and drive the queue with their own HTTP/SSE
    clients — they intentionally don't call this helper.
    """
    if not hasattr(agent, "get_feature"):
        return

    security_feature = agent.get_feature("SecurityFeature")
    if not security_feature or not getattr(security_feature, "permission_store", None):
        return

    from kestrel_sovereign.features.security.permissions import PermissionLevel

    store = security_feature.permission_store

    # Tools registered via SecurityFeature._register_all_tools (the
    # PRE_TOOL_USE path).
    tree = await store.get_permission_tree()
    for feature_perms in tree:
        for tool in feature_perms.tools:
            await store.set_permission(
                feature_name=feature_perms.feature_name,
                tool_name=tool.tool_name,
                level=PermissionLevel.ALLOW,
                scope="always",
                reason=reason,
            )

    # Subagent dispatch names (PRE_SUBAGENT_CALL path) and the
    # direct-tool dispatch path use different feature_name conventions
    # in the SecurityHook input — the subagent path uses the class
    # name (type(feature).__name__) while _dispatch_direct_tool uses
    # the lowercase tool_name from _tool_to_feature.  Grant ALLOW for
    # both shapes so neither path queues for human approval:
    #
    #   1. (ClassName, feature.tool_name) — subagent dispatch on
    #      _dispatch_feature_tool, e.g. ("ModelAgent", "model_agent").
    #   2. (feature.tool_name, registered_tool_name) — direct tool
    #      dispatch on _dispatch_direct_tool, e.g. ("model_agent",
    #      "list_models").  These rows aren't in the registered-tools
    #      tree because that tree is keyed by class name.
    if hasattr(agent, "features"):
        for feature_name, feature in agent.features.items():
            tool_name = getattr(feature, "tool_name", None)
            if not tool_name:
                continue
            # Subagent dispatch: (ClassName, feature.tool_name)
            await store.set_permission(
                feature_name=feature_name,
                tool_name=tool_name,
                level=PermissionLevel.ALLOW,
                scope="always",
                reason=reason,
            )
            # Direct-tool dispatch: (feature.tool_name, each_tool_name)
            get_tools = getattr(feature, "get_tools", None)
            if callable(get_tools):
                try:
                    for tool_obj in get_tools():
                        inner_tool_name = getattr(tool_obj, "name", None)
                        if not inner_tool_name:
                            continue
                        await store.set_permission(
                            feature_name=tool_name,
                            tool_name=inner_tool_name,
                            level=PermissionLevel.ALLOW,
                            scope="always",
                            reason=reason,
                        )
                except Exception:
                    # Some features build their tool list lazily; if
                    # we can't enumerate, the registered-tools tree
                    # path above already covers them under the class
                    # name.
                    pass


@pytest.fixture
def skip_bootstrap():
    """
    Fixture that provides the complete_bootstrap helper.

    Usage in tests:
        async def test_something(kestrel_agent, skip_bootstrap):
            await skip_bootstrap(kestrel_agent)
            # Now the agent won't show bootstrap prompts
    """
    return complete_bootstrap
