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
    """Prevent multi_agent.toml detection in integration tests.

    When multi_agent.toml exists in the project root, server.py starts in
    multi-agent mode, which breaks TestClient-based tests that expect
    a single agent on app.state.agent.
    """
    monkeypatch.setenv(
        "KESTREL_MULTI_AGENT_CONFIG",
        str(tmp_path / "nonexistent_multi_agent.toml"),
    )


@pytest.fixture(autouse=True)
def _isolate_from_lighthouse_snapshots(monkeypatch):
    """Ensure tests don't restore prior-run state from Lighthouse.

    KestrelAgent.initialize does a cold-start restore from Lighthouse
    when LIGHTHOUSE_API_KEY is set and the storage_path doesn't
    exist yet (kestrel_agent.py around line 328).  Locally that's
    fine for production, but in tests it pulls a previous run's
    snapshot into a fresh tempdir DB — which can leak permission
    rows, conversation history, and audit log across tests.

    Concretely: a permission_store row written by yesterday's
    integration run can land in today's bare_agent fixture and
    silently grant ALLOW for a tool the test wanted at default
    ASK.  Test passes locally; fails on CI (where the env var
    isn't set).  Saw exactly that with #879's per-test grants.

    Strip the env var globally for the integration suite.  Tests
    that specifically want to exercise the Lighthouse restore path
    can re-set it inside their own fixture.
    """
    monkeypatch.delenv("LIGHTHOUSE_API_KEY", raising=False)


# NOTE: There is intentionally no autouse blanket-grant fixture here.
#
# An earlier iteration patched KestrelAgent.initialize to bulk-grant
# ALLOW for every registered (feature, tool) pair on every test
# agent.  That made the SecurityHook short-circuit on every
# permission check, which fixed the timeout — but it also stopped
# every integration test from exercising the security path at all.
# A regression that, say, dispatched a tool through a code path
# that bypassed the SecurityHook would have gone completely unseen.
#
# Tests that exercise the orchestrator hook chain (PRE_TOOL_USE /
# PRE_SUBAGENT_CALL) must now explicitly declare the (feature_name,
# tool_name) pairs they expect to dispatch, via ``grant_permissions``
# below.  This makes the security surface of each test self-
# documenting and means a regression that introduces a new gated
# tool surfaces as a hang in exactly the test that needs to know.


async def complete_bootstrap(agent):
    """
    Mark an agent's bootstrap as complete for testing.

    Integration tests typically want to test specific features
    without going through the bootstrap discovery flow.
    """
    if hasattr(agent, 'bootstrap_service') and agent.bootstrap_service:
        from kestrel_sovereign.bootstrap import BootstrapState
        await agent.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)


async def grant_permissions(agent, *tool_specs, reason: str = "integration-test"):
    """
    Grant PermissionLevel.ALLOW for the explicit (feature_name, tool_name) pairs.

    Each test that exercises a security-gated dispatch path lists the
    specific tools it expects to use.  Anything not listed remains at
    the default ASK and will hang the SecurityHook on
    request_approval — which is correct: a test that triggers an
    unexpected security-gated dispatch should fail loudly, not be
    silently waved through.

    Three (feature_name, tool_name) shapes are common, depending on
    how the orchestrator dispatches:

    1. Registered feature tools (PRE_TOOL_USE path through
       SecurityFeature._register_all_tools), e.g.
       ``("ModelAgent", "list_models")``.
    2. Subagent dispatch (PRE_SUBAGENT_CALL path through
       _dispatch_feature_tool), e.g. ``("ModelAgent", "model_agent")``
       — class name + feature.tool_name.
    3. Direct-tool dispatch (PRE_TOOL_USE through
       _dispatch_direct_tool), e.g. ``("model_agent", "list_models")``
       — lowercase feature.tool_name + the inner tool name, because
       _tool_to_feature is keyed by the lowercase tool_name.

    Example:
        await grant_permissions(
            agent,
            ("ModelAgent", "model_agent"),  # subagent dispatch
            ("model_agent", "list_models"),  # direct-tool dispatch
            reason="orchestrator-loop-test",
        )

    Background: commit 61b431a4 ("take the watchdog clock off
    human-input hooks") removed the hook manager's 5-second wait_for
    around hooks marked awaits_user_input=True.  That's the correct
    production behaviour — the approval modal stays open until the
    human actually clicks.  But integration tests have no UI to
    click; the SecurityHook's request_approval blocks indefinitely
    until pytest-timeout fires.  The integration-tests CI job has
    been silently skipped on every recent main commit, so the
    regression went unnoticed until #879 re-enabled the suite.

    Production code paths are untouched — the approval queue still
    blocks indefinitely waiting for a human.  Tests that *do*
    exercise the approval flow itself (test_approval_*.py) leave
    permissions at the default ASK and drive the queue with their
    own HTTP/SSE clients — they intentionally don't call this helper.
    """
    if not tool_specs:
        return
    if not hasattr(agent, "get_feature"):
        return

    security_feature = agent.get_feature("SecurityFeature")
    if not security_feature or not getattr(security_feature, "permission_store", None):
        return

    from kestrel_sovereign.features.security.permissions import PermissionLevel

    store = security_feature.permission_store
    for spec in tool_specs:
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise TypeError(
                f"grant_permissions expects (feature_name, tool_name) tuples, "
                f"got {spec!r}"
            )
        feature_name, tool_name = spec
        await store.set_permission(
            feature_name=feature_name,
            tool_name=tool_name,
            level=PermissionLevel.ALLOW,
            scope="always",
            reason=reason,
        )


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
