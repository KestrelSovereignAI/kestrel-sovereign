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
def _auto_approve_security_requests(monkeypatch):
    """Auto-approve every SecurityHook request raised during a test.

    Background: commit 61b431a4 ("take the watchdog clock off
    human-input hooks") removed the hook manager's 5-second
    asyncio.wait_for around hooks marked awaits_user_input=True. That
    fix is correct in production — it's what makes the approval modal
    stay open until a user actually clicks. But integration tests
    that exercise PRE_TOOL_USE / PRE_SUBAGENT_CALL through the real
    orchestrator have no UI to click; without a responder the
    SecurityHook's request_approval call blocks indefinitely until
    pytest-timeout fires.

    The integration-tests CI job has been silently skipped on every
    recent main commit, so this regression went unnoticed until issue
    #879 re-enabled the suite for a PR.

    Patch ApprovalQueue.__init__ for the test session: any queue
    created (one per KestrelAgent) gets a default on_request_added
    that immediately submits an "always"-scoped approval. The
    original SSE callback (when set by SecurityFeature) is preserved
    and runs first so observers still see the request transit.
    Production code paths are untouched.
    """
    from kestrel_sovereign.features.security import approval_queue as aq_mod

    original_init = aq_mod.ApprovalQueue.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        original_callback = self._on_request_added

        async def _auto_approve(request):
            if original_callback:
                try:
                    await original_callback(request)
                except Exception:
                    pass
            self.submit_decision(request.id, approved=True, scope="always")

        self._on_request_added = _auto_approve

    monkeypatch.setattr(aq_mod.ApprovalQueue, "__init__", patched_init)


async def complete_bootstrap(agent):
    """
    Mark an agent's bootstrap as complete for testing.

    Integration tests typically want to test specific features
    without going through the bootstrap discovery flow.

    Note: SecurityHook approval auto-approval is wired separately
    via the ``_auto_approve_security_requests`` autouse fixture
    above, so all integration tests get the responder regardless of
    whether they call this helper.
    """
    if hasattr(agent, 'bootstrap_service') and agent.bootstrap_service:
        from kestrel_sovereign.bootstrap import BootstrapState
        await agent.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)


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
