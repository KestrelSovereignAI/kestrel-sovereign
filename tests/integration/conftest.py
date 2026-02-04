"""
Pytest configuration for integration tests.

Provides fixtures and utilities specific to integration testing,
including handling of bootstrap state for test agents.
"""
import pytest
import pytest_asyncio


async def complete_bootstrap(agent):
    """
    Mark an agent's bootstrap as complete for testing.

    Integration tests typically want to test specific features
    without going through the bootstrap discovery flow.
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
