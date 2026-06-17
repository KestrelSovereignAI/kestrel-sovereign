"""Regression: a COGNITION dispatch reaching process_input before the agent
finishes initialize() must DEFER (clean retryable error), not crash.

The restart.completed wake (RestartCoordinatorFeature.initialize → post-restart
sweep) dispatches a COGNITION signal to the requesting agent. If that turn
reaches process_input before initialize() has constructed self.context_manager,
the old code raised `AttributeError: 'KestrelAgent' object has no attribute
'context_manager'` — observed live in Emma's signal_log. The dispatcher records
that as Status.FAILED and #1797 retries, but it burns retries, runs a half-built
turn, and the wake lands late/unreliably.

Fix: default context_manager to None in __init__ and, in process_input, defer
with a clear RuntimeError when it is still None (dispatcher → Status.FAILED →
retried after init completes). Bootstrap / safe-mode / !command paths still run.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.kestrel_agent import KestrelAgent


def _preinit_agent() -> KestrelAgent:
    """A KestrelAgent stopped at the pre-context-manager point of init.

    Built via __new__ so we control exactly which attributes exist: the heavy
    pre-context steps are stubbed and context_manager is None, mirroring the
    window between __init__ and initialize() completing.
    """
    agent = KestrelAgent.__new__(KestrelAgent)
    agent._safe_mode = False
    agent._maybe_audit = AsyncMock()
    agent.bootstrap_service = None          # not yet constructed
    agent.context_manager = None            # the race: still unset
    agent.did = "did:test:preinit"

    @asynccontextmanager
    async def _noop_lifecycle():
        yield
    agent._turn_lifecycle = _noop_lifecycle
    return agent


@pytest.mark.asyncio
async def test_cognition_turn_before_init_defers_with_clear_error():
    agent = _preinit_agent()

    with pytest.raises(RuntimeError) as exc:
        await agent.process_input("restart.completed wake")

    msg = str(exc.value)
    # Must be the intentional deferral, not the opaque AttributeError.
    assert "not fully initialized" in msg
    assert "context_manager" in msg


@pytest.mark.asyncio
async def test_preinit_turn_does_not_raise_attributeerror():
    """Belt-and-suspenders: the failure must never resurface as AttributeError
    (which the dispatcher would still retry, but noisily and after a partial
    turn)."""
    agent = _preinit_agent()
    with pytest.raises(RuntimeError):
        await agent.process_input("wake")
    # If context_manager were accessed unguarded it would be AttributeError,
    # which is a subclass relationship we explicitly do NOT want here.
    try:
        await agent.process_input("wake")
    except AttributeError:  # pragma: no cover - regression tripwire
        pytest.fail("process_input raised AttributeError on a pre-init agent")
    except RuntimeError:
        pass
