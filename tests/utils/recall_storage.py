"""A real agent-bound storage behind the real privacy wrapper, for recall tests.

The generic recall read (#4155) reads only through ``PrivacyEnforcingStorage``
and checks a governed node's canonical assertion, so its tests and the tests
of its aliases need both: a graph store bound to the agent and an assertion
store. ``graph`` swaps a test double in for the graph store, for tests whose
subject is the strategy index rather than the read.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Tuple

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_assertion_store import (
    _issue_assertion_tenant_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


async def open_governed_storage(
    path: str,
    agent_id: str,
    *,
    mode: PrivacyMode = PrivacyMode.NORMAL,
    graph: Optional[Any] = None,
) -> Tuple[PrivacyEnforcingStorage, AsyncStorage]:
    """Open ``(wrapper, raw)``; the caller closes ``raw``."""
    raw = AsyncStorage(
        path,
        agent_id=agent_id,
        _assertion_tenant_capability=_issue_assertion_tenant_capability(agent_id),
    )
    await raw.initialize()
    if graph is not None:
        raw.graph = graph
    return PrivacyEnforcingStorage(raw, mode), raw


@asynccontextmanager
async def governed_storage(
    path: str,
    agent_id: str,
    *,
    mode: PrivacyMode = PrivacyMode.NORMAL,
    graph: Optional[Any] = None,
) -> AsyncIterator[Tuple[PrivacyEnforcingStorage, AsyncStorage]]:
    wrapper, raw = await open_governed_storage(path, agent_id, mode=mode, graph=graph)
    try:
        yield wrapper, raw
    finally:
        await raw.close()
