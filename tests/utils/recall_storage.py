"""A real agent-bound storage behind the real privacy wrapper, for recall tests.

The generic recall read (#4155) reads only through ``PrivacyEnforcingStorage``
and checks a governed node's canonical assertion, so its tests need both: a
graph store bound to the agent and an assertion store.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Tuple

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_assertion_store import (
    _issue_assertion_tenant_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


@asynccontextmanager
async def governed_storage(
    path: str,
    agent_id: str,
    *,
    mode: PrivacyMode = PrivacyMode.NORMAL,
) -> AsyncIterator[Tuple[PrivacyEnforcingStorage, AsyncStorage]]:
    """``(wrapper, raw)`` over one SQLite file, closed on exit."""
    raw = AsyncStorage(
        path,
        agent_id=agent_id,
        _assertion_tenant_capability=_issue_assertion_tenant_capability(agent_id),
    )
    await raw.initialize()
    try:
        yield PrivacyEnforcingStorage(raw, mode), raw
    finally:
        await raw.close()
