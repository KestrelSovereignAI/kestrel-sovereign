"""Regression test: ``AsyncStorage`` exposes delegator methods for every
conversation-session write path the privacy wrapper calls through it.

Context (observed on live Meridian, 2026-04-24):

- ``PrivacyEnforcingStorage`` wraps an ``AsyncStorage`` facade and calls
  ``self._storage.delete_conversation_session(...)`` etc.
- ``AsyncStorage`` delegates most conversation reads/writes to its inner
  ``AsyncConversationStore`` via explicit wrapper methods
  (``add_conversation``, ``get_conversation_history``, …).
- When issues #715 / #716 added ``delete_conversation_session`` /
  ``set_conversation_name`` / ``get_conversation_name`` /
  ``get_conversation_names`` to ``AsyncConversationStore``, they did NOT
  add the matching delegator to ``AsyncStorage``.  The privacy wrapper
  then got ``AttributeError: 'AsyncStorage' object has no attribute
  'delete_conversation_session'`` and the endpoint returned 500.

This test pins the contract at the facade boundary so the next person
adding a conversation-session method can't ship it without wiring the
wrapper.
"""

import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_storage import AsyncStorage


REQUIRED_DELEGATORS = [
    # (method name, kwargs, description)
    ("delete_conversation_session", {"session_id": "sess-1"}, "#715 delete"),
    ("set_conversation_name", {"session_id": "sess-1", "name": "x"}, "#716 rename"),
    ("get_conversation_name", {"session_id": "sess-1"}, "#716 read"),
    ("get_conversation_names", {}, "#716 bulk read"),
]


@pytest.fixture
async def storage():
    """Real SQLite-backed AsyncStorage — no mocks, because the regression
    is specifically about method resolution between the facade and its
    inner store.  Mocks would hide it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        s = await AsyncStorage.create_sqlite(str(db_path))
        s.agent_id = "test-agent"
        # Re-initialize the conversation store so agent_id propagates into
        # its own attribute (create_sqlite caches before we set it).
        from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
        s.conversation = AsyncConversationStore(s.db, agent_id="test-agent")
        yield s
        await s.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method,kwargs,description", REQUIRED_DELEGATORS)
async def test_async_storage_exposes_conversation_session_delegators(
    storage, method, kwargs, description
):
    """Every session-level conversation method on AsyncConversationStore
    must have a matching delegator on AsyncStorage.  The privacy wrapper
    reaches the underlying store through the facade; missing delegators
    surfaced as live 500 errors on conversation delete / rename.
    """
    assert hasattr(storage, method), (
        f"AsyncStorage is missing delegator for {method!r} "
        f"({description}).  PrivacyEnforcingStorage calls "
        f"self._storage.{method}(...), so this missing method breaks "
        "that path at runtime with AttributeError."
    )
    # The method must be callable and must not blow up on well-formed
    # input.  We don't assert on return values here — those invariants
    # live in the per-issue storage tests.  What this test defends is
    # purely the delegation contract.
    fn = getattr(storage, method)
    await fn(**kwargs)


@pytest.mark.asyncio
async def test_delete_conversation_session_returns_zero_for_unknown_session(storage):
    """Defensive sanity: delegating to storage for an unknown session
    returns 0 instead of crashing — matches the endpoint's 404 path.
    """
    result = await storage.delete_conversation_session("never-existed")
    assert result == 0


@pytest.mark.asyncio
async def test_set_conversation_name_roundtrips_through_facade(storage):
    """Facade-level rename + read round-trips correctly.  Previously the
    privacy-wrapper → facade → store chain was broken at the first arrow;
    verify it works end-to-end now.
    """
    stored = await storage.set_conversation_name("sess-rt", "My Thread")
    assert stored == "My Thread"
    fetched = await storage.get_conversation_name("sess-rt")
    assert fetched == "My Thread"

    # Bulk read sees it too.
    names = await storage.get_conversation_names()
    assert names == {"sess-rt": "My Thread"}

    # Clear via facade.
    cleared = await storage.set_conversation_name("sess-rt", "")
    assert cleared is None
    assert await storage.get_conversation_name("sess-rt") is None
    assert await storage.get_conversation_names() == {}
