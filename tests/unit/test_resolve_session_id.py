"""
``resolve_session_id`` is the public surface that lets the streaming
endpoint echo the effective session_id back to the client. Without
it, the server would derive an implicit UUID inside add_conversation,
the client would never see it, and the frontend pane would stay
anchored to ``sessionId = null`` forever — leaving auto-load and
context-status fragile (reviewer flagged this).

These tests pin the resolver's contract:
- Explicit session_id always wins
- Missing session_id is resolved by the same time-gap heuristic
  add_conversation uses internally (so the client gets back the
  same id that ends up in the row's metadata)
- EPHEMERAL / ISOLATED privacy modes pass through (they have no
  durable history to derive from)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_resolve_returns_explicit_value_unchanged():
    """If the caller passes a session_id, the resolver must echo it
    back verbatim — never re-derive."""
    from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore

    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store._derive_implicit_session_id = AsyncMock(side_effect=AssertionError(
        "explicit value must NOT trigger re-derivation"
    ))

    out = await store.resolve_session_id("explicit-sess-123")
    assert out == "explicit-sess-123"


@pytest.mark.asyncio
async def test_resolve_derives_when_caller_passes_none():
    """When the caller passes None, the resolver delegates to the
    internal time-gap derivation. Without this surfacing the implicit
    id, the client would never learn it."""
    from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore

    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store._derive_implicit_session_id = AsyncMock(return_value="derived-uuid-abc")

    out = await store.resolve_session_id(None)
    assert out == "derived-uuid-abc"
    store._derive_implicit_session_id.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_propagates_none_when_derive_returns_none():
    """If the derivation step fails or returns None (legacy data, no
    prior rows), the resolver returns None instead of fabricating
    a value. The endpoint then omits the X-Session-Id header — caller
    sees the absence and knows no session_id is available yet."""
    from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore

    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store._derive_implicit_session_id = AsyncMock(return_value=None)

    out = await store.resolve_session_id(None)
    assert out is None


@pytest.mark.asyncio
async def test_privacy_wrapper_resolve_passes_through_for_ephemeral():
    """EPHEMERAL has no persistent history, so the time-gap heuristic
    has nothing to read. Passing through the caller-provided value
    (even if None) is the correct semantics — it stays explicit."""
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    wrapper = PrivacyEnforcingStorage.__new__(PrivacyEnforcingStorage)
    wrapper._privacy_config = MagicMock()
    wrapper._privacy_config.is_ephemeral = MagicMock(return_value=True)
    wrapper._policy = MagicMock()
    wrapper._policy.use_session_storage = False
    wrapper._storage = MagicMock()
    wrapper._storage.resolve_session_id = AsyncMock(side_effect=AssertionError(
        "EPHEMERAL must NOT delegate to persistent store"
    ))

    assert await wrapper.resolve_session_id("explicit") == "explicit"
    assert await wrapper.resolve_session_id(None) is None


@pytest.mark.asyncio
async def test_privacy_wrapper_resolve_passes_through_for_isolated():
    """ISOLATED uses an in-memory session-local list — no durable
    history to drive time-gap derivation, so we pass through."""
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    wrapper = PrivacyEnforcingStorage.__new__(PrivacyEnforcingStorage)
    wrapper._privacy_config = MagicMock()
    wrapper._privacy_config.is_ephemeral = MagicMock(return_value=False)
    wrapper._policy = MagicMock()
    wrapper._policy.use_session_storage = True
    wrapper._storage = MagicMock()
    wrapper._storage.resolve_session_id = AsyncMock(side_effect=AssertionError(
        "ISOLATED must NOT delegate to persistent store"
    ))

    assert await wrapper.resolve_session_id("explicit") == "explicit"
    assert await wrapper.resolve_session_id(None) is None


@pytest.mark.asyncio
async def test_privacy_wrapper_resolve_delegates_for_normal():
    """NORMAL/PUBLIC uses persistent storage — delegate to its
    resolver so the time-gap heuristic runs and returns the same id
    add_conversation will stamp into row metadata."""
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    wrapper = PrivacyEnforcingStorage.__new__(PrivacyEnforcingStorage)
    wrapper._privacy_config = MagicMock()
    wrapper._privacy_config.is_ephemeral = MagicMock(return_value=False)
    wrapper._policy = MagicMock()
    wrapper._policy.use_session_storage = False
    wrapper._storage = MagicMock()
    wrapper._storage.resolve_session_id = AsyncMock(return_value="from-store-xyz")

    assert await wrapper.resolve_session_id(None) == "from-store-xyz"
    wrapper._storage.resolve_session_id.assert_awaited_once_with(None)
