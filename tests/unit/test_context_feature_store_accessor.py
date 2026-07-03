"""Regression tests for F152 — the context/save tool paths that call
``context_manager._get_conversation_store()``.

``ContextManager`` previously defined no ``_get_conversation_store``
accessor, so five tool paths raised ``AttributeError`` at runtime:

  - context feature: restore_excluded (target="recent"), query "excluded",
    query "compacted:"/"summary:"
  - save feature: two paths

These tests drive the accessor through a **real** ``ContextManager`` and a
realistic conversation store (not a MagicMock context_manager), so the
missing method cannot silently regress.
"""

from __future__ import annotations

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.features.context.feature import ContextFeature
from kestrel_sovereign.features.save.feature import SaveFeature


class _FakeConversationStore:
    """Minimal realistic conversation store.

    Exposes the async methods the F152 tool paths reach for. Records
    whether they were called so tests can assert the real store was hit.
    """

    def __init__(self, excluded=None):
        self._excluded = excluded or []
        self.get_excluded_calls = 0

    async def get_excluded_messages(self, limit=10):
        self.get_excluded_calls += 1
        return list(self._excluded)[:limit]


class _FakeStorage:
    """Storage facade shaped like production: exposes ``.conversation``.

    ``ConversationManager._get_conversation_store`` resolves the store from
    ``storage.conversation`` (or the nested ``_storage.conversation``); this
    fake mirrors the flat facade shape.
    """

    def __init__(self, conversation):
        self.conversation = conversation


def _make_context_manager(conv_store):
    return ContextManager(
        storage=_FakeStorage(conv_store),
        agent_id="did:test:agent",
    )


def test_context_manager_get_conversation_store_returns_real_store():
    """The accessor exists on ContextManager and delegates to the
    ConversationManager, resolving the real store."""
    conv = _FakeConversationStore()
    cm = _make_context_manager(conv)

    resolved = cm._get_conversation_store()

    assert resolved is conv
    # And it is the same store the ConversationManager resolves.
    assert resolved is cm.conversation_manager._get_conversation_store()


@pytest.mark.asyncio
async def test_restore_excluded_recent_resolves_real_store():
    """restore_excluded(target="recent") drives the accessor through a real
    ContextManager without raising AttributeError (F152 path @ feature.py:627)."""
    conv = _FakeConversationStore(excluded=[])
    cm = _make_context_manager(conv)

    feature = ContextFeature.__new__(ContextFeature)
    feature.context_manager = cm

    result = await feature.restore_excluded(target="recent")

    # Empty exclusions -> honest no-op OK, but the store was really hit.
    assert conv.get_excluded_calls == 1
    assert result.status == ToolResultStatus.OK


@pytest.mark.asyncio
async def test_save_feature_resolves_real_store():
    """A SaveFeature configured with a real ContextManager resolves the
    same store (F152 path @ save/feature.py)."""
    conv = _FakeConversationStore()
    cm = _make_context_manager(conv)

    feature = SaveFeature.__new__(SaveFeature)
    feature.context_manager = cm

    assert feature.context_manager._get_conversation_store() is conv
