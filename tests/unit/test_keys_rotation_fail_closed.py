"""Key rotation fails CLOSED when the approval queue raises (#1723).

Previously a constitutional-approval-queue exception during rotation was logged
and rotation proceeded anyway (reported PARTIAL). That silently downgrades a
security control. Rotation must now refuse by default and only proceed under an
explicit KESTREL_KEYS_ROTATE_WITHOUT_APPROVAL override.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.features.keys.feature import KeyManagementFeature


def _feature_with_crashing_approval():
    feature = object.__new__(KeyManagementFeature)  # bypass __init__/storage wiring
    feature._persistent_key_storage_hidden = lambda: False
    feature._ensure_storage = lambda: True
    store_key = AsyncMock(return_value="key-123")
    feature._storage = MagicMock(store_key=store_key)

    approval_queue = MagicMock()
    approval_queue.request_approval = AsyncMock(side_effect=RuntimeError("queue down"))
    security = MagicMock(approval_queue=approval_queue)
    agent = MagicMock()
    agent.get_feature = MagicMock(return_value=security)
    feature.agent = agent
    return feature, store_key


@pytest.mark.asyncio
async def test_rotation_fails_closed_when_approval_queue_raises(monkeypatch):
    monkeypatch.delenv("KESTREL_KEYS_ROTATE_WITHOUT_APPROVAL", raising=False)
    feature, store_key = _feature_with_crashing_approval()

    result = await feature.rotate_service_key("openai", "new-secret")

    assert result.status == "error"  # fail closed
    store_key.assert_not_awaited()   # rotation did NOT happen


@pytest.mark.asyncio
async def test_rotation_proceeds_with_partial_under_override(monkeypatch):
    monkeypatch.setenv("KESTREL_KEYS_ROTATE_WITHOUT_APPROVAL", "1")
    feature, store_key = _feature_with_crashing_approval()

    result = await feature.rotate_service_key("openai", "new-secret")

    # Override: rotation proceeds but is surfaced as PARTIAL (not silent OK).
    assert result.status == "partial"
    store_key.assert_awaited_once()
