"""Borrowed host storage stays open until its actual owner retires it."""

from __future__ import annotations

import pytest

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.context_store import ContextStore
from kestrel_sovereign.a2a.stores.unified.feedback_store import FeedbackStore
from kestrel_sovereign.a2a.stores.unified.memory_service import MemoryService
from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.a2a.stores.unified.orchestration_store import OrchestrationStore
from kestrel_sovereign.a2a.stores.unified.session_service import SessionService
from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore


@pytest.mark.parametrize(
    "store_type",
    [
        ContextStore,
        FeedbackStore,
        MemoryService,
        ObservabilityStore,
        OrchestrationStore,
        SessionService,
        TaskStore,
    ],
)
@pytest.mark.parametrize("owns_backend", [True, False])
async def test_each_store_honors_the_configured_resource_owner(
    native_a2a_backend_factory, store_type, owns_backend
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        kwargs = (
            {"scope": NativeRecipientScope("did:kestrel:recipient")}
            if store_type is TaskStore
            else {}
        )
        store = store_type(backend, owns_backend=owns_backend, **kwargs)
        await store.close()
        assert backend.is_connected is (not owns_backend)
        if not owns_backend:
            assert await backend.fetch_one("SELECT 1") == (1,)
    finally:
        await backend.close()


@pytest.mark.parametrize("value", [None, "false", 0])
def test_backend_ownership_cannot_be_a_truthy_configuration_string(value):
    with pytest.raises(TypeError, match="ownership"):
        ContextStore(None, owns_backend=value)
