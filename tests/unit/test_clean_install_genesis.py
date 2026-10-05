"""Release rehearsal may defer test genesis, never synthesize a pass."""

from __future__ import annotations

import pytest

from kestrel_sovereign.constitution.genesis_audit import defer_test_genesis_audit
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore


@pytest.mark.parametrize(
    ("test_instance", "mode", "injected", "expected"),
    [
        (True, "skip", False, True),
        (True, "skip", True, False),
        (False, "skip", False, False),
        (True, "strict", False, False),
    ],
)
def test_defer_predicate_requires_test_marker_and_no_auditor(
    test_instance, mode, injected, expected
):
    auditor = (lambda prompt: prompt) if injected else None
    assert defer_test_genesis_audit(
        test_instance, auditor=auditor, environ={"KESTREL_AUDIT_MODE": mode}
    ) is expected


@pytest.mark.asyncio
async def test_test_instance_pending_genesis_never_calls_embedding_model(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    embedding_requests: list[bool] = []
    original_chunk = AsyncRAGStore.chunk_document

    async def captured_chunk(self, *args, **kwargs):
        embedding_requests.append(kwargs["compute_embeddings"])
        return await original_chunk(self, *args, **kwargs)

    def forbidden_model(self):
        raise AssertionError("clean-install inception called an embedding model")

    monkeypatch.setattr(AsyncRAGStore, "chunk_document", captured_chunk)
    monkeypatch.setattr(AsyncRAGStore, "_get_embedding_service", forbidden_model)
    credentials = await create_kestrel_identity_async(
        str(tmp_path / "agent"),
        agent_name="InstallTest",
        identity_method="did:pkh",
        is_test_instance=True,
    )

    assert embedding_requests and all(value is False for value in embedding_requests)
    async with Storage(credentials.db_path, agent_id=credentials.agent_did) as storage:
        node = await storage.get_node(credentials.agent_did)
        receipt = node.properties["genesis_audit"]
        assert receipt["status"] == "pending"
        assert receipt["audited"] is False
        assert receipt["constitution_hash"] == node.properties["constitution_hash"]
