import asyncio
from unittest.mock import patch

import pytest

from kestrel_sovereign.storage.providers.base import StorageResult, StorageTier
from kestrel_sovereign.features.sovereignty.feature import SovereigntyFeature


class _FakeStorage:
    async def create_backup_blob(self, include_db=True):
        return b"x" * 1024

    async def record_backup_artifact(self, agent_id, result):
        return "backup-node-1"

    async def add_node(self, node):
        self.receipt = node


class _FakeAgent:
    agent_id = "agent-progress"

    def __init__(self):
        self.storage = _FakeStorage()
        self.features = {}
        self.events = []

    async def emit_event(self, event_type, data):
        self.events.append((event_type, data))


@pytest.mark.asyncio
async def test_export_sovereignty_reports_upload_progress():
    agent = _FakeAgent()
    feature = SovereigntyFeature(agent)
    progress = []

    def fake_store_content(*args, **kwargs):
        on_progress = kwargs["on_progress"]
        on_progress(256, 1024)
        on_progress(768, 1024)
        return StorageResult(
            content_hash="abc123",
            cid="QmProgress",
            tier=StorageTier.IPFS,
            provider="ipfs",
            encrypted=False,
            size_bytes=1024,
        )

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter.store_content",
        side_effect=fake_store_content,
    ):
        result = await feature.export_sovereignty(
            storage_tier="local",
            encrypt=False,
            on_progress=lambda sent, total: progress.append((sent, total)),
        )

    for _ in range(5):
        await asyncio.sleep(0)

    assert result.data["cid"] == "QmProgress"
    assert (0, 1024) in progress
    assert (256, 1024) in progress
    assert (768, 1024) in progress
    assert progress[-1] == (1024, 1024)

    progress_events = [
        data
        for event_type, data in agent.events
        if event_type == "sovereignty_export_progress"
    ]
    assert progress_events
    assert progress_events[-1]["percent"] == 100
