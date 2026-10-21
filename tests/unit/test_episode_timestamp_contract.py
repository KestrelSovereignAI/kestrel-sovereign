"""Backend episode timestamp bindings and canonical native row materialization."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator
from kestrel_sovereign.storage.memory_models import MemoryEpisode


@pytest.mark.asyncio
async def test_episode_reader_uses_the_existing_native_datetime_decoder():
    now = datetime(2026, 10, 20, 17, 15)
    row = ("one", "did:test:episode", "title", "summary", now, now, '["message"]', "steady", now, 0.5, 0)
    db = SimpleNamespace(backend_type="postgres", fetchall=AsyncMock(return_value=[row]))
    consolidator = MemoryConsolidator(db, "did:test:episode")
    assert await consolidator.get_episodes() == [MemoryEpisode.from_row(row)]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_type", ["sqlite", "postgres"])
async def test_episode_write_uses_backend_timestamp_contract_without_provider(backend_type):
    db = SimpleNamespace(backend_type=backend_type, execute=AsyncMock())
    start = datetime(2026, 10, 20, 17, 15, 44, 971730, tzinfo=timezone(timedelta(hours=9)))
    episode = MemoryEpisode("one", "did:test:episode", "title", "summary", timespan_start=start)
    consolidator = MemoryConsolidator(db, episode.agent_id)
    await consolidator._save_episode(episode)
    params = db.execute.await_args.args[1]
    assert params[4] == (start.isoformat() if backend_type == "sqlite" else start.astimezone(timezone.utc).replace(tzinfo=None))
    assert params[5] is None
    assert isinstance(params[8], str if backend_type == "sqlite" else datetime)
