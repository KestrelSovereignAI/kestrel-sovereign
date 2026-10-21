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
    episodes = await consolidator.get_episodes()
    assert episodes[0].timespan_start == now
    assert episodes[0].timespan_end == now
    assert episodes[0].created_at == now
    assert episodes == [MemoryEpisode.from_row(row)]


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


@pytest.mark.parametrize("stored,expected", [
    (None, []), ("", []), ("malformed", []), ('["message"]', ["message"]), (["message"], ["message"]),
])
def test_canonical_episode_decoder_preserves_legacy_list_tolerance(stored, expected):
    row = ("one", "did:test:episode", "title", "summary", None, None, stored, None, None)
    assert MemoryEpisode.from_row(row).key_message_ids == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_archival_write_uses_the_same_postgres_timestamp_boundary(legacy, monkeypatch):
    import json
    from kestrel_sovereign.storage import memory_consolidator as module

    archived = datetime(2026, 10, 20, 17, 15, tzinfo=timezone(timedelta(hours=9)))
    metadata = {"archived": True, "archived_at": archived.isoformat()} if legacy else {"importance": 0.0}
    db = SimpleNamespace(
        backend_type="postgres", table_exists=AsyncMock(return_value=False),
        fetchall=AsyncMock(return_value=[("one", json.dumps(metadata), archived)]),
        execute=AsyncMock(),
    )
    monkeypatch.setattr(module, "calculate_decay", lambda *a, **kw: 0.0)
    assert await MemoryConsolidator(db, "did:test:episode")._archive_decayed() == 1
    stamp = db.execute.await_args.args[1][1]
    assert isinstance(stamp, datetime) and stamp.tzinfo is None
    if legacy:
        assert stamp == archived.astimezone(timezone.utc).replace(tzinfo=None)
