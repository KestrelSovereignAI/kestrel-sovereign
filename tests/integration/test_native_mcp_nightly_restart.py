"""Original auto-seeded nightly sleep consumes external rows after restart.

Only the owned schedule's due instant is advanced. No poller, clock, claim,
dispatcher, storage, consolidator or model result is replaced. Generation is
disabled, so these are scheduling/persistence tests, not LLM reflection proof.
"""

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.encoders import jsonable_encoder

from kestrel_sovereign.features.scheduler import runner as scheduler_runner
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import memory_consolidator
from kestrel_sovereign.storage.conversation_created_at import canonical_created_at
from tests.integration.test_native_mcp_memory_restart import (
    native_restart_process,
    persist_external_exchanges,
    restart_request,
    retire_original_publication,
)
from tests.integration.test_native_mcp_publication import (
    assistant_agent as _assistant_agent,
)
from tests.integration.test_native_mcp_publication import publication as _publication

assistant_agent = _assistant_agent
publication = _publication

pytestmark = [
    pytest.mark.asyncio, pytest.mark.integration,
    pytest.mark.parametrize("assistant_agent", [{"features": [
        "MemoryAgencyFeature", "MemoryFeature", "SchedulerFeature",
    ]}], indirect=True),
    pytest.mark.parametrize("publication", [{"clients": [{
        "client_id": "owned-provider", "client_name": "Owned provider",
        "redirect_uris": ["https://provider.example/callback"], "client_secret": "owned-secret",
        "scopes": ["memory:read", "memory:write"],
    }]}], indirect=True),
]


@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL])
async def test_boot_armed_nightly_sleep_consumes_original_native_rows(publication, mode, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    owner, _ = publication
    runner = owner.agent.features["SchedulerFeature"]._runner
    await runner.stop()
    original, receipts, persisted = await persist_external_exchanges(publication)
    db = original._raw_storage.db
    seeds = await db.fetchall(
        "SELECT id, cron_expression, args_json FROM scheduled_tasks WHERE agent_id = ? AND task_name = 'sleep'",
        (original.did,))
    assert len(seeds) == 1, "The original MemoryFeature did not receive exactly one nightly sleep schedule"
    schedule_id, cron, arguments = seeds[0]
    assert cron == "0 4 * * *" and json.loads(arguments) == {"skip_export": True}
    assert not await db.fetchall("SELECT id FROM task_execution_log WHERE task_id = ?", (schedule_id,))
    if mode == PrivacyMode.EPHEMERAL:
        # Conversation stamps and the established privacy purge boundary have
        # whole-second resolution. These NORMAL rows must actually precede
        # the transition second; rows in that second are deliberately purged.
        latest = max(canonical_created_at(row["created_at"]) for row in persisted)
        async with asyncio.timeout(2):
            while canonical_created_at((await db.fetchone("SELECT CURRENT_TIMESTAMP"))[0]) <= latest:
                await asyncio.sleep(0.01)
    await original.set_privacy_mode(mode)
    due = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    await db.execute("UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ? AND agent_id = ?",
        (due, schedule_id, original.did))
    await retire_original_publication(owner)

    results = []
    for number in range(2):
        receipt_path = Path(original.storage_path).parent / f"{mode.value}-nightly-{number}.json"
        assert not receipt_path.exists()
        request = restart_request(original, receipt_path, receipts=receipts, mode=mode)
        request.update(processing="scheduled_sleep", sleep_schedule_id=schedule_id,
            occurrence_at=due, prepare_repeat=number == 0)
        async with native_restart_process() as process:
            stdout, stderr = await process.communicate(json.dumps(request).encode())
            receipt_path.with_suffix(".stdout").write_bytes(stdout)
            receipt_path.with_suffix(".stderr").write_bytes(stderr)
            assert process.returncode == 0, (stderr.decode()[-5000:], stdout.decode()[-1000:])
        result = json.loads(receipt_path.read_text())
        assert result["pid"] == process.pid and result["pid"] != os.getpid()
        assert result["core_source"] == str(Path(memory_consolidator.__file__).resolve())
        assert result["scheduler_source"] == str(Path(scheduler_runner.__file__).resolve())
        assert result["did"] == original.did and result["generation_disabled"]
        assert result["cleanup"]["primary_sqlite_closed"]
        assert {row["id"]: row for row in result["rows"]} == {
            row["id"]: row for row in jsonable_encoder(persisted)}, "Nightly restart changed original native rows"
        history = result["history"]
        assert len(history) == 1 and history[0][2] == due and history[0][3], history
        # The runner acknowledges successful dispatch. The sleep report owns
        # the distinct privacy/consolidation verdict, including a benign skip.
        assert history[0][1] == "success", history
        report = json.loads(history[0][4])
        schedule = result["schedule"]
        assert schedule[:3] == [schedule_id, "sleep", cron] and json.loads(schedule[3]) == json.loads(arguments)
        assert datetime.fromisoformat(schedule[4]) > datetime.fromisoformat(due)
        assert schedule[5:] == [None, None, None], "Nightly occurrence retained its original claim"
        assert len(result["signals"]) == number + 1
        assert all(row[1:] == ["cron.sleep", "ok", None] for row in result["signals"]), (
            "Nightly completion bypassed the original signal journal", result["signals"])
        if mode == PrivacyMode.EPHEMERAL:
            assert report["skipped"] and report["success"] is False
            assert report["error"] == "consolidation_skipped"
            assert report["consolidation"]["episodes_created"] == 0
            assert not result["stored_episode_ids"], "Private scheduled sleep persisted an external derivative"
        else:
            assert report["success"] is True and not report["error"]
            assert report["consolidation"]["episodes_created"] == (1 if number == 0 else 0)
            assert len(result["episodes"]) == 1, "The actual boot-armed nightly poller did not consolidate the native exchange"
            assert set(result["episodes"][0]["key_message_ids"]) == set(map(str, receipts))
        results.append(result)
        due = result.get("repeat_occurrence_at", due)
    assert results[0]["pid"] != results[1]["pid"]
    assert results[0]["history"][0][0] != results[1]["history"][0][0]
    assert results[0]["stored_episode_ids"] == results[1]["stored_episode_ids"], "Nightly retry duplicated an external episode"
