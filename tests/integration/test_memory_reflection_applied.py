from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.memory_system import MemorySystem


AGENT_ID = "did:test:memory-reflection-applied"


class _FakeLLM:
    async def generate(self, *, system_prompt: str, user_prompt: str, **kwargs):
        memory_content = user_prompt.split("Memory content:\n", 1)[1].split(
            "\n\nSession context:", 1
        )[0]
        if "load-bearing preference" in memory_content:
            return json.dumps({
                "applied": True,
                "reason": "It changed the assistant's recommendation.",
            })
        return json.dumps({
            "applied": False,
            "reason": "It was retrieved but not used.",
        })


class _Agent:
    def __init__(self, storage: AsyncStorage, memory_system: MemorySystem):
        self.did = AGENT_ID
        self.agent_id = AGENT_ID
        self._raw_storage = storage
        self.storage = storage
        self.memory_system = memory_system
        self.memory_consolidator = memory_system.consolidator
        self.llm_service = _FakeLLM()


def _metadata(row):
    raw = row["metadata"]
    return raw if isinstance(raw, dict) else json.loads(raw or "{}")


@pytest.mark.asyncio
async def test_pre_sleep_marks_only_llm_attested_retrieved_memories(
    tmp_path,
    caplog,
):
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        memory_system = MemorySystem(storage, AGENT_ID)
        await memory_system.initialize()

        await storage.conversation.add_conversation(
            "assistant",
            "Remember the user's load-bearing preference for concise plans.",
        )
        await storage.conversation.add_conversation(
            "assistant",
            "Decorative context that did not steer this session.",
        )
        history = await storage.conversation.get_full_history_with_ids()
        applied_id = history[0]["id"]
        unused_id = history[1]["id"]

        retrieved_at = datetime.now(timezone.utc).isoformat()
        for msg_id in (applied_id, unused_id):
            await storage.conversation.update_message_metadata(
                msg_id,
                {
                    "importance": 0.5,
                    "access_count": 1,
                    "last_accessed": retrieved_at,
                },
            )

        agent = _Agent(storage, memory_system)
        hook = ReflectionSleepHook()

        caplog.set_level(logging.INFO)
        result = await hook.on_pre_sleep(agent)

        assert result["success"] is True
        assert result["candidates"] == 2
        assert result["applied_count"] == 1
        assert result["attested_message_ids"] == [applied_id]

        history = await storage.conversation.get_full_history_with_ids()
        applied_meta = _metadata(next(row for row in history if row["id"] == applied_id))
        unused_meta = _metadata(next(row for row in history if row["id"] == unused_id))

        assert applied_meta["applied_count"] == 1
        assert applied_meta["last_applied"]
        assert unused_meta.get("applied_count", 0) == 0
        assert "reason=It changed the assistant's recommendation." in caplog.text


@pytest.mark.asyncio
async def test_applied_count_changes_archive_set_on_consolidation(tmp_path):
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        memory_system = MemorySystem(storage, AGENT_ID)
        await memory_system.initialize()

        old_created_at = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        base_metadata = {"importance": 0.5}
        await storage.conversation.add_conversation(
            "assistant",
            "Applied old memory that should survive because it was load-bearing.",
            metadata=base_metadata,
        )
        await storage.conversation.add_conversation(
            "assistant",
            "Equally old unused memory that should archive.",
            metadata=base_metadata,
        )
        history = await storage.conversation.get_full_history_with_ids()
        applied_id = history[0]["id"]
        unused_id = history[1]["id"]

        await storage.db.execute_commit(
            "UPDATE conversation_history SET created_at = ? WHERE id IN (?, ?)",
            (old_created_at, applied_id, unused_id),
        )

        for _ in range(3):
            await memory_system.mark_applied(
                applied_id,
                reason="Integration test load-bearing attestation.",
            )

        result = await memory_system.consolidate()
        assert result["messages_archived"] == 1

        rows = await storage.db.fetchall(
            "SELECT id, metadata FROM conversation_history WHERE id IN (?, ?)",
            (applied_id, unused_id),
        )
        metadata_by_id = {
            row[0]: json.loads(row[1] or "{}")
            for row in rows
        }

        assert metadata_by_id[unused_id]["archived"] is True
        assert "archived" not in metadata_by_id[applied_id]
        assert metadata_by_id[applied_id]["applied_count"] == 3
