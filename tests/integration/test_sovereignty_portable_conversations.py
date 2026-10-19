"""Actual encrypted SQLite -> CAR -> distinct destination, no model/cloud I/O."""

import json
from uuid import uuid4

import pytest

from kestrel_sovereign.security.encryption import DecryptionError
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.db import QueryError
from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter


class CapturedArchive(SovereignStorageAdapter):
    """Own only the upload transport; CAR construction/encryption stay real."""

    async def _upload_content(
        self, content, storage_tier, metadata=None, *, store_guard=None
    ):
        assert store_guard is None
        self.car = content
        return "owned-car-transport"


class LegacyEnvelopeArchive(CapturedArchive):
    """Emulate the historical CAR fields with actual source at-rest bytes."""

    async def _get_conversations(self):
        row = await self.db.fetchone(
            "SELECT role,content,metadata,id,rendered_content FROM conversation_history WHERE agent_id=?",
            (self.agent_id,),
        )
        return [
            {
                "role": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]),
                "id": row[3],
                "rendered_content": row[4],
            }
        ]


@pytest.mark.asyncio
async def test_import_success_requires_an_acknowledged_audit_row(monkeypatch, tmp_path):
    monkeypatch.setenv("KESTREL_DATA_KEY", "owned-ignored-audit-key-0123456789abcdef")
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    did = "did:kestrel:owned-ignored-audit"
    async with AsyncStorage(
        backend="sqlite", db_path=str(tmp_path / "ignored-audit.db"), agent_id=did
    ) as storage:
        await storage.add_conversation("user", "owned ignored-audit fixture")
        archive = CapturedArchive(
            storage.db, "owned-ignored-audit-secret", object(), agent_id=did
        )
        await archive.export_agent(did)
        await archive._ensure_import_log()
        await storage.db.execute(
            "CREATE TRIGGER owned_ignore_audit BEFORE INSERT ON agent_import_log BEGIN SELECT RAISE(IGNORE); END"
        )
        with pytest.raises(QueryError, match="not acknowledged"):
            await archive.import_agent(archive.car)
        assert await storage.db.fetchone("SELECT count(*) FROM agent_import_log") == (
            0,
        )


@pytest.mark.asyncio
async def test_encrypted_conversations_are_readable_under_distinct_destination_key(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("KESTREL_DATA_KEY", "owned-portable-source-key-0123456789abcdef")
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    source_did, target_did = (
        "did:kestrel:owned-portable-source",
        "did:kestrel:owned-portable-target",
    )
    session_id = str(uuid4())
    async with AsyncStorage(
        backend="sqlite", db_path=str(tmp_path / "source.db"), agent_id=source_did
    ) as source:
        await source.add_conversation(
            "assistant",
            "owned logical plaintext",
            rendered_content="owned exact rendered transport ⟂",
            model="owned-model-metadata",
            provider="owned-provider-metadata",
            session_id=session_id,
            metadata={"timestamp": "2026-10-19T12:00:00+00:00"},
        )
        raw = await source.db.fetchone(
            "SELECT content,rendered_content,metadata FROM conversation_history WHERE agent_id=?",
            (source_did,),
        )
        assert (
            raw[0].startswith("KSAv2:")
            and raw[1].startswith("KSAv2:")
            and json.loads(raw[2])["enc"]
        )
        exported = CapturedArchive(
            source.db, "owned-portable-archive-secret", object(), agent_id=source_did
        )
        await exported.export_agent(source_did)
        assert b"owned logical plaintext" not in exported.car
        assert b"owned exact rendered transport" not in exported.car
    # The archive must carry logical data inside its own encryption, not need
    # the source host's at-rest key at the receiving storage boundary.
    monkeypatch.setenv(
        "KESTREL_DATA_KEY", "owned-distinct-destination-key-0123456789abcdef"
    )
    async with AsyncStorage(
        backend="sqlite", db_path=str(tmp_path / "target.db"), agent_id=target_did
    ) as target:
        imported = SovereignStorageAdapter(
            target.db, "owned-portable-archive-secret", object(), agent_id=target_did
        )
        result = await imported.import_agent(exported.car)
        assert result.success and result.messages_restored == 1
        history = await target.get_conversation_history(limit=10)
        assert history[0]["content"] == "owned logical plaintext"
        assert history[0]["rendered_content"] == "owned exact rendered transport ⟂"
        stored = await target.db.fetchone(
            "SELECT content,rendered_content,metadata,session_id,created_at,model,provider FROM conversation_history WHERE agent_id=?",
            (target_did,),
        )
        assert stored[0].startswith("KSAv2:") and stored[1].startswith("KSAv2:")
        assert stored[0] != raw[0] and stored[1] != raw[1]
        assert json.loads(stored[2])["enc"] is True
        assert stored[3] == session_id and stored[4] == "2026-10-19 12:00:00"
        assert stored[5:] == ("owned-model-metadata", "owned-provider-metadata")


@pytest.mark.asyncio
async def test_conflicting_host_audit_schema_cannot_be_a_logged_success(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("KESTREL_DATA_KEY", "owned-audit-key-0123456789abcdef")
    async with AsyncStorage(
        backend="sqlite",
        db_path=str(tmp_path / "host.db"),
        agent_id="did:kestrel:owned-audit",
    ) as storage:
        await storage.db.execute(
            "CREATE TABLE agent_import_log(id TEXT PRIMARY KEY,companion_id TEXT NOT NULL,source_system TEXT NOT NULL)"
        )
        adapter = SovereignStorageAdapter(
            storage.db,
            "owned-audit-secret",
            object(),
            agent_id="did:kestrel:owned-audit",
        )
        with pytest.raises(QueryError, match="agent_did"):
            await adapter._log_import_attempt(
                agent_did="did:kestrel:owned-audit",
                source_did="did:kestrel:source",
                package_hash="a" * 64,
                continuity_score=1.0,
                status="imported",
                reject_reason=None,
                messages_restored=1,
                shards_restored=1,
                manifest_version="3",
                source_cid="owned-cid",
            )


@pytest.mark.asyncio
async def test_corrupt_source_at_rest_is_refused_before_upload(monkeypatch, tmp_path):
    monkeypatch.setenv("KESTREL_DATA_KEY", "owned-corrupt-source-key-0123456789abcdef")
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    async with AsyncStorage(
        backend="sqlite",
        db_path=str(tmp_path / "corrupt.db"),
        agent_id="did:kestrel:corrupt-source",
    ) as storage:
        await storage.add_conversation("user", "owned corrupt-source fixture")
        await storage.db.execute(
            "UPDATE conversation_history SET content=? WHERE agent_id=?",
            ("KSAv2:invalid-owned-ciphertext", storage.agent_id),
        )
        adapter = CapturedArchive(
            storage.db, "owned-car-secret", object(), agent_id=storage.agent_id
        )
        with pytest.raises(DecryptionError):
            await adapter.export_agent(storage.agent_id)
        assert not hasattr(adapter, "car")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "same_target,audit_outcome",
    [(False, "accepted"), (True, "accepted"), (False, "refused"), (False, "ignored")],
)
async def test_legacy_envelopes_require_the_actual_destination_key_before_replace(
    monkeypatch, tmp_path, same_target, audit_outcome
):
    monkeypatch.setenv("KESTREL_DATA_KEY", "owned-legacy-key-0123456789abcdef")
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    source_did = "did:kestrel:legacy-source"
    target_did = source_did if same_target else "did:kestrel:distinct-legacy-target"
    async with AsyncStorage(
        backend="sqlite",
        db_path=str(tmp_path / "legacy-source.db"),
        agent_id=source_did,
    ) as source:
        await source.add_conversation("user", "owned legacy archive message")
        archive = LegacyEnvelopeArchive(
            source.db, "owned-legacy-car-secret", object(), agent_id=source_did
        )
        await archive.export_agent(source_did)
    async with AsyncStorage(
        backend="sqlite",
        db_path=str(tmp_path / "legacy-target.db"),
        agent_id=target_did,
    ) as target:
        await target.add_conversation(
            "user", "owned existing target retained on refusal"
        )
        before = await target.db.fetchall(
            "SELECT id,content,metadata FROM conversation_history WHERE agent_id=?",
            (target_did,),
        )
        receiver = SovereignStorageAdapter(
            target.db, "owned-legacy-car-secret", object(), agent_id=target_did
        )
        if same_target:
            result = await receiver.import_agent(archive.car)
            assert result.success
            history = await target.get_conversation_history(limit=10)
            assert [row["content"] for row in history] == [
                "owned legacy archive message"
            ]
        else:
            if audit_outcome != "accepted":
                await receiver._ensure_import_log()
                action = (
                    "SELECT RAISE(IGNORE)"
                    if audit_outcome == "ignored"
                    else "SELECT RAISE(ABORT,'owned audit refused')"
                )
                await target.db.execute(
                    f"CREATE TRIGGER owned_refuse_audit BEFORE INSERT ON agent_import_log BEGIN {action}; END"
                )
            with pytest.raises(DecryptionError) as caught:
                await receiver.import_agent(archive.car)
            if audit_outcome != "accepted":
                assert isinstance(caught.value.__cause__, ExceptionGroup)
                assert caught.value.__cause__.subgroup(QueryError) is not None
            assert (
                await target.db.fetchall(
                    "SELECT id,content,metadata FROM conversation_history WHERE agent_id=?",
                    (target_did,),
                )
                == before
            )
            assert [
                row["content"]
                for row in await target.get_conversation_history(limit=10)
            ] == ["owned existing target retained on refusal"]
