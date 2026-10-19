"""Real SQLite/PostgreSQL recovery tests, using Core's current SDK dependency."""

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from types import SimpleNamespace

from cryptography.fernet import Fernet
import pytest
import pytest_asyncio

from kestrel_sdk.security.aead import AEADCipher
from kestrel_sovereign.security.key_rotation import (
    KeyRotationService, RotationRecord, RotationStatus,
    _hash_key, _master_bytes_from_key,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest_asyncio.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.native_postgres)])
async def rotation_store(tmp_path, monkeypatch, request):
    old_key, new_key = (Fernet.generate_key().decode() for _ in range(2))
    monkeypatch.setenv("KESTREL_DATA_KEY", old_key)
    new_key_file = tmp_path / "next-key"
    new_key_file.write_text(new_key)
    new_key_file.chmod(0o600)
    async def no_core_schema(db):
        pass
    async with AsyncExitStack() as cleanup:
        if request.param == "postgres":
            from tests.utils.postgres_schema import postgres_test_url, disposable_postgres_schema, with_search_path
            dsn = postgres_test_url()
            if not dsn:
                pytest.skip("PostgreSQL test URL is not configured")
            admin = await AsyncDatabase.postgres(dsn, min_pool_size=1, max_pool_size=2, schema_initializer=no_core_schema)
            cleanup.push_async_callback(admin.close)
            schema = await cleanup.enter_async_context(disposable_postgres_schema(admin, "rotation_test"))
            db = await AsyncDatabase.postgres(with_search_path(dsn, schema), min_pool_size=1, max_pool_size=2,
                                               schema_initializer=no_core_schema)
        else:
            db = await AsyncDatabase.sqlite(str(tmp_path / "rotation.db"), schema_initializer=no_core_schema)
        cleanup.push_async_callback(db.close)
        service = KeyRotationService(SimpleNamespace(database=db))
        await service.initialize()
        cleanup.push_async_callback(service.shutdown)
        blob_type = "BYTEA" if db.backend_type == "postgres" else "BLOB"
        await db.execute("CREATE TABLE conversation_history (id INTEGER PRIMARY KEY, content TEXT, rendered_content TEXT, agent_id TEXT, metadata TEXT)")
        await db.execute(f"CREATE TABLE files (content_hash TEXT PRIMARY KEY, content {blob_type}, metadata TEXT)")
        await db.execute(f"CREATE TABLE artifact_keys (key_id TEXT PRIMARY KEY, wrapped_key {blob_type} NOT NULL)")
        old_cipher, new_cipher = (AEADCipher(_master_bytes_from_key(key)) for key in (old_key, new_key))
        root = bytes(range(32))
        assert len(root) == 32
        await db.execute("INSERT INTO artifact_keys VALUES (?, ?)", ("v1", old_cipher.encrypt(root)))
        await db.execute("INSERT INTO conversation_history (id, content, metadata) VALUES (?, ?, '{\"enc\": true}')", (1, old_cipher.encrypt(b"conversation").decode()))
        for index in range(8):
            await db.execute("INSERT INTO files (content_hash, content, metadata) VALUES (?, ?, ?)", (str(index), old_cipher.encrypt(b"file"), '{"enc": true}'))
        await db.execute("INSERT INTO files (content_hash, content, metadata) VALUES (?, ?, ?)", ("plain", b"legacy plaintext", None))
        yield SimpleNamespace(db=db, service=service, old_key=old_key, new_key=new_key,
                              old_cipher=old_cipher, new_cipher=new_cipher, root=root, key_file=new_key_file)


async def reject_writes(db, table, failure):
    assert table in {"artifact_keys", "files", "conversation_history"}
    if db.backend_type == "sqlite":
        action = "SELECT RAISE(ABORT, 'update denied');" if failure == "raise" else "SELECT RAISE(IGNORE);"
        await db.execute(f"CREATE TRIGGER rotation_unavailable BEFORE UPDATE ON {table} BEGIN {action} END")
    else:
        action = "RAISE EXCEPTION 'update denied';" if failure == "raise" else "RETURN NULL;"
        await db.execute(f"CREATE OR REPLACE FUNCTION rotation_unavailable() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN {action} END $$")
        await db.execute(f"CREATE TRIGGER rotation_unavailable BEFORE UPDATE ON {table} FOR EACH ROW EXECUTE FUNCTION rotation_unavailable()")


async def assert_rotated(store, *, has_root=True):
    rows = await store.db.fetchall("SELECT content FROM files WHERE content_hash IN ('0', '1', '2', '3', '4', '5', '6', '7')")
    assert len(rows) == 8 and all(store.new_cipher.decrypt(bytes(row[0])) == b"file" for row in rows)
    text = (await store.db.fetchone("SELECT content FROM conversation_history WHERE id = 1"))[0]
    assert store.new_cipher.decrypt(text.encode()) == b"conversation"
    assert (await store.db.fetchone("SELECT content FROM files WHERE content_hash = 'plain'"))[0] == b"legacy plaintext"
    if has_root:
        wrapped = (await store.db.fetchone("SELECT wrapped_key FROM artifact_keys WHERE key_id = 'v1'"))[0]
        assert store.new_cipher.decrypt(bytes(wrapped)) == store.root


@pytest.mark.asyncio
@pytest.mark.parametrize("table", ["artifact_keys", "files"])
@pytest.mark.parametrize("failure", ["raise", "ignored"])
async def test_partial_failure_resumes_original_journal(rotation_store, monkeypatch, tmp_path, table, failure):
    store = rotation_store
    await reject_writes(store.db, table, failure)
    rotation_id = await store.service.start_rotation(new_key=store.new_key)
    await store.service.drain_rotations()
    failed = await store.service.get_status(rotation_id)
    assert failed.status == RotationStatus.FAILED
    assert failed.records_processed == (9 if table == "artifact_keys" else 1)
    # Atomic journal entries, not a possibly lagging checkpoint count, block a
    # replacement rotation and restore the cumulative counter on resume.
    await store.db.execute("UPDATE key_rotations SET records_processed = 0 WHERE id = ?", (rotation_id,))
    with pytest.raises(RuntimeError, match="Resume"):
        await store.service.start_rotation(new_key=store.new_key)
    if store.db.backend_type == "sqlite":
        await store.db.execute("DROP TRIGGER rotation_unavailable")
    else:
        await store.db.execute(f"DROP TRIGGER rotation_unavailable ON {table}")
    with monkeypatch.context() as patch:
        patch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
        with pytest.raises(ValueError, match="old key"):
            await store.service.resume_rotation(rotation_id, str(store.key_file))
    wrong_key = tmp_path / "wrong-key"
    wrong_key.write_text(Fernet.generate_key().decode())
    with pytest.raises(ValueError, match="new key"):
        await store.service.resume_rotation(rotation_id, str(wrong_key))
    execution_allowed = asyncio.Event()
    execute = store.service._execute_rotation
    async def held_execution(*args, **kwargs):
        await execution_allowed.wait()
        await execute(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(store.service, "_execute_rotation", held_execution)
        results = await asyncio.gather(store.service.resume_rotation(rotation_id, str(store.key_file)),
                                       store.service.resume_rotation(rotation_id, str(store.key_file)), return_exceptions=True)
        assert sum(result is True for result in results) == 1
        assert sum(isinstance(result, (ValueError, RuntimeError)) for result in results) == 1
        execution_allowed.set()
        await store.service.drain_rotations()
    completed = await store.service.get_status(rotation_id)
    assert completed.status == RotationStatus.COMPLETED
    assert completed.records_processed == completed.records_total == 10
    assert completed.error_message is None
    await assert_rotated(store)
    with pytest.raises(ValueError, match="not resumable"):
        await store.service.resume_rotation(rotation_id, str(store.key_file))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "malformed", "mixed", "resumed"])
async def test_mandatory_envelope_rows_cannot_be_skipped(rotation_store, failure):
    store = rotation_store
    if failure == "missing":
        await store.db.execute("DELETE FROM artifact_keys")
    elif failure == "mixed":
        await store.db.execute("INSERT INTO artifact_keys VALUES (?, ?)", ("damaged", b"malformed"))
    else:
        await store.db.execute("UPDATE artifact_keys SET wrapped_key = ?", (b"malformed",))
    rotation = RotationRecord(id="mandatory-root", started_at=datetime.now(UTC),
                              old_key_hash=_hash_key(store.old_key), new_key_hash=_hash_key(store.new_key),
                              status=RotationStatus.IN_PROGRESS)
    if failure == "resumed":
        await store.db.execute("INSERT INTO rotation_progress VALUES (?, ?, ?, ?)",
                               (rotation.id, "artifact_keys", "v1", datetime.now(UTC).isoformat()))
    await store.service._execute_rotation(rotation, _master_bytes_from_key(store.old_key), _master_bytes_from_key(store.new_key))
    assert rotation.status == RotationStatus.FAILED
    assert rotation.error_message


@pytest.mark.asyncio
async def test_legacy_database_without_artifact_table_still_rotates(rotation_store):
    store = rotation_store
    await store.db.execute("DROP TABLE artifact_keys")
    rotation_id = await store.service.start_rotation(new_key=store.new_key)
    await store.service.drain_rotations()
    completed = await store.service.get_status(rotation_id)
    assert completed.status == RotationStatus.COMPLETED
    assert completed.records_processed == completed.records_total == 9
    await assert_rotated(store, has_root=False)


@pytest.mark.asyncio
async def test_valid_journal_can_be_resumed_after_an_interruption(rotation_store):
    store = rotation_store
    rotation_id = await store.service.start_rotation(new_key=store.new_key)
    await store.service.drain_rotations()
    await store.db.execute("UPDATE key_rotations SET status = ?, records_processed = 0 WHERE id = ?", ("in_progress", rotation_id))
    assert await store.service.resume_rotation(rotation_id, str(store.key_file))
    await store.service.drain_rotations()
    completed = await store.service.get_status(rotation_id)
    assert completed.status == RotationStatus.COMPLETED
    assert completed.records_processed == 10
    await assert_rotated(store)


async def allow_writes(db, table):
    if db.backend_type == "sqlite":
        await db.execute("DROP TRIGGER rotation_unavailable")
    else:
        await db.execute(f"DROP TRIGGER rotation_unavailable ON {table}")


async def zero_progress_journal(store, tmp_path):
    key = Fernet.generate_key().decode()
    key_file = tmp_path / "earlier-key"
    key_file.write_text(key)
    key_file.chmod(0o600)
    await reject_writes(store.db, "conversation_history", "raise")
    rotation_id = await store.service.start_rotation(new_key=key)
    await store.service.drain_rotations()
    failed = await store.service.get_status(rotation_id)
    assert failed.status == RotationStatus.FAILED and failed.records_processed == 0
    await allow_writes(store.db, "conversation_history")
    return rotation_id, key_file


@pytest.mark.asyncio
async def test_stale_zero_progress_journal_cannot_override_partial_rotation(rotation_store, tmp_path):
    store = rotation_store
    stale_id, stale_key_file = await zero_progress_journal(store, tmp_path)
    await reject_writes(store.db, "artifact_keys", "raise")
    current_id = await store.service.start_rotation(new_key=store.new_key)
    await store.service.drain_rotations()
    failed = await store.service.get_status(current_id)
    assert failed.status == RotationStatus.FAILED and failed.records_processed == 9
    await allow_writes(store.db, "artifact_keys")
    with pytest.raises(RuntimeError, match="unresolved"):
        await store.service.resume_rotation(stale_id, str(stale_key_file))
    assert await store.service.resume_rotation(current_id, str(store.key_file))
    await store.service.drain_rotations()
    assert (await store.service.get_status(current_id)).status == RotationStatus.COMPLETED
    assert (await store.service.get_status(stale_id)).records_processed == 0
    await assert_rotated(store)


@pytest.mark.asyncio
async def test_running_owner_blocks_every_start_and_resume(rotation_store, tmp_path):
    store = rotation_store
    stale_id, stale_key_file = await zero_progress_journal(store, tmp_path)
    # Native retirement can remain owned after the terminal status was saved.
    # Even a task whose ID differs must block all new work until it joins.
    retirement = asyncio.Event()
    store.service._track_rotation_task(retirement.wait(), "different-retiring-owner")
    with pytest.raises(RuntimeError, match="running"):
        await store.service.start_rotation(new_key=store.new_key)
    with pytest.raises(RuntimeError, match="running"):
        await store.service.resume_rotation(stale_id, str(stale_key_file))
    retirement.set()
    await store.service.drain_rotations()
    assert await store.service.resume_rotation(stale_id, str(stale_key_file))
    await store.service.drain_rotations()
    assert (await store.service.get_status(stale_id)).status == RotationStatus.COMPLETED
    cipher = AEADCipher(_master_bytes_from_key(stale_key_file.read_text()))
    assert cipher.decrypt(bytes((await store.db.fetchone("SELECT wrapped_key FROM artifact_keys"))[0])) == store.root


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["KSAv2:", "gAAAAA"])
@pytest.mark.parametrize("metadata", [None, '{"enc": false}'])
async def test_encryption_metadata_preserves_prefixed_plaintext(rotation_store, prefix, metadata):
    store = rotation_store
    plaintext = prefix + " is a literal in this document"
    await store.db.execute(
        "INSERT INTO conversation_history (id, content, rendered_content, metadata) VALUES (?, ?, ?, ?)",
        (2, plaintext, plaintext, metadata))
    await store.db.execute("INSERT INTO files VALUES (?, ?, ?)", ("prefix-plain", plaintext.encode(), metadata))
    rotation_id = await store.service.start_rotation(new_key=store.new_key)
    await store.service.drain_rotations()
    completed = await store.service.get_status(rotation_id)
    assert completed.status == RotationStatus.COMPLETED
    assert completed.records_total == completed.records_processed == 10
    assert tuple(await store.db.fetchone("SELECT content, rendered_content FROM conversation_history WHERE id = 2")) == (plaintext, plaintext)
    assert bytes((await store.db.fetchone("SELECT content FROM files WHERE content_hash = 'prefix-plain'"))[0]) == plaintext.encode()
    await assert_rotated(store)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["content", "rendered_content", "files"])
@pytest.mark.parametrize("damaged", ["no-ciphertext-marker", "KSAv2:broken", "gAAAAAbroken"])
async def test_flagged_encrypted_rows_cannot_hide_behind_prefix_filter(rotation_store, target, damaged):
    store = rotation_store
    if target == "files":
        await store.db.execute("UPDATE files SET content = ? WHERE content_hash = '0'", (damaged.encode(),))
    else:
        assert target in {"content", "rendered_content"}
        await store.db.execute(f"UPDATE conversation_history SET {target} = ? WHERE id = 1", (damaged,))
    rotation_id = await store.service.start_rotation(new_key=store.new_key)
    await store.service.drain_rotations()
    failed = await store.service.get_status(rotation_id)
    assert failed.status == RotationStatus.FAILED
    assert failed.error_message


@pytest.mark.asyncio
async def test_metadata_selection_rotates_per_agent_content_and_rendered_bytes(rotation_store):
    from kestrel_sovereign.security.key_rotation import _derive_agent_cipher
    store = rotation_store
    old = _derive_agent_cipher(_master_bytes_from_key(store.old_key), "alice")
    new = _derive_agent_cipher(_master_bytes_from_key(store.new_key), "alice")
    await store.db.execute(
        "INSERT INTO conversation_history (id, content, rendered_content, agent_id, metadata) VALUES (?, ?, ?, ?, ?)",
        (2, old.encrypt(b"raw").decode(), old.encrypt(b"rendered").decode(), "alice", '{"enc": true, "key_version": 1}'))
    rotation_id = await store.service.start_rotation(new_key=store.new_key)
    await store.service.drain_rotations()
    completed = await store.service.get_status(rotation_id)
    assert completed.status == RotationStatus.COMPLETED
    assert completed.records_total == completed.records_processed == 12
    row = await store.db.fetchone("SELECT content, rendered_content FROM conversation_history WHERE id = 2")
    assert new.decrypt(row[0].encode()) == b"raw"
    assert new.decrypt(row[1].encode()) == b"rendered"
    await assert_rotated(store)


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", ["{", "[1]"])
async def test_invalid_encryption_metadata_refuses_before_writes(rotation_store, metadata):
    store = rotation_store
    await store.db.execute("UPDATE conversation_history SET metadata = ? WHERE id = 1", (metadata,))
    with pytest.raises(RuntimeError, match="Invalid encryption metadata"):
        await store.service.start_rotation(new_key=store.new_key)
    assert (await store.db.fetchone("SELECT COUNT(*) FROM key_rotations"))[0] == 0
    content = (await store.db.fetchone("SELECT content FROM conversation_history WHERE id = 1"))[0]
    assert store.old_cipher.decrypt(content.encode()) == b"conversation"
    root = (await store.db.fetchone("SELECT wrapped_key FROM artifact_keys"))[0]
    assert store.old_cipher.decrypt(bytes(root)) == store.root


@pytest.mark.asyncio
async def test_same_key_upgrade_checks_damaged_aead_instead_of_trusting_prefix(rotation_store):
    store = rotation_store
    await store.db.execute("UPDATE conversation_history SET content = ? WHERE id = 1", ("KSAv2:broken",))
    rotation_id = await store.service.upgrade_to_aead()
    await store.service.drain_rotations()
    assert (await store.service.get_status(rotation_id)).status == RotationStatus.FAILED
