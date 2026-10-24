"""Real SQLite publication, ownership and durability checks for saved outputs."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime, timedelta

from cryptography.fernet import Fernet
import pytest
import pytest_asyncio

from kestrel_sdk.artifacts import ArtifactCommit, ArtifactLink, ArtifactManifest
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceScope, ResourceOutcomeUnknown
from kestrel_sdk.storage.database.interface import TransactionError
from kestrel_sovereign.artifacts.repository import ArtifactIntegrityError, ArtifactLimits, ArtifactQuotaExceeded, ArtifactRepository
from kestrel_sovereign.storage.async_database import AsyncDatabase


SCOPE = ResourceScope(tenant_id="tenant", principal_id="person", agent_id="agent")


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def repository(tmp_path, monkeypatch, request):
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    path = tmp_path / "artifacts.db"
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
            schema = await cleanup.enter_async_context(disposable_postgres_schema(admin, "artifact_test"))
            async def reopen():
                return await AsyncDatabase.postgres(with_search_path(dsn, schema), min_pool_size=1, max_pool_size=2,
                                                     schema_initializer=no_core_schema)
        else:
            async def reopen():
                return await AsyncDatabase.sqlite(str(path), schema_initializer=no_core_schema)
        db = await reopen()
        cleanup.push_async_callback(db.close)
        store = ArtifactRepository(db, storage_id="local")
        await store.initialize()
        yield store, reopen



async def publish(store, *, scope=SCOPE, key="publish", content=b"private report", artifact=None, title="Private title"):
    file = await store.stage(scope, content, publication_key=key, path="report.md", media_type="text/markdown")
    request = ArtifactCommit(idempotency_key=key, title=title,
                             artifact_id=artifact.artifact_id if artifact else None,
                             expected_revision_id=artifact.revision_id if artifact else None,
                             manifest=ArtifactManifest(files=(file,), entrypoint=file.path))
    return await store.commit(scope, request), request


@pytest.mark.asyncio
async def test_publish_survives_restart_without_source_session(repository):
    store, reopen = repository
    receipt, request = await publish(store)
    async def no_core_schema(db):
        pass
    restarted_db = await reopen()
    restarted = ArtifactRepository(restarted_db, storage_id="local")
    try:
        await restarted.initialize()
        assert await restarted.publication_status(SCOPE, "publish") == receipt
        assert await restarted.commit(SCOPE, request) == receipt
        assert await restarted.read(SCOPE, receipt.ref, "report.md") == b"private report"
        revision = await restarted.get(SCOPE, receipt.ref)
        assert revision.provenance.session_id is None
        row = await restarted_db.fetchone("SELECT payload FROM artifact_revisions")
        blob = await restarted_db.fetchone("SELECT content FROM artifact_blobs")
        assert b"Private title" not in row[0]
        assert b"private report" not in blob[0]
    finally:
        await restarted_db.close()


@pytest.mark.asyncio
async def test_publication_failure_rolls_back_head_revision_and_receipt(repository):
    store, _ = repository
    file = await store.stage(SCOPE, b"hello", publication_key="fail", path="a.md", media_type="text/markdown")
    if store.db.backend_type == "sqlite":
        await store.db.execute("CREATE TRIGGER fail_publication BEFORE INSERT ON artifact_publications BEGIN SELECT RAISE(ABORT, 'publication failed'); END")
    else:
        await store.db.execute("CREATE FUNCTION fail_publication() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'publication failed'; END $$")
        await store.db.execute("CREATE TRIGGER fail_publication BEFORE INSERT ON artifact_publications FOR EACH ROW EXECUTE FUNCTION fail_publication()")
    request = ArtifactCommit(idempotency_key="fail", title="Report", manifest=ArtifactManifest(files=(file,), entrypoint="a.md"))
    with pytest.raises(TransactionError, match="publication failed"):
        await store.commit(SCOPE, request)
    for table in ("artifact_heads", "artifact_revisions", "artifact_publications", "artifact_blob_refs"):
        assert (await store.db.fetchone(f"SELECT COUNT(*) FROM {table}"))[0] == 0
    assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_blobs"))[0] == 1


@pytest.mark.asyncio
async def test_uncommitted_caller_transaction_cannot_receive_saved_receipt(repository):
    store, _ = repository
    async with store.db.transaction():
        with pytest.raises(ResourceConflict, match="own durable transaction"):
            await publish(store)


@pytest.mark.asyncio
async def test_owner_scope_does_not_leak_hashes_or_revisions(repository):
    store, _ = repository
    receipt, request = await publish(store)
    for other in (ResourceScope(tenant_id="other", principal_id="person", agent_id="agent"),
                  ResourceScope(tenant_id="tenant", principal_id="other", agent_id="agent"),
                  ResourceScope(tenant_id="tenant", principal_id="person")):
        with pytest.raises(ResourceNotFound):
            await store.get(other, receipt.ref)
        with pytest.raises(ResourceNotFound):
            await store.read(other, receipt.ref, "report.md")
        with pytest.raises(ResourceNotFound):
            await store.commit(other, request)
        assert await store.publication_status(other, "publish") is None
        assert (await store.page(other)).items == ()
    personal = ResourceScope(tenant_id="tenant", principal_id="person")
    own, _ = await publish(store, scope=personal)
    assert own.scope.agent_id is None and own.ref != receipt.ref


@pytest.mark.asyncio
async def test_updates_and_deletes_compare_current_head(repository):
    store, _ = repository
    first, original = await publish(store)
    second, _ = await publish(store, key="edit", content=b"new", artifact=first.ref)
    assert second.ref.artifact_id == first.ref.artifact_id
    assert (await store.get(SCOPE, second.ref)).parent_revision_id == first.ref.revision_id
    assert await store.read(SCOPE, first.ref, "report.md") == b"private report"
    with pytest.raises(ResourceConflict, match="head changed"):
        await publish(store, key="stale", content=b"conflicting edit", artifact=first.ref)
    with pytest.raises(ResourceConflict, match="head changed"):
        await store.delete(SCOPE, first.ref)
    with pytest.raises(ResourceConflict, match="another request"):
        await store.commit(SCOPE, original.model_copy(update={"title": "different"}))
    await store.delete(SCOPE, second.ref)
    with pytest.raises(ResourceNotFound):
        await store.commit(SCOPE, original)
    with pytest.raises(ResourceNotFound):
        await store.read(SCOPE, first.ref, "report.md")


@pytest.mark.asyncio
async def test_simultaneous_edits_have_exactly_one_winner(repository):
    store, reopen = repository
    first, _ = await publish(store)
    async def no_core_schema(db):
        pass
    db2 = await reopen()
    peer = ArtifactRepository(db2, storage_id="local")
    try:
        await peer.initialize()
        results = await asyncio.gather(publish(store, key="edit-one", artifact=first.ref),
                                       publish(peer, key="edit-two", artifact=first.ref), return_exceptions=True)
        assert sum(isinstance(item, ResourceConflict) for item in results) == 1
        assert sum(isinstance(item, tuple) for item in results) == 1
        assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_revisions"))[0] == 2
    finally:
        await db2.close()


@pytest.mark.asyncio
async def test_unlink_keeps_content_and_delete_keeps_other_artifact_refs(repository):
    store, _ = repository
    one, _ = await publish(store)
    two, _ = await publish(store, key="second")
    link = ArtifactLink(ref=one.ref, context_kind="run", context_id="run")
    await store.link(SCOPE, link)
    await store.link(SCOPE, link, remove=True)
    assert await store.read(SCOPE, one.ref, "report.md") == b"private report"
    await store.delete(SCOPE, one.ref)
    store.clock = lambda: datetime.now(UTC) + timedelta(days=1)
    await store.collect_staging(SCOPE)
    assert await store.read(SCOPE, two.ref, "report.md") == b"private report"
    await store.delete(SCOPE, two.ref)
    assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_blobs"))[0] == 0


@pytest.mark.asyncio
async def test_quota_counts_deduplicated_bytes_and_expired_staging_is_collectible(repository):
    store, _ = repository
    store.limits = ArtifactLimits(max_file_bytes=5, max_owner_bytes=5, max_revisions=1, staging_seconds=1)
    instant = datetime.now(UTC)
    store.clock = lambda: instant
    await store.stage(SCOPE, b"hello", publication_key="physical", path="a", media_type="text/plain")
    await store.stage(SCOPE, b"hello", publication_key="pending-b", path="b", media_type="text/plain")
    with pytest.raises(ArtifactQuotaExceeded):
        await store.stage(SCOPE, b"!", publication_key="pending-c", path="c", media_type="text/plain")
    store.clock = lambda: instant + timedelta(seconds=2)
    assert await store.collect_staging(SCOPE) == 1
    first, _ = await publish(store, content=b"hi")
    with pytest.raises(ArtifactQuotaExceeded, match="revision"):
        await publish(store, key="edit", content=b"hi", artifact=first.ref)
    store.clock = lambda: instant + timedelta(days=1)
    assert await store.collect_staging(SCOPE) == 0
    assert await store.read(SCOPE, first.ref, "report.md") == b"hi"


@pytest.mark.asyncio
async def test_corrupted_blob_never_publishes_or_reads_as_valid(repository):
    store, _ = repository
    receipt, request = await publish(store)
    await store.stage(SCOPE, b"private report", publication_key="new", path="report.md", media_type="text/markdown")
    await store.db.execute("UPDATE artifact_blobs SET size_bytes = size_bytes + 1")
    with pytest.raises(ArtifactIntegrityError):
        await store.read(SCOPE, receipt.ref, "report.md")
    with pytest.raises(ArtifactIntegrityError):
        await store.commit(SCOPE, request.model_copy(update={"idempotency_key": "new"}))
    with pytest.raises(ArtifactIntegrityError):
        await store.stage(SCOPE, b"private report", publication_key="another", path="a", media_type="text/plain")


@pytest.mark.asyncio
async def test_plaintext_storage_is_never_implicit(repository, monkeypatch):
    store, _ = repository
    monkeypatch.delenv("KESTREL_DATA_KEY")
    monkeypatch.delenv("KESTREL_DATA_KEY_FILE", raising=False)
    uninitialized = ArtifactRepository(store.db, storage_id="unconfigured")
    with pytest.raises(ArtifactIntegrityError, match="requires encryption"):
        await uninitialized.initialize()
    assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_blobs"))[0] == 0


@pytest.mark.asyncio
async def test_pagination_is_bounded_and_returns_pinned_revisions(repository):
    store, _ = repository
    receipts = [(await publish(store, key=f"item{i}"))[0] for i in range(3)]
    page1 = await store.page(SCOPE, limit=2)
    page2 = await store.page(SCOPE, limit=2, cursor=page1.next_cursor)
    assert len(page1.items) == 2 and len(page2.items) == 1
    assert {item.ref for item in page1.items + page2.items} == {item.ref for item in receipts}
    assert page2.next_cursor is None
    for limit in (0, 101, True):
        with pytest.raises(ValueError):
            await store.page(SCOPE, limit=limit)


@pytest.mark.asyncio
async def test_deletion_recovers_publication_quota_and_erases_consumed_bytes(repository):
    store, _ = repository
    store.limits = ArtifactLimits(max_publications=1)
    for index in range(3):
        receipt, _ = await publish(store, key=f"deleted{index}")
        await store.delete(SCOPE, receipt.ref)
        assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_blobs"))[0] == 0
    receipt, _ = await publish(store, key="still-usable")
    assert await store.read(SCOPE, receipt.ref, "report.md") == b"private report"


@pytest.mark.asyncio
async def test_sensitive_lookup_metadata_is_keyed_and_owner_identity_is_explicit(repository, monkeypatch):
    import hashlib
    import json
    store, _ = repository
    receipt, request = await publish(store)
    owner = store._owner(SCOPE)
    assert owner != hashlib.sha256(SCOPE.model_dump_json().encode()).hexdigest()
    row = await store.db.fetchone("SELECT owner, blob_key FROM artifact_blobs")
    assert row[1] != request.manifest.files[0].sha256
    # Knowing the index format, owner address and guessed content must not
    # expose a dictionary oracle without the encrypted envelope root.
    guess = json.dumps(["kestrel-artifacts-index-v1", "blob", owner, request.manifest.files[0].sha256],
                       ensure_ascii=True, separators=(",", ":")).encode()
    assert row[1] != hashlib.sha256(guess).hexdigest()
    publication = await store.db.fetchone("SELECT publication_key FROM artifact_publications")
    assert publication[0] != request.idempotency_key
    # Serialization changes and future non-identity metadata cannot re-key owners.
    monkeypatch.setattr(ResourceScope, "model_dump_json", lambda *args, **kwargs: "changed serialization")
    assert store._owner(SCOPE) == owner


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["committed", "unknown"])
@pytest.mark.parametrize("carrier", ["transaction", "retirement"])
async def test_production_commit_outcome_signal_reconciles_to_saved_revision(repository, monkeypatch, outcome, carrier):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError, execution_commit_outcome
    store, _ = repository
    file = await store.stage(SCOPE, b"hello", publication_key="physical", path="a", media_type="text/plain")
    request = ArtifactCommit(idempotency_key="physical", title="Report", manifest=ArtifactManifest(files=(file,), entrypoint="a"))
    original = store.db.transaction
    @asynccontextmanager
    async def lost_ack(**options):
        async with original(**options):
            yield
        if carrier == "retirement":
            from kestrel_sovereign.execution_custody import execution_error_with_cause
            raise execution_error_with_cause(RuntimeError("native retirement refused"), ExecutionCommitOutcomeError(outcome))
        raise ExecutionCommitOutcomeError(outcome)
    with monkeypatch.context() as patch:
        patch.setattr(store.db, "transaction", lost_ack)
        with pytest.raises(ResourceOutcomeUnknown) as raised:
            await store.commit(SCOPE, request)
        assert execution_commit_outcome(raised.value) == outcome
    receipt = await store.publication_status(SCOPE, "physical")
    assert receipt is not None
    assert await store.commit(SCOPE, request) == receipt


@pytest.mark.asyncio
async def test_pending_staging_hold_survives_unrelated_delete_then_host_sweep_reclaims_it(repository, monkeypatch):
    store, _ = repository
    instant = datetime.now(UTC)
    store.clock = lambda: instant
    await store.stage(SCOPE, b"private report", publication_key="pending", path="pending.md", media_type="text/markdown")
    receipt, _ = await publish(store)
    await store.delete(SCOPE, receipt.ref)
    assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_blobs"))[0] == 1
    assert await store.collect_expired_staging() == 0
    store.clock = lambda: instant + timedelta(days=1)
    # Maintenance needs only opaque owner addresses, not a returning owner/key.
    monkeypatch.delenv("KESTREL_DATA_KEY")
    assert await store.collect_expired_staging() == 1
    assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_blobs"))[0] == 0


@pytest.mark.asyncio
async def test_blob_ciphertext_metadata_and_context_links_have_separate_bounds(repository):
    store, _ = repository
    store.limits = ArtifactLimits(max_blobs=1)
    await store.stage(SCOPE, b"a", publication_key="pending-a", path="a", media_type="text/plain")
    with pytest.raises(ArtifactQuotaExceeded, match="blob storage"):
        await store.stage(SCOPE, b"b", publication_key="pending-b", path="b", media_type="text/plain")
    size = (await store.db.fetchone("SELECT length(content) FROM artifact_blobs"))[0]
    store.limits = ArtifactLimits(max_stored_bytes=size)
    with pytest.raises(ArtifactQuotaExceeded, match="blob storage"):
        await store.stage(SCOPE, b"b", publication_key="pending-b", path="b", media_type="text/plain")
    store.limits = ArtifactLimits(max_metadata_bytes=1)
    with pytest.raises(ArtifactQuotaExceeded, match="metadata"):
        await publish(store)
    assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_heads"))[0] == 0
    store.limits = ArtifactLimits(max_links=1)
    receipt, _ = await publish(store)
    link = ArtifactLink(ref=receipt.ref, context_kind="run", context_id="first")
    await store.link(SCOPE, link)
    await store.link(SCOPE, link)  # Idempotent even at the quota.
    with pytest.raises(ArtifactQuotaExceeded, match="context link"):
        await store.link(SCOPE, link.model_copy(update={"context_id": "second"}))
    assert (await store.db.fetchone("SELECT context_id FROM artifact_links"))[0] != "first"
    await store.link(SCOPE, link, remove=True)
    await store.link(SCOPE, link.model_copy(update={"context_id": "second"}))


@pytest.mark.asyncio
async def test_history_is_chronological_and_reads_do_not_mutate_owner_rows(repository):
    store, _ = repository
    instant = datetime.now(UTC)
    store.clock = lambda: instant
    first, _ = await publish(store)
    store.clock = lambda: instant + timedelta(seconds=1)
    second, _ = await publish(store, key="second", artifact=first.ref)
    store.clock = lambda: instant + timedelta(seconds=2)
    third, _ = await publish(store, key="third", artifact=second.ref)
    before = await store.db.fetchone("SELECT generation FROM artifact_owners")
    page = await store.page(SCOPE, artifact_id=first.ref.artifact_id, limit=2)
    assert [item.ref for item in page.items] == [third.ref, second.ref]
    last = await store.page(SCOPE, artifact_id=first.ref.artifact_id, cursor=page.next_cursor, limit=2)
    assert [item.ref for item in last.items] == [first.ref]
    assert last.next_cursor is None
    await store.get(SCOPE, first.ref)
    await store.read(SCOPE, first.ref, "report.md")
    await store.publication_status(SCOPE, "publish")
    assert (await store.db.fetchone("SELECT generation FROM artifact_owners")) == before
    for cursor in ("garbage", page.next_cursor):
        with pytest.raises(ValueError, match="cursor"):
            await store.page(SCOPE, cursor=cursor)


@pytest.mark.asyncio
async def test_master_key_rotation_preserves_owner_addresses_and_all_saved_content(repository, monkeypatch):
    import os
    from types import SimpleNamespace
    from kestrel_sovereign.security.key_rotation import KeyRotationService, RotationRecord, RotationStatus, _master_bytes_from_key
    store, reopen = repository
    receipt, request = await publish(store)
    old_owner = store._owner(SCOPE)
    old_blob = (await store.db.fetchone("SELECT content FROM artifact_blobs"))[0]
    old_key = os.environ["KESTREL_DATA_KEY"]
    new_key = Fernet.generate_key().decode()
    rotation_service = KeyRotationService(SimpleNamespace(database=store.db))
    await rotation_service.initialize()
    # Only these two empty legacy tables are needed by the complete registry.
    await store.db.execute("CREATE TABLE conversation_history (id INTEGER PRIMARY KEY, content TEXT, rendered_content TEXT, agent_id TEXT, metadata TEXT)")
    blob_type = "BYTEA" if store.db.backend_type == "postgres" else "BLOB"
    await store.db.execute(f"CREATE TABLE files (content_hash TEXT PRIMARY KEY, content {blob_type}, metadata TEXT)")
    rotation = RotationRecord(id="artifact-rotation", started_at=datetime.now(UTC),
                              old_key_hash="old", new_key_hash="new", status=RotationStatus.IN_PROGRESS)
    await rotation_service._execute_rotation(rotation, _master_bytes_from_key(old_key), _master_bytes_from_key(new_key))
    assert rotation.status == RotationStatus.COMPLETED
    assert rotation.records_processed == 1
    # A resumed pass validates an already-rotated root with the new key.
    rotation.status = RotationStatus.IN_PROGRESS
    await rotation_service._execute_rotation(rotation, _master_bytes_from_key(old_key), _master_bytes_from_key(new_key))
    assert rotation.status == RotationStatus.COMPLETED
    assert rotation.records_processed == 1
    assert (await store.db.fetchone("SELECT content FROM artifact_blobs"))[0] == old_blob
    monkeypatch.setenv("KESTREL_DATA_KEY", new_key)
    db2 = await reopen()
    try:
        restored = ArtifactRepository(db2, storage_id="local")
        await restored.initialize()
        assert restored._owner(SCOPE) == old_owner
        assert await restored.read(SCOPE, receipt.ref, "report.md") == b"private report"
        assert await restored.commit(SCOPE, request) == receipt
    finally:
        await db2.close()


@pytest.mark.asyncio
async def test_publication_cannot_consume_another_publications_staged_file(repository):
    store, _ = repository
    file = await store.stage(SCOPE, b"shared", publication_key="pending", path="same.txt", media_type="text/plain")
    request = ArtifactCommit(idempotency_key="different", title="Output",
                             manifest=ArtifactManifest(files=(file,), entrypoint=file.path))
    with pytest.raises(ResourceNotFound, match="not staged for this publication"):
        await store.commit(SCOPE, request)
    await store.stage(SCOPE, b"shared", publication_key="different", path="same.txt", media_type="text/plain")
    receipt = await store.commit(SCOPE, request)
    await store.delete(SCOPE, receipt.ref)
    # Completion and deletion remove only the matching publication's holds.
    assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_staging"))[0] == 1
    saved = await store.commit(SCOPE, request.model_copy(update={"idempotency_key": "pending"}))
    assert await store.read(SCOPE, saved.ref, "same.txt") == b"shared"


@pytest.mark.asyncio
async def test_staging_replay_is_bound_bounded_and_closed_after_publication(repository):
    store, _ = repository
    store.limits = ArtifactLimits(max_staging_files=1, staging_seconds=1)
    instant = datetime.now(UTC)
    store.clock = lambda: instant
    file = await store.stage(SCOPE, b"one", publication_key="save", path="a", media_type="text/plain")
    assert await store.stage(SCOPE, b"one", publication_key="save", path="a", media_type="text/plain") == file
    assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_staging"))[0] == 1
    for content, media_type in ((b"two", "text/plain"), (b"one", "text/markdown")):
        with pytest.raises(ResourceConflict, match="another file"):
            await store.stage(SCOPE, content, publication_key="save", path="a", media_type=media_type)
    with pytest.raises(ArtifactQuotaExceeded, match="staging file"):
        await store.stage(SCOPE, b"one", publication_key="other", path="a", media_type="text/plain")
    store.clock = lambda: instant + timedelta(seconds=2)
    # Expired data can be restaged and does not permanently consume quota.
    file = await store.stage(SCOPE, b"one", publication_key="save", path="a", media_type="text/plain")
    request = ArtifactCommit(idempotency_key="save", title="Output", manifest=ArtifactManifest(files=(file,), entrypoint=file.path))
    receipt = await store.commit(SCOPE, request)
    for deleted in (False, True):
        if deleted:
            await store.delete(SCOPE, receipt.ref)
        with pytest.raises(ResourceConflict, match="closed"):
            await store.stage(SCOPE, b"one", publication_key="save", path="a", media_type="text/plain")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "ignored", "malformed", "missing", "resumed", "mixed"])
async def test_artifact_root_rotation_failure_cannot_report_completed(repository, failure):
    import os
    from types import SimpleNamespace
    from kestrel_sovereign.security.key_rotation import KeyRotationService, RotationRecord, RotationStatus, _master_bytes_from_key
    store, _ = repository
    await publish(store)
    service = KeyRotationService(SimpleNamespace(database=store.db))
    await service.initialize()
    await store.db.execute("CREATE TABLE conversation_history (id INTEGER PRIMARY KEY, content TEXT, rendered_content TEXT, agent_id TEXT, metadata TEXT)")
    blob_type = "BYTEA" if store.db.backend_type == "postgres" else "BLOB"
    await store.db.execute(f"CREATE TABLE files (content_hash TEXT PRIMARY KEY, content {blob_type}, metadata TEXT)")
    if failure == "mixed":
        await store.db.execute("INSERT INTO artifact_keys (key_id, wrapped_key) VALUES (?, ?)", ("damaged-key", b"damaged"))
    elif failure == "missing":
        await store.db.execute("DELETE FROM artifact_keys")
    elif failure in {"malformed", "resumed"}:
        await store.db.execute("UPDATE artifact_keys SET wrapped_key = ?", (b"damaged",))
        if failure == "resumed":
            await store.db.execute("INSERT INTO rotation_progress (rotation_id, table_name, record_id, rotated_at) VALUES (?, ?, ?, ?)",
                                   ("failed-root", "artifact_keys", "v1", datetime.now(UTC).isoformat()))
    elif store.db.backend_type == "sqlite":
        action = "SELECT RAISE(ABORT, 'root update denied');" if failure == "raise" else "SELECT RAISE(IGNORE);"
        await store.db.execute(f"CREATE TRIGGER reject_key_rotation BEFORE UPDATE ON artifact_keys BEGIN {action} END")
    else:
        action = "RAISE EXCEPTION 'root update denied';" if failure == "raise" else "RETURN NULL;"
        await store.db.execute(f"CREATE FUNCTION reject_key_rotation() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN {action} END $$")
        await store.db.execute("CREATE TRIGGER reject_key_rotation BEFORE UPDATE ON artifact_keys FOR EACH ROW EXECUTE FUNCTION reject_key_rotation()")
    rotation = RotationRecord(id="failed-root", started_at=datetime.now(UTC), status=RotationStatus.IN_PROGRESS)
    await service._execute_rotation(rotation, _master_bytes_from_key(os.environ["KESTREL_DATA_KEY"]),
                                    _master_bytes_from_key(Fernet.generate_key().decode()))
    assert rotation.status == RotationStatus.FAILED
    assert rotation.records_processed in ({0, 1} if failure == "mixed" else {0})
    assert rotation.error_message
    assert (await store.db.fetchone("SELECT COUNT(*) FROM rotation_progress WHERE table_name = 'artifact_keys'"))[0] == rotation.records_processed + int(failure == "resumed")


@pytest.mark.asyncio
async def test_listing_decodes_one_snapshot_when_a_peer_deletes_after_selection(repository, monkeypatch):
    store, reopen = repository
    if store.db.backend_type != "postgres":
        pytest.skip("PostgreSQL READ COMMITTED statement race")
    receipt, _ = await publish(store)
    peer_db = await reopen()
    try:
        peer = ArtifactRepository(peer_db, storage_id="local")
        await peer.initialize()
        original = store.db.fetchall
        async def delete_after_selection(sql, *args, **kwargs):
            rows = await original(sql, *args, **kwargs)
            if sql.startswith("SELECT r.artifact_id"):
                await peer.delete(SCOPE, receipt.ref)
            return rows
        monkeypatch.setattr(store.db, "fetchall", delete_after_selection)
        page = await store.page(SCOPE)
        assert [item.ref for item in page.items] == [receipt.ref]
        with pytest.raises(ResourceNotFound):
            await store.get(SCOPE, receipt.ref)
    finally:
        await peer_db.close()


@pytest.mark.asyncio
async def test_confirmed_cas_rollback_allows_rebase_without_consuming_staging(repository):
    store, _ = repository
    first, _ = await publish(store)
    second, _ = await publish(store, key="winner", artifact=first.ref)
    file = await store.stage(SCOPE, b"my edits", publication_key="rebase", path="a", media_type="text/plain")
    request = ArtifactCommit(idempotency_key="rebase", artifact_id=first.ref.artifact_id,
                             expected_revision_id=first.ref.revision_id, title="Edited",
                             manifest=ArtifactManifest(files=(file,), entrypoint=file.path))
    with pytest.raises(ResourceConflict, match="head changed"):
        await store.commit(SCOPE, request)
    assert await store.publication_status(SCOPE, "rebase") is None
    rebased = request.model_copy(update={"expected_revision_id": second.ref.revision_id})
    third = await store.commit(SCOPE, rebased)
    assert await store.read(SCOPE, third.ref, "a") == b"my edits"
    assert (await store.get(SCOPE, third.ref)).parent_revision_id == second.ref.revision_id


@pytest.mark.asyncio
async def test_missing_envelope_root_cannot_silently_create_an_empty_namespace(repository):
    store, reopen = repository
    await publish(store)
    await store.db.execute("DELETE FROM artifact_keys")
    db2 = await reopen()
    try:
        restored = ArtifactRepository(db2, storage_id="local")
        with pytest.raises((ArtifactIntegrityError, TransactionError), match="envelope key is missing"):
            await restored.initialize()
        assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_keys"))[0] == 0
    finally:
        await db2.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["root", "files"])
@pytest.mark.parametrize("failure", ["raise", "ignored"])
async def test_partial_rotation_resumes_its_journal_after_root_failure(repository, tmp_path, monkeypatch, target, failure):
    import os
    from types import SimpleNamespace
    from kestrel_sdk.security.aead import AEADCipher
    from kestrel_sovereign.security.key_rotation import KeyRotationService, RotationStatus, _master_bytes_from_key
    store, reopen = repository
    receipt, _ = await publish(store)
    old_key = os.environ["KESTREL_DATA_KEY"]
    new_key = Fernet.generate_key().decode()
    old_cipher, new_cipher = (AEADCipher(_master_bytes_from_key(key)) for key in (old_key, new_key))
    new_key_file = tmp_path / "new-key"
    new_key_file.write_text(new_key)
    new_key_file.chmod(0o600)
    service = KeyRotationService(SimpleNamespace(database=store.db))
    await service.initialize()
    await store.db.execute("CREATE TABLE conversation_history (id INTEGER PRIMARY KEY, content TEXT, rendered_content TEXT, agent_id TEXT, metadata TEXT)")
    blob_type = "BYTEA" if store.db.backend_type == "postgres" else "BLOB"
    await store.db.execute(f"CREATE TABLE files (content_hash TEXT PRIMARY KEY, content {blob_type}, metadata TEXT)")
    await store.db.execute("INSERT INTO conversation_history (id, content, metadata) VALUES (?, ?, '{\"enc\": true}')", (1, old_cipher.encrypt(b"existing conversation").decode()))
    for index in range(8):
        await store.db.execute("INSERT INTO files (content_hash, content, metadata) VALUES (?, ?, '{\"enc\": true}')", (str(index), old_cipher.encrypt(b"existing file")))
    table = "artifact_keys" if target == "root" else "files"
    if store.db.backend_type == "sqlite":
        action = "SELECT RAISE(ABORT, 'update denied');" if failure == "raise" else "SELECT RAISE(IGNORE);"
        await store.db.execute(f"CREATE TRIGGER root_unavailable BEFORE UPDATE ON {table} BEGIN {action} END")
    else:
        action = "RAISE EXCEPTION 'update denied';" if failure == "raise" else "RETURN NULL;"
        await store.db.execute(f"CREATE FUNCTION root_unavailable() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN {action} END $$")
        await store.db.execute(f"CREATE TRIGGER root_unavailable BEFORE UPDATE ON {table} FOR EACH ROW EXECUTE FUNCTION root_unavailable()")
    try:
        rotation_id = await service.start_rotation(new_key=new_key)
        await service.drain_rotations()
        failed = await service.get_status(rotation_id)
        assert failed.status == RotationStatus.FAILED
        assert failed.records_processed == (9 if target == "root" else 1)
        file_cipher = new_cipher if target == "root" else old_cipher
        assert all(file_cipher.decrypt(bytes(row[0])) == b"existing file" for row in await store.db.fetchall("SELECT content FROM files"))
        assert new_cipher.decrypt((await store.db.fetchone("SELECT content FROM conversation_history"))[0].encode()) == b"existing conversation"
        # A checkpoint counter may lag the atomic progress journal after a crash.
        await store.db.execute("UPDATE key_rotations SET records_processed = 0 WHERE id = ?", (rotation_id,))
        with pytest.raises(RuntimeError, match="Resume"):
            await service.start_rotation(new_key=new_key)
        if store.db.backend_type == "sqlite":
            await store.db.execute("DROP TRIGGER root_unavailable")
        else:
            await store.db.execute(f"DROP TRIGGER root_unavailable ON {table}")
        # Verify both keys, not only the destination: after a mistaken config swap
        # the operator must restore the old key before resuming the same journal.
        with monkeypatch.context() as patch:
            patch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
            with pytest.raises(ValueError, match="old key"):
                await service.resume_rotation(rotation_id, str(new_key_file))
        wrong_key_file = tmp_path / "wrong-key"
        wrong_key_file.write_text(Fernet.generate_key().decode())
        with pytest.raises(ValueError, match="new key"):
            await service.resume_rotation(rotation_id, str(wrong_key_file))
        execution_allowed = asyncio.Event()
        execute = service._execute_rotation
        async def held_execution(*args, **kwargs):
            await execution_allowed.wait()
            await execute(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(service, "_execute_rotation", held_execution)
            attempts = await asyncio.gather(service.resume_rotation(rotation_id, str(new_key_file)),
                                            service.resume_rotation(rotation_id, str(new_key_file)), return_exceptions=True)
            assert sum(value is True for value in attempts) == 1
            assert sum(isinstance(value, (RuntimeError, ValueError)) for value in attempts) == 1
            execution_allowed.set()
            await service.drain_rotations()
        complete = await service.get_status(rotation_id)
        assert complete.status == RotationStatus.COMPLETED
        assert complete.records_processed == 10
        assert complete.error_message is None
        assert all(new_cipher.decrypt(bytes(row[0])) == b"existing file" for row in await store.db.fetchall("SELECT content FROM files"))
        monkeypatch.setenv("KESTREL_DATA_KEY", new_key)
        db2 = await reopen()
        try:
            restored = ArtifactRepository(db2, storage_id="local")
            await restored.initialize()
            assert await restored.read(SCOPE, receipt.ref, "report.md") == b"private report"
        finally:
            await db2.close()
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_head_uses_committed_pointer_despite_clock_reversal(repository):
    store, _ = repository
    first, _ = await publish(store)
    before = store.clock()
    store.clock = lambda: before - timedelta(days=1)
    second, _ = await publish(store, key="later-commit", artifact=first.ref)
    assert (await store.page(SCOPE, artifact_id=first.ref.artifact_id)).items[0].ref == first.ref
    assert (await store.head(SCOPE, first.ref.artifact_id)).ref == second.ref
    for other in (SCOPE.model_copy(update={"tenant_id": "other"}),
                  SCOPE.model_copy(update={"principal_id": "other"}),
                  SCOPE.model_copy(update={"agent_id": "other"}),
                  SCOPE.model_copy(update={"agent_id": None})):
        with pytest.raises(ResourceNotFound):
            await store.head(other, first.ref.artifact_id)
    await store.delete(SCOPE, second.ref)
    with pytest.raises(ResourceNotFound):
        await store.head(SCOPE, first.ref.artifact_id)


@pytest.mark.asyncio
async def test_head_validates_identifier_and_decodes_owner_bound_revision(repository):
    store, _ = repository
    with pytest.raises(ValueError):
        await store.head(SCOPE, "../invalid")
    with pytest.raises(ResourceNotFound):
        await store.head(SCOPE, "absent")
    first, _ = await publish(store)
    second, _ = await publish(store, key="different-artifact")
    # A corrupt head link must not disclose a different artifact's payload.
    await store.db.execute("UPDATE artifact_heads SET revision_id = ? WHERE artifact_id = ?",
                           (second.ref.revision_id, first.ref.artifact_id))
    with pytest.raises(ArtifactIntegrityError, match="head revision is unavailable"):
        await store.head(SCOPE, first.ref.artifact_id)
