"""The evidence-gated de-identified save on PrivacyEnforcingStorage (#1761).

DEIDENTIFIED refuses every generic write. ``store_deidentified_records`` is the
one durable write it admits, and only with a valid evidence artifact persisted
alongside the records in the same transaction. Records are synthetic.
"""

import asyncio
import dataclasses
import hashlib
import json
import sqlite3
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from kestrel_sdk.storage.database.interface import TransactionError

from kestrel_sovereign.deidentification import (
    EXPORT_SCHEMA,
    ActualKnowledgeAttestation,
    DeidentificationEvidence,
    DeidentificationPipeline,
    DeidentificationResult,
    EvidenceValidationError,
    ExpertDeterminationReference,
    FieldSpec,
    OperatorContext,
    SafeHarborIdentifier,
    SourceRecord,
    validate_evidence,
)
from kestrel_sovereign.endpoints.files import serve_file
from kestrel_sovereign.privacy import PrivacyConfig, PrivacyMode
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.privacy_wrapper import (
    DEIDENTIFICATION_EVIDENCE_KIND,
    DEIDENTIFIED_RECORDS_KIND,
    PrivacyEnforcingStorage,
    PrivacyViolationError,
)

FIXED_NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
SCHEMA = {
    "name": FieldSpec.remove(SafeHarborIdentifier.NAMES),
    "zip": FieldSpec.zip_code(),
    "dob": FieldSpec.birth_date(),
    "diagnosis": FieldSpec.non_identifying(),
}
RECORD = {
    "name": "Avery Quillfeather",
    "zip": "02139",
    "dob": "1950-02-03",
    "diagnosis": "synthetic condition",
}
EXPERT_REPORT = ExpertDeterminationReference(
    report_reference="doc:expert-report/synthetic-1",
    report_digest=hashlib.sha256(b"synthetic expert report").hexdigest(),
    expert_id="expert:synthetic",
)


def _result(method="safe_harbor"):
    pipeline = DeidentificationPipeline(
        SCHEMA,
        source_digest_key=b"synthetic-source-digest-key-0001",
        clock=lambda: FIXED_NOW,
    )
    if method == "safe_harbor":
        extra = {"attestation": ActualKnowledgeAttestation(
            "operator:synthetic", True, attested_at=FIXED_NOW.isoformat()
        )}
    else:
        extra = {"method": method, "expert_determination": EXPERT_REPORT}
    return pipeline.run(
        [SourceRecord("chart-778812", RECORD)],
        operator=OperatorContext("operator:synthetic", "request-0001"),
        **extra,
    )


def _mock_storage():
    storage = Mock()
    storage.store_file = AsyncMock(side_effect=["evidence-hash", "records-hash"])
    storage.add_conversation = AsyncMock()
    storage.owns_open_transaction = False

    @asynccontextmanager
    async def transaction(*, immediate=False):
        yield

    storage.transaction = transaction
    return storage


@pytest.fixture
async def sqlite_storage(tmp_path):
    storage = await AsyncStorage.create_sqlite(str(tmp_path / "deid.db"))
    try:
        yield storage
    finally:
        await storage.close()


async def test_generic_writes_stay_refused_in_deidentified_mode():
    storage = _mock_storage()
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    with pytest.raises(PrivacyViolationError):
        await wrapper.add_conversation("user", "raw clinical note")
    with pytest.raises(PrivacyViolationError):
        await wrapper.store_file(b"raw", "note.txt")
    storage.add_conversation.assert_not_called()
    storage.store_file.assert_not_called()
    assert wrapper.allows_persistent_writes() is False


async def test_deidentified_save_persists_records_with_their_evidence(sqlite_storage):
    wrapper = PrivacyEnforcingStorage(sqlite_storage, PrivacyMode.DEIDENTIFIED)
    result = _result()

    receipt = await wrapper.store_deidentified_records(result)

    assert receipt.evidence_id == result.evidence.evidence_id
    assert receipt.assurance == "safe_harbor"
    assert receipt.record_count == 1

    evidence_bytes = await wrapper.retrieve_file(receipt.evidence_file_hash)
    stored_evidence = DeidentificationEvidence.from_json_bytes(evidence_bytes)
    assert stored_evidence == result.evidence

    # The records document is the export bundle: it carries its own artifact.
    records_bytes = await wrapper.retrieve_file(receipt.records_file_hash)
    assert records_bytes == result.export_bundle()
    assert json.loads(records_bytes)["schema"] == EXPORT_SCHEMA
    restored = DeidentificationResult.from_export_bundle(
        records_bytes, required_assurance="safe_harbor"
    )
    assert restored.evidence == stored_evidence
    assert restored.records_as_dicts() == result.records_as_dicts()
    validate_evidence(stored_evidence, restored.records, required_assurance="safe_harbor")

    evidence_meta = await sqlite_storage.get_file_metadata(receipt.evidence_file_hash)
    records_meta = await sqlite_storage.get_file_metadata(receipt.records_file_hash)
    assert evidence_meta["kind"] == DEIDENTIFICATION_EVIDENCE_KIND
    assert records_meta["kind"] == DEIDENTIFIED_RECORDS_KIND
    assert records_meta["evidence_file_hash"] == receipt.evidence_file_hash
    for meta in (evidence_meta, records_meta):
        assert meta["evidence_id"] == receipt.evidence_id
        assert meta["artifact_digest"] == result.evidence.artifact_digest
        assert meta["mime_type"] == "application/json"


async def test_downloaded_records_carry_a_verifiable_evidence_artifact(tmp_path):
    """GET /api/files/{hash} serves stored bytes as they are, so the records
    document itself must embed the artifact that authorizes it."""
    agent_id = "did:test:deidentified-download"
    storage = AsyncStorage(str(tmp_path / "download.db"), agent_id=agent_id)
    await storage.initialize()
    try:
        await storage.graph.add_node(GraphNode(agent_id, "agent", "Synthetic", {}))
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
        result = _result()
        receipt = await wrapper.store_deidentified_records(result)

        request = SimpleNamespace(
            state=SimpleNamespace(agent=SimpleNamespace(storage=wrapper, agent_id=agent_id))
        )
        response = await serve_file(receipt.records_file_hash, request)

        assert response.media_type == "application/json"
        downloaded = DeidentificationResult.from_export_bundle(
            bytes(response.body), required_assurance="safe_harbor"
        )
        assert downloaded.evidence == result.evidence
        assert downloaded.records_as_dicts() == result.records_as_dicts()
        assert downloaded.evidence.attestation.no_actual_knowledge is True
        with pytest.raises(EvidenceValidationError, match="not the required"):
            DeidentificationResult.from_export_bundle(
                bytes(response.body), required_assurance="expert_determination"
            )
    finally:
        await storage.close()


SOURCE_VALUES = ("Avery", "Quillfeather", "chart-778812", "02139", "1950-02-03")


async def test_saved_documents_leak_no_source_value_and_are_encrypted(tmp_path, monkeypatch):
    """Metadata and file names are plaintext columns; they must be content-free.
    The documents themselves are encrypted when a data key is configured."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "synthetic-deidentification-test-key")
    storage = await AsyncStorage.create_sqlite(str(tmp_path / "encrypted.db"))
    try:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
        receipt = await wrapper.store_deidentified_records(_result())
        rows = await storage.db.fetchall(
            "SELECT content_hash, original_name, metadata, content FROM files"
        )
        assert {row[0] for row in rows} == {
            receipt.evidence_file_hash, receipt.records_file_hash
        }
        for _, original_name, metadata, content in rows:
            plaintext_columns = f"{original_name} {metadata}"
            assert json.loads(metadata)["enc"] is True
            for value in SOURCE_VALUES:
                assert value not in plaintext_columns
            assert b"evidence_id" not in bytes(content)  # stored ciphertext
        evidence_bytes = await wrapper.retrieve_file(receipt.evidence_file_hash)
        for value in SOURCE_VALUES:
            assert value.encode() not in evidence_bytes
    finally:
        await storage.close()


async def test_records_are_never_stored_without_their_evidence(sqlite_storage, monkeypatch):
    """The two writes share a transaction: a failed records write leaves no evidence row,
    and evidence is always written first."""
    wrapper = PrivacyEnforcingStorage(sqlite_storage, PrivacyMode.DEIDENTIFIED)
    result = _result()
    real_store_file = sqlite_storage.store_file
    kinds = []

    async def store_file(content, original_name, metadata=None):
        kinds.append(metadata["kind"])
        if metadata["kind"] == DEIDENTIFIED_RECORDS_KIND:
            raise RuntimeError("synthetic write failure")
        return await real_store_file(content, original_name, metadata)

    monkeypatch.setattr(sqlite_storage, "store_file", store_file)
    with pytest.raises(TransactionError, match="synthetic write failure"):
        await wrapper.store_deidentified_records(result)

    assert kinds == [DEIDENTIFICATION_EVIDENCE_KIND, DEIDENTIFIED_RECORDS_KIND]
    evidence_hash = hashlib.sha256(result.evidence.to_json_bytes()).hexdigest()
    assert await sqlite_storage.files.file_exists(evidence_hash) is False
    assert wrapper._active_deidentified_save_leases == 0


class _ClaimsSafeHarbor(DeidentificationEvidence):
    """Evidence that reports ``safe_harbor`` whatever its method is, and
    serializes that claim."""

    @property
    def assurance(self):
        return "safe_harbor"


class _ClaimsSafeHarborOnlyInProcess(_ClaimsSafeHarbor):
    """Serializes its true assurance, so its bytes are a valid Expert
    Determination artifact; only the in-process property claims safe_harbor."""

    def _body(self):
        body = super()._body()
        body["assurance"] = self.method.value
        return body


@pytest.mark.parametrize(
    "evidence_type, message",
    [
        (_ClaimsSafeHarbor, "assurance does not match method"),
        (_ClaimsSafeHarborOnlyInProcess, "requires 'safe_harbor'"),
    ],
)
async def test_save_takes_its_assurance_from_the_verified_bytes(evidence_type, message):
    """Regression: an Expert Determination artifact with no attestation, wrapped
    to report safe_harbor, passed the DEIDENTIFIED gate on the object alone."""
    expert = _result("expert_determination")
    fields = {f.name: getattr(expert.evidence, f.name) for f in dataclasses.fields(expert.evidence)}
    draft = evidence_type(**fields)
    claim = dataclasses.replace(draft, artifact_digest=draft.compute_digest())
    result = DeidentificationResult(expert.records, claim)
    assert result.evidence.assurance == "safe_harbor"
    assert result.evidence.attestation is None

    storage = _mock_storage()
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    with pytest.raises(PrivacyViolationError, match=message):
        await wrapper.store_deidentified_records(result)
    storage.store_file.assert_not_called()


@pytest.mark.parametrize("not_a_result", [None, {"records": [], "evidence": {}}, "bundle"])
async def test_save_requires_a_pipeline_result(not_a_result):
    storage = _mock_storage()
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    with pytest.raises(PrivacyViolationError, match="DeidentificationResult"):
        await wrapper.store_deidentified_records(not_a_result)
    storage.store_file.assert_not_called()


async def test_save_rejects_an_altered_evidence_artifact():
    storage = _mock_storage()
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    result = _result()
    # Structurally valid, so only the artifact digest can catch it.
    altered = dataclasses.replace(
        result.evidence, operator=OperatorContext("operator:someone-else")
    )
    object.__setattr__(result, "evidence", altered)

    with pytest.raises(PrivacyViolationError, match="artifact_digest does not match"):
        await wrapper.store_deidentified_records(result)
    storage.store_file.assert_not_called()


async def test_save_refuses_a_result_subclass():
    """A subclass could override verify() or the records it hands over."""

    class Permissive(DeidentificationResult):
        def verify(self, *, required_assurance=None):
            return None

    storage = _mock_storage()
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    genuine = _result()
    forged = Permissive(({"name": "Avery Quillfeather"},), genuine.evidence)
    with pytest.raises(PrivacyViolationError, match="DeidentificationResult"):
        await wrapper.store_deidentified_records(forged)
    storage.store_file.assert_not_called()


async def test_save_rejects_records_the_evidence_does_not_describe():
    storage = _mock_storage()
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    result = _result()
    object.__setattr__(result, "records", (dict(result.records[0], diagnosis="other"),))

    with pytest.raises(PrivacyViolationError, match="output digest"):
        await wrapper.store_deidentified_records(result)
    storage.store_file.assert_not_called()


async def test_deidentified_mode_requires_the_configured_assurance():
    storage = _mock_storage()
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    with pytest.raises(PrivacyViolationError, match="requires 'safe_harbor'"):
        await wrapper.store_deidentified_records(_result("expert_determination"))
    storage.store_file.assert_not_called()

    expert_config = PrivacyConfig(storage="deidentified", assurance="expert_determination")
    wrapper.set_privacy_mode(expert_config)
    with pytest.raises(PrivacyViolationError, match="requires 'expert_determination'"):
        await wrapper.store_deidentified_records(_result())
    receipt = await wrapper.store_deidentified_records(_result("expert_determination"))
    assert receipt.assurance == "expert_determination"


async def test_a_configured_assurance_is_enforced_outside_deidentified_storage():
    storage = _mock_storage()
    config = PrivacyConfig(storage="full", assurance="expert_determination")
    wrapper = PrivacyEnforcingStorage(storage, config)
    with pytest.raises(PrivacyViolationError, match="requires 'expert_determination'"):
        await wrapper.store_deidentified_records(_result())
    storage.store_file.assert_not_called()
    receipt = await wrapper.store_deidentified_records(_result("expert_determination"))
    assert receipt.assurance == "expert_determination"


@pytest.mark.parametrize("mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED])
async def test_volatile_modes_refuse_deidentified_saves(mode):
    storage = _mock_storage()
    wrapper = PrivacyEnforcingStorage(storage, mode)
    with pytest.raises(PrivacyViolationError, match="persistent writes are disabled"):
        await wrapper.store_deidentified_records(_result())
    storage.store_file.assert_not_called()


@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.ANONYMOUS])
async def test_persistent_modes_accept_valid_evidence(mode):
    storage = _mock_storage()
    wrapper = PrivacyEnforcingStorage(storage, mode)
    receipt = await wrapper.store_deidentified_records(_result())
    assert (receipt.evidence_file_hash, receipt.records_file_hash) == (
        "evidence-hash",
        "records-hash",
    )
    assert storage.store_file.await_count == 2


async def test_privacy_transition_is_refused_while_a_save_is_in_flight():
    storage = _mock_storage()
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_store_file(content, original_name, metadata=None):
        entered.set()
        await release.wait()
        return hashlib.sha256(content).hexdigest()

    storage.store_file = slow_store_file
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    save = asyncio.create_task(wrapper.store_deidentified_records(_result()))
    await asyncio.wait_for(entered.wait(), timeout=30)

    with pytest.raises(PrivacyViolationError, match="de-identified record save"):
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert wrapper.privacy_mode is PrivacyMode.DEIDENTIFIED

    release.set()
    await asyncio.wait_for(save, timeout=30)
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert wrapper.privacy_mode is PrivacyMode.EPHEMERAL


@pytest.fixture
def delayed_sqlite_commit(sqlite_storage, monkeypatch):
    """Hold the SQLite worker inside COMMIT until the test releases it.

    The commit runs on aiosqlite's worker thread exactly as a real one does, so
    a caller cancelled while awaiting it returns at once while the worker
    still commits afterwards.
    """
    connection = sqlite_storage.db._backend._connection
    real_commit = connection._conn.commit
    entered, release = threading.Event(), threading.Event()

    def blocked_commit():
        entered.set()
        release.wait(timeout=30)
        real_commit()

    async def commit():
        await connection._execute(blocked_commit)

    monkeypatch.setattr(connection, "commit", commit)
    try:
        yield SimpleNamespace(entered=entered, release=release)
    finally:
        release.set()  # never leave the worker blocked behind a failed assertion


async def _file_rows(storage):
    return await storage.db.fetchall("SELECT content_hash FROM files")


async def test_a_cancelled_save_holds_its_lease_until_its_commit_resolves(
    sqlite_storage, delayed_sqlite_commit
):
    """Regression: cancelling a save while SQLite's worker was committing handed
    the commit to the backend's cancellation drain and returned at once. The
    lease was released, a transition to EPHEMERAL succeeded, and the worker
    then committed both documents under EPHEMERAL. The save now owns its
    commit: the transition stays refused until the commit has resolved, and
    the documents it wrote were committed before any transition."""
    wrapper = PrivacyEnforcingStorage(sqlite_storage, PrivacyMode.DEIDENTIFIED)
    save = asyncio.create_task(wrapper.store_deidentified_records(_result()))
    await asyncio.wait_for(asyncio.to_thread(delayed_sqlite_commit.entered.wait, 30), timeout=30)

    save.cancel()
    for _ in range(20):  # the old path released the lease within one step
        await asyncio.sleep(0)
    assert not save.done()
    with pytest.raises(PrivacyViolationError, match="de-identified record save"):
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert wrapper.privacy_mode is PrivacyMode.DEIDENTIFIED

    delayed_sqlite_commit.release.set()
    await asyncio.wait({save}, timeout=30)
    assert save.cancelled()  # the caller's cancellation is still delivered
    assert len(await _file_rows(sqlite_storage)) == 2
    assert wrapper._active_deidentified_save_leases == 0

    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert len(await _file_rows(sqlite_storage)) == 2


def _commit_task():
    (commit,) = [
        task for task in asyncio.all_tasks()
        if task.get_name().startswith("deidentified-save:")
    ]
    return commit


async def test_a_commit_task_cancelled_during_commit_keeps_the_fence_closed(
    sqlite_storage, delayed_sqlite_commit
):
    """Only event-loop teardown can cancel the commit task itself. Cancelled
    during COMMIT, whether its documents committed is unknown, so the lease is
    kept and transitions stay refused rather than following a commit that may
    still land."""
    wrapper = PrivacyEnforcingStorage(sqlite_storage, PrivacyMode.DEIDENTIFIED)
    save = asyncio.create_task(wrapper.store_deidentified_records(_result()))
    await asyncio.wait_for(asyncio.to_thread(delayed_sqlite_commit.entered.wait, 30), timeout=30)

    _commit_task().cancel()
    with pytest.raises(PrivacyViolationError, match="commit outcome is unknown"):
        await asyncio.wait_for(save, timeout=30)
    assert wrapper._active_deidentified_save_leases == 1
    with pytest.raises(PrivacyViolationError, match="de-identified record save"):
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)


async def test_a_commit_that_fails_after_commit_was_issued_keeps_the_fence_closed(
    sqlite_storage, monkeypatch
):
    """A COMMIT that raises may still have landed (a COMMIT sent to PostgreSQL
    completes on the server whatever the client saw), so the wrapper cannot
    treat the failure as a known rollback."""
    wrapper = PrivacyEnforcingStorage(sqlite_storage, PrivacyMode.DEIDENTIFIED)
    connection = sqlite_storage.db._backend._connection

    async def failing_commit():
        raise sqlite3.OperationalError("synthetic disk I/O error")

    monkeypatch.setattr(connection, "commit", failing_commit)
    with pytest.raises(PrivacyViolationError, match="commit outcome is unknown") as failed:
        await wrapper.store_deidentified_records(_result())
    assert isinstance(failed.value.__cause__, TransactionError)
    assert wrapper._active_deidentified_save_leases == 1
    with pytest.raises(PrivacyViolationError, match="de-identified record save"):
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)


async def test_a_commit_task_cancelled_before_commit_releases_the_lease():
    """Cancelled while still writing, the transaction rolls back and nothing
    commits, so the outcome is known and the lease is released."""
    storage = _mock_storage()
    writing = asyncio.Event()

    async def blocked_store_file(content, original_name, metadata=None):
        writing.set()
        await asyncio.Event().wait()

    storage.store_file = blocked_store_file
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    save = asyncio.create_task(wrapper.store_deidentified_records(_result()))
    await asyncio.wait_for(writing.wait(), timeout=30)

    _commit_task().cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(save, timeout=30)
    assert wrapper._active_deidentified_save_leases == 0
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)


def _recording_storage(on_commit=None):
    """Mock storage whose transaction records how it ended; ``on_commit``
    runs where leaving the transaction would issue COMMIT."""
    storage = _mock_storage()
    storage.ended = []

    @asynccontextmanager
    async def transaction(*, immediate=False):
        try:
            yield
        except BaseException:
            storage.ended.append("rolled back")
            raise
        if on_commit is not None:
            await on_commit()
        storage.ended.append("committed")

    storage.transaction = transaction
    return storage


async def test_a_caller_cancelled_while_writing_withdraws_the_save():
    """Before COMMIT is issued, cancelling the commit task rolls it back, so
    the caller's cancellation withdraws the save: a wait inside the
    transaction (a row lock its own parent holds, say) stays cancellable."""
    storage = _recording_storage()
    writing = asyncio.Event()

    async def blocked_store_file(content, original_name, metadata=None):
        writing.set()
        await asyncio.Event().wait()

    storage.store_file = blocked_store_file
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    save = asyncio.create_task(wrapper.store_deidentified_records(_result()))
    await asyncio.wait_for(writing.wait(), timeout=30)

    save.cancel("caller timeout")
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(save, timeout=30)
    assert storage.ended == ["rolled back"]
    assert wrapper._active_deidentified_save_leases == 0
    assert not getattr(cancelled.value, "__notes__", None)  # withdrawn, not failed
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)


async def test_a_caller_cancelled_once_commit_is_issued_lets_it_finish():
    """Once COMMIT is issued, the caller's cancellation no longer withdraws
    the save, even when it lands before the caller has woken for that point.
    Interrupting a COMMIT would leave its outcome unknown and the fence closed
    for good, so the commit finishes first."""
    release, save = asyncio.Event(), None

    async def on_commit():
        save.cancel()  # the caller has not yet seen COMMIT being issued
        await release.wait()

    storage = _recording_storage(on_commit)
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    save = asyncio.create_task(wrapper.store_deidentified_records(_result()))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not save.done()
    with pytest.raises(PrivacyViolationError, match="de-identified record save"):
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)

    release.set()
    await asyncio.wait({save}, timeout=30)
    assert save.cancelled()
    assert storage.ended == ["committed"]
    assert wrapper._active_deidentified_save_leases == 0


async def test_a_save_cancelled_before_its_transaction_starts_is_withdrawn(sqlite_storage):
    """Until its transaction starts nothing is written, so the caller's
    cancellation (a timeout around a save queued behind another writer) still
    withdraws it: the lease is released and nothing is written later."""
    wrapper = PrivacyEnforcingStorage(sqlite_storage, PrivacyMode.DEIDENTIFIED)
    holding, release = asyncio.Event(), asyncio.Event()

    async def hold_a_transaction():
        async with sqlite_storage.transaction():
            holding.set()
            await release.wait()

    holder = asyncio.create_task(hold_a_transaction())
    await asyncio.wait_for(holding.wait(), timeout=30)
    save = asyncio.create_task(wrapper.store_deidentified_records(_result()))
    while wrapper._active_deidentified_save_leases == 0:
        assert not save.done()
        await asyncio.sleep(0)

    save.cancel()
    await asyncio.wait({save}, timeout=30)
    assert save.cancelled()
    assert wrapper._active_deidentified_save_leases == 0
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    release.set()
    await asyncio.wait_for(holder, timeout=30)
    assert await _file_rows(sqlite_storage) == []


@pytest.mark.parametrize("outer", ["storage", "wrapper"])
async def test_a_save_inside_an_open_transaction_is_refused(sqlite_storage, outer):
    """Regression: a save inside a caller's transaction joined it, so its lease
    was released before the documents committed and a transition to EPHEMERAL
    could land in between. The save is refused, writes nothing, and holds no
    lease afterwards."""
    wrapper = PrivacyEnforcingStorage(sqlite_storage, PrivacyMode.DEIDENTIFIED)
    transaction = (sqlite_storage if outer == "storage" else wrapper).transaction
    async with transaction():
        with pytest.raises(PrivacyViolationError, match="already has one open"):
            await wrapper.store_deidentified_records(_result())
        assert wrapper._active_deidentified_save_leases == 0
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert await sqlite_storage.db.fetchall("SELECT content_hash FROM files") == []


async def test_another_tasks_transaction_does_not_refuse_a_save(sqlite_storage):
    """The check is per task: a save beside another task's transaction opens
    its own, waits for SQLite's writer, and commits before releasing."""
    wrapper = PrivacyEnforcingStorage(sqlite_storage, PrivacyMode.DEIDENTIFIED)
    holding, release = asyncio.Event(), asyncio.Event()

    async def hold_a_transaction():
        async with sqlite_storage.transaction():
            holding.set()
            await release.wait()

    holder = asyncio.create_task(hold_a_transaction())
    await asyncio.wait_for(holding.wait(), timeout=30)
    save = asyncio.create_task(wrapper.store_deidentified_records(_result()))

    async def lease_taken():
        while wrapper._active_deidentified_save_leases == 0:
            if save.done():
                save.result()  # surface a refusal instead of spinning
            await asyncio.sleep(0)

    await asyncio.wait_for(lease_taken(), timeout=30)
    assert not save.done()  # waiting for the other task's writer slot
    with pytest.raises(PrivacyViolationError, match="de-identified record save"):
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    release.set()
    await asyncio.wait_for(holder, timeout=30)
    receipt = await asyncio.wait_for(save, timeout=30)
    assert await sqlite_storage.files.file_exists(receipt.records_file_hash)


_MISSING = object()


@pytest.mark.parametrize("answer", [_MISSING, None, True, "no"])
async def test_a_save_needs_storage_that_reports_no_open_transaction(answer):
    storage = _mock_storage()
    if answer is _MISSING:
        del storage.owns_open_transaction
    else:
        storage.owns_open_transaction = answer
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.DEIDENTIFIED)
    with pytest.raises(PrivacyViolationError, match="already has one open"):
        await wrapper.store_deidentified_records(_result())
    storage.store_file.assert_not_called()
    assert wrapper._active_deidentified_save_leases == 0
