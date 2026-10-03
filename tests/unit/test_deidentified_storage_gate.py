"""The evidence-gated de-identified save on PrivacyEnforcingStorage (#1761).

DEIDENTIFIED refuses every generic write. ``store_deidentified_records`` is the
one durable write it admits, and only with a valid evidence artifact persisted
alongside the records in the same transaction. Records are synthetic.
"""

import asyncio
import dataclasses
import hashlib
import json
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
