"""Authority-boundary tests for the loopback Kite core-erasure drill."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kestrel_sovereign.knowledge import kite_evidence_signing
from kestrel_sovereign.knowledge.kite_evidence_signing import (
    KiteEvidenceSigningError,
    consume_kite_evidence_nonce,
)
from kestrel_sovereign.knowledge.kite_erasure_authority import (
    ERASURE_CORE_SNAPSHOT_OPERATION,
    KiteErasureDrillAuthorityError,
    KiteErasureDrillCapability,
    _consume_kite_erasure_drill_capability,
    _issue_kite_erasure_drill_capability,
    _typed_kite_erasure_endpoint_issuance_scope,
)


def _enabled_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "kite-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("KESTREL_KITE_RELEASE_EVIDENCE", "1")
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.setenv("KESTREL_KITE_RELEASE_EVIDENCE_ROOT", str(home))


def _receipt(nonce: str):
    receipt = consume_kite_evidence_nonce(nonce, issue_receipt=True)
    assert receipt is not None
    return receipt


def test_erasure_authority_rejects_direct_construction_and_same_process_issuance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enabled_home(monkeypatch, tmp_path)

    with pytest.raises(TypeError, match="typed evidence endpoint"):
        KiteErasureDrillCapability()
    with pytest.raises(KiteErasureDrillAuthorityError, match="endpoint task"):
        _issue_kite_erasure_drill_capability(
            _receipt("a" * 64), operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )
    with pytest.raises(KiteErasureDrillAuthorityError, match="active endpoint task"):
        with _typed_kite_erasure_endpoint_issuance_scope():
            pass


@pytest.mark.asyncio
async def test_erasure_authority_is_exactly_bound_burned_and_route_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enabled_home(monkeypatch, tmp_path)
    with _typed_kite_erasure_endpoint_issuance_scope():
        capability = _issue_kite_erasure_drill_capability(
            _receipt("b" * 64), operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )
        with pytest.raises(KiteErasureDrillAuthorityError, match="another operation"):
            _consume_kite_erasure_drill_capability(
                capability, expected_operation="diagnostics",
            )
        # A cross-operation attempt burns the one-shot capability; it cannot
        # be retried against the correct storage operation.
        with pytest.raises(KiteErasureDrillAuthorityError, match="already consumed"):
            _consume_kite_erasure_drill_capability(
                capability, expected_operation=ERASURE_CORE_SNAPSHOT_OPERATION,
            )

    with _typed_kite_erasure_endpoint_issuance_scope():
        capability = _issue_kite_erasure_drill_capability(
            _receipt("c" * 64), operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )
        binding = _consume_kite_erasure_drill_capability(
            capability, expected_operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )
    assert binding.operation_id == f"kite-erasure-{'c' * 64}"
    assert len(binding.correlation) == 64
    with pytest.raises(KiteErasureDrillAuthorityError, match="endpoint task"):
        _consume_kite_erasure_drill_capability(
            capability, expected_operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )


@pytest.mark.asyncio
async def test_child_task_cannot_issue_or_consume_copied_erasure_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enabled_home(monkeypatch, tmp_path)

    async def issue_in_child(receipt: object):
        return _issue_kite_erasure_drill_capability(
            receipt, operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )

    async def consume_in_child(capability: object):
        return _consume_kite_erasure_drill_capability(
            capability, expected_operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )

    with _typed_kite_erasure_endpoint_issuance_scope():
        child_issue = asyncio.create_task(issue_in_child(_receipt("d" * 64)))
        with pytest.raises(KiteErasureDrillAuthorityError, match="endpoint task"):
            await child_issue
        capability = _issue_kite_erasure_drill_capability(
            _receipt("e" * 64), operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )
        child_consume = asyncio.create_task(consume_in_child(capability))
        with pytest.raises(KiteErasureDrillAuthorityError, match="endpoint task"):
            await child_consume
        # Child rejection does not convert the valid parent's authority into a
        # replay; the exact owner can still consume it once.
        _consume_kite_erasure_drill_capability(
            capability, expected_operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )

    release_child = asyncio.Event()

    async def consume_after_route(capability: object):
        await release_child.wait()
        return await consume_in_child(capability)

    with _typed_kite_erasure_endpoint_issuance_scope():
        delayed_capability = _issue_kite_erasure_drill_capability(
            _receipt("f" * 64), operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )
        delayed_child = asyncio.create_task(consume_after_route(delayed_capability))
    release_child.set()
    with pytest.raises(KiteErasureDrillAuthorityError, match="endpoint task"):
        await delayed_child


@pytest.mark.asyncio
async def test_erasure_authority_rejects_malformed_receipts_and_non_erasure_receipts_do_not_accumulate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enabled_home(monkeypatch, tmp_path)
    before = len(kite_evidence_signing._nonce_receipts)
    assert consume_kite_evidence_nonce("d" * 64) is None
    assert len(kite_evidence_signing._nonce_receipts) == before

    with _typed_kite_erasure_endpoint_issuance_scope():
        with pytest.raises(KiteEvidenceSigningError, match="receipt is invalid"):
            _issue_kite_erasure_drill_capability(
                object(), operation=ERASURE_CORE_SNAPSHOT_OPERATION,
            )
        with pytest.raises(KiteErasureDrillAuthorityError, match="operation is invalid"):
            _issue_kite_erasure_drill_capability(
                _receipt("a" * 64), operation="diagnostics",
            )
