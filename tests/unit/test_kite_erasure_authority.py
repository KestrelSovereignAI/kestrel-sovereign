"""Authority-boundary tests for the loopback Kite core-erasure drill."""

from __future__ import annotations

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
    with pytest.raises(KiteErasureDrillAuthorityError, match="typed endpoint"):
        _issue_kite_erasure_drill_capability(
            _receipt("a" * 64), operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )


def test_erasure_authority_is_exactly_bound_burned_and_route_scoped(
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
    with pytest.raises(KiteErasureDrillAuthorityError, match="typed endpoint"):
        _consume_kite_erasure_drill_capability(
            capability, expected_operation=ERASURE_CORE_SNAPSHOT_OPERATION,
        )


def test_erasure_authority_rejects_malformed_receipts_and_non_erasure_receipts_do_not_accumulate(
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
                _receipt("e" * 64), operation="diagnostics",
            )
