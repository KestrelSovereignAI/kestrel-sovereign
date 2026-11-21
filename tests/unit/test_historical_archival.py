"""Real verify-only SPHINCS+ compatibility for pre-August rotation records."""
from dataclasses import replace
import os
from pathlib import Path
import sys

import pytest

from kestrel_sovereign.constitution.historical_archival import (
    ENV_PYTHON, HistoricalVerifierError, verify_historical_archival_signature,
)
from kestrel_sovereign.identity.succession import SuccessionStatement
from kestrel_sovereign.security.crypto_suite import ALG_SLH_DSA_SHA2_128S, get_suite
from kestrel_sovereign.security.multikey import public_key_to_multibase


@pytest.fixture
def archived():
    # Public key followed by the genuine pqcrypto 0.4.0 signature over the
    # canonical payload below, generated once with a disposable key.
    material = (Path(__file__).resolve().parents[1] / "fixtures" / "archival"
                / "sphincs-sha2-128s-simple.v0.4.0.bin").read_bytes()
    multibase = public_key_to_multibase(get_suite(ALG_SLH_DSA_SHA2_128S), material[:32])
    statement = SuccessionStatement(
        predecessor_did="did:example:predecessor", successor_did="did:example:successor",
        effective_from="2026-05-05T00:00:00+00:00", reason="Synthetic archival verification fixture",
        archival_verification_method={"id": "did:example:successor#archival-1", "publicKeyMultibase": multibase},
        archival_signature={"kid": "archival-1", "alg": ALG_SLH_DSA_SHA2_128S, "sig": material[32:].hex()},
    )
    return statement, multibase


def test_historical_mode_requires_the_actual_algorithm(archived, monkeypatch):
    statement, pin = archived
    monkeypatch.delenv(ENV_PYTHON, raising=False)
    with pytest.raises(HistoricalVerifierError, match="absolute path"):
        verify_historical_archival_signature(statement, trusted_archival_multibase=pin)
    monkeypatch.setenv(ENV_PYTHON, sys.executable)
    with pytest.raises(HistoricalVerifierError, match="pqcrypto 0.4.0"):
        verify_historical_archival_signature(statement, trusted_archival_multibase=pin)


def test_historical_mode_refuses_wrong_pin_kid_algorithm_and_length(archived):
    statement, pin = archived
    assert not verify_historical_archival_signature(statement, trusted_archival_multibase=pin + "x")
    for updates in ({"kid": "other"}, {"alg": "ed25519"}, {"sig": "00"}, {"sig": "x" * 15712}):
        changed = replace(statement, archival_signature={**statement.archival_signature, **updates})
        assert not verify_historical_archival_signature(changed, trusted_archival_multibase=pin)


@pytest.mark.skipif(not os.environ.get(ENV_PYTHON), reason="requires isolated historical verifier")
def test_original_signature_verifies_only_over_the_signed_statement(archived):
    statement, pin = archived
    assert verify_historical_archival_signature(statement, trusted_archival_multibase=pin)
    assert not verify_historical_archival_signature(replace(statement, reason="Changed"), trusted_archival_multibase=pin)
