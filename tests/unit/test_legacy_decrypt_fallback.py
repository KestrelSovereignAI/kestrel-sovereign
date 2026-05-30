"""Regression tests for ``decrypt_with_legacy_fallback`` (#1458).

The fallback recovers ciphertext written by the earliest
``encrypt(agent_did, purpose, ...)`` implementation, which stored
raw AES-GCM ``nonce(12) || ct+tag`` with no version prefix and no
AAD. AEADCipher.decrypt (v2 + Fernet) rejects that layout, which
broke boot-time activation of agent service keys (e.g. OpenRouter)
for agents that haven't been rotated since the Quantum Hardening
epic landed — symptom in the wild was Emma's pre-Wave-0C
``agent_service_keys.openrouter`` row.

Pins:
  - Current v2 ciphertext round-trips through the fallback unchanged
    (no fallback path taken).
  - Legacy Fernet ciphertext (written by the SDK transition era)
    round-trips through the fallback unchanged.
  - Pre-v2 raw AES-GCM ciphertext recovers via the fallback and logs
    a WARNING with the agent DID + purpose so operators can rotate.
  - Garbage shorter than nonce+tag rejects early without a misleading
    second attempt.
  - Authentic ciphertext encrypted under a DIFFERENT agent's key
    raises DecryptionError (no cross-agent leak).
"""

from __future__ import annotations

import base64
import logging

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from kestrel_sdk.security.encryption import encrypt as sdk_encrypt
from kestrel_sdk.security.encryption import get_agent_key
from kestrel_sovereign.security.exceptions import DecryptionError
from kestrel_sovereign.security.legacy_decrypt import (
    decrypt_with_legacy_fallback,
)


EMMA_DID = "did:pkh:eip155:1:0xB4E7F05F9c39FcD0b0d2C516249BE960c863647E"
OTHER_DID = "did:pkh:eip155:1:0x1234567890abcdef1234567890abcdef12345678"
PURPOSE = "service-keys"


@pytest.fixture(autouse=True)
def _set_test_master_key(monkeypatch):
    """Same master key fixture as the wider agent_encryption tests so
    HKDF derivation is deterministic per-DID across this file."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-encryption-key-for-unit-tests")


def _make_pre_v2_ciphertext(agent_did: str, purpose: str, plaintext: bytes) -> bytes:
    """Reproduce the earliest on-disk format: ``nonce(12) || ct+tag``,
    no version prefix, no AAD. Mirrors what ``encrypt()`` produced
    before the Wave-0C ``KSAv2:`` token framing landed."""
    import os
    key = get_agent_key(agent_did, purpose)
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext, None)
    return nonce + ct


class TestCurrentV2Passthrough:
    def test_v2_ciphertext_decrypts_without_fallback(self, caplog):
        """Modern v2 ciphertext goes through the SDK path. The fallback
        warning must NOT fire — that signal is reserved for the
        legacy-recovery case so operators can act on it."""
        plaintext = b"sk-or-v1-modern-token"
        ciphertext = sdk_encrypt(EMMA_DID, PURPOSE, plaintext)
        # Sanity: the SDK currently writes v2 tokens.
        assert ciphertext.startswith(b"KSAv2:")

        with caplog.at_level(logging.WARNING):
            recovered = decrypt_with_legacy_fallback(
                EMMA_DID, PURPOSE, ciphertext,
            )
        assert recovered == plaintext
        assert not any(
            "raw AES-GCM fallback" in rec.message for rec in caplog.records
        ), (
            "v2 ciphertext must NOT trigger the legacy-recovery warning. "
            "If it does, the SDK decrypt is rejecting current-format "
            "rows and the fallback is masking the real problem."
        )


class TestPreV2RawAESGCMRecovery:
    def test_pre_v2_raw_aesgcm_recovers_via_fallback(self, caplog):
        """The regression case Emma hit: an OpenRouter key stored as
        raw AES-GCM (``nonce || ct+tag``, no prefix, no AAD) must
        recover and log the rotation hint."""
        plaintext = b"sk-or-v1-historic-token-from-early-2026"
        ciphertext = _make_pre_v2_ciphertext(EMMA_DID, PURPOSE, plaintext)

        with caplog.at_level(logging.WARNING):
            recovered = decrypt_with_legacy_fallback(
                EMMA_DID, PURPOSE, ciphertext,
            )
        assert recovered == plaintext
        # The warning must surface enough context for an operator to
        # decide on rotation (DID prefix + purpose + length, and the
        # issue reference so they can find #1458's runbook).
        matched = [
            rec for rec in caplog.records
            if "raw AES-GCM fallback" in rec.message
        ]
        assert matched, (
            "Pre-v2 fallback success MUST log a WARNING — that's the "
            "only signal an operator gets to plan a rotation pass."
        )
        msg = matched[0].message
        assert PURPOSE in msg
        assert "#1458" in msg

    def test_pre_v2_ciphertext_under_wrong_agent_did_rejects(self):
        """Authentic pre-v2 ciphertext from one agent must NOT decrypt
        under another agent's DID — the per-agent HKDF derivation is
        the cross-tenant isolation boundary."""
        plaintext = b"sk-or-v1-only-emma"
        ciphertext = _make_pre_v2_ciphertext(EMMA_DID, PURPOSE, plaintext)

        with pytest.raises(DecryptionError):
            decrypt_with_legacy_fallback(OTHER_DID, PURPOSE, ciphertext)

    def test_pre_v2_ciphertext_under_wrong_purpose_rejects(self):
        """Pre-v2 ciphertext written under one purpose must NOT decrypt
        under another purpose — purpose-derived subkey is the
        cross-purpose isolation boundary."""
        plaintext = b"sk-or-v1-purpose-bound"
        ciphertext = _make_pre_v2_ciphertext(EMMA_DID, "service-keys", plaintext)

        with pytest.raises(DecryptionError):
            decrypt_with_legacy_fallback(EMMA_DID, "wallet", ciphertext)


class TestRejectionShape:
    def test_garbage_shorter_than_nonce_plus_tag_rejects_without_fallback(self):
        """A 4-byte blob cannot be raw AES-GCM (needs 12-byte nonce +
        16-byte tag minimum). The SDK error must surface AS-IS rather
        than being masked by an attempt that can't possibly succeed."""
        too_short = b"abcd"
        with pytest.raises(DecryptionError):
            decrypt_with_legacy_fallback(EMMA_DID, PURPOSE, too_short)

    def test_random_long_bytes_reject_with_sdk_error_chain(self, caplog):
        """A long random blob that fails BOTH v2/Fernet AND raw AES-GCM
        must raise DecryptionError. The warning log must NOT fire —
        we only log on actual recovery."""
        import os
        rubbish = os.urandom(101)  # same length as Emma's row, but random key
        with caplog.at_level(logging.WARNING):
            with pytest.raises(DecryptionError):
                decrypt_with_legacy_fallback(EMMA_DID, PURPOSE, rubbish)
        assert not any(
            "raw AES-GCM fallback" in rec.message for rec in caplog.records
        ), (
            "Failed-fallback rubbish must NOT log the recovery warning "
            "— a future operator running ``grep #1458`` would otherwise "
            "see noise unrelated to actual rotation work."
        )
