"""
Fail-closed tests for ScriptSigner — Wave 0B (#914).

The historical HMAC fallback used the agent's public DID as the HMAC key,
which made the resulting tag forgeable by anyone who could read the script.
Wave 0B removes the fallback entirely: signing fails closed when keys are
unavailable, and verification rejects every ``hmac:``-prefixed signature
regardless of whether the HMAC math would have 'verified.'

These tests prove the fix and prevent silent regressions.
"""

import asyncio
import base64
import hashlib
import hmac
import os
import tempfile

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from kestrel_sovereign.features.compute.models import ComputeScript, ScriptState
from kestrel_sovereign.features.compute.script_signer import (
    ScriptSigner,
    ScriptSigningKeysUnavailable,
)


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def script():
    return ComputeScript(
        id="fail-closed-001",
        name="fc",
        language="python",
        content="print('payload')",
        purpose="fail-closed test",
        state=ScriptState.DRAFT,
    )


def _signer_with_keys(temp_db, did="did:ethr:0xtest"):
    s = ScriptSigner(did, temp_db)
    s._private_key = ec.generate_private_key(ec.SECP256K1())
    s._public_key = s._private_key.public_key()

    async def _ok():
        return True
    s._load_keys = _ok
    return s


def _canonical_hash(script: ComputeScript) -> str:
    canonical = f"{script.name}|{script.language}|{script.content}|{script.purpose}"
    return hashlib.sha256(canonical.encode()).hexdigest()


# -----------------------------------------------------------------------------
# Sign-or-fail
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sign_without_keys_raises(temp_db, script):
    signer = ScriptSigner("did:key:nokeys", temp_db)
    with pytest.raises(ScriptSigningKeysUnavailable) as excinfo:
        await signer.sign(script)
    # The diagnostic must mention the DID so operators can chase the missing key
    assert "did:key:nokeys" in str(excinfo.value)


@pytest.mark.asyncio
async def test_sign_without_did_raises(temp_db, script):
    signer = ScriptSigner(None, temp_db)
    with pytest.raises(ScriptSigningKeysUnavailable):
        await signer.sign(script)


@pytest.mark.asyncio
async def test_sign_propagates_signing_failure(temp_db, script):
    """If the underlying ECDSA call blows up, sign() must raise — not produce a fallback."""
    signer = _signer_with_keys(temp_db)
    # Replace the private key with something that will fail to sign
    class BrokenKey:
        def sign(self, *a, **kw):
            raise RuntimeError("hardware fault")
    signer._private_key = BrokenKey()
    with pytest.raises(ScriptSigningKeysUnavailable) as excinfo:
        await signer.sign(script)
    assert "hardware fault" in str(excinfo.value)


# -----------------------------------------------------------------------------
# Verify-or-fail
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_rejects_hmac_prefix_even_when_math_verifies(temp_db, script):
    """The forgery vector: anyone with the script + DID can produce a 'valid' hmac: tag.

    Post-Wave-0B verifiers must refuse to look at hmac:-prefixed signatures
    *before* checking the math. This test produces a tag the legacy verifier
    would have accepted, then asserts the new verifier rejects it.
    """
    signer = _signer_with_keys(temp_db, did="did:ethr:0xtest")

    # Forge a tag the way the old fallback did
    content_hash = _canonical_hash(script)
    forged = hmac.new(b"did:ethr:0xtest", content_hash.encode(), hashlib.sha256).digest()
    script.signature = "hmac:" + base64.b64encode(forged).decode()
    script.signed_by = "did:ethr:0xtest"

    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_verify_rejects_hmac_with_unsigned_default(temp_db, script):
    """The literal 'kestrel-unsigned' fallback key path must also be rejected."""
    signer = _signer_with_keys(temp_db)
    content_hash = _canonical_hash(script)
    forged = hmac.new(b"kestrel-unsigned", content_hash.encode(), hashlib.sha256).digest()
    script.signature = "hmac:" + base64.b64encode(forged).decode()
    script.signed_by = None
    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_verify_genuine_ecdsa_signature(temp_db, script):
    signer = _signer_with_keys(temp_db)
    sig = await signer.sign(script)
    script.signature = sig
    assert sig.startswith("ecdsa:")
    assert await signer.verify(script) is True


@pytest.mark.asyncio
async def test_verify_rejects_garbled_ecdsa_signature(temp_db, script):
    signer = _signer_with_keys(temp_db)
    script.signature = "ecdsa:" + base64.b64encode(b"\x00" * 70).decode()
    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_verify_rejects_unknown_prefix(temp_db, script):
    signer = _signer_with_keys(temp_db)
    script.signature = "future:" + base64.b64encode(b"placeholder").decode()
    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_verify_rejects_empty_signature(temp_db, script):
    signer = _signer_with_keys(temp_db)
    script.signature = ""
    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_verify_rejects_wrong_did_signature(temp_db, script):
    """A signature produced under DID A must not verify under DID B's keys."""
    signer_a = _signer_with_keys(temp_db, did="did:ethr:0xAAAA")
    signer_b = _signer_with_keys(temp_db, did="did:ethr:0xBBBB")

    sig = await signer_a.sign(script)
    script.signature = sig

    # signer_b has a different keypair → must reject
    assert await signer_b.verify(script) is False


@pytest.mark.asyncio
async def test_verify_detects_content_tampering(temp_db, script):
    signer = _signer_with_keys(temp_db)
    script.signature = await signer.sign(script)
    # Tamper after signing
    script.content = "print('malicious')"
    assert await signer.verify(script) is False


# -----------------------------------------------------------------------------
# Migration / production-script sweep
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_existing_hmac_scripts_treated_as_unsigned(temp_db, script):
    """Any script in a database carrying an hmac: signature must fail verify.

    Operators encountering this on the migration sweep should re-sign with
    ECDSA keys (which by now exist for every active agent) or invalidate.
    """
    signer = _signer_with_keys(temp_db)
    # Simulate a legacy DB row
    script.signature = "hmac:" + base64.b64encode(b"\x42" * 32).decode()
    script.signed_by = "did:ethr:0xlegacy"
    assert await signer.verify(script) is False
