"""Native SDK authenticated records retain one domain, agent, identity and expiry."""

import json

import pytest
from kestrel_sdk.security.aead import AEADCipher
from kestrel_sdk.security.encryption import get_agent_key
from kestrel_sdk.security.exceptions import DecryptionError

from kestrel_sovereign.mcp_server.records import (
    MCPAuthorizationRecords,
    MCPRecordKind,
    hash_mcp_secret,
    new_mcp_secret,
)


@pytest.fixture
def records(monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "o" * 32)
    return MCPAuthorizationRecords("did:owned:mcp-records", 1024)


def test_pending_record_uses_the_original_native_sdk_aad_and_cipher(records):
    secret = new_mcp_secret()
    digest = hash_mcp_secret(secret)
    aad = json.dumps(["kestrel-mcp-server-pending-v1", records.agent_id,
        digest, 1600], separators=(",", ":")).encode()
    cipher = AEADCipher(get_agent_key(records.agent_id, "service-keys"))
    original = cipher.encrypt(b'{"state":"original"}', aad=aad).decode("ascii")
    assert records.open(MCPRecordKind.PENDING, digest, 1600, original) == {"state": "original"}
    sealed = records.seal(MCPRecordKind.PENDING, digest, 1600, {"state": "original"})
    assert secret not in sealed and "original" not in sealed
    assert json.loads(cipher.decrypt(sealed.encode("ascii"), aad=aad)) == {"state": "original"}


@pytest.mark.parametrize("change", ["agent", "kind", "identity", "expiry"])
def test_native_record_refuses_authenticated_domain_retargeting(records, change):
    digest = hash_mcp_secret(new_mcp_secret())
    sealed = records.seal(MCPRecordKind.CODE, digest, 1600, {"subject": "owned"})
    selected, kind, expiry = records, MCPRecordKind.CODE, 1600
    if change == "agent":
        selected = MCPAuthorizationRecords("did:owned:other-agent", 1024)
    elif change == "kind":
        kind = MCPRecordKind.ACCESS
    elif change == "identity":
        digest = hash_mcp_secret(new_mcp_secret())
    else:
        expiry += 1
    with pytest.raises(DecryptionError):
        selected.open(kind, digest, expiry, sealed)


def test_current_record_budget_applies_to_both_writes_and_authenticated_reads(records):
    digest = hash_mcp_secret(new_mcp_secret())
    value = {"subject": "owned"}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    bounded = MCPAuthorizationRecords(records.agent_id, len(raw))
    sealed = bounded.seal(MCPRecordKind.GRANT, digest, 1600, value)
    assert bounded.open(MCPRecordKind.GRANT, digest, 1600, sealed) == value
    with pytest.raises(ValueError, match="storage budget"):
        bounded.seal(MCPRecordKind.GRANT, digest, 1600, {"subject": "owned!"})
    with pytest.raises(ValueError, match="current storage budget"):
        MCPAuthorizationRecords(records.agent_id, len(raw) - 1).open(
            MCPRecordKind.GRANT, digest, 1600, sealed)


@pytest.mark.parametrize("secret", [None, "", "a" * 42, "a" * 44, "!" * 43])
def test_only_native_opaque_secret_shapes_have_lookup_hashes(secret):
    with pytest.raises(ValueError, match="authorization secret"):
        hash_mcp_secret(secret)


@pytest.mark.parametrize("agent,budget", [("", 1024), ("bad\nagent", 1024),
    ("did:owned:mcp-records", True), ("did:owned:mcp-records", 0)])
def test_record_codec_requires_explicit_original_identity_and_budget(agent, budget):
    with pytest.raises(ValueError):
        MCPAuthorizationRecords(agent, budget)
