"""Unit tests for !reanchor-constitution command."""
import json
import pytest
import hashlib
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.amendment_artifact import (
    MAX_REANCHOR_ARTIFACT_BYTES,
    build_legacy_signed_reanchor_artifact,
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite


_SUITE = Secp256k1Suite()
ROOT_KEYPAIR = _SUITE.generate_keypair()
ROOT_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000000587"
ROOT_DID_DOCUMENT = did_document_from_legacy_public_key(
    ROOT_DID,
    ROOT_KEYPAIR.public_key,
)
AGENT_KEYPAIR = _SUITE.generate_keypair()
AGENT_DID = "did:pkh:eip155:1:0x000000000000000000000000000000000000a587"
AGENT_DID_DOCUMENT = did_document_from_legacy_public_key(
    AGENT_DID,
    AGENT_KEYPAIR.public_key,
)


@pytest.fixture(autouse=True)
def _operator_pinned_root(tmp_path, monkeypatch):
    """Every authorization test uses a root outside the mocked graph DB."""
    root_path = tmp_path / "operator-sovereign-root.did.json"
    root_path.write_text(json.dumps(ROOT_DID_DOCUMENT), encoding="utf-8")
    monkeypatch.setenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH", str(root_path))
    return root_path


def _make_agent(stored_hash="oldhash", safe_mode=False):
    """Create a mock agent with ConstitutionMixin methods bound."""
    agent = MagicMock(spec=KestrelAgent)
    agent._safe_mode = safe_mode
    agent._get_timestamp = MagicMock(return_value="2026-04-06T00:00:00Z")
    agent.extension = None
    agent.agent_id = AGENT_DID
    agent.identity = SimpleNamespace(
        legacy_did=AGENT_DID,
        signing_did=AGENT_DID,
        legacy_did_document=AGENT_DID_DOCUMENT,
    )
    agent._sovereign_trust_root_path = None

    node = MagicMock()
    node.properties = {
        "constitution_hash": stored_hash,
        "sovereign_root_did_document": ROOT_DID_DOCUMENT,
    }
    agent.storage = AsyncMock()
    agent.storage.get_node = AsyncMock(return_value=node)
    agent.storage.store_file = AsyncMock()
    agent.storage.add_node = AsyncMock()
    agent.privacy_agent = MagicMock()
    agent.privacy_agent.add_conversation = AsyncMock()

    agent.reanchor_constitution = ConstitutionMixin.reanchor_constitution.__get__(
        agent, KestrelAgent
    )
    agent._trusted_sovereign_did_document = (
        ConstitutionMixin._trusted_sovereign_did_document.__get__(agent, KestrelAgent)
    )
    agent._agent_signing_dids = ConstitutionMixin._agent_signing_dids.__get__(
        agent, KestrelAgent
    )
    agent._get_governing_constitution = ConstitutionMixin._get_governing_constitution.__get__(
        agent, KestrelAgent
    )
    return agent, node


FAKE_CONSTITUTION = b"# Kestrel Constitution v2\n\nAmended content here.\n"
FAKE_HASH = hashlib.sha256(FAKE_CONSTITUTION).hexdigest()


def _write_artifact(tmp_path, constitution_hash=FAKE_HASH, keypair=ROOT_KEYPAIR, did=ROOT_DID):
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=did,
        constitution_sha256=constitution_hash,
        private_key=keypair.private_key,
        created_at="2026-04-06T00:00:00Z",
        reason="unit test",
    )
    path = tmp_path / "constitution-reanchor.signed.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _open_handles(*contents):
    handles = []
    for content in contents:
        handle = MagicMock()
        handle.read.return_value = content
        handle.__enter__.return_value = handle
        handle.__exit__.return_value = False
        handles.append(handle)
    return handles


# --- Authorization: expected_hash required ---

@pytest.mark.asyncio
async def test_reanchor_rejects_missing_signed_artifact():
    """Re-anchor refuses without a signed amendment artifact."""
    agent, _ = _make_agent()
    result = await agent.reanchor_constitution()
    assert "error" in result.lower()
    assert "signed amendment artifact required" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_rejects_short_hash(tmp_path):
    """Re-anchor refuses hash prefix shorter than 8 characters."""
    agent, _ = _make_agent()
    artifact_path = _write_artifact(tmp_path)
    result = await agent.reanchor_constitution(
        expected_hash="abc",
        amendment_artifact_path=str(artifact_path),
    )
    assert "error" in result.lower()
    assert "at least 8" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_rejects_wrong_hash(tmp_path):
    """Re-anchor refuses when expected hash doesn't match file on disk."""
    agent, _ = _make_agent()
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash="deadbeef",
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "hash mismatch" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_rejects_oversized_artifact_before_storage(tmp_path):
    agent, _ = _make_agent()
    artifact_path = tmp_path / "oversized.signed.json"
    artifact_path.write_bytes(b"x" * (MAX_REANCHOR_ARTIFACT_BYTES + 1))

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION)
        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "exceeds" in result.lower()
    agent.storage.store_file.assert_not_called()


# --- Happy path with valid hash ---

@pytest.mark.asyncio
async def test_reanchor_succeeds_with_sovereign_signed_artifact(tmp_path):
    """Re-anchor stores new constitution when expected hash matches."""
    agent, node = _make_agent(stored_hash="oldhash", safe_mode=False)
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            authorization="admin_command",
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower()
    assert FAKE_HASH[:16] in result
    assert "admin_command" in result
    assert node.properties["constitution_hash"] == FAKE_HASH
    assert node.properties["constitution_reanchor"]["old_hash"] == "oldhash"
    assert node.properties["constitution_reanchor"]["new_hash"] == FAKE_HASH
    assert node.properties["constitution_reanchor"]["authorization"] == "admin_command"
    assert node.properties["constitution_reanchor"]["expected_hash_prefix"] == FAKE_HASH[:8]
    assert node.properties["constitution_reanchor"]["signed_artifact_hash"] == FAKE_HASH
    assert node.properties["constitution_reanchor"]["signed_artifact_signer"] == ROOT_DID
    assert agent.storage.store_file.call_count == 2
    agent.storage.store_file.assert_any_call(FAKE_CONSTITUTION, "KESTREL_CONSTITUTION.md")
    agent.privacy_agent.add_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_reanchor_rejects_unpinned_controller_root(tmp_path, monkeypatch):
    """A controller resolver is discovery, not an operator-pinned authority."""
    agent, node = _make_agent(stored_hash="oldhash")
    agent.identity.legacy_did_document = {
        **AGENT_DID_DOCUMENT,
        "controller": ROOT_DID,
    }
    node.properties = {"constitution_hash": "oldhash"}
    agent.a2a_did_resolver = lambda did: ROOT_DID_DOCUMENT if did == ROOT_DID else None
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)
    monkeypatch.delenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH")

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "no external sovereign trust root is configured" in result.lower()
    assert "operator-owned json file" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_rejects_wrongly_signed_artifact(tmp_path):
    """Re-anchor refuses artifacts not signed by the trusted Sovereign root key."""
    agent, _ = _make_agent(stored_hash="oldhash")
    other_keypair = _SUITE.generate_keypair()
    artifact_path = _write_artifact(tmp_path, keypair=other_keypair)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "signed amendment verification failed" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_rejects_agent_owned_signature(tmp_path):
    """An agent-owned legacy key must never authorize its own re-anchor."""
    agent, _ = _make_agent(stored_hash="oldhash")
    artifact_path = _write_artifact(
        tmp_path,
        keypair=AGENT_KEYPAIR,
        did=AGENT_DID,
    )

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "signed amendment verification failed" in result.lower()
    assert "not trusted sovereign did" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_rejects_self_identity_as_trust_anchor(tmp_path):
    """The agent's DID document is not a Sovereign root trust source."""
    agent, node = _make_agent(stored_hash="oldhash")
    node.properties["sovereign_root_did_document"] = AGENT_DID_DOCUMENT
    artifact_path = _write_artifact(
        tmp_path,
        keypair=AGENT_KEYPAIR,
        did=AGENT_DID,
    )

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "not trusted sovereign did" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_db_injected_root_and_hash_cannot_authorize_reanchor(tmp_path):
    """A DB writer cannot replace the root and self-sign a hostile reanchor."""
    attacker_keypair = _SUITE.generate_keypair()
    attacker_did = (
        "did:pkh:eip155:1:0x000000000000000000000000000000000000bad0"
    )
    attacker_doc = did_document_from_legacy_public_key(
        attacker_did,
        attacker_keypair.public_key,
    )
    agent, node = _make_agent(stored_hash="attacker-overwrote-this-hash", safe_mode=True)
    node.properties.update(
        {
            "sovereign_root_did_document": attacker_doc,
            "trusted_sovereign_did_document": attacker_doc,
            "sovereign_root_did": attacker_did,
            "sovereign_root_public_key_hex": attacker_doc["publicKey"][0][
                "publicKeyHex"
            ],
            "emancipation_contract": {"enabled": False},
        }
    )
    before = deepcopy(node.properties)
    artifact_path = _write_artifact(
        tmp_path,
        keypair=attacker_keypair,
        did=attacker_did,
    )

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION)
        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "signed amendment verification failed" in result.lower()
    assert "not trusted sovereign did" in result.lower()
    assert node.properties == before
    assert agent._safe_mode is True
    agent.storage.store_file.assert_not_called()
    agent.storage.add_node.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_accepts_full_hash(tmp_path):
    """Re-anchor works with full hash, not just prefix."""
    agent, node = _make_agent(stored_hash="oldhash")
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH,
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower()


# --- Safe mode is NOT auto-exited ---

@pytest.mark.asyncio
async def test_reanchor_does_not_exit_safe_mode(tmp_path):
    """Re-anchor updates hash but leaves safe mode active."""
    agent, node = _make_agent(stored_hash="oldhash", safe_mode=True)
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert agent._safe_mode is True
    assert "safe mode" in result.lower()
    assert node.properties["constitution_hash"] == FAKE_HASH


# --- Edge cases ---

@pytest.mark.asyncio
async def test_reanchor_noop_when_already_current(tmp_path):
    """Re-anchor is a no-op if constitution hasn't changed."""
    agent, node = _make_agent(stored_hash=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "already anchored" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_fails_when_no_identity_node():
    """Re-anchor fails gracefully when agent node is missing."""
    agent, _ = _make_agent()
    agent.storage.get_node = AsyncMock(return_value=None)

    result = await agent.reanchor_constitution(
        expected_hash=FAKE_HASH[:8],
        amendment_artifact_path="artifact.json",
    )

    assert "error" in result.lower()
    assert "identity node" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_fails_when_no_file_on_disk(tmp_path):
    """Re-anchor fails gracefully when constitution file is missing."""
    agent, _ = _make_agent()
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", side_effect=FileNotFoundError):
        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "no constitution file" in result.lower()


# --- Governance source: post-reanchor reads from storage, not disk ---

@pytest.mark.asyncio
async def test_governing_constitution_reads_from_storage_after_reanchor(tmp_path):
    """After reanchor, _get_governing_constitution reads from anchored storage, not disk."""
    agent, node = _make_agent(stored_hash="oldhash")
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    agent.agent_id = "test-agent"
    artifact_path = _write_artifact(tmp_path)

    # Step 1: reanchor so node gets the new hash
    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower()
    assert node.properties["constitution_hash"] == FAKE_HASH

    # Step 2: _get_governing_constitution should retrieve from storage using the new hash
    agent.storage.retrieve_file = AsyncMock(return_value=b"stored constitution content")

    constitution = await agent._get_governing_constitution()

    agent.storage.retrieve_file.assert_called_once_with(FAKE_HASH)
    assert constitution == "stored constitution content"


@pytest.mark.asyncio
async def test_governing_constitution_does_not_read_disk_when_hash_exists():
    """_get_governing_constitution never opens disk files when a hash is anchored."""
    agent, node = _make_agent(stored_hash=FAKE_HASH)
    agent.agent_id = "test-agent"
    agent.storage.retrieve_file = AsyncMock(return_value=FAKE_CONSTITUTION)

    with patch("builtins.open", create=True) as mock_open:
        constitution = await agent._get_governing_constitution()

    mock_open.assert_not_called()
    agent.storage.retrieve_file.assert_called_once_with(FAKE_HASH)
    assert constitution == FAKE_CONSTITUTION.decode("utf-8")
