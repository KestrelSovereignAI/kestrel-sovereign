"""Unit tests for !reanchor-constitution command."""
import json
import pytest
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.amendment_artifact import (
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


def _make_agent(stored_hash="oldhash", safe_mode=False):
    """Create a mock agent with ConstitutionMixin methods bound."""
    agent = MagicMock(spec=KestrelAgent)
    agent._safe_mode = safe_mode
    agent._get_timestamp = MagicMock(return_value="2026-04-06T00:00:00Z")
    agent.extension = None
    agent.identity = SimpleNamespace(legacy_did_document=ROOT_DID_DOCUMENT)

    node = MagicMock()
    node.properties = {"constitution_hash": stored_hash}
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
