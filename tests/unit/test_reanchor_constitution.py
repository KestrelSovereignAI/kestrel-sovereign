"""Unit tests for !reanchor-constitution command."""
import pytest
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.agent.constitution import ConstitutionMixin


def _make_agent(stored_hash="oldhash", safe_mode=False):
    """Create a mock agent with ConstitutionMixin methods bound."""
    agent = MagicMock(spec=KestrelAgent)
    agent._safe_mode = safe_mode
    agent._get_timestamp = MagicMock(return_value="2026-04-06T00:00:00Z")
    agent.extension = None

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
    agent._get_governing_constitution = ConstitutionMixin._get_governing_constitution.__get__(
        agent, KestrelAgent
    )
    return agent, node


FAKE_CONSTITUTION = b"# Kestrel Constitution v2\n\nAmended content here.\n"
FAKE_HASH = hashlib.sha256(FAKE_CONSTITUTION).hexdigest()


# --- Authorization: expected_hash required ---

@pytest.mark.asyncio
async def test_reanchor_rejects_missing_hash():
    agent, _ = _make_agent()
    result = await agent.reanchor_constitution()
    assert "error" in result.lower()
    assert "expected hash required" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_rejects_short_hash():
    agent, _ = _make_agent()
    result = await agent.reanchor_constitution(expected_hash="abc")
    assert "error" in result.lower()
    assert "min 8" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_rejects_wrong_hash():
    agent, _ = _make_agent()
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value=FAKE_CONSTITUTION)
        result = await agent.reanchor_constitution(expected_hash="deadbeef")
    assert "error" in result.lower()
    assert "hash mismatch" in result.lower()
    agent.storage.store_file.assert_not_called()


# --- Happy path ---

@pytest.mark.asyncio
async def test_reanchor_succeeds_with_correct_hash():
    agent, node = _make_agent(stored_hash="oldhash", safe_mode=False)
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value=FAKE_CONSTITUTION)
        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8], authorization="sovereign_api_key",
        )
    assert "re-anchored successfully" in result.lower()
    assert node.properties["constitution_hash"] == FAKE_HASH
    assert node.properties["constitution_reanchor"]["authorization"] == "sovereign_api_key"


@pytest.mark.asyncio
async def test_reanchor_accepts_full_hash():
    agent, node = _make_agent(stored_hash="oldhash")
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value=FAKE_CONSTITUTION)
        result = await agent.reanchor_constitution(expected_hash=FAKE_HASH)
    assert "re-anchored successfully" in result.lower()


# --- Safe mode NOT auto-exited ---

@pytest.mark.asyncio
async def test_reanchor_does_not_exit_safe_mode():
    agent, node = _make_agent(stored_hash="oldhash", safe_mode=True)
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value=FAKE_CONSTITUTION)
        result = await agent.reanchor_constitution(expected_hash=FAKE_HASH[:8])
    assert agent._safe_mode is True
    assert "safe mode" in result.lower()


# --- Edge cases ---

@pytest.mark.asyncio
async def test_reanchor_noop_when_already_current():
    agent, node = _make_agent(stored_hash=FAKE_HASH)
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value=FAKE_CONSTITUTION)
        result = await agent.reanchor_constitution(expected_hash=FAKE_HASH[:8])
    assert "already anchored" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_fails_when_no_identity_node():
    agent, _ = _make_agent()
    agent.storage.get_node = AsyncMock(return_value=None)
    result = await agent.reanchor_constitution(expected_hash=FAKE_HASH[:8])
    assert "error" in result.lower()
    assert "identity node" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_fails_when_no_file_on_disk():
    agent, _ = _make_agent()
    with patch("builtins.open", side_effect=FileNotFoundError):
        result = await agent.reanchor_constitution(expected_hash=FAKE_HASH[:8])
    assert "error" in result.lower()
    assert "no constitution file" in result.lower()


# --- Governance source: post-reanchor reads from storage, not disk ---

@pytest.mark.asyncio
async def test_governing_constitution_reads_from_storage_after_reanchor():
    agent, node = _make_agent(stored_hash="oldhash")
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    agent.agent_id = "test-agent"

    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value=FAKE_CONSTITUTION)
        result = await agent.reanchor_constitution(expected_hash=FAKE_HASH[:8])

    assert "re-anchored successfully" in result.lower()
    agent.storage.retrieve_file = AsyncMock(return_value=b"stored constitution content")
    constitution = await agent._get_governing_constitution()
    agent.storage.retrieve_file.assert_called_once_with(FAKE_HASH)
    assert constitution == "stored constitution content"


@pytest.mark.asyncio
async def test_governing_constitution_does_not_read_disk_when_hash_exists():
    agent, node = _make_agent(stored_hash=FAKE_HASH)
    agent.agent_id = "test-agent"
    agent.storage.retrieve_file = AsyncMock(return_value=FAKE_CONSTITUTION)

    with patch("builtins.open", create=True) as mock_open:
        constitution = await agent._get_governing_constitution()

    mock_open.assert_not_called()
    agent.storage.retrieve_file.assert_called_once_with(FAKE_HASH)
    assert constitution == FAKE_CONSTITUTION.decode("utf-8")
