"""Unit tests for !reanchor-constitution command."""
import pytest
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.agent.constitution import ConstitutionMixin


def _make_agent(stored_hash="oldhash", safe_mode=False):
    """Create a mock agent with ConstitutionMixin.reanchor_constitution bound."""
    agent = MagicMock(spec=KestrelAgent)
    agent._safe_mode = safe_mode
    agent._get_timestamp = MagicMock(return_value="2026-04-06T00:00:00Z")

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
    return agent, node


FAKE_CONSTITUTION = b"# Kestrel Constitution v2\n\nAmended content here.\n"
FAKE_HASH = hashlib.sha256(FAKE_CONSTITUTION).hexdigest()


# --- Authorization: expected_hash required ---

@pytest.mark.asyncio
async def test_reanchor_rejects_missing_hash():
    """Re-anchor refuses without expected_hash argument."""
    agent, _ = _make_agent()
    result = await agent.reanchor_constitution()
    assert "error" in result.lower()
    assert "expected hash required" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_rejects_short_hash():
    """Re-anchor refuses hash prefix shorter than 8 characters."""
    agent, _ = _make_agent()
    result = await agent.reanchor_constitution(expected_hash="abc")
    assert "error" in result.lower()
    assert "min 8" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_rejects_wrong_hash():
    """Re-anchor refuses when expected hash doesn't match file on disk."""
    agent, _ = _make_agent()

    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value=FAKE_CONSTITUTION)

        result = await agent.reanchor_constitution(expected_hash="deadbeef")

    assert "error" in result.lower()
    assert "hash mismatch" in result.lower()
    agent.storage.store_file.assert_not_called()


# --- Happy path with valid hash ---

@pytest.mark.asyncio
async def test_reanchor_succeeds_with_correct_hash():
    """Re-anchor stores new constitution when expected hash matches."""
    agent, node = _make_agent(stored_hash="oldhash", safe_mode=False)
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value=FAKE_CONSTITUTION)

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            authorization="admin_command",
        )

    assert "re-anchored successfully" in result.lower()
    assert FAKE_HASH[:16] in result
    assert "admin_command" in result
    assert node.properties["constitution_hash"] == FAKE_HASH
    assert node.properties["constitution_reanchor"]["old_hash"] == "oldhash"
    assert node.properties["constitution_reanchor"]["new_hash"] == FAKE_HASH
    assert node.properties["constitution_reanchor"]["authorization"] == "admin_command"
    assert node.properties["constitution_reanchor"]["expected_hash_prefix"] == FAKE_HASH[:8]
    agent.storage.store_file.assert_called_once_with(FAKE_CONSTITUTION, "KESTREL_CONSTITUTION.md")
    agent.privacy_agent.add_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_reanchor_accepts_full_hash():
    """Re-anchor works with full hash, not just prefix."""
    agent, node = _make_agent(stored_hash="oldhash")
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value=FAKE_CONSTITUTION)

        result = await agent.reanchor_constitution(expected_hash=FAKE_HASH)

    assert "re-anchored successfully" in result.lower()


# --- Safe mode is NOT auto-exited ---

@pytest.mark.asyncio
async def test_reanchor_does_not_exit_safe_mode():
    """Re-anchor updates hash but leaves safe mode active."""
    agent, node = _make_agent(stored_hash="oldhash", safe_mode=True)
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = MagicMock(return_value=FAKE_CONSTITUTION)

        result = await agent.reanchor_constitution(expected_hash=FAKE_HASH[:8])

    assert agent._safe_mode is True
    assert "safe mode" in result.lower()
    assert node.properties["constitution_hash"] == FAKE_HASH


# --- Edge cases ---

@pytest.mark.asyncio
async def test_reanchor_noop_when_already_current():
    """Re-anchor is a no-op if constitution hasn't changed."""
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
    """Re-anchor fails gracefully when agent node is missing."""
    agent, _ = _make_agent()
    agent.storage.get_node = AsyncMock(return_value=None)

    result = await agent.reanchor_constitution(expected_hash=FAKE_HASH[:8])

    assert "error" in result.lower()
    assert "identity node" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_fails_when_no_file_on_disk():
    """Re-anchor fails gracefully when constitution file is missing."""
    agent, _ = _make_agent()

    with patch("builtins.open", side_effect=FileNotFoundError):
        result = await agent.reanchor_constitution(expected_hash=FAKE_HASH[:8])

    assert "error" in result.lower()
    assert "no constitution file" in result.lower()
