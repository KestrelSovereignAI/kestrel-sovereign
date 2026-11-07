"""Borrowed native storage keeps its caller's original encryption lifetime."""

import os

from cryptography.fernet import Fernet
import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_storage import AsyncStorage


@pytest_asyncio.fixture(autouse=True)
async def caller_encrypted_storage(tmp_path, monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv('KESTREL_DATA_KEY', key)
    storage = AsyncStorage(str(tmp_path / 'caller.db'), agent_id='did:test:borrowed-key')
    await storage.initialize()
    content = b'Original encrypted caller data'
    digest = await storage.store_file(content, 'owned-caller.txt')
    try:
        yield storage, digest, content, key
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_native_agent_reads_original_encrypted_borrowed_storage(
    caller_encrypted_storage, native_tool_effect_agent,
):
    storage, digest, content, key = caller_encrypted_storage
    agent = await native_tool_effect_agent(
        did=storage.agent_id, borrowed_storage=storage,
    )
    assert agent.tool_effect_store.database is storage.db
    assert await agent._raw_storage.retrieve_file(digest) == content
    assert os.environ['KESTREL_DATA_KEY'] == key
    assert not agent._runtime_publication_ready.is_set()


@pytest.mark.asyncio
async def test_native_agent_storage_initialization_keeps_its_original_journal_database(
    native_tool_effect_agent,
):
    agent = await native_tool_effect_agent(did='did:test:one-native-primary')
    database = agent.tool_effect_store.database
    await agent._raw_storage.initialize()
    assert agent._raw_storage.db is database
    assert agent._raw_storage.agent_id == agent.did
    content = b'One canonical native primary database'
    digest = await agent._raw_storage.store_file(content, 'owned-native-primary.txt')
    assert await agent._raw_storage.retrieve_file(digest) == content
    await agent._initialize_tool_effect_store()
    assert agent.tool_effect_store.database is database


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid', ['uninitialized', 'foreign-did'])
async def test_native_agent_refuses_incompatible_borrowed_storage(
    caller_encrypted_storage, native_tool_effect_agent, invalid,
):
    storage, digest, content, _key = caller_encrypted_storage
    supplied = (AsyncStorage(backend=storage.db.backend, agent_id=storage.agent_id)
        if invalid == 'uninitialized' else storage)
    did = storage.agent_id if invalid == 'uninitialized' else 'did:test:foreign'
    with pytest.raises(ValueError, match='initialized for the Agent DID'):
        await native_tool_effect_agent(did=did, borrowed_storage=supplied)
    assert storage.db.backend.is_connected
    assert await storage.retrieve_file(digest) == content
