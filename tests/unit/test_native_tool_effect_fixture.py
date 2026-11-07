"""Borrowed native storage keeps its caller's original encryption lifetime."""

import os
from unittest.mock import patch

from cryptography.fernet import Fernet
import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.async_file_store import AsyncFileStore


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
    reader = AsyncFileStore(storage.db, agent_id=storage.agent_id)
    assert await reader.retrieve_file(digest) == content
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


@pytest.mark.asyncio
async def test_native_fixture_retires_its_patches_before_the_outer_provider_fixture(tmp_path, monkeypatch):
    from kestrel_sovereign.llm import service
    from kestrel_sovereign.testing.fixtures import native_tool_effect_agent

    original = service.load_section
    with patch.object(service, 'load_section', return_value={}) as outer:
        lifetime = native_tool_effect_agent.__wrapped__(tmp_path, monkeypatch)
        try:
            create = await anext(lifetime)
            agent = await create(did='did:test:nested-provider-fixture')
            assert not agent._runtime_publication_ready.is_set()
        finally:
            await lifetime.aclose()
        assert service.load_section is outer, 'Native fixture patches outlived their original lifetime'
    assert service.load_section is original


@pytest.mark.timeout(30)
def test_native_fixture_is_available_to_integration_without_plugin_autoload(tmp_path):
    """The native CI tier must not depend on package entry-point autoload."""
    from pathlib import Path
    import subprocess
    import sys

    import kestrel_sovereign

    repository = Path(__file__).resolve().parents[2]
    core_file = Path(kestrel_sovereign.__file__).resolve()
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    # A wheel-consumer run must exercise that SAME installed Core in its child,
    # even though pytest adds the test checkout to the child's import path.
    same_core = (
        "import pathlib,sys; sys.path.insert(0,sys.argv.pop(1)); "
        "import kestrel_sovereign; "
        "assert str(pathlib.Path(kestrel_sovereign.__file__).resolve()) == sys.argv.pop(1); "
        "import pytest; raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", same_core, str(core_file.parent.parent), str(core_file), "-q",
         "-p", "pytest_asyncio.plugin", "-p", "pytest_timeout",
         "-p", "no:cacheprovider", "-o", "addopts=",
         "--basetemp=" + str(tmp_path / "native-fixture-child"),
         "tests/integration/test_native_input_fixture.py::"
         "test_native_fixture_owns_original_storage_and_effect"],
        cwd=repository, env=environment, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, (
        "Native integration fixture registration failed without plugin autoload:\n"
        + result.stdout + result.stderr
    )
    assert "1 passed" in result.stdout
