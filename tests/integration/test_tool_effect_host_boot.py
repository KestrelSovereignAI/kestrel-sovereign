"""The normal process host owns journal boot and refuses a corrupt envelope."""

from pathlib import Path
import sqlite3
import sys

import pytest

from tests.integration.test_mcp_permission_host_restart import host_campaign, native_host


pytestmark = [pytest.mark.mcp_integration, pytest.mark.timeout(120),
    pytest.mark.skipif(sys.platform == 'win32', reason='Requires native graceful SIGTERM')]


@pytest.mark.asyncio
async def test_normal_host_opens_journal_on_its_original_primary_database(host_campaign):
    c = host_campaign
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 'journal-boot') as host:
        assert host.state['journal_initialized'] is True
        database = Path(host.state['agent_database']).resolve()
        assert database.is_relative_to(c.home.resolve())
    assert host.process.returncode is not None
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        assert db.execute('SELECT COUNT(*) FROM tool_effect_keys').fetchone() == (1,)
        assert db.execute('SELECT COUNT(*) FROM tool_effect_operations').fetchone() == (0,)


@pytest.mark.asyncio
async def test_corrupt_journal_cannot_be_ignored_during_normal_host_boot(host_campaign):
    c = host_campaign
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 'journal-original') as original:
        database = Path(original.state['agent_database']).resolve()
        assert database.is_relative_to(c.home.resolve())
    with sqlite3.connect(database) as db:
        db.execute('UPDATE tool_effect_keys SET wrapped_key = ?', (b'owned-corrupt-envelope',))
    with pytest.raises(AssertionError, match='tool effect envelope cannot be opened'):
        async with native_host(c.home, c.descriptor, c.environment, c.peer, 'journal-corrupt'):
            pytest.fail('A configured persistent journal was silently ignored')
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        assert db.execute('SELECT wrapped_key FROM tool_effect_keys').fetchone() == (b'owned-corrupt-envelope',)
    assert c.peer.witness_file is not None
    assert not c.peer.witness_file.exists() or not c.peer.witness_file.read_text().splitlines()
