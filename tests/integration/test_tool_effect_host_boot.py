"""The normal process host owns journal boot and refuses a corrupt envelope."""

from pathlib import Path
from contextlib import closing
import sqlite3
import sys
import uuid

import pytest

from tests.integration.test_mcp_permission_host_restart import (
    host_campaign as host_campaign, host_client, native_host, observe_registry,
)


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


@pytest.mark.asyncio
@pytest.mark.parametrize('corrupt', [False, True])
async def test_volatile_restart_reopens_original_journal_when_persistence_returns(host_campaign, corrupt):
    c = host_campaign
    endpoint = '/api/agents/permission-host/api/agent/privacy-mode'
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 'before-volatile') as first:
        database = Path(first.state['agent_database']).resolve()
        assert database.is_relative_to(c.home.resolve())
        async with host_client(c, first.state) as client:
            response = await client.post(endpoint, json={'mode': 'ephemeral'})
            assert response.status_code == 200 and response.json()['success'], response.text
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        original = db.execute('SELECT key_id, scope_digest, wrapped_key FROM tool_effect_keys').fetchall()
        assert len(original) == 1
    if corrupt:
        with closing(sqlite3.connect(database)) as db, db:
            db.execute('UPDATE tool_effect_keys SET wrapped_key = ?', (b'owned-corrupt-envelope',))
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 'volatile-restart') as second:
        assert second.state['pid'] != first.state['pid']
        assert second.state['journal_initialized'] is False
        async with host_client(c, second.state) as client:
            nonce = uuid.uuid4().hex
            response = await client.post(endpoint, json={'mode': 'normal'},
                headers={'X-Owned-Registry-Observation': nonce})
            observed = await observe_registry(c, second.state, nonce)
            assert observed['privacy_mode'] == 'normal'
            if corrupt:
                assert response.status_code == 500, 'Transition ignored the original corrupt envelope'
                assert observed['journal_initialized'] is False
                assert observed['journal_on_primary'] is False
            else:
                assert response.status_code == 200 and response.json()['success'], response.text
                assert observed['journal_initialized'], 'Persistent transition left the journal unavailable'
                assert observed['journal_on_primary'], 'Transition replaced the original primary database'
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        expected = [(original[0][0], original[0][1], b'owned-corrupt-envelope')] if corrupt else original
        assert db.execute('SELECT key_id, scope_digest, wrapped_key FROM tool_effect_keys').fetchall() == expected
        assert db.execute('SELECT COUNT(*) FROM tool_effect_operations').fetchone() == (0,)
    assert await c.peer.witness(None, None) == []


@pytest.mark.asyncio
async def test_actual_http_retry_preserves_client_input_before_session_resolution(host_campaign):
    c = host_campaign
    endpoint = '/api/agents/permission-host/api/agent/invoke'
    body = {'input': '!help', 'request_id': 'owned-http-intent'}
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 'http-retry-root') as host:
        database = Path(host.state['agent_database']).resolve()
        async with host_client(c, host.state) as client:
            response = await client.post(endpoint, json=body)
            assert response.status_code == 200, response.text
            with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
                first, = db.execute('SELECT attempt_id, revision, state FROM tool_effect_operations').fetchall()
                assert first[2] == 'CLOSED'
            response = await client.post(endpoint, json={
                'input': body['input'], 'invocation_id': body['request_id'], 'user_passphrase': None,
            })
            assert response.status_code == 409, response.text
            assert response.json()['error']['code'] == 'tool_effect_reconciliation_required', (
                'Unchanged client input was mistaken for another operation: ' + response.text)
            with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
                assert db.execute('SELECT attempt_id, revision, state FROM tool_effect_operations').fetchall() == [first]
            response = await client.post(endpoint, json=dict(body, input='!status'))
            assert response.status_code == 409, response.text
            assert response.json()['error']['code'] == 'tool_effect_operation_conflict'
            with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
                assert db.execute('SELECT attempt_id, revision, state FROM tool_effect_operations').fetchall() == [first]
                assert db.execute('SELECT COUNT(*) FROM tool_effect_claims').fetchone() == (0,)
            response = await client.post(endpoint, json=dict(body, request_id='owned-http-fresh-intent'))
            assert response.status_code == 200, response.text
            with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
                rows = db.execute('SELECT attempt_id, revision, state FROM tool_effect_operations').fetchall()
                assert len(rows) == 2 and first in rows
                fresh, = [row for row in rows if row != first]
                assert fresh[0] != first[0] and fresh[2] == 'CLOSED'
                assert db.execute('SELECT COUNT(*) FROM tool_effect_claims').fetchone() == (0,)
    assert await c.peer.witness(None, None) == []


@pytest.mark.asyncio
async def test_actual_http_stream_admits_and_closes_its_original_journal_turn(host_campaign):
    c = host_campaign
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 'http-stream-root') as host:
        database = Path(host.state['agent_database']).resolve()
        async with host_client(c, host.state) as client:
            response = await client.post('/api/agents/permission-host/api/agent/stream',
                json={'input': '!help', 'request_id': 'owned-stream-root'})
            assert response.status_code == 200, response.text
        with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
            rows = db.execute('SELECT state FROM tool_effect_operations').fetchall()
            assert rows == [('CLOSED',)], 'Native streaming producer bypassed its original journal turn'
            assert db.execute('SELECT COUNT(*) FROM tool_effect_claims').fetchone() == (0,)
    assert await c.peer.witness(None, None) == []
