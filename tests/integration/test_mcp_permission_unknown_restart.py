"""A repeated transport identity cannot repeat an unconfirmed remote effect."""

import json
import uuid

from tests.integration.test_mcp_permission_host_restart import (
    configure_remote, host_campaign as host_campaign, host_client, native_host,
    observe_registry,
    pytestmark as pytestmark, set_permission,
)
import pytest


@pytest.mark.parametrize('host_campaign', [{'drop_first_tool_reply': True}], indirect=True)
async def test_restarted_host_does_not_replay_a_lost_remote_effect(host_campaign):
    c = host_campaign
    payload = dict(input='Record the owned marker using the connected MCP tool.',
        provider='deepseek:api', model='owned-permission-model',
        session_id='owned-host-restart', request_id=uuid.uuid4().hex)
    path = '/api/agents/permission-host/api/agent/invoke'
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 1) as first:
        async with host_client(c, first.state) as client:
            owner, tool = await configure_remote(client, c, first.state)
            await set_permission(client, owner, tool, 'allow')
            original = await client.post(path, json=payload)
            assert await c.peer.witness(None, None) == ['owned-provider-effect']
            assert 'Owned MCP reply lost after its real external effect' in (
                c.home / 'peer/remote-peer.log').read_text()
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 2) as second:
        assert second.state['did'] == first.state['did'] == c.did
        assert second.state['pid'] != first.state['pid']
        async with host_client(c, second.state) as client:
            nonce = uuid.uuid4().hex
            permissions = await client.get(
                '/api/agents/permission-host/api/security/permissions/tree',
                headers={'X-Owned-Registry-Observation': nonce})
            assert permissions.status_code == 200, permissions.text
            observed = await observe_registry(c, second.state, nonce)
            assert observed['remote_tools'] == {tool: owner}
            remote, = [entry for entry in permissions.json()['tree']
                if entry['name'] == owner]
            restored, = [entry for entry in remote['tools']
                if entry['name'] == tool]
            assert restored['level'] == 'allow'
            repeated = await client.post(path, json=payload)
            effects = await c.peer.witness(None, None)
            (c.home / 'request-retry-observation.json').write_text(json.dumps(dict(
                first_http_status=original.status_code, first_reply=original.json(),
                repeated_http_status=repeated.status_code, repeated_reply=repeated.json(),
                effects=effects), indent=2) + '\n')
            assert effects == ['owned-provider-effect']
            assert repeated.status_code == 409, repeated.text
            assert repeated.json()['error']['code'] == 'tool_effect_reconciliation_required'
            # The restored host remains usable: a new operation may perform
            # one new effect. A broken connection or blanket refusal is no fence.
            fresh = await client.post(path, json=dict(payload, request_id=uuid.uuid4().hex))
            assert fresh.status_code == 200, fresh.text
            assert await c.peer.witness(None, None) == [
                'owned-provider-effect', 'owned-provider-effect']
