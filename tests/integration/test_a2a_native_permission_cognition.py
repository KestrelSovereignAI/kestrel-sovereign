"""Actual A2A HTTP reaches booted cognition, SDK tools and native permission effects.

Both agents, signed cards, sender policy, prompt, model transport, tools and
Task completion are real. Only the finite external model/MCP peer is scripted.
This is owned local interoperability, not a foreign-framework or vendor claim.
"""

import asyncio
import sqlite3
import sys
import time
from contextlib import closing
from uuid import uuid4

import httpx
import pytest
from google.protobuf.json_format import ParseDict
from kestrel_sdk.native_a2a.client import ClientConfig, ClientFactory
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
from kestrel_sdk.native_a2a.types import (
    AgentCard,
    GetTaskRequest,
    Role,
    SendMessageRequest,
    TaskState,
)

from kestrel_sovereign.a2a.native_card import (
    REQUIRED_NATIVE_EXTENSIONS,
    verify_native_agent_card,
)
from kestrel_sovereign.a2a.native_client import NativeSenderInterceptor
from kestrel_sovereign.a2a.native_sender import NativeProofPolicy
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from tests.integration.test_mcp_permission_host_restart import (
    configure_mcp,
    host_client,
    native_host,
    set_permission,
)
from tests.integration.test_mcp_permission_host_restart import (
    host_campaign as _host_campaign,
)

host_campaign = _host_campaign


pytestmark = [pytest.mark.mcp_integration, pytest.mark.timeout(120),
    pytest.mark.skipif(sys.platform == 'win32', reason='This test requires native graceful SIGTERM')]


@pytest.mark.parametrize('host_campaign', [{'native_a2a': True}], indirect=True)
@pytest.mark.parametrize('protocol', ['JSONRPC', 'HTTP+JSON'])
@pytest.mark.parametrize('permission,decision,expected', [
    ('deny', 'auto_denied', []),
    ('allow', 'auto_allowed', ['owned-provider-effect']),
    ('always_ask', 'no_approver', []),
])
async def test_signed_native_http_cognition_obeys_the_real_remote_tool_permission(
    host_campaign, protocol, permission, decision, expected,
):
    c = host_campaign
    async with (
        native_host(c.home, c.descriptor, c.environment, c.peer, 1) as host,
        host_client(c, host.state) as operator,
    ):
        owner, handle = await configure_mcp(operator, c, host.state)
        await set_permission(operator, owner, handle, permission)
        response = await operator.post('/api/agents/permission-host/api/security/permissions',
            json={'feature': 'TaskFeature', 'tool': 'respond_to_a2a_task', 'level': 'allow'})
        assert response.status_code == 200, response.text
        origin = f'http://127.0.0.1:{host.state["port"]}'
        async with httpx.AsyncClient(base_url=origin, trust_env=False, timeout=30) as wire:
            discovery = await wire.get('/.well-known/agent-card.json')
            assert discovery.status_code == 200, discovery.text
            card = ParseDict(discovery.json(), AgentCard())
            identity = c.recipient_identity
            verify_native_agent_card(card, signing_did=identity.signing_did,
                trusted_document={'id': identity.signing_did,
                    'verificationMethod': identity.new_verification_methods})
            scope = NativeRecipientScope(c.did)
            signer = NativeSenderInterceptor.for_identity(c.sender_identity,
                recipient=scope, extensions=REQUIRED_NATIVE_EXTENSIONS,
                policy=NativeProofPolicy(), clock=time.time, nonce=lambda: uuid4().hex)
            client = ClientFactory(ClientConfig(httpx_client=wire, streaming=False,
                polling=True, supported_protocol_bindings=[protocol])).create(card,
                    interceptors=[signer])
            request = SendMessageRequest(tenant=scope.tenant,
                message=new_text_message(
                    'Attempt the connected marker tool once under your permission policy, '
                    'then finish the inbox task to report that the permission evaluation is complete.',
                    role=Role.ROLE_USER),
                metadata={'sender': c.did, 'sender_verified': True, 'creator_agent_id': c.did})
            async def complete_request():
                async with asyncio.timeout(30):
                    events = [event async for event in client.send_message(request)]
                    task = events[-1].task
                    while task.status.state not in {
                        TaskState.TASK_STATE_COMPLETED, TaskState.TASK_STATE_FAILED,
                        TaskState.TASK_STATE_CANCELED, TaskState.TASK_STATE_REJECTED,
                    }:
                        await asyncio.sleep(0.1)
                        task = await client.get_task(GetTaskRequest(id=task.id, tenant=scope.tenant))
                return task

            task = await complete_request()
            if permission == 'always_ask':
                pending = await operator.get('/api/agents/permission-host/api/security/pending')
                assert pending.status_code == 200, pending.text
                assert pending.json()['pending'] == [], (
                    'A signed peer without an interactive approver manufactured a private approval'
                )
            assert task.status.state == TaskState.TASK_STATE_COMPLETED, str(task)
            assert task.status.message.parts[0].text == 'Owned permission evaluation complete'
        audit = await operator.get('/api/agents/permission-host/api/security/audit')
        assert audit.status_code == 200, audit.text
        decisions = [entry['decision'] for entry in audit.json()['logs']
            if entry['feature'] == owner and entry['tool'] == handle]
        assert decisions == [decision], audit.text
        assert await c.peer.witness(None, None) == expected
    assert await c.peer.witness(None, None) == expected, 'No native effect may escape after host retirement'
    with closing(sqlite3.connect((c.home / 'agent/kestrel_prime.db').as_uri()
            + '?mode=ro', uri=True)) as database:
        creator = database.execute(
            'SELECT creator_agent_id FROM a2a_tasks '
            'WHERE id = ? AND recipient_agent_id = ? AND tenant = ?',
            (task.id, c.did, scope.tenant)).fetchone()
    assert creator == (c.sender_identity.signing_did,), (
        'Caller-authored recipient metadata replaced the cryptographically verified task creator'
    )
