"""Actual native proof, DB policy and lifecycle-lease fixtures."""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_text_message
from kestrel_sovereign._vendor.a2a.types import Role, SendMessageRequest
from google.protobuf.json_format import MessageToDict
from starlette.exceptions import HTTPException
from starlette.requests import Request

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.a2a.extensions import RECIPIENT_POLICY_URI, SIGNED_SENDER_URI
from kestrel_sovereign.a2a.native_context import (
    NativeRecipientBinding,
    NativeServerCallContextBuilder,
)
from kestrel_sovereign.a2a.native_security import (
    NativeDIDRequestGuard,
    NativeRequestAuthority,
)
from kestrel_sovereign.a2a.native_sender import (
    SENDER_PROOF_HEADER,
    NativeProofPolicy,
    sign_native_request,
)
from kestrel_sovereign.a2a.replay_store import SharedReplayNonceStore
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from tests.utils.native_a2a import manager_for, signing_identity

SENDER = "did:web:sender.test:agent"
PRINCIPAL = "did:kestrel:authorized-sender"
NOW = 1_700_000_000
EXTENSIONS = frozenset({SIGNED_SENDER_URI, RECIPIENT_POLICY_URI})


@pytest_asyncio.fixture
async def native_push_runtime():
    """Wire tests opt into the real managed service explicitly."""
    yield None


@pytest.fixture(scope="module", name="native_signing")
def signing():
    return signing_identity(SENDER)


@pytest_asyncio.fixture(name="native_request_admission")
async def admission(native_a2a_backend_factory, native_signing):
    backend = native_a2a_backend_factory()
    await backend.connect()
    scope = NativeRecipientScope("did:kestrel:recipient", "tenant-alpha")
    manager = manager_for(backend, scope)
    lock = AsyncReaderWriterLock()
    state = SimpleNamespace(current=True, authorizations=[])

    def require_current(sender):
        assert lock.locked()
        if not state.current:
            raise HTTPException(status_code=403, detail="Recipient policy changed")

    async def authorize(sender):
        assert lock.locked()
        state.authorizations.append(sender)
        row = await backend.fetch_one(
            "SELECT principal FROM native_test_principals WHERE signing_did = ?",
            (sender,),
        )
        return row[0] if row else None

    async def authorize_retained(principal_id):
        assert lock.locked()
        return await backend.fetch_one(
            "SELECT principal FROM native_test_principals WHERE principal = ?",
            (principal_id,),
        ) is not None

    authority = NativeRequestAuthority(
        resolver=lambda did: native_signing[1],
        authorize_sender=authorize,
        require_current=require_current,
        authorize_retained_principal=authorize_retained,
    )
    guard = NativeDIDRequestGuard(
        routing_key="registered-agent",
        scope=scope,
        required_extensions=EXTENSIONS,
        proof_policy=NativeProofPolicy(),
        authority_factory=lambda: authority,
        replay_store=SharedReplayNonceStore(backend),
        lifecycle_lease=lock.read,
        clock=lambda: NOW,
    )
    try:
        await manager.initialize()
        await guard.replay_store.ensure_table()
        await backend.execute(
            "CREATE TABLE native_test_principals (signing_did TEXT PRIMARY KEY, principal TEXT NOT NULL)"
        )
        await backend.execute(
            "INSERT INTO native_test_principals VALUES (?, ?)", (SENDER, PRINCIPAL)
        )
        yield SimpleNamespace(
            guard=guard,
            backend=backend,
            manager=manager,
            scope=scope,
            lock=lock,
            state=state,
            signing=native_signing,
        )
    finally:
        await manager.close()


def signed_context(
    admission,
    params,
    *,
    operation="SendMessage",
    protocol="JSONRPC",
    proof_changes=None,
    headers=None,
    activated=EXTENSIONS,
):
    guard = admission.guard
    options = {
        "sender": SENDER,
        "recipient": guard.scope.recipient_id,
        "operation": operation,
        "timestamp": NOW,
        "nonce": uuid4().hex,
        "extensions": EXTENSIONS,
        "classical_kid": "key-1",
        "pq_kid": "key-2",
        "policy": guard.proof_policy,
    }
    options.update(proof_changes or {})
    proof = sign_native_request(admission.signing[0], params, **options)
    header_values = (
        [(SENDER_PROOF_HEADER, proof)] if headers is None else headers(proof)
    )
    header_values.append(("A2A-Extensions", ",".join(sorted(activated))))
    body = MessageToDict(params)
    if protocol == "JSONRPC":
        body = {"jsonrpc": "2.0", "id": "request", "method": operation, "params": body}
    encoded = json.dumps(body).encode()

    body_sent = False
    disconnected = asyncio.Event()

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": encoded, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/",
            "query_string": b"",
            "headers": [
                (key.lower().encode(), value.encode()) for key, value in header_values
            ],
            "path_params": {"tenant": guard.scope.tenant}
            if protocol == "HTTP+JSON"
            else {},
        },
        receive,
    )
    context = NativeServerCallContextBuilder(
        NativeRecipientBinding(
            guard.routing_key, guard.scope.recipient_id, guard.scope.tenant, protocol
        )
    ).build(request)
    context.tenant = params.tenant
    return context


def send(admission):
    return SendMessageRequest(
        tenant=admission.scope.tenant,
        message=new_text_message("request", role=Role.ROLE_USER),
        metadata={
            "sender": "forged",
            "sender_verified": True,
            "creator_agent_id": "forged",
        },
    )
