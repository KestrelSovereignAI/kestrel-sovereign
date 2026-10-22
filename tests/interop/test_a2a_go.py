"""Public Go SDK interoperability with an initialized, isolated Kestrel agent.

Run explicitly with Go on PATH. The mandatory CI lane builds the foreign peer
from go.sum; a missing toolchain is a failure, never a skipped conformance gate.
The deterministic business handler completes real TaskManager tasks. This tests
the native protocol and persistence, not an LLM's response quality.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2ACardResolver
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, SendMessageRequest, TaskState

from kestrel_sovereign.a2a.native_card import (
    verify_native_agent_card,
)
from kestrel_sovereign.a2a.native_client import (
    NativePeerSession,
    NativeSenderInterceptor,
)
from kestrel_sovereign.a2a.native_sender import NativeProofPolicy
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import owned_native_http_client
from tests.utils.native_go_peer import run_peer, serve_go_peer, write_peer_config


@pytest.mark.timeout(120)
async def test_public_go_sdk_and_booted_kestrel_exchange_native_tasks(native_go_environment):
    environment = native_go_environment
    go_peer, config, config_path = environment.binary, environment.config, environment.config_path
    recipient, foreign_doc, public = environment.recipient, environment.foreign_document, environment.foreign_public
    identity, manager = recipient.identity, recipient.task_manager
    core_id, foreign_id, foreign_did = recipient.did, config["foreignStable"], foreign_doc["id"]
    owner = recipient._native_a2a_runtime_authority
    async with serve_go_peer(go_peer, config_path, config, environment.root) as foreign_base:
        # A validly shaped but unrelated trust anchor must fail independent
        # Agent Card verification before a task can be sent.
        trusted_ed = config["coreEdPublic"]
        config.update(coreEdPublic=public["edPublic"], protocol="JSONRPC")
        write_peer_config(config_path, config)
        error = await run_peer(go_peer, "client", config_path, accepted=False)
        assert "signature" in error.lower(), error
        config["coreEdPublic"] = trusted_ed

        for protocol in ("JSONRPC", "HTTP+JSON"):
            config["protocol"] = protocol
            write_peer_config(config_path, config)
            result = await run_peer(go_peer, "client", config_path)
            assert result["cardVerified"] is True
            snapshot = await manager.get_task_snapshot(result["task"]["id"], recipient_agent_id=core_id)
            assert snapshot.ownership.creator_agent_id == foreign_id
            assert snapshot.task.status.state == TaskState.TASK_STATE_COMPLETED
            config["coreStable"] = "did:kestrel:wrong-recipient"
            write_peer_config(config_path, config)
            error = await run_peer(go_peer, "client", config_path, accepted=False)
            assert "HTTP status" in error and "401" in error, error
            config["coreStable"] = core_id
            write_peer_config(config_path, config)
        assert len(await manager.list_tasks(recipient_agent_id=core_id)) == 2, "Rejected proofs wrote tasks"

        async with owned_native_http_client(httpx.AsyncClient(
            timeout=20, trust_env=False, follow_redirects=False
        )) as http:
            card = await A2ACardResolver(http, foreign_base).get_agent_card()
            verify_native_agent_card(card, signing_did=foreign_did, trusted_document=foreign_doc)
            for protocol in ("JSONRPC", "HTTP+JSON"):
                interface = next(item for item in card.supported_interfaces if item.protocol_binding == protocol)
                scope = NativeRecipientScope(foreign_id)
                signer = NativeSenderInterceptor.for_identity(
                    identity, recipient=scope, extensions=frozenset(config["extensions"]),
                    policy=NativeProofPolicy(), clock=time.time, nonce=lambda: uuid4().hex,
                )

                async def authorize():
                    owner.require_current()

                session = NativePeerSession(
                    card=card, signing_did=foreign_did, trusted_document=foreign_doc,
                    recipient=scope, interface=interface, signer=signer, http_client=http,
                    outbound_store=manager.outbound_store, question_store=manager.question_store,
                    authorize=authorize,
                )
                try:
                    dispatch, wire = await session.reserve(SendMessageRequest(
                        message=new_text_message("Kestrel to Go SDK", role=Role.ROLE_USER)
                    ), dispatch_tool="send_a2a_message", verb="message")
                    receipt = await session.send(dispatch, wire)
                    observed = await session.get_task(receipt.dispatch)
                    assert observed.response.status.state == TaskState.TASK_STATE_COMPLETED
                    assert await manager.outbound_store.get(dispatch.dispatch_id) == observed.dispatch
                finally:
                    await session.close()
    receipts = json.loads(Path(config["receipts"]).read_text())
    assert len(receipts) == 4
    assert {item["operation"] for item in receipts} == {"SendMessage", "GetTask"}
    assert {item["sender"] for item in receipts} == {identity.signing_did}
