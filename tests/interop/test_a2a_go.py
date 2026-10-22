"""Public Go SDK interoperability with an initialized, isolated Kestrel agent.

Run explicitly with Go on PATH. The mandatory CI lane builds the foreign peer
from go.sum; a missing toolchain is a failure, never a skipped conformance gate.
The deterministic business handler completes real TaskManager tasks. This tests
the native protocol and persistence, not an LLM's response quality.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2ACardResolver
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, SendMessageRequest, TaskState
from starlette.applications import Starlette

from kestrel_sovereign.a2a.inbound_authorization import RecipientA2ASenderAuthorizer
from kestrel_sovereign.a2a.native_card import (
    REQUIRED_NATIVE_EXTENSIONS,
    verify_native_agent_card,
)
from kestrel_sovereign.a2a.native_client import (
    NativePeerSession,
    NativeSenderInterceptor,
)
from kestrel_sovereign.a2a.native_routes import native_card_path
from kestrel_sovereign.a2a.native_sender import NativeProofPolicy
from kestrel_sovereign.a2a.native_standalone import (
    prepare_single_agent_native_publication,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import owned_native_http_client
from kestrel_sovereign.features.peers.directory import PeerRequester
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.security.crypto_suite import (
    ALG_ED25519,
    ALG_ML_DSA_65,
    get_suite,
)
from tests.shared.genesis_audit import deterministic_safe_genesis_auditor
from tests.utils.native_a2a import serve_native_app


@pytest.fixture(scope="session")
def go_peer(tmp_path_factory):
    """Compile and test the pinned public SDK, without private SDK patches."""
    go = shutil.which("go")
    assert go is not None, "Go is required for the native A2A interoperability gate"
    source = Path(__file__).with_name("a2a_go_peer")
    binary = tmp_path_factory.mktemp("native-go-peer") / "peer"
    for command in ([go, "test", "./..."], [go, "build", "-o", str(binary), "."]):
        result = subprocess.run(
            command, cwd=source, capture_output=True, text=True, timeout=180, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return binary


async def retire_peer(process):
    """Join the original process even when graceful shutdown exceeds its bound."""
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            process.kill()
            await process.wait()


async def run_peer(binary, mode, config_path, *, accepted=True):
    process = await asyncio.create_subprocess_exec(
        str(binary), mode, str(config_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 45)
        if accepted:
            assert process.returncode == 0, stderr.decode()
            return json.loads(stdout)
        assert process.returncode != 0, "Foreign peer accepted an invalid trust binding"
        return stderr.decode()
    finally:
        await retire_peer(process)


@asynccontextmanager
async def serve_go_peer(binary, config_path, config, root):
    log_path = root / "go-server.log"
    with log_path.open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            str(binary), "server", str(config_path), stdout=log, stderr=log
        )
        try:
            async with asyncio.timeout(10):
                ready = Path(config["ready"])
                while not ready.exists():
                    assert process.returncode is None, log_path.read_text()
                    await asyncio.sleep(0.01)
            yield json.loads(ready.read_text())["base"]
        finally:
            await retire_peer(process)
    assert process.returncode == 0, log_path.read_text()


@pytest.mark.timeout(120)
async def test_public_go_sdk_and_booted_kestrel_exchange_native_tasks(
    go_peer, tmp_path, monkeypatch
):
    foreign_id, foreign_did = "did:kestrel:interop-go", "did:web:interop.go.test:agent"
    config_path = tmp_path / "peer.json"
    config = {
        "keys": str(tmp_path / "keys.json"),
        "replay": str(tmp_path / "nonces"),
        "ready": str(tmp_path / "ready.json"),
        "receipts": str(tmp_path / "receipts.json"),
        "foreignStable": foreign_id,
        "extensions": sorted(REQUIRED_NATIVE_EXTENSIONS),
    }

    def save_config():
        config_path.write_text(json.dumps(config))

    save_config()
    public = await run_peer(go_peer, "keys", config_path)
    ed, pq = get_suite(ALG_ED25519), get_suite(ALG_ML_DSA_65)
    foreign_doc = {
        "id": foreign_did,
        "verificationMethod": build_verification_methods(foreign_did, [
            (ed, ed.deserialize_public_key(base64.urlsafe_b64decode(public["edPublic"]["x"] + "=="))),
            (pq, pq.deserialize_public_key(base64.b64decode(public["pqPublic"]))),
        ]),
    }
    agent_root, home = tmp_path / "agent", tmp_path / "home"
    agent_root.mkdir()
    home.mkdir()
    (home / "kestrel.toml").write_text("[llm]\nprovider_priority = []\nproviders = []\n")
    for name, value in {
        "KESTREL_HOME": str(home),
        "KESTREL_DB_PATH": str(agent_root),
        "KESTREL_HOST_DB_PATH": str(tmp_path / "host.db"),
        "KESTREL_DB_BACKEND": "sqlite",
        "KESTREL_HOLD_BACKEND": "sqlite",
        "KESTREL_PHOENIX_ENABLED": "0",
        "KESTREL_DATA_KEY": uuid4().hex,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("LIGHTHOUSE_API_KEY", raising=False)
    credentials = await create_kestrel_identity_async(
        str(agent_root), is_test_instance=True, identity_method="did:web",
        did_web_domain="interop.core.test", did_web_slug="agent",
        genesis_auditor=deterministic_safe_genesis_auditor,
        genesis_audit_provenance="test:native_foreign_go_interop",
    )
    core_id = credentials.agent_did

    class ScopedPolicy:
        async def authorize_inbound_sender(self, requester, sender_id):
            return requester.identity == core_id and sender_id == foreign_id

    async with AsyncExitStack() as cleanup:
        llm = LLMService(agent_data_dir=agent_root)
        cleanup.push_async_callback(llm.close)
        llm.disabled = True
        recipient = KestrelAgent(
            did=core_id, storage_path=str(agent_root / "kestrel_prime.db"),
            llm_service=llm, allowed_features=set(), peer_directory_router=ScopedPolicy(),
            peer_requester=PeerRequester(core_id, object()),
        )
        cleanup.push_async_callback(recipient.shutdown)
        recipient.a2a_did_resolver = lambda did: foreign_doc if did == foreign_did else None
        recipient.a2a_inbound_sender_authorizer = RecipientA2ASenderAuthorizer(
            None, recipient=recipient,
            sender_id_resolver=lambda did: foreign_id if did == foreign_did else None,
        )
        await recipient.initialize()
        assert recipient._runtime_publication_ready.is_set(), "Real agent did not reach READY"
        identity, manager = recipient.identity, recipient.task_manager
        config.update(
            coreStable=core_id, foreignDocument=foreign_doc,
            coreDocument={"id": identity.signing_did, "verificationMethod": identity.new_verification_methods},
            coreEdPublic={"x": base64.urlsafe_b64encode(ed.serialize_public_key(identity.hybrid_keypair.classical.public_key)).rstrip(b"=").decode()},
            corePqPublic=base64.b64encode(pq.serialize_public_key(identity.hybrid_keypair.pq.public_key)).decode(),
        )
        save_config()
        workers, worker_errors = set(), []

        async def join_business_handlers():
            if workers:
                await asyncio.gather(*workers)
            assert not worker_errors, worker_errors

        cleanup.push_async_callback(join_business_handlers)

        def submitted(snapshot):
            task = asyncio.create_task(manager.complete_task(
                snapshot.task.id, "Completed by Kestrel TaskManager", recipient_agent_id=core_id
            ))
            workers.add(task)

            def done(task):
                workers.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    worker_errors.append(task.exception())

            task.add_done_callback(done)

        manager._on_task_submitted = submitted
        foreign_base = await cleanup.enter_async_context(
            serve_go_peer(go_peer, config_path, config, tmp_path)
        )
        app = Starlette()
        core_base = await cleanup.enter_async_context(serve_native_app(app))
        await prepare_single_agent_native_publication(app, recipient, {
            "KESTREL_BASE_URL": core_base, "KESTREL_A2A_DISCOVERY_AGENT": "core",
            "KESTREL_A2A_PUSH_ENABLED": "false",
        })
        owner = recipient._native_a2a_runtime_authority
        cleanup.push_async_callback(owner.close)
        config["coreCardUrl"] = core_base + native_card_path("core")

        # A validly shaped but unrelated trust anchor must fail independent
        # Agent Card verification before a task can be sent.
        trusted_ed = config["coreEdPublic"]
        config.update(coreEdPublic=public["edPublic"], protocol="JSONRPC")
        save_config()
        error = await run_peer(go_peer, "client", config_path, accepted=False)
        assert "signature" in error.lower(), error
        config["coreEdPublic"] = trusted_ed

        for protocol in ("JSONRPC", "HTTP+JSON"):
            config["protocol"] = protocol
            save_config()
            result = await run_peer(go_peer, "client", config_path)
            assert result["cardVerified"] is True
            snapshot = await manager.get_task_snapshot(result["task"]["id"], recipient_agent_id=core_id)
            assert snapshot.ownership.creator_agent_id == foreign_id
            assert snapshot.task.status.state == TaskState.TASK_STATE_COMPLETED
            config["coreStable"] = "did:kestrel:wrong-recipient"
            save_config()
            error = await run_peer(go_peer, "client", config_path, accepted=False)
            assert "HTTP status" in error and "401" in error, error
            config["coreStable"] = core_id
            save_config()
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
