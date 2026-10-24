"""Owned native interop runtime and foreign Go process, shared across wire gates."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import psutil
from starlette.applications import Starlette

from kestrel_sovereign.a2a.inbound_authorization import RecipientA2ASenderAuthorizer
from kestrel_sovereign.a2a.native_card import (
    REQUIRED_NATIVE_EXTENSIONS,
)
from kestrel_sovereign.a2a.native_routes import native_card_path
from kestrel_sovereign.a2a.native_standalone import (
    prepare_single_agent_native_publication,
)
from kestrel_sovereign.a2a.stores.unified.task_store import NativeTaskSnapshot
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


@dataclass(frozen=True)
class NativeGoEnvironment:
    """An actual initialized agent with temporary peer identities and HTTP routes."""

    binary: Path
    root: Path
    config: dict
    config_path: Path
    recipient: KestrelAgent
    foreign_document: dict
    foreign_public: dict
    set_business_handler: Callable[[Callable[[NativeTaskSnapshot], Awaitable[None]]], None]


def write_peer_config(path, config):
    """Publish the owned test peer configuration to its child process."""
    path.write_text(json.dumps(config))


async def retire_peer(process, *, process_group=False):
    """Join the original process even when graceful shutdown exceeds its bound."""
    def interrupt(sig):
        try:
            if process_group:
                os.killpg(process.pid, sig)
            elif process.returncode is None:
                process.send_signal(sig)
        except ProcessLookupError:
            pass

    if process.returncode is None:
        interrupt(signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            interrupt(signal.SIGKILL)
            await process.wait()
    if process_group:
        # The launcher uses start_new_session=True; only its original session
        # is ours. A normally exited uv/runner must not strand its pytest child.
        def live_members():
            members = []
            for child in psutil.process_iter():
                try:
                    if os.getpgid(child.pid) == process.pid and child.status() != psutil.STATUS_ZOMBIE:
                        members.append(child.pid)
                except (ProcessLookupError, psutil.NoSuchProcess):
                    continue
            return members

        if live_members():
            interrupt(signal.SIGTERM)
            try:
                async with asyncio.timeout(5):
                    while live_members():
                        await asyncio.sleep(0.01)
            except TimeoutError:
                interrupt(signal.SIGKILL)
                async with asyncio.timeout(5):
                    while live_members():
                        await asyncio.sleep(0.01)


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


@asynccontextmanager
async def boot_native_go_environment(binary, tmp_path, monkeypatch):
    """Boot one real recipient and retain every original owner until cleanup."""
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

    write_peer_config(config_path, config)
    public = await run_peer(binary, "keys", config_path)
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
        write_peer_config(config_path, config)
        workers, worker_errors = set(), []

        async def join_business_handlers():
            if workers:
                await asyncio.gather(*workers)
            assert not worker_errors, worker_errors

        cleanup.push_async_callback(join_business_handlers)

        async def complete(snapshot):
            await manager.complete_task(
                snapshot.task.id, "Completed by Kestrel TaskManager", recipient_agent_id=core_id
            )

        business_handler = complete

        def set_business_handler(handler):
            nonlocal business_handler
            business_handler = handler

        def submitted(snapshot):
            task = asyncio.create_task(business_handler(snapshot))
            workers.add(task)

            def done(task):
                workers.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    worker_errors.append(task.exception())

            task.add_done_callback(done)

        manager._on_task_submitted = submitted
        app = Starlette()
        core_base = await cleanup.enter_async_context(serve_native_app(app))
        await prepare_single_agent_native_publication(app, recipient, {
            "KESTREL_BASE_URL": core_base, "KESTREL_A2A_DISCOVERY_AGENT": "core",
            "KESTREL_A2A_PUSH_ENABLED": "false",
        })
        owner = recipient._native_a2a_runtime_authority
        cleanup.push_async_callback(owner.close)
        config["coreCardUrl"] = core_base + native_card_path("core")

        write_peer_config(config_path, config)
        yield NativeGoEnvironment(
            binary=binary, root=tmp_path, config=config, config_path=config_path,
            recipient=recipient, foreign_document=foreign_doc, foreign_public=public,
            set_business_handler=set_business_handler,
        )
