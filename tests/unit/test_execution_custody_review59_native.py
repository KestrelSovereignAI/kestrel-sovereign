"""Real SQLite effects and native SDK discovery ownership across retirement."""

import asyncio
import errno
import json
import os
from dataclasses import replace
from types import SimpleNamespace

import ollama
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.a2a.inbound_authorization import RecipientA2ASenderAuthorizer
from kestrel_sovereign.a2a.native_security import NativeRequestAuthority
from kestrel_sovereign.endpoints.features import _settle_feature_transition
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    execution_evidence_carriers,
    is_execution_control_error,
)
from kestrel_sovereign.features.peers.directory import PeerDirectoryError, PeerRequester
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_execution_custody_review32 import receipt
from tests.utils.native_a2a_security import send, signed_context


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group", "authority", "committed"])
async def test_cancelled_feature_transition_keeps_original_sqlite_receipt(tmp_path, kind):
    db = await AsyncDatabase.sqlite(str(tmp_path / "feature.db"))
    await db.execute("CREATE TABLE effects(id INTEGER)")
    committed, release = asyncio.Event(), asyncio.Event()
    original = ExecutionCommitOutcomeError("committed") if kind == "committed" else receipt(kind)

    async def operation():
        await db.execute("INSERT INTO effects VALUES(1)")
        committed.set()
        await release.wait()
        raise original

    task = create_owned_task(_settle_feature_transition(
        SimpleNamespace(), operation, feature_name="offline", operation="disable",
    ))
    try:
        await asyncio.wait_for(committed.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "caller cancelled away the admitted native operation"
        release.set()
        outcome = await await_owned_task(task)
        assert outcome.error is original, "caller cancellation erased the original effect/authority receipt"
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        release.set()
        await await_owned_task(task)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["directory", "host"])
@pytest.mark.parametrize("kind", ["unknown", "group", "authority", "directory-cause", "ordinary"])
async def test_a2a_authorization_preserves_original_native_control(tmp_path, native_request_admission, boundary, kind):
    db = await AsyncDatabase.sqlite(str(tmp_path / "authorization.db"))
    await db.execute("CREATE TABLE effects(id INTEGER)")
    original = None if kind == "ordinary" else receipt(kind)
    if kind == "directory-cause":
        original = PeerDirectoryError("native directory query failed")
        original.__cause__ = receipt("unknown")

    actors = []

    class Directory:
        async def authorize_inbound_sender(self, requester, sender_id):
            nonlocal original
            before = asyncio.current_task()
            await db.execute("INSERT INTO effects VALUES(1)")
            actors.append((before, asyncio.current_task()))
            if kind == "ordinary":
                descriptor = os.open(os.devnull, os.O_WRONLY)
                try:
                    os.read(descriptor, 1)
                except OSError as error:
                    original = error
                    raise
                finally:
                    os.close(descriptor)
            raise original

    recipient = SimpleNamespace(
        did="did:test:recipient",
        peer_directory_router=Directory(),
        peer_requester=PeerRequester("did:test:recipient", object()),
    )
    authorizer = RecipientA2ASenderAuthorizer(
        SimpleNamespace(), recipient=recipient, sender_id_resolver=lambda did: did,
    )
    async def native_admission():
        admission = native_request_admission
        original_authority = admission.guard.authority_factory()
        authority = NativeRequestAuthority(
            original_authority.resolver, authorizer.authorize_principal,
            original_authority.require_current,
            original_authority.authorize_retained_principal,
        )
        guard = replace(admission.guard, authority_factory=lambda: authority)
        params = send(admission)
        async with guard.admit(params, "SendMessage", signed_context(admission, params)):
            return True

    operation = (authorizer.authorize("did:test:sender") if boundary == "directory"
                 else native_admission())
    try:
        outcome = await await_owned_task(create_owned_task(operation))
        if boundary == "host" and kind == "ordinary":
            # Opaque I/O now enters native control at the host boundary, never
            # ordinary authorization-denial data. Keep the exact physical cause.
            assert isinstance(outcome.error, ExecutionAuthorityError)
            assert is_execution_control_error(outcome.error)
            assert outcome.error.__cause__ is original
        else:
            assert outcome.error is original, "Original native authorization evidence was replaced"
        assert any(node is original for node in execution_evidence_carriers(outcome.error))
        if kind == "ordinary":
            assert isinstance(original, OSError) and original.errno == errno.EBADF
        assert len(actors) == 1 and actors[0][0] is actors[0][1]
        assert not native_request_admission.lock.locked()
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_ollama_close_joins_active_native_discovery_borrower(monkeypatch):
    requested, response, peer_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    writers = []
    async def peer(reader, writer):
        writers.append(writer)
        try:
            await reader.readuntil(b"\r\n\r\n")
            requested.set()
            await response.wait()
            body = json.dumps({"models": []}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(body)).encode() + b"\r\n\r\n" + body)
            await writer.drain()
            await reader.read(1)
        finally:
            writer.close()
            await writer.wait_closed()
            peer_closed.set()

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    sdk = ollama.AsyncClient(host=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    monkeypatch.setattr(ollama, "AsyncClient", lambda: sdk)
    adapter = OllamaAdapter()
    discovery = create_owned_task(adapter.list_models())
    closing = None
    try:
        await asyncio.wait_for(requested.wait(), 2)
        closing = create_owned_task(adapter.aclose())
        await asyncio.sleep(0.1)
        assert not closing.done(), "shutdown retired an SDK pool while discovery still borrowed it"
        response.set()
        result = await await_owned_task(discovery)
        assert result.error is None and result.result == [], "native borrower lost its retirement entry"
        assert (await await_owned_task(closing)).error is None
        await asyncio.wait_for(peer_closed.wait(), 2)
        assert not adapter._discovery_clients and sdk._client.is_closed
    finally:
        response.set()
        await await_owned_task(discovery)
        if closing is not None:
            await await_owned_task(closing)
        await sdk._client.aclose()
        for writer in writers:
            writer.close()
        server.close()
        await server.wait_closed()
