"""Lane revocation joins the original sub-operations before retirement.

IO is explicitly scripted here; the Tasks, stream close Future and Unix socket
are original stdlib objects. These cases do not claim native TLS conformance.
"""
import asyncio
from types import SimpleNamespace
import hashlib
import socket

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, execution_evidence_carriers,
)
from kestrel_sovereign.visual_computer import worker_broker as broker
from tests.unit.visual_computer.test_worker_stream import PeerTransport, stalled_tls_peer
from tests.unit.visual_computer.test_two_domain_worker_profile import child_settings


@pytest.mark.asyncio
@pytest.mark.parametrize(("stage", "kind"), [
    ("upgrade", "ordinary"), ("close", "ordinary"), ("close", "peer-reset"),
    ("connect", "ordinary"),
])
async def test_lane_revoke_during_owned_step_does_not_poison_retirement(stage, kind, monkeypatch):
    loop = asyncio.get_running_loop()
    entered, finish = asyncio.Event(), asyncio.Event()
    observed = {}
    original = ValueError("scripted exchange refused")
    reset = ConnectionResetError("original scripted peer reset")

    class Peer:
        def version(self):
            return "TLSv1.3"
        def getpeercert(self, binary_form):
            return b"scripted peer"

    class ConnectedTransport(PeerTransport):
        def get_extra_info(self, name, default=None):
            return Peer() if name == "ssl_object" else default

    class IPC:
        path = "never-connected.sock"
        def require_same(self, original):
            assert original is ipc
        def close(self):
            observed["ipc_closed"] = True

    ipc = IPC()
    if stage == "connect":
        monkeypatch.setattr(broker, "require_ipc", lambda *args: ipc)

    class DeferredClose(PeerTransport):
        def close(self):
            self.close_calls += 1
            self.closing = True
            entered.set()

    class Loop:
        def __getattr__(self, name):
            return getattr(loop, name)

        async def connect_accepted_socket(self, factory, raw, **kwargs):
            if stage == "upgrade":
                entered.set()
                await finish.wait()
                raise original
            protocol = factory()
            transport = (ConnectedTransport(protocol, None) if stage == "connect" else
                         DeferredClose(protocol, reset if kind == "peer-reset" else None))
            protocol.connection_made(transport)
            if stage == "connect":
                protocol._stream_reader.feed_data(b"{}\n")
            observed.update(protocol=protocol, transport=transport)
            return transport, protocol

        async def sock_connect(self, sock, path):
            assert path == ipc.path and sock.fileno() >= 0
            entered.set()
            await finish.wait()
            raise original

    def check():
        if stage != "connect":
            raise original

    slot = broker._Slot(SimpleNamespace(loop=Loop(), context=object(), check=check,
        role=broker.LaneRole.NORMAL, original_ipc=ipc,
        value={"client_sha256": hashlib.sha256(b"scripted peer").hexdigest()}))
    task = create_owned_task(slot.exchange(), admission=slot.admission,
                             name="original-revoked-broker-step")
    try:
        await asyncio.wait_for(entered.wait(), 2)
        slot.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "revocation must retain the original owned join"
        if stage == "close":
            observed["protocol"].connection_lost(observed["transport"].error)
        finish.set()
        outcome = await await_owned_task(task)
        assert task.done() and slot.admission.task is task
        assert broker._original_slot_error(slot, outcome.error) is None, (
            "a lane's own revocation must not poison a physically joined exchange")
        assert original in execution_evidence_carriers(slot.failure)
        if stage == "close":
            waiter = observed["protocol"]._get_close_waiter(slot.writer)
            assert waiter.done() and not waiter.cancelled()
            assert waiter.exception() is observed["transport"].error
            assert slot.writer.is_closing() and observed["transport"].close_calls == 1
            if kind == "peer-reset":
                assert reset in execution_evidence_carriers(slot.failure)
        if stage == "connect":
            assert slot.unix.fileno() == -1 and observed["ipc_closed"]
    finally:
        finish.set()
        if "protocol" in observed:
            waiter = observed["protocol"]._get_close_waiter(slot.writer)
            if not waiter.done():
                observed["protocol"].connection_lost(observed["transport"].error)
        await await_owned_task(task)


@pytest.mark.asyncio
async def test_lane_revoke_joins_actual_TLS_shutdown_timeout(tmp_path):
    async with stalled_tls_peer(tmp_path) as writer:
        class RefusedUpgrade:
            async def connect_accepted_socket(self, *args, **kwargs):
                raise ValueError("scripted exchange loss before native TLS close")
        slot = broker._Slot(SimpleNamespace(loop=RefusedUpgrade(), context=object()))
        slot.writer = writer
        task = create_owned_task(slot.exchange(), admission=slot.admission,
                                 name="revoked-broker-native-TLS-join")
        try:
            while not writer.is_closing():
                await asyncio.sleep(0)
            waiter = writer._protocol._get_close_waiter(writer)
            assert not waiter.done()
            slot.cancel()
            outcome = await await_owned_task(task)
            assert waiter.done() and type(waiter.exception()) is TimeoutError
            assert writer.transport._ssl_protocol._transport is None
            assert broker._original_slot_error(slot, outcome.error) is None, (
                "own revocation must preserve the completed native TLS retirement")
            assert waiter.exception() in execution_evidence_carriers(slot.failure)
        finally:
            await await_owned_task(task)


@pytest.mark.asyncio
async def test_lane_retirement_closes_socket_when_original_payload_never_started():
    lane = broker.WorkerLane(child_settings(broker.LaneRole.PRIORITY),
                             broker.LaneRole.PRIORITY, object())
    raw, peer = socket.socketpair()
    slot = broker._Slot(lane, raw=raw)
    lane.slots[0] = slot
    task = create_owned_task(slot.exchange(), admission=slot.admission,
                             name="unstarted-original-accepted-slot")
    slot.cancel()
    try:
        await lane._close_originals()
        assert task.done() and not slot.started
        assert raw.fileno() == -1, "original accepted socket must close before lane acknowledgment"
    finally:
        await await_owned_task(task)
        raw.close()
        peer.close()


class NativeAbort(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["authority", "commit", "retirement", "foreign-control"])
async def test_own_revoke_does_not_hide_original_owned_step_control(kind):
    entered, finish = asyncio.Event(), asyncio.Event()
    marker = object()
    if kind == "authority":
        original = ExecutionAuthorityError("original revoked native scope")
    elif kind == "commit":
        original = ExecutionCommitOutcomeError("unknown")
    elif kind == "retirement":
        original = RuntimeError("original physical retirement debt")
        original._kestrel_native_retirement_debts = ((marker, None),)
    else:
        original = NativeAbort("original foreign execution control")
    class Loop:
        async def connect_accepted_socket(self, *args, **kwargs):
            entered.set()
            await finish.wait()
            raise original
    slot = broker._Slot(SimpleNamespace(loop=Loop(), context=object()))
    task = create_owned_task(slot.exchange(), admission=slot.admission,
                             name="original-broker-native-control")
    try:
        await asyncio.wait_for(entered.wait(), 2)
        slot.cancel()
        finish.set()
        outcome = await await_owned_task(task)
        error = broker._original_slot_error(slot, outcome.error)
        assert error is not None
        assert original in execution_evidence_carriers(error)
    finally:
        finish.set()
        await await_owned_task(task)
