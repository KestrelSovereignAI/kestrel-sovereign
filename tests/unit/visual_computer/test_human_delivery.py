"""Actual buffered host ASGI sends; SDK doubles do not prove Docker isolation."""
import asyncio

import httpx
import pytest

from tests.unit.visual_computer.test_host import AUTH, visual_host as visual_host
from tests.unit.visual_computer.test_human_http import setup


class OuterTransport:
    def __init__(self, app, path, *, suppress=False, fail=False):
        self.app, self.path = app, path
        self.suppress, self.fail = suppress, fail
        self.entered, self.cancelled = asyncio.Event(), asyncio.Event()
        self.release, self.closed = asyncio.Event(), asyncio.Event()
        self.child, self.bodies = None, []

    async def __call__(self, scope, receive, send):
        original_path = scope['path']
        async def outer_send(message):
            if original_path == self.path and message['type'] == 'http.response.body' and message.get('body'):
                self.child = asyncio.current_task()
                self.entered.set()
                try:
                    if self.fail:
                        # Transport acceptance is not evidence that no bytes
                        # escaped. The response must fail without replay.
                        self.bodies.append(message['body'][:1])
                        raise BrokenPipeError('owned test transport lost acknowledgement')
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        self.cancelled.set()
                        if self.suppress:
                            await self.release.wait()
                        raise
                    self.bodies.append(message['body'])
                finally:
                    self.closed.set()
            await send(message)
        await self.app(scope, receive, outer_send)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['stop', 'revoke'])
async def test_safety_joins_actual_outer_human_send_before_ack(visual_host, harness, operation):
    client, feature, _, sid, base, _ = await setup(visual_host, harness)
    transport = OuterTransport(visual_host[2], base+'/human/control/observation')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=transport), base_url='http://test') as delayed:
        active = asyncio.create_task(delayed.get(transport.path, headers=AUTH))
        try:
            await asyncio.wait_for(transport.entered.wait(), 2)
            assert transport.child is not active
            assert any(transport.child in call.accepted_jobs for call in feature.runtime.egress._calls.values())
            if operation == 'stop':
                response = await client.post(base+'/stop', headers=AUTH, json={
                    'request_id':'stop-send', 'session_id':sid, 'expected_control_epoch':1, 'operation':'pause'})
            else:
                response = await client.post(base+'/human/control/revoke', headers=AUTH,
                    json={'request_id':'revoke-send', 'control_epoch':1})
            assert response.status_code == 200, response.text
            assert transport.closed.is_set() and transport.child.done()
            assert not transport.bodies
        finally:
            transport.release.set()
            await asyncio.gather(active, return_exceptions=True)
    assert not feature.runtime.egress._calls


@pytest.mark.asyncio
async def test_stop_and_repeated_cancellation_cannot_detach_suppressed_outer_send(visual_host, harness):
    client, feature, _, sid, base, _ = await setup(visual_host, harness)
    transport = OuterTransport(visual_host[2], base+'/human/control/observation', suppress=True)
    request = {'request_id':'stop-retained', 'session_id':sid, 'expected_control_epoch':1, 'operation':'pause'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=transport), base_url='http://test') as delayed:
        active = asyncio.create_task(delayed.get(transport.path, headers=AUTH))
        stop = None
        writer = None
        try:
            await asyncio.wait_for(transport.entered.wait(), 2)
            agent = visual_host[1]['kite']
            transitioned = asyncio.Event()
            async def privacy_writer():
                async with agent.gate.exclusive():
                    transitioned.set()
            writer = asyncio.create_task(privacy_writer())
            await asyncio.sleep(0)
            assert not transitioned.is_set()
            stop = asyncio.create_task(client.post(base+'/stop', headers=AUTH, json=request))
            await asyncio.wait_for(transport.cancelled.wait(), 2)
            for _ in range(3):
                stop.cancel()
                active.cancel()
                await asyncio.sleep(0)
            assert not stop.done() and not active.done() and not transport.child.done()
            assert not transitioned.is_set()
            transport.release.set()
            await asyncio.gather(active, stop, return_exceptions=True)
            await asyncio.wait_for(writer, 2)
            assert transport.closed.is_set() and transport.child.done()
            assert not transport.bodies
            # The interrupted safety caller did not acknowledge success. Only
            # an exact original-ID retry can reconcile the actual control.
            response = await client.post(base+'/stop', headers=AUTH, json=request)
            assert response.status_code == 200, response.text
        finally:
            transport.release.set()
            await asyncio.gather(*(task for task in (active, stop, writer) if task), return_exceptions=True)
    assert not feature.runtime.egress._calls


@pytest.mark.asyncio
async def test_partial_transport_failure_has_unknown_delivery_without_replay_or_leak(visual_host, harness):
    _, feature, _, _, base, _ = await setup(visual_host, harness)
    transport = OuterTransport(visual_host[2], base+'/human/control/observation', fail=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=transport), base_url='http://test') as delayed:
        with pytest.raises((BrokenPipeError, BaseExceptionGroup)):
            await delayed.get(transport.path, headers=AUTH)
    assert transport.closed.is_set() and transport.child.done()
    assert len(transport.bodies) == 1 and len(transport.bodies[0]) == 1
    assert not feature.runtime.egress._calls


@pytest.mark.asyncio
async def test_fixed_human_expiry_joins_outer_send_and_never_delivers_late_body(visual_host, harness):
    _, feature, _, _, base, _ = await setup(visual_host, harness, lease_seconds=1)
    transport = OuterTransport(visual_host[2], base+'/human/control/observation')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=transport), base_url='http://test') as delayed:
        active = asyncio.create_task(delayed.get(transport.path, headers=AUTH))
        try:
            await asyncio.wait_for(transport.entered.wait(), 2)
            await asyncio.wait_for(transport.cancelled.wait(), 2)
            await asyncio.gather(active, return_exceptions=True)
            assert transport.closed.is_set() and transport.child.done()
            assert not transport.bodies
        finally:
            transport.release.set()
            await asyncio.gather(active, return_exceptions=True)
    assert not feature.runtime.egress._calls
