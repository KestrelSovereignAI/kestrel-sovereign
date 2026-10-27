"""Real instrumented native HTTP for registered Human routes; no model calls."""
import asyncio
import logging
import socket

import httpx
import pytest

from tests.unit.visual_computer.test_host import AUTH
from tests.unit.visual_computer.test_host import visual_host as visual_host
from tests.unit.visual_computer.test_human_http import setup


@pytest.mark.asyncio
async def test_registered_human_http_is_private_before_spans_and_native_access_logs(
    visual_host, harness, monkeypatch, caplog,
):
    pytest.importorskip('opentelemetry.instrumentation.fastapi')
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from kestrel_sovereign import telemetry
    from kestrel_sovereign.visual_computer.runtime import VisualAction
    import uvicorn

    from fastapi import FastAPI
    # Other tests serve the shared host singleton first. Instrument a fresh
    # public ASGI wrapper before its first request, retaining the genuinely
    # registered host routes/middleware rather than retrofitting a cached stack.
    app = FastAPI()
    app.add_route('/health', visual_host[2], methods=['GET'])
    app.mount('/', visual_host[2])
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrument = FastAPIInstrumentor.instrument_app
    monkeypatch.setattr(FastAPIInstrumentor, 'instrument_app',
        lambda target, **kwargs: instrument(target, tracer_provider=provider, **kwargs))
    telemetry._instrument_fastapi(app)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    runner = uvicorn.Server(uvicorn.Config(app, log_config=None, access_log=True, lifespan='off'))
    serving = None
    caplog.set_level(logging.INFO, logger='uvicorn.access')
    sentinel = 'PRIVATE-HUMAN-URL-AND-ERROR-SENTINEL'
    try:
        _, feature, extension, sid, base, lease = await setup(visual_host, harness)
        serving = asyncio.create_task(runner.serve(sockets=[sock]))
        async with asyncio.timeout(5):
            while not runner.started:
                await asyncio.sleep(.01)
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{sock.getsockname()[1]}', trust_env=False) as client:
            viewed = await client.get(base+'/human/control/observation', headers=AUTH)
            assert viewed.status_code == 200 and viewed.headers['cache-control'].startswith('private, no-store')
            frame_id = viewed.json()['frame']['observation_id']
            refused = await client.get(base+'/human/'+sentinel+'/observation', params={'private-query':sentinel})
            assert refused.status_code == 401
            missing = await client.get(base+'/human/'+sentinel+'/observation', headers=AUTH)
            assert missing.status_code == 404
            malformed = await client.post(base+'/human/control/actions', headers=AUTH, json={'private':sentinel})
            assert malformed.status_code == 400 and sentinel not in malformed.text
            feature._cfg['granted_actions'].remove(VisualAction.HUMAN_INPUT.value)
            denied = await client.post(base+'/human/control/actions', headers=AUTH, json={})
            assert denied.status_code == 403
            feature._cfg['granted_actions'].append(VisualAction.HUMAN_INPUT.value)
            async def original_failure(*args):
                raise RuntimeError(sentinel)
            monkeypatch.setattr(extension, 'observe', original_failure)
            failed = await client.get(base+'/human/control/observation', headers=AUTH)
            assert failed.status_code == 500 and sentinel not in failed.text
            healthy = await client.get('/health')
            assert healthy.status_code == 200
        spans = exporter.get_finished_spans()
        assert any(span.kind.name == 'SERVER' and '/health' in span.name for span in spans)
        projected = repr([(span.name, dict(span.attributes), [(event.name, dict(event.attributes)) for event in span.events]) for span in spans])
        assert all(value not in projected for value in (sentinel, sid, frame_id, 'private-human-pixels', '/human/'))
        # Client-side httpx diagnostics are not the server's access/error sinks.
        records = [record.getMessage() for record in caplog.records if record.name.startswith(('uvicorn', 'kestrel_sovereign.api_errors'))]
        assert records and any('/health' in text for text in records)
        assert all(value not in '\n'.join(records) for value in (sentinel, sid, frame_id, 'private-human-pixels'))
        assert extension.effects == 0
    finally:
        runner.should_exit = True
        if serving is not None:
            await asyncio.wait_for(serving, 5)
        sock.close()
        FastAPIInstrumentor.uninstrument_app(app)
        provider.shutdown()
    assert serving is not None and serving.done() and sock.fileno() == -1
