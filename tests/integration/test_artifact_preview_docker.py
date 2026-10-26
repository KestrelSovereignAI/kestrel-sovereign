"""Real local Linux renderer, aggregate cgroup bounds, cancellation and cleanup.

Set ARTIFACT_TEST_RENDERER_IMAGE to the immutable ID built from preview_worker.
These tests never pull an image, mount a host directory or admit network egress.
"""

import asyncio
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import subprocess
from uuid import uuid4
import zlib

from PIL import Image
from pypdf import PdfWriter
import pytest
import pytest_asyncio

from kestrel_sovereign.artifacts.preview import (
    ArtifactPreviewPool, ArtifactPreviewRefused, ArtifactPreviewUnavailable, DockerArtifactRenderer,
)


@pytest_asyncio.fixture(autouse=True)
async def no_unowned_asyncio_failures():
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    failures = []
    def record(current_loop, context):
        failures.append(context)
        if previous is None:
            current_loop.default_exception_handler(context)
        else:
            previous(current_loop, context)
    loop.set_exception_handler(record)
    try:
        yield
        await asyncio.sleep(0.05)
        assert not failures, failures
    finally:
        loop.set_exception_handler(previous)


@pytest.fixture
def renderer_image():
    value = os.environ.get('ARTIFACT_TEST_RENDERER_IMAGE')
    if not value:
        pytest.skip('Real isolated renderer requires ARTIFACT_TEST_RENDERER_IMAGE immutable local ID')
    assert value.startswith('sha256:') and len(value) == 71
    return value


def png():
    out = BytesIO(); Image.new('RGB', (64, 32), 'red').save(out, format='PNG'); return out.getvalue()


def pdf():
    out = BytesIO(); writer = PdfWriter(); writer.add_blank_page(width=720, height=576); writer.write(out); return out.getvalue()


async def absent(renderer, names):
    endpoint = await renderer._endpoint()
    for name in names:
        status, present = await renderer._command(['--host', endpoint, 'container', 'ls', '--all',
                                                  '--filter', 'name=^/' + name + '$', '--format', '{{.ID}}'])
        assert status == 0 and not present.strip()


class ObservedRenderer(DockerArtifactRenderer):
    def __init__(self, image):
        super().__init__(image); self.names = []; self.descriptors = []; self.started = asyncio.Event()
    async def _command(self, arguments, **kwargs):
        if 'create' in arguments:
            self.names.append(arguments[arguments.index('--name') + 1])
        if 'start' in arguments:
            self.started.set()
        status, output = await super()._command(arguments, **kwargs)
        if 'inspect' in arguments and 'image' not in arguments and 'context' not in arguments and status == 0:
            self.descriptors.extend(json.loads(output))
        return status, output


@pytest.mark.asyncio
@pytest.mark.parametrize('media,source', [('image/png', png), ('application/pdf', pdf)])
async def test_real_worker_produces_bounded_inert_png_under_verified_cgroup(renderer_image, media, source):
    renderer = ObservedRenderer(renderer_image); pool = ArtifactPreviewPool()
    result = await renderer.render(pool, source(), media, lambda: None)
    assert 0 < result.width <= 1024 and 0 < result.height <= 1024
    assert result.png.startswith(b'\x89PNG\r\n\x1a\n') and len(result.png) < 4 * 1024 * 1024
    assert pool.active_count == 0
    host = renderer.descriptors[0]['HostConfig']
    assert host['Memory'] == host['MemorySwap'] == 512 * 1024 * 1024
    assert host['NanoCpus'] == 500000000 and host['PidsLimit'] == 16
    await absent(renderer, renderer.names)


@pytest.mark.asyncio
async def test_dimension_bomb_and_invalid_pdf_refuse_boundedly_without_source_exposure(renderer_image):
    def chunk(kind, data): return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    bomb = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', 100000, 100000, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b'')) + chunk(b'IEND', b'')
    renderer = ObservedRenderer(renderer_image); pool = ArtifactPreviewPool()
    for source, media in [(bomb, 'image/png'), (b'%PDF-1.7\nprivate-broken-source', 'application/pdf')]:
        with pytest.raises((ArtifactPreviewRefused, ArtifactPreviewUnavailable)):
            await asyncio.wait_for(renderer.render(pool, source, media, lambda: None), 20)
        assert pool.active_count == 0
    await absent(renderer, renderer.names)


@pytest.fixture
def sandbox_image_factory(renderer_image, tmp_path):
    tags = []
    root = Path(__file__).resolve().parents[2]
    def build(code):
        tag = 'kestrel-artifact-preview-test:' + uuid4().hex
        base_tag = tag + '-base'; tags.extend([tag, base_tag])
        subprocess.run(['docker', 'tag', renderer_image, base_tag], cwd=root, check=True)
        dockerfile = 'FROM ' + base_tag + '\nENTRYPOINT ' + json.dumps(['python', '-c', code]) + '\n'
        context = tmp_path / ('image' + str(len(tags))); context.mkdir(); (context / 'Dockerfile').write_text(dockerfile)
        subprocess.run(['docker', 'build', '--pull=false', '-t', tag, str(context)], cwd=root,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return subprocess.check_output(['docker', 'image', 'inspect', tag, '--format', '{{.Id}}'], cwd=root, text=True).strip()
    try:
        yield build
    finally:
        if tags:
            subprocess.run(['docker', 'image', 'rm', *tags], cwd=root,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.fixture
def slow_image(sandbox_image_factory):
    return sandbox_image_factory('import time; time.sleep(60)')



@pytest.mark.asyncio
async def test_cancellation_removes_exact_native_container_and_releases_slot(slow_image):
    renderer = ObservedRenderer(slow_image); pool = ArtifactPreviewPool()
    pending = asyncio.create_task(renderer.render(pool, png(), 'image/png', lambda: None))
    await asyncio.wait_for(renderer.started.wait(), 10)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError): await pending
    assert pool.active_count == 0
    await absent(renderer, renderer.names)


@pytest.mark.asyncio
async def test_repeated_cancellation_joins_the_same_native_container_cleanup(slow_image):
    entered, release = asyncio.Event(), asyncio.Event()
    class PausedCleanup(ObservedRenderer):
        async def _command(self, arguments, **kwargs):
            if 'rm' in arguments:
                entered.set()
                await release.wait()
            return await super()._command(arguments, **kwargs)
    renderer = PausedCleanup(slow_image)
    pool = ArtifactPreviewPool()
    pending = asyncio.create_task(renderer.render(pool, png(), 'image/png', lambda: None))
    await asyncio.wait_for(renderer.started.wait(), 10)
    pending.cancel()
    await asyncio.wait_for(entered.wait(), 10)
    pending.cancel()
    await asyncio.sleep(.02)
    assert not pending.done() and pool.active_count == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 10)
    assert pool.active_count == 0
    await absent(renderer, renderer.names)


@pytest.mark.asyncio
async def test_host_deadline_kills_a_noncooperative_decoder_process_tree(slow_image):
    renderer = ObservedRenderer(slow_image); pool = ArtifactPreviewPool()
    started = asyncio.get_running_loop().time()
    with pytest.raises((ArtifactPreviewUnavailable, ArtifactPreviewRefused)):
        await renderer.render(pool, png(), 'image/png', lambda: None)
    assert asyncio.get_running_loop().time() - started < 20
    assert pool.active_count == 0
    await absent(renderer, renderer.names)


@pytest.mark.asyncio
async def test_global_pool_bounds_two_real_process_trees_and_refuses_a_third_without_queueing(slow_image):
    pool = ArtifactPreviewPool(); renderers = [ObservedRenderer(slow_image) for _ in range(2)]
    tasks = [asyncio.create_task(renderer.render(pool, png(), 'image/png', lambda: None)) for renderer in renderers]
    try:
        await asyncio.wait_for(asyncio.gather(*(renderer.started.wait() for renderer in renderers)), 10)
        assert pool.active_count == 2
        assert sum(renderer.descriptors[0]['HostConfig']['Memory'] for renderer in renderers) == 1024 * 1024 * 1024
        with pytest.raises(ArtifactPreviewUnavailable):
            await DockerArtifactRenderer(slow_image).render(pool, png(), 'image/png', lambda: None)
    finally:
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert pool.active_count == 0
    for renderer in renderers: await absent(renderer, renderer.names)


@pytest.mark.asyncio
async def test_kernel_memory_bound_covers_decoder_descendants_and_cpu_quota(sandbox_image_factory):
    # Two children exceed their SHARED 512 MiB limit. A per-process bound would
    # admit both and fail this probe. Only synthetic pixels enter the sandbox.
    code = """import subprocess,sys,json
from pathlib import Path
children=[subprocess.Popen([sys.executable,'-c','import time; x=bytearray(360*1024*1024); time.sleep(1)']) for _ in range(2)]
for child in children: child.wait()
events=dict(line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())
print(json.dumps({'oom_kill':int(events['oom_kill']),'peak':int(Path('/sys/fs/cgroup/memory.peak').read_text()),'cpu':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'pids_max':Path('/sys/fs/cgroup/pids.max').read_text().strip()}))
"""
    class MemoryObserved(ObservedRenderer):
        proof = None
        async def _command(self, arguments, **kwargs):
            result = await super()._command(arguments, **kwargs)
            if 'start' in arguments and result[0] == 0:
                self.proof = json.loads(result[1])
            return result
    renderer=MemoryObserved(sandbox_image_factory(code));pool=ArtifactPreviewPool()
    with pytest.raises((ArtifactPreviewUnavailable, ArtifactPreviewRefused)):
        await renderer.render(pool,png(),'image/png',lambda:None)
    assert renderer.proof['oom_kill'] >= 1
    assert renderer.proof['peak'] <= 512*1024*1024
    assert renderer.proof['cpu'] == '50000 100000' and renderer.proof['pids_max'] == '16'
    assert pool.active_count == 0
    print(json.dumps({'kernel_aggregate_probe':renderer.proof}))
    await absent(renderer,renderer.names)

@pytest.mark.asyncio
async def test_lost_native_create_ack_reconciles_exact_job_before_releasing_capacity(renderer_image):
    class LostAck(ObservedRenderer):
        async def _command(self,arguments,**kwargs):
            result=await super()._command(arguments,**kwargs)
            if 'create' in arguments:raise OSError('lost create acknowledgement')
            return result
    renderer=LostAck(renderer_image);pool=ArtifactPreviewPool()
    with pytest.raises(ArtifactPreviewUnavailable):
        await renderer.render(pool,png(),'image/png',lambda:None)
    assert pool.active_count==0 and not renderer.started.is_set()
    await absent(renderer,renderer.names)


@pytest.mark.asyncio
async def test_lost_cleanup_ack_retains_capacity_until_exact_absence_is_reconciled(renderer_image):
    class LostCleanupAck(ObservedRenderer):
        lost = False
        async def _command(self, arguments, **kwargs):
            result = await super()._command(arguments, **kwargs)
            if 'rm' in arguments and not self.lost:
                self.lost = True
                raise OSError('lost native cleanup acknowledgement')
            return result
    renderer = LostCleanupAck(renderer_image)
    pool = ArtifactPreviewPool()
    with pytest.raises(ArtifactPreviewUnavailable, match='cleanup remains unknown'):
        await renderer.render(pool, png(), 'image/png', lambda: None)
    assert pool.active_count == 1
    await absent(renderer, renderer.names)
    assert not await renderer.reconcile_cleanup(pool)
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_definitive_daemon_create_refusal_frees_slot_after_exact_absence(renderer_image):
    class RejectedCreate(ObservedRenderer):
        async def _command(self, arguments, **kwargs):
            if 'create' in arguments:
                arguments = list(arguments)
                arguments[arguments.index('--runtime=runc')] = '--runtime=kestrel-proof-missing-' + uuid4().hex
            return await super()._command(arguments, **kwargs)
    renderer = RejectedCreate(renderer_image)
    pool = ArtifactPreviewPool()
    try:
        for _ in range(pool.capacity + 1):
            with pytest.raises(ArtifactPreviewUnavailable):
                await renderer.render(pool, png(), 'image/png', lambda: None)
            await absent(renderer, renderer.names)
            assert pool.active_count == 0, 'definitive daemon create refusal leaked preview capacity'
        assert len(renderer.names) == pool.capacity + 1
    finally:
        await renderer.reconcile_cleanup(pool)
        await absent(renderer, renderer.names)


@pytest.mark.asyncio
async def test_unknown_create_absence_retains_slot_until_late_native_owner_reconciles(renderer_image):
    release = asyncio.Event()
    class LateCreate(ObservedRenderer):
        pending = None
        async def _command(self, arguments, **kwargs):
            if 'create' in arguments:
                # This fixture owns a real delayed daemon call. Its lost ACK
                # must not turn an early absence observation into free capacity.
                self.names.append(arguments[arguments.index('--name') + 1])
                async def native():
                    await release.wait()
                    return await DockerArtifactRenderer._command(self, arguments, **kwargs)
                self.pending = asyncio.create_task(native())
                raise OSError('create outcome is unacknowledged')
            return await super()._command(arguments, **kwargs)
    renderer = LateCreate(renderer_image); pool = ArtifactPreviewPool(); job = pool.reserve()
    try:
        with pytest.raises(ArtifactPreviewUnavailable):
            await renderer.render(pool, png(), 'image/png', lambda: None, job=job)
        await absent(renderer, renderer.names)
        assert pool.active_count == 1, 'unacknowledged create was prematurely reclaimed'
        assert await renderer.reconcile_cleanup(pool)
        assert pool.active_count == 1
        release.set()
        status, _ = await renderer.pending
        assert status == 0
        assert not await renderer.reconcile_cleanup(pool)
        assert pool.active_count == 0
        await absent(renderer, renderer.names)
    finally:
        release.set()
        if renderer.pending is not None:
            await asyncio.gather(renderer.pending, return_exceptions=True)
        await renderer._cleanup(pool, job)
        await absent(renderer, renderer.names)


@pytest_asyncio.fixture
async def no_command_image(renderer_image, tmp_path):
    renderer = DockerArtifactRenderer(renderer_image)
    endpoint = await renderer._endpoint()
    identity = uuid4().hex;tag = 'kestrel-artifact-no-command:' + identity
    context = tmp_path / 'empty-image';context.mkdir()
    (context/'Dockerfile').write_text('FROM scratch\nLABEL io.kestrel.proof="'+identity+'"\n')
    status, _ = await renderer._command(['--host',endpoint,'build','--pull=false','--network=none','--tag',tag,str(context)],timeout=20)
    assert status == 0
    status, value = await renderer._command(['--host',endpoint,'image','inspect',tag])
    assert status == 0
    image=json.loads(value)[0];image_id=image['Id']
    assert image['Config']['Labels']['io.kestrel.proof']==identity
    try:
        yield image_id
    finally:
        status, value = await renderer._command(['--host',endpoint,'image','inspect',image_id])
        assert status == 0 and json.loads(value)[0]['Config']['Labels']['io.kestrel.proof']==identity
        status, _ = await renderer._command(['--host',endpoint,'image','rm',image_id])
        assert status == 0


@pytest.mark.asyncio
async def test_actual_daemon_no_command_refusal_does_not_consume_capacity(no_command_image):
    renderer=ObservedRenderer(no_command_image);pool=ArtifactPreviewPool()
    try:
        for _ in range(pool.capacity+1):
            with pytest.raises(ArtifactPreviewUnavailable):
                await renderer.render(pool,png(),'image/png',lambda:None)
            await absent(renderer,renderer.names)
            assert pool.active_count==0,'definitive no-command refusal leaked preview capacity'
        assert len(renderer.names)==pool.capacity+1
    finally:
        await renderer.reconcile_cleanup(pool)
        await absent(renderer,renderer.names)
