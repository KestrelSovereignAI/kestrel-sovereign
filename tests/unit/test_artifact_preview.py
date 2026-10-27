"""Bounded derivative wire format and process-wide admission, with no decoder."""
import struct
import zlib
from types import SimpleNamespace

import pytest

from kestrel_sovereign.artifacts.preview import (
    ArtifactPreviewPool, ArtifactPreviewRefused, ArtifactPreviewUnavailable,
    DockerArtifactRenderer, MAX_PREVIEW_PIXELS, encode_preview, host_preview_pool,
)


def test_png_encoder_emits_only_rgb_ihdr_idat_iend_with_exact_bounded_pixels():
    raw = b'KAP1' + struct.pack('!II', 2, 1) + b'\xff\0\0\0\xff\0'
    result = encode_preview(raw)
    assert (result.width, result.height) == (2, 1)
    offset, chunks = 8, []
    while offset < len(result.png):
        length = struct.unpack('!I', result.png[offset:offset + 4])[0]
        kind = result.png[offset + 4:offset + 8]
        data = result.png[offset + 8:offset + 8 + length]
        crc = struct.unpack('!I', result.png[offset + 8 + length:offset + 12 + length])[0]
        assert zlib.crc32(kind + data) == crc
        chunks.append((kind, data)); offset += length + 12
    assert [kind for kind, _ in chunks] == [b'IHDR', b'IDAT', b'IEND']
    assert zlib.decompress(chunks[1][1]) == b'\0\xff\0\0\0\xff\0'


@pytest.mark.parametrize('raw', [b'', b'KAP1', b'BAD1' + bytes(8),
    b'KAP1' + struct.pack('!II', 0, 1), b'KAP1' + struct.pack('!II', 1, 0),
    b'KAP1' + struct.pack('!II', 1025, 1), b'KAP1' + struct.pack('!II', 2**32 - 1, 2**32 - 1),
    b'KAP1' + struct.pack('!II', 1, 1) + b'\0\0',
    b'KAP1' + struct.pack('!II', 1, 1) + b'\0\0\0extra'])
def test_malformed_or_oversized_raw_output_never_reaches_a_decoder(raw):
    with pytest.raises(ArtifactPreviewRefused):
        encode_preview(raw)


def test_host_pool_is_shared_across_agents_and_unknown_cleanup_keeps_capacity():
    app = SimpleNamespace(state=SimpleNamespace())
    pool = host_preview_pool(app)
    assert host_preview_pool(app) is pool
    first, second = pool.reserve(), pool.reserve()
    with pytest.raises(ArtifactPreviewUnavailable): pool.reserve()
    first.creation_started = True
    with pytest.raises(ArtifactPreviewUnavailable): pool.complete(first)
    assert pool.active_count == 2
    first.cleanup_confirmed = True; pool.complete(first); pool.complete(second)
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('image', [None, 'renderer:latest', 'renderer@sha256:bad'])
async def test_missing_or_mutable_image_fails_before_docker_or_slot_admission(image):
    pool = ArtifactPreviewPool()
    with pytest.raises(ArtifactPreviewUnavailable):
        await DockerArtifactRenderer(image).render(pool, b'x', 'image/png', lambda: None)
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("message,rejected", [
    ("Error response from daemon: no command specified\n", True),
    ("Error response from daemon: unknown or invalid runtime name: owned-test-123\n", True),
    ("Error response from daemon: internal server error\n", False),
    ("Error response from daemon: EOF\n", False),
    ("Error response from daemon: unknown or invalid runtime name: \n", False),
    ("Error response from daemon: unknown or invalid runtime name: valid\nEOF\n", False),
    ("connection reset by peer\n", False),
])
async def test_only_known_precreation_daemon_refusals_are_definitive_in_owned_native_process(monkeypatch, message, rejected):
    import sys
    from kestrel_sovereign.artifacts import preview
    original = preview.start_async_process
    children = []
    async def launch(arguments, **kwargs):
        child = await original([sys.executable, "-B", "-c", "import sys;sys.stderr.write("+repr(message)+");sys.exit(1)"], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(preview, "start_async_process", launch)
    renderer = DockerArtifactRenderer("sha256:"+"1"*64)
    if rejected:
        with pytest.raises(preview._CreateRejected, match="renderer creation was refused") as error:
            await renderer._command(["create"], classify_create=True)
        assert message.strip() not in str(error.value)
    else:
        status, output = await renderer._command(["create"], classify_create=True)
        assert status == 1 and not output
    assert len(children) == 1 and children[0].returncode == 1


@pytest.mark.asyncio
async def test_known_rejection_still_retains_slot_without_successful_absence_oracle(monkeypatch):
    renderer = DockerArtifactRenderer("sha256:"+"1"*64);pool = ArtifactPreviewPool();job = pool.reserve()
    job.creation_started = True;job.create_rejected = True;job.finished = True
    job.endpoint = "unix:///owned-test.sock";job.image_id = renderer.image
    calls = []
    async def unavailable(arguments, **kwargs):
        calls.append(arguments)
        return 1, b""
    monkeypatch.setattr(renderer, "_command", unavailable)
    await renderer._cleanup_native(pool, job)
    assert pool.active_count == 1 and not job.cleanup_confirmed, "failed absence oracle released capacity"
    assert any("ls" in call for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["label", "image", "name", "id"])
async def test_recovery_never_removes_a_foreign_or_malformed_container(monkeypatch, damage):
    renderer=DockerArtifactRenderer("sha256:"+"1"*64);pool=ArtifactPreviewPool();job=pool.reserve()
    job.creation_started=True;job.finished=True;job.endpoint="unix:///owned-test.sock";job.image_id=renderer.image
    value={"Id":"2"*64,"Name":"/"+job.name,"Config":{"Image":job.image_id,"Labels":{"io.kestrel.artifact-preview.job":job.name}}}
    if damage=="label":value["Config"]["Labels"]={}
    if damage=="image":value["Config"]["Image"]="sha256:"+"3"*64
    if damage=="name":value["Name"]="/foreign"
    if damage=="id":value["Id"]="not-a-container-id"
    calls=[]
    async def inspect_only(arguments, **kwargs):
        import json
        calls.append(arguments)
        return 0,json.dumps([value]).encode()
    monkeypatch.setattr(renderer,"_command",inspect_only)
    await renderer._cleanup_native(pool,job)
    assert pool.active_count==1 and not job.cleanup_confirmed, "foreign container reclaimed preview slot"
    assert not any("rm" in args for args in calls), "foreign cleanup was dispatched"
