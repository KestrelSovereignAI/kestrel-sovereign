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
