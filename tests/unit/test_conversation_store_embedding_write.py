"""Pure vector serialization contracts; native write cases live in integration tests."""

import struct

import pytest

from kestrel_sovereign.storage.async_conversation_store import (
    _format_pgvector_text,
    _serialize_embedding,
)


def test_serialize_embedding_round_trips_float32():
    """The PurePythonBackend's ``_unpack`` is the inverse. If
    ``_serialize_embedding`` ever drifted (endianness, dtype) the
    backend would silently read garbage."""
    vec = [0.1, -0.2, 0.3, 1e6]
    packed = _serialize_embedding(vec)
    assert len(packed) == len(vec) * 4
    unpacked = list(struct.unpack(f"<{len(vec)}f", packed))
    assert unpacked == pytest.approx(vec, rel=1e-5)


def test_format_pgvector_text_round_trips():
    """pgvector's text shape is ``[v1,v2,…]``. Asyncpg ships this as a
    string + ``::vector`` cast handles the rest."""
    out = _format_pgvector_text([0.1, -0.2, 0.3])
    assert out.startswith("[") and out.endswith("]")
    # Parseable as a comma-separated float list.
    parsed = [float(x) for x in out[1:-1].split(",")]
    assert parsed == pytest.approx([0.1, -0.2, 0.3], rel=1e-5)
