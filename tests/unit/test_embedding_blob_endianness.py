"""#1653: vector BLOBs must be explicit little-endian with an alignment guard.

saved_items_store and async_rag_store previously packed/unpacked with native
byte order and ``len(data) // 4`` (silent truncation), so a big-endian host or
a cross-architecture migration produced byte-swapped garbage and a misaligned
row silently lost its tail. They now mirror async_conversation_store: explicit
``<`` little-endian + skip-on-misalignment. Both stores decode through the one
stored-embedding reader in ``embedding_column`` (#3409).
"""
import struct

import pytest

from kestrel_sovereign.storage.embedding_column import decode_stored_embedding
from kestrel_sovereign.storage.saved_items_store import (
    _serialize_embedding as saved_items_serialize,
)
from kestrel_sovereign.storage.async_rag_store import (
    _serialize_embedding as rag_serialize,
)

_SERIALIZERS = [saved_items_serialize, rag_serialize]


@pytest.mark.parametrize("serialize", _SERIALIZERS)
def test_roundtrip_is_explicit_little_endian(serialize):
    vec = [0.0, 1.5, -2.25, 3.125, 42.0]
    blob = serialize(vec)
    # Encoding is little-endian regardless of host byte order.
    assert blob == struct.pack(f"<{len(vec)}f", *vec)
    assert decode_stored_embedding(blob) == pytest.approx(vec)


def test_decode_skips_misaligned_blob_instead_of_truncating():
    # 10 bytes is not a multiple of 4 — must be skipped, not // 4
    # truncated to two floats of noise.
    assert decode_stored_embedding(b"\x00" * 10) is None


@pytest.mark.parametrize("serialize", _SERIALIZERS)
def test_empty_embedding_is_no_embedding(serialize):
    assert serialize([]) == b""
    assert decode_stored_embedding(b"") is None
