"""#1653: vector BLOBs must be explicit little-endian with an alignment guard.

saved_items_store and async_rag_store previously packed/unpacked with native
byte order and ``len(data) // 4`` (silent truncation), so a big-endian host or
a cross-architecture migration produced byte-swapped garbage and a misaligned
row silently lost its tail. They now mirror async_conversation_store: explicit
``<`` little-endian + skip-on-misalignment.
"""
import struct

import pytest

from kestrel_sovereign.storage.saved_items_store import (
    _serialize_embedding as saved_items_serialize,
    _deserialize_embedding as saved_items_deserialize,
)
from kestrel_sovereign.storage.async_rag_store import (
    _serialize_embedding as rag_serialize,
    _deserialize_embedding as rag_deserialize,
)

_STORES = [
    (saved_items_serialize, saved_items_deserialize),
    (rag_serialize, rag_deserialize),
]


@pytest.mark.parametrize("serialize,deserialize", _STORES)
def test_roundtrip_is_explicit_little_endian(serialize, deserialize):
    vec = [0.0, 1.5, -2.25, 3.125, 42.0]
    blob = serialize(vec)
    # Encoding is little-endian regardless of host byte order.
    assert blob == struct.pack(f"<{len(vec)}f", *vec)
    assert deserialize(blob) == pytest.approx(vec)


@pytest.mark.parametrize("serialize,deserialize", _STORES)
def test_deserialize_skips_misaligned_blob_instead_of_truncating(serialize, deserialize):
    # 10 bytes is not a multiple of 4 — must return [] (skip), not // 4
    # truncate to two floats of noise.
    assert deserialize(b"\x00" * 10) == []


@pytest.mark.parametrize("serialize,deserialize", _STORES)
def test_empty_embedding_roundtrips(serialize, deserialize):
    assert serialize([]) == b""
    assert deserialize(b"") == []
