"""Portable column types for sovereign-core SQLAlchemy entities.

PortableVector dual-implements an embedding column so the same ORM
mapping works on Postgres (using pgvector's ``vector(N)``) and SQLite
(using ``LargeBinary``). At write time, the column accepts either:

- A Python ``list[float]`` (pgvector adapter handles PG natively;
  SQLite path packs into float32 bytes via ``struct``).
- ``bytes`` already in float32-packed form — passed through on
  SQLite, unpacked + handed to pgvector on PG.

At read time:

- PG returns a ``numpy.ndarray`` (pgvector's default deserialization).
  Callers that expect bytes / lists must accept ndarray too — the
  vector backends in ``kestrel_sovereign.storage.vector.python`` do.
- SQLite returns ``bytes`` (the raw BLOB) which callers can ``struct``
  unpack themselves.

Used by ``SavedItem.embedding`` and any future sovereign-core entity
that stores embeddings.
"""

from __future__ import annotations

import struct
from typing import Any, Optional

from sqlalchemy import LargeBinary
from sqlalchemy.types import TypeDecorator


class PortableVector(TypeDecorator):
    """Dialect-aware embedding column.

    On Postgres: ``vector(N)`` via the pgvector SQLAlchemy adapter.
    On SQLite (and other non-PG dialects): ``LargeBinary`` (BLOB) with
    float32 little-endian packing on the way in, raw bytes on the way
    out.

    The PG path requires the ``vector`` extension to be installed
    (``CREATE EXTENSION IF NOT EXISTS vector``). The sovereign-core
    migration that swaps ``saved_items.embedding`` to this type runs
    that statement guarded with ``IF NOT EXISTS``.
    """

    impl = LargeBinary
    cache_ok = True

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(
                f"PortableVector.dimension must be positive, got {dimension}"
            )
        self.dimension = dimension
        super().__init__()

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            # Lazy import — pgvector is a hard dep of sovereign (since
            # the vector-search-backend lift in #1445) but feature
            # pkgs may extend this code path against dialects where
            # pgvector isn't relevant. Local import keeps the failure
            # mode obvious if it's ever missing on PG.
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dimension))
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            # pgvector.sqlalchemy.Vector accepts list[float] or
            # numpy.ndarray natively. If the caller handed us bytes
            # (legacy AsyncDatabase writes), unpack them first.
            if isinstance(value, (bytes, bytearray)):
                count = len(value) // 4
                if count != self.dimension:
                    raise ValueError(
                        f"PortableVector(dim={self.dimension}) got bytes of "
                        f"length {len(value)} = {count} floats; mismatched."
                    )
                return list(struct.unpack(f"<{count}f", bytes(value)))
            return value
        # SQLite / other: store as raw float32 little-endian bytes so
        # the existing ``PurePythonBackend`` path keeps working
        # unchanged. Lists get packed; pre-packed bytes pass through.
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        # ``list``-like — could be list[float], tuple, or numpy array.
        return struct.pack(f"<{len(value)}f", *(float(v) for v in value))

    def process_result_value(self, value: Any, dialect: Any) -> Optional[Any]:
        # Read path: return the dialect-native shape. PG → ndarray (via
        # the pgvector adapter), SQLite → bytes. Both shapes are
        # accepted by PurePythonBackend (ndarray, list, and bytes are
        # all handled).
        return value
