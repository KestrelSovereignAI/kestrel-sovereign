"""Verify and backfill ``embedding_vec`` from the legacy ``embedding`` column.

Phase 1 of retiring the legacy raw-SQL ``embedding`` column on
``saved_items`` and ``document_chunks`` (#3402, parent #2684). The
inventory of every legacy reader and writer lives in
``docs/architecture/storage/EMBEDDING_COLUMN_RETIREMENT.md``.

The Phase-2 startup migrations in ``sqla/migrations.py`` copy legacy
vectors only when they first *create* ``embedding_vec``. A row that is
later left with only the legacy value (a failed dual-write, a PG
database whose column was created after the row was written) is never
repaired by them. :func:`backfill_embedding_vec` repairs those rows and
:func:`verify_embedding_vec` reports on them.

Neither function changes the schema, touches the legacy ``embedding``
value, or overwrites a non-NULL ``embedding_vec``. After
``kestrel embeddings reindex`` the two columns legitimately disagree
(reindex rewrites ``embedding_vec`` only), so a disagreement is
counted, never repaired.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from .embedding_reindex import _format_pgvector_text

if TYPE_CHECKING:
    from .async_database import AsyncDatabase

logger = logging.getLogger(__name__)

# Tables that still carry the legacy ``embedding`` column, mapped to
# their primary-key column. Table and column names are interpolated
# into SQL, so only these literals are accepted.
LEGACY_EMBEDDING_TABLES = {
    "saved_items": "id",
    "document_chunks": "chunk_id",
}

DEFAULT_BATCH_SIZE = 500


class EmbeddingVecBackfillError(RuntimeError):
    """The ``embedding_vec`` column cannot be verified or backfilled."""


@dataclass(frozen=True)
class EmbeddingVecReport:
    """State of one table's two embedding columns.

    Counts describe the table after the call. ``rows_backfilled`` and
    ``rows_unbackfillable`` describe the rows the call scanned.
    """

    table: str
    embedding_vec_present: bool
    total_rows: int
    rows_with_both: int
    rows_missing_embedding_vec: int
    rows_embedding_vec_only: int
    rows_disagreeing: int
    rows_backfilled: int
    rows_unbackfillable: int


@dataclass(frozen=True)
class _VecColumn:
    """The ``embedding_vec`` column as the backend declares it."""

    present: bool
    # Declared ``vector(N)`` width on PostgreSQL; ``None`` when the
    # column is unconstrained or the backend stores bytes.
    dimension: Optional[int] = None


async def verify_embedding_vec(db: "AsyncDatabase", table: str) -> EmbeddingVecReport:
    """Report on *table*'s embedding columns without writing anything."""
    return await _run(db, table, batch_size=DEFAULT_BATCH_SIZE, write=False)


async def backfill_embedding_vec(
    db: "AsyncDatabase",
    table: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingVecReport:
    """Copy legacy ``embedding`` into ``embedding_vec`` where it is NULL.

    Rows are walked by primary key in batches of *batch_size*, and each
    batch commits in its own transaction, so an interrupted run keeps
    its finished batches and a re-run resumes with the rest. Every
    ``UPDATE`` re-checks ``embedding_vec IS NULL``: a concurrent
    dual-write is never overwritten and a second run writes 0 rows.

    Rows whose legacy bytes are not whole float32 values, whose width
    differs from a PostgreSQL ``vector(N)`` column, or that hold a NaN or
    infinite component, are left alone and counted in
    ``rows_unbackfillable``. When ``embedding_vec`` does not
    exist yet (PostgreSQL defers creating it until a legacy row exists),
    nothing is written: the startup migration owns creating the column.
    """
    return await _run(db, table, batch_size=batch_size, write=True)


async def _run(
    db: "AsyncDatabase", table: str, *, batch_size: int, write: bool
) -> EmbeddingVecReport:
    if table not in LEGACY_EMBEDDING_TABLES:
        raise ValueError(
            f"table must be one of {sorted(LEGACY_EMBEDDING_TABLES)}, got {table!r}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    id_col = LEGACY_EMBEDDING_TABLES[table]
    is_postgres = _is_postgres(db)
    column = await _vec_column(db, table, is_postgres)

    if not column.present:
        total, legacy = await _fetch_counts(
            db, f"SELECT COUNT(*), {_count_when('embedding IS NOT NULL')} FROM {table}"
        )
        logger.warning(
            "%s.embedding_vec does not exist; %d legacy embeddings cannot be "
            "backfilled until the startup migration creates it.",
            table, legacy,
        )
        return EmbeddingVecReport(
            table=table,
            embedding_vec_present=False,
            total_rows=total,
            rows_with_both=0,
            rows_missing_embedding_vec=legacy,
            rows_embedding_vec_only=0,
            rows_disagreeing=0,
            rows_backfilled=0,
            rows_unbackfillable=legacy,
        )

    backfilled, unbackfillable = await _backfill_missing(
        db, table, id_col, column, is_postgres, batch_size=batch_size, write=write
    )
    total, both, missing, vec_only = await _fetch_counts(
        db,
        f"SELECT COUNT(*), "
        f"{_count_when('embedding IS NOT NULL AND embedding_vec IS NOT NULL')}, "
        f"{_count_when('embedding IS NOT NULL AND embedding_vec IS NULL')}, "
        f"{_count_when('embedding IS NULL AND embedding_vec IS NOT NULL')} "
        f"FROM {table}",
    )
    disagreeing = await _count_disagreeing(
        db, table, id_col, is_postgres, batch_size=batch_size
    )
    report = EmbeddingVecReport(
        table=table,
        embedding_vec_present=True,
        total_rows=total,
        rows_with_both=both,
        rows_missing_embedding_vec=missing,
        rows_embedding_vec_only=vec_only,
        rows_disagreeing=disagreeing,
        rows_backfilled=backfilled,
        rows_unbackfillable=unbackfillable,
    )
    logger.info("embedding_vec %s: %s", "backfill" if write else "verify", report)
    return report


def _is_postgres(db: "AsyncDatabase") -> bool:
    backend_type = getattr(db, "backend_type", None)
    if backend_type not in ("postgres", "sqlite"):
        raise EmbeddingVecBackfillError(
            f"unsupported database backend {backend_type!r}"
        )
    return backend_type == "postgres"


async def _vec_column(db: "AsyncDatabase", table: str, is_postgres: bool) -> _VecColumn:
    if not is_postgres:
        rows = await db.fetchall(
            f"SELECT 1 FROM pragma_table_info('{table}') WHERE name = 'embedding_vec'",
            (),
        )
        return _VecColumn(present=bool(rows))

    # ``to_regclass`` resolves the relation unqualified SQL will hit, the
    # same one every UPDATE below targets (see AsyncDatabase._column_exists).
    rows = await db.fetchall(
        "SELECT t.typname, a.atttypmod FROM pg_attribute a "
        "JOIN pg_type t ON t.oid = a.atttypid "
        f"WHERE a.attrelid = to_regclass('{table}') "
        "AND a.attname = 'embedding_vec' AND NOT a.attisdropped",
        (),
    )
    if not rows:
        return _VecColumn(present=False)
    type_name, typmod = rows[0]
    if type_name != "vector":
        raise EmbeddingVecBackfillError(
            f"{table}.embedding_vec is {type_name!r}, expected pgvector 'vector'"
        )
    return _VecColumn(present=True, dimension=typmod if typmod > 0 else None)


def _count_when(predicate: str) -> str:
    return f"COALESCE(SUM(CASE WHEN {predicate} THEN 1 ELSE 0 END), 0)"


async def _fetch_counts(db: "AsyncDatabase", sql: str) -> Tuple[int, ...]:
    row = await db.fetchone(sql, ())
    return tuple(int(value) for value in row)


async def _keyset_batches(
    db: "AsyncDatabase",
    table: str,
    id_col: str,
    columns: str,
    predicate: str,
    *,
    batch_size: int,
):
    """Yield batches of ``(id, *columns)`` rows matching *predicate* by id.

    Walking by primary key, not by re-querying the predicate, is what lets
    rows that stay matching (unbackfillable ones) not stall the scan.
    """
    cursor: Any = None
    while True:
        where = predicate
        params: Tuple[Any, ...] = ()
        if cursor is not None:
            where += f" AND {id_col} > ?"
            params = (cursor,)
        rows = await db.fetchall(
            f"SELECT {id_col}, {columns} FROM {table} WHERE {where} "
            f"ORDER BY {id_col} LIMIT ?",
            params + (batch_size,),
        )
        if not rows:
            return
        yield rows
        if len(rows) < batch_size:
            return
        cursor = rows[-1][0]


async def _backfill_missing(
    db: "AsyncDatabase",
    table: str,
    id_col: str,
    column: _VecColumn,
    is_postgres: bool,
    *,
    batch_size: int,
    write: bool,
) -> Tuple[int, int]:
    """Return ``(rows written, rows that cannot be written)``."""
    update = (
        f"UPDATE {table} SET embedding_vec = ?::vector "
        f"WHERE {id_col} = ? AND embedding_vec IS NULL"
        if is_postgres
        else f"UPDATE {table} SET embedding_vec = ? "
        f"WHERE {id_col} = ? AND embedding_vec IS NULL"
    )
    backfilled = 0
    unbackfillable = 0
    async for rows in _keyset_batches(
        db, table, id_col, "embedding",
        "embedding IS NOT NULL AND embedding_vec IS NULL",
        batch_size=batch_size,
    ):
        writes: List[Tuple[Any, Any]] = []
        for row_id, blob in rows:
            value = _backfill_value(bytes(blob), column, is_postgres)
            if value is None:
                logger.warning(
                    "%s row %s: legacy embedding of %d bytes is not a finite "
                    "float32 vector%s; embedding_vec left NULL.",
                    table, row_id, len(blob),
                    f" matching vector({column.dimension})"
                    if column.dimension else "",
                )
                unbackfillable += 1
                continue
            writes.append((value, row_id))
        if not write or not writes:
            continue
        async with db.transaction():
            for value, row_id in writes:
                backfilled += await db.execute(update, (value, row_id))
    return backfilled, unbackfillable


def _backfill_value(blob: bytes, column: _VecColumn, is_postgres: bool) -> Any:
    """The ``embedding_vec`` bind value for *blob*, or ``None`` if unusable.

    pgvector rejects a NaN or infinite element, which would abort the
    whole batch's transaction. SQLite would store one, but applies the same
    rule so both backends report the same rows as unbackfillable.
    """
    if not blob or len(blob) % 4:
        return None
    dimension = len(blob) // 4
    if column.dimension is not None and dimension != column.dimension:
        return None
    values = struct.unpack(f"<{dimension}f", blob)
    if not all(math.isfinite(value) for value in values):
        return None
    return _format_pgvector_text(values) if is_postgres else blob


async def _count_disagreeing(
    db: "AsyncDatabase",
    table: str,
    id_col: str,
    is_postgres: bool,
    *,
    batch_size: int,
) -> int:
    """Rows whose two columns hold different float32 vectors."""
    vec_expr = "embedding_vec::text" if is_postgres else "embedding_vec"
    disagreeing = 0
    async for rows in _keyset_batches(
        db, table, id_col, f"embedding, {vec_expr}",
        "embedding IS NOT NULL AND embedding_vec IS NOT NULL",
        batch_size=batch_size,
    ):
        for _row_id, legacy, vec in rows:
            vec_bytes = _pgvector_text_to_bytes(vec) if is_postgres else bytes(vec)
            if bytes(legacy) != vec_bytes:
                disagreeing += 1
    return disagreeing


def _pgvector_text_to_bytes(text: str) -> bytes:
    """Pack pgvector's ``[v1,v2,...]`` text output as float32 bytes.

    pgvector prints each element as the shortest decimal that round-trips
    its float4, so repacking reproduces the stored float32 exactly.
    """
    body = text.strip()[1:-1]
    values = [float(part) for part in body.split(",")] if body else []
    return struct.pack(f"<{len(values)}f", *values)
