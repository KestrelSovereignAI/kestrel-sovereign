"""Unit tests for ``kestrel_sovereign.storage.vector``.

Covers the spec validation, both backends' filter/error contracts, and
the factory's dialect dispatch. The backends are exercised with a
mocked async session — a real DB round-trip is covered by the existing
story-archive test suite (which now consumes these primitives) and by
the property-based parity tests on the feature-pkg side.
"""

from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, Integer, String

from kestrel_sovereign.storage.vector import (
    PgVectorBackend,
    PurePythonBackend,
    VectorSearchError,
    VectorTableSpec,
    get_vector_backend,
)


# A throwaway "entity" with the InstrumentedAttribute shape SQLAlchemy
# would produce — just enough for the backends to compose where()
# clauses and label() expressions against.
class _FakeEntity:
    __tablename__ = "fake_items"
    __name__ = "FakeEntity"
    id = Column("id", Integer, primary_key=True)
    embedding = Column("embedding", String)  # type doesn't matter for these tests
    tenant_id = Column("tenant_id", String)
    item_type = Column("item_type", String)


def _spec(**overrides) -> VectorTableSpec:
    base = dict(
        entity=_FakeEntity,
        id_column=_FakeEntity.id,
        embedding_column=_FakeEntity.embedding,
        dimension=4,
        required_filter_keys=("tenant_id",),
        filter_columns={
            "tenant_id": _FakeEntity.tenant_id,
            "item_type": _FakeEntity.item_type,
        },
    )
    base.update(overrides)
    return VectorTableSpec(**base)


def _packed(*floats) -> bytes:
    return struct.pack(f"<{len(floats)}f", *floats)


def _session_with_rows(rows: list) -> MagicMock:
    """Mock session whose execute() returns rows usable by both backends."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    result = MagicMock()
    result.all = MagicMock(return_value=rows)
    session.execute = AsyncMock(return_value=result)
    return session


def _factory_for(session: MagicMock, *, dialect: str = "sqlite") -> MagicMock:
    factory = MagicMock()
    factory.engine = SimpleNamespace(dialect=SimpleNamespace(name=dialect))
    factory.read_session = MagicMock(return_value=session)
    return factory


# ----------------------------------------------------------------- VectorTableSpec


def test_spec_rejects_non_positive_dimension():
    with pytest.raises(ValueError, match="dimension"):
        VectorTableSpec(
            entity=_FakeEntity,
            id_column=_FakeEntity.id,
            embedding_column=_FakeEntity.embedding,
            dimension=0,
        )


def test_spec_rejects_tenant_key_not_in_required_keys():
    # ``tenant_id_filter_key`` must be in ``required_filter_keys`` so the
    # session-scoping code actually sees the value to use. Otherwise
    # multi-tenant scoping would silently never apply.
    with pytest.raises(ValueError, match="tenant_id_filter_key"):
        VectorTableSpec(
            entity=_FakeEntity,
            id_column=_FakeEntity.id,
            embedding_column=_FakeEntity.embedding,
            dimension=4,
            required_filter_keys=("agent_id",),
            tenant_id_filter_key="tenant_id",  # not in required_filter_keys
        )


# ----------------------------------------------------------------- factory


def test_factory_picks_pg_for_postgresql():
    session = _session_with_rows([])
    factory = _factory_for(session, dialect="postgresql")
    backend = get_vector_backend(factory, _spec())
    assert isinstance(backend, PgVectorBackend)


def test_factory_falls_back_to_pure_python_for_sqlite():
    session = _session_with_rows([])
    factory = _factory_for(session, dialect="sqlite")
    backend = get_vector_backend(factory, _spec())
    assert isinstance(backend, PurePythonBackend)


def test_factory_falls_back_to_pure_python_for_unknown_dialect():
    session = _session_with_rows([])
    factory = _factory_for(session, dialect="mysql")
    backend = get_vector_backend(factory, _spec())
    assert isinstance(backend, PurePythonBackend)


# ----------------------------------------------------------------- filter contract


@pytest.mark.asyncio
async def test_pure_python_raises_when_required_filter_missing():
    session = _session_with_rows([])
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec())
    with pytest.raises(VectorSearchError, match="tenant_id"):
        await backend.knn(_packed(1.0, 0, 0, 0), k=5, filter={})


@pytest.mark.asyncio
async def test_pure_python_raises_when_filter_is_none():
    session = _session_with_rows([])
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec())
    with pytest.raises(VectorSearchError, match="tenant_id"):
        await backend.knn(_packed(1.0, 0, 0, 0), k=5, filter=None)


@pytest.mark.asyncio
async def test_pure_python_raises_on_wrong_size_embedding():
    session = _session_with_rows([])
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec())
    with pytest.raises(VectorSearchError, match="bytes"):
        # Dim=4 → 16 bytes expected; passing 8 (=2 floats).
        await backend.knn(_packed(1.0, 0.0), k=5, filter={"tenant_id": "t1"})


# ----------------------------------------------------------------- unfiltered specs


def _spec_no_filters() -> VectorTableSpec:
    return VectorTableSpec(
        entity=_FakeEntity,
        id_column=_FakeEntity.id,
        embedding_column=_FakeEntity.embedding,
        dimension=4,
    )


@pytest.mark.asyncio
async def test_pure_python_accepts_filter_none_when_no_required_filters():
    """Regression: a spec with no required filters must accept
    ``filter=None``. Earlier the backend asserted ``filter is not None``
    after validation, which made unfiltered single-tenant tables
    unusable. (Caught by codex review on the sovereign vector-lift PR.)
    """
    rows = [("row-A", _packed(1, 0, 0, 0))]
    session = _session_with_rows(rows)
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec_no_filters())
    out = await backend.knn(_packed(1, 0, 0, 0), k=1, filter=None)
    assert out[0][0] == "row-A"


@pytest.mark.asyncio
async def test_pg_accepts_filter_none_when_no_required_filters():
    """Same regression for PgVectorBackend."""
    rows = [("row-A", 0.9)]
    session = _session_with_rows(rows)
    factory = _factory_for(session, dialect="postgresql")
    backend = PgVectorBackend(factory, _spec_no_filters())
    out = await backend.knn(_packed(1, 0, 0, 0), k=1, filter=None)
    assert out == [("row-A", 0.9)]


# ----------------------------------------------------------------- zero-norm short-circuit


@pytest.mark.asyncio
async def test_pure_python_returns_empty_on_zero_norm_query():
    session = _session_with_rows([("row-1", _packed(1, 0, 0, 0))])
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec())
    # All-zero query vector → cosine is NaN; backend short-circuits.
    out = await backend.knn(_packed(0, 0, 0, 0), k=5, filter={"tenant_id": "t1"})
    assert out == []
    # Session not opened — short-circuit happens before any DB I/O.
    session.execute.assert_not_called()


# ----------------------------------------------------------------- kNN behavior


@pytest.mark.asyncio
async def test_pure_python_ranks_by_cosine_similarity():
    # Query: e1=[1,0,0,0]. Rows: row-A=[1,0,0,0] (sim 1.0), row-B=[0,1,0,0]
    # (sim 0.0), row-C=[0.5,0,0,0] (sim 1.0 — normalized direction same as A).
    rows = [
        ("row-A", _packed(1, 0, 0, 0)),
        ("row-B", _packed(0, 1, 0, 0)),
        ("row-C", _packed(0.5, 0, 0, 0)),
    ]
    session = _session_with_rows(rows)
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec())

    out = await backend.knn(_packed(1, 0, 0, 0), k=5, filter={"tenant_id": "t1"})
    out_ids = [pair[0] for pair in out]
    assert out_ids[0] in {"row-A", "row-C"}  # both have similarity 1.0
    assert out[-1][0] == "row-B"
    # Each similarity is in [-1, 1]. Floating-point math on row-B's dot
    # product can land at exactly 0.0 or a tiny epsilon either side.
    assert all(-1.0 <= sim <= 1.0 for _, sim in out)


@pytest.mark.asyncio
async def test_pure_python_respects_k():
    rows = [(f"row-{i}", _packed(1, 0, 0, 0)) for i in range(10)]
    session = _session_with_rows(rows)
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec())

    out = await backend.knn(_packed(1, 0, 0, 0), k=3, filter={"tenant_id": "t1"})
    assert len(out) == 3


@pytest.mark.asyncio
async def test_pure_python_returns_empty_when_no_rows():
    session = _session_with_rows([])
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec())
    out = await backend.knn(_packed(1, 0, 0, 0), k=5, filter={"tenant_id": "t1"})
    assert out == []


@pytest.mark.asyncio
async def test_pure_python_skips_zero_norm_rows():
    rows = [
        ("good", _packed(1, 0, 0, 0)),
        ("zero", _packed(0, 0, 0, 0)),
    ]
    session = _session_with_rows(rows)
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec())

    out = await backend.knn(_packed(1, 0, 0, 0), k=5, filter={"tenant_id": "t1"})
    assert [pair[0] for pair in out] == ["good"]


@pytest.mark.asyncio
async def test_pure_python_accepts_ndarray_embedding_from_orm():
    """Regression: pgvector's SQLAlchemy adapter deserializes embeddings
    to ``numpy.ndarray``, not list or bytes. The earlier
    ``if not embedding_data`` truthiness check raised ``ValueError`` on
    every ndarray row, aborting the entire kNN scan. (Caught by codex
    review on the sovereign vector-lift PR.)
    """
    import numpy as np

    rows = [("row-A", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))]
    session = _session_with_rows(rows)
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec())

    out = await backend.knn(_packed(1, 0, 0, 0), k=5, filter={"tenant_id": "t1"})
    assert out[0][0] == "row-A"
    assert out[0][1] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_pure_python_accepts_list_embedding_from_orm():
    # When PortableVector deserializes to a list (PG path under ORM),
    # backend should consume it directly without trying to unpack bytes.
    rows = [("row-A", [1.0, 0.0, 0.0, 0.0])]
    session = _session_with_rows(rows)
    factory = _factory_for(session)
    backend = PurePythonBackend(factory, _spec())

    out = await backend.knn(_packed(1, 0, 0, 0), k=5, filter={"tenant_id": "t1"})
    assert out[0][0] == "row-A"
    assert out[0][1] == pytest.approx(1.0)


# ----------------------------------------------------------------- tenant-scope leak guard


@pytest.mark.asyncio
async def test_tenant_scoped_session_unwinds_context_on_session_open_failure():
    """Regression: when ``read_session().__aenter__()`` fails after
    ``TenantContext.use().__enter__()`` succeeds, ``__aexit__`` is NOT
    called by Python (the ``async with`` never bound a value), so the
    tenant context would leak into subsequent queries in the same task.
    The wrapper must catch the failure and unwind the tenant manually.
    (Caught by codex review on the sovereign vector-lift PR.)
    """
    from kestrel_sovereign.storage.vector.pg import _TenantScopedSession

    enter_calls = []
    exit_calls = []

    class _FakeTenantContext:
        def __init__(self, tid):
            self.tid = tid
            self._cm = None

        @classmethod
        def use(cls, tid):
            return cls(tid)

        def __enter__(self):
            enter_calls.append(self.tid)
            return self

        def __exit__(self, *args):
            exit_calls.append(self.tid)
            return False

    class _FailingFactory:
        def read_session(self):
            class _CM:
                async def __aenter__(self_inner):
                    raise RuntimeError("pool exhausted")
                async def __aexit__(self_inner, *a):
                    return False
            return _CM()

    # Patch the lazy ``from kestrel_feature_entities import TenantContext``
    # inside ``__aenter__`` so the wrapper picks up our fake.
    import sys, types
    fake_mod = types.ModuleType("kestrel_feature_entities")
    fake_mod.TenantContext = _FakeTenantContext
    saved = sys.modules.get("kestrel_feature_entities")
    sys.modules["kestrel_feature_entities"] = fake_mod
    try:
        wrapper = _TenantScopedSession(_FailingFactory(), tenant_id="t1")
        with pytest.raises(RuntimeError, match="pool exhausted"):
            async with wrapper:
                pass
    finally:
        if saved is None:
            del sys.modules["kestrel_feature_entities"]
        else:
            sys.modules["kestrel_feature_entities"] = saved

    # Tenant context was entered then unwound — no leak.
    assert enter_calls == ["t1"]
    assert exit_calls == ["t1"]


# ----------------------------------------------------------------- PG backend filter contract


@pytest.mark.asyncio
async def test_pg_raises_when_required_filter_missing():
    session = _session_with_rows([])
    factory = _factory_for(session, dialect="postgresql")
    backend = PgVectorBackend(factory, _spec())
    with pytest.raises(VectorSearchError, match="tenant_id"):
        await backend.knn(_packed(1, 0, 0, 0), k=5, filter={})


@pytest.mark.asyncio
async def test_pg_zero_norm_short_circuits():
    session = _session_with_rows([])
    factory = _factory_for(session, dialect="postgresql")
    backend = PgVectorBackend(factory, _spec())
    out = await backend.knn(_packed(0, 0, 0, 0), k=5, filter={"tenant_id": "t1"})
    assert out == []
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_pg_returns_rows_as_tuples():
    # PG backend doesn't compute cosine in Python — it lets pgvector do
    # it server-side and just returns whatever the mocked execute()
    # yields. We just check the tuple shape is preserved.
    rows = [("row-A", 0.95), ("row-B", 0.42)]
    session = _session_with_rows(rows)
    factory = _factory_for(session, dialect="postgresql")
    backend = PgVectorBackend(factory, _spec())
    out = await backend.knn(_packed(1, 0, 0, 0), k=2, filter={"tenant_id": "t1"})
    assert out == [("row-A", 0.95), ("row-B", 0.42)]
