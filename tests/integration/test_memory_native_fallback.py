"""Native read selection preserves live and denying ORIGINAL execution scopes."""

from contextlib import nullcontext

import pytest
from sqlalchemy import text

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    bind_execution_cleanup,
    bind_execution_custody,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.storage import sqla
from tests.integration.native_memory_original_control import actual_native_loss
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


def original_reader(n, route, *, sqlite=False):
    memory, saved = (
        (n.sqlite_memory, n.sqlite_saved) if sqlite else (n.pg_memory, n.pg_saved)
    )
    return (
        memory.retriever
        if route == "retriever"
        else memory.consolidator
        if route == "consolidator"
        else saved
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["retriever", "consolidator", "saved-items"])
@pytest.mark.parametrize(
    "state", ["live", "revoked", "unknown", "ambient-denied", "cleanup"]
)
@pytest.mark.parametrize("cached_unavailable", [False, True])
async def test_native_reader_selects_original_scope_before_availability(
    native_memory_owners, route, state, cached_unavailable
):
    n = native_memory_owners
    reader = original_reader(n, route)
    if cached_unavailable:
        # Actual unbound from_pool limitation, not a manufactured cache flag.
        assert reader._get_vector_session_factory() is None
        assert reader._sqla_factory_unavailable
    scope = ExecutionCustody(GenerationFence())
    n.backend._execution_custody = scope
    if state == "revoked":
        scope.revoke("original runtime revoked")
    elif state == "unknown":
        # Declared legacy outcome input; not driver-generated UNKNOWN evidence.
        scope.preserve_commit_uncertainty("unknown")
    context = (
        bind_execution_cleanup(n.backend)
        if state == "cleanup"
        else bind_execution_custody(GenerationFence())
        if state == "ambient-denied"
        else nullcontext()
    )
    failure = result = None
    with context as caller:
        if state == "ambient-denied":
            caller.revoke("original narrow caller revoked")
        try:
            result = reader._get_vector_session_factory()
        except BaseException as error:
            failure = error
    assert n.backend._execution_custody is scope
    if state == "live":
        assert failure is None and result is None
        assert (
            await n.backend.fetch_val("SELECT generation FROM authority WHERE id=1")
            == 1
        )
    else:
        assert failure is not None and is_execution_control_error(failure), (
            f"{state} became unavailable {result!r}"
        )
        if state == "unknown":
            assert execution_commit_outcome(failure) == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["consolidator", "saved-items"])
async def test_warmed_reader_cannot_publish_unfenced_factory_in_captured_scope(
    native_memory_owners, route
):
    n = native_memory_owners
    reader = original_reader(n, route, sqlite=True)
    original = reader._get_vector_session_factory()
    assert original is n.sqlite_factory
    # Declared captured admission, capability-only conformance: this does NOT
    # claim a PostgreSQL fence can authorize a SQLite mutation.
    backend = n.sqlite_saved.db.backend
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    try:
        assert reader._get_vector_session_factory() is None
        assert backend._execution_custody is scope
    finally:
        backend._execution_custody = None


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["retriever", "consolidator", "saved-items"])
async def test_unbound_factory_callback_keeps_original_native_denial(
    native_memory_owners, monkeypatch, route
):
    n = native_memory_owners
    reader = original_reader(n, route, sqlite=True)
    actual, originals, returned = sqla.make_session_factory, [], []

    def real_factory(*args, **kwargs):
        result = actual(*args, **kwargs)
        returned.append(result)
        actual_native_loss(n.lease, originals)
        return result

    monkeypatch.setattr(sqla, "make_session_factory", real_factory)
    failure = None
    try:
        reader._get_vector_session_factory()
    except BaseException as error:
        failure = error
    assert returned == [n.sqlite_factory] and originals
    assert failure is originals[0], "unexpected actual native denial became unavailable"
    assert not getattr(reader, "_sqla_factory_unavailable", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["revoked", "unknown", "cleanup"])
async def test_canonical_native_selection_validates_every_original_scope(
    native_memory_owners, state
):
    n = native_memory_owners
    scope = ExecutionCustody(GenerationFence())
    n.backend._execution_custody = scope
    if state == "revoked":
        scope.revoke("original runtime revoked")
    elif state == "unknown":
        scope.preserve_commit_uncertainty("unknown")
    context = bind_execution_cleanup(n.backend) if state == "cleanup" else nullcontext()
    with context:
        with pytest.raises(BaseException) as caught:
            sqla.make_session_factory(n.pg_storage.db, native_read_fallback=True)
    assert is_execution_control_error(caught.value)
    if state == "unknown":
        assert execution_commit_outcome(caught.value) == "unknown"


@pytest.mark.asyncio
async def test_canonical_native_selection_keeps_unbound_factory_available(
    native_memory_owners,
):
    n = native_memory_owners
    factory = sqla.make_session_factory(n.sqlite_saved.db, native_read_fallback=True)
    assert factory is n.sqlite_factory
    async with factory.read_session() as session:
        assert (await session.execute(text("SELECT 1"))).scalar_one() == 1
