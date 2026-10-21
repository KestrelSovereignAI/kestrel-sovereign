"""Actual native retrieval factories and public save/recall tools; no models."""

import asyncpg
import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.features.save.feature import SaveFeature
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.memory_system import MemorySystem
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_storage_backend_parity import _incepted_assertion_identity
from tests.unit.test_cleanup_task_admission import _partial_native_agent


@pytest.fixture
async def native_recall(native_pg, tmp_path, monkeypatch):
    backend, observer, schema = native_pg
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    did, identity = await _incepted_assertion_identity(tmp_path, "native-recall")
    storage = AsyncStorage(
        backend=backend,
        agent_id=did,
        _assertion_tenant_capability=_resolve_authenticated_agent_assertion_capability(
            did, identity
        ),
    )
    await storage.initialize()
    memory = MemorySystem(storage, did)
    await memory.initialize()
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent.did = did
    agent.storage = agent._raw_storage = storage
    feature = SaveFeature(agent)
    await feature.initialize()
    store = SavedItemsStore(storage.db, did, llm_service=agent.llm_service)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    assert await backend.fetch_val("SELECT generation FROM authority WHERE id=1") == 1
    try:
        yield backend, storage, memory, agent, feature, store
    finally:
        backend._execution_custody = None
        await storage.close()
        await agent.llm_service.close()
        await usage_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["retriever", "consolidator", "saved-items"])
async def test_native_bound_vector_factory_selects_original_native_reads(
    native_recall, monkeypatch, door
):
    backend, storage, memory, agent, feature, store = native_recall
    from kestrel_sovereign.storage import sqla

    actual = sqla.make_session_factory
    original_errors = []

    def observe_actual_factory(db, **options):
        try:
            return actual(db, **options)
        except BaseException as error:
            original_errors.append(error)
            raise

    monkeypatch.setattr(sqla, "make_session_factory", observe_actual_factory)
    owner = (
        memory.retriever
        if door == "retriever"
        else memory.consolidator
        if door == "consolidator"
        else store
    )
    failure = result = None
    try:
        result = owner._get_vector_session_factory()
    except BaseException as error:
        failure = error
    assert failure is None and result is None
    assert not original_errors, "capability selection attempted a forbidden executor"
    # The default API still refuses a custody-bound SQLAlchemy executor.
    # Native fallback selection is explicit and never renews authority.
    with pytest.raises(BaseException) as strict:
        actual(storage.db)
    assert is_execution_control_error(strict.value)
    assert await backend.fetch_val("SELECT generation FROM authority WHERE id=1") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["save", "recall"])
async def test_public_saved_item_tool_keeps_native_read_ack_control(
    native_recall, monkeypatch, door
):
    backend, storage, memory, agent, feature, store = native_recall
    observed_sql, faulted = [], []
    for name in ("fetch", "fetchrow", "fetchval", "execute"):
        actual = getattr(asyncpg.Connection, name)

        def forward(actual):
            async def operation(connection, sql, *args, **kwargs):
                result = await actual(connection, sql, *args, **kwargs)
                if "FROM saved_items" in sql:
                    observed_sql.append(sql)
                return result

            return operation

        monkeypatch.setattr(asyncpg.Connection, name, forward(actual))
    original = OSError("actual native public save/recall release ACK lost")
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__

    async def lose_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if observed_sql and not faulted and typ is None:
            faulted.append(checkout)
            raise original
        return result

    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    result = failure = None
    try:
        if door == "save":
            result = await feature.save_item(
                item_type="excerpt", name="native test", content="local fixture"
            )
        else:
            result = await feature.recall("local fixture")
    except BaseException as error:
        failure = error
    assert observed_sql and faulted, (
        "did not reach actual saved-items native query/release"
    )
    assert failure is not None and original in tuple(
        _explicit_control_chain(failure)
    ), f"native saved-items control became tool result {result!r}"
    assert is_execution_control_error(failure)
