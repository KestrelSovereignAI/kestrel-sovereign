"""Native release-receipt and owner-aware retry scheduling regressions."""

import asyncpg
import pytest
import asyncio

from kestrel_sovereign.execution_custody import ExecutionCustody
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as _native_pg
from tests.integration.test_execution_custody_review20_pg import setup_delivery, release, claim
from tests.integration.test_execution_custody_review18_pg import join_owned

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("kind", ["nack", "managed", "hold"])
@pytest.mark.parametrize("exact", [False, True])
async def test_same_owner_cannot_replace_token_before_actual_release_ack(native_pg, monkeypatch, kind, exact):
    backend, control, _ = native_pg
    agent, dispatcher, delivery, _ = await setup_delivery(native_pg)
    agent._execution_custody = backend._execution_custody = ExecutionCustody(GenerationFence())
    native_exit = asyncpg.transaction.Transaction.__aexit__
    injected = False
    claims = []

    async def delay_ack(transaction, error_type, error, traceback):
        nonlocal injected
        result = await native_exit(transaction, error_type, error, traceback)
        if error_type is None and not injected:
            injected = True
            # COMMIT really happened. The same resident store/drainer sees
            # RETRY while its original release is still awaiting its receipt.
            claims.append(await claim(dispatcher._durable_store, agent, delivery, "dispatcher:original", exact))
            raise OSError("actual release ACK lost after same-owner claim opportunity")
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", delay_ack)
    try:
        with pytest.raises(BaseException) as caught:
            await release(dispatcher._durable_store, agent, delivery, kind)
        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_exit)
        assert claims == [None], "unsettled original release allowed token replacement"
        assert await control.fetchval("SELECT lease_token FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == delivery.lease_token
        await dispatcher._terminalize_failed_cognition(delivery, caught.value)
        assert not dispatcher._retained_cognition_control_debt
        assert await control.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "failed"
    finally:
        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_exit)
        await join_owned(agent, dispatcher)


async def test_actual_standalone_shutdown_purges_scoped_leak_while_work_stays_retired(native_pg, tmp_path, monkeypatch):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError, ProcessRuntimeExecutionFence
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    from kestrel_sovereign.privacy import PrivacyMode
    from tests.unit.test_agent_boot_phases import _make_agent

    backend, control, schema = native_pg
    agent = _make_agent(tmp_path)
    service = agent.llm_service
    await backend.execute("CREATE TABLE conversation_history (id INTEGER PRIMARY KEY, agent_id TEXT, role TEXT, content TEXT, metadata TEXT, created_at TIMESTAMP, deleted_at TIMESTAMP, lexical_index_id TEXT)")
    await backend.execute("CREATE TABLE conversation_lexical_tokens (agent_id TEXT, lexical_index_id TEXT, token_hash TEXT)")
    await backend.execute("CREATE TABLE graph_nodes (node_id TEXT PRIMARY KEY, properties TEXT, node_type TEXT DEFAULT 'memory', label TEXT DEFAULT 'test memory')")
    await backend.execute("CREATE TABLE graph_node_owners (node_id TEXT, agent_id TEXT)")
    await backend.execute("CREATE TABLE graph_edges (source_id TEXT, target_id TEXT, label TEXT)")
    await backend.execute("CREATE TABLE graph_edge_owners (source_id TEXT, target_id TEXT, label TEXT, agent_id TEXT)")
    await backend.execute("CREATE TABLE channel_messages (id INTEGER PRIMARY KEY, agent_id TEXT, created_at TEXT)")
    await backend.execute("CREATE TABLE conversation_session_watermarks (agent_id TEXT PRIMARY KEY)")
    await backend.execute("CREATE TABLE conversation_sessions (agent_id TEXT)")
    await backend.execute("CREATE TABLE conversation_history_changes (agent_id TEXT)")
    storage = AsyncStorage(backend=backend, agent_id=agent.did)
    storage.db = AsyncDatabase(backend)
    storage.db._initialized = storage._initialized = True
    storage.conversation = AsyncConversationStore(storage.db, agent_id=agent.did)
    storage.graph = AsyncGraphStore(storage.db, agent_id=agent.did)
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
    wrapper._entered_ephemeral_at = "2026-10-10 12:00:00"
    await control.execute(f'SET search_path TO "{schema}"')
    await control.execute("INSERT INTO conversation_history VALUES (1,$1,'user','legitimate','{}','2026-10-10 11:00:00',NULL,'keep'),(2,$1,'user','leaked','{}','2026-10-10 13:00:00',NULL,'purge'),(3,'foreign','user','foreign','{}','2026-10-10 13:00:00',NULL,'foreign')", agent.did)
    await control.execute("INSERT INTO conversation_lexical_tokens VALUES ($1,'keep','kept'),($1,'purge','leaked'),('foreign','foreign','foreign')", agent.did)
    await control.execute("INSERT INTO graph_nodes (node_id,properties) VALUES ('old','{\"created_at\":\"2026-10-10T11:00:00Z\"}'),('leak','{\"created_at\":\"2026-10-10T13:00:00Z\"}'),('shared','{\"created_at\":\"2026-10-10T13:00:00Z\"}')")
    await control.execute("INSERT INTO graph_node_owners VALUES ('old',$1),('leak',$1),('shared',$1),('shared','foreign')", agent.did)
    await control.execute("INSERT INTO channel_messages VALUES (1,$1,'2026-10-10T11:00:00Z'),(2,$1,'2026-10-10T13:00:00Z'),(3,'foreign','2026-10-10T13:00:00Z')", agent.did)
    for table in ("conversation_session_watermarks", "conversation_sessions", "conversation_history_changes"):
        await control.execute(f"INSERT INTO {table} VALUES ($1),('foreign')", agent.did)
    scope = ExecutionCustody(ProcessRuntimeExecutionFence(agent.did))
    agent._execution_custody = backend._execution_custody = scope
    agent.storage = wrapper
    agent._privacy_mode = PrivacyMode.EPHEMERAL
    agent.features = {}
    agent.llm_service = agent.task_manager = agent.memory_system = agent._sync_service = None
    # No active feature/provider is required to exercise the actual lifecycle.
    agent.dispatcher = None
    native_execute = asyncpg.Connection.execute
    denied_during_purge = []

    async def observe_denial(connection, query, *params, **kwargs):
        if query.startswith("DELETE FROM conversation_history "):
            with pytest.raises(ExecutionAuthorityError, match="retired"):
                await backend.execute("INSERT INTO effects VALUES (91, 'ordinary shutdown escape')")
            denied_during_purge.append(True)
        return await native_execute(connection, query, *params, **kwargs)

    monkeypatch.setattr(asyncpg.Connection, "execute", observe_denial)
    try:
        await agent.shutdown()
        if agent._durable_shutdown_continuation is not None:
            await agent._durable_shutdown_continuation
        rows = await control.fetch("SELECT id FROM conversation_history ORDER BY id")
        assert [row[0] for row in rows] == [1, 3], "mandatory shutdown purge was denied after process retirement"
        assert [row[0] for row in await control.fetch("SELECT token_hash FROM conversation_lexical_tokens ORDER BY token_hash")] == ["foreign", "kept"]
        assert [row[0] for row in await control.fetch("SELECT node_id FROM graph_nodes ORDER BY node_id")] == ["old", "shared"]
        assert await control.fetchval("SELECT count(*) FROM graph_node_owners WHERE node_id='shared' AND agent_id=$1", agent.did) == 0
        assert [row[0] for row in await control.fetch("SELECT id FROM channel_messages ORDER BY id")] == [1, 3]
        for table in ("conversation_session_watermarks", "conversation_sessions"):
            assert [row[0] for row in await control.fetch(f"SELECT agent_id FROM {table}")] == ["foreign"]
        assert await control.fetchval("SELECT count(*) FROM conversation_history_changes WHERE agent_id=$1", agent.did) == 1
        assert denied_during_purge
        assert await control.fetchval("SELECT count(*) FROM effects") == 0
        with pytest.raises(ExecutionAuthorityError, match="retired"):
            scope.require_work()
        assert backend._pool is None
    finally:
        monkeypatch.setattr(asyncpg.Connection, "execute", native_execute)
        tasks = tuple(agent._background_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await service.close()


async def test_sibling_drain_query_excludes_live_owner_retry_before_limit(native_pg):
    _, control, _ = native_pg
    agent, dispatcher, delivery, _ = await setup_delivery(native_pg)
    try:
        # The drainer can rebuild this source without a stored caller. The
        # fixture's payload-elided row must enter the executable window.
        agent.rehydrate_durable_cognition_signal = lambda *args: None
        agent.durable_rehydratable_sources = frozenset({"channel.message"})
        assert await release(dispatcher._durable_store, agent, delivery, "managed")
        dispatcher._durable_delivery_owner = "dispatcher:sibling"
        assert await dispatcher._drainable_durable_deliveries(delivery.consumer_id, limit=1) == [], "claim-ineligible live-owner retry must not re-arm an immediate scan"
        delays = []
        dispatcher._schedule_durable_cognition_drain = lambda consumer, *, delay: delays.append(delay)
        await dispatcher._schedule_next_durable_cognition_drain(delivery.consumer_id)
        assert delays and min(delays) >= 1, "excluded rows require bounded polling for eventual owner recovery"
        await control.execute("UPDATE durable_signal_runtime_owners SET stopped_at=NOW() WHERE owner_id='dispatcher:original'")
        rows = await dispatcher._drainable_durable_deliveries(delivery.consumer_id, limit=1)
        assert [row.delivery_id for row in rows] == [delivery.delivery_id]
        dispatcher._durable_delivery_owner = "dispatcher:original"
    finally:
        dispatcher._durable_delivery_owner = "dispatcher:original"
        await join_owned(agent, dispatcher)


async def test_excluded_live_retry_does_not_crowd_executable_delivery_out(native_pg):
    _, control, _ = native_pg
    agent, dispatcher, delivery, _ = await setup_delivery(native_pg)
    agent.rehydrate_durable_cognition_signal = lambda *args: None
    agent.durable_rehydratable_sources = frozenset({"channel.message"})
    try:
        assert await release(dispatcher._durable_store, agent, delivery, "managed")
        await control.execute("UPDATE durable_signal_deliveries SET created_at=NOW()-INTERVAL '1 day' WHERE delivery_id='delivery1'")
        await control.execute("UPDATE durable_signal_deliveries SET status='pending',lease_owner=NULL,lease_token=NULL WHERE delivery_id='delivery2'")
        dispatcher._durable_delivery_owner = "dispatcher:sibling"
        rows = await dispatcher._drainable_durable_deliveries(delivery.consumer_id, limit=1)
        assert [row.delivery_id for row in rows] == ["delivery2"]
    finally:
        dispatcher._durable_delivery_owner = "dispatcher:original"
        await join_owned(agent, dispatcher)
