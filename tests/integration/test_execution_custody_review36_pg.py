"""Real host inception and SQLite-to-PG birth replay retain native custody."""
from types import SimpleNamespace

import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign.execution_custody import (
    ExecutionCustody, ExecutionAuthorityError, bind_execution_custody_snapshot, is_execution_control_error,
)
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg, GenerationFence
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign._async_ownership import await_owned_task
from tests.unit.test_execution_custody_review32 import receipt
import asyncio

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def auditor(_prompt):
    return {"risk_level": 1, "reasoning": "Deterministic offline native birth probe."}


async def test_real_host_inception_under_original_pg_parent(native_pg, tmp_path):
    backend, _, _ = native_pg
    scope = ExecutionCustody(GenerationFence())
    with bind_execution_custody_snapshot((scope,)):
        credentials = await create_kestrel_identity_async(
            output_dir=str(tmp_path), agent_name="Native parent child", is_test_instance=True,
            genesis_auditor=auditor, genesis_audit_provenance="test:review36-native",
        )
        # Host inception must not globally authorize SQLite for tenant work.
        with pytest.raises(ExecutionAuthorityError):
            await AsyncDatabase.sqlite(str(tmp_path / "ordinary-tenant.db"))
        await backend.execute("INSERT INTO effects VALUES (101, 'original PG authority')")
    anchor = await AsyncDatabase.sqlite(credentials.db_path)
    try:
        assert await AsyncGraphStore(anchor).get_node(credentials.agent_did) is not None
    finally:
        await anchor.close()


@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_inception_borrower_native_close_precedes_owned_birth_database(native_pg, tmp_path, monkeypatch, kind):
    from kestrel_sovereign.llm import service as service_module
    import httpx2 as httpx
    import openai
    failure = receipt(kind)
    source = OriginalFailure.capture(failure)
    providers = []
    def construct(**kwargs):
        service = LLMService(**kwargs)
        db = kwargs["usage_db"]
        http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail("provider request during native close probe")))
        client = openai.AsyncOpenAI(api_key="offline-test-only", http_client=http,
                                    base_url="https://offline.invalid/v1", max_retries=0)
        calls = []
        providers.append((service, db, http, calls, client))
        native_close = client.close
        async def native_close_then_receipt():
            assert db.backend._connection is not None, "birth database closed below its provider borrower"
            await native_close()
            calls.append(True)
            raise failure
        monkeypatch.setattr(client, "close", native_close_then_receipt)
        original_close = service.close
        async def close_then_receipt():
            if not any(item.get("client") is client for item in service.providers):
                service.providers.append({"client": client})
            await original_close()
        monkeypatch.setattr(service, "close", close_then_receipt)
        return service
    monkeypatch.setattr(service_module, "LLMService", construct)
    async def incept():
        with bind_execution_custody_snapshot((ExecutionCustody(GenerationFence()),)):
            return await create_kestrel_identity_async(
                output_dir=str(tmp_path), agent_name="Native borrower bird", is_test_instance=True,
                genesis_auditor=auditor, genesis_audit_provenance="test:review36-borrower",
            )
    outcome = await await_owned_task(asyncio.create_task(incept()))
    retirement = getattr(outcome.error, "inception_retirement_owner", None)
    try:
        assert len(providers) == 1, "birth consumers did not share one owned accounting borrower"
        service, database, http, calls, client = providers[0]
        source.assert_retained(outcome.error, resource=client)
        assert retirement is not None
        # Even a physically closed socket does not prove that a close which
        # returned an uncertain effect receipt retired all native borrowers.
        assert retirement.database is database
        assert database.backend._connection is not None
        assert http.is_closed and calls == [True]
        original_owners = tuple(task for task in service._llm_close_tasks.values() if task is not None)
        assert original_owners and all(task.done() for task in original_owners)
        assert service.has_pending_close_owners()
        with pytest.raises(BaseException) as repeated:
            await retirement.retire()
        source.assert_retained(repeated.value, resource=client)
        assert retirement.database is database
        assert calls == [True], "reconciliation repeated the failed native close"
        assert tuple(task for task in service._llm_close_tasks.values() if task is not None) == original_owners
    finally:
        # Dispose only this fixture's original database/lock. This is not a
        # production success acknowledgement and never clears close evidence.
        for _service, database, http, _calls, _client in providers:
            await http.aclose()
            await database.close()
        if retirement is not None and retirement.directory_custody is not None:
            retirement.directory_custody.close()


@pytest.mark.parametrize("generation_changed", ["none", "before", "after_anchor"])
async def test_actual_agent_replays_birth_to_fenced_pg_destination(native_pg, tmp_path, monkeypatch, generation_changed):
    backend, control, schema = native_pg
    await control.execute(f'SET search_path TO "{schema}"')
    runtime_db = await AsyncDatabase.from_connected_backend(backend)
    credentials = await create_kestrel_identity_async(
        output_dir=str(tmp_path), agent_name="Native replay bird", is_test_instance=True,
        genesis_auditor=auditor, genesis_audit_provenance="test:review36-replay",
    )
    scope = ExecutionCustody(GenerationFence())
    service = LLMService()
    agent = KestrelAgent(did=credentials.agent_did, storage_path=credentials.db_path,
                         llm_service=service, execution_custody=scope, pg_pool=backend.operational_pool,
                         db_backend="postgres")
    agent.identity = SimpleNamespace(agent_did=credentials.agent_did)
    agent._db_backend = "postgres"
    agent._raw_storage = agent.storage = AsyncStorage(backend=backend, agent_id=agent.did)
    agent._raw_storage.db = runtime_db
    anchor_observed = []
    if generation_changed == "before":
        await control.execute("UPDATE authority SET generation=2 WHERE id=1")
    if generation_changed == "after_anchor":
        from kestrel_sovereign.identity import birth_record
        original_read = birth_record.anchor_holds_birth_record
        async def revoke_after_native_anchor_read(**kwargs):
            result = await original_read(**kwargs)
            anchor_observed.append(result)
            await control.execute("UPDATE authority SET generation=2 WHERE id=1")
            return result
        monkeypatch.setattr(birth_record, "anchor_holds_birth_record", revoke_after_native_anchor_read)
    try:
        with bind_execution_custody_snapshot((scope,)):
            if generation_changed != "none":
                with pytest.raises(BaseException) as caught:
                    await agent._reconcile_birth_record_with_runtime_database()
                assert is_execution_control_error(caught.value)
            else:
                await agent._reconcile_birth_record_with_runtime_database()
        rows = await control.fetch("SELECT node_id, label FROM graph_nodes WHERE node_type='agent'")
        if generation_changed != "none":
            assert not rows, "identity replay bypassed original PG generation fence"
            if generation_changed == "after_anchor":
                assert anchor_observed == [True]
        else:
            assert [(row[0], row[1]) for row in rows] == [(agent.did, "Native replay bird")]
            assert await control.fetchval("SELECT COUNT(*) FROM graph_edges WHERE label='governed_by'") == 1
            assert await control.fetchval("SELECT COUNT(*) FROM document_chunk_owners WHERE agent_id=$1", agent.did) > 0
    finally:
        await service.close()


@pytest.mark.parametrize("boundary", ["ack_error", "ack_cancel", "post_ack_retirement"])
async def test_local_anchor_commit_uncertainty_denies_same_native_pg_parent(native_pg, tmp_path, monkeypatch, boundary):
    import sqlite3
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError, _host_identity_anchor_access, execution_commit_outcome,
    )
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
    backend, control, schema = native_pg
    await control.execute(f'SET search_path TO "{schema}"')
    scopes = (ExecutionCustody(GenerationFence()), ExecutionCustody(GenerationFence()))
    path = tmp_path / "kestrel_prime.db"
    with bind_execution_custody_snapshot(scopes), _host_identity_anchor_access(path):
        anchor = SQLiteBackend(path)
        await anchor.connect()
        await anchor.execute("CREATE TABLE native_birth(id INTEGER PRIMARY KEY)")
        original_commit = anchor._connection.commit
        async def native_then_ack_loss():
            await original_commit()
            if boundary == "post_ack_retirement":
                scopes[-1].revoke("original parent retired after native commit")
            elif boundary == "ack_cancel":
                raise asyncio.CancelledError("native anchor COMMIT acknowledgement lost")
            else:
                raise OSError("native anchor COMMIT acknowledgement lost")
        monkeypatch.setattr(anchor._connection, "commit", native_then_ack_loss)
        try:
            outcome = await await_owned_task(asyncio.create_task(anchor.execute("INSERT INTO native_birth VALUES(1)")))
            expected = "committed" if boundary == "post_ack_retirement" else "unknown"
            assert execution_commit_outcome(outcome.error) == expected
            with sqlite3.connect(path) as observer:
                assert observer.execute("SELECT COUNT(*) FROM native_birth").fetchone()[0] == 1
            with pytest.raises(ExecutionCommitOutcomeError):
                await backend.execute("INSERT INTO effects VALUES(102, 'must not replay')")
            assert await control.fetchval("SELECT COUNT(*) FROM effects WHERE id=102") == 0
            for scope in scopes:
                with pytest.raises(ExecutionCommitOutcomeError) as caught:
                    scope.require_work()
                assert execution_commit_outcome(caught.value) == expected
        finally:
            await anchor.close()
