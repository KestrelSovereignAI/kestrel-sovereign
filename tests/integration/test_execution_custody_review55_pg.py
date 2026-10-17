"""Native PostgreSQL commit/release ACKs at signed reanchor and Stop."""
import asyncio
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_custody_snapshot, execution_commit_outcome
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.stop import StopReceiptStore, StopDoor
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_constitution_turn_admission import _ready_turn
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files
from tests.unit.test_stop_receipts import _request, _outcomes
from tests.unit.test_execution_custody_review43_native import carries

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("route", ["reanchor", "genesis", "stop"])
@pytest.mark.parametrize("boundary", ["commit", "release"])
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_actual_control_publication_does_not_become_success_or_error_text(native_pg, tmp_path, monkeypatch, route, boundary, carrier):
    backend, observer, schema = native_pg
    storage = AsyncStorage(backend=backend, agent_id="did:test:review55:" + uuid4().hex)
    await storage.initialize()
    agent = None
    if route in {"reanchor", "genesis"}:
        agent, content = await _ready_turn(storage)
    if route == "genesis":
        node = await storage.get_node(storage.agent_id)
        node.properties.pop("genesis_audit", None)
        await storage.add_node(node)
        async def offline_auditor(prompt):
            return {"risk_level": 1, "reasoning": "Deterministic offline audit receipt"}
        agent.get_audit_response = offline_auditor
    elif route == "reanchor":
        replacement = content + b"\nSigned offline review55 replacement.\n"
        path = tmp_path / "KESTREL_CONSTITUTION.md"
        path.write_bytes(replacement)
        import kestrel_sovereign.config as config
        monkeypatch.setattr(config, "CONSTITUTION_PATH", str(path))
        artifact, root = _write_authority_files(tmp_path, replacement)
        agent._sovereign_trust_root_path = root
    else:
        store = StopReceiptStore(storage.db)
        await store.ensure_schema()
        request = _request()
    native_execute = asyncpg.Connection.execute
    native_commit = asyncpg.transaction.Transaction.__aexit__
    native_release = asyncpg.pool.PoolAcquireContext.__aexit__
    written, acknowledgements = [], []
    original = OSError("native control publication ACK") if carrier == "error" else asyncio.CancelledError("native control publication ACK")
    async def observed_execute(connection, query, *args, **kwargs):
        result = await native_execute(connection, query, *args, **kwargs)
        if (route in {"reanchor", "genesis"} and "INSERT INTO graph_nodes" in query) or (route == "stop" and "INSERT INTO stop_receipts" in query):
            written.append(True)
        return result
    async def lost_commit(transaction, error_type, error, traceback):
        result = await native_commit(transaction, error_type, error, traceback)
        if boundary == "commit" and written and not acknowledgements and error is None and not transaction._nested:
            acknowledgements.append(True)
            raise original
        return result
    async def lost_release(checkout, error_type, error, traceback):
        result = await native_release(checkout, error_type, error, traceback)
        if boundary == "release" and written and not acknowledgements and error is None:
            acknowledgements.append(True)
            raise original
        return result
    monkeypatch.setattr(asyncpg.Connection, "execute", observed_execute)
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", lost_commit)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lost_release)
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    try:
        with bind_execution_custody_snapshot((scope,)):
            with pytest.raises(BaseException) as caught:
                if route == "reanchor":
                    await ConstitutionMixin.reanchor_constitution(agent, amendment_artifact_path=str(artifact))
                elif route == "genesis":
                    await ConstitutionMixin.perform_genesis_audit(agent)
                else:
                    await store.persist(request, _outcomes(request), door=StopDoor.AGENT)
        assert written and acknowledgements == [True]
        assert carries(caught.value, original)
        assert execution_commit_outcome(caught.value) == ("unknown" if boundary == "commit" else "committed")
        if route == "reanchor":
            import hashlib, json
            properties = await observer.fetchval(f'SELECT properties FROM "{schema}".graph_nodes WHERE node_id=$1', storage.agent_id)
            assert json.loads(properties)["constitution_hash"] == hashlib.sha256(replacement).hexdigest()
            assert agent._safe_mode is True
        elif route == "genesis":
            import json
            properties = await observer.fetchval(f'SELECT properties FROM "{schema}".graph_nodes WHERE node_id=$1', storage.agent_id)
            assert json.loads(properties)["genesis_audit"]["status"] == "passed"
        else:
            assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".stop_receipts') == 1
    finally:
        monkeypatch.undo()
        backend._execution_custody = None
        await storage.close()
