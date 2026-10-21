"""Native SOUL/bootstrap helpers; not complete Agent genesis/READY acceptance."""

from dataclasses import dataclass
from uuid import uuid4

import asyncpg
import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@dataclass
class SoulBootstrapInput:
    agent_id: str
    storage: AsyncStorage
    context_builder: ContextBuilder


@pytest.mark.asyncio
async def test_native_postgres_soul_resource_ciphertext_round_trip(
    native_pg, monkeypatch
):
    backend, observer, schema = native_pg
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    did = "did:web:native-soul-roundtrip:" + uuid4().hex
    storage = AsyncStorage.from_backend(backend, agent_id=did)
    await storage.initialize()
    try:
        resource = await storage.promote_soul_seed(
            "private disposable native identity", created_by=did
        )
        current = await storage.get_current_agent_resource()
        assert current.id == resource.id
        assert current.content == "private disposable native identity"
        row = await storage.db.fetchone(
            "SELECT content_ciphertext FROM agent_identity_resources WHERE id = ?",
            (resource.id,),
        )
        assert isinstance(row[0], bytes)
        assert b"private disposable native identity" not in row[0]
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    ["canonical", "agent-read", "promote", "refresh", "graph-link", "bootstrap"],
)
async def test_original_native_soul_control_is_not_seed_or_bootstrap_fallback(
    native_pg, tmp_path, monkeypatch, route
):
    backend, observer, schema = native_pg
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    did = "did:web:native-soul:" + uuid4().hex
    storage = AsyncStorage.from_backend(backend, agent_id=did)
    await storage.initialize()
    await storage.add_node(
        GraphNode(
            node_id=did, node_type="agent", label="Local native input", properties={}
        )
    )
    (tmp_path / "SOUL.md").write_text(
        "# Local native seed\nPrivate fixture identity.\n"
    )
    builder = ContextBuilder(
        storage, agent_data_path=str(tmp_path), db=storage.db, agent_id=did
    )
    runtime = SoulBootstrapInput(did, storage, builder)
    original = OSError("actual SOUL native release acknowledgement lost")
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    selected = route in {"canonical", "agent-read", "bootstrap"}
    faulted, native_errors, writes_after_fault = [], [], []
    canonical_reads = 0
    actual_canonical = builder.load_canonical_soul_resource

    async def canonical():
        nonlocal canonical_reads, selected
        canonical_reads += 1
        if route == "refresh" and canonical_reads == 2:
            selected = True
        return await actual_canonical()

    monkeypatch.setattr(builder, "load_canonical_soul_resource", canonical)

    async def lose_actual_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if selected and not faulted and typ is None:
            faulted.append(checkout)
            raise original
        return result

    def forward(name, actual):
        async def operation(*args, **kwargs):
            nonlocal selected
            if name == "execute" and args:
                query = args[0].upper()
                if faulted:
                    writes_after_fault.append(query)
                if route == "promote" and "INTO AGENT_IDENTITY_RESOURCES" in query:
                    selected = True
                if route == "graph-link" and "INTO GRAPH_EDGES" in query:
                    selected = True
            try:
                return await actual(*args, **kwargs)
            except BaseException as error:
                native_errors.append(error)
                raise

        return operation

    for name in ("execute", "fetch_one", "fetch_all", "fetch_val"):
        monkeypatch.setattr(backend, name, forward(name, getattr(backend, name)))
    monkeypatch.setattr(storage, "add_edge", forward("add_edge", storage.add_edge))
    backend._execution_custody = ExecutionCustody(GenerationFence())
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_actual_ack)
    failure, result = None, None
    try:
        try:
            if route == "canonical":
                result = await builder.load_canonical_soul_resource()
            elif route == "bootstrap":
                result = await builder.load_bootstrap_db_config()
            else:
                result = await KestrelAgent._load_or_promote_soul_resource(
                    runtime, str(tmp_path)
                )
        except BaseException as error:
            failure = error
        assert faulted and native_errors, "fixture did not reach actual native storage"
        assert failure is not None and original in tuple(
            _explicit_control_chain(failure)
        ), f"original native SOUL/bootstrap control became fallback: {result!r}"
        assert is_execution_control_error(failure)
        assert execution_commit_outcome(failure) == "committed"
        assert not writes_after_fault, "fallback performed later durable seed work"
    finally:
        backend._execution_custody = None
        await storage.close()
