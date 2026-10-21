"""Native retrieval helper boundaries, not model/Agent READY acceptance."""

from dataclasses import dataclass

import asyncpg
import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_storage_backend_parity import (
    _RECALL_PROFILE,
    _incepted_assertion_identity,
)


class NoEmbeddingsInput:
    def get_embedding_service(self):
        return None


@dataclass
class NativeRetrievalBudget:
    rag: int = 512


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["rag", "semantic"])
@pytest.mark.parametrize("door", ["builder", "manager"])
async def test_native_retrieval_control_is_not_context_fallback(
    native_pg, tmp_path, monkeypatch, lane, door
):
    backend, observer, schema = native_pg
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    did, identity = await _incepted_assertion_identity(tmp_path, "native-retrieval")
    capability = _resolve_authenticated_agent_assertion_capability(did, identity)
    no_models = NoEmbeddingsInput()
    storage = AsyncStorage(
        backend=backend,
        agent_id=did,
        llm_service=no_models,
        _assertion_tenant_capability=capability,
    )
    await storage.initialize()
    builder = ContextBuilder(
        storage,
        agent_data_path=str(tmp_path),
        llm_service=no_models,
        semantic_inference_profile=_RECALL_PROFILE,
    )
    manager = ContextManager(storage, context_builder=builder)
    from kestrel_sovereign import config

    actual_config = config.load_section

    def retrieval_config(section, *args, **kwargs):
        if section == "retrieval":
            return {"semantic_recall_enabled": lane == "semantic"}
        return actual_config(section, *args, **kwargs)

    monkeypatch.setattr(config, "load_section", retrieval_config)
    original = OSError("actual native retrieval release acknowledgement lost")
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    faulted, source_errors = [], []

    async def lose_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if not faulted and typ is None:
            faulted.append(checkout)
            raise original
        return result

    def forward(actual):
        async def operation(*args, **kwargs):
            try:
                return await actual(*args, **kwargs)
            except BaseException as error:
                source_errors.append(error)
                raise

        return operation

    for name in ("fetch_one", "fetch_all", "fetch_val", "execute"):
        monkeypatch.setattr(backend, name, forward(getattr(backend, name)))
    backend._execution_custody = ExecutionCustody(GenerationFence())
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    result, failure = None, None
    try:
        try:
            query = "native local query" if lane == "rag" else ""
            if door == "builder":
                result = await builder.retrieve_context(query)
            else:
                result = await manager._produce_rag(
                    NativeRetrievalBudget(),
                    query,
                    {"rag_min_score": 0},
                    include_rag=True,
                    trivial_turn=False,
                )
        except BaseException as error:
            failure = error
        assert faulted and source_errors, (
            "prototype did not reach real native PostgreSQL"
        )
        assert failure is not None and original in tuple(
            _explicit_control_chain(failure)
        ), f"native control became context fallback {result!r}"
        assert (
            is_execution_control_error(failure)
            and execution_commit_outcome(failure) == "committed"
        )
    finally:
        backend._execution_custody = None
        await storage.close()
