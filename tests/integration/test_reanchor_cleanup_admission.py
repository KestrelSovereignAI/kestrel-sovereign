"""Reanchor's original embedding span keeps committed work and its raw owner.

This drives the real embedding context and PostgreSQL COMMIT, not the separate
signed-amendment admission API. Resolution is local and makes no provider call.
"""

import asyncio
import inspect

import httpx
import pytest

from kestrel_sovereign import cli_embeddings
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_cleanup_debts,
    execution_commit_outcome,
)
from kestrel_sovereign.llm import service as service_module
from kestrel_sovereign.llm.lifecycle import UnpublishedLLMOwner
from kestrel_sovereign.setup.constitution_reanchor import _agent_embedding
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_execution_custody_review43_native import carries


@pytest.mark.asyncio
@pytest.mark.parametrize("body_outcome", ["committed", "unknown"])
async def test_reanchor_closer_admission_refusal_keeps_actual_commit_and_service(
    native_pg, tmp_path, monkeypatch, body_outcome
):
    backend, observer, schema = native_pg
    monkeypatch.setattr(service_module, "load_section", lambda _: {})
    service = service_module.LLMService(
        usage_db=AsyncDatabase(backend), agent_data_dir=tmp_path
    )
    client = httpx.AsyncClient()
    service.providers = [{"adapter": client}]
    resolution = cli_embeddings.AgentEmbeddingResolution(
        service, None, None, _private_owner=UnpublishedLLMOwner(service, ()),
    )

    async def resolve(_db, _did):
        return resolution

    monkeypatch.setattr(cli_embeddings, "resolve_agent_embedding", resolve)
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    refused, primary = [], None
    backend._execution_custody = ExecutionCustody(GenerationFence())

    def factory(loop, coroutine, **kwargs):
        frame = getattr(coroutine, "cr_frame", None)
        inner = frame.f_locals.get("operation") if frame is not None else None
        candidate = inner if inspect.iscoroutine(inner) else coroutine
        code = getattr(candidate, "cr_code", None)
        if code is not None and code.co_qualname == "UnpublishedLLMOwner._close":
            refused.append((coroutine, candidate))
            raise RuntimeError("reanchor physical closer Task allocation refused")
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        loop.set_task_factory(factory)
        with pytest.raises(BaseException) as propagated:
            async with _agent_embedding(
                AsyncDatabase(backend), "did:test:original-reanchor", needed=True
            ):
                if body_outcome == "unknown":
                    lose_native_ack(
                        monkeypatch,
                        boundary="commit",
                        original=OSError("actual repair COMMIT ACK lost"),
                    )
                try:
                    await backend.execute(
                        "INSERT INTO effects VALUES (109, 'actual repair span')"
                    )
                except BaseException as error:
                    primary = error
                    raise
        assert len(refused) == 1 and not client.is_closed
        assert execution_commit_outcome(propagated.value) == body_outcome, (
            "closer allocation erased the original repair outcome"
        )
        assert (service, None) in execution_cleanup_debts(propagated.value)
        if primary is not None:
            assert carries(propagated.value, primary)
        if body_outcome == "unknown":
            assert (
                primary is not None and execution_commit_outcome(primary) == "unknown"
            )
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=109'
            )
            == 1
        )
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        )
    finally:
        backend._execution_custody = None
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await service.close()
        await service.wait_for_close_completion()
        await client.aclose()
