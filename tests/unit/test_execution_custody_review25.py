"""Native host COMMIT acknowledgment and local provisioning retry boundaries."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, ProcessRuntimeExecutionFence,
    bind_execution_runtime, execution_commit_outcome, is_execution_control_error,
)
from kestrel_sovereign.features.llm_keys import openrouter_provisioning as provisioning
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_execution_custody_review24 import host_accessor


async def host_write(db, operation):
    sql = "INSERT INTO service_providers (id,name) VALUES ('receipt','receipt')"
    if operation == "execute":
        await db.execute(sql)
    elif operation == "batch":
        await db.execute_many(sql, [()])
    elif operation == "script":
        await db.execute_script(sql)
    else:
        async with db.transaction():
            await db.execute(sql)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["execute", "batch", "script", "transaction"])
@pytest.mark.parametrize("native_committed", [True, False])
async def test_host_interrupted_commit_ack_retains_uncertainty(tmp_path, monkeypatch, operation, native_committed):
    owner, db, path = await host_accessor(tmp_path)
    entered = asyncio.Event()
    original_commit = db.backend._connection.commit

    async def interrupted_ack():
        if native_committed:
            await original_commit()
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(db.backend._connection, "commit", interrupted_ack)
    task = asyncio.create_task(host_write(db, operation))  # No ambient opener: retained custody must suffice.
    try:
        await entered.wait()
        task.cancel()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        assert isinstance(result, BaseException)
        assert execution_commit_outcome(result) == "unknown", "COMMIT handoff was mistaken for rollback"
        with pytest.raises(ExecutionAuthorityError) as retained:
            owner._execution_custody.require_work()
        assert execution_commit_outcome(retained.value) == "unknown"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await db.close()
    verifier = await AsyncDatabase.sqlite(str(path))
    try:
        assert await verifier.fetchone("SELECT COUNT(*) FROM service_providers WHERE id='receipt'") == (int(native_committed),)
    finally:
        await verifier.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["execute", "batch", "script", "transaction"])
async def test_host_known_commit_retirement_keeps_receipt_for_all_original_scopes(tmp_path, monkeypatch, operation):
    owner, db, _ = await host_accessor(tmp_path)
    occurrence = SimpleNamespace(_execution_custody=ExecutionCustody(ProcessRuntimeExecutionFence("original-occurrence")))
    original_commit = db.backend._connection.commit

    async def committed_then_retired():
        await original_commit()
        occurrence._execution_custody.fence.retire()

    monkeypatch.setattr(db.backend._connection, "commit", committed_then_retired)
    try:
        with bind_execution_runtime(occurrence):
            with pytest.raises(BaseException) as result:
                await host_write(db, operation)
            assert execution_commit_outcome(result.value) == "committed"
        for scope in (owner._execution_custody, occurrence._execution_custody):
            with pytest.raises(ExecutionAuthorityError) as retained:
                scope.require_work()
            assert execution_commit_outcome(retained.value) == "committed"
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retire_during", ["backoff", "client", "live"])
async def test_real_provisioning_post_attempts_revalidate_original_custody(monkeypatch, retire_during):
    owner = SimpleNamespace(_execution_custody=ExecutionCustody(ProcessRuntimeExecutionFence("provisioning")))
    entered, release = asyncio.Event(), asyncio.Event()
    posts = []

    async def transport(request):
        posts.append(request.url.path)
        if len(posts) == 1:
            return httpx.Response(429, json={"error": "local rate limit"})
        return httpx.Response(200, json={"key": "synthetic-not-a-provider-key", "data": {"hash": "local-only"}})

    service = provisioning.OpenRouterProvisioningService(management_key="synthetic-local-test-only")
    service._client = httpx.AsyncClient(base_url="https://local.invalid", transport=httpx.MockTransport(transport))
    original_get_client = service._get_client

    async def client_boundary():
        if retire_during == "client":
            entered.set()
            await release.wait()
        return await original_get_client()

    async def backoff(_delay):
        if retire_during != "client":
            entered.set()
            await release.wait()

    monkeypatch.setattr(service, "_get_client", client_boundary)
    monkeypatch.setattr(provisioning, "asyncio", SimpleNamespace(sleep=backoff))
    task = None
    try:
        with bind_execution_runtime(owner):
            task = asyncio.create_task(service.create_agent_key("fixture"))
        await entered.wait()
        if retire_during != "live":
            owner._execution_custody.fence.retire()
        release.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        if retire_during == "live":
            assert result.key_hash == "local-only" and len(posts) == 2
        else:
            assert isinstance(result, BaseException) and is_execution_control_error(result)
            assert len(posts) == (0 if retire_during == "client" else 1)
    finally:
        release.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["execute", "batch", "script", "transaction"])
@pytest.mark.parametrize("cleanup_error", ["cancel", "error"])
async def test_host_commit_receipt_survives_secondary_rollback_failure(tmp_path, monkeypatch, operation, cleanup_error):
    owner, db, _ = await host_accessor(tmp_path)
    connection = db.backend._connection
    original_commit, original_rollback = connection.commit, connection.rollback
    committing, rolling_back = asyncio.Event(), asyncio.Event()
    release_cleanup = asyncio.Event()

    async def commit_ack():
        await original_commit()
        committing.set()
        await asyncio.Event().wait()

    async def rollback_ack():
        rolling_back.set()
        await release_cleanup.wait()
        if cleanup_error == "error":
            raise ValueError("local rollback failure")
        await original_rollback()

    monkeypatch.setattr(connection, "commit", commit_ack)
    monkeypatch.setattr(connection, "rollback", rollback_ack)
    task = asyncio.create_task(host_write(db, operation))
    try:
        await committing.wait()
        task.cancel()
        await rolling_back.wait()
        if cleanup_error == "cancel":
            task.cancel()
        release_cleanup.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        assert execution_commit_outcome(result) == "unknown", "secondary cleanup discarded irreversible receipt"
        with pytest.raises(ExecutionAuthorityError) as retained:
            owner._execution_custody.require_work()
        assert execution_commit_outcome(retained.value) == "unknown"
    finally:
        release_cleanup.set()
        monkeypatch.setattr(connection, "rollback", original_rollback)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper", ["rate_limit", "unexpected"])
async def test_real_provisioning_normalizer_does_not_retry_cause_carried_control(wrapper):
    calls = []
    control = ExecutionAuthorityError("original custody lost")

    async def transport(request):
        calls.append(request)
        try:
            raise control
        except ExecutionAuthorityError as cause:
            if wrapper == "rate_limit":
                response = httpx.Response(429, request=request)
                raise httpx.HTTPStatusError("local wrapper", request=request, response=response) from cause
            raise ValueError("local wrapper") from cause

    service = provisioning.OpenRouterProvisioningService(management_key="synthetic-local-test-only")
    service._client = httpx.AsyncClient(base_url="https://local.invalid", transport=httpx.MockTransport(transport))
    try:
        with pytest.raises(BaseException) as result:
            await service.create_agent_key("fixture")
        assert is_execution_control_error(result.value)
        assert not isinstance(result.value, provisioning.OpenRouterProvisioningError)
        assert len(calls) == 1
    finally:
        await service.close()
