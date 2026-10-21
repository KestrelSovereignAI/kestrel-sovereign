"""Actual job publication and cancellation before its original body starts."""

import asyncio
import inspect
import json
from contextlib import asynccontextmanager
from types import MethodType

import pytest
from fastapi import Request

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.endpoints import models
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_embedding_job_original_control import (
    _job_agent,
    _seed_job_corpus,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("first_reader", ["public", "prune"])
async def test_actual_shutdown_settles_unstarted_reindex_job(
    native_memory_owners, monkeypatch, first_reader
):
    n = native_memory_owners
    monkeypatch.setenv("KESTREL_EMBEDDING_DIM", "3")
    monkeypatch.setattr(models, "_REINDEX_JOBS", {})
    monkeypatch.setattr(models, "_REINDEX_JOBS_MAX", 1)
    pool = n.backend._pool
    owners, first_observation = [], []
    actual_create = asyncio.create_task

    @asynccontextmanager
    async def checkout(native):
        async with native as connection:
            await connection.execute(f'SET search_path TO "{n.schema}", public')
            yield connection

    n.backend._pool = PostgresPoolAdapter(pool, checkout)
    try:
        await _seed_job_corpus(n)
        agent = _job_agent(n)
        agent._raise_shutdown_control = MethodType(
            KestrelAgent._raise_shutdown_control, agent
        )
        monkeypatch.setattr(models, "get_agent", lambda request: agent)

        def observe_first(done):
            # This real Task-factory observer runs before the publisher's
            # callbacks. No completion, status or native ACK is fabricated.
            if first_reader == "public":
                first_observation.append(models._reindex_job_public(job)["status"])
            else:
                models._prune_reindex_jobs(reserve=1)
                first_observation.append(job["job_id"] not in models._REINDEX_JOBS)

        def publish(operation, *args, **kwargs):
            task = actual_create(operation, *args, **kwargs)
            if (kwargs.get("name") or "").startswith("embedding-reindex-"):
                owners.append(task)
                task.add_done_callback(observe_first)
            return task

        monkeypatch.setattr(asyncio, "create_task", publish)
        body = json.dumps({"dry_run": False, "tables": ["saved_items"]}).encode()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(
            {"type": "http", "method": "POST", "path": "/component"}, receive
        )
        started = await models.reindex_embeddings(request)
        job = models._REINDEX_JOBS[started["job_id"]]
        owner = job["_owner_task"]
        assert owners == [owner]
        assert inspect.getcoroutinestate(owner.get_coro()) == inspect.CORO_CREATED
        # The actual Agent helper cancels and joins its actual background set.
        # It is not a fabricated Task outcome or an Agent READY assertion.
        await KestrelAgent._shutdown_background_tasks(agent)
        assert isinstance((await await_owned_task(owner)).error, asyncio.CancelledError)
        assert first_observation == (
            ["error"] if first_reader == "public" else [True]
        ), (
            f"first completed-owner observer still sees an unstarted running job: "
            f"{first_reader} {first_observation!r} {job!r}"
        )
        assert job["status"] == "error" and job["error"] == "Reindex interrupted"
        updated = job["updated_at"]
        assert models._reindex_job_public(job)["updated_at"] == updated
        assert models._reindex_job_public(job)["updated_at"] == updated
        models._prune_reindex_jobs(reserve=1)
        assert not models._REINDEX_JOBS and not agent._background_tasks
        assert (
            await n.observer.fetchval(
                f'SELECT count(*) FROM "{n.schema}".saved_items WHERE agent_id=$1 AND embedding_profile_id=$2',
                n.did,
                "old-profile",
            )
            == 2
        )
        assert not n.requests, "unstarted work cannot issue an embedding request"
    finally:
        for owner in owners:
            if not owner.done():
                owner.cancel()
            await await_owned_task(owner)
        n.backend._pool = pool
