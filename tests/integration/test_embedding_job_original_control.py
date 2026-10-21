"""Actual background reindex work, native receipts and lifecycle ownership.

Agent/request fields are explicit component inputs, not READY or auth proof.
Only the local fixture SDK and the fixture-owned PostgreSQL schema are used.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import MethodType, SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.endpoints import models
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.utils.postgres_schema import pgvector_schema, quoted_search_path


async def _seed_job_corpus(n):
    db = n.pg_storage.db
    vector_namespace = quoted_search_path(await pgvector_schema(db))
    for table in ("saved_items", "conversation_history", "document_chunks"):
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT"
        )
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_vec {vector_namespace}.vector(3)"
        )
    await db.execute(
        "CREATE TABLE IF NOT EXISTS embedding_profiles (id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER NOT NULL, space_id TEXT NOT NULL, normalized BOOLEAN NOT NULL)"
    )
    for index in range(2):
        await db.execute(
            "INSERT INTO saved_items (id,agent_id,item_type,name,content,content_hash,embedding_profile_id) VALUES (?,?,?,?,?,?,?)",
            (
                f"job-{index}",
                n.did,
                "excerpt",
                "native",
                f"Native local source {index}.",
                f"job-{index}",
                "old-profile",
            ),
        )


def _job_agent(n):
    agent = SimpleNamespace(
        agent_id=n.did,
        storage=n.pg_storage,
        llm_service=n.service,
        _background_tasks=set(),
    )
    agent._remember_shutdown_control = MethodType(
        KestrelAgent._remember_shutdown_control, agent
    )
    agent._track_background_task = MethodType(
        KestrelAgent._track_background_task, agent
    )
    agent._finish_background_task = MethodType(
        KestrelAgent._finish_background_task, agent
    )
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["native-control", "native-cancel", "ordinary", "healthy"]
)
async def test_actual_reindex_job_keeps_original_control_and_owner(
    native_memory_owners, monkeypatch, kind
):
    n = native_memory_owners
    monkeypatch.setenv("KESTREL_EMBEDDING_DIM", "3")
    monkeypatch.setattr(models, "_REINDEX_JOBS", {})
    db, pool = n.pg_storage.db, n.backend._pool
    armed, reached, fired, sources, launched = False, [], [], [], []
    search_path = quoted_search_path(n.schema, await pgvector_schema(db))
    original = OSError("actual background reindex release acknowledgement lost")

    @asynccontextmanager
    async def vector_checkout(native):
        async with native as connection:
            # Real per-connection setting, reset by the actual pool on release.
            await connection.execute(f'SET search_path TO {search_path}')
            yield connection
        if armed and not fired and kind.startswith("native"):
            fired.append(True)
            if kind == "native-cancel":
                raise asyncio.CancelledError(
                    "native release observer withdrawn"
                ) from original
            raise original

    n.backend._pool = PostgresPoolAdapter(pool, vector_checkout)
    actual_commit = db.execute_commit
    try:
        await _seed_job_corpus(n)

        async def commit(sql, *args, **kwargs):
            nonlocal armed
            selected = "UPDATE saved_items SET embedding_vec" in sql and not reached
            if selected:
                reached.append(sql)
                armed = kind.startswith("native")
            try:
                result = await actual_commit(sql, *args, **kwargs)
            except BaseException as error:
                if selected:
                    sources.append(error)
                raise
            if selected and kind == "ordinary":
                raise OSError("ordinary callback unavailable after native ACK")
            return result

        monkeypatch.setattr(db, "execute_commit", commit)
        n.backend._execution_custody = ExecutionCustody(GenerationFence())
        agent = _job_agent(n)
        monkeypatch.setattr(models, "get_agent", lambda request: agent)
        actual_create = asyncio.create_task

        def capture_create(operation, *args, **kwargs):
            task = actual_create(operation, *args, **kwargs)
            name = kwargs.get("name") or ""
            code = getattr(operation, "cr_code", None)
            if getattr(code, "co_name", None) == "_run_reindex_job" or name.startswith(
                "embedding-reindex-"
            ):
                launched.append(task)
            return task

        monkeypatch.setattr(asyncio, "create_task", capture_create)
        body = json.dumps({"dry_run": False, "tables": ["saved_items"]}).encode()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(
            {"type": "http", "method": "POST", "path": "/component"}, receive
        )
        started = await models.reindex_embeddings(request)
        job = models._REINDEX_JOBS[started["job_id"]]
        assert len(launched) == 1, "actual reindex launcher was not observed"
        owner = launched[0]
        outcome = await await_owned_task(owner)
        await asyncio.sleep(0)  # native done callbacks, not CPU-load generation
        assert reached, f"actual reindex write not reached: {outcome.error!r} {job!r}"
        target = n.service.get_embedding_service().current_profile_id()
        committed = await n.observer.fetchval(
            f'SELECT count(*) FROM "{n.schema}".saved_items WHERE agent_id=$1 AND embedding_profile_id=$2',
            n.did,
            target,
        )
        if kind.startswith("native"):
            assert (
                fired
                and sources
                and execution_commit_outcome(sources[0]) == "committed"
            )
            assert original in tuple(_explicit_control_chain(sources[0]))
            assert outcome.error is not None and is_execution_control_error(
                outcome.error
            ), (
                f"actual reindex owner reported ordinary completion: {kind} {outcome.error!r} {job!r}"
            )
            assert sources[0] in tuple(_explicit_control_chain(outcome.error))
            assert committed == 1, (
                "first committed write cannot continue to the next row"
            )
            control = job.get("_control")
            assert control is not None and sources[0] in tuple(
                _explicit_control_chain(control)
            )
            failure = None
            try:
                await models.get_reindex_job(job["job_id"], request)
            except BaseException as error:
                failure = error
            assert failure is not None and is_execution_control_error(failure)
            assert sources[0] in tuple(_explicit_control_chain(failure))
            assert job.get("_owner_task") is owner and owner.done()
            assert original in tuple(
                _explicit_control_chain(vars(agent).get("_shutdown_control"))
            )
            monkeypatch.setattr(models, "_REINDEX_JOBS_MAX", 1)
            models._prune_reindex_jobs(reserve=1)
            assert models._REINDEX_JOBS.get(job["job_id"]) is job, (
                "a completed native Task does not settle its retained control"
            )
        else:
            assert outcome.error is None and committed == 2
            status = await models.get_reindex_job(job["job_id"], request)
            assert status["status"] == ("partial" if kind == "ordinary" else "done")
            assert status["total_failed"] == (1 if kind == "ordinary" else 0)
            assert "_control" not in status and "_owner_task" not in status
        assert not agent._background_tasks
        assert len(n.requests) == 1 and len(n.requests[0]["input"]) == 2, (
            "both source rows use one actual batched local SDK request"
        )
    finally:
        for task in launched:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        n.backend._pool = pool
        n.backend._execution_custody = None


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["missing-owner", "native-admission"])
async def test_actual_reindex_refusal_cannot_publish_an_unowned_job(
    native_memory_owners, monkeypatch, refusal
):
    import inspect

    n = native_memory_owners
    monkeypatch.setenv("KESTREL_EMBEDDING_DIM", "3")
    monkeypatch.setattr(models, "_REINDEX_JOBS", {})
    db, pool = n.pg_storage.db, n.backend._pool
    submitted, sources = [], []
    search_path = quoted_search_path(n.schema, await pgvector_schema(db))

    @asynccontextmanager
    async def checkout(native):
        async with native as connection:
            await connection.execute(f'SET search_path TO {search_path}')
            yield connection

    n.backend._pool = PostgresPoolAdapter(pool, checkout)
    try:
        await _seed_job_corpus(n)
        n.backend._execution_custody = ExecutionCustody(GenerationFence())
        agent = _job_agent(n)
        monkeypatch.setattr(models, "get_agent", lambda request: agent)
        if refusal == "missing-owner":
            del agent._track_background_task
        else:
            actual_publish = agent._track_background_task

            def publish(operation, *, name):
                submitted.append(operation)
                return actual_publish(operation, name=name)

            agent._track_background_task = publish
            actual_create = asyncio.create_task

            def refuse(operation, *args, **kwargs):
                if (kwargs.get("name") or "").startswith("embedding-reindex-"):
                    try:
                        n.lease.require_live()
                    except BaseException as error:
                        sources.append(error)
                        raise
                return actual_create(operation, *args, **kwargs)

            monkeypatch.setattr(asyncio, "create_task", refuse)
        body = json.dumps({"dry_run": False, "tables": ["saved_items"]}).encode()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(
            {"type": "http", "method": "POST", "path": "/component"}, receive
        )
        failure = None
        try:
            await models.reindex_embeddings(request)
        except BaseException as error:
            failure = error
        if refusal == "missing-owner":
            assert isinstance(failure, HTTPException) and failure.status_code == 503
        else:
            assert (
                sources
                and failure is sources[0]
                and is_execution_control_error(failure)
            ), f"native job admission control was replaced: {failure!r}"
            assert submitted and all(
                inspect.getcoroutinestate(op) == inspect.CORO_CLOSED for op in submitted
            )
        assert not models._REINDEX_JOBS and not agent._background_tasks
        assert (
            await n.observer.fetchval(
                f'SELECT count(*) FROM "{n.schema}".saved_items WHERE agent_id=$1 AND embedding_profile_id=$2',
                n.did,
                "old-profile",
            )
            == 2
        )
        assert not n.requests, "refusal cannot start the actual SDK reindex work"
    finally:
        for operation in submitted:
            operation.close()
        n.backend._pool = pool
        n.backend._execution_custody = None


@pytest.mark.asyncio
async def test_actual_background_owner_keeps_cancelled_native_lease_receipt(
    native_memory_owners,
    monkeypatch,
):
    n = native_memory_owners
    agent = _job_agent(n)
    originals = []
    observed = []
    actual_create = asyncio.create_task

    def observe_terminal(done):
        try:
            done.exception()
        except asyncio.CancelledError as error:
            observed.append(error)

    def publish(operation, *args, **kwargs):
        task = actual_create(operation, *args, **kwargs)
        if kwargs.get("name") == "native-lease-receipt":
            # A real Task-factory observer can inspect cancellation before
            # the agent's done callback. It grants no authority and forwards
            # the exact native Task; no result or completion is fabricated.
            task.add_done_callback(observe_terminal)
        return task

    monkeypatch.setattr(asyncio, "create_task", publish)

    async def operation():
        await n.pg_storage.db.execute_commit(
            "INSERT INTO agent_metadata (agent_id,key,value) VALUES (?,?,?)",
            (n.did, "background-native-receipt", "committed"),
        )
        try:
            n.lease.require_live()
        except BaseException as original:
            originals.append(original)
            # An explicit observer carrier around the actual lease denial,
            # not a fabricated commit outcome or successful native Task.
            raise asyncio.CancelledError(
                "native lease observer withdrawn"
            ) from original

    owner = agent._track_background_task(operation(), name="native-lease-receipt")
    try:
        outcome = await await_owned_task(owner)
        assert (
            originals
            and observed
            and originals[0] in tuple(_explicit_control_chain(observed[0]))
        ), "first native Task observer did not receive the original lease denial"
        assert (
            originals
            and outcome.error is not None
            and originals[0] in tuple(_explicit_control_chain(outcome.error))
        ), "background Task cancellation erased the original native lease denial"
        assert originals[0] in tuple(
            _explicit_control_chain(vars(agent).get("_shutdown_control"))
        ), "agent lifecycle lost the completed background native receipt"
        assert (
            await n.observer.fetchval(
                f'SELECT value FROM "{n.schema}".agent_metadata WHERE agent_id=$1 AND key=$2',
                n.did,
                "background-native-receipt",
            )
            == "committed"
        )
        assert owner.done() and not agent._background_tasks
    finally:
        await await_owned_task(owner)


@pytest.mark.asyncio
async def test_actual_background_owner_refuses_borrowed_native_task(
    native_memory_owners,
):
    n = native_memory_owners
    agent = _job_agent(n)
    borrowed = asyncio.create_task(n.pg_storage.db.fetchval("SELECT 42"))
    unexpected_owner = None
    failure = None
    try:
        try:
            unexpected_owner = agent._track_background_task(
                borrowed, name="borrowed-native-background"
            )
        except TypeError as error:
            failure = error
        assert failure is not None, (
            "background publisher adopted a borrowed native Task"
        )
        assert not borrowed.cancelled() and borrowed not in agent._background_tasks
        assert (await await_owned_task(borrowed)).result == 42
    finally:
        await await_owned_task(borrowed)
        if unexpected_owner is not None:
            await await_owned_task(unexpected_owner)


@pytest.mark.asyncio
async def test_actual_background_refusal_closes_unstarted_native_coroutine(
    native_memory_owners, monkeypatch
):
    import inspect

    n = native_memory_owners
    agent = _job_agent(n)
    sources = []
    actual_create = asyncio.create_task

    def refuse(operation, *args, **kwargs):
        if kwargs.get("name") == "refused-native-background":
            try:
                n.lease.require_live()
            except BaseException as error:
                sources.append(error)
                raise
        return actual_create(operation, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", refuse)
    operation = n.pg_storage.db.execute(
        "INSERT INTO agent_metadata (agent_id,key,value) VALUES (?,?,?)",
        (n.did, "refused-background", "must-not-write"),
    )
    failure = None
    try:
        try:
            agent._track_background_task(operation, name="refused-native-background")
        except BaseException as error:
            failure = error
        assert sources and failure is sources[0] and is_execution_control_error(failure)
        assert inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED, (
            "refused background publication left its native coroutine unowned"
        )
        assert not agent._background_tasks
        assert (
            await n.observer.fetchval(
                f'SELECT count(*) FROM "{n.schema}".agent_metadata WHERE agent_id=$1 AND key=$2',
                n.did,
                "refused-background",
            )
            == 0
        )
    finally:
        operation.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["finish", "withdraw"])
async def test_actual_reindex_capacity_refuses_before_second_owner(
    native_memory_owners, monkeypatch, completion
):
    n = native_memory_owners
    monkeypatch.setenv("KESTREL_EMBEDDING_DIM", "3")
    monkeypatch.setattr(models, "_REINDEX_JOBS", {})
    monkeypatch.setattr(models, "_REINDEX_JOBS_MAX", 1)
    db, pool = n.pg_storage.db, n.backend._pool
    committed, release = asyncio.Event(), asyncio.Event()
    agent = None
    search_path = quoted_search_path(n.schema, await pgvector_schema(db))

    @asynccontextmanager
    async def checkout(native):
        async with native as connection:
            await connection.execute(f'SET search_path TO {search_path}')
            yield connection

    n.backend._pool = PostgresPoolAdapter(pool, checkout)
    actual_commit = db.execute_commit
    try:
        await _seed_job_corpus(n)

        async def gate_after_ack(sql, *args, **kwargs):
            result = await actual_commit(sql, *args, **kwargs)
            if "UPDATE saved_items SET embedding_vec" in sql and not committed.is_set():
                committed.set()
                await release.wait()
            return result

        monkeypatch.setattr(db, "execute_commit", gate_after_ack)
        n.backend._execution_custody = ExecutionCustody(GenerationFence())
        agent = _job_agent(n)
        monkeypatch.setattr(models, "get_agent", lambda request: agent)
        body = json.dumps({"dry_run": False, "tables": ["saved_items"]}).encode()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(
            {"type": "http", "method": "POST", "path": "/component"}, receive
        )
        first = await models.reindex_embeddings(request)
        await asyncio.wait_for(committed.wait(), timeout=10)
        owner = models._REINDEX_JOBS[first["job_id"]]["_owner_task"]
        assert agent._background_tasks == {owner} and not owner.done()
        # The mutable business projection is deliberately adversarial input,
        # not a fabricated native completion. The ACTUAL owner is unfinished.
        live_job = models._REINDEX_JOBS[first["job_id"]]
        live_job["status"] = "done"
        models._prune_reindex_jobs(reserve=1)
        assert models._REINDEX_JOBS.get(first["job_id"]) is live_job, (
            "a business status cannot retire the original pending native Task"
        )
        live_job["status"] = "running"
        assert (
            await n.observer.fetchval(
                f'SELECT count(*) FROM "{n.schema}".saved_items WHERE agent_id=$1 AND embedding_profile_id=$2',
                n.did,
                n.service.get_embedding_service().current_profile_id(),
            )
            == 1
        )
        failure = None
        try:
            await models.reindex_embeddings(request)
        except HTTPException as error:
            failure = error
        assert failure is not None and failure.status_code == 503, (
            f"reindex capacity admitted a second native owner: {failure!r} owners={len(agent._background_tasks)}"
        )
        assert agent._background_tasks == {owner} and len(models._REINDEX_JOBS) == 1
        if completion == "withdraw":
            owner.cancel()
        release.set()
        outcome = await await_owned_task(owner)
        status = await models.get_reindex_job(first["job_id"], request)
        if completion == "finish":
            assert outcome.error is None
            assert status["status"] == "done" and status["total_reembedded"] == 2
        else:
            assert isinstance(outcome.error, asyncio.CancelledError)
            assert not is_execution_control_error(outcome.error)
            assert (
                status["status"] == "error" and status["error"] == "Reindex interrupted"
            )
        models._prune_reindex_jobs(reserve=1)
        assert not models._REINDEX_JOBS
    finally:
        release.set()
        owners = set(agent._background_tasks) if agent is not None else set()
        owners.update(job["_owner_task"] for job in models._REINDEX_JOBS.values())
        for task in owners:
            await await_owned_task(task)
        n.backend._pool = pool
        n.backend._execution_custody = None
