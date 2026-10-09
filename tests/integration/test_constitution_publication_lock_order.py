"""Native writers reserve governance before taking file-owner row locks."""

import asyncio
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.setup import constitution_reanchor as offline
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files
from tests.integration.test_constitution_refusal_races import _agent


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("rejection", ["foreign-owner", "oversized", "unowned-blob"])
async def test_rejected_avatar_has_no_joined_transaction_side_effects(
    db_backend, monkeypatch, rejection
):
    if db_backend.backend_type != "sqlite":
        pytest.skip("SQLite joined transaction rejection boundary")
    from kestrel_sovereign.storage import async_file_store

    identity = "did:test:avatar-rejection:" + uuid4().hex
    target = (
        "did:test:avatar-foreign:" + uuid4().hex
        if rejection == "foreign-owner"
        else identity
    )
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    try:
        if rejection == "oversized":
            monkeypatch.setattr(async_file_store, "MAX_FILE_SIZE", 1)
        elif rejection == "unowned-blob":
            import hashlib

            await storage.db.execute_commit(
                "INSERT INTO files (content_hash,original_name,content) VALUES (?,?,?)",
                (hashlib.sha256(b"avatar").hexdigest(), "legacy.jpg", b"avatar"),
            )
        async with storage.transaction():
            # A caller may catch a validation error and commit other work.
            # Native SQLite joins rather than rolling back the nested scope.
            with pytest.raises(ValueError):
                await storage.files.store_avatar(b"avatar", target)
        assert (
            await storage.db.fetchone(
                "SELECT node_id FROM graph_node_owners WHERE node_id=? AND agent_id=?",
                (target, target),
            )
            is None
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["runtime", "offline", "bootstrap", "avatar"])
async def test_postgres_governance_custody_precedes_file_owner_write(
    db_backend,
    tmp_path,
    monkeypatch,
    writer,
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL cross-connection physical lock-order proof")
    import asyncpg

    identity = "did:test:graph-before-file:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    task = None
    blocker = probe = None
    transaction = None
    try:
        agent = await _agent(storage)
        content = resolve_governing_constitution_bytes(None)
        digest = await storage.store_file(content, "KESTREL_CONSTITUTION.md")
        await storage.add_node(
            GraphNode(
                node_id=identity, node_type="agent", label="lock order", properties={}
            )
        )
        artifact, root = _write_authority_files(tmp_path, content)
        agent._sovereign_trust_root_path = root
        if writer != "bootstrap":
            initial = await ConstitutionMixin.reanchor_constitution(
                agent, amendment_artifact_path=str(artifact)
            )
            assert not initial.startswith("Error:"), initial

        backend_class = type(db_backend)
        native_fetch_all = backend_class.fetch_all
        reached = asyncio.Event()
        writer_pid = None

        async def observed_fetch_all(backend, query, params=()):
            nonlocal writer_pid
            if (
                asyncio.current_task() is task
                and "SELECT node_id FROM graph_nodes" in query
                and "FOR UPDATE" in query
                and identity in params
            ):
                writer_pid = await backend.fetch_val("SELECT pg_backend_pid()")
                reached.set()
            return await native_fetch_all(backend, query, params)

        monkeypatch.setattr(backend_class, "fetch_all", observed_fetch_all)
        target = offline.ReanchorTarget(None, "postgres", identity, db_backend._dsn)

        async def exact_target(*args, **kwargs):
            return target

        @asynccontextmanager
        async def no_embedding(*args, **kwargs):
            yield None

        monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
        monkeypatch.setattr(offline, "_agent_embedding", no_embedding)

        async def repair():
            if writer == "avatar":
                return await storage.files.store_avatar(content, identity)
            if writer == "runtime":
                return await ConstitutionMixin.reanchor_constitution(
                    agent, amendment_artifact_path=str(artifact)
                )
            if writer == "bootstrap":
                return await ConstitutionMixin._get_governing_constitution(agent)
            return await offline.reanchor_constitution(
                agent_name="lock-order proof",
                agent_dir=None,
                force=True,
                sovereign_trust_root_path=root,
                amendment_artifact_path=artifact,
                runtime_backend="postgres",
                runtime_dsn=target.dsn,
                hosted_agent_did=identity,
                environ={},
            )

        blocker = await asyncpg.connect(db_backend._dsn)
        probe = await asyncpg.connect(db_backend._dsn)
        await probe.execute("SET lock_timeout = '100ms'")
        transaction = blocker.transaction()
        await transaction.start()
        await blocker.fetchrow(
            "SELECT node_id FROM graph_nodes WHERE node_id=$1 FOR UPDATE", identity
        )
        task = asyncio.create_task(repair())
        await asyncio.wait_for(reached.wait(), 5)
        # Confirm that native SQL really is waiting on governance, not merely
        # paused before a helper. While it waits it must own no file-row lock.
        async with asyncio.timeout(5):
            while (
                await probe.fetchval(
                    "SELECT wait_event_type FROM pg_stat_activity WHERE pid=$1",
                    writer_pid,
                )
                != "Lock"
            ):
                await asyncio.sleep(0.02)
        assert (
            await probe.execute(
                "UPDATE file_owners SET agent_id=agent_id WHERE content_hash=$1 AND agent_id=$2",
                digest,
                identity,
            )
        ) == "UPDATE 1"
        await transaction.rollback()
        transaction = None
        result = await asyncio.wait_for(task, 10)
        if writer == "offline":
            assert result.reanchored and result.error is None, result.error
        else:
            assert not result.startswith("Error:"), result
        assert await storage.retrieve_file(digest) == content
        assert (await storage.get_node(identity)).properties[
            "constitution_hash"
        ] == digest
    finally:
        if transaction is not None:
            await transaction.rollback()
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task
        if probe is not None:
            await probe.close()
        if blocker is not None:
            await blocker.close()
        await storage.close()
