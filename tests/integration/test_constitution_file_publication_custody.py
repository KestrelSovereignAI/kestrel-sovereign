"""Signed native file publication retains physical content custody."""

import hashlib
from uuid import uuid4

import pytest

from kestrel_sovereign.constitution import anchored_bytes
from kestrel_sovereign.constitution.amendment_artifact import (
    load_verified_reanchor_artifact,
)
from kestrel_sovereign.constitution.trust_root import load_sovereign_trust_root
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_signed_file_validation_holds_actual_blob_through_publication(
    db_backend,
    tmp_path,
    monkeypatch,
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL physical blob custody")
    import asyncpg

    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:blob-custody:" + uuid4().hex
    )
    await storage.initialize()
    external = await asyncpg.connect(db_backend._dsn)
    content = ("signed native blob " + uuid4().hex).encode()
    digest = hashlib.sha256(content).hexdigest()
    artifact, root = _write_authority_files(tmp_path, content)
    verification = load_verified_reanchor_artifact(
        artifact,
        trusted_did_document=load_sovereign_trust_root(explicit_path=root, environ={}),
        expected_constitution_sha256=digest,
    )[2]
    native_retrieve = AsyncFileStore.retrieve_file
    observed = []
    try:
        await storage.store_file(content, "KESTREL_CONSTITUTION.md")
        await external.execute("SET lock_timeout = '100ms'")

        async def retrieve_and_attempt_corruption(files, requested):
            result = await native_retrieve(files, requested)
            if files.agent_id == "" and requested == digest:
                with pytest.raises(asyncpg.LockNotAvailableError):
                    await external.execute(
                        "UPDATE files SET content=$1 WHERE content_hash=$2",
                        b"corrupt",
                        digest,
                    )
                observed.append(True)
            return result

        monkeypatch.setattr(
            AsyncFileStore, "retrieve_file", retrieve_and_attempt_corruption
        )
        async with storage.transaction():
            await storage.lock_nodes_for_update([storage.agent_id, digest])
            assert (
                await anchored_bytes.store_verified_governing_file(
                    storage, content, verification=verification
                )
                == digest
            )
        assert observed == [True]
        assert await storage.retrieve_file(digest) == content
        assert (
            await external.execute(
                "UPDATE files SET content=content WHERE content_hash=$1", digest
            )
            == "UPDATE 1"
        )
    finally:
        await external.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("winner", ["correct", "corrupt"])
async def test_signed_file_creation_validates_actual_concurrent_winner(
    db_backend,
    tmp_path,
    monkeypatch,
    winner,
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL absent-file concurrent creation")
    import asyncpg

    identity = "did:test:absent-signed-file:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    external = await asyncpg.connect(db_backend._dsn)
    content = ("signed race blob " + uuid4().hex).encode()
    digest = hashlib.sha256(content).hexdigest()
    artifact, root = _write_authority_files(tmp_path, content)
    verification = load_verified_reanchor_artifact(
        artifact,
        trusted_did_document=load_sovereign_trust_root(explicit_path=root, environ={}),
        expected_constitution_sha256=digest,
    )[2]
    native_store = AsyncFileStore.store_file
    reached = []
    try:

        async def concurrently_created(files, data, name, *args, **kwargs):
            if data == content:
                await external.execute(
                    "INSERT INTO files (content_hash,original_name,content,metadata) VALUES ($1,'winner',$2,NULL)",
                    digest,
                    content if winner == "correct" else b"corrupt",
                )
                await external.execute(
                    "INSERT INTO file_owners (content_hash,agent_id,original_name,metadata) VALUES ($1,'did:test:other-owner','winner',NULL)",
                    digest,
                )
                reached.append(True)
            return await native_store(files, data, name, *args, **kwargs)

        monkeypatch.setattr(AsyncFileStore, "store_file", concurrently_created)

        async def publish():
            async with storage.transaction():
                await storage.lock_nodes_for_update([identity, digest])
                return await anchored_bytes.store_verified_governing_file(
                    storage, content, verification=verification
                )

        if winner == "corrupt":
            with pytest.raises(
                TransactionError, match="differs from exact signed content"
            ) as refused:
                await publish()
            assert isinstance(refused.value.__cause__, RuntimeError)
            assert await storage.retrieve_file(digest) is None
        else:
            assert await publish() == digest
            assert await storage.retrieve_file(digest) == content
        assert reached == [True]
        assert (
            await external.fetchval(
                "SELECT COUNT(*) FROM file_owners WHERE content_hash=$1 AND agent_id='did:test:other-owner'",
                digest,
            )
            == 1
        )
    finally:
        await external.close()
        await storage.close()
