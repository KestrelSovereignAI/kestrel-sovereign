"""Installed SDK transaction contracts retain real native rollback isolation."""
from contextlib import asynccontextmanager
import hashlib

import pytest
from kestrel_sdk.storage.database.interface import DatabaseBackend
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore


class SDKAdapter(DatabaseBackend):
    """Conforming SDK-only signature, delegating SQL to real SQLite."""
    transaction_options = ()
    nested_transaction_strategy = "joined"

    def __init__(self, native):
        self.native = native

    @property
    def backend_type(self):
        return self.native.backend_type

    @property
    def is_connected(self):
        return self.native.is_connected

    @property
    def owns_open_transaction(self):
        return self.native.owns_open_transaction

    async def connect(self):
        await self.native.connect()

    async def close(self):
        await self.native.close()

    async def execute(self, query, params=()):
        return await self.native.execute(query, params)

    async def execute_many(self, query, params_list):
        return await self.native.execute_many(query, params_list)

    async def execute_script(self, script):
        return await self.native.execute_script(script)

    async def fetch_one(self, query, params=()):
        return await self.native.fetch_one(query, params)

    async def fetch_all(self, query, params=()):
        return await self.native.fetch_all(query, params)

    async def fetch_val(self, query, params=()):
        return await self.native.fetch_val(query, params)

    @asynccontextmanager
    async def transaction(self):
        async with self.native.transaction():
            yield


class ImmediateOnlyAdapter(SDKAdapter):
    @asynccontextmanager
    async def transaction(self, *, immediate=False):
        async with self.native.transaction(immediate=immediate):
            yield


class UnknownCustodyAdapter(SDKAdapter):
    def __init__(self, native, kind, explicit_isolation):
        super().__init__(native)
        self.kind = kind
        self.transaction_options = ("savepoint",) if explicit_isolation else ()

    @property
    def backend_type(self):
        return self.kind

    @property
    def owns_open_transaction(self):
        raise AttributeError("The installed SDK does not require caller custody")

    @asynccontextmanager
    async def transaction(self, *, savepoint=False):
        async with self.native.transaction(savepoint=savepoint):
            yield


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sqlite", "sdk-joined"])
@pytest.mark.parametrize("explicit_isolation", [False, True])
async def test_unknown_sdk_custody_never_commits_partial_avatar(tmp_path, kind, explicit_isolation):
    from kestrel_sovereign.storage.async_graph_store import GraphNode

    seed = await AsyncDatabase.sqlite(str(tmp_path / "unknown-sdk.db"))

    async def already_initialized(db):
        pass

    db = await AsyncDatabase.from_connected_backend(
        UnknownCustodyAdapter(seed.backend, kind, explicit_isolation),
        schema_initializer=already_initialized,
    )
    identity = "did:test:unknown-sdk-avatar"
    files = AsyncFileStore(db, agent_id=identity)
    graph = AsyncGraphStore(db, agent_id=identity)
    content = b"native unknown-custody avatar"
    digest = hashlib.sha256(content).hexdigest()
    avatar_id = files._avatar_node_id(identity, "primary", digest)
    try:
        await graph.add_node(GraphNode(node_id=identity, node_type="agent", label="root", properties={}))
        await graph.add_node(GraphNode(node_id=avatar_id, node_type="avatar", label="old avatar", properties={"retained": True}))
        await graph.add_edge(identity, avatar_id, "has_avatar", {})
        await db.execute_commit(
            "DELETE FROM graph_edge_owners WHERE source_id=? AND target_id=? AND label='has_avatar'",
            (identity, avatar_id),
        )
        before = await graph.get_node(avatar_id)
        async with db.transaction():
            with pytest.raises(Exception) as refusal:
                await files.store_avatar(content, identity)
            if not explicit_isolation:
                assert isinstance(refusal.value, NotImplementedError)
        assert await db.fetchone("SELECT 1 FROM files WHERE content_hash=?", (digest,)) is None
        assert (await graph.get_node(avatar_id)).properties == before.properties
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [SDKAdapter, ImmediateOnlyAdapter])
async def test_sdk_avatar_top_level_commit_rollback_and_nested_refusal(tmp_path, monkeypatch, adapter):
    seed = await AsyncDatabase.sqlite(str(tmp_path / "sdk.db"))

    async def already_initialized(db):
        pass

    # This supported public constructor consumes an already initialized SDK
    # backend; Core-specific schema migration extensions are not this adapter's
    # transaction contract. All subsequent avatar SQL remains real SQLite.
    db = await AsyncDatabase.from_connected_backend(adapter(seed.backend), schema_initializer=already_initialized)
    identity = "did:test:sdk-native-avatar"
    files = AsyncFileStore(db, agent_id=identity)
    try:
        assert await files.store_avatar(b"valid avatar", identity) == hashlib.sha256(b"valid avatar").hexdigest()
        native_add = AsyncGraphStore.add_edge

        async def refuse_edge(*args, **kwargs):
            raise ValueError("native avatar edge refusal")

        monkeypatch.setattr(AsyncGraphStore, "add_edge", refuse_edge)
        with pytest.raises(Exception, match="native avatar edge refusal"):
            await files.store_avatar(b"rollback avatar", identity)
        assert await db.fetchone("SELECT 1 FROM files WHERE content_hash=?", (hashlib.sha256(b"rollback avatar").hexdigest(),)) is None
        monkeypatch.setattr(AsyncGraphStore, "add_edge", native_add)
        async with db.transaction():
            with pytest.raises(NotImplementedError, match="isolated nested"):
                await files.store_avatar(b"nested refusal", identity)
        assert await db.fetchone("SELECT 1 FROM files WHERE content_hash=?", (hashlib.sha256(b"nested refusal").hexdigest(),)) is None
        if adapter is ImmediateOnlyAdapter:
            async with db.transaction(immediate=True):
                assert db.owns_open_transaction is True
    finally:
        await db.close()
