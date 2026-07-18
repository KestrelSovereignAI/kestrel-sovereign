"""Storage-backend-aware server identity discovery tests (#2472)."""

from types import SimpleNamespace

import pytest

from kestrel_sovereign import main as main_module


class _FakeStorage:
    instances = []
    nodes = [SimpleNamespace(node_id="did:web:agents.example.com:kestrel")]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.initialized = False
        self.closed = False
        self.__class__.instances.append(self)

    async def initialize(self):
        self.initialized = True

    async def get_nodes_by_type(self, node_type):
        assert node_type == "agent"
        return self.__class__.nodes

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_postgres_identity_discovery_uses_durable_database(monkeypatch):
    _FakeStorage.instances.clear()
    _FakeStorage.nodes = [
        SimpleNamespace(node_id="did:web:agents.example.com:kestrel")
    ]
    monkeypatch.setattr(main_module, "AsyncStorage", _FakeStorage)

    did = await main_module.get_agent_did_async(
        "/disposable/identity",
        db_backend="postgres",
        database_url="postgresql://durable.example/kestrel",
    )

    assert did == "did:web:agents.example.com:kestrel"
    storage = _FakeStorage.instances[-1]
    assert storage.args == ()
    assert storage.kwargs == {
        "backend": "postgres",
        "dsn": "postgresql://durable.example/kestrel",
    }
    assert storage.initialized and storage.closed


@pytest.mark.asyncio
async def test_postgres_identity_discovery_requires_dsn(monkeypatch):
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="KESTREL_DATABASE_URL is required"):
        await main_module.get_agent_did_async(
            "/disposable/identity",
            db_backend="postgres",
        )


@pytest.mark.asyncio
async def test_sqlite_identity_discovery_keeps_local_path(monkeypatch):
    _FakeStorage.instances.clear()
    _FakeStorage.nodes = [
        SimpleNamespace(node_id="did:web:agents.example.com:kestrel")
    ]
    monkeypatch.setattr(main_module, "AsyncStorage", _FakeStorage)

    await main_module.get_agent_did_async("/local/agent", db_backend="sqlite")

    storage = _FakeStorage.instances[-1]
    assert storage.args == ("/local/agent/kestrel_prime.db",)
    assert storage.kwargs == {}


@pytest.mark.asyncio
async def test_postgres_identity_discovery_refuses_ambiguous_database(monkeypatch):
    _FakeStorage.instances.clear()
    _FakeStorage.nodes = [
        SimpleNamespace(node_id="did:web:agents.example.com:kestrel"),
        SimpleNamespace(node_id="did:web:agents.example.com:other"),
    ]
    monkeypatch.setattr(main_module, "AsyncStorage", _FakeStorage)

    with pytest.raises(ValueError, match="exactly one agent node"):
        await main_module.get_agent_did_async(
            "/disposable/identity",
            db_backend="postgres",
            database_url="postgresql://durable.example/kestrel",
        )

    assert _FakeStorage.instances[-1].closed
