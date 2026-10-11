"""Failed construction keeps the real provider and its original close receipt."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.execution_custody import current_execution_custody
from kestrel_sovereign.llm.lifecycle import drain_unpublished_llm_owners
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_execution_custody_review22 import stub_single_startup
from tests.unit.test_execution_custody_review30 import service_owner
from tests.unit.test_execution_custody_review32 import receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["server", "manager"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group", "pending", "ordinary"])
async def test_actual_failed_constructor_retains_native_provider_cleanup(
    monkeypatch, tmp_path, surface, kind,
):
    from kestrel_sovereign.llm import service as service_module
    from kestrel_sovereign.multi_agent import agent_manager as manager_module
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig

    service = service_owner()
    db = await AsyncDatabase.sqlite(str(tmp_path / "provider.db"))
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    entered, release = asyncio.Event(), asyncio.Event()
    scopes, calls = [], []
    failure = receipt(kind)
    if kind != "pending":
        release.set()

    async def close_adapter():
        # The actual native worker retires before its delayed receipt arrives.
        # Cleanup does not need (and must not mint) ordinary mutation authority.
        await db.backend._connection.close()
        calls.append(True)
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        if kind != "pending":
            raise failure

    service.providers = [{"adapter": SimpleNamespace(aclose=close_adapter)}]
    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.01)

    def provider(**kwargs):
        scopes.extend(current_execution_custody())
        return service

    def constructor(**kwargs):
        raise ValueError("actual constructor rejected birth")

    if surface == "server":
        server, app = stub_single_startup(monkeypatch, tmp_path)
        monkeypatch.setattr(server, "LLMService", provider)
        monkeypatch.setattr(server, "KestrelAgent", constructor)

        async def initialize():
            async with server._lifespan_startup(app):
                pass

        state = app.state
        owners_name = "unpublished_llm_owners"
    else:
        monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
        monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
        manager = manager_module.AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(data_dir=str(tmp_path), port=18888)
        monkeypatch.setattr(LocalAgentConfig, "validate_runtime", lambda *_a, **_kw: [])
        monkeypatch.setattr(manager_module, "read_anchor_agent_did", AsyncMock(return_value="did:test:unborn"))
        monkeypatch.setattr(manager_module, "LLMService", provider)
        monkeypatch.setattr(manager_module, "KestrelAgent", constructor)

        async def initialize():
            await manager._initialize_agent("child", config)

        state = manager
        owners_name = "_unpublished_llm_owners"
    try:
        if kind in {"unknown", "cancel_unknown", "group"}:
            with pytest.raises(BaseException) as caught:
                await initialize()
            assert caught.value is failure
        elif surface == "manager":
            with pytest.raises(ValueError, match="constructor rejected"):
                await initialize()
        else:
            await initialize()
        assert entered.is_set() and calls == [True]
        owners = getattr(state, owners_name, {})
        if kind == "ordinary":
            assert not owners
        else:
            assert len(owners) == 1
            (owner,) = owners.values()
            assert owner.provider is service
            assert owner.scopes == tuple(scopes)
            if surface == "server":
                with pytest.raises(Exception, match="retire|cleanup|provider"):
                    async with server._lifespan_startup(app):
                        pass
            else:
                with pytest.raises(Exception, match="retire|cleanup|provider"):
                    await manager._initialize_agent("child", config)
            assert calls == [True]
            if kind == "pending":
                with pytest.raises(RuntimeError, match="incomplete|pending"):
                    await drain_unpublished_llm_owners(owners)
                assert len(owners) == 1 and calls == [True]
                release.set()
                await service.wait_for_close_completion()
                await drain_unpublished_llm_owners(owners)
                assert not owners and calls == [True]
            else:
                with pytest.raises(BaseException) as caught:
                    await drain_unpublished_llm_owners(owners)
                assert caught.value is failure
                assert len(owners) == 1 and calls == [True]
    finally:
        release.set()
        try:
            await service.wait_for_close_completion()
        except BaseException:
            pass
        await db.close()
