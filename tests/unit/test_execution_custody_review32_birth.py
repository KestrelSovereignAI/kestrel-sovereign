"""Failed construction keeps the real provider and its original close receipt."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    current_execution_custody,
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.llm.lifecycle import drain_unpublished_llm_owners
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_execution_custody_review22 import stub_single_startup
from tests.unit.test_execution_custody_review30 import service_owner
from tests.unit.test_execution_custody_review32 import receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["server", "manager"])
@pytest.mark.parametrize("kind,late_receipt", [
    (kind, False) for kind in ("unknown", "cancel_unknown", "group", "pending", "ordinary")
] + [(kind, True) for kind in ("unknown", "cancel_unknown", "group")])
async def test_actual_failed_constructor_retains_native_provider_cleanup(
    monkeypatch, tmp_path, surface, kind, late_receipt,
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
    source = OriginalFailure.capture(failure)
    if kind != "pending" and not late_receipt:
        release.set()

    async def close_adapter():
        # The actual native worker retires before its delayed receipt arrives.
        # Cleanup does not need (and must not mint) ordinary mutation authority.
        # The injected receipt belongs to the physical native close, not to
        # cancellation of the test's waiter before that close has completed.
        native = asyncio.create_task(db.backend._connection.close())
        outcome = await await_owned_task(native)
        assert outcome.error is None
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
    if kind == "pending" or late_receipt:
        # Only delayed-receipt cases force a bounded startup handoff. Immediate
        # receipts use the real production deadline, not a 10ms scheduling bet.
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
        if kind in {"unknown", "cancel_unknown", "group", "ordinary"} and not late_receipt:
            with pytest.raises(BaseException) as caught:
                await initialize()
            assert failure in tuple(_explicit_control_chain(caught.value))
            if kind == "ordinary":
                # This error came from an ORIGINAL physical adapter close,
                # not an ordinary writer. Its failed ACK now truthfully owns
                # native retirement debt and must make startup fail closed.
                source.assert_retained(caught.value, resource=service.providers[0]["adapter"])
                assert is_execution_control_error(caught.value)
                assert not is_execution_control_error(failure)
                assert any(resource is service.providers[0]["adapter"]
                           for resource, _ in execution_cleanup_debts(caught.value))
        elif surface == "manager":
            with pytest.raises(ValueError, match="constructor rejected"):
                await initialize()
        else:
            await initialize()
        await asyncio.wait_for(entered.wait(), 5)
        assert calls == [True]
        owners = getattr(state, owners_name, {})
        if kind == "ordinary":
            # An adapter's failed close receipt is not positive retirement
            # proof, even if this test knows its underlying SQLite is closed.
            # The production owner cannot assume that about arbitrary adapters.
            assert len(owners) == 1
            with pytest.raises(BaseException) as closed:
                await drain_unpublished_llm_owners(owners)
            source.assert_retained(closed.value, resource=service.providers[0]["adapter"])
            assert service.has_pending_close_owners()
            assert calls == [True]
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
            if kind == "pending" or late_receipt:
                drain = asyncio.create_task(drain_unpublished_llm_owners(owners))
                done, _ = await asyncio.wait({drain}, timeout=0.4)
                assert not done, "host drain must own the native join itself"
                assert len(owners) == 1 and calls == [True]
                release.set()
                if late_receipt:
                    with pytest.raises(BaseException) as caught:
                        await drain
                    assert failure in tuple(_explicit_control_chain(caught.value))
                    assert len(owners) == 1 and calls == [True]
                    with pytest.raises(BaseException) as caught:
                        await drain_unpublished_llm_owners(owners)
                    assert failure in tuple(_explicit_control_chain(caught.value))
                    assert calls == [True]
                else:
                    await drain
                    assert not owners and calls == [True]
            else:
                with pytest.raises(BaseException) as caught:
                    await drain_unpublished_llm_owners(owners)
                assert failure in tuple(_explicit_control_chain(caught.value))
                assert len(owners) == 1 and calls == [True]
    finally:
        release.set()
        try:
            await service.wait_for_close_completion()
        except BaseException:
            pass
        await db.close()
