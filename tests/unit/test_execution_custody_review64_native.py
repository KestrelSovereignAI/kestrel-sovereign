"""Native temporary transport and cold console-chain custody regressions."""
import asyncio

import httpx
import pytest
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts, _explicit_control_chain
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["local-record", "openrouter"])
async def test_discovery_failure_retains_original_temporary_native_client_for_shutdown(tmp_path, monkeypatch, lane):
    peers, attempts, requests = set(), [], []
    original = OSError("original temporary catalog client close acknowledgement")
    async def peer(reader, writer):
        peers.add(asyncio.current_task())
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            body = b'{"data": []}'
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" +
                f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            peers.discard(asyncio.current_task())
    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    monkeypatch.setattr("kestrel_sovereign.llm.service.ProviderRegistry.initialize_providers", lambda self: [])
    service = LLMService(agent_data_dir=tmp_path)
    native_close = httpx.AsyncClient.aclose
    async def failed_close(client):
        await native_close(client)
        attempts.append(client)
        raise original
    monkeypatch.setattr(httpx.AsyncClient, "aclose", failed_close)
    source_graph = tuple(_explicit_control_chain(original))
    source_debts = tuple(execution_cleanup_debts(original))
    try:
        with pytest.raises(BaseException) as caught:
            if lane == "local-record":
                await service._record_discovery("offline-local", service._discover_local_openai_compatible(
                    "offline-local", {"base_url": base, "context_limit": 4096}))
            else:
                adapter = OpenRouterAdapter()
                adapter.api_key, adapter.base_url = "loopback-not-a-secret", base
                await service._safe_list_models("offline-openrouter", adapter, None)
        assert original in tuple(_explicit_control_chain(caught.value))
        assert tuple(_explicit_control_chain(original)) == source_graph
        assert tuple(execution_cleanup_debts(original)) == source_debts
        assert len(requests) == len(attempts) == 1
        resource, closer = execution_cleanup_debts(caught.value)[0]
        assert resource is attempts[0] and closer.done() and closer.exception() is original
        assert service.has_pending_close_owners()
        for _ in range(2):
            with pytest.raises(BaseException) as shutdown:
                await service.close()
            assert original in tuple(_explicit_control_chain(shutdown.value))
            assert tuple(_explicit_control_chain(original)) == source_graph
            assert tuple(execution_cleanup_debts(original)) == source_debts
        assert attempts == [resource], "shutdown retried a failed original temporary transport close"
        assert resource in service._llm_native_retirement_resources.values()
    finally:
        monkeypatch.undo()
        try:
            await service.close()
        except BaseException as retained:
            assert original in tuple(_explicit_control_chain(retained))
        server.close()
        await server.wait_closed()
        for peer_task in tuple(peers):
            await await_owned_task(peer_task)


def test_accepted_console_chain_retains_mutable_namespace_custody_after_restart(tmp_path, monkeypatch):
    from kestrel_sovereign.features import isolated_runtime as native
    from tests.unit.test_isolated_feature_runtime import (
        _hosted_postgres_agent, _isolated_runtime, _write_prebuilt_venv_shape,
    )
    runtime, prebuilt = _isolated_runtime(), tmp_path / "immutable-operator-venv"
    _write_prebuilt_venv_shape(prebuilt, console_service=runtime.service)
    agent = _hosted_postgres_agent(tmp_path / "owned-runtime", "original-console")
    feature = native.ProxyFeature(agent, runtime)
    workspace = feature._prepare_runtime_workspace()
    public = native._console_script_path(prebuilt, runtime.service)
    target = public.with_name(public.name + "-original")
    public.rename(target)
    intermediate = workspace / "immutable-console-link"
    intermediate.symlink_to(target)
    public.symlink_to(intermediate)
    setting = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(setting, str(prebuilt))
    feature = native.ProxyFeature(agent, runtime)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    # Distribution probing is independent of this disk-only chain acceptance;
    # no isolated child/provider is executed by this regression.
    monkeypatch.setattr(feature, "_verify_prebuilt_feature_distribution", lambda *args: None)
    monkeypatch.setattr(feature, "_warn_on_sdk_mismatch", lambda *args: None)
    feature.ensure_venv()
    assert feature._validated_hosted_console_path == target.resolve()
    scope = feature._isolated_runtime_scope
    agent._isolated_runtime_cleanup_owners.clear()
    monkeypatch.delenv(setting)
    with pytest.raises(native.IsolatedRuntimePreparationError, match="immutable"):
        native.remove_runtime_namespace(scope, agent.did)
    assert intermediate.is_symlink() and public.resolve() == target.resolve()
