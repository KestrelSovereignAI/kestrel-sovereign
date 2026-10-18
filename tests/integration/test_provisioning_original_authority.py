"""Original provisioning authority reaches actual HTTP dispatch and disclosure.

The only peer is owned loopback HTTP. All credential-shaped values are declared
inert protocol inputs, never real provider credentials or key creation.
"""

import asyncio
import json

import aiosqlite
import httpx
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    ExecutionCustody,
    bind_execution_custody,
    execution_denial_scopes,
    is_execution_control_error,
)
from kestrel_sovereign.features.llm_keys import openrouter_provisioning as module
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize(("entry", "phase"), [
    ("service", "response"), ("mint", "response"), ("legacy", "response"),
    ("mint", "close"), ("legacy", "close"),
    ("service", "dispatch"), ("mint", "dispatch"), ("service", "client"),
    ("mint", "response-cleanup"), ("legacy", "response-cleanup"),
    ("service-usage", "response"), ("service-update", "response"),
    ("service-delete", "response"), ("service-list", "response"),
    ("sdk-usage", "close"), ("sdk-update", "close"),
    ("legacy-usage", "close"), ("legacy-delete", "close"),
    ("service", "normalized"), ("service-usage", "entry"),
    ("service-update", "entry"), ("service-delete", "entry"), ("service-list", "entry"),
    ("bootstrap", "close"), ("registry", "close"),
    ("bootstrap", "entry"), ("bootstrap", "response-cleanup"),
    ("bootstrap", "cached-entry"),
    ("payer", "close"), ("payer", "response-cleanup"),
    ("payer-revoke", "revoke-control"),
])
async def test_provisioning_original_authority_fences_native_dispatch_and_disclosure(
    native_pg, tmp_path, monkeypatch, entry, phase,
):
    entered, release = asyncio.Event(), asyncio.Event()
    peers, requests, clients, scopes = set(), [], [], []
    originals, graphs, cleanup_errors = [], [], []
    from kestrel_sovereign.llm import provider_registry as registry_module
    monkeypatch.setattr(registry_module, "_BOOTSTRAP_OPENROUTER_KEYS", {})
    monkeypatch.setattr(registry_module, "_BOOTSTRAP_OPENROUTER_LOCK", None)
    if phase == "cached-entry":
        registry_module._BOOTSTRAP_OPENROUTER_KEYS["declared-inert-loopback-management"] = "declared-inert-cached-child"
    registries = []
    payer_db = payer_storage = payer = None
    if entry in {"payer", "payer-revoke"}:
        from kestrel_sovereign.security.host_key_storage import HostKeyStorage
        from kestrel_sovereign.services.payer_resolver import FoundationPayerResolver
        from kestrel_sovereign.storage.async_storage import AsyncStorage
        from tests.unit.test_payer_resolver_mint import _host_master_policy, _seed_agent_graph_node
        monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-32-bytes-fixed--")
        # Declared component graph input, not Agent genesis or READY evidence.
        payer_storage = AsyncStorage(backend=native_pg[0])
        await payer_storage.initialize()
        payer_db = payer_storage.db
        await _seed_agent_graph_node(payer_db, "did:test:inert-payer-input")
        await HostKeyStorage(payer_db).store_key("openrouter", "declared-inert-loopback-management")
        payer = FoundationPayerResolver(_host_master_policy(), db=payer_db)
    foreign_scope = foreign_original = None
    foreign_graph = ()
    if phase == "revoke-control":
        async with native_pg[0].advisory_locks([(3569, 18231)]) as lease:
            foreign_scope = ExecutionCustody(GenerationFence(lease=lease))
        try:
            foreign_scope.require_work()
        except BaseException as error:
            foreign_original = error
            foreign_graph = tuple(_explicit_control_chain(error))
        assert is_execution_control_error(foreign_original)
    closed_db = None
    if phase == "response-cleanup":
        closed_db = await aiosqlite.connect(":memory:")
        await closed_db.close()
    body = json.dumps({"key": "declared-inert-loopback-child",
                       "data": [] if entry == "service-list" else
                       {"hash": "declared-inert-loopback-hash"}}).encode()

    async def peer(reader, writer):
        task = asyncio.current_task()
        peers.add(task)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            method, path, _ = head.split(b"\r\n", 1)[0].split(b" ")
            assert method in {b"POST", b"GET", b"PATCH", b"DELETE"}
            assert path in {b"/keys", b"/keys/declared-inert-loopback-hash"}
            length = next((int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                           if line.lower().startswith(b"content-length:")), 0)
            requests.append((method, path, await reader.readexactly(length)))
            if phase.startswith("response"):
                entered.set()
                await release.wait()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                         + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            peers.discard(task)

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    endpoint = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    monkeypatch.setattr(module, "get_openrouter_api_base", lambda: endpoint)
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_API_KEY", "declared-inert-loopback-management")
    actual_get = module.OpenRouterProvisioningService._get_client
    actual_close = module.OpenRouterProvisioningService.close
    actual_create = module.OpenRouterProvisioningService.create_agent_key
    actual_status = httpx.Response.raise_for_status
    close_observed = set()

    def observed_status(response):
        result = actual_status(response)
        if phase == "normalized":
            scopes[0].revoke("actual original runtime retired during response normalization")
            entered.set()
        return result

    monkeypatch.setattr(httpx.Response, "raise_for_status", observed_status)

    async def hold_request(_request):
        entered.set()
        await release.wait()

    async def original_revoke_control(response):
        if response.request.method == "DELETE":
            entered.set()
            await release.wait()
            raise foreign_original

    async def observed_get(service):
        client = await actual_get(service)
        if client not in clients:
            clients.append(client)
            if phase == "revoke-control":
                client.event_hooks["response"].append(original_revoke_control)
            if phase == "dispatch":
                client.event_hooks["request"].append(hold_request)
            elif phase == "client":
                entered.set()
                await release.wait()
        return client

    async def observed_close(service):
        await actual_close(service)
        if closed_db is not None:
            # The actual HTTP close has settled. A real closed native handle
            # supplies an independent epilogue failure, not a fake HTTP ACK.
            try:
                await closed_db.execute("SELECT 1")
            except ValueError as error:
                cleanup_errors.append(error)
                raise
        if phase == "close" and id(service) not in close_observed:
            close_observed.add(id(service))
            entered.set()
            await release.wait()

    async def observed_create(service, *args, **kwargs):
        try:
            result = await actual_create(service, *args, **kwargs)
            if entry == "payer-revoke":
                # The same real graph row disappears after native mint ACK.
                await payer_db.execute("DELETE FROM graph_nodes WHERE node_id = ?",
                                       ("did:test:inert-payer-input",))
            return result
        except BaseException as error:
            originals.append(error)
            graphs.append(tuple(_explicit_control_chain(error)))
            raise

    monkeypatch.setattr(module.OpenRouterProvisioningService, "_get_client", observed_get)
    monkeypatch.setattr(module.OpenRouterProvisioningService, "close", observed_close)
    monkeypatch.setattr(module.OpenRouterProvisioningService, "create_agent_key", observed_create)
    direct = module.OpenRouterProvisioningService("declared-inert-loopback-management")

    async def run():
        with bind_execution_custody(GenerationFence()) as original:
            scopes.append(original)
            if phase in {"entry", "cached-entry"}:
                original.revoke("actual original runtime retired before service admission")
                entered.set()
            if entry.startswith("service"):
                operation = {
                    "service": lambda: direct.create_agent_key("native-loopback-agent"),
                    "service-usage": lambda: direct.get_key_usage("declared-inert-loopback-hash"),
                    "service-update": lambda: direct.update_key_limit("declared-inert-loopback-hash", 0.10),
                    "service-delete": lambda: direct.delete_key("declared-inert-loopback-hash"),
                    "service-list": direct.list_keys,
                }[entry]
                return await operation()
            if entry == "mint":
                return await module.mint_managed_openrouter_key(
                    "declared-inert-loopback-management", "native-loopback-agent",
                )
            if entry == "bootstrap":
                return await registry_module._mint_bootstrap_openrouter_key(
                    "declared-inert-loopback-management", host_db=None,
                )
            if entry in {"payer", "payer-revoke"}:
                from kestrel_sdk.payer_policy import ResourceClass
                return await payer.resolve_for("did:test:inert-payer-input", ResourceClass.LLM)
            if entry == "registry":
                monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
                # Explicit fixture installation inventory: only the configured
                # built-in loopback route. Never instantiate unrelated installed
                # third-party providers or their credential discovery chains.
                monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes",
                                    lambda group, base: {})
                registry = registry_module.ProviderRegistry(config={
                    "route_priority": ["openrouter:api"], "vendors": {"openrouter": {
                        "is_cloud": True, "routes": {"api": {
                            "adapter": "OpenRouterAdapter", "model": "auto",
                            "api_key_env": "OPENROUTER_API_KEY", "base_url": endpoint,
                        }},
                    }},
                })
                registries.append(registry)
                assert registry.initialize_providers() == []
                return await registry.finalize_providers(host_db=None)
            if entry == "sdk-usage":
                return await module.get_provider_key_usage(
                    "declared-inert-loopback-management", "declared-inert-loopback-hash",
                )
            if entry == "sdk-update":
                return await module.update_provider_key_limit(
                    "declared-inert-loopback-management", "declared-inert-loopback-hash", 0.10,
                )
            if entry == "legacy-usage":
                return await module.get_agent_usage("declared-inert-loopback-hash")
            if entry == "legacy-delete":
                return await module.delete_agent_key("declared-inert-loopback-hash")
            return await module.provision_agent_key("native-loopback-agent")

    task = create_owned_task(run(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert len(scopes) == 1
        if phase != "revoke-control":
            scopes[0].revoke("actual original provisioning runtime retired")
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(task), 5)
        if entry in {"bootstrap", "registry"} and phase != "cached-entry":
            assert not registry_module._BOOTSTRAP_OPENROUTER_KEYS, "bootstrap published a credential after original runtime authority retired"
            assert all(not registry.providers for registry in registries), "bootstrap published a route after original runtime authority retired"
        assert outcome.error is not None and outcome.result is None, (
            "provisioning disclosed a key after original runtime authority retired"
        )
        assert is_execution_control_error(outcome.error), "provisioning normalized original runtime control"
        if phase == "revoke-control":
            assert foreign_original in tuple(_explicit_control_chain(outcome.error))
            assert tuple(_explicit_control_chain(foreign_original)) == foreign_graph
            assert foreign_scope in (execution_denial_scopes(outcome.error) or ())
        else:
            assert scopes[0] in (execution_denial_scopes(outcome.error) or ())
        expected_requests = 2 if entry == "payer-revoke" or entry == "sdk-update" and phase == "close" else (
            1 if phase.startswith("response") or phase in {"close", "normalized"} else 0
        )
        assert len(requests) == expected_requests, (
            "provisioning dispatched after its original native admission retired"
        )
        if phase == "entry":
            assert not clients, "provisioning acquired a native client after original admission retired"
        if phase == "response-cleanup":
            chain = tuple(_explicit_control_chain(outcome.error))
            assert originals and all(error in chain for error in originals), (
                "provisioning cleanup replaced the original native body denial"
            )
            assert cleanup_errors and all(error in chain for error in cleanup_errors), (
                "provisioning cleanup erased its independent native epilogue failure"
            )
            assert all(tuple(_explicit_control_chain(error)) == graph
                       for error, graph in zip(originals, graphs)), (
                "provisioning cleanup mutated the original shared body receipt"
            )
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await await_owned_task(task)
        await actual_close(direct)
        for client in clients:
            await client.aclose()
        for registry in registries:
            for provider in registry.providers:
                await provider.client.close()
        if payer_storage is not None:
            await payer_storage.close()
        server.close()
        await server.wait_closed()
        for peer_task in tuple(peers):
            await await_owned_task(peer_task)
