"""The same real permission oracles, through the installed Docker MCP Gateway."""

import asyncio
from contextlib import nullcontext
import json
import os
import signal
from pathlib import Path

import pytest

from tests.integration.test_mcp_permission_boundary import (
    mcp_boundary as mcp_boundary,
    test_denied_tool_has_no_generic_dispatcher_bypass as _denial,
    test_one_shot_approval_cannot_be_replayed_for_an_external_call as _once,
    test_pending_approval_cannot_execute_against_a_replacement_connection as _replacement,
)


pytestmark = [pytest.mark.mcp_integration, pytest.mark.docker_mcp, pytest.mark.timeout(120)]


@pytest.mark.parametrize("mcp_boundary", ["gateway"], indirect=True)
@pytest.mark.parametrize("oracle", [_denial, _once, _replacement], ids=["deny", "approve-once", "replace"])
async def test_actual_docker_gateway_obeys_the_host_permission_boundary(mcp_boundary, oracle):
    boundary = mcp_boundary
    gateway = boundary.feature.gateway_manager.gateway
    process = gateway.process
    assert process.poll() is None and process.pid > 0
    assert process.args[:4] == ["docker", "mcp", "gateway", "run"]
    assert "--profile=" + gateway._profile_id in process.args
    # Startup capability discovery has no client session and retires its
    # discovery container. The first permitted call acquires the session-owned
    # long-lived peer; denial can therefore leave no peer container at all.
    assert len(await boundary.peer.containers()) <= 1
    if oracle is _denial:
        await oracle(boundary)
    else:
        # A permitted Gateway call includes a fresh native stdio process
        # handshake. Delegate its deadline to the actual MCP connection,
        # bounded again by this test's existing 120-second overall deadline.
        # Approval/replay/refusal observations retain their five-second bounds.
        await oracle(boundary, approved_timeout=None)
    assert boundary.current_connection().is_connected
    containers = await boundary.peer.containers()
    assert len(containers) <= 1
    if oracle is not _denial:
        assert len(containers) == 1, "A permitted native call must retain its session-owned stdio peer"


@pytest.mark.parametrize("mcp_boundary", ["gateway-held"], indirect=True)
@pytest.mark.parametrize("fail_witness", [False, True], ids=["normal", "witness-error"])
async def test_cancelled_native_gateway_call_physically_retires_its_peer(mcp_boundary, fail_witness):
    from kestrel_sovereign.features.security.permissions import PermissionLevel
    from tests.utils.virtual_clock import await_within_hang_guard

    boundary = mcp_boundary
    gateway = boundary.feature.gateway_manager.gateway
    process, profile, configuration = gateway.process, gateway._profile_id, gateway._config_directory
    connection = boundary.current_connection()
    await boundary.store.set_permission(boundary.owner, boundary.handle, PermissionLevel.ASK)
    call = boundary.invoke("cancelled-cold-peer")
    request = await asyncio.wait_for(boundary.requests.get(), 5)
    assert await boundary.queue.submit_decision(request.id, True, scope="once")

    async def entered_native_call():
        while await boundary.witness() != ["cancelled-cold-peer"]:
            assert not call.done(), "Native call settled before its external effect witness"
            await asyncio.sleep(0.01)

    expectation = (
        pytest.raises(AssertionError, match="Owned native witness observation failed")
        if fail_witness else nullcontext()
    )
    with expectation:
        try:
            await await_within_hang_guard(entered_native_call(), step="held native peer effect")
            if fail_witness:
                raise AssertionError("Owned native witness observation failed")
            assert await boundary.peer.held_call_witness() == ["cancelled-cold-peer"]
            assert not call.done(), "The externally witnessed peer barrier returned a reply"
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
        finally:
            # A failed effect/held-witness observation must not leave an
            # already-approved infinite native call for fixture drain_calls.
            try:
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
            finally:
                await boundary.feature.shutdown()
    assert call.cancelled()
    assert not await boundary.queue.submit_decision(request.id, True, scope="once")
    assert gateway.process is None and process.poll() is not None
    assert not connection.is_connected
    assert gateway._profile_id is None and gateway._profile_pending is None, profile
    assert not configuration.exists()
    assert await boundary.peer.containers() == [], "Gateway acknowledged retirement with a live native peer"
    settled = await boundary.witness()
    assert settled == ["cancelled-cold-peer"]
    assert await boundary.witness() == settled, "A native effect arrived after retirement acknowledgement"


@pytest.mark.parametrize("mcp_boundary", ["gateway"], indirect=True)
async def test_invalid_gateway_replacement_preserves_the_original_native_profile(mcp_boundary):
    boundary = mcp_boundary
    gateway = boundary.feature.gateway_manager.gateway
    original = gateway.process, gateway._profile_id, gateway._config_directory, boundary.current_connection()
    with pytest.raises(ValueError, match="exact catalog server names"):
        await gateway.start(["../not-an-exact-server"])
    assert (gateway.process, gateway._profile_id, gateway._config_directory, boundary.current_connection()) == original
    assert gateway.process.poll() is None and boundary.current_connection().is_connected
    assert await boundary.witness() == []


@pytest.mark.parametrize("mcp_boundary", ["gateway"], indirect=True)
async def test_gateway_cli_configuration_is_private_and_retired_after_native_processes(mcp_boundary):
    boundary = mcp_boundary
    gateway = boundary.feature.gateway_manager.gateway
    operator_config = Path(os.environ["DOCKER_CONFIG"]) / "config.json"
    original = operator_config.read_bytes()
    private = gateway._config_directory
    assert private != operator_config.parent
    assert private.stat().st_mode & 0o777 == 0o700
    assert (private / "config.json").stat().st_mode & 0o777 == 0o600
    assert json.loads(original)["features"]["dynamic-tools"] == "enabled"
    assert json.loads((private / "config.json").read_text())["features"]["dynamic-tools"] == "disabled"
    assert not any("mcp-exec" in name or "code-mode" in name for name in boundary.agent._tool_to_feature)
    await boundary.feature.shutdown()
    assert gateway.process is None and gateway._profile_id is None and gateway._profile_pending is None
    assert not private.exists() and operator_config.read_bytes() == original
    assert await boundary.peer.containers() == []


@pytest.mark.parametrize("mcp_boundary", ["gateway"], indirect=True)
async def test_native_profile_cli_error_preserves_the_existing_gateway(mcp_boundary):
    from uuid import uuid4
    from kestrel_feature_mcp.gateway import DockerMCPGatewayError

    boundary = mcp_boundary
    gateway = boundary.feature.gateway_manager.gateway
    original = gateway.process, gateway._profile_id, gateway._config_directory
    with pytest.raises(DockerMCPGatewayError, match="profile operation failed"):
        await gateway._profile_command(["remove", "missing-owned-test-" + uuid4().hex])
    assert (gateway.process, gateway._profile_id, gateway._config_directory) == original
    assert gateway.process.poll() is None and gateway._profile_pending is None
    assert boundary.current_connection().is_connected
    assert await boundary.witness() == []


@pytest.mark.parametrize("mcp_boundary", ["gateway"], indirect=True)
@pytest.mark.parametrize("entrypoint", ["restart", "manager-start", "manager-enable", "feature-start", "feature-load", "missing-profile"])
async def test_public_gateway_replacement_validates_the_actual_catalog_before_retirement(mcp_boundary, entrypoint, monkeypatch):
    from kestrel_feature_mcp.gateway import DockerMCPGatewayError

    boundary = mcp_boundary
    manager = boundary.feature.gateway_manager
    gateway = manager.gateway
    original = gateway.process, gateway._profile_id, gateway._config_directory, manager.connection
    # time is in the bundled catalog but absent from this owned OCI catalog.
    if entrypoint == "missing-profile":
        with monkeypatch.context() as runtime:
            runtime.setenv("KESTREL_MCP_DOCKER_PROFILE", "missing-owned-source-profile")
            result = await boundary.feature.gateway_start(boundary.peer.selector)
            assert result.status.value == "error" and "source profile is absent" in result.error
    elif entrypoint == "feature-start":
        result = await boundary.feature.gateway_start("time")
        assert result.status.value == "error", result.to_dict()
    elif entrypoint == "feature-load":
        result = await boundary.feature.load_tool("time")
        assert result.status.value == "error", result.to_dict()
    else:
        with pytest.raises(DockerMCPGatewayError, match="absent from the configured"):
            if entrypoint == "restart":
                await gateway.restart(["time"])
            elif entrypoint == "manager-start":
                await manager.start(["time"])
            else:
                await manager.enable_server("time")
    assert boundary.feature.gateway_manager is manager
    assert (gateway.process, gateway._profile_id, gateway._config_directory, manager.connection) == original
    assert gateway.process.poll() is None and manager.connection.is_connected
    assert await boundary.witness() == []


@pytest.mark.parametrize("mcp_boundary", ["gateway-lingering"], indirect=True)
@pytest.mark.parametrize("public_action", ["direct", "stop", "unload", "replace"])
async def test_killed_gateway_cannot_acknowledge_a_surviving_native_container(mcp_boundary, public_action):
    from kestrel_sovereign.features.security.permissions import PermissionLevel

    boundary = mcp_boundary
    gateway = boundary.feature.gateway_manager.gateway
    process, private = gateway.process, gateway._config_directory
    await boundary.store.set_permission(boundary.owner, boundary.handle, PermissionLevel.ALLOW)
    result = await boundary.invoke("forced-retirement-witness")
    assert result["status"] == "ok" and await boundary.witness() == ["forced-retirement-witness"]
    containers = await boundary.peer.containers()
    assert len(containers) == 1
    try:
        os.killpg(process.pid, signal.SIGSTOP)
        if public_action == "direct":
            with pytest.raises(ExceptionGroup, match="MCP cleanup") as caught:
                await gateway.stop()
            assert caught.value.subgroup(lambda error: "Docker supplies no per-profile owner receipt" in str(error)) is not None
        else:
            profile = gateway._profile_id
            if public_action == "stop":
                result = await boundary.feature.gateway_stop()
            elif public_action == "unload":
                result = await boundary.feature.unload_tool(boundary.peer.selector)
            else:
                result = await boundary.feature.gateway_start(boundary.peer.selector)
            assert result.status.value == "error", result.to_dict()
            assert "Native MCP container retirement is unconfirmed for profile " + profile in result.error, result.to_dict()
            assert "Docker supplies no per-profile owner receipt" in result.error, result.to_dict()
        assert gateway.process is None and gateway._profile_id is not None and private.exists()
        assert {container.id for container in await boundary.peer.containers()} == {container.id for container in containers}
    finally:
        # The adversarial peer deliberately survives EOF and SIGTERM. On a red
        # run, remove only containers already attested to this private fixture.
        for container in await boundary.peer.containers():
            await asyncio.to_thread(container.remove, force=True)
        await gateway.stop()


@pytest.mark.parametrize("mcp_boundary", ["gateway-lingering"], indirect=True)
async def test_failed_native_container_inventory_retains_reconciliation_custody(mcp_boundary):
    boundary = mcp_boundary
    gateway = boundary.feature.gateway_manager.gateway
    private, profile, environment = gateway._config_directory, gateway._profile_id, gateway._process_env
    original_host = environment["DOCKER_HOST"]
    os.killpg(gateway.process.pid, signal.SIGSTOP)
    environment["DOCKER_HOST"] = "unix://" + str(private / "missing-owned-daemon.sock")
    try:
        with pytest.raises(ExceptionGroup, match="MCP cleanup") as caught:
            await gateway.stop()
        assert caught.value.subgroup(lambda error: "container inventory operation failed" in str(error)) is not None
        assert gateway._profile_id == profile and private.exists()
        assert gateway._containers_retired is False
    finally:
        environment["DOCKER_HOST"] = original_host
        await gateway.stop()


@pytest.mark.parametrize("mcp_boundary", ["gateway"], indirect=True)
@pytest.mark.parametrize("prefixed", [False, True])
async def test_concurrent_native_profiles_cannot_retire_each_others_server(mcp_boundary, prefixed):
    from kestrel_feature_mcp.manager import MCPGatewayManager
    from kestrel_sovereign.features.security.permissions import PermissionLevel

    boundary = mcp_boundary
    first = boundary.feature.gateway_manager
    second = MCPGatewayManager()
    operator_configuration = Path(os.environ["DOCKER_CONFIG"]) / "config.json"
    original = operator_configuration.read_bytes()
    configuration = json.loads(original)
    if prefixed:
        configuration["features"]["tool-name-prefix"] = "enabled"
    operator_configuration.write_text(json.dumps(configuration))
    try:
        try:
            await second.start([boundary.peer.selector])
        finally:
            operator_configuration.write_bytes(original)
        boundary.peer.remember_gateway(second.gateway)
        assert first.gateway._profile_id != second.gateway._profile_id
        second_tool = (boundary.peer.selector + "__" if prefixed else "") + "record_marker"
        assert set(second.tools) == {second_tool}
        await second.call_tool(second_tool, {"marker": "second-native-profile"})
        await boundary.store.set_permission(boundary.owner, boundary.handle, PermissionLevel.ALLOW)
        assert (await boundary.invoke("first-native-profile"))["status"] == "ok"
        assert len(await boundary.peer.containers()) == 2
        with pytest.raises(ExceptionGroup, match="MCP cleanup") as caught:
            await boundary.feature.shutdown()
        assert caught.value.subgroup(lambda error: "Docker supplies no per-profile owner receipt" in str(error)) is not None
        assert first.gateway.process is None and first.gateway._profile_id is not None
        remaining = await boundary.peer.containers()
        assert len(remaining) == 1 and second.is_connected
        await second.call_tool(second_tool, {"marker": "second-still-connected"})
        assert await boundary.witness() == ["second-native-profile", "first-native-profile", "second-still-connected"]
        await second.stop()
        await boundary.feature.shutdown()
        assert first.gateway._profile_id is None and second.gateway._profile_id is None
    finally:
        operator_configuration.write_bytes(original)
        for container in await boundary.peer.containers():
            await asyncio.to_thread(container.remove, force=True)
        await second.stop()
        await boundary.feature.shutdown()


@pytest.mark.parametrize("close_failure", [None, OSError("owned native connection close epilogue"), asyncio.CancelledError("owned native connection close cancellation")], ids=["normal-close", "failed-close", "cancelled-close"])
async def test_native_configured_profile_retains_values_and_template_namespace(tmp_path, monkeypatch, close_failure):
    from kestrel_feature_mcp.gateway import DockerMCPGateway
    from kestrel_feature_mcp.connection import MCPConnection
    from tests.integration.mcp_permission_peers import gateway_peer

    async with gateway_peer(tmp_path, monkeypatch, configured=True) as peer:
        gateway = DockerMCPGateway()
        connection = None
        original_process = None
        original_configuration = None
        armed_failure = None
        try:
            try:
                token = await gateway.start([peer.selector])
                original_process = gateway.process
                original_configuration = gateway._config_directory
                exported = gateway._config_directory / "configured-profile-proof.json"
                exported.touch(mode=0o600)
                await gateway._profile_command(["export", gateway._profile_id, str(exported)])
                configured = json.loads(exported.read_text())
                definition, = configured["servers"]
                assert definition["snapshot"]["server"]["name"] == peer.selector
                assert definition["config"] == {"mode": "owned-configured-mode"}
                assert definition["tools"] == ["record_marker"]
                assert definition["secrets"] == "owned-source-binding"
                assert configured["secrets"]["owned-source-binding"] == {"provider": "docker-desktop-store"}
                connection = MCPConnection(gateway.endpoint_url, headers={"Authorization": "Bearer " + token})
                tools = await connection.connect()
                assert [tool.name for tool in tools] == ["record_marker"]
                result = await connection.call_tool("record_marker", {"marker": "configured-native-profile"})
                assert not result.is_error
                assert await peer.witness(None, None) == ["configured-native-profile"]
                if close_failure is not None:
                    original_close = connection.close

                    async def close_then_fail():
                        await original_close()
                        raise close_failure

                    monkeypatch.setattr(connection, "close", close_then_fail)
                    armed_failure = close_failure
            finally:
                peer.remember_gateway(gateway)
                expected = pytest.raises(type(armed_failure)) if armed_failure is not None else nullcontext()
                with expected as caught:
                    try:
                        if connection is not None:
                            await connection.close()
                    finally:
                        await gateway.stop()
                if armed_failure is not None:
                    assert caught.value is armed_failure
            assert gateway.process is None and original_process.poll() is not None, "connection close failure skipped independent Gateway retirement"
            assert gateway._profile_id is None and not original_configuration.exists()
        finally:
            # Explicit test-owner reconciliation on a deliberately broken guard.
            await gateway.stop()


@pytest.mark.parametrize("mcp_boundary", ["gateway"], indirect=True)
async def test_native_remote_profile_keeps_its_original_oauth_identity(mcp_boundary):
    from uuid import uuid4
    from kestrel_feature_mcp.gateway import DockerMCPGateway
    from tests.integration.mcp_permission_peers import create_owned_mcp_catalog

    boundary = mcp_boundary
    server = "remote-identity-" + uuid4().hex
    reference = "kestrel-permission/remote:" + uuid4().hex
    catalog_file = Path(os.environ["DOCKER_CONFIG"]) / "mcp/catalogs/remote-identity.yaml"
    await create_owned_mcp_catalog(catalog_file, {"version": 2, "name": "docker-mcp", "registry": {
        server: {"name": server, "type": "remote", "remote": {"url": "https://metadata-only.invalid/mcp"},
                 "description": "Native metadata identity; no server is contacted"},
    }}, reference)
    gateway = DockerMCPGateway()
    try:
        gateway._prepare_gateway_configuration()
        await gateway._create_profile(reference, [server])
        export = gateway._config_directory / "identity-proof.json"
        await gateway._profile_command(["export", gateway._profile_id, str(export)])
        definition, = json.loads(export.read_text())["servers"]
        assert definition["snapshot"]["server"]["name"] == server
        assert definition["snapshot"]["server"]["remote"]["url"] == "https://metadata-only.invalid/mcp"
    finally:
        await gateway.stop()


@pytest.mark.parametrize("mcp_boundary", ["gateway"], indirect=True)
async def test_public_unload_validates_remaining_servers_before_handles_or_persistence(mcp_boundary, monkeypatch):
    from uuid import uuid4
    from tests.integration.mcp_permission_peers import create_owned_mcp_catalog

    boundary = mcp_boundary
    source = Path(os.environ["DOCKER_CONFIG"])
    original_reference = os.environ["KESTREL_MCP_DOCKER_CATALOG"]
    catalog = json.loads((source / "mcp/catalogs/docker-mcp.yaml").read_text())
    second_server = "permission-" + uuid4().hex
    catalog["registry"][second_server] = dict(catalog["registry"][boundary.peer.selector], name=second_server)
    reference = "kestrel-permission/two-servers:" + uuid4().hex
    await create_owned_mcp_catalog(source / "mcp/catalogs/two-native-servers.yaml", catalog, reference)
    operator_configuration = source / "config.json"
    original_bytes = operator_configuration.read_bytes()
    configuration = json.loads(original_bytes)
    configuration["features"]["tool-name-prefix"] = "enabled"
    operator_configuration.write_text(json.dumps(configuration))
    with monkeypatch.context() as override:
        override.setenv("KESTREL_MCP_DOCKER_CATALOG", reference)
        try:
            loaded = await boundary.feature.gateway_start(boundary.peer.selector + "," + second_server)
            assert loaded.status.value == "ok", loaded.to_dict()
        finally:
            operator_configuration.write_bytes(original_bytes)
        manager = boundary.feature.gateway_manager
        boundary.peer.remember_gateway(manager.gateway)
        handles = dict(boundary.agent._tool_to_feature)
        saved = await boundary.agent.get_enablement_deltas("mcp_server")
        assert len([name for name, owner in handles.items() if owner == boundary.owner]) == 2
        override.setenv("KESTREL_MCP_DOCKER_CATALOG", original_reference)
        result = await boundary.feature.unload_tool(boundary.peer.selector)
        assert result.status.value == "error", result.to_dict()
        assert boundary.feature.gateway_manager is manager and manager.is_connected
        assert boundary.agent._tool_to_feature == handles, "Rejected unload withdrew the original callable handles"
        assert await boundary.agent.get_enablement_deltas("mcp_server") == saved
        assert manager.gateway.requested_servers == {boundary.peer.selector, second_server}
        assert await boundary.witness() == []


@pytest.mark.parametrize('field', ['command', 'env'])
async def test_native_source_settings_cannot_move_to_a_changed_execution_snapshot(tmp_path, monkeypatch, field):
    from kestrel_feature_mcp.gateway import DockerMCPGateway, DockerMCPGatewayError
    from tests.integration.mcp_permission_peers import gateway_peer, create_owned_mcp_catalog

    async with gateway_peer(tmp_path, monkeypatch, configured=True) as peer:
        gateway = DockerMCPGateway()
        source_profile = await gateway._read_source_profile()
        catalog_file = Path(os.environ['DOCKER_CONFIG']) / 'mcp/catalogs/docker-mcp.yaml'
        catalog = json.loads(catalog_file.read_text())
        definition = catalog['registry'][peer.selector]
        if field == 'command':
            definition[field].append('changed-owned-executable-target')
        else:
            definition[field].append({'name': 'CHANGED_OWNED_EXECUTION', 'value': 'different'})
        reference = 'kestrel-permission/changed-snapshot:' + peer.owner
        await create_owned_mcp_catalog(catalog_file, catalog, reference)
        try:
            gateway._prepare_gateway_configuration()
            with pytest.raises(DockerMCPGatewayError, match='different execution identity'):
                await gateway._create_profile(reference, [peer.selector], source_profile=source_profile)
            assert gateway.process is None
            assert await peer.witness(None, None) == []
        finally:
            peer.remember_gateway(gateway)
            await gateway.stop()


async def test_native_startup_timeout_survives_independent_retirement_failure(tmp_path, monkeypatch):
    import kestrel_feature_mcp.gateway as gateway_module
    from tests.integration.mcp_permission_peers import gateway_peer

    async with gateway_peer(tmp_path, monkeypatch) as peer:
        gateway = gateway_module.DockerMCPGateway()
        monkeypatch.setattr(gateway_module, 'GATEWAY_STARTUP_TIMEOUT', 0)
        original_host = os.environ['DOCKER_HOST']
        missing_host = 'unix://' + str(Path(os.environ['DOCKER_CONFIG']) / 'missing-owned-daemon.sock')
        async def fail_inventory_after_original_process_allocation():
            while gateway.process is None:
                await asyncio.sleep(0)
            gateway._process_env['DOCKER_HOST'] = missing_host

        fault = asyncio.create_task(fail_inventory_after_original_process_allocation())
        try:
            with pytest.raises(ExceptionGroup) as caught:
                await gateway.start([peer.selector])
            await fault
            assert caught.value.subgroup(lambda error: isinstance(error, gateway_module.DockerMCPGatewayError)
                and 'Gateway startup timeout' in str(error)) is not None, 'Original native startup timeout was erased'
            assert caught.value.subgroup(lambda error: 'container inventory operation failed' in str(error)) is not None
            assert gateway.process is None and gateway._profile_id is not None
            assert gateway._config_directory.exists()
        finally:
            if not fault.done():
                fault.cancel()
                try:
                    await fault
                except asyncio.CancelledError:
                    pass
            else:
                await fault
            peer.remember_gateway(gateway)
            if gateway._process_env is not None:
                gateway._process_env['DOCKER_HOST'] = original_host
            await gateway.stop()


async def test_gateway_fixture_diagnostics_cannot_skip_original_process_retirement(tmp_path, monkeypatch):
    from kestrel_feature_mcp.gateway import DockerMCPGateway
    from tests.integration.mcp_permission_peers import gateway_peer

    gateway = DockerMCPGateway()
    process = None
    private = None
    try:
        with pytest.raises(AssertionError):
            async with gateway_peer(tmp_path, monkeypatch) as peer:
                peer.remember_gateway(gateway)
                await gateway.start([peer.selector])
                peer.remember_gateway(gateway)
                process = gateway.process
                private = gateway._config_directory
                # Only this fixture's private source configuration is changed.
                (Path(os.environ["DOCKER_CONFIG"]) / "config.json").write_text("{}")
        assert process is not None
        assert process.poll() is not None, "fixture diagnostic failure skipped original Gateway retirement"
        assert gateway.process is None and gateway._profile_id is None
        assert not private.exists()
    finally:
        await gateway.stop()


async def test_owned_configured_body_failure_is_not_an_expected_cleanup_failure(tmp_path, monkeypatch):
    from kestrel_feature_mcp.connection import MCPConnection

    original_call = MCPConnection.call_tool
    body_failure = OSError("owned real tool-call epilogue")

    async def call_then_fail(connection, *args, **kwargs):
        await original_call(connection, *args, **kwargs)
        raise body_failure

    monkeypatch.setattr(MCPConnection, "call_tool", call_then_fail)
    with pytest.raises(OSError) as caught:
        await test_native_configured_profile_retains_values_and_template_namespace(
            tmp_path, monkeypatch, OSError("owned expected close epilogue"),
        )
    assert caught.value is body_failure
