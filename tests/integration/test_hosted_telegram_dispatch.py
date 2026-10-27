"""Real SDK wire and SQLite ownership proof, not live provider acceptance."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from kestrel_sdk.isolated_feature import (
    IsolatedFeatureClient,
    IsolatedFeatureService,
    SubprocessIsolatedFeatureClient,
    ToolMetadata,
)

from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from tests.shared.original_task_admission import assert_unknown_task_admission
from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.channels.route_ownership import (
    ChannelRouteOwnershipStore,
)
from kestrel_sovereign.features.hosted_telegram_dispatch import (
    HostedTelegramBinding,
    HostedTelegramOutbound,
)
from kestrel_sovereign.features.isolated_runtime import (
    HostedTelegramRouteAttestation,
    ProxyChannelAdapter,
    ProxyFeature,
    set_hosted_telegram_route_attestation_resolver,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_isolated_feature_config_transition import (
    _hosted_lifecycle_agent,
    _QueueReader,
    _QueueWriter,
    _SDKWireClient,
)


class DispatchService(IsolatedFeatureService):
    """Supported external protocol fixture, no mocked native host state."""

    def __init__(self, events):
        super().__init__(name="hosted-dispatch-wire", version="1")
        self.events = events
        self.activation_entered = asyncio.Event()
        self.activation_release = asyncio.Event()
        self.activation_release.set()
        self.activation_response = {"status": "ok", "http_status": 200, "transport": "hosted"}
        self.activated = False
        self.sends = []
        self.send_entered = asyncio.Event()
        self.send_release = asyncio.Event()
        self.send_release.set()
        self.register_host_ingress("telegram-hosted-activate", self.activate)
        self.register_tool(ToolMetadata(
            name="telegram_send", description="Controlled protocol send",
            input_schema={"type": "object", "properties": {
                "to": {"type": "string"}, "message": {"type": "string"}},
                "required": ["to", "message"]},
        ), self.send)

    async def activate(self, payload):
        self.events.append("activate")
        self.activation_entered.set()
        await self.activation_release.wait()
        self.activated = self.activation_response.get("status") == "ok"
        return self.activation_response

    async def send(self, arguments):
        assert self.activated, "Unprepared child reached send"
        self.events.append("send")
        self.sends.append(arguments)
        self.send_entered.set()
        await self.send_release.wait()
        return {"status": "ok", "confirmation": "Controlled send completed",
                "message_id": "777", "to": arguments["to"]}


class WireClient(_SDKWireClient):
    def __init__(self, *args):
        super().__init__(*args)
        self.stop_calls = 0
        self.wire_closed = asyncio.Event()
        self.stop_release = asyncio.Event()
        self.stop_release.set()

    async def stop(self):
        self.stop_calls += 1
        await super().stop()
        self.wire_closed.set()
        await self.stop_release.wait()

    async def call_tool(self, name, arguments, **kwargs):
        return await self._rpc.call_tool(name, arguments, **kwargs)

    @property
    def supports_tool_execution_context(self):
        return self._rpc.supports_tool_execution_context


@asynccontextmanager
async def sdk_host(tmp_path, monkeypatch, factory, *, timeout=30, expected_unknown=False, expected_denied=False):
    backend = SQLiteBackend(str(tmp_path / "route.db"))
    await backend.connect()
    store = ChannelRouteOwnershipStore(backend)
    events, services, tasks, clients = [], [], [], []

    def client_factory(**kwargs):
        host_reader, service_reader = _QueueReader(), _QueueReader()
        service = DispatchService(events)
        services.append(service)
        tasks.append(asyncio.create_task(service.serve(service_reader, _QueueWriter(host_reader))))
        rpc = IsolatedFeatureClient(host_reader, _QueueWriter(service_reader))
        client = WireClient(rpc, dict(kwargs.get("config") or {}))
        clients.append(client)
        return client

    agent = _hosted_lifecycle_agent(tmp_path)
    binding = HostedTelegramBinding("tenant", agent.did, "123", "555", "g" * 43)
    authority = factory(binding, events) if factory is not None else None
    set_hosted_telegram_route_attestation_resolver(agent, lambda _: HostedTelegramRouteAttestation(
        store, "123", outbound_authority=authority, outbound_timeout_seconds=timeout,
    ))
    monkeypatch.setenv("KESTREL_FEATURE_TELEGRAMFEATURE_BIN", "/bin/test-service")
    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature", entry_point="test_pkg.feature:TelegramFeature",
        distribution="kestrel-channel-telegram", runtime="isolated-venv", service="test-service",
    )
    proxy = ProxyFeature(agent, runtime, client_factory=client_factory)
    try:
        await proxy.initialize()
        assert proxy._hosted_telegram_startup_attested
        yield proxy, services, clients, events
    finally:
        for service in services:
            service.activation_release.set()
            service.send_release.set()
        for client in clients:
            client.stop_release.set()
        try:
            try:
                await proxy.shutdown()
            except ExecutionAuthorityError as error:
                from kestrel_sovereign.execution_custody import (
                    ExecutionCommitOutcomeError,
                    execution_commit_outcome,
                )
                if expected_unknown:
                    assert type(error) is ExecutionCommitOutcomeError
                    assert execution_commit_outcome(error) == "unknown"
                else:
                    assert expected_denied and type(error) is ExecutionAuthorityError
                assert all(client.stopped for client in clients)
                assert not proxy._terminal_retirement_clients
        finally:
            await KestrelAgent._shutdown_background_tasks(agent)
            await asyncio.gather(*tasks)
            await backend.close()


def ordinary_authority(binding, events):
    @asynccontextmanager
    async def authority(proxy, arguments):
        events.append("authority-enter")
        selected = HostedTelegramOutbound(binding)
        try:
            yield selected
        finally:
            events.append("authority-exit")
        assert selected.response["message_id"] == "777"
    return authority


@pytest.mark.asyncio
async def test_actual_sdk_first_send_prepares_before_send_and_confirms_before_release(tmp_path, monkeypatch):
    async with sdk_host(tmp_path, monkeypatch, ordinary_authority) as (proxy, services, _, events):
        assert services[0].activated is False
        response = await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
        assert response["status"] == "ok" and response["message_id"] == "777"
        assert events == ["authority-enter", "activate", "send", "authority-exit"]
        assert services[0].sends == [{"to": "555", "message": "controlled"}]


@pytest.mark.asyncio
async def test_missing_host_authority_never_uses_activated_legacy_send(tmp_path, monkeypatch):
    async with sdk_host(tmp_path, monkeypatch, None) as (proxy, services, _, events):
        services[0].activated = True  # Negative control: persistent binding grants no host authority.
        response = await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
        assert response["status"] == "error"
        assert services[0].sends == [] and events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    {"status": "error", "http_status": 200, "transport": "hosted"},
    {"status": "ok", "http_status": True, "transport": "hosted"},
    {"status": "ok", "http_status": 200},
])
async def test_refused_or_malformed_activation_never_calls_tool(tmp_path, monkeypatch, response):
    async with sdk_host(tmp_path, monkeypatch, ordinary_authority) as (proxy, services, _, events):
        services[0].activation_response = response
        result = await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
        assert result["status"] == "error"
        assert services[0].sends == []
        assert events == ["authority-enter", "activate", "authority-exit"]


@pytest.mark.asyncio
async def test_actual_bridge_and_replacement_both_traverse_current_authority(tmp_path, monkeypatch):
    async with sdk_host(tmp_path, monkeypatch, ordinary_authority) as (proxy, services, clients, events):
        adapter = ProxyChannelAdapter(proxy, channel_type="telegram", send_tool="telegram_send")
        first = await adapter.send_message("555", "first")
        assert first.message_id == "777"
        await proxy.reload()
        assert clients[0].stopped and len(services) == 2
        assert services[1].activated is False
        second = await adapter.send_message("555", "second")
        assert second.message_id == "777"
        assert [len(service.sends) for service in services] == [1, 1]
        assert events == ["authority-enter", "activate", "send", "authority-exit"] * 2


@pytest.mark.asyncio
async def test_cancellation_during_activation_drains_original_and_prohibits_send(tmp_path, monkeypatch):
    @asynccontextmanager
    async def suppressed(proxy, arguments):
        try:
            yield HostedTelegramOutbound(HostedTelegramBinding("tenant", proxy.agent.did, "123", "555", "g" * 43))
        except asyncio.CancelledError:
            pass  # Negative control: host context must not manufacture success.

    async with sdk_host(tmp_path, monkeypatch, lambda *_: suppressed) as (proxy, services, clients, _):
        service = services[0]
        service.activation_release.clear()
        task = asyncio.create_task(proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"}))
        reload_task = None
        try:
            await asyncio.wait_for(service.activation_entered.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            reload_task = asyncio.create_task(proxy.reload())
            await asyncio.sleep(0.02)
            assert not task.done() and not reload_task.done() and not clients[0].stopped
            service.activation_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            await asyncio.wait_for(reload_task, 1)
            assert service.sends == []
        finally:
            service.activation_release.set()
            for operation in (task, reload_task):
                if operation is not None:
                    if not operation.done():
                        operation.cancel()
                    await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_confirmed_prior_message_is_not_reactivated_or_resent(tmp_path, monkeypatch):
    def factory(binding, events):
        @asynccontextmanager
        async def authority(proxy, arguments):
            selected = HostedTelegramOutbound(binding, prior_message_id="confirmed-original")
            yield selected
            assert selected.response is None  # No new provider response is invented.
        return authority

    async with sdk_host(tmp_path, monkeypatch, factory) as (proxy, services, _, events):
        response = await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
        assert response["status"] == "ok" and response["message_id"] == "confirmed-original"
        assert services[0].sends == [] and events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [{"chat_id": "different"}, {"agent_id": "did:other"}, {"bot_id": "000123"}, {"bot_id": "124"}])
@pytest.mark.parametrize("prior", [None, "prior-confirmed"])
async def test_binding_must_match_exact_agent_bot_and_destination(tmp_path, monkeypatch, changed, prior):
    def factory(binding, events):
        @asynccontextmanager
        async def authority(proxy, arguments):
            yield HostedTelegramOutbound(replace(binding, **changed), prior_message_id=prior)
        return authority

    async with sdk_host(tmp_path, monkeypatch, factory) as (proxy, services, _, events):
        response = await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
        assert response["status"] == "error"
        assert services[0].sends == [] and events == []


@pytest.mark.asyncio
async def test_public_argument_mutation_cannot_change_authorized_send(tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    def factory(binding, events):
        @asynccontextmanager
        async def authority(proxy, arguments):
            with pytest.raises(TypeError):
                arguments["to"] = "different"
            entered.set()
            await release.wait()
            yield HostedTelegramOutbound(binding)
        return authority

    async with sdk_host(tmp_path, monkeypatch, factory) as (proxy, services, _, _):
        arguments = {"to": "555", "message": "original"}
        operation = asyncio.create_task(proxy.call_isolated_tool("telegram_send", arguments))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            arguments.update(to="different", message="changed")
            release.set()
            assert (await operation)["status"] == "ok"
            assert services[0].sends == [{"to": "555", "message": "original"}]
        finally:
            release.set()
            await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_production_subprocess_wrapper_prepares_each_actual_replacement_child(tmp_path, monkeypatch):
    journal = tmp_path / "dispatch-journal.jsonl"
    service_path = tmp_path / "hosted-dispatch-service"
    fixture = Path(__file__).parents[1] / "fixtures" / "hosted_telegram_dispatch_service.py"
    service_path.write_text(f"#!{sys.executable}\n" + fixture.read_text())
    service_path.chmod(0o700)
    monkeypatch.setenv("KESTREL_FEATURE_TELEGRAMFEATURE_BIN", str(service_path))
    monkeypatch.setenv("HOSTED_DISPATCH_JOURNAL", str(journal))
    backend = SQLiteBackend(str(tmp_path / "process-route.db"))
    await backend.connect()
    agent = _hosted_lifecycle_agent(tmp_path)
    binding = HostedTelegramBinding("tenant", agent.did, "123", "555", "g" * 43)
    confirmed = []

    @asynccontextmanager
    async def authority(proxy, arguments):
        selected = HostedTelegramOutbound(binding)
        yield selected
        confirmed.append(selected.response["message_id"])

    set_hosted_telegram_route_attestation_resolver(agent, lambda _: HostedTelegramRouteAttestation(
        ChannelRouteOwnershipStore(backend), "123", outbound_authority=authority,
    ))
    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature", entry_point="test_pkg.feature:TelegramFeature",
        distribution="kestrel-channel-telegram", runtime="isolated-venv", service="test-service",
    )
    proxy = ProxyFeature(agent, runtime)
    processes = []
    try:
        await proxy.initialize()
        for index in range(2):
            assert type(proxy._client) is SubprocessIsolatedFeatureClient
            process = proxy._client.process
            assert process is not None and process.returncode is None
            processes.append(process)
            response = await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": f"message-{index}"})
            assert response["status"] == "ok" and response["message_id"] == str(process.pid)
            if index == 0:
                await proxy.reload()
                assert process.returncode is not None, "Original child must be physically reaped"
        await proxy.shutdown()
        assert all(process.returncode is not None for process in processes)
        assert len({process.pid for process in processes}) == 2
        entries = [json.loads(line) for line in journal.read_text().splitlines()]
        assert [entry["kind"] for entry in entries] == ["started", "activate", "send"] * 2
        assert confirmed == [str(process.pid) for process in processes]
        for offset, process in zip((0, 3), processes):
            assert all(entry["pid"] == process.pid for entry in entries[offset:offset + 3])
            assert entries[offset + 1]["binding"] == {
                "scope": {"tenant_id": "tenant", "agent_id": agent.did},
                "bot_id": "123", "chat_id": "555", "activation_generation": "g" * 43,
            }
    finally:
        try:
            await proxy.shutdown()
        finally:
            await KestrelAgent._shutdown_background_tasks(agent)
            for process in processes:
                if process.returncode is None:
                    process.kill()
                await process.wait()
            await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["construct", "enter", "exit"])
async def test_host_callback_errors_do_not_expose_secret_text(tmp_path, monkeypatch, caplog, phase):
    secret = "credential-shaped-host-secret-DO-NOT-EXPOSE"

    def factory(binding, events):
        @asynccontextmanager
        async def context(proxy, arguments):
            if phase == "enter":
                raise RuntimeError(secret)
            yield HostedTelegramOutbound(binding)
            if phase == "exit":
                raise RuntimeError(secret)

        def authority(proxy, arguments):
            if phase == "construct":
                raise RuntimeError(secret)
            return context(proxy, arguments)
        return authority

    async with sdk_host(tmp_path, monkeypatch, factory, expected_unknown=phase == "exit") as (proxy, services, _, _):
        from kestrel_sovereign.execution_custody import (
            ExecutionCommitOutcomeError,
            execution_commit_outcome,
        )

        if phase == "exit":
            with pytest.raises(ExecutionCommitOutcomeError) as unknown:
                await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
            assert execution_commit_outcome(unknown.value) == "unknown"
            assert secret not in str(unknown.value) + repr(unknown.value)
            assert len(services[0].sends) == 1
        else:
            response = await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
            assert response["status"] == "error" and secret not in repr(response)
            assert services[0].sends == []
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_authority_does_not_exit_on_stream_failure_before_original_stop_completes(tmp_path, monkeypatch):
    exited = asyncio.Event()

    def factory(binding, events):
        @asynccontextmanager
        async def authority(proxy, arguments):
            try:
                yield HostedTelegramOutbound(binding)
            finally:
                exited.set()
        return authority

    async with sdk_host(tmp_path, monkeypatch, factory) as (proxy, services, clients, _):
        service, client = services[0], clients[0]
        service.activation_release.clear()
        client.stop_release.clear()
        operation = asyncio.create_task(proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"}))
        shutdown = None
        try:
            await asyncio.wait_for(service.activation_entered.wait(), 1)
            shutdown = asyncio.create_task(proxy.shutdown())
            await asyncio.wait_for(client.wire_closed.wait(), 1)
            await asyncio.sleep(0.02)
            assert not exited.is_set(), "Host authority released before original stop completion"
            assert not operation.done() and not shutdown.done()
            assert client.stop_calls == 1
            client.stop_release.set()
            # No provider RPC was admitted; the original activation failed.
            response = await asyncio.wait_for(operation, 2)
            assert response["status"] == "error"
            await asyncio.wait_for(shutdown, 2)
            assert exited.is_set() and client.stop_calls == 1
        finally:
            service.activation_release.set()
            client.stop_release.set()
            for task in (operation, shutdown):
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["activation", "provider"])
async def test_deadline_is_bounded_but_original_authority_survives_until_physical_stop(tmp_path, monkeypatch, phase):
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError,
        ExecutionCommitOutcomeError,
        execution_commit_outcome,
    )

    exited = asyncio.Event()

    def factory(binding, events):
        @asynccontextmanager
        async def authority(proxy, arguments):
            try:
                yield HostedTelegramOutbound(binding)
            finally:
                exited.set()
        return authority

    async with sdk_host(tmp_path, monkeypatch, factory, timeout=0.1,
                        expected_unknown=phase == "provider", expected_denied=phase == "activation") as (proxy, services, clients, _):
        service, client = services[0], clients[0]
        release = service.activation_release if phase == "activation" else service.send_release
        entered = service.activation_entered if phase == "activation" else service.send_entered
        release.clear()
        client.stop_release.clear()
        operation = asyncio.create_task(proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"}))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            expected = ExecutionAuthorityError if phase == "activation" else ExecutionCommitOutcomeError
            with pytest.raises(expected) as denied:
                await asyncio.wait_for(asyncio.shield(operation), 2)
            assert execution_commit_outcome(denied.value) == (None if phase == "activation" else "unknown")
            await asyncio.wait_for(client.wire_closed.wait(), 1)
            assert not exited.is_set(), "Deadline released authority before original physical stop"
            assert client.stop_calls == 1
            assert any(not admission.admission.task.done() for admission in proxy._host_ingress_admissions)
            client.stop_release.set()
            await asyncio.wait_for(exited.wait(), 2)
            assert len(service.sends) == (0 if phase == "activation" else 1)
            # A deadline seals this facade. No automatic second provider RPC.
            with pytest.raises(expected):
                await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "do not replay"})
            assert len(service.sends) == (0 if phase == "activation" else 1)
        finally:
            release.set()
            client.stop_release.set()
            await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_wait_callback_refusal_cannot_abandon_original_dispatch(tmp_path, monkeypatch):
    class RefuseWaitCallback(asyncio.Task):
        refused = False

        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "_on_completion" and not self.refused:
                self.refused = True
                raise RuntimeError("controlled wait observation refusal")
            return super().add_done_callback(callback, context=context)

    async with sdk_host(tmp_path, monkeypatch, ordinary_authority, expected_denied=True) as (proxy, services, _, _):
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()
        observed = []

        def factory(loop, coroutine, **kwargs):
            if not observed and coroutine.cr_code.co_qualname == "create_owned_task.<locals>.run":
                task = RefuseWaitCallback(coroutine, loop=loop, **kwargs)
                observed.append(task)
                return task
            return (asyncio.Task(coroutine, loop=loop, **kwargs) if previous is None
                    else previous(loop, coroutine, **kwargs))

        loop.set_task_factory(factory)
        try:
            with pytest.raises(ExecutionAuthorityError):
                await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
            assert len(observed) == 1 and observed[0].refused
            assert services[0].sends == []
            assert observed[0].done() or any(admission.admission.task is observed[0] for admission in proxy._host_ingress_admissions)
        finally:
            loop.set_task_factory(previous)
            await asyncio.gather(*observed, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_after_provider_start_permanently_retains_unknown(tmp_path, monkeypatch):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    async with sdk_host(tmp_path, monkeypatch, ordinary_authority, expected_unknown=True) as (proxy, services, _, _):
        service = services[0]
        service.send_release.clear()
        operation = asyncio.create_task(proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"}))
        try:
            await asyncio.wait_for(service.send_entered.wait(), 1)
            operation.cancel()
            await asyncio.sleep(0)
            service.send_release.set()
            with pytest.raises(ExecutionCommitOutcomeError, match="unknown"):
                await operation
            with pytest.raises(ExecutionCommitOutcomeError, match="unknown"):
                await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "no replay"})
            assert len(service.sends) == 1
        finally:
            service.send_release.set()
            await asyncio.gather(operation, return_exceptions=True)


def scheduled_scope():
    from kestrel_sovereign.features.scheduler.runner import (
        SchedulerExecution,
        _SchedulerExecutionScope,
    )

    return _SchedulerExecutionScope(SchedulerExecution(
        id="execution-1", schedule_id="schedule-1", agent_id="did:test:isolated-integration",
        task_name="telegram_send", args={}, scheduled_for="2026-10-21T15:00:00+00:00",
        idempotency_key="controlled-stable-effect", attempt=1, owner="controlled-owner",
    ))


@pytest.mark.asyncio
async def test_scheduled_context_translation_refusal_stays_scheduler_failure(tmp_path, monkeypatch):
    from kestrel_sovereign.features import isolated_runtime
    from kestrel_sovereign.features.scheduler.runner import (
        bind_scheduler_execution_scope,
    )

    def refused():
        raise isolated_runtime.SchedulerExecutionContextUnavailable("controlled missing SDK context")

    async with sdk_host(tmp_path, monkeypatch, ordinary_authority) as (proxy, services, _, _):
        monkeypatch.setattr(isolated_runtime, "_scheduled_tool_execution_context", refused)
        with bind_scheduler_execution_scope(scheduled_scope()), pytest.raises(isolated_runtime.SchedulerExecutionContextUnavailable):
            await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
        assert services[0].sends == []


@pytest.mark.asyncio
async def test_scheduler_revocation_during_activation_prevents_actual_provider_dispatch(tmp_path, monkeypatch):
    from kestrel_sovereign.features.isolated_runtime import (
        SchedulerExecutionContextUnavailable,
    )
    from kestrel_sovereign.features.scheduler.runner import (
        bind_scheduler_execution_scope,
    )

    async with sdk_host(tmp_path, monkeypatch, ordinary_authority) as (proxy, services, _, _):
        service, scope = services[0], scheduled_scope()
        service.activation_release.clear()
        with bind_scheduler_execution_scope(scope):
            operation = asyncio.create_task(proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"}))
        try:
            await asyncio.wait_for(service.activation_entered.wait(), 1)
            scope.revoke()
            service.activation_release.set()
            with pytest.raises(SchedulerExecutionContextUnavailable):
                await operation
            assert service.sends == []
        finally:
            service.activation_release.set()
            await asyncio.gather(operation, return_exceptions=True)


def test_host_admission_annotations_resolve_on_eager_annotation_python():
    from typing import Any, get_type_hints

    from kestrel_sovereign.features.channels.feature import ChannelFeature

    # Python3.14 postpones annotation evaluation. Explicit resolution catches
    # the same missing import that prevents supported older Python importing.
    for method in (ChannelFeature.handle_inbound, ChannelFeature.handle_terminal_inbound):
        assert get_type_hints(method)["admission_fence"] is Any


@pytest.mark.asyncio
@pytest.mark.parametrize("hide_identity", [False, True])
async def test_string_subclass_cannot_skip_hosted_authority(tmp_path, monkeypatch, hide_identity):
    class Name(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            return False if hide_identity else str.__eq__(self, other)

        def __str__(self):
            return "not-the-wire-name" if hide_identity else str.__str__(self)

    async with sdk_host(tmp_path, monkeypatch, None) as (proxy, services, _, _):
        services[0].activated = True
        response = await proxy.call_isolated_tool(Name("telegram_send"), {"to": "555", "message": "controlled"})
        assert response["status"] == "error", "String subclass bypassed hosted authority"
        assert services[0].sends == []


@pytest.mark.asyncio
async def test_authorized_string_subclass_uses_canonical_dispatch(tmp_path, monkeypatch):
    class Name(str):
        pass

    async with sdk_host(tmp_path, monkeypatch, ordinary_authority) as (proxy, services, _, events):
        response = await proxy.call_isolated_tool(Name("telegram_send"), {"to": "555", "message": "controlled"})
        assert response["status"] == "ok"
        assert len(services[0].sends) == 1
        assert events == ["authority-enter", "activate", "send", "authority-exit"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["tool", "bridge"])
async def test_revoked_entry_does_not_export_hosted_arguments_or_poison_proxy(tmp_path, monkeypatch, path):
    from kestrel_sovereign.execution_custody import bind_execution_custody

    secret = "controlled-sensitive-message-entry-DO-NOT-EXPORT"
    arguments = {"to": "555", "message": secret}

    class Fence:
        backend_type = "postgres"

        def require_work(self):
            pass

        async def lock_and_validate(self, connection):
            pass

    async with sdk_host(tmp_path, monkeypatch, ordinary_authority) as (proxy, services, _, _):
        with bind_execution_custody(Fence()) as scope:
            scope.revoke("original caller retired before hosted admission")
            with pytest.raises(ExecutionAuthorityError) as denied:
                if path == "tool":
                    await proxy.call_isolated_tool("telegram_send", arguments)
                else:
                    await ProxyChannelAdapter(proxy, channel_type="telegram", send_tool="telegram_send").send_message("555", secret)
        pending, seen = [denied.value], set()
        while pending:
            error = pending.pop()
            if id(error) in seen:
                continue
            seen.add(id(error))
            pending.extend(item for item in (error.__cause__, error.__context__) if item is not None)
            pending.extend(getattr(error, "exceptions", ()))
            trace = error.__traceback__
            while trace is not None:
                frame = trace.tb_frame
                if "/kestrel_sovereign/" in frame.f_code.co_filename:
                    values = tuple(frame.f_locals.values())
                    assert all(value is not arguments and value is not proxy and not (type(value) is str and value == secret) for value in values), frame.f_code.co_name
                    assert all(not (type(value) is dict and value.get("message") == secret) for value in values), frame.f_code.co_name
                trace = trace.tb_next
        assert services[0].sends == [] and proxy._host_ingress_public_control is None
        assert not proxy._host_ingress_admissions
        response = await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "new valid caller"})
        assert response["status"] == "ok" and len(services[0].sends) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["allocation", "publication"])
async def test_scheduled_task_refusal_is_not_success_shaped_result(tmp_path, monkeypatch, phase):
    from kestrel_sovereign.execution_custody import (
        _explicit_control_chain, execution_commit_outcome,
    )
    from kestrel_sovereign.features.isolated_runtime import (
        SchedulerExecutionContextUnavailable,
    )
    from kestrel_sovereign.features.scheduler.runner import (
        bind_scheduler_execution_scope,
    )

    refusal = RuntimeError("controlled original task " + phase + " refusal")

    class RefusePublication(asyncio.Task):
        refused = False

        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "close_unstarted" and not self.refused:
                self.refused = True
                raise refusal
            return super().add_done_callback(callback, context=context)

    retained = None
    async with sdk_host(
        tmp_path, monkeypatch, ordinary_authority, expected_denied=phase == "allocation",
    ) as (proxy, services, _, _):
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()
        observed, refused, attempted = [], [], False

        def factory(loop, coroutine, **kwargs):
            nonlocal attempted
            if not attempted and coroutine.cr_code.co_qualname == "create_owned_task.<locals>.run":
                attempted = True
                if phase == "allocation":
                    refused.append(coroutine)
                    raise refusal
                task = RefusePublication(coroutine, loop=loop, **kwargs)
                observed.append(task)
                return task
            return (asyncio.Task(coroutine, loop=loop, **kwargs) if previous is None
                    else previous(loop, coroutine, **kwargs))

        loop.set_task_factory(factory)
        try:
            expected = ExecutionAuthorityError if phase == "allocation" else SchedulerExecutionContextUnavailable
            with bind_scheduler_execution_scope(scheduled_scope()), pytest.raises(expected) as caught:
                await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
            assert type(caught.value) is expected
            if phase == "allocation":
                import inspect
                assert execution_commit_outcome(caught.value) is None
                assert refusal not in tuple(_explicit_control_chain(caught.value)), "public frames leaked private ingress refusal"
                assert len(proxy._host_ingress_admissions) == 1
                retained = next(iter(proxy._host_ingress_admissions))
                assert_unknown_task_admission(retained.opening, refusal, admission=retained.admission)
                assert_unknown_task_admission(retained.native_error, refusal, admission=retained.admission)
                assert not retained.provider_rpc_started and not retained.scopes
                assert proxy._host_ingress_public_control.commit_outcome is None
                assert refused and all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused)
            assert attempted
            assert services[0].sends == []
            assert all(task.done() for task in observed)
        finally:
            loop.set_task_factory(previous)
            await asyncio.gather(*observed, return_exceptions=True)
    if retained is not None:
        assert retained in proxy._host_ingress_admissions
        assert_unknown_task_admission(retained.native_error, refusal, admission=retained.admission)


@pytest.mark.asyncio
async def test_same_callback_withdrawal_prevents_next_provider_rpc(tmp_path, monkeypatch):
    async with sdk_host(tmp_path, monkeypatch, ordinary_authority) as (proxy, services, clients, _):
        client, ready = clients[0], asyncio.Event()
        release = asyncio.get_running_loop().create_future()
        original = client.call_host_ingress

        async def observed(name, payload=None):
            response = await original(name, payload)
            ready.set()
            await release
            return response

        monkeypatch.setattr(client, "call_host_ingress", observed)
        operation = asyncio.create_task(proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"}))
        try:
            await asyncio.wait_for(ready.wait(), 2)

            def withdraw():
                # Queue the dispatch continuation BEFORE the public waiter's
                # cancellation continuation, in the same actual loop callback.
                release.set_result(None)
                operation.cancel()

            asyncio.get_running_loop().call_soon(withdraw)
            with pytest.raises(asyncio.CancelledError):
                await operation
            assert services[0].sends == []
        finally:
            if not release.done():
                release.set_result(None)
            await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_previously_handled_cancellation_does_not_withdraw_new_dispatch(tmp_path, monkeypatch):
    async with sdk_host(tmp_path, monkeypatch, ordinary_authority) as (proxy, services, _, _):
        async def call():
            task = asyncio.current_task()
            task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass  # Retain its old count deliberately; no new withdrawal.
            assert task.cancelling() == 1
            return await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})

        response = await asyncio.create_task(call())
        assert response["status"] == "ok"
        assert len(services[0].sends) == 1


@pytest.mark.asyncio
async def test_deadline_publication_refusal_retires_original_and_redacts(tmp_path, monkeypatch):
    async with sdk_host(tmp_path, monkeypatch, ordinary_authority, expected_denied=True) as (proxy, services, _, _):
        loop, attempted = asyncio.get_running_loop(), False
        original = loop.call_later

        def refused(delay, callback, *args, **kwargs):
            nonlocal attempted
            if not attempted and 0 < delay <= 30 and getattr(callback, "__name__", "") == "set_result":
                attempted = True
                raise RuntimeError("controlled-deadline-publication-secret")
            return original(delay, callback, *args, **kwargs)

        monkeypatch.setattr(loop, "call_later", refused)
        with pytest.raises(ExecutionAuthorityError) as failure:
            await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
        assert attempted and "controlled-deadline-publication-secret" not in str(failure.value)
        assert proxy._terminal_lifecycle_latched
        assert services[0].sends == []


@pytest.mark.asyncio
@pytest.mark.parametrize("withdraw", [False, True])
async def test_custody_denied_while_gate_waits_remains_native_control(tmp_path, monkeypatch, withdraw):
    from kestrel_sovereign.execution_custody import (
        ConfirmedLeaseDeadline,
        bind_execution_custody,
    )

    async with sdk_host(tmp_path, monkeypatch, ordinary_authority, expected_denied=True) as (proxy, services, _, _):
        gate, entered = proxy._traffic_gate, asyncio.Event()
        original = gate._condition.wait

        async def waiting():
            entered.set()
            return await original()

        monkeypatch.setattr(gate._condition, "wait", waiting)
        await gate.close()
        deadline = ConfirmedLeaseDeadline()
        assert deadline.confirm(asyncio.get_running_loop().time() + 30)

        class Fence:
            backend_type = "postgres"

            def require_work(self):
                deadline.require_work()

            async def lock_and_validate(self, connection):
                deadline.require_work()

        operation = None
        try:
            with bind_execution_custody(Fence()) as scope:
                operation = asyncio.create_task(proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"}))
                await asyncio.wait_for(entered.wait(), 2)
                scope.revoke("controlled original admission retired")
                if withdraw:
                    operation.cancel()
                await gate.reopen()
                with pytest.raises(ExecutionAuthorityError):
                    await operation
            assert services[0].sends == []
        finally:
            await gate.reopen()
            if operation is not None:
                await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_shaped", [False, True])
async def test_foreign_control_shaped_callback_cannot_forge_native_provenance(tmp_path, monkeypatch, commit_shaped):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    def factory(binding, events):
        def refused(proxy, arguments):
            if commit_shaped:
                raise ExecutionCommitOutcomeError("unknown")
            raise ExecutionAuthorityError("foreign-controlled-denial-text")
        return refused

    async with sdk_host(tmp_path, monkeypatch, factory) as (proxy, services, _, _):
        response = await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
        assert response["status"] == "error"
        assert "foreign-controlled-denial-text" not in repr(response)
        assert proxy._host_ingress_public_control is None
        assert services[0].sends == []


@pytest.mark.asyncio
@pytest.mark.parametrize("allocation", ["completion", "shield"])
async def test_retirement_observer_allocation_cannot_release_authority_before_stop(tmp_path, monkeypatch, allocation):
    from contextlib import asynccontextmanager
    from kestrel_sovereign.features.hosted_telegram_dispatch import HostedTelegramOutbound

    exited, refused, target = asyncio.Event(), False, None

    def factory(binding, events):
        @asynccontextmanager
        async def authority(proxy, arguments):
            try:
                yield HostedTelegramOutbound(binding)
            finally:
                exited.set()
        return authority

    async with sdk_host(tmp_path, monkeypatch, factory) as (proxy, services, clients, _):
        loop, client, service = asyncio.get_running_loop(), clients[0], services[0]
        original = loop.create_future

        def allocate():
            nonlocal refused
            frame = sys._getframe(1)
            if allocation == "shield" and frame.f_code.co_name == "shield":
                frame = frame.f_back
            if (not refused and frame.f_code.co_name == 'await_owned_task'
                    and frame.f_locals.get('task') is target and target is not None
                    and (allocation == "completion" or sys._getframe(1).f_code.co_name == "shield")):
                refused = True
                raise RuntimeError('controlled original retirement observation refusal')
            return original()

        monkeypatch.setattr(loop, 'create_future', allocate)
        service.activation_release.clear()
        client.stop_release.clear()
        operation = asyncio.create_task(proxy.call_isolated_tool('telegram_send', {'to': '555', 'message': 'controlled'}))
        shutdown = None
        try:
            await asyncio.wait_for(service.activation_entered.wait(), 2)
            target, = [admission.physical_client_retired for admission in proxy._host_ingress_admissions
                       if admission.physical_client_retired is not None]
            shutdown = asyncio.create_task(proxy.shutdown())
            await asyncio.wait_for(client.wire_closed.wait(), 2)
            await asyncio.sleep(0.02)
            assert refused
            assert not exited.is_set(), 'Observer allocation released original host authority before stop'
            client.stop_release.set()
            await asyncio.wait_for(operation, 2)
            await asyncio.wait_for(shutdown, 2)
            assert exited.is_set()
        finally:
            service.activation_release.set()
            client.stop_release.set()
            await asyncio.gather(*(task for task in (operation, shutdown) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_real_deadline_precedes_provider_even_if_activation_continuation_runs_first(tmp_path, monkeypatch):
    async with sdk_host(tmp_path, monkeypatch, ordinary_authority, timeout=1, expected_denied=True) as (proxy, services, clients, _):
        loop, client, ready = asyncio.get_running_loop(), clients[0], asyncio.Event()
        release, attempts = loop.create_future(), []
        ingress, send, later = client.call_host_ingress, client.call_tool, loop.call_later
        installed = False

        async def observed(name, payload=None):
            response = await ingress(name, payload)
            ready.set()
            await release
            return response

        async def sending(*args, **kwargs):
            attempts.append('actual-sdk-call')
            return await send(*args, **kwargs)

        def install(delay, callback, *args, **kwargs):
            nonlocal installed
            if not installed and 0 < delay <= 1 and getattr(callback, '__name__', '') == 'set_result':
                installed = True

                def expire():
                    assert ready.is_set(), 'Activation response must already be held at the real deadline'
                    release.set_result(None)
                    callback(*args)

                return later(delay, expire, **kwargs)
            return later(delay, callback, *args, **kwargs)

        monkeypatch.setattr(client, 'call_host_ingress', observed)
        monkeypatch.setattr(client, 'call_tool', sending)
        monkeypatch.setattr(loop, 'call_later', install)
        operation = asyncio.create_task(proxy.call_isolated_tool('telegram_send', {'to': '555', 'message': 'controlled'}))
        try:
            with pytest.raises(ExecutionAuthorityError) as denied:
                await operation
            assert installed and type(denied.value) is ExecutionAuthorityError
            assert attempts == []
            assert services[0].sends == []
        finally:
            if not release.done():
                release.set_result(None)
            await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation_state", [None, "committed", "unknown"])
@pytest.mark.parametrize("denial_phase", ["confirmation", "after-confirmation"])
async def test_confirmation_custody_denial_cannot_erase_unknown_send(tmp_path, monkeypatch, confirmation_state, denial_phase):
    from kestrel_sovereign.execution_custody import (
        ConfirmedLeaseDeadline, ExecutionCommitOutcomeError,
        bind_execution_custody, require_execution_work,
    )

    confirmed, exit_release = asyncio.Event(), asyncio.Event()

    def factory(binding, events):
        @asynccontextmanager
        async def authority(proxy, arguments):
            selected = HostedTelegramOutbound(binding)
            yield selected
            assert selected.response["message_id"] == "777"
            if denial_phase == "confirmation":
                require_execution_work(proxy.agent)
            else:
                confirmed.set()
                await exit_release.wait()
        return authority

    async with sdk_host(tmp_path, monkeypatch, factory, expected_unknown=True) as (proxy, services, _, _):
        deadline = ConfirmedLeaseDeadline()
        assert deadline.confirm(asyncio.get_running_loop().time() + 30)

        class Fence:
            backend_type = "postgres"

            def require_work(self):
                deadline.require_work()

            async def lock_and_validate(self, connection):
                deadline.require_work()

        service = services[0]
        service.send_release.clear()
        operation = None
        try:
            with bind_execution_custody(Fence()) as scope:
                operation = asyncio.create_task(proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"}))
                await asyncio.wait_for(service.send_entered.wait(), 2)
                if denial_phase == "after-confirmation":
                    service.send_release.set()
                    await asyncio.wait_for(confirmed.wait(), 2)
                if confirmation_state is not None:
                    scope.preserve_commit_uncertainty(confirmation_state)
                scope.revoke("original confirmation authority retired")
                service.send_release.set()
                exit_release.set()
                with pytest.raises(ExecutionCommitOutcomeError, match="unknown"):
                    await operation
            with pytest.raises(ExecutionCommitOutcomeError, match="unknown"):
                await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "no replay"})
            assert len(service.sends) == 1
            assert proxy._host_ingress_public_control.commit_outcome == "unknown"
            assert any(item.core_outcome.commit_outcome == "unknown" for item in proxy._host_ingress_admissions)
        finally:
            service.send_release.set()
            exit_release.set()
            if operation is not None:
                await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_post_send_tool_envelope_failure_cannot_authorize_replay(tmp_path, monkeypatch):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    async with sdk_host(tmp_path, monkeypatch, ordinary_authority, expected_unknown=True) as (proxy, services, _, _):
        def refuse_envelope(_name):
            raise RuntimeError("controlled post-send envelope failure")

        monkeypatch.setattr(proxy, "_maybe_emit_channel_link_part", refuse_envelope)
        with pytest.raises(ExecutionCommitOutcomeError, match="unknown"):
            await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "controlled"})
        with pytest.raises(ExecutionCommitOutcomeError, match="unknown"):
            await proxy.call_isolated_tool("telegram_send", {"to": "555", "message": "no replay"})
        assert len(services[0].sends) == 1
