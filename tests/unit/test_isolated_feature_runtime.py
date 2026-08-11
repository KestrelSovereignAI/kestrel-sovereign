"""Tests for isolated feature runtime proxy behavior."""

import asyncio
import gc
import json
import os
import threading
import types
import weakref
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kestrel_sdk.isolated_feature import (
    MAX_HOST_INGRESS_PAYLOAD_BYTES,
    ConfigTransitionError,
    ConfigTransitionResult,
    HostIngressCapabilities,
    HostIngressError,
    HostIngressUnknownNameError,
    HostIngressUnsupportedError,
)
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features import isolated_runtime
from kestrel_sovereign.features.isolated_runtime import (
    ProxyFeature,
    SchedulerExecutionContextUnavailable,
    SchedulerTerminalAdmissionError,
)
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    ScheduledTask,
    SchedulerExecution,
    SchedulerRunner,
    _current_execution,
    _SchedulerExecutionScope,
    get_current_scheduler_execution,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

_TEST_AGENT_DID = "did:test:isolated-runtime"
_TEST_CONFIG_NODE_ID = f"feature_config:v2:{_TEST_AGENT_DID}:TestFeature"


class FakeIsolatedClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.event_handler = None
        self.calls = []

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def health(self):
        return True

    async def list_tools(self):
        # Real wire contract: services advertise a JSON-Schema ``input_schema``
        # (kestrel_sdk.isolated_feature.protocol.ToolMetadata), NOT a bare
        # ``parameters`` dict. The host must convert it into ToolParameters.
        return [
            {
                "name": "ping",
                "description": "Ping the isolated service",
                "category": "utility",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "text to echo",
                        }
                    },
                    "required": ["message"],
                },
                "command_prefix": "!ping",
            }
        ]

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return {"echo": args}

    def on_event(self, handler):
        self.event_handler = handler


@pytest.mark.asyncio
async def test_proxy_feature_mirrors_tools_and_forwards_calls(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}

    runtime = InstalledFeatureRuntime(
        class_name="TestFeature",
        entry_point="test_pkg.feature:TestFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test_service",
        description="Test proxy",
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, runtime, client_factory=client_factory)
    await feature.initialize()

    tools = feature.get_tools()
    assert feature.name == "TestFeature"
    assert feature.tool_description == "Test proxy"
    assert len(tools) == 1
    assert tools[0].name == "ping"
    assert tools[0].schema.command_prefix == "!ping"
    # F004: the advertised input_schema is converted into real ToolParameters,
    # so the tool reaches the LLM with usable arguments (not an empty list).
    params = tools[0].schema.parameters
    assert [p.name for p in params] == ["message"]
    assert params[0].type == "string"
    assert params[0].required is True
    assert params[0].description == "text to echo"
    # And the schema survives OpenAI conversion (crashes if params are dicts).
    openai = tools[0].schema.to_openai_format()
    props = openai["function"]["parameters"]["properties"]
    assert props["message"]["type"] == "string"

    result = await tools[0].execute(message="hello")
    assert result["success"] is True
    assert result["result"] == {"echo": {"message": "hello"}}
    assert clients[0].calls == [("ping", {"message": "hello"})]

    await feature.shutdown()
    assert clients[0].stopped is True


@pytest.mark.asyncio
async def test_scheduled_isolated_call_fails_before_dispatch_without_context_capability(
    tmp_path,
):
    """A legacy isolated service must not receive an unkeyed scheduled effect."""

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    client = FakeIsolatedClient()
    feature._client = client
    execution = SchedulerExecution(
        id="execution-1",
        schedule_id="schedule-1",
        agent_id="agent-1",
        task_name="ping",
        args={"message": "hello"},
        scheduled_for="2026-07-25T15:00:00+00:00",
        idempotency_key="stable-effect-key",
        attempt=1,
        owner="runner-1",
    )

    token = _current_execution.set(_SchedulerExecutionScope(execution))
    try:
        with pytest.raises(SchedulerExecutionContextUnavailable, match="advertises"):
            await feature.call_isolated_tool("ping", {"message": "hello"})
    finally:
        _current_execution.reset(token)

    # No RPC is attempted, so a legacy child can never perform a duplicate
    # effect without the scheduler's stable idempotency key.
    assert client.calls == []


@pytest.mark.asyncio
async def test_scheduler_records_terminal_isolated_admission_as_failed(tmp_path, monkeypatch):
    """A sealed proxy raises for scheduler delivery, so the durable row fails."""

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)

    backend = SQLiteBackend(str(tmp_path / "scheduler.db"))
    await backend.connect()
    db = AsyncDatabase(backend)

    async def dispatch_terminal_isolated_call(_task_name, _args):
        return await feature.call_isolated_tool("ping", {})

    runner = SchedulerRunner(
        db,
        _TEST_AGENT_DID,
        dispatch_terminal_isolated_call,
        owner_id="terminal-admission-runner",
    )
    try:
        await feature.initialize()
        client = feature._client
        await feature.shutdown()
        assert feature._traffic_gate.sealed is True

        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, schedule_kind, timezone_name,
                 misfire_policy, idempotency_key, scheduler_protocol_version)
            VALUES (?, ?, 'isolated_ping', '* * * * *', '{}', 1, ?, ?,
                    'cron', 'UTC', 'skip', 'terminal-admission', ?)
            """,
            ("terminal-admission-task", _TEST_AGENT_DID, due, due, SCHEDULER_PROTOCOL_VERSION),
        )

        await runner._tick()

        status, result_text = await db.fetchone(
            "SELECT status, result_text FROM task_execution_log WHERE task_id = ?",
            ("terminal-admission-task",),
        )
        assert status == "failed"
        assert result_text == (
            "SchedulerTerminalAdmissionError: isolated feature traffic is unavailable"
        )
        assert client.calls == []
    finally:
        if not feature._stopping:
            await feature.shutdown()

        await db.close()


@pytest.mark.asyncio
async def test_scheduler_revokes_context_for_detached_core_child_before_late_isolated_call(
    tmp_path,
):
    """A child that outlives dispatch cannot reuse a completed occurrence.

    ``asyncio.create_task`` copies the scheduler ContextVar.  Exercise the
    production runner and proxy together: the child inherits the context
    during dispatch, waits for the runner to clear the occurrence, then calls
    the isolated tool.  That late call must be a normal untrusted call, never
    an RPC bearing the stale scheduler idempotency identity.
    """

    class ContextAwareClient(FakeIsolatedClient):
        supports_tool_execution_context = True

        async def call_tool(self, name, args, *, context=None):
            self.calls.append((name, args, context))
            return {"echo": args}

    class NoStorageRunner(SchedulerRunner):
        # This test isolates context revocation rather than persistence. The
        # real runner proves its durable token before and at effect admission;
        # retain that precondition without giving this deliberately storageless
        # test double a database implementation.
        async def _renew_lease_once(self, task):
            return True

        async def _claim_token_is_live(self, task):
            return True

        async def _renew_lease(self, task):
            await asyncio.Future()

        async def _finalize(self, *args, **kwargs):
            return None

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    client = ContextAwareClient()
    feature._client = client

    child_started = asyncio.Event()
    release_child = asyncio.Event()
    child: asyncio.Task[None] | None = None

    async def late_isolated_call() -> None:
        child_started.set()
        await release_child.wait()
        await feature.call_isolated_tool("ping", {"message": "late"})

    async def executor(name, args):
        nonlocal child
        assert get_current_scheduler_execution() is not None
        child = asyncio.create_task(late_isolated_call())
        await child_started.wait()
        return "dispatched"

    now = datetime.now(timezone.utc).isoformat()
    task = ScheduledTask(
        id="schedule-1",
        agent_id="agent-1",
        task_name="ping",
        cron_expression="* * * * *",
        args_json='{"message": "scheduled"}',
        enabled=True,
        last_run_at=None,
        next_run_at=now,
        created_at=now,
        idempotency_key="stable-effect-key",
        claim_token="claim-token",
        claim_execution_id="execution-1",
        claim_scheduled_for=now,
        attempt_count=1,
    )
    runner = NoStorageRunner(object(), "agent-1", executor, owner_id="runner-1")

    await runner._execute_claim(task)
    assert get_current_scheduler_execution() is None

    release_child.set()
    assert child is not None
    await child

    # A stale scope would give this call a ToolExecutionContext with the
    # completed occurrence ID and stable idempotency key.  It must be absent.
    assert client.calls == [("ping", {"message": "late"}, None)]


def test_service_command_console_script(tmp_path):
    """`service` resolves to a console-script in the venv bin/, not `python -m`."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    runtime = InstalledFeatureRuntime(
        class_name="WhatsAppWebFeature",
        entry_point="wa.feature:WhatsAppWebFeature",
        distribution="kestrel-channel-whatsapp",
        runtime="isolated-venv",
        service="kestrel-whatsapp-web",
        project="service",
    )
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    cmd = feature._service_command()
    assert cmd == [str(feature._venv_path / "bin" / "kestrel-whatsapp-web")]
    # the install target is `project`, never the `service` runnable
    assert (runtime.project or runtime.distribution) == "service"


def test_service_command_module_func(tmp_path):
    """`service` of the form module:func runs via the venv python, not `-m`."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    runtime = InstalledFeatureRuntime(
        class_name="SvcFeature",
        entry_point="svc.feature:SvcFeature",
        distribution="svc-pkg",
        runtime="isolated-venv",
        service="svc_pkg.service:main",
    )
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    cmd = feature._service_command()
    assert cmd[0] == str(feature._venv_path / "bin" / "python")
    assert cmd[1] == "-c"
    assert "from svc_pkg.service import main" in cmd[2]
    assert "-m" not in cmd  # never `python -m <install-target>`


@pytest.mark.asyncio
async def test_supervision_registered_and_child_stopped_on_cancel(tmp_path):
    """Leak guard: supervision task registers with the agent's background-task
    lifecycle, and cancelling it (agent shutdown path) stops the child."""
    import asyncio

    tracked = []

    class FakeAgent:
        storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
        features: dict = {}

        def _track_background_task(self, coro, *, name):
            task = asyncio.create_task(coro, name=name)
            tracked.append(task)
            return task

    runtime = InstalledFeatureRuntime(
        class_name="SvcFeature",
        entry_point="svc.feature:SvcFeature",
        distribution="svc-pkg",
        runtime="isolated-venv",
        service="svc",
    )
    feature = ProxyFeature(FakeAgent(), runtime, client_factory=FakeIsolatedClient)
    feature._client_factory = lambda **kw: FakeIsolatedClient(**kw)
    monkey_bin = tmp_path  # avoid real venv work
    import os
    os.environ["KESTREL_FEATURE_SVCFEATURE_BIN"] = str(monkey_bin / "svc-bin")
    try:
        await feature.initialize()
        # registered through the agent's tracker, not a bare task
        assert feature._supervision_task in tracked
        client = feature._client
        await asyncio.sleep(0.05)  # let the supervision loop enter its body
        # simulate agent shutdown cancelling tracked background tasks
        feature._supervision_task.cancel()
        try:
            await feature._supervision_task
        except asyncio.CancelledError:
            pass
        assert client.stopped is True  # child torn down despite no shutdown() call
    finally:
        os.environ.pop("KESTREL_FEATURE_SVCFEATURE_BIN", None)


class FakeChannelRegistry:
    def __init__(self):
        self.adapters = {}

    def register(self, adapter):
        self.adapters[adapter.channel_type] = adapter

    def get(self, channel_type):
        return self.adapters.get(channel_type)

    def unregister(self, channel_type):
        return self.adapters.pop(channel_type, None)


class FakeChannelFeature:
    def __init__(self):
        self.registry = FakeChannelRegistry()
        self.inbound = []

    async def handle_inbound(self, message):
        self.inbound.append(message)
        return SimpleNamespace(durably_admitted=True)


def _isolated_runtime():
    return InstalledFeatureRuntime(
        class_name="WhatsAppFeature",
        entry_point="wa.feature:WhatsAppFeature",
        distribution="kestrel-channel-whatsapp",
        runtime="isolated-venv",
        service="wa-service",
    )


@pytest.mark.asyncio
async def test_route_link_qr_persists_png_only(tmp_path):
    """A channel.link_qr event is written to the agent data dir as the latest
    PNG (served by the endpoint the persisted channel_link card fetches). It is
    NO LONGER pushed as a live SSE bubble / sticky event (#2081) — that path
    orphaned on refresh."""
    import base64

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    emitted = []

    async def emit_event(event_type, data):
        emitted.append((event_type, data))

    agent.emit_event = emit_event

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)

    png = b"\x89PNG\r\n\x1a\n" + b"fake-qr-bytes"
    await feature._route_link_qr(
        {
            "channel_type": "WhatsApp",  # mixed case → normalized to lowercase
            "png_b64": base64.b64encode(png).decode("ascii"),
            "caption": "Scan me",
        }
    )

    out = tmp_path / "agent" / "channel_link_artifacts" / "whatsapp_link_qr.png"
    assert out.exists()
    assert out.read_bytes() == png

    # No SSE emit and no sticky replay — the card is a persisted typed part now.
    assert emitted == []
    agent.set_sticky_event.assert_not_called()


@pytest.mark.asyncio
async def test_route_link_cleared_removes_png(tmp_path):
    """Linking clears the persisted PNG so the channel_link card resolves to
    'expired or already linked'. No sticky/SSE state to retract (#2081)."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    emitted = []

    async def emit_event(event_type, data):
        emitted.append((event_type, data))

    agent.emit_event = emit_event
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)

    art = tmp_path / "agent" / "channel_link_artifacts"
    art.mkdir(parents=True)
    (art / "whatsapp_link_qr.png").write_bytes(b"\x89PNG\r\n\x1a\nqr")

    await feature._route_link_cleared({"channel_type": "whatsapp"})

    assert not (art / "whatsapp_link_qr.png").exists()
    assert emitted == []
    agent.clear_sticky_event.assert_not_called()


@pytest.mark.asyncio
async def test_route_link_qr_rejects_malformed_payloads(tmp_path):
    """Bad channel_type / missing PNG / traversal attempts are dropped, no emit."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    emitted = []

    async def emit_event(event_type, data):
        emitted.append((event_type, data))

    agent.emit_event = emit_event
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)

    # missing png
    await feature._route_link_qr({"channel_type": "whatsapp"})
    # path-traversal / invalid channel type
    await feature._route_link_qr({"channel_type": "../etc", "png_b64": "AAAA"})
    # non-dict
    await feature._route_link_qr(None)

    assert emitted == []
    assert not (tmp_path / "agent" / "channel_link_artifacts").exists()


@pytest.mark.asyncio
async def test_channel_link_tool_emits_persisted_part(monkeypatch, tmp_path):
    """When the bridged channel's pairing tool runs on the streaming turn, the
    host emits a persisted ``channel_link`` typed part (a reference, not the QR
    bytes) so the pairing card rides the conversation that asked for it (#2081)."""
    from kestrel_sovereign.agent import parts as parts_mod

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {"ChannelFeature": channel_feature}

    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")

    class ChannelClient(FakeIsolatedClient):
        capabilities = {
            "channel": {"channel_type": "whatsapp", "send_tool": "whatsapp_send"}
        }

        async def list_tools(self):
            return [
                {"name": "whatsapp_link", "description": "Link WhatsApp", "category": "utility"},
                {"name": "whatsapp_send", "description": "Send", "category": "utility"},
            ]

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=ChannelClient)
    await feature.initialize()

    # Convention fallback: <channel_type>_link is the pairing tool.
    assert feature._link_tool == "whatsapp_link"

    # The pairing tool, run inside a streaming turn's part collector, emits the
    # persisted channel_link part.
    with parts_mod.part_collector():
        await feature.call_isolated_tool("whatsapp_link", {})
        drained = parts_mod.drain_parts()
    assert drained == [{"type": "channel_link", "data": {"channel_type": "whatsapp"}}]

    # A non-link tool (e.g. send) emits nothing.
    with parts_mod.part_collector():
        await feature.call_isolated_tool("whatsapp_send", {"to": "x", "message": "y"})
        assert parts_mod.drain_parts() == []

    await feature.shutdown()


@pytest.mark.asyncio
async def test_proxy_forwards_host_config_into_client(monkeypatch, tmp_path):
    """Persisted host config is loaded and handed to the client (-> initialize handshake)."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}

    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")
    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=client_factory)

    async def fake_load(**_kwargs):
        return {"provider": "web", "allowed_senders": ["+13035551234"]}

    feature.load_persisted_config = fake_load  # type: ignore[assignment]
    await feature.initialize()

    assert captured["config"] == {
        "provider": "web",
        "allowed_senders": ["+13035551234"],
    }
    await feature.shutdown()


@pytest.mark.asyncio
async def test_transient_startup_config_read_failure_recovers_without_losing_secret(
    monkeypatch, tmp_path
):
    """A failed durable read must not boot an authoritative empty config.

    The feature starts only after storage recovers, at which point a
    write-only-secret-preserving PATCH receives the actual durable value rather
    than replacing it with the empty config used by the old best-effort path.
    """

    stored_config = {"enabled": True, "token": "stored-secret-not-for-logs"}

    class TransientReadStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.fail_reads = False

        async def get_node(self, node_id):
            if self.fail_reads:
                raise OSError("storage temporarily unavailable")
            return await super().get_node(node_id)

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = TransientReadStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(stored_config)
    agent.storage.fail_reads = True

    with pytest.raises(RuntimeError, match="failed to load persisted config"):
        await feature.initialize()
    assert clients == []
    assert feature._host_config_loaded is False

    agent.storage.fail_reads = False
    await feature.initialize()
    assert clients[0].kwargs["config"] == stored_config

    # This mirrors the endpoint's write-only-secret preservation: the request
    # omits ``token``, so the recovered current config supplies it before save.
    partial_patch = {"enabled": False}
    current = await feature.get_config()
    partial_patch["token"] = current["token"]
    await feature.set_config(partial_patch)

    assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == {
        "enabled": False,
        "token": stored_config["token"],
    }
    await feature.shutdown()


@pytest.mark.asyncio
async def test_proxy_bridges_channel_capability_into_registry(monkeypatch, tmp_path):
    """A service advertising a channel capability is registered as a forwarding adapter,
    and channels_send-style routing reaches the service tool."""
    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {"ChannelFeature": channel_feature}

    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")

    class ChannelClient(FakeIsolatedClient):
        capabilities = {
            "channel": {
                "channel_type": "whatsapp",
                "send_tool": "whatsapp_send",
                "status_tool": "whatsapp_status",
            }
        }

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            return {"ok": True, "data": {"message_id": "WAMID.1"}, "message": "sent"}

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=lambda **kw: ChannelClient(**kw))
    await feature.initialize()

    adapter = channel_feature.registry.adapters.get("whatsapp")
    assert adapter is not None
    assert adapter.is_connected is True

    receipt = await adapter.send_message(to="+13035551234", content="hi")
    assert receipt.status.value == "success"
    assert receipt.message_id == "WAMID.1"
    assert feature._client.calls == [
        ("whatsapp_send", {"to": "+13035551234", "message": "hi"})
    ]

    await feature.shutdown()
    assert "whatsapp" not in channel_feature.registry.adapters


@pytest.mark.asyncio
async def test_replacement_bridge_uses_effective_config_for_outbound_and_inbound_authorization(
    monkeypatch, tmp_path
):
    """A replacement bridge must not inherit stale sender/enabled policy."""

    from kestrel_sovereign.features.channels.feature import ChannelFeature
    from kestrel_sovereign.features.channels.models import (
        ChannelMessage,
        MessageDirection,
    )

    old_config = {
        "agent_id": "old-agent",
        "enabled": False,
        "allowed_senders": ["old-sender"],
    }
    target_config = {
        "agent_id": "target-agent",
        "enabled": True,
        "allowed_senders": ["target-sender"],
    }
    routed_senders: list[str] = []

    async def route_inbound(message):
        routed_senders.append(message.sender)

    channel_agent = SimpleNamespace(
        did=_TEST_AGENT_DID,
        storage=SimpleNamespace(agent_id=_TEST_AGENT_DID),
        dispatcher=None,
        signal_registry=None,
        features={},
    )
    channel_feature = ChannelFeature(channel_agent)
    await channel_feature.initialize()
    channel_feature.registry.set_inbound_router(route_inbound)

    class ChannelClient(FakeIsolatedClient):
        capabilities = {
            "channel": {
                "channel_type": "whatsapp",
                "send_tool": "whatsapp_send",
            }
        }

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            return {"ok": True, "data": {"message_id": "WAMID.target"}}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")
    clients: list[ChannelClient] = []

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=client_factory)
    feature._host_config = dict(old_config)
    feature._host_config_loaded = True
    try:
        await feature._connect_client(old_config)
        # ``_replace_client(target_config)`` models forced reconciliation: the
        # target child starts before the caller updates ``_host_config``.
        await feature._replace_client(target_config)

        adapter = channel_feature.registry.get("whatsapp")
        assert adapter is not None
        assert feature._host_config == old_config
        assert adapter.config.enabled is True
        assert adapter.config.allowed_senders == ["target-sender"]
        assert adapter.config.agent_id == "target-agent"

        # Outbound authorization reads ``enabled`` from the bridge config.
        outbound = await channel_feature.channels_send("whatsapp", "+1", "hello")
        assert outbound.status is ToolResultStatus.OK
        assert clients[1].calls == [
            ("whatsapp_send", {"to": "+1", "message": "hello"})
        ]

        # Inbound authorization reads ``allowed_senders`` from that same
        # bridge config, so old policy neither admits old senders nor blocks
        # the target sender.
        await channel_feature.handle_inbound(
            ChannelMessage(
                channel_type="whatsapp",
                direction=MessageDirection.INBOUND,
                sender="old-sender",
                recipient="bot",
                content="stale policy",
            )
        )
        await channel_feature.handle_inbound(
            ChannelMessage(
                channel_type="whatsapp",
                direction=MessageDirection.INBOUND,
                sender="target-sender",
                recipient="bot",
                content="target policy",
            )
        )
        assert routed_senders == ["target-sender"]
    finally:
        await feature.shutdown()
        await channel_feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_does_not_evict_replacement_adapter(monkeypatch, tmp_path):
    """If another adapter replaced our channel_type, shutdown must not remove it."""
    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {"ChannelFeature": channel_feature}
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")

    class ChannelClient(FakeIsolatedClient):
        capabilities = {"channel": {"channel_type": "whatsapp", "send_tool": "whatsapp_send"}}

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=lambda **kw: ChannelClient(**kw))
    await feature.initialize()
    assert channel_feature.registry.get("whatsapp") is feature._channel_adapter

    # A native/replacement adapter takes over the same channel_type.
    replacement = object.__new__(type(feature._channel_adapter))
    replacement._channel_type = "whatsapp"  # type: ignore[attr-defined]
    channel_feature.registry.adapters["whatsapp"] = replacement

    await feature.shutdown()
    assert channel_feature.registry.get("whatsapp") is replacement


@pytest.mark.asyncio
async def test_proxy_send_maps_failure_receipt(monkeypatch, tmp_path):
    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {"ChannelFeature": channel_feature}
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")

    class FailingChannelClient(FakeIsolatedClient):
        capabilities = {
            "channel": {"channel_type": "whatsapp", "send_tool": "whatsapp_send"}
        }

        async def call_tool(self, name, args):
            return {"ok": False, "error": "not linked"}

    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=lambda **kw: FailingChannelClient(**kw)
    )
    await feature.initialize()
    adapter = channel_feature.registry.adapters["whatsapp"]
    receipt = await adapter.send_message(to="+1", content="x")
    assert receipt.status.value == "failure"
    assert "not linked" in (receipt.error or "")
    await feature.shutdown()


@pytest.mark.asyncio
async def test_proxy_send_maps_toolresult_envelopes(monkeypatch, tmp_path):
    """ToolResult wire shapes (status=error/partial) must not read as success."""
    from kestrel_sovereign.features.isolated_runtime import (
        _delivery_receipt_from_result,
    )

    # status=error wrapped as a successful transport call must be a FAILURE
    err = _delivery_receipt_from_result(
        "whatsapp", {"success": True, "result": {"status": "error", "error": "not linked"}}
    )
    assert err.status.value == "failure"
    assert "not linked" in (err.error or "")

    # status=partial -> PENDING (honesty: not yet confirmed)
    part = _delivery_receipt_from_result(
        "whatsapp",
        {"success": True, "result": {"status": "partial", "data": {"receipt": {"message_id": "M2"}}}},
    )
    assert part.status.value == "pending"
    assert part.message_id == "M2"

    # status=ok -> SUCCESS
    ok = _delivery_receipt_from_result(
        "whatsapp", {"success": True, "result": {"status": "ok", "data": {"message_id": "M3"}}}
    )
    assert ok.status.value == "success"
    assert ok.message_id == "M3"

    # top-level message_id on a plain {"ok": True} envelope is preserved
    top = _delivery_receipt_from_result(
        "whatsapp", {"success": True, "result": {"ok": True, "message_id": "WAMID.top"}}
    )
    assert top.status.value == "success"
    assert top.message_id == "WAMID.top"


def test_proxy_feature_resolves_default_per_agent_venv(tmp_path):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    runtime = InstalledFeatureRuntime(
        class_name="VoiceFeature",
        entry_point="voice.feature:VoiceFeature",
        distribution="kestrel-feature-voice",
        runtime="isolated-venv",
    )

    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    venv, bin_path = feature.resolve_runtime_paths()

    assert (
        venv
        == Path(agent.storage_path).parent / "feature_venvs" / "VoiceFeature" / ".venv"
    )
    assert bin_path is None


@pytest.mark.asyncio
async def test_supervision_restarts_child_on_wedged_health_probe(tmp_path):
    """F013: a health() that never returns must not wedge supervision forever —
    the probe is bounded and a timeout drops through to stop()/start()."""
    import asyncio

    import kestrel_sovereign.features.isolated_runtime as ir

    class WedgedThenHealthyClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.health_calls = 0
            self.starts = 0

        async def start(self):
            self.starts += 1
            await super().start()

        async def health(self):
            self.health_calls += 1
            if self.health_calls == 1:
                # First probe hangs past the bound → treated as wedged.
                await asyncio.sleep(3600)
            return True

    runtime = InstalledFeatureRuntime(
        class_name="WedgeFeature",
        entry_point="w.feature:WedgeFeature",
        distribution="w-pkg",
        runtime="isolated-venv",
        service="w",
    )
    client_holder = {}

    def factory(**kw):
        client_holder["c"] = WedgedThenHealthyClient(**kw)
        return client_holder["c"]

    feature = ProxyFeature(Mock(storage_path=str(tmp_path / "a" / "db.db"), features={}),
                           runtime, client_factory=factory)
    import os
    os.environ["KESTREL_FEATURE_WEDGEFEATURE_BIN"] = str(tmp_path / "w-bin")
    # Shrink the probe bound and backoff so the test is fast.
    monkey = pytest.MonkeyPatch()
    monkey.setattr(ir, "_HEALTH_PROBE_TIMEOUT", 0.05)
    try:
        await feature.initialize()
        client = client_holder["c"]
        # Wait for: first (hanging) probe to time out, then the restart path.
        for _ in range(200):
            await asyncio.sleep(0.02)
            if client.starts >= 2:
                break
        assert client.stopped is True
        assert client.starts >= 2  # child was restarted after the wedged probe
    finally:
        feature._stopping = True
        if feature._supervision_task:
            feature._supervision_task.cancel()
            try:
                await feature._supervision_task
            except asyncio.CancelledError:
                pass
        monkey.undo()
        os.environ.pop("KESTREL_FEATURE_WEDGEFEATURE_BIN", None)


def test_ensure_venv_reprovisions_when_host_sdk_upgrades(tmp_path, monkeypatch):
    """F019: a provisioned venv whose stamped host SDK version no longer matches
    the running host is reinstalled (--upgrade), not silently reused."""
    import json

    import kestrel_sovereign.features.isolated_runtime as ir

    runtime = InstalledFeatureRuntime(
        class_name="StampFeature",
        entry_point="s.feature:StampFeature",
        distribution="stamp-pkg",
        runtime="isolated-venv",
        service="s",
        project="stamp-pkg",
    )
    feature = ProxyFeature(Mock(storage_path=str(tmp_path / "a" / "db.db")),
                           runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()

    runs = []

    def fake_run(cmd):
        runs.append(cmd)
        # Materialize the venv python so the "exists" branch is taken next time.
        py = feature._venv_path / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.touch()

    monkeypatch.setattr(feature, "_run", fake_run)
    # Child venv python is a stub (empty file) — report a concrete SDK version
    # rather than shelling out to it.
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.28.0")

    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.28.0")
    feature.ensure_venv()  # fresh: uv venv + uv pip install (no --upgrade)
    assert any(c[:3] == ["uv", "venv"] or c[0] == "uv" and "venv" in c for c in runs)
    install_cmds = [c for c in runs if "pip" in c and "install" in c]
    assert install_cmds and "--upgrade" not in install_cmds[-1]
    manifest = json.loads((feature._venv_path / ".kestrel_provision.json").read_text())
    assert manifest["provisioned_against_host_sdk"] == "0.28.0"
    assert manifest["child_sdk_version"] == "0.28.0"

    runs.clear()
    # Same host SDK → no reprovision.
    feature.ensure_venv()
    assert runs == []

    # Host upgraded → reprovision with --upgrade, stamped against the new host.
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.29.0")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.29.0")
    feature.ensure_venv()
    upgrades = [c for c in runs if "pip" in c and "install" in c]
    assert upgrades and "--upgrade" in upgrades[-1]
    manifest = json.loads((feature._venv_path / ".kestrel_provision.json").read_text())
    assert manifest["provisioned_against_host_sdk"] == "0.29.0"

    runs.clear()
    # A feature that pins an OLD sdk installs "successfully" but the child stays
    # behind: we still stamp against the new host so we don't reinstall every
    # startup, but the mismatch is recorded (and warned) rather than hidden.
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.30.0")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.28.0")
    feature.ensure_venv()
    manifest = json.loads((feature._venv_path / ".kestrel_provision.json").read_text())
    assert manifest["provisioned_against_host_sdk"] == "0.30.0"
    assert manifest["child_sdk_version"] == "0.28.0"
    runs.clear()
    feature.ensure_venv()  # not stale now (host unchanged) → no thrash
    assert runs == []


def test_ensure_venv_does_not_mutate_operator_override_venv(tmp_path, monkeypatch):
    """#2125 regression: an operator-supplied (override) venv that already exists
    must NOT be `uv pip install --upgrade`'d — that rewrites a prebuilt/pinned
    environment the operator provided (and hard-fails the feature at startup if
    the index is unreachable). ensure_venv verifies + warns but leaves it
    untouched and stamps no manifest we don't own."""
    import kestrel_sovereign.features.isolated_runtime as ir

    override_venv = tmp_path / "prebuilt" / ".venv"
    py = override_venv / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.touch()  # a venv that ALREADY exists (operator built it)

    runtime = InstalledFeatureRuntime(
        class_name="OverrideFeature",
        entry_point="o.feature:OverrideFeature",
        distribution="override-pkg",
        runtime="isolated-venv",
        service="o",
        project="override-pkg",
        venv=str(override_venv),  # pyproject `venv =` override
    )
    feature = ProxyFeature(Mock(storage_path=str(tmp_path / "a" / "db.db")),
                           runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    assert feature._venv_path == override_venv.resolve()

    runs = []
    monkeypatch.setattr(feature, "_run", lambda cmd: runs.append(cmd))
    # SDK mismatch present — must warn, but still NOT mutate.
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.28.0")
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.29.0")

    feature.ensure_venv()

    assert runs == [], f"override venv must not be touched, ran: {runs}"
    assert not (override_venv / ".kestrel_provision.json").exists()


def test_ensure_venv_reprovisions_host_created_override_venv(tmp_path, monkeypatch):
    """An override venv that KESTREL created earlier (carries our manifest) keeps
    the reprovision lifecycle — a host SDK upgrade still triggers --upgrade. Only
    prebuilt override venvs (no manifest) are left untouched (#2125)."""
    import kestrel_sovereign.features.isolated_runtime as ir

    override_venv = tmp_path / "hostbuilt" / ".venv"
    runtime = InstalledFeatureRuntime(
        class_name="OverrideFeature", entry_point="o.feature:OverrideFeature",
        distribution="override-pkg", runtime="isolated-venv", service="o",
        project="override-pkg", venv=str(override_venv),
    )
    feature = ProxyFeature(Mock(storage_path=str(tmp_path / "a" / "db.db")),
                           runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()

    runs = []

    def fake_run(cmd):
        runs.append(cmd)
        py = feature._venv_path / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.touch()

    monkeypatch.setattr(feature, "_run", fake_run)
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.28.0")
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.28.0")

    # First startup: override path missing → bootstrap (create + install + stamp).
    feature.ensure_venv()
    assert any("install" in c for c in runs)
    assert (override_venv / ".kestrel_provision.json").exists()

    runs.clear()
    feature.ensure_venv()  # unchanged host → no thrash even though it's an override
    assert runs == []

    # Host SDK upgrade → OUR override venv reprovisions (--upgrade).
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.29.0")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.29.0")
    feature.ensure_venv()
    upgrades = [c for c in runs if "pip" in c and "install" in c]
    assert upgrades and "--upgrade" in upgrades[-1]


# --- F023: isolated service launch env must not inherit interpreter shadowing --


def test_isolated_child_env_strips_shadowing_vars(monkeypatch, tmp_path):
    """The launch env must drop host PYTHONPATH/PYTHONHOME/PYTHONSTARTUP/
    VIRTUAL_ENV so the host interpreter can't shadow the isolated venv (F023)."""
    from kestrel_sovereign.features.isolated_runtime import (
        _isolated_child_env,
        _venv_bin_dir,
    )

    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("PYTHONSTARTUP", "/host/startup.py")
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_TOKEN", "keep-me")

    venv = tmp_path / "svc-venv"
    env = _isolated_child_env(venv)

    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PYTHONSTARTUP" not in env
    # Feature config/secrets still pass through (documented config channel).
    assert env["KESTREL_FEATURE_WHATSAPPFEATURE_TOKEN"] == "keep-me"
    # VIRTUAL_ENV re-points at the isolated venv and its bin leads PATH.
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"].split(os.pathsep)[0] == str(_venv_bin_dir(venv))


def test_build_client_passes_stripped_env(monkeypatch, tmp_path):
    """``_build_client`` must hand the stripped env to the client factory."""
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=client_factory)
    feature._venv_path = tmp_path / "svc-venv"
    feature._bin_path = None

    feature._build_client()

    assert "env" in captured
    assert "PYTHONPATH" not in captured["env"]


def test_venv_sdk_version_uses_canonical_isolated_child_env(monkeypatch, tmp_path):
    """The version probe cannot resolve the host SDK through hostile env vars."""
    import kestrel_sovereign.features.isolated_runtime as ir

    venv = tmp_path / "feature-venv"
    python = venv / "bin" / "python"
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("PYTHONSTARTUP", "/host/startup.py")
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    captured = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return ir.subprocess.CompletedProcess([], 0, stdout="0.35.1\n")

    monkeypatch.setattr(ir.subprocess, "run", fake_run)

    assert ir._venv_sdk_version(python) == "0.35.1"
    env = captured["env"]
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PYTHONSTARTUP" not in env
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"].split(os.pathsep)[0] == str(venv / "bin")


def test_build_client_preserves_explicit_empty_config(tmp_path):
    """An effective empty config is sent as ``{}``, not omitted as ``None``."""

    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    feature = ProxyFeature(
        Mock(features={}),
        _isolated_runtime(),
        client_factory=client_factory,
    )
    feature._venv_path = tmp_path / "svc-venv"
    feature._bin_path = tmp_path / "test-service"
    feature._host_config = {}

    feature._build_client()

    assert captured["config"] == {}


class _FakeStorage:
    """Minimal graph store double with the production CAS contract."""

    def __init__(self):
        self.nodes = {}
        # Isolated config migration is only permitted through a graph-store
        # capability bound to the same DID as the proxy.
        self.agent_id = _TEST_AGENT_DID

    async def add_node(self, node):
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def compare_and_swap_node(self, node_id, expected, new_node):
        current = self.nodes.get(node_id)
        current_properties = None if current is None else current.properties
        if current_properties != expected:
            return "predicate_failed"
        self.nodes[node_id] = new_node
        return "swapped"


class _CASStorage(_FakeStorage):
    """Named CAS double retained for tests that model hosted contention."""


def _cfg_runtime():
    return InstalledFeatureRuntime(
        class_name="TestFeature",
        entry_point="test_pkg.feature:TestFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test_service",
        description="Test proxy",
    )


@pytest.mark.asyncio
async def test_set_config_persists_node_reflects_in_get_config_and_reloads(monkeypatch, tmp_path):
    """#2214: set_config must persist to the feature_config:<name> graph node,
    get_config must reflect it (not an empty client passthrough), and the running
    service must be reloaded so the new config actually takes effect."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    clients = []

    def client_factory(**kwargs):
        c = FakeIsolatedClient(**kwargs)
        clients.append(c)
        return c

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.initialize()
    assert len(clients) == 1

    new_cfg = {"enabled": True, "allowed_senders": ["8825903191"], "token": "12345:abc"}
    await feature.set_config(new_cfg)

    # 1) Persisted to the graph node the isolated runtime reads at startup.
    node = agent.storage.nodes.get(_TEST_CONFIG_NODE_ID)
    assert node is not None, "set_config did not persist the feature_config node"
    assert node.properties["config"]["allowed_senders"] == ["8825903191"]
    assert "pending_config" not in node.properties
    assert "_isolated_pending_generation" not in node.properties
    assert "_isolated_pending_owner" not in node.properties
    assert "_isolated_pending_lease_expires_at" not in node.properties

    # 2) get_config reflects it (previously always returned {}), incl. the secret
    #    so the endpoint's write-only-secret preservation works.
    got = await feature.get_config()
    assert got["allowed_senders"] == ["8825903191"]
    assert got["token"] == "12345:abc"

    # 3) Reloaded: old client stopped, a NEW client built with the new config
    #    forwarded through the initialize handshake (config only flows at init).
    assert clients[0].stopped is True
    assert len(clients) == 2
    assert clients[1].kwargs.get("config") == new_cfg


@pytest.mark.asyncio
async def test_post_promotion_start_failure_rebuilds_active_child_before_traffic_reopens(
    monkeypatch, tmp_path
):
    """A failed promoted candidate is repaired before finite traffic reopens.

    ``_replace_client`` must unpublish the old child before stopping it, so a
    startup failure after promotion leaves no client published.  The cleanup
    path must rebuild from the now-durable active config rather than reopening
    an empty proxy.
    """

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": False, "revision": "next"}

    class PromotionCandidate(FakeIsolatedClient):
        async def start(self):
            self.started = True
            raise RuntimeError("promoted candidate could not start")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FakeIsolatedClient(**kwargs)
            if len(clients) != 1
            else PromotionCandidate(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    reopen_clients = []
    original_reopen = feature._reopen_traffic_gate

    async def assert_reconciled_then_reopen():
        # This is the finite gate's only reopen point for the transition.  The
        # recovery child must already be reachable and configured from the
        # durable promotion before any traffic can be admitted again.
        assert feature._traffic_gate.closed is True
        assert feature._client is clients[2]
        assert feature._client.kwargs["config"] == next_config
        assert feature._host_config == next_config
        reopen_clients.append(feature._client)
        await original_reopen()

    monkeypatch.setattr(feature, "_reopen_traffic_gate", assert_reconciled_then_reopen)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(RuntimeError, match="promoted candidate could not start"):
            await feature.set_config(next_config)

        assert clients[0].stopped is True
        assert clients[1].stopped is True
        assert clients[2].started is True
        assert feature._client is clients[2]
        assert feature.get_tools()
        assert feature._traffic_gate.closed is False
        assert feature._traffic_gate.sealed is False
        assert reopen_clients == [clients[2]]
        assert (
            await feature.call_isolated_tool("ping", {"message": "recovered"})
        ) == {
            "success": True,
            "result": {"echo": {"message": "recovered"}},
            "tool": "ping",
        }
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_post_promotion_recovery_start_failure_seals_traffic(monkeypatch, tmp_path):
    """A failed forced rebuild leaves the promoted state terminally unreachable."""

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": False, "revision": "next"}

    class StartFailingClient(FakeIsolatedClient):
        async def start(self):
            self.started = True
            raise RuntimeError("replacement child could not start")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs) if not clients else StartFailingClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)

    async def fail_if_traffic_reopens():
        raise AssertionError("failed recovery must not reopen traffic")

    monkeypatch.setattr(feature, "_reopen_traffic_gate", fail_if_traffic_reopens)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(RuntimeError, match="replacement child could not start"):
            await feature.set_config(next_config)

        # The first failed child was the post-promotion candidate and the
        # second was the forced durable recovery. Neither is publishable.
        assert clients[0].stopped is True
        assert clients[1].stopped is True
        assert clients[2].stopped is True
        assert clients[2].kwargs["config"] == next_config
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._stopping is True
        assert feature._traffic_gate.sealed is True
        assert await feature.call_isolated_tool("ping", {"message": "blocked"}) == {
            "status": "error",
            "error": "isolated feature traffic is unavailable",
            "tool": "ping",
            "success": False,
        }
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "next_config",
    [
        {
            "enabled": False,
            "token": "12345678:old-token",
            "transport": "webhook",
            "webhook_url": "https://ingress.example.test/telegram",
            "webhook_secret": "old-webhook-secret",
        },
        {
            "enabled": True,
            "token": "87654321:new-token",
            "transport": "webhook",
            "webhook_url": "https://ingress.example.test/telegram",
            "webhook_secret": "new-webhook-secret",
        },
    ],
    ids=["disable", "credential-rotation"],
)
async def test_supported_transition_cleans_up_before_replacing_old_service(
    monkeypatch, tmp_path, caplog, next_config
):
    """A webhook-shaped service prepares both disable and token-rotation first.

    The test deliberately records only ordering and public config shape. It
    never logs the credentials it carries, which proves the host can honor the
    generic SDK lifecycle without introducing Telegram-specific dispatch.
    """

    old_config = {
        "enabled": True,
        "token": "12345678:old-token",
        "transport": "webhook",
        "webhook_url": "https://ingress.example.test/telegram",
        "webhook_secret": "old-webhook-secret",
    }
    events = []

    class WebhookCleanupClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            events.append(("cleanup", dict(config)))
            return ConfigTransitionResult.restart_required()

        async def stop(self):
            events.append(("stop", None))
            await super().stop()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    clients = []

    def client_factory(**kwargs):
        client = (
            WebhookCleanupClient(**kwargs)
            if not clients
            else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(old_config)
    await feature.initialize()

    await feature.set_config(next_config)

    # The lifecycle RPC receives the complete effective next config and its
    # ordered resource cleanup completes before the old process is stopped.
    assert events == [("cleanup", next_config), ("stop", None)]
    assert clients[0].kwargs["config"] == old_config
    assert clients[1].kwargs["config"] == next_config
    assert clients[0].stopped is True
    for secret in (
        old_config["token"],
        old_config["webhook_secret"],
        next_config["token"],
        next_config["webhook_secret"],
    ):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_supported_transition_failure_keeps_old_config_and_service(monkeypatch, tmp_path):
    """A failed negotiated cleanup cannot masquerade as a successful apply."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": True, "token": "next-token"}

    class FailingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            raise ConfigTransitionError("config transition failed")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(
        agent,
        _cfg_runtime(),
        client_factory=FailingTransitionClient,
    )
    await feature.persist_config(old_config)
    await feature.initialize()
    old_client = feature._client

    with pytest.raises(ConfigTransitionError, match="config transition failed"):
        await feature.set_config(next_config)

    # The normal SDK failure contract preserves the known-safe old child. The
    # host also restores durable/in-memory config so a later restart cannot
    # launch the candidate without the cleanup that failed above.
    assert feature._client is old_client
    assert old_client.stopped is False
    assert feature._host_config == old_config
    assert (await feature.get_config()) == old_config
    assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == old_config


@pytest.mark.asyncio
async def test_external_ingress_quiesces_callback_while_admission_is_open_then_drains(
    monkeypatch, tmp_path
):
    """A real pending producer callback cannot deadlock Core's transition."""

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": True, "revision": "next"}
    delivered = []
    clients = []
    callback_started = asyncio.Event()
    callback_finished = asyncio.Event()
    release_callback = asyncio.Event()

    class QuiescingIngressClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.quiesced = False
            self.pending_events = []
            self.ingress_calls = []
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=(
                    "telegram-polling-ack",
                    "external-ingress-quiesce",
                    "external-ingress-resume",
                )
            )

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if name == "telegram-polling-ack":
                callback_started.set()
                await release_callback.wait()
                callback_finished.set()
                return {"status": "ok", "http_status": 200, "state": "acknowledged"}
            if name == "external-ingress-quiesce":
                # The provider quiesce waits for its actual callback to
                # finish. Core must leave admission open for that wait.
                assert feature._traffic_gate.closed is False
                await callback_finished.wait()
                self.quiesced = True
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            assert name == "external-ingress-resume"
            # Resume is an owned finalizer: deferred work cannot replay until
            # this exact source has resumed and Core reopens admission.
            assert feature._traffic_gate.closed is True
            self.quiesced = False
            for event in self.pending_events:
                await self.event_handler(event)
            self.pending_events.clear()
            return {"status": "ok", "http_status": 200, "state": "resumed"}

        async def emit_external_update(self, event):
            if self.quiesced:
                self.pending_events.append(event)
                return
            await self.event_handler(event)

        async def prepare_config_transition(self, config):
            assert config == next_config
            # Quiesce has now completed, so Core closes admission before the
            # hook runs. New producer events are retained until resume.
            assert feature._traffic_gate.closed is True
            await self.emit_external_update(
                {
                    "type": "channel.inbound",
                    "payload": {
                        "message": {
                            "id": "late-update",
                            "metadata": {"dedupe_key": "late-update"},
                        },
                        "_host_ingress_ack": {
                            "name": "telegram-polling-ack",
                            "payload": {"dedupe_key": "late-update"},
                        },
                    },
                }
            )
            return ConfigTransitionResult.applied()

    def client_factory(**kwargs):
        client = QuiescingIngressClient(**kwargs)
        clients.append(client)
        return client

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)

    async def record(event):
        delivered.append(event)
        return SimpleNamespace(durably_admitted=True)

    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        feature._route_inbound = record  # type: ignore[method-assign]

        await clients[0].event_handler(
            _acknowledged_telegram_event("telegram:v2:bot:42:update:early")
        )
        await asyncio.wait_for(callback_started.wait(), timeout=1)

        transition_task = asyncio.create_task(feature.set_config(next_config))
        await asyncio.sleep(0)
        assert not transition_task.done()
        release_callback.set()
        await asyncio.wait_for(transition_task, timeout=1)

        assert [
            call[0]
            for call in clients[0].ingress_calls
            if call[0].startswith("external-ingress-")
        ] == [
            "external-ingress-quiesce",
            "external-ingress-resume",
        ]
        for _ in range(20):
            if delivered:
                break
            await asyncio.sleep(0)
        assert delivered == [
            {
                "channel_type": "telegram",
                "direction": "inbound",
                "sender": "555",
                "recipient": "42",
                "content": "hello",
                "id": "telegram:v2:bot:42:update:early",
                "metadata": {"dedupe_key": "telegram:v2:bot:42:update:early"},
            },
            {"id": "late-update", "metadata": {"dedupe_key": "late-update"}}
        ]
        assert clients[0].pending_events == []
        assert feature._traffic_gate.closed is False
        assert feature._host_config == next_config
    finally:
        release_callback.set()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_external_ingress_resumes_after_failed_transition_rollback(monkeypatch, tmp_path):
    """A failed staged transition resumes polling before reopening Core admission."""

    old_config = {"enabled": True, "revision": "old"}
    rejected_config = {"enabled": True, "revision": "rejected"}

    class RollbackIngressClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.quiesced = False
            self.ingress_calls = []
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if name == "external-ingress-quiesce":
                self.quiesced = True
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            assert name == "external-ingress-resume"
            assert feature._traffic_gate.closed is True
            self.quiesced = False
            return {"status": "ok", "http_status": 200, "state": "resumed"}

        async def prepare_config_transition(self, config):
            assert self.quiesced is True
            assert config == rejected_config
            raise ConfigTransitionError("transition rejected")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=RollbackIngressClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        old_client = feature._client

        with pytest.raises(ConfigTransitionError, match="transition rejected"):
            await feature.set_config(rejected_config)

        assert feature._client is old_client
        assert old_client.quiesced is False
        assert [call[0] for call in old_client.ingress_calls] == [
            "external-ingress-quiesce",
            "external-ingress-resume",
        ]
        assert feature._traffic_gate.closed is False
        assert feature._host_config == old_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_external_ingress_resumes_after_cancelled_transition_rollback(monkeypatch, tmp_path):
    """Cancellation after admission closes releases the exact producer."""

    old_config = {"enabled": True, "revision": "old"}
    pending_config = {"enabled": True, "revision": "pending"}
    quiesce_started = asyncio.Event()
    release_quiesce = asyncio.Event()

    class CancelledIngressClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.quiesced = False
            self.ingress_calls = []
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if name == "external-ingress-quiesce":
                self.quiesced = True
                quiesce_started.set()
                await release_quiesce.wait()
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            assert name == "external-ingress-resume"
            assert feature._traffic_gate.closed is True
            self.quiesced = False
            return {"status": "ok", "http_status": 200, "state": "resumed"}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=CancelledIngressClient)
    update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        client = feature._client

        update = asyncio.create_task(feature.set_config(pending_config))
        await asyncio.wait_for(quiesce_started.wait(), timeout=1)
        update.cancel()
        release_quiesce.set()
        with pytest.raises(asyncio.CancelledError):
            await update

        assert feature._client is client
        assert client.quiesced is False
        assert [call[0] for call in client.ingress_calls] == [
            "external-ingress-quiesce",
            "external-ingress-resume",
        ]
        assert feature._traffic_gate.closed is False
        assert feature._host_config == old_config
    finally:
        if update is not None and not update.done():
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
        await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resume_error", "expected_error"),
    [
        (RuntimeError("resume failed"), HostIngressError),
        (asyncio.CancelledError("resume cancelled"), asyncio.CancelledError),
    ],
    ids=["failure", "cancellation"],
)
async def test_external_ingress_resume_failure_never_reports_config_success(
    monkeypatch, tmp_path, resume_error, expected_error
):
    """A failed resume quarantines and remains the successful body's outcome."""

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": True, "revision": "next"}

    class ResumeFailingClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_host_ingress(self, name, payload=None):
            if name == "external-ingress-quiesce":
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            assert name == "external-ingress-resume"
            raise resume_error

        async def prepare_config_transition(self, config):
            assert config == next_config
            return ConfigTransitionResult.applied()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=ResumeFailingClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(expected_error):
            await feature.set_config(next_config)

        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._stopping is True
        assert feature._traffic_gate.sealed is True
        assert feature._host_config == next_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_external_ingress_resume_failure_preserves_active_transition_error(
    monkeypatch, tmp_path
):
    """Resume cleanup cannot hide the transition error already being unwound."""

    old_config = {"enabled": True, "revision": "old"}
    rejected_config = {"enabled": True, "revision": "rejected"}

    class BodyFailingClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_host_ingress(self, name, payload=None):
            if name == "external-ingress-quiesce":
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            assert name == "external-ingress-resume"
            raise RuntimeError("resume failed during unwind")

        async def prepare_config_transition(self, config):
            assert config == rejected_config
            raise ConfigTransitionError("transition rejected")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BodyFailingClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(ConfigTransitionError, match="transition rejected"):
            await feature.set_config(rejected_config)

        assert feature._client is None
        assert feature._traffic_gate.sealed is True
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_transition_failure_clears_pending_config_when_promotion_fails(
    monkeypatch, tmp_path
):
    """A failed hook clears its generation instead of wedging future updates."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": True, "token": "next-token"}

    class RollbackFailingStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            is_owned_cleanup = (
                isinstance(expected, dict)
                and "pending_config" in expected
                and "pending_config" not in new_node.properties
            )
            if is_owned_cleanup and self.cas_calls == 3:
                raise OSError("storage offline during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class FailingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            raise ConfigTransitionError("config transition failed")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = RollbackFailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FailingTransitionClient)
    await feature.persist_config(old_config)
    await feature.initialize()

    with pytest.raises(ConfigTransitionError, match="config transition failed"):
        await feature.set_config(next_config)

    node = agent.storage.nodes[_TEST_CONFIG_NODE_ID]
    assert node.properties["config"] == old_config
    assert "pending_config" not in node.properties
    assert "_isolated_pending_generation" not in node.properties
    # stage → post-reconciliation lease renewal → failed cleanup → retry
    assert agent.storage.cas_calls == 4

    fresh = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await fresh.initialize()
    assert fresh._host_config == old_config
    assert (await fresh.get_config()) == old_config


@pytest.mark.asyncio
async def test_hook_failure_clears_generation_and_allows_immediate_retry(
    monkeypatch, tmp_path
):
    """A known hook failure leaves no durable barrier to the next update."""

    old_config = {"enabled": True, "token": "old-token"}
    failed_config = {"enabled": False, "token": "failed-token"}
    retry_config = {"enabled": False, "token": "retry-token"}

    class FlakyTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.preparations = 0

        async def prepare_config_transition(self, config):
            self.preparations += 1
            if self.preparations == 1:
                assert config == failed_config
                raise ConfigTransitionError("hook rejected staged config")
            assert config == retry_config
            return ConfigTransitionResult.restart_required()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FlakyTransitionClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(ConfigTransitionError, match="hook rejected"):
            await feature.set_config(failed_config)

        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert "_isolated_pending_generation" not in properties

        await feature.set_config(retry_config)

        assert clients[0].preparations == 2
        assert feature._host_config == retry_config
        assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == retry_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_cancellation_after_stage_clears_owned_generation(monkeypatch, tmp_path):
    """Cancelling a staged hook aborts that generation and starts no candidate."""

    old_config = {"enabled": True, "token": "old-token"}
    pending_config = {"enabled": False, "token": "pending-token"}
    hook_started = asyncio.Event()

    class BlockingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == pending_config
            hook_started.set()
            await asyncio.Event().wait()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = BlockingTransitionClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        update = asyncio.create_task(feature.set_config(pending_config))
        await asyncio.wait_for(hook_started.wait(), timeout=1)

        staged = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert staged["pending_config"] == pending_config
        assert isinstance(staged["_isolated_pending_owner"], str)
        assert isinstance(staged["_isolated_pending_lease_expires_at"], str)

        update.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update

        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert all(client.kwargs["config"] != pending_config for client in clients)
    finally:
        if update is not None and not update.done():
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
        await feature.shutdown()


@pytest.mark.asyncio
async def test_abort_commit_then_raise_is_reconciled_before_retry(monkeypatch, tmp_path):
    """An abort connection error after commit does not strand its generation."""

    old_config = {"enabled": True, "token": "old-token"}
    rejected_config = {"enabled": False, "token": "rejected-token"}

    class CommitThenRaiseAbortStorage(_CASStorage):
        def __init__(self):
            super().__init__()
            self.abort_raised = False

        async def compare_and_swap_node(self, node_id, expected, new_node):
            result = await super().compare_and_swap_node(node_id, expected, new_node)
            is_abort = (
                result == "swapped"
                and isinstance(expected, dict)
                and "pending_config" in expected
                and "pending_config" not in new_node.properties
            )
            if is_abort and not self.abort_raised:
                self.abort_raised = True
                raise ConnectionError("connection dropped after abort commit")
            return result

    class FailingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == rejected_config
            raise ConfigTransitionError("hook rejected staged config")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = CommitThenRaiseAbortStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FailingTransitionClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(ConfigTransitionError, match="hook rejected"):
            await feature.set_config(rejected_config)

        assert agent.storage.abort_raised is True
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_fresh_replica_cannot_steal_unexpired_pending_generation(
    monkeypatch, tmp_path
):
    """A new proxy fences itself behind a healthy writer's durable lease."""

    old_config = {"enabled": True, "token": "old-token"}
    first_config = {"enabled": False, "token": "first-token"}
    second_config = {"enabled": False, "token": "second-token"}
    hook_started = asyncio.Event()
    release_hook = asyncio.Event()

    class BlockingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == first_config
            hook_started.set()
            await release_hook.wait()
            return ConfigTransitionResult.restart_required()

    storage = _CASStorage()
    first_agent = Mock(did=_TEST_AGENT_DID, features={})
    first_agent.storage = storage
    first_agent.storage_path = str(tmp_path / "first" / "kestrel_prime.db")
    second_agent = Mock(did=_TEST_AGENT_DID, features={})
    second_agent.storage = storage
    second_agent.storage_path = str(tmp_path / "second" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    first = ProxyFeature(first_agent, _cfg_runtime(), client_factory=BlockingTransitionClient)
    second_clients = []

    def second_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        second_clients.append(client)
        return client

    second = ProxyFeature(second_agent, _cfg_runtime(), client_factory=second_factory)
    first_update = None
    try:
        await first.persist_config(old_config)
        await first.initialize()
        first_update = asyncio.create_task(first.set_config(first_config))
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        staged = dict(storage.nodes[_TEST_CONFIG_NODE_ID].properties)

        await second.initialize()
        with pytest.raises(RuntimeError, match="already in progress"):
            await second.set_config(second_config)

        assert storage.nodes[_TEST_CONFIG_NODE_ID].properties == staged
        assert second_clients[0].kwargs["config"] == old_config
        assert all(client.kwargs["config"] != first_config for client in second_clients)

        release_hook.set()
        await first_update
    finally:
        release_hook.set()
        if first_update is not None and not first_update.done():
            await first_update
        await second.shutdown()
        await first.shutdown()


@pytest.mark.asyncio
async def test_expired_pending_generation_is_cleared_then_retried_from_active_config(
    monkeypatch, tmp_path
):
    """Takeover removes only expired metadata and never starts its candidate."""

    now = datetime(2026, 7, 26, 12, tzinfo=timezone.utc)
    old_config = {"enabled": True, "token": "old-token"}
    abandoned_config = {"enabled": False, "token": "abandoned-token"}
    next_config = {"enabled": False, "token": "next-token"}
    monkeypatch.setattr(isolated_runtime, "_utc_now", lambda: now)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    storage = _CASStorage()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    seed = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await seed.persist_config(old_config)
    storage.nodes[_TEST_CONFIG_NODE_ID].properties.update(
        {
            "pending_config": abandoned_config,
            "_isolated_pending_generation": "abandoned-generation",
            "_isolated_pending_owner": "abandoned-owner",
            "_isolated_pending_lease_expires_at": (
                now - isolated_runtime._PENDING_CONFIG_CLOCK_SKEW - timedelta(seconds=1)
            ).isoformat(),
        }
    )

    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.initialize()
        await feature.set_config(next_config)

        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
        assert all(client.kwargs["config"] != abandoned_config for client in clients)
        assert clients[0].kwargs["config"] == old_config
        assert clients[1].kwargs["config"] == next_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_malformed_pending_lease_metadata_fails_closed(monkeypatch, tmp_path):
    """A corrupt lease is never treated as expired or cleared speculatively."""

    old_config = {"enabled": True, "token": "old-token"}
    malformed_config = {"enabled": False, "token": "malformed-token"}
    next_config = {"enabled": False, "token": "next-token"}
    storage = _CASStorage()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    seed = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await seed.persist_config(old_config)
    storage.nodes[_TEST_CONFIG_NODE_ID].properties.update(
        {
            "pending_config": malformed_config,
            "_isolated_pending_generation": "malformed-generation",
            "_isolated_pending_owner": "malformed-owner",
            "_isolated_pending_lease_expires_at": "not-a-timestamp",
        }
    )
    original_properties = dict(storage.nodes[_TEST_CONFIG_NODE_ID].properties)
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    try:
        await feature.initialize()
        with pytest.raises(RuntimeError, match="stored pending config lease is invalid"):
            await feature.set_config(next_config)

        assert storage.nodes[_TEST_CONFIG_NODE_ID].properties == original_properties
        assert feature._host_config == old_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_empty_active_config_hides_pending_candidate_from_concurrent_reads(
    monkeypatch, tmp_path
):
    """An initialized empty config is active, not a signal to read persistence."""

    next_config = {"enabled": True, "token": "next-token"}
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()

    class SlowFailingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            preparation_started.set()
            await release_preparation.wait()
            raise ConfigTransitionError("config transition failed")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(
        agent,
        _cfg_runtime(),
        client_factory=SlowFailingTransitionClient,
    )
    await feature.initialize()
    assert feature._host_config == {}

    update = asyncio.create_task(feature.set_config(next_config))
    await asyncio.wait_for(preparation_started.wait(), timeout=1)

    properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
    assert properties["config"] == {}
    assert properties["pending_config"] == next_config
    assert isinstance(properties["_isolated_pending_generation"], str)
    assert await feature.get_config() == {}

    release_preparation.set()
    with pytest.raises(ConfigTransitionError, match="config transition failed"):
        await update
    assert await feature.get_config() == {}


@pytest.mark.asyncio
async def test_persistence_failure_prevents_transition_hook_from_running(monkeypatch, tmp_path):
    """A non-privacy storage failure is a fail-closed transition boundary."""

    class FailingStorage(_FakeStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            raise OSError("storage unavailable")

    class TransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.preparation_calls = 0

        async def prepare_config_transition(self, config):
            self.preparation_calls += 1
            return ConfigTransitionResult.restart_required()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = FailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=TransitionClient)
    await feature.initialize()
    old_client = feature._client

    with pytest.raises(RuntimeError, match="failed to persist config"):
        await feature.set_config({"enabled": True, "token": "next-token"})

    assert old_client.preparation_calls == 0
    assert old_client.stopped is False
    assert feature._client is old_client
    assert feature._host_config == {}


@pytest.mark.asyncio
async def test_volatile_privacy_noop_allows_transition_without_durable_config(
    monkeypatch, tmp_path
):
    """The explicit privacy-policy no-op is the one allowed write exception."""

    class VolatileStorage(_FakeStorage):
        def allows_persistent_writes(self):
            return False

    class TransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared = []

        async def prepare_config_transition(self, config):
            self.prepared.append(dict(config))
            return ConfigTransitionResult.applied()

    next_config = {"enabled": True, "token": "volatile-token"}
    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = VolatileStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=TransitionClient)
    await feature.initialize()
    client = feature._client

    await feature.set_config(next_config)

    assert client.prepared == [next_config]
    assert feature._client is client
    assert feature._host_config == next_config
    assert agent.storage.nodes == {}


@pytest.mark.asyncio
async def test_fenced_transition_failure_replaces_with_next_config(monkeypatch, tmp_path):
    """An unknown lifecycle outcome follows the SDK's required replacement path."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": True, "token": "next-token"}

    class FencedTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            self.replacement_required = True
            raise ConfigTransitionError("config transition failed")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FencedTransitionClient(**kwargs)
            if not clients
            else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(old_config)
    await feature.initialize()

    with pytest.raises(ConfigTransitionError, match="config transition failed"):
        await feature.set_config(next_config)

    # A cancellation or transport break can leave the child partly transitioned;
    # the SDK marks that state explicitly. The apply still fails to its caller,
    # but the host must not keep using that fenced old process.
    assert clients[0].stopped is True
    assert clients[1].kwargs["config"] == next_config
    assert feature._host_config == next_config


@pytest.mark.asyncio
async def test_fenced_transition_promotion_failure_restores_active_config_for_restart(
    monkeypatch, tmp_path
):
    """A fenced candidate is retired when its durable promotion fails.

    A cancelled/transport-broken SDK transition can have applied ``next_config``
    in the old child, so the proxy first replaces it with a candidate child on
    that config.  Fault the *following* promotion write: the live replacement,
    host state, and a fresh proxy must all return to the still-durable old
    config.  The existing supervisor must remain live on that restored child.
    """

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": False, "token": "next-token"}

    class PromotionFailingStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            # Identify promotion by its exact state shape rather than write
            # ordinal: a pre-hook lease renewal is now a separate CAS.
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                raise OSError("storage offline during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class FencedTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            assert config == next_config
            self.replacement_required = True
            raise ConfigTransitionError("config transition failed")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = PromotionFailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FencedTransitionClient(**kwargs)
            if not clients
            else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    fresh = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(ConfigTransitionError, match="config transition failed"):
            await feature.set_config(next_config)

        # The fenced process is retired, but no next-config child is ever
        # started: its promotion failed while next_config was still pending.
        # The replacement and host-facing config therefore agree with the
        # durable active value.
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == old_config
        assert feature._client is clients[1]
        assert feature._host_config == old_config
        assert await feature.get_config() == old_config
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert "_isolated_pending_generation" not in properties

        # The recovery stays inside the serialized reload section without
        # disabling supervision. A later fresh proxy (host restart) also sees
        # and launches only the durable old config.
        assert feature._supervision_task is not None
        assert feature._supervision_task.done() is False
        restart_clients = []

        def restart_factory(**kwargs):
            client = FakeIsolatedClient(**kwargs)
            restart_clients.append(client)
            return client

        fresh = ProxyFeature(agent, _cfg_runtime(), client_factory=restart_factory)
        await fresh.initialize()
        assert fresh._host_config == old_config
        assert restart_clients[0].kwargs["config"] == old_config
    finally:
        if fresh is not None:
            await fresh.shutdown()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_recovery_never_starts_candidate_before_durable_promotion(
    monkeypatch, tmp_path
):
    """A pending fenced candidate is never started, even detached.

    Child startup may allocate external resources. While promotion is blocked
    (and then fails), the only permitted replacement is the old configuration
    that the graph node still names as active.
    """

    old_config = {"enabled": True, "token": "old-token-not-for-logs"}
    next_config = {"enabled": False, "token": "next-token-not-for-logs"}
    promotion_started = asyncio.Event()
    release_promotion = asyncio.Event()

    class BlockingPromotionStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                promotion_started.set()
                await release_promotion.wait()
                raise OSError("storage unavailable during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class FencedTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            assert config == next_config
            self.replacement_required = True
            raise ConfigTransitionError("transition outcome unknown")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = BlockingPromotionStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FencedTransitionClient(**kwargs)
            if not clients
            else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    update = None

    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(promotion_started.wait(), timeout=1)
        await asyncio.sleep(0)

        # The old child is fenced and removed from host traffic, but a pending
        # next-config candidate has not even been constructed or started.
        assert len(clients) == 1
        assert feature._client is None
        assert feature.get_tools() == []

        # Direct traffic is admitted only after recovery: it cannot reach the
        # fenced child while its candidate remains merely pending.
        blocked_call = asyncio.create_task(
            feature.call_isolated_tool("ping", {"message": "blocked"})
        )
        await asyncio.sleep(0)
        assert blocked_call.done() is False

        release_promotion.set()
        with pytest.raises(ConfigTransitionError, match="transition outcome unknown"):
            await update

        result = await blocked_call
        assert result["success"] is True

        # The only recovery child starts from the still-durable old config.
        assert len(clients) == 2
        assert clients[1].kwargs["config"] == old_config
        assert feature._client is clients[1]
        assert feature._host_config == old_config
    finally:
        release_promotion.set()
        if update is not None and not update.done():
            try:
                await update
            except (RuntimeError, ConfigTransitionError):
                pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_promotion_commit_then_raise_reconciles_to_committed_next_config(
    monkeypatch, tmp_path
):
    """A post-commit connection failure is not misreported as a rollback."""

    old_config = {"enabled": True, "token": "old-token-not-for-errors"}
    next_config = {"enabled": False, "token": "next-token-not-for-errors"}

    class CommitThenRaiseStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            result = await super().compare_and_swap_node(node_id, expected, new_node)
            if (
                result == "swapped"
                and new_node.properties.get("config") == next_config
                and "pending_config" not in new_node.properties
            ):
                raise ConnectionError("connection dropped after commit")
            return result

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = CommitThenRaiseStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(RuntimeError, match="failed to persist config") as exc_info:
            await feature.set_config(next_config)

        # The caller learns that the transport outcome was uncertain, while
        # the local child and the durable source of truth follow the committed
        # next generation rather than a fictional rollback.
        assert old_config["token"] not in str(exc_info.value)
        assert next_config["token"] not in str(exc_info.value)
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == next_config
        assert feature._client is clients[1]
        assert feature._host_config == next_config
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
        assert isinstance(properties["_isolated_config_generation"], str)
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_promotion_cancellation_after_commit_reconciles_before_propagating(
    monkeypatch, tmp_path
):
    """Cancellation after a committed CAS leaves no stale child or rollback lie."""

    old_config = {"enabled": True, "token": "old-token-not-for-errors"}
    next_config = {"enabled": False, "token": "next-token-not-for-errors"}
    promotion_committed = asyncio.Event()

    class CommitThenCancelStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            result = await super().compare_and_swap_node(node_id, expected, new_node)
            if (
                result == "swapped"
                and new_node.properties.get("config") == next_config
                and "pending_config" not in new_node.properties
            ):
                promotion_committed.set()
                await asyncio.Event().wait()
            return result

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = CommitThenCancelStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(promotion_committed.wait(), timeout=1)
        update.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update

        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == next_config
        assert feature._client is clients[1]
        assert feature._host_config == next_config
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
        assert isinstance(properties["_isolated_config_generation"], str)
    finally:
        if update is not None and not update.done():
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
        await feature.shutdown()


@pytest.mark.asyncio
async def test_stale_replica_cas_conflict_cannot_overwrite_winner_and_reconciles(
    monkeypatch, tmp_path
):
    """A loser with an old read reloads the winner instead of clobbering it."""

    old_config = {"enabled": True, "token": "old-token"}
    winner_config = {"enabled": False, "token": "winner-token"}
    loser_config = {"enabled": False, "token": "loser-token"}
    loser_stage_attempted = asyncio.Event()
    winner_promoted = asyncio.Event()

    class CoordinatedCASStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            is_loser_stage = (
                new_node.properties.get("pending_config") == loser_config
            )
            if is_loser_stage:
                loser_stage_attempted.set()
                await winner_promoted.wait()
            result = await super().compare_and_swap_node(node_id, expected, new_node)
            if (
                result == "swapped"
                and new_node.properties.get("config") == winner_config
                and "pending_config" not in new_node.properties
            ):
                winner_promoted.set()
            return result

    storage = CoordinatedCASStorage()
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    winner_agent = Mock(did=_TEST_AGENT_DID, features={})
    winner_agent.storage = storage
    winner_agent.storage_path = str(tmp_path / "winner" / "kestrel_prime.db")
    loser_agent = Mock(did=_TEST_AGENT_DID, features={})
    loser_agent.storage = storage
    loser_agent.storage_path = str(tmp_path / "loser" / "kestrel_prime.db")
    winner_clients = []
    loser_clients = []

    def winner_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        winner_clients.append(client)
        return client

    def loser_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        loser_clients.append(client)
        return client

    winner = ProxyFeature(winner_agent, _cfg_runtime(), client_factory=winner_factory)
    loser = ProxyFeature(loser_agent, _cfg_runtime(), client_factory=loser_factory)
    loser_update = None
    try:
        await winner.persist_config(old_config)
        await winner.initialize()
        await loser.initialize()

        # The loser reads ``old_config`` and pauses immediately before its CAS.
        # The winner then stages/promotes from the same former snapshot.
        loser_update = asyncio.create_task(loser.set_config(loser_config))
        await asyncio.wait_for(loser_stage_attempted.wait(), timeout=1)
        await winner.set_config(winner_config)

        with pytest.raises(RuntimeError, match="conflicts with a newer durable state") as exc_info:
            await loser_update

        assert loser_config["token"] not in str(exc_info.value)
        assert winner_config["token"] not in str(exc_info.value)
        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == winner_config
        assert "pending_config" not in properties
        assert loser_clients[0].stopped is True
        assert loser_clients[1].kwargs["config"] == winner_config
        assert loser._host_config == winner_config
        assert await loser.get_config() == winner_config
    finally:
        if loser_update is not None and not loser_update.done():
            loser_update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await loser_update
        await loser.shutdown()
        await winner.shutdown()


@pytest.mark.asyncio
async def test_promotion_conflict_reconciles_a_live_applied_child_to_durable_winner(
    monkeypatch, tmp_path
):
    """A losing promotion replaces a locally applied child from the re-read node."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": False, "token": "next-token"}
    winner_config = {"enabled": False, "token": "winner-token"}

    class PromotionConflictStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            if (
                new_node.properties.get("config") == next_config
                and "pending_config" not in new_node.properties
            ):
                # Another replica has already promoted a different generation
                # after this proxy staged and live-applied its own candidate.
                current = self.nodes[node_id]
                current.properties = {
                    "config": dict(winner_config),
                    "_isolated_config_generation": "other-replica",
                }
                return "predicate_failed"
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class LiveApplyClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            return ConfigTransitionResult.applied()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = PromotionConflictStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = LiveApplyClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(RuntimeError, match="conflicts with a newer durable state") as exc_info:
            await feature.set_config(next_config)

        assert old_config["token"] not in str(exc_info.value)
        assert next_config["token"] not in str(exc_info.value)
        assert winner_config["token"] not in str(exc_info.value)
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == winner_config
        assert feature._client is clients[1]
        assert feature._host_config == winner_config
        assert await feature.get_config() == winner_config
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == winner_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_promotion_policy_probe_failure_reconciles_live_child_to_staged_active(
    monkeypatch, tmp_path
):
    """A post-hook policy failure cannot leave an in-process next config live."""

    next_config = {"enabled": False, "token": "next-token-not-for-errors"}

    class FlappingPolicyStorage(_CASStorage):
        def __init__(self):
            super().__init__()
            self.policy_calls = 0

        def allows_persistent_writes(self):
            self.policy_calls += 1
            if self.policy_calls == 2:
                raise OSError("policy backend unavailable")
            return True

    class LiveApplyClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            return ConfigTransitionResult.applied()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = FlappingPolicyStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = LiveApplyClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.initialize()

        with pytest.raises(RuntimeError, match="failed to persist config") as exc_info:
            await feature.set_config(next_config)

        assert next_config["token"] not in str(exc_info.value)
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == {}
        assert feature._client is clients[1]
        assert feature._host_config == {}
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == {}
        assert "pending_config" not in properties
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_promotion_recovery_quarantines_when_old_child_cannot_restart(
    monkeypatch, tmp_path
):
    """A failed restoration exposes no candidate-config client to the host."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": False, "token": "next-token"}

    class PromotionFailingStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                raise OSError("storage offline during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class FencedTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            self.replacement_required = True
            raise ConfigTransitionError("config transition failed")

    class FailingRestoreClient(FakeIsolatedClient):
        async def start(self):
            raise RuntimeError("old-config child could not start")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = PromotionFailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        if not clients:
            client = FencedTransitionClient(**kwargs)
        else:
            client = FailingRestoreClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(RuntimeError, match="old-config child could not start"):
            await feature.set_config(next_config)

        # No pending next-config candidate starts. The sole attempted recovery
        # child uses durable old config, fails, and remains detached; the
        # supervisor latch prevents a later candidate resurrection.
        assert clients[1].kwargs["config"] == old_config
        assert clients[1].stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._host_config == old_config
        assert feature._stopping is True
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert "_isolated_pending_generation" not in properties
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_live_applied_transition_retains_the_initialized_service(monkeypatch, tmp_path):
    """An SDK ``applied`` result keeps the existing process alive."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": False, "token": "next-token"}

    class LiveApplyClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared = []

        async def prepare_config_transition(self, config):
            self.prepared.append(dict(config))
            return ConfigTransitionResult.applied()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=LiveApplyClient)
    await feature.persist_config(old_config)
    await feature.initialize()
    client = feature._client

    await feature.set_config(next_config)

    assert feature._client is client
    assert client.prepared == [next_config]
    assert client.stopped is False
    assert feature._host_config == next_config


@pytest.mark.asyncio
@pytest.mark.parametrize("transition_action", ["applied", "restart"])
async def test_failed_promotion_restores_previous_child_and_config(
    monkeypatch, tmp_path, transition_action
):
    """A failed second write cannot leave a successful hook's config live.

    Both negotiated outcomes can mutate or retire the current service before
    durable promotion. The host must therefore replace it from the staged
    active config when that promotion write fails.
    """

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": False, "token": "next-token"}

    class PromotionFailingStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                raise OSError("storage offline during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class TransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.transitioned_to = None

        async def prepare_config_transition(self, config):
            self.transitioned_to = dict(config)
            if transition_action == "applied":
                return ConfigTransitionResult.applied()
            return ConfigTransitionResult.restart_required()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = PromotionFailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = TransitionClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(old_config)
    await feature.initialize()

    with pytest.raises(RuntimeError, match="failed to persist config"):
        await feature.set_config(next_config)

    # The child that ran the successful transition is retired. Its replacement,
    # proxy state, and durable active value all agree on the prior config.
    assert clients[0].transitioned_to == next_config
    assert clients[0].stopped is True
    assert clients[1].kwargs["config"] == old_config
    assert feature._client is clients[1]
    assert feature._host_config == old_config
    assert await feature.get_config() == old_config
    properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
    assert properties["config"] == old_config
    assert "pending_config" not in properties
    assert "_isolated_pending_generation" not in properties


@pytest.mark.asyncio
async def test_supervision_restarts_on_sdk_fenced_health_response(tmp_path, monkeypatch):
    """The real SDK's fenced health envelope is not a healthy non-empty dict."""

    import kestrel_sovereign.features.isolated_runtime as ir

    health_checked = asyncio.Event()
    restarted = asyncio.Event()

    class FencedHealthClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            health_checked.set()
            if self.start_calls:
                return {"status": "ready", "ready": True}
            # This is the exact return shape from SDK health() after a
            # transition fences a child for replacement.
            return {"status": "restart-required", "ready": False}

        async def stop(self):
            self.stop_calls += 1
            await super().stop()

        async def start(self):
            self.start_calls += 1
            restarted.set()
            await super().start()

    client = FencedHealthClient()
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db"), features={}),
        _cfg_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._client = client

    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(ir.asyncio, "sleep", immediate_sleep)
    supervisor = asyncio.create_task(feature._supervise())
    try:
        await asyncio.wait_for(health_checked.wait(), timeout=1)
        await asyncio.wait_for(restarted.wait(), timeout=1)
        assert client.stop_calls == 1
        assert client.start_calls == 1
    finally:
        feature._stopping = True
        supervisor.cancel()
        try:
            await supervisor
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_config_transition_serializes_with_a_pending_health_restart(
    monkeypatch, tmp_path
):
    """A stale failed health probe cannot restart through a config transition."""

    import kestrel_sovereign.features.isolated_runtime as ir

    health_started = asyncio.Event()
    release_health = asyncio.Event()
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()

    class HealthAndTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def health(self):
            health_started.set()
            await release_health.wait()
            return False

        async def prepare_config_transition(self, config):
            preparation_started.set()
            await release_preparation.wait()
            return ConfigTransitionResult.restart_required()

        async def stop(self):
            self.stop_calls += 1
            await super().stop()

    old_client = HealthAndTransitionClient()
    replacement_clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        replacement_clients.append(client)
        return client

    agent = Mock(did=_TEST_AGENT_DID, storage_path=str(tmp_path / "agent" / "kestrel_prime.db"), features={})
    agent.storage = _FakeStorage()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    feature._client = old_client
    feature._host_config = {"enabled": True, "token": "old-token"}

    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(ir.asyncio, "sleep", immediate_sleep)
    supervisor = asyncio.create_task(feature._supervise())
    try:
        await asyncio.wait_for(health_started.wait(), timeout=1)
        update = asyncio.create_task(
            feature.set_config({"enabled": False, "token": "next-token"})
        )
        await asyncio.wait_for(preparation_started.wait(), timeout=1)

        # The supervisor has a stale unhealthy result but cannot acquire the
        # shared lifecycle lock until the transition has fully prepared and
        # replaced the child.
        release_health.set()
        release_preparation.set()
        await update
        await real_sleep(0)

        assert old_client.stop_calls == 1
        assert len(replacement_clients) == 1
        assert replacement_clients[0].started is True
        assert replacement_clients[0].stopped is False
    finally:
        feature._stopping = True
        supervisor.cancel()
        try:
            await supervisor
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_persisted_config_survives_restart(monkeypatch, tmp_path):
    """A config set on one ProxyFeature is loaded by a fresh one (restart)."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    f1 = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await f1.initialize()
    await f1.set_config({"allowed_senders": ["777"], "token": "t"})

    # Fresh feature (simulating a host/agent restart) reads the persisted node.
    f2 = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await f2.initialize()
    assert (await f2.get_config())["allowed_senders"] == ["777"]
    # And the reloaded service was launched with that config.
    assert f2._host_config["allowed_senders"] == ["777"]


@pytest.mark.asyncio
async def test_reenable_restarts_live_health_supervisor(tmp_path):
    """Runtime disable → re-enable on the SAME ProxyFeature instance must leave a
    LIVE health supervisor.

    ``shutdown()`` latches ``_stopping=True`` to unwind the supervisor; the
    canonical re-enable (``_activate_feature_runtime``) re-runs ``initialize()``
    on the same instance. Before the fix the stale ``_stopping`` made the new
    ``_supervise()`` task exit on its first ``while not self._stopping`` check,
    silently leaving the re-enabled service with NO health supervisor
    (kestrel-sovereign#2522 P2). This drives the real initialize/shutdown/
    initialize lifecycle and proves the second supervisor stays alive AND
    actively probes the freshly launched client."""
    import asyncio
    import os

    import kestrel_sovereign.features.isolated_runtime as ir

    clients = []

    class CountingClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.health_calls = 0

        async def health(self):
            self.health_calls += 1
            return True

    def factory(**kw):
        client = CountingClient(**kw)
        clients.append(client)
        return client

    runtime = InstalledFeatureRuntime(
        class_name="ReenableFeature",
        entry_point="r.feature:ReenableFeature",
        distribution="r-pkg",
        runtime="isolated-venv",
        service="r",
    )
    agent = Mock(did=_TEST_AGENT_DID, storage_path=str(tmp_path / "a" / "db.db"), features={})
    feature = ProxyFeature(agent, runtime, client_factory=factory)
    os.environ["KESTREL_FEATURE_REENABLEFEATURE_BIN"] = str(tmp_path / "r-bin")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(ir, "_HEALTH_PROBE_TIMEOUT", 0.05)
    try:
        # First enable → supervisor #1 against client #1.
        await feature.initialize()
        first_task = feature._supervision_task
        assert first_task is not None and not first_task.done()
        assert feature._stopping is False

        # Canonical disable latches _stopping and unwinds supervisor #1.
        await feature.shutdown()
        assert feature._stopping is True
        assert first_task.done()

        # Re-enable: initialize() re-runs on the SAME instance.
        await feature.initialize()
        second_task = feature._supervision_task
        # The lifecycle flag was reset to the fresh-start baseline.
        assert feature._stopping is False
        assert second_task is not first_task
        second_client = clients[-1]

        # The second supervisor is LIVE — it does NOT exit on its first check and
        # it actively probes the newly launched client. Before the fix
        # second_task would already be done() and health_calls would stay 0.
        for _ in range(150):
            if second_client.health_calls >= 1:
                break
            await asyncio.sleep(0.02)
        assert not second_task.done(), (
            "re-enabled health supervisor exited immediately (stale _stopping)"
        )
        assert second_client.health_calls >= 1, (
            "re-enabled supervisor never probed health — supervisor is dead"
        )
    finally:
        feature._stopping = True
        if feature._supervision_task:
            feature._supervision_task.cancel()
            try:
                await feature._supervision_task
            except asyncio.CancelledError:
                pass
        monkey.undo()
        os.environ.pop("KESTREL_FEATURE_REENABLEFEATURE_BIN", None)


@pytest.mark.asyncio
async def test_live_apply_blocks_tools_and_channel_but_drops_stale_inbound_until_promotion(
    monkeypatch, tmp_path
):
    """No host-visible effect may observe an applied candidate before its CAS.

    The client intentionally mutates its local mode before reporting ``applied``.
    Promotion is then held at the durable write boundary while a direct tool,
    the generic channel adapter, and an SDK inbound callback all try to enter.
    Tool/channel calls wait for the finite transition. An SDK callback is
    instead dropped immediately: it originated from the old child, and replay
    after promotion could apply stale inbound traffic under the new config.
    """

    from kestrel_sdk.channels import ChannelMessage, MessageDirection

    old_config = {"mode": "old", "enabled": True}
    next_config = {"mode": "next", "enabled": True}
    promotion_started = asyncio.Event()
    release_promotion = asyncio.Event()

    class BlockingPromotionStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            if (
                new_node.properties.get("config") == next_config
                and "pending_config" not in new_node.properties
            ):
                promotion_started.set()
                await release_promotion.wait()
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class LiveApplyChannelClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mode = dict(kwargs["config"])["mode"]
            self.effects = []

        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

        async def prepare_config_transition(self, config):
            self.mode = config["mode"]
            return ConfigTransitionResult.applied()

        async def call_tool(self, name, args):
            self.effects.append((name, self.mode))
            return {"ok": True, "data": {"message_id": "receipt"}}

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = BlockingPromotionStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=LiveApplyChannelClient)
    update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        client = feature._client
        adapter = channel_feature.registry.get("whatsapp")
        assert adapter is not None

        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(promotion_started.wait(), timeout=1)
        assert client.mode == "next"  # hook already applied locally
        assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == old_config

        tool_call = asyncio.create_task(feature.call_isolated_tool("ping", {}))
        channel_send = asyncio.create_task(adapter.send_message(to="+1", content="hello"))
        inbound = ChannelMessage(
            channel_type="whatsapp",
            direction=MessageDirection.INBOUND,
            sender="sender",
            recipient="agent",
            content="hello",
        )
        inbound_callback = asyncio.create_task(
            feature._handle_event({"type": "channel.inbound", "payload": inbound.to_dict()})
        )
        await asyncio.sleep(0)

        assert not tool_call.done()
        assert not channel_send.done()
        assert inbound_callback.done()
        assert client.effects == []
        assert channel_feature.inbound == []

        release_promotion.set()
        await update
        assert (await tool_call)["success"] is True
        assert (await channel_send).status.value == "success"
        await inbound_callback
        assert client.effects == [("ping", "next"), ("whatsapp_send", "next")]
        assert channel_feature.inbound == []
    finally:
        release_promotion.set()
        if update is not None and not update.done():
            await update
        await feature.shutdown()


@pytest.mark.asyncio
async def test_live_apply_republishes_config_dependent_tool_and_channel_inventory(
    monkeypatch, tmp_path
):
    """A successful in-process transition refreshes every host advertisement."""

    old_config = {"mode": "old", "enabled": True}
    next_config = {"mode": "next", "enabled": True}

    class InventoryClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mode = kwargs["config"]["mode"]

        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": f"{self.mode}_send",
                }
            }

        async def list_tools(self):
            return [{"name": f"{self.mode}_tool", "description": self.mode}]

        async def prepare_config_transition(self, config):
            self.mode = config["mode"]
            return ConfigTransitionResult.applied()

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=InventoryClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        old_adapter = channel_feature.registry.get("whatsapp")
        assert [tool.name for tool in feature.get_tools()] == ["old_tool"]
        assert old_adapter._send_tool == "old_send"

        await feature.set_config(next_config)

        new_adapter = channel_feature.registry.get("whatsapp")
        assert feature._client.stopped is False
        assert [tool.name for tool in feature.get_tools()] == ["next_tool"]
        assert new_adapter is not old_adapter
        assert new_adapter._send_tool == "next_send"
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_persistent_transition_requires_compare_and_swap_before_hook(monkeypatch, tmp_path):
    """An upsert-only storage surface fails before an SDK hook can run."""

    class NoCASStorage:
        def __init__(self):
            self.nodes = {}
            self.agent_id = _TEST_AGENT_DID

        async def add_node(self, node):
            self.nodes[node.node_id] = node

        async def get_node(self, node_id):
            return self.nodes.get(node_id)

    class TransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepare_calls = 0

        async def prepare_config_transition(self, config):
            self.prepare_calls += 1
            return ConfigTransitionResult.restart_required()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = NoCASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=TransitionClient)
    try:
        with pytest.raises(RuntimeError, match="requires compare_and_swap_node"):
            await feature.persist_config({"enabled": True})
        await feature.initialize()
        with pytest.raises(RuntimeError, match="failed to persist config"):
            await feature.set_config({"enabled": False})
        assert feature._client.prepare_calls == 0
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_healthy_long_hook_renews_lease_before_another_replica_can_takeover(
    monkeypatch, tmp_path
):
    """A live hook remains owned beyond its original lease interval."""

    old_config = {"enabled": True}
    first_config = {"enabled": False, "revision": "first"}
    second_config = {"enabled": False, "revision": "second"}
    hook_started = asyncio.Event()
    release_hook = asyncio.Event()
    lease_renewed = asyncio.Event()
    monkeypatch.setattr(isolated_runtime, "_PENDING_CONFIG_LEASE_TTL", timedelta(milliseconds=60))
    monkeypatch.setattr(isolated_runtime, "_PENDING_CONFIG_CLOCK_SKEW", timedelta(0))
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    class LongHookClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            hook_started.set()
            await release_hook.wait()
            return ConfigTransitionResult.restart_required()

    class RenewalObservingStorage(_CASStorage):
        def __init__(self):
            super().__init__()
            self.initial_lease = None

        async def compare_and_swap_node(self, node_id, expected, new_node):
            lease = new_node.properties.get("_isolated_pending_lease_expires_at")
            if isinstance(lease, str):
                if self.initial_lease is None:
                    self.initial_lease = datetime.fromisoformat(lease)
                elif (
                    datetime.fromisoformat(lease)
                    > self.initial_lease + timedelta(milliseconds=5)
                ):
                    # The post-reconciliation fence renewal happens almost
                    # immediately. Wait for the heartbeat's later renewal so
                    # this test still proves long-hook ownership beyond the
                    # original lease deadline.
                    lease_renewed.set()
            return await super().compare_and_swap_node(node_id, expected, new_node)

    storage = RenewalObservingStorage()
    first_agent = Mock(did=_TEST_AGENT_DID, features={})
    first_agent.storage = storage
    first_agent.storage_path = str(tmp_path / "first" / "kestrel_prime.db")
    second_agent = Mock(did=_TEST_AGENT_DID, features={})
    second_agent.storage = storage
    second_agent.storage_path = str(tmp_path / "second" / "kestrel_prime.db")
    first = ProxyFeature(first_agent, _cfg_runtime(), client_factory=LongHookClient)
    second = ProxyFeature(second_agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    first_update = None
    try:
        await first.persist_config(old_config)
        await first.initialize()
        await second.initialize()
        first_update = asyncio.create_task(first.set_config(first_config))
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        await asyncio.wait_for(lease_renewed.wait(), timeout=1)
        assert storage.initial_lease is not None
        # Advance the takeover clock past the original lease.  It remains
        # before the renewed generation-scoped lease, so a healthy hook cannot
        # be reclaimed merely because its first two-minute interval elapsed.
        monkeypatch.setattr(
            isolated_runtime,
            "_utc_now",
            lambda: storage.initial_lease + timedelta(milliseconds=1),
        )
        renewed_expires_at = datetime.fromisoformat(
            storage.nodes[_TEST_CONFIG_NODE_ID].properties[
                "_isolated_pending_lease_expires_at"
            ]
        )
        assert renewed_expires_at > isolated_runtime._utc_now()

        with pytest.raises(RuntimeError, match="already in progress"):
            await second.set_config(second_config)
        assert storage.nodes[_TEST_CONFIG_NODE_ID].properties["pending_config"] == first_config

        release_hook.set()
        await first_update
    finally:
        release_hook.set()
        if first_update is not None and not first_update.done():
            await first_update
        await second.shutdown()
        await first.shutdown()


@pytest.mark.asyncio
async def test_second_cancellation_waits_for_owned_cleanup_before_releasing_reload_lock(
    monkeypatch, tmp_path
):
    """Repeated cancellation cannot leave a cleanup task mutating after unlock."""

    old_config = {"enabled": True}
    next_config = {"enabled": False}
    hook_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    class BlockingCleanupStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            is_owned_cleanup = (
                isinstance(expected, dict)
                and "pending_config" in expected
                and "pending_config" not in new_node.properties
            )
            if is_owned_cleanup:
                cleanup_started.set()
                await release_cleanup.wait()
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class BlockingClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            hook_started.set()
            await asyncio.Event().wait()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = BlockingCleanupStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingClient)
    update = None
    second_update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        update.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)

        update.cancel()
        second_update = asyncio.create_task(feature.set_config({"enabled": True, "revision": 2}))
        await asyncio.sleep(0)
        assert not update.done()
        assert not second_update.done()
        assert "pending_config" in agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties

        # It is still waiting on the first transition's reload lock, so this
        # cancellation must prevent a second generation from being staged once
        # cleanup releases the lock.
        second_update.cancel()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await update
        assert "pending_config" not in agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties

        with pytest.raises(asyncio.CancelledError):
            await second_update
    finally:
        release_cleanup.set()
        for task in (update, second_update):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_stale_empty_replica_patch_preserves_concurrent_secret_rotation_at_stage_cas(
    monkeypatch, tmp_path
):
    """An initially empty replica cannot re-stage an old write-only secret."""

    first_secret_config = {"enabled": True, "api_key": "first-secret"}
    rotated_config = {"enabled": True, "api_key": "rotated-secret"}
    stale_patch = {"enabled": False}
    stale_stage_ready = asyncio.Event()
    release_stale_stage = asyncio.Event()

    class CoordinatedStorage(_CASStorage):
        def __init__(self):
            super().__init__()
            self._paused = False

        async def compare_and_swap_node(self, node_id, expected, new_node):
            pending = new_node.properties.get("pending_config")
            if (
                not self._paused
                and pending == {"enabled": False, "api_key": "first-secret"}
            ):
                self._paused = True
                stale_stage_ready.set()
                await release_stale_stage.wait()
            return await super().compare_and_swap_node(node_id, expected, new_node)

    storage = CoordinatedStorage()
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    stale_agent = Mock(did=_TEST_AGENT_DID, features={})
    stale_agent.storage = storage
    stale_agent.storage_path = str(tmp_path / "stale" / "kestrel_prime.db")
    writer_agent = Mock(did=_TEST_AGENT_DID, features={})
    writer_agent.storage = storage
    writer_agent.storage_path = str(tmp_path / "writer" / "kestrel_prime.db")
    stale = ProxyFeature(stale_agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    writer = ProxyFeature(writer_agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    stale_update = None
    try:
        # The stale replica has legitimately loaded an empty config before any
        # credential exists.  Hosted ``get_config`` must subsequently re-read
        # storage, not return this cached empty dict forever.
        await stale.initialize()
        await writer.initialize()
        assert await stale.get_config() == {}

        await writer.set_config(first_secret_config)
        assert await stale.get_config() == first_secret_config

        stale_update = asyncio.create_task(
            stale.set_config_with_secret_preservation(
                stale_patch,
                {"api_key"},
                lambda effective: None,
            )
        )
        await asyncio.wait_for(stale_stage_ready.wait(), timeout=1)

        # Rotate the credential after stale preservation chose the first value
        # but before its stage predicate commits.  The stale CAS must lose,
        # re-read, and preserve this winner instead.
        await writer.set_config(rotated_config)
        release_stale_stage.set()
        await stale_update

        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == {"enabled": False, "api_key": "rotated-secret"}
        assert "pending_config" not in properties
    finally:
        release_stale_stage.set()
        if stale_update is not None and not stale_update.done():
            await stale_update
        await writer.shutdown()
        await stale.shutdown()


@pytest.mark.asyncio
async def test_replica_get_does_not_mask_stale_child_before_next_patch(
    monkeypatch, tmp_path
):
    """A durable GET cannot make a stale local child skip reconciliation.

    Replica two is still running ``old_config`` when replica one promotes
    ``winner_config``.  Its GET must return the durable winner, but a later
    PATCH must first replace the locally stale child before it invokes the
    negotiated hook.
    """

    old_config = {"enabled": True, "revision": "old"}
    winner_config = {"enabled": True, "revision": "winner"}
    next_config = {"enabled": False, "revision": "next"}
    storage = _CASStorage()
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    winner_agent = Mock(did=_TEST_AGENT_DID, features={})
    winner_agent.storage = storage
    winner_agent.storage_path = str(tmp_path / "winner" / "kestrel_prime.db")
    stale_agent = Mock(did=_TEST_AGENT_DID, features={})
    stale_agent.storage = storage
    stale_agent.storage_path = str(tmp_path / "stale" / "kestrel_prime.db")
    stale_clients = []

    class ConfigAwareClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared = []

        async def prepare_config_transition(self, config):
            # This hook owns resources created by the child config supplied at
            # initialize.  It must never receive the next PATCH while those
            # resources still describe the old durable generation.
            assert self.kwargs["config"] == winner_config
            self.prepared.append(dict(config))
            return ConfigTransitionResult.applied()

    def stale_factory(**kwargs):
        client = ConfigAwareClient(**kwargs)
        stale_clients.append(client)
        return client

    winner = ProxyFeature(
        winner_agent, _cfg_runtime(), client_factory=FakeIsolatedClient
    )
    stale = ProxyFeature(stale_agent, _cfg_runtime(), client_factory=stale_factory)
    try:
        await winner.persist_config(old_config)
        await winner.initialize()
        await stale.initialize()
        stale_child = stale_clients[0]

        await winner.set_config(winner_config)
        assert await stale.get_config() == winner_config
        # GET reports durable state without rewriting the identity of the
        # child that is actually still running old_config.
        assert stale._host_config == old_config

        await stale.set_config(next_config)

        assert stale_child.stopped is True
        assert stale._client is stale_clients[1]
        assert stale_clients[1].kwargs["config"] == winner_config
        assert stale_clients[1].prepared == [next_config]
        assert (
            storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"]
            == next_config
        )
    finally:
        await stale.shutdown()
        await winner.shutdown()


@pytest.mark.asyncio
async def test_stale_equal_config_reconciles_only_after_external_ingress_fence(
    monkeypatch, tmp_path
):
    """A stale ``B -> B`` update fences old polling before replacing child A."""

    old_config = {"enabled": True, "revision": "old"}
    winner_config = {"enabled": True, "revision": "winner"}
    storage = _CASStorage()
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    winner_agent = Mock(did=_TEST_AGENT_DID, features={})
    winner_agent.storage = storage
    winner_agent.storage_path = str(tmp_path / "winner" / "kestrel_prime.db")
    stale_agent = Mock(did=_TEST_AGENT_DID, features={})
    stale_agent.storage = storage
    stale_agent.storage_path = str(tmp_path / "stale" / "kestrel_prime.db")
    stale_clients = []

    class FencedStaleClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.quiesced = False
            self.ingress_calls = []
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if name == "external-ingress-quiesce":
                assert stale._traffic_gate.closed is False
                self.quiesced = True
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            raise AssertionError("a retired child must not be resumed")

        async def stop(self):
            if self is stale_clients[0]:
                assert self.quiesced is True
                assert stale._traffic_gate.closed is True
            await super().stop()

    def stale_factory(**kwargs):
        client = FencedStaleClient(**kwargs)
        stale_clients.append(client)
        return client

    winner = ProxyFeature(
        winner_agent, _cfg_runtime(), client_factory=FakeIsolatedClient
    )
    stale = ProxyFeature(stale_agent, _cfg_runtime(), client_factory=stale_factory)
    try:
        await winner.persist_config(old_config)
        await winner.initialize()
        await stale.initialize()
        stale_child = stale_clients[0]

        await winner.set_config(winner_config)
        # Stale's local applied identity remains A even though its stage reads
        # durable B, which is exactly the prior no-op fast-path hole.
        assert stale._host_config == old_config
        await stale.set_config(winner_config)

        assert stale_child.stopped is True
        assert [call[0] for call in stale_child.ingress_calls] == [
            "external-ingress-quiesce"
        ]
        assert stale._client is stale_clients[1]
        assert stale_clients[1].kwargs["config"] == winner_config
        assert stale._host_config == winner_config
    finally:
        await stale.shutdown()
        await winner.shutdown()


@pytest.mark.asyncio
async def test_ambiguous_promotion_reread_quarantine_cannot_publish_recovery_child(
    monkeypatch, tmp_path
):
    """Storage recovery after quarantine stays terminal until initialize()."""

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": False, "revision": "next"}

    class PromotionReadFailureStorage(_CASStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0
            self.fail_next_read = False

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            # The stage and pre-hook renewal succeed.  The promotion response
            # is ambiguous, and its required durable re-read then fails,
            # latching quarantine.
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                self.fail_next_read = True
                raise OSError("promotion transport failed")
            return await super().compare_and_swap_node(node_id, expected, new_node)

        async def get_node(self, node_id):
            if self.fail_next_read:
                self.fail_next_read = False
                raise OSError("promotion reread unavailable")
            return await super().get_node(node_id)

    class LiveApplyClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            return ConfigTransitionResult.applied()

    storage = PromotionReadFailureStorage()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            LiveApplyClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        old_child = clients[0]

        with pytest.raises(RuntimeError, match="could not reconcile durable config"):
            await feature.set_config(next_config)

        # The follow-up cleanup can read storage again and clear its pending
        # generation, but it must not force-start a replacement behind the
        # terminal gate that the ambiguous reread already sealed.
        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert old_child.stopped is True
        assert clients == [old_child]
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._supervision_task is None
        assert feature._stopping is True
        assert feature._traffic_gate.sealed is True

        # Only an explicit enable-cycle initialization may create a new child.
        await feature.initialize()
        assert len(clients) == 2
        assert clients[1].kwargs["config"] == old_config
        assert feature._client is clients[1]
        assert feature._traffic_gate.sealed is False
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_cancellation_promotes_generation_before_starting_replacement(
    monkeypatch, tmp_path
):
    """A fenced cancellation publishes only a durably promoted next child."""

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": False, "revision": "next"}
    hook_started = asyncio.Event()

    class FencedCancellationClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            hook_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.replacement_required = True
                raise

    storage = _CASStorage()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FencedCancellationClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        update.cancel()

        async def _wait_for_fence():
            while not clients[0].replacement_required:
                await asyncio.sleep(0)

        await asyncio.wait_for(_wait_for_fence(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await update

        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == next_config
        assert feature._client is clients[1]
    finally:
        if update is not None and not update.done():
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
        await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_recovery_old_client_stop_error_cleans_owned_pending_and_quarantines(
    monkeypatch, tmp_path
):
    """A failed old-child stop never leaves its generation wedged in storage."""

    old_config = {"enabled": True}
    next_config = {"enabled": False}

    class StopFailingFencedClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            self.replacement_required = True
            raise ConfigTransitionError("transition transport failed")

        async def stop(self):
            self.stopped = True
            raise OSError("old child would not stop")

    storage = _CASStorage()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=StopFailingFencedClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        with pytest.raises(OSError, match="old child would not stop"):
            await feature.set_config(next_config)

        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert feature._client is None
        assert feature._stopping is True
        assert feature._terminal_retirement_clients
        assert feature._traffic_gate.sealed is True
    finally:
        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ):
            await feature.shutdown()


@pytest.mark.asyncio
async def test_set_config_cancellation_while_gate_drains_reopens_before_staging(
    monkeypatch, tmp_path
):
    """Cancelling an admitted-call drain cannot wedge config traffic closed."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingToolClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            self.calls.append((name, args))
            active_started.set()
            await release_active.wait()
            return {"echo": args}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingToolClient)
    active = update = None
    try:
        await feature.persist_config({"enabled": True})
        await feature.initialize()
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)

        update = asyncio.create_task(feature.set_config({"enabled": False}))
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed

        update.cancel()
        await asyncio.sleep(0)
        assert not update.done()
        release_active.set()

        with pytest.raises(asyncio.CancelledError):
            await update
        assert (await active)["success"] is True
        assert feature._traffic_gate.closed is False
        assert feature._traffic_gate.sealed is False
        assert feature._reloading is False
        assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == {
            "enabled": True
        }
    finally:
        release_active.set()
        for task in (update, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_reload_cancellation_while_gate_drains_reopens_without_restart(
    monkeypatch, tmp_path
):
    """Reload owns and reopens its gate even when cancellation wins the drain."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingToolClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            self.calls.append((name, args))
            active_started.set()
            await release_active.wait()
            return {"echo": args}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingToolClient)
    active = reload_task = None
    try:
        await feature.initialize()
        client = feature._client
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)

        reload_task = asyncio.create_task(feature.reload())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed

        reload_task.cancel()
        await asyncio.sleep(0)
        assert not reload_task.done()
        release_active.set()

        with pytest.raises(asyncio.CancelledError):
            await reload_task
        assert (await active)["success"] is True
        assert client.stopped is False
        assert feature._traffic_gate.closed is False
        assert feature._reloading is False
    finally:
        release_active.set()
        for task in (reload_task, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_second_cancellation_during_gate_reopen_waits_for_final_boundary(
    monkeypatch, tmp_path
):
    """A second cancellation cannot strand a reopened finite gate behind reload."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()
    reopen_started = asyncio.Event()
    release_reopen = asyncio.Event()

    class BlockingToolClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            active_started.set()
            await release_active.wait()
            return {"echo": args}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingToolClient)
    active = update = None
    try:
        await feature.initialize()
        original_reopen = feature._traffic_gate.reopen

        async def blocking_reopen():
            reopen_started.set()
            await release_reopen.wait()
            await original_reopen()

        monkeypatch.setattr(feature._traffic_gate, "reopen", blocking_reopen)
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)
        update = asyncio.create_task(feature.set_config({"enabled": False}))
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)

        update.cancel()
        release_active.set()
        await asyncio.wait_for(reopen_started.wait(), timeout=1)
        update.cancel()
        await asyncio.sleep(0)
        assert not update.done()

        release_reopen.set()
        with pytest.raises(asyncio.CancelledError):
            await update
        assert feature._traffic_gate.closed is False
        assert feature._traffic_gate.sealed is False
        assert feature._reloading is False
    finally:
        release_active.set()
        release_reopen.set()
        for task in (update, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_quarantine_wakes_tool_and_channel_waiters_without_rpc(
    monkeypatch, tmp_path
):
    """Sealing quarantine releases finite waiters with no post-terminal effect."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            if name == "hold":
                active_started.set()
                await release_active.wait()
            return {"ok": True}

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingChannelClient)
    active = close = tool_waiter = channel_waiter = quarantine = None
    try:
        await feature.initialize()
        client = feature._client
        adapter = channel_feature.registry.get("whatsapp")
        assert adapter is not None
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)
        close = asyncio.create_task(feature._close_traffic_gate())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)

        tool_waiter = asyncio.create_task(feature.call_isolated_tool("ping", {}))
        channel_waiter = asyncio.create_task(adapter.send_message(to="+1", content="hello"))
        await asyncio.sleep(0)
        assert not tool_waiter.done()
        assert not channel_waiter.done()

        quarantine = asyncio.create_task(feature._quarantine_unreconciled_client())
        await asyncio.sleep(0)
        tool_result = await tool_waiter
        channel_result = await channel_waiter
        assert tool_result["error"] == "isolated feature traffic is unavailable"
        assert channel_result.status.value == "failure"
        assert channel_result.error == "isolated feature traffic is unavailable"
        assert client.calls == [("hold", {})]
        assert feature._traffic_gate.sealed is True
        assert not quarantine.done()

        release_active.set()
        assert (await active)["success"] is True
        await quarantine
        with pytest.raises(RuntimeError, match="isolated feature traffic is unavailable"):
            await close
    finally:
        release_active.set()
        for task in (quarantine, close, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, RuntimeError):
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_shutdown_wakes_tool_waiter(monkeypatch, tmp_path):
    """Shutdown seals admission before draining an already admitted RPC."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingToolClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            self.calls.append((name, args))
            if name == "hold":
                active_started.set()
                await release_active.wait()
            return {"echo": args}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingToolClient)
    active = close = waiter = shutdown_task = None
    try:
        await feature.initialize()
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)
        close = asyncio.create_task(feature._close_traffic_gate())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)

        waiter = asyncio.create_task(feature.call_isolated_tool("ping", {}))
        await asyncio.sleep(0)
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        result = await waiter
        assert result["error"] == "isolated feature traffic is unavailable"
        assert feature._traffic_gate.sealed is True
        assert not shutdown_task.done()

        execution = SchedulerExecution(
            id="terminal-execution",
            schedule_id="terminal-schedule",
            agent_id="agent-1",
            task_name="ping",
            args={},
            scheduled_for="2026-07-26T00:00:00+00:00",
            idempotency_key="terminal-effect-key",
            attempt=1,
            owner="runner-1",
        )
        token = _current_execution.set(_SchedulerExecutionScope(execution))
        try:
            with pytest.raises(
                SchedulerTerminalAdmissionError,
                match="isolated feature traffic is unavailable",
            ):
                await feature.call_isolated_tool("ping", {})
        finally:
            _current_execution.reset(token)

        release_active.set()
        await active
        await shutdown_task
        with pytest.raises(RuntimeError, match="isolated feature traffic is unavailable"):
            await close
    finally:
        release_active.set()
        for task in (shutdown_task, close, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, RuntimeError):
                    pass
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_repeated_cancellation_during_terminal_seal_finishes_gate_boundary(
    monkeypatch, tmp_path
):
    """Repeated shutdown cancellation cannot leave a terminal waiter asleep."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingToolClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            self.calls.append((name, args))
            if name == "hold":
                active_started.set()
                await release_active.wait()
            return {"echo": args}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingToolClient)
    active = close = waiter = shutdown_task = None
    try:
        await feature.initialize()
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)
        close = asyncio.create_task(feature._close_traffic_gate())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)

        waiter = asyncio.create_task(feature.call_isolated_tool("ping", {}))
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        assert (await waiter)["error"] == "isolated feature traffic is unavailable"
        assert feature._traffic_gate.sealed is True

        shutdown_task.cancel("first terminal cleanup cancellation")
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        shutdown_task.cancel("later terminal cleanup cancellation")
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        release_active.set()
        await active
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await shutdown_task
        assert cancelled.value.args == ("first terminal cleanup cancellation",)
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 0
        with pytest.raises(RuntimeError, match="isolated feature traffic is unavailable"):
            await close
    finally:
        release_active.set()
        for task in (shutdown_task, close, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, RuntimeError):
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_inbound_event_is_dropped_without_callback_error(monkeypatch, tmp_path):
    """A terminal SDK callback is deliberately ignored rather than propagated."""

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    delivered = []

    async def record_delivery(event):
        delivered.append(event)

    try:
        await feature.initialize()
        feature._handle_event_admitted = record_delivery  # type: ignore[method-assign]
        await feature.shutdown()

        await feature._handle_event({"type": "channel.inbound", "payload": {}})
        assert delivered == []
    finally:
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_reinitialize_after_terminal_quarantine_reopens_only_after_child_init(
    monkeypatch, tmp_path
):
    """A durable re-enable keeps terminal admission sealed until its child is ready."""

    second_start = asyncio.Event()
    release_second_start = asyncio.Event()
    clients = []

    class DelayedReinitializeClient(FakeIsolatedClient):
        async def start(self):
            self.started = True
            if len(clients) == 2:
                second_start.set()
                await release_second_start.wait()

    def client_factory(**kwargs):
        client = DelayedReinitializeClient(**kwargs)
        clients.append(client)
        return client

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    initialize_task = None
    try:
        await feature.persist_config({"enabled": True})
        await feature.initialize()
        await feature._quarantine_unreconciled_client()
        assert feature._traffic_gate.sealed is True

        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.wait_for(second_start.wait(), timeout=1)
        assert feature._traffic_gate.sealed is True
        blocked = await feature.call_isolated_tool("ping", {"message": "blocked"})
        assert blocked["error"] == "isolated feature traffic is unavailable"
        assert clients[1].calls == []

        release_second_start.set()
        await initialize_task
        assert feature._traffic_gate.sealed is False
        result = await feature.call_isolated_tool("ping", {"message": "ready"})
        assert result["success"] is True
        assert clients[1].calls == [("ping", {"message": "ready"})]
    finally:
        release_second_start.set()
        if initialize_task is not None and not initialize_task.done():
            initialize_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await initialize_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_soft_disabled_terminal_proxy_persists_repaired_config_until_reenable(
    monkeypatch, tmp_path
):
    """A disabled proxy can rotate config without reviving its sealed child."""

    old_config = {"enabled": True, "token": "old-token"}
    repaired_config = {"enabled": True, "token": "rotated-token"}
    clients = []

    class TransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared = []

        async def prepare_config_transition(self, config):
            self.prepared.append(dict(config))
            return ConfigTransitionResult.applied()

    def client_factory(**kwargs):
        client = TransitionClient(**kwargs)
        clients.append(client)
        return client

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        old_child = clients[0]
        await feature.shutdown()

        assert old_child.stopped is True
        assert feature._client is None
        assert feature._supervision_task is None
        assert feature._stopping is True
        assert feature._traffic_gate.sealed is True

        await feature.set_config(repaired_config)

        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == repaired_config
        assert "pending_config" not in properties
        # A config repair is storage-only while disabled: no transition hook,
        # child, supervision task, or admission reset is permitted.
        assert old_child.prepared == []
        assert clients == [old_child]
        assert feature._client is None
        assert feature._supervision_task is None
        assert feature._stopping is True
        assert feature._traffic_gate.sealed is True

        await feature.initialize()

        assert len(clients) == 2
        assert clients[1].kwargs["config"] == repaired_config
        assert feature._client is clients[1]
        assert feature._supervision_task is not None
        assert feature._stopping is False
        assert feature._traffic_gate.sealed is False
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cancellation_waiting_for_lifecycle_lock_finishes_teardown(
    monkeypatch, tmp_path
):
    """Caller cancellation waits through lock ownership and final retirement."""

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=ChannelClient)
    shutdown_task = None
    lock_held = False
    try:
        await feature.initialize()
        client = feature._client
        adapter = channel_feature.registry.get("whatsapp")
        assert client is not None
        assert adapter is not None

        await feature._reload_lock.acquire()
        lock_held = True
        shutdown_task = asyncio.create_task(feature.shutdown())
        for _ in range(100):
            if feature._traffic_gate.sealed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )
        assert not shutdown_task.done()

        shutdown_task.cancel()
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        feature._reload_lock.release()
        lock_held = False

        with pytest.raises(asyncio.CancelledError):
            await shutdown_task
        assert client.stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
        assert feature._stopping is True
    finally:
        if lock_held:
            feature._reload_lock.release()
        if shutdown_task is not None and not shutdown_task.done():
            shutdown_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await shutdown_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_quarantine_repeated_cancellation_waits_for_lock_and_retires_child(
    monkeypatch, tmp_path
):
    """Terminal quarantine cannot leave its cleanup task behind the lifecycle lock."""

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=ChannelClient)
    quarantine_task = None
    lock_held = False
    try:
        await feature.initialize()
        client = feature._client
        assert client is not None
        assert channel_feature.registry.get("whatsapp") is not None

        await feature._reload_lock.acquire()
        lock_held = True
        quarantine_task = asyncio.create_task(feature._quarantine_unreconciled_client())
        for _ in range(100):
            if feature._traffic_gate.sealed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )

        quarantine_task.cancel()
        quarantine_task.cancel()
        await asyncio.sleep(0)
        assert not quarantine_task.done()
        feature._reload_lock.release()
        lock_held = False

        with pytest.raises(asyncio.CancelledError):
            await quarantine_task
        assert client.stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
    finally:
        if lock_held:
            feature._reload_lock.release()
        if quarantine_task is not None and not quarantine_task.done():
            quarantine_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await quarantine_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_initialize_cancellation_after_gate_reset_unpublishes_and_retires_child(
    monkeypatch, tmp_path
):
    """A cancelled post-connect initialize has no tool, channel, or event path."""

    reset_started = asyncio.Event()
    release_reset = asyncio.Event()
    delivered = []

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    original_reset = feature._traffic_gate.reset_and_reopen

    async def blocking_reset():
        reset_started.set()
        await release_reset.wait()
        await original_reset()

    async def record_delivery(event):
        delivered.append(event)

    monkeypatch.setattr(feature._traffic_gate, "reset_and_reopen", blocking_reset)
    feature._handle_event_admitted = record_delivery  # type: ignore[method-assign]
    initialize_task = None
    try:
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.wait_for(reset_started.wait(), timeout=1)
        client = clients[0]
        assert feature._client is client
        assert channel_feature.registry.get("whatsapp") is not None

        initialize_task.cancel()
        release_reset.set()
        with pytest.raises(asyncio.CancelledError):
            await initialize_task

        assert client.stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )
        assert client.event_handler is not None
        await client.event_handler({"type": "channel.inbound", "payload": {}})
        assert delivered == []
    finally:
        release_reset.set()
        if initialize_task is not None and not initialize_task.done():
            initialize_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await initialize_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_during_event_registration_keeps_initialize_terminal(
    monkeypatch, tmp_path
):
    """A terminal latch during post-publication registration cannot revive the proxy."""

    registration_started = asyncio.Event()
    release_registration = asyncio.Event()
    clients = []

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

        async def set_event_handler(self, handler):
            if self is clients[0]:
                registration_started.set()
                await release_registration.wait()
            self.event_handler = handler

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs)
        clients.append(client)
        return client

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    initialize_task = None
    shutdown_task = None
    try:
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.wait_for(registration_started.wait(), timeout=1)
        first_child = clients[0]
        assert feature._client is first_child
        assert feature.get_tools()
        assert channel_feature.registry.get("whatsapp") is not None

        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        assert feature._terminal_lifecycle_latched is True

        release_registration.set()
        with pytest.raises(RuntimeError, match="terminal lifecycle is latched"):
            await asyncio.wait_for(initialize_task, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)

        assert first_child.stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._supervision_task is None
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True

        # Cleanup has completed, so a later explicit initialize owns a new
        # lifecycle and may open admission again.
        await feature.initialize()
        assert len(clients) == 2
        assert feature._client is clients[1]
        assert feature._supervision_task is not None
        assert feature._traffic_gate.sealed is False
    finally:
        release_registration.set()
        for task in (initialize_task, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_registration_failure_after_shutdown_retirement_does_not_stop_twice(
    monkeypatch, tmp_path
):
    """Registration cleanup never reclaims a client terminal cleanup already owns."""

    registration_started = asyncio.Event()
    release_registration = asyncio.Event()
    stop_finished = asyncio.Event()
    clients = []

    class RegistrationFailureClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def set_event_handler(self, _handler):
            registration_started.set()
            await release_registration.wait()
            raise RuntimeError("event registration failed")

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True
            stop_finished.set()

    def client_factory(**kwargs):
        client = RegistrationFailureClient(**kwargs)
        clients.append(client)
        return client

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    initialize_task = shutdown_task = None
    try:
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.wait_for(registration_started.wait(), timeout=1)
        client = clients[0]

        # Shutdown wins publication ownership while registration remains
        # blocked.  It has retired the exact client, but must wait for the
        # initializer's reload lock before it can complete its final fence.
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.wait_for(stop_finished.wait(), timeout=1)
        assert feature._client is None
        assert not shutdown_task.done()

        release_registration.set()
        with pytest.raises(RuntimeError, match="event registration failed"):
            await asyncio.wait_for(initialize_task, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)

        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_cleanup_task is None
        assert feature._traffic_gate.sealed is True
        assert feature._supervision_task is None
    finally:
        release_registration.set()
        for task in (initialize_task, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_initialize_gate_reset_failure_unpublishes_and_retires_child(monkeypatch, tmp_path):
    """A post-connect gate failure follows the same terminal cleanup path."""

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)

    async def failing_reset():
        raise RuntimeError("gate reset failed")

    monkeypatch.setattr(feature._traffic_gate, "reset_and_reopen", failing_reset)
    try:
        with pytest.raises(RuntimeError, match="gate reset failed"):
            await feature.initialize()

        assert clients[0].stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_reload_candidate_start_failure_quarantines_stopped_old_child(
    monkeypatch, tmp_path
):
    """Direct reload never reopens old tools when candidate startup fails."""

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    class StartFailingClient(ChannelClient):
        async def start(self):
            self.started = True
            raise RuntimeError("candidate start failed")

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs) if not clients else StartFailingClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.initialize()
        old_client = clients[0]
        assert channel_feature.registry.get("whatsapp") is not None

        with pytest.raises(RuntimeError, match="candidate start failed"):
            await feature.reload()

        assert old_client.stopped is True
        assert clients[1].stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_reload_cancellation_after_old_stop_quarantines_publication(
    monkeypatch, tmp_path
):
    """Reload cancellation after old retirement cannot reopen stale traffic."""

    candidate_started = asyncio.Event()
    release_candidate = asyncio.Event()

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    class BlockingCandidate(ChannelClient):
        async def start(self):
            self.started = True
            candidate_started.set()
            await release_candidate.wait()

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs) if not clients else BlockingCandidate(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    reload_task = None
    try:
        await feature.initialize()
        old_client = clients[0]
        reload_task = asyncio.create_task(feature.reload())
        await asyncio.wait_for(candidate_started.wait(), timeout=1)
        assert old_client.stopped is True
        assert feature._client is None

        reload_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reload_task

        assert clients[1].stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )
    finally:
        release_candidate.set()
        if reload_task is not None and not reload_task.done():
            reload_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reload_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_channel_inbound_acknowledges_exact_source_only_after_host_delivery(
    monkeypatch, tmp_path
):
    """A notification's producer waits for post-handler private acknowledgement."""

    dedupe_key = "telegram:v2:bot:42:update:101"
    delivery_started = asyncio.Event()
    release_delivery = asyncio.Event()
    acknowledgement_started = asyncio.Event()
    release_acknowledgement = asyncio.Event()
    delivered = []

    class ChannelFeature:
        async def handle_inbound(self, message):
            delivery_started.set()
            await release_delivery.wait()
            delivered.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class AcknowledgingClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            acknowledgement_started.set()
            await release_acknowledgement.wait()
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=AcknowledgingClient)
    event_task = None
    try:
        await feature.initialize()
        client = feature._client
        event_task = asyncio.create_task(
            client.event_handler(
                {
                    "type": "channel.inbound",
                    "payload": {
                        "message": {
                            "channel_type": "telegram",
                            "direction": "inbound",
                            "sender": "555",
                            "recipient": "42",
                            "content": "hello",
                            "id": dedupe_key,
                            "metadata": {"dedupe_key": dedupe_key},
                        },
                        "_host_ingress_ack": {
                            "name": "telegram-polling-ack",
                            "payload": {"dedupe_key": dedupe_key},
                        },
                    },
                }
            )
        )
        await asyncio.wait_for(delivery_started.wait(), timeout=1)
        assert not acknowledgement_started.is_set()

        release_delivery.set()
        await asyncio.wait_for(event_task, timeout=1)
        await asyncio.wait_for(acknowledgement_started.wait(), timeout=1)
        assert delivered == [dedupe_key]
        assert client.acknowledgements == [
            ("telegram-polling-ack", {"dedupe_key": dedupe_key})
        ]
        # The acknowledgement remains independently in-flight so an SDK event
        # reader can return to consume its JSON-RPC response instead of waiting
        # on that same response inline.
        assert feature._event_ack_tasks
        release_acknowledgement.set()
        for _ in range(20):
            if not feature._event_ack_tasks:
                break
            await asyncio.sleep(0)
        assert not feature._event_ack_tasks
    finally:
        release_delivery.set()
        release_acknowledgement.set()
        if event_task is not None and not event_task.done():
            await event_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_current_replacement_acknowledged_event_waits_for_reopened_gate(
    monkeypatch, tmp_path
):
    """A replacement poller cannot lose its first update to Core's closed gate."""

    dedupe_key = "telegram:v2:bot:42:update:102"
    delivered = []
    acknowledged = asyncio.Event()

    class ChannelFeature:
        async def handle_inbound(self, message):
            delivered.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class AckClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )

        async def call_host_ingress(self, name, payload=None):
            assert (name, payload) == (
                "telegram-polling-ack",
                {"dedupe_key": dedupe_key},
            )
            acknowledged.set()
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=AckClient)
    try:
        await feature.initialize()
        await feature._close_traffic_gate()
        await feature._client.event_handler(
            {
                "type": "channel.inbound",
                "payload": {
                    "message": {
                        "channel_type": "telegram",
                        "direction": "inbound",
                        "sender": "555",
                        "recipient": "42",
                        "content": "first after replacement",
                        "id": dedupe_key,
                        "metadata": {"dedupe_key": dedupe_key},
                    },
                    "_host_ingress_ack": {
                        "name": "telegram-polling-ack",
                        "payload": {"dedupe_key": dedupe_key},
                    },
                },
            }
        )
        assert delivered == []
        assert not acknowledged.is_set()

        await feature._reopen_traffic_gate()
        await asyncio.wait_for(acknowledged.wait(), timeout=1)
        assert delivered == [dedupe_key]
    finally:
        await feature.shutdown()


def _acknowledged_telegram_event(dedupe_key: str) -> dict:
    """Build the exact private ingress envelope shared by polling and hosted dedupe."""

    return {
        "type": "channel.inbound",
        "payload": {
            "message": {
                "channel_type": "telegram",
                "direction": "inbound",
                "sender": "555",
                "recipient": "42",
                "content": "hello",
                "id": dedupe_key,
                "metadata": {"dedupe_key": dedupe_key},
            },
            "_host_ingress_ack": {
                "name": "telegram-polling-ack",
                "payload": {"dedupe_key": dedupe_key},
            },
        },
    }


def _retryable_telegram_event(dedupe_key: str) -> dict:
    """Build a polling envelope whose provider callback can be NACKed."""

    event = _acknowledged_telegram_event(dedupe_key)
    event["payload"]["_host_ingress_retry"] = {
        "name": "telegram-polling-nack",
        "payload": {"dedupe_key": dedupe_key},
    }
    return event


@pytest.mark.asyncio
async def test_inbound_reader_returns_before_cognition_and_nacks_retryable_result(
    monkeypatch, tmp_path
):
    """Cognition can make an outbound channel call without pinning the SDK reader."""

    dedupe_key = "telegram:v2:bot:42:update:reader-free"
    cognition_started = asyncio.Event()
    release_cognition = asyncio.Event()
    retry_completed = asyncio.Event()

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, _message):
            cognition_started.set()
            # This is the same data-plane route cognition uses for
            # ``channels_send``. It must be able to await the child RPC after
            # the notification reader has already returned to its stream.
            await feature._channel_adapter.send_message("555", "reply during cognition")
            await release_cognition.wait()
            return SimpleNamespace(durably_admitted=False)

    class RetryClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "telegram",
                    "send_tool": "telegram_send",
                }
            }

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack", "telegram-polling-nack")
            )
            self.completions = []

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            return {"status": "ok", "data": {"message_id": "reply-1"}}

        async def call_host_ingress(self, name, payload=None):
            self.completions.append((name, payload))
            retry_completed.set()
            return {"status": "ok", "http_status": 200, "state": "retrying"}

    channel_feature = ChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=RetryClient)
    try:
        await feature.initialize()
        client = feature._client
        # The notification invocation returns before even a deliberately
        # blocked cognition turn, freeing its serial reader for outbound RPC
        # responses and the later provider completion.
        await asyncio.wait_for(
            client.event_handler(_retryable_telegram_event(dedupe_key)), timeout=0.1
        )
        await asyncio.wait_for(cognition_started.wait(), timeout=1)
        assert client.calls == [
            ("telegram_send", {"to": "555", "message": "reply during cognition"})
        ]
        assert client.completions == []

        release_cognition.set()
        await asyncio.wait_for(retry_completed.wait(), timeout=1)
        assert client.completions == [
            ("telegram-polling-nack", {"dedupe_key": dedupe_key})
        ]
    finally:
        release_cognition.set()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_ack_bearing_ingress_never_acknowledges_legacy_router_fallback(monkeypatch, tmp_path):
    """A legacy/non-durable ChannelFeature result leaves the provider cursor intact."""

    dedupe_key = "telegram:v2:bot:42:update:201"

    class LegacyChannelFeature:
        async def handle_inbound(self, message):
            return SimpleNamespace(durably_admitted=False)

    class AckClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": LegacyChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=AckClient)
    try:
        await feature.initialize()
        client = feature._client
        await client.event_handler(_acknowledged_telegram_event(dedupe_key))
        await asyncio.sleep(0)
        assert client.acknowledgements == []
        assert feature._event_ack_tasks == set()
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_retired_source_event_is_rejected_under_traffic_admission(monkeypatch, tmp_path):
    """A late callback from a retired child cannot route or ACK after replacement."""

    dedupe_key = "telegram:v2:bot:42:update:202"
    delivered = []

    class ChannelFeature:
        async def handle_inbound(self, message):
            delivered.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class AckClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=AckClient)
    old_client = None
    try:
        await feature.initialize()
        old_client = feature._client
        # The callback closure still carries old_client, while publication has
        # moved to the replacement. Identity is checked *inside* admission.
        replacement = AckClient()
        feature._client = replacement
        await old_client.event_handler(_acknowledged_telegram_event(dedupe_key))
        await asyncio.sleep(0)
        assert delivered == []
        assert old_client.acknowledgements == []
        assert replacement.acknowledgements == []
    finally:
        if old_client is not None:
            await old_client.stop()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_deferred_acknowledged_ingress_uses_detached_k1_snapshot(monkeypatch, tmp_path):
    """Sol regression: mutating a queued k1 envelope to k2 cannot redirect delivery/ACK."""

    k1 = "telegram:v2:bot:42:update:203"
    k2 = "telegram:v2:bot:42:update:204"
    delivered = []
    acknowledged = []

    class ChannelFeature:
        async def handle_inbound(self, message):
            delivered.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class AckClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )

        async def call_host_ingress(self, name, payload=None):
            acknowledged.append((name, payload))
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=AckClient)
    try:
        await feature.initialize()
        event = _acknowledged_telegram_event(k1)
        await feature._close_traffic_gate()
        await feature._client.event_handler(event)
        # This is the caller-owned graph Sol mutated after Core deferred it.
        event["payload"]["message"]["id"] = k2
        event["payload"]["message"]["metadata"]["dedupe_key"] = k2
        event["payload"]["_host_ingress_ack"]["payload"]["dedupe_key"] = k2

        await feature._reopen_traffic_gate()
        for _ in range(20):
            if acknowledged:
                break
            await asyncio.sleep(0)
        assert delivered == [k1]
        assert acknowledged == [("telegram-polling-ack", {"dedupe_key": k1})]
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_ingress_workers_are_bounded_to_one_per_source_under_1000_events(monkeypatch, tmp_path):
    """Sol regression: 1,000 callbacks cannot create a host-memory route queue."""

    release_ack = asyncio.Event()
    ack_started = asyncio.Event()
    delivered = []

    class ChannelFeature:
        async def handle_inbound(self, message):
            delivered.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class SlowAckClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            ack_started.set()
            await release_ack.wait()
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=SlowAckClient)
    try:
        await feature.initialize()
        client = feature._client
        for update_id in range(1_000):
            await client.event_handler(
                _acknowledged_telegram_event(f"telegram:v2:bot:42:update:{1_000 + update_id}")
            )
        await asyncio.wait_for(ack_started.wait(), timeout=1)
        # The serial provider must retain every later update while the first
        # detached route/ACK owns the source. Core intentionally does not turn
        # those notifications into 999 unbounded cognition tasks.
        assert len(delivered) == 1
        assert len(feature._event_ingress_tasks) == 0
        assert len(feature._event_ack_tasks) == 1
        assert len(client.acknowledgements) == 1
        release_ack.set()
    finally:
        release_ack.set()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_rejected_ack_retries_then_terminally_retires_exact_source(monkeypatch, tmp_path):
    """A rejected polling ACK retries idempotently, then leaves Telegram's cursor for restart."""

    monkeypatch.setattr(isolated_runtime, "_EVENT_INGRESS_ACK_BACKOFF", 0)
    dedupe_key = "telegram:v2:bot:42:update:205"

    class ChannelFeature:
        async def handle_inbound(self, message):
            return SimpleNamespace(durably_admitted=True)

    class RejectingAckClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            return {"status": "error", "http_status": 409, "error": "still pending"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=RejectingAckClient)
    try:
        await feature.initialize()
        client = feature._client
        await client.event_handler(_acknowledged_telegram_event(dedupe_key))
        for _ in range(50):
            if feature._terminal_lifecycle_latched and client.stopped:
                break
            await asyncio.sleep(0)
        assert client.acknowledgements == [
            ("telegram-polling-ack", {"dedupe_key": dedupe_key})
        ] * 3
        assert feature._terminal_lifecycle_latched is True
        assert client.stopped is True
        assert feature._client is None
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_hung_ack_times_out_with_bounded_retries_then_retires_source(
    monkeypatch, tmp_path
):
    """A never-completing ACK is fenced; it cannot retain an unbounded task swarm."""

    monkeypatch.setattr(isolated_runtime, "_EVENT_INGRESS_ACK_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_EVENT_INGRESS_ACK_CANCELLATION_GRACE", 0.01)
    monkeypatch.setattr(isolated_runtime, "_EVENT_INGRESS_ACK_BACKOFF", 0)
    dedupe_key = "telegram:v2:bot:42:update:208"

    class ChannelFeature:
        async def handle_inbound(self, message):
            return SimpleNamespace(durably_admitted=True)

    class HungAckClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            await asyncio.Event().wait()

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=HungAckClient)
    try:
        await feature.initialize()
        client = feature._client
        await client.event_handler(_acknowledged_telegram_event(dedupe_key))
        for _ in range(100):
            if feature._terminal_lifecycle_latched and client.stopped:
                break
            await asyncio.sleep(0.01)
        assert client.acknowledgements == [
            ("telegram-polling-ack", {"dedupe_key": dedupe_key})
        ] * 3
        assert feature._terminal_lifecycle_latched is True
        assert client.stopped is True
        assert feature._event_ack_tasks == set()
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_owner_joins_ack_and_deferred_workers_and_blocks_new_ack(
    monkeypatch, tmp_path
):
    """Terminal success is fenced until both detached ingress worker kinds settle."""

    ack_started = asyncio.Event()
    release_ack = asyncio.Event()

    class ChannelFeature:
        async def handle_inbound(self, message):
            return SimpleNamespace(durably_admitted=True)

    class SlowAckClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )

        async def call_host_ingress(self, name, payload=None):
            ack_started.set()
            await release_ack.wait()
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=SlowAckClient)
    try:
        await feature.initialize()
        client = feature._client
        first = _acknowledged_telegram_event("telegram:v2:bot:42:update:206")
        await client.event_handler(first)
        await asyncio.wait_for(ack_started.wait(), timeout=1)
        await feature._close_traffic_gate()
        await client.event_handler(_acknowledged_telegram_event("telegram:v2:bot:42:update:207"))
        for _ in range(20):
            if feature._deferred_acknowledged_event_tasks:
                break
            await asyncio.sleep(0)
        assert feature._event_ack_tasks
        assert feature._deferred_acknowledged_event_tasks

        feature._latch_terminal_lifecycle()
        release_ack.set()
        await feature._complete_terminal_cleanup()
        assert feature._event_ack_tasks == set()
        assert feature._deferred_acknowledged_event_tasks == set()
        request = feature._event_ingress_acknowledgement(first)
        assert request is not None
        feature._schedule_event_ingress_acknowledgement(client, request)
        assert feature._event_ack_tasks == set()
        assert client.stopped is True
    finally:
        release_ack.set()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_exact_lifecycle_rpc_bypasses_closed_data_plane_and_waits_for_drain(
    monkeypatch, tmp_path
):
    """A slow already-admitted send cannot consume Core's quiesce lifecycle budget."""

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    lifecycle_calls = []

    class SlowToolClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_tool(self, name, args):
            tool_started.set()
            await release_tool.wait()
            return {"echo": args}

        async def call_host_ingress(self, name, payload=None):
            lifecycle_calls.append((name, payload))
            state = "quiesced" if name == "external-ingress-quiesce" else "resumed"
            return {"status": "ok", "http_status": 200, "state": state}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=SlowToolClient)
    tool_task = None
    try:
        await feature.initialize()
        tool_task = asyncio.create_task(feature.call_isolated_tool("ping", {}))
        await asyncio.wait_for(tool_started.wait(), timeout=1)
        quiesce = feature._new_external_ingress_quiesce()
        assert quiesce is not None
        await feature._close_traffic_gate_admission()
        assert await asyncio.wait_for(feature._quiesce_external_ingress(quiesce), timeout=1) is None
        assert not tool_task.done()
        assert lifecycle_calls == [
            ("external-ingress-quiesce", {"transition_id": quiesce.transition_id})
        ]
        release_tool.set()
        await tool_task
        await feature._drain_traffic_gate()
        await feature._resume_external_ingress(quiesce)
        await feature._reopen_traffic_gate()
    finally:
        release_tool.set()
        if tool_task is not None and not tool_task.done():
            await tool_task
        await feature.shutdown()


class _HostIngressClient(FakeIsolatedClient):
    """SDK-client double that advertises a typed private ingress contract."""

    def __init__(self, *, ingress_capabilities=None, **kwargs):
        super().__init__(**kwargs)
        self.ingress_capabilities = ingress_capabilities
        self.ingress_calls = []

    @property
    def host_ingress_capabilities(self):
        return self.ingress_capabilities

    async def call_host_ingress(self, name, payload=None):
        self.ingress_calls.append((name, payload))
        return {"accepted": True}


async def _initialized_host_ingress_proxy(
    monkeypatch, tmp_path, client_factory, *, agent=None
):
    agent = agent or Mock(did=_TEST_AGENT_DID, features={})
    agent.features = getattr(agent, "features", {})
    storage = _CASStorage()
    storage.agent_id = agent.did
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.initialize()
    return feature, agent


def _walk_exception_chain(error):
    """Yield every exception recursively reachable through chaining links."""

    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend((current.__context__, current.__cause__))


def _assert_host_ingress_error_is_detached(error, *, secret, external=None):
    """A redacted public failure must retain no private exception graph."""

    chain = list(_walk_exception_chain(error))
    assert all(secret not in str(item) for item in chain)
    assert all(secret not in repr(item) for item in chain)
    assert all(type(item) is HostIngressError for item in chain)
    if external is not None:
        assert all(item is not external for item in chain)


def _assert_host_ingress_tracebacks_are_secret_free(error, *, secret):
    """Inspect every internal traceback frame on every chained error.

    ``raise ... from None`` only changes formatting; exception context and
    traceback locals remain directly reachable to debuggers/telemetry.  Keep
    this intentionally stronger than a message-redaction assertion.
    """

    forbidden_frames = {"_call_host_ingress_rpc", "_run_host_ingress"}
    for chained in _walk_exception_chain(error):
        traceback = chained.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if frame.f_code.co_filename == isolated_runtime.__file__:
                assert frame.f_code.co_name not in forbidden_frames
                assert secret not in repr(dict(frame.f_locals))
            traceback = traceback.tb_next


def _safe_traceback_references(value):
    """Yield bounded, non-executing references for traceback graph checks.

    Do not use ``repr()``, ordinary ``getattr()``, or application-defined
    properties here: the adversarial values under test can make each of those
    secret-bearing or executable.  Never expand modules, globals, or classes:
    those are ambient process state rather than a public-error retention path.
    Native runtime objects are different: a public traceback can retain an
    operation task, and a done task can retain its secret-bearing result.  Walk
    only their built-in, non-executing reference edges below.
    """

    value_type = type(value)
    if value_type is dict:
        yield from value.keys()
        yield from value.values()
        return
    if value_type in (list, tuple, set, frozenset):
        yield from value
        return

    if value_type in (asyncio.Task, asyncio.Future):
        # Use the base Future descriptors, never a facade/subclass override.
        # A completed operation owns both its result and exception internally;
        # either can otherwise retain private response/client data.
        try:
            done = asyncio.Future.done(value)
            cancelled = asyncio.Future.cancelled(value)
        except (AttributeError, TypeError):
            return
        if done and not cancelled:
            try:
                yield asyncio.Future.result(value)
            except BaseException as error:  # native task failure graph
                yield error
            try:
                error = asyncio.Future.exception(value)
            except BaseException as error:  # native task failure graph
                yield error
            else:
                if error is not None:
                    yield error
        for attribute in ("_fut_waiter", "_callbacks"):
            try:
                reference = object.__getattribute__(value, attribute)
            except (AttributeError, TypeError):
                continue
            if type(reference) in (list, tuple):
                yield from reference
            elif reference is not None:
                yield reference
        return

    if value_type is types.CoroutineType:
        for attribute in ("cr_frame", "cr_await"):
            try:
                reference = object.__getattribute__(value, attribute)
            except (AttributeError, TypeError):
                continue
            if reference is not None:
                yield reference
        return

    if value_type is types.GeneratorType:
        for attribute in ("gi_frame", "gi_yieldfrom"):
            try:
                reference = object.__getattribute__(value, attribute)
            except (AttributeError, TypeError):
                continue
            if reference is not None:
                yield reference
        return

    if value_type is types.AsyncGeneratorType:
        for attribute in ("ag_frame", "ag_await"):
            try:
                reference = object.__getattribute__(value, attribute)
            except (AttributeError, TypeError):
                continue
            if reference is not None:
                yield reference
        return

    if value_type is types.TracebackType:
        yield value.tb_frame
        if value.tb_next is not None:
            yield value.tb_next
        return

    if value_type is types.FrameType:
        # ``f_locals`` is a runtime mapping, not application code.  Do not
        # inspect ``f_globals``: modules/global graphs are intentionally leaves.
        namespace = value.f_locals
        if type(namespace) is dict:
            yield from namespace.values()
        return

    if issubclass(value_type, BaseException):
        for attribute in ("args", "__cause__", "__context__", "__traceback__"):
            try:
                reference = object.__getattribute__(value, attribute)
            except (AttributeError, TypeError):
                continue
            if reference is not None:
                yield reference

    if value_type is types.CellType:
        try:
            yield value.cell_contents
        except ValueError:
            pass
        return

    if value_type is types.FunctionType:
        closure = object.__getattribute__(value, "__closure__")
        if type(closure) is tuple:
            yield from closure
        return

    if value_type in (types.MethodType, types.BuiltinMethodType):
        try:
            yield object.__getattribute__(value, "__self__")
        except (AttributeError, TypeError):
            pass
        if value_type is types.MethodType:
            yield object.__getattribute__(value, "__func__")
        return

    if isinstance(
        value,
        (
            types.ModuleType,
            types.BuiltinFunctionType,
            type,
            types.CodeType,
        ),
    ):
        return

    # Calling object.__getattribute__ bypasses an instance's hostile
    # __getattribute__ implementation. A class that replaces __dict__ with a
    # property is skipped rather than executed.
    class_namespace = type.__getattribute__(value_type, "__dict__")
    if isinstance(class_namespace.get("__dict__"), property):
        return
    try:
        namespace = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return
    if type(namespace) is dict:
        yield from namespace.keys()
        yield from namespace.values()


def _assert_runtime_traceback_locals_are_detached(
    error, *, forbidden_values, operation=None
):
    """Assert public runtime frames retain no task or private input/output.

    This direct check complements the graph traversal: if a future refactor
    forgets to clear the local operation task before raising, fail immediately
    rather than relying on a particular task implementation's result graph.
    """

    forbidden_ids = {id(value) for value in forbidden_values}
    if operation is not None:
        forbidden_ids.add(id(operation))
    for chained in _walk_exception_chain(error):
        traceback = chained.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if frame.f_code.co_filename == isolated_runtime.__file__:
                local_values = tuple(frame.f_locals.values())
                assert not any(
                    type(value) in (asyncio.Task, asyncio.Future)
                    for value in local_values
                )
                assert not any(id(value) in forbidden_ids for value in local_values)
            traceback = traceback.tb_next


def _traceback_locals_reach_any(
    error, forbidden_values, *, runtime_filename=isolated_runtime.__file__
):
    """Whether runtime traceback locals reach an exact forbidden object."""

    forbidden_ids = {id(value) for value in forbidden_values}
    pending = []
    for chained in _walk_exception_chain(error):
        traceback = chained.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if frame.f_code.co_filename == runtime_filename:
                pending.extend(frame.f_locals.values())
            traceback = traceback.tb_next

    seen = set()
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in forbidden_ids:
            return True
        if identity in seen:
            continue
        seen.add(identity)
        # This limit is a circuit breaker, not a lossy global traversal: the
        # leaf policy above excludes the objects that could otherwise explode.
        assert len(seen) < 4096
        pending.extend(_safe_traceback_references(value))
    return False


@pytest.mark.asyncio
async def test_traceback_reachability_follows_done_operation_task_result():
    """The cancellation guard reaches a secret retained only by a task result."""

    response_secret = "TRACEBACK-TASK-RESULT-SECRET-2755"

    async def raise_with_operation_local():
        operation = asyncio.create_task(
            asyncio.sleep(0, result={"response": response_secret})
        )
        await operation
        raise RuntimeError("synthetic public failure")

    with pytest.raises(RuntimeError) as raised:
        await raise_with_operation_local()
    assert _traceback_locals_reach_any(
        raised.value,
        (response_secret,),
        runtime_filename=__file__,
    )


@pytest.mark.asyncio
async def test_host_ingress_is_private_and_uses_the_negotiated_sdk_contract(
    monkeypatch, tmp_path
):
    """Host ingress is callable without becoming an agent-visible tool."""

    client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        result = await feature.call_host_ingress(
            "telegram-webhook", {"update_id": 7}
        )

        assert result == {"accepted": True}
        assert client.ingress_calls == [("telegram-webhook", {"update_id": 7})]
        # The service's ordinary tools remain the only agent/LLM tool surface.
        assert [tool.name for tool in feature.get_tools()] == ["ping"]
        assert all(
            "telegram-webhook" not in tool.schema.description
            for tool in feature.get_tools()
        )
        assert "telegram-webhook" not in feature.tool_description
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_uses_the_published_wrapper_not_its_inner_transport(
    monkeypatch, tmp_path
):
    """A subprocess-style facade owns both capability passthrough and RPC."""

    inner = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )

    class Wrapper:
        def __init__(self):
            self.client = inner
            self.wrapper_calls = []

        @property
        def host_ingress_capabilities(self):
            return self.client.host_ingress_capabilities

        async def call_host_ingress(self, name, payload=None):
            self.wrapper_calls.append((name, payload))
            return await self.client.call_host_ingress(name, payload)

        def __getattr__(self, name):
            return getattr(self.client, name)

    wrapper = Wrapper()
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: wrapper
    )
    try:
        assert await feature.call_host_ingress("telegram-webhook", {"id": 1}) == {
            "accepted": True
        }
        assert wrapper.wrapper_calls == [("telegram-webhook", {"id": 1})]
        assert inner.ingress_calls == [("telegram-webhook", {"id": 1})]
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capabilities", "error_type"),
    [
        (None, HostIngressUnsupportedError),
        ({"version": 1, "names": ["telegram-webhook"]}, HostIngressUnsupportedError),
        (
            HostIngressCapabilities(names=("other-webhook",)),
            HostIngressUnknownNameError,
        ),
    ],
)
async def test_host_ingress_fails_closed_for_missing_malformed_or_unknown_capabilities(
    monkeypatch, tmp_path, capabilities, error_type
):
    """No ingress RPC is emitted unless typed metadata names it exactly."""

    client = _HostIngressClient(ingress_capabilities=capabilities)
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        with pytest.raises(error_type) as raised:
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

        assert type(raised.value) is error_type
        assert client.ingress_calls == []
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_rejects_str_subclasses_before_capability_comparison(
    monkeypatch, tmp_path
):
    """A hostile ``str`` subclass cannot equality-match an advertised name."""

    class EqualToEverything(str):
        def __eq__(self, other):
            return True

        def __hash__(self):
            return hash(str(self))

    client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("other-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        with pytest.raises(HostIngressError) as raised:
            await feature.call_host_ingress(
                EqualToEverything("not-an-advertised-webhook"),
                {"update_id": 7},
            )

        assert type(raised.value) is HostIngressError
        assert str(raised.value) == "host ingress failed"
        assert client.ingress_calls == []
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_rejects_mutable_and_malformed_capabilities(
    monkeypatch, tmp_path
):
    """Frozen SDK dataclasses with mutable or corrupted fields fail closed."""

    mutable_names = ["telegram-webhook"]
    mutable = HostIngressCapabilities(names=mutable_names)
    mutable_names.append("unexpected-webhook")
    malformed = HostIngressCapabilities(names=("telegram-webhook",))
    object.__setattr__(malformed, "version", 2)

    for capabilities in (mutable, malformed):
        client = _HostIngressClient(ingress_capabilities=capabilities)
        feature, _ = await _initialized_host_ingress_proxy(
            monkeypatch, tmp_path, lambda client=client, **_: client
        )
        try:
            with pytest.raises(HostIngressUnsupportedError) as raised:
                await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

            assert type(raised.value) is HostIngressUnsupportedError
            assert client.ingress_calls == []
        finally:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_rejects_subclass_with_overridden_supports(
    monkeypatch, tmp_path
):
    """An untrusted capability subclass cannot grant an unadvertised name."""

    class PermissiveCapabilities(HostIngressCapabilities):
        def supports(self, name):
            return True

    client = _HostIngressClient(
        ingress_capabilities=PermissiveCapabilities(names=("other-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        with pytest.raises(HostIngressUnsupportedError) as raised:
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

        assert type(raised.value) is HostIngressUnsupportedError
        assert client.ingress_calls == []
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_does_not_resolve_or_fall_back_to_another_agents_proxy(
    monkeypatch, tmp_path
):
    """The caller's already-resolved proxy is the only ingress authority."""

    alice_client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    bob_client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    alice_agent = Mock(did="did:test:alice", features={})
    bob_agent = Mock(did="did:test:bob", features={})
    alice, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path / "alice", lambda **_: alice_client, agent=alice_agent
    )
    bob, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path / "bob", lambda **_: bob_client, agent=bob_agent
    )
    # An ambient registry-like reference must not influence a proxy call.
    alice_agent.features["CompanionFeature"] = bob
    try:
        assert (
            await alice.call_host_ingress("telegram-webhook", {"agent": "alice"})
            == {"accepted": True}
        )
        assert alice_client.ingress_calls == [
            ("telegram-webhook", {"agent": "alice"})
        ]
        assert bob_client.ingress_calls == []
    finally:
        await alice.shutdown()
        await bob.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_redacts_invalid_payload_and_transport_failure(
    monkeypatch, tmp_path, caplog
):
    """Host-only request/config/transport details never cross the proxy API."""

    secret = "super-secret-webhook-token"
    transport_external = RuntimeError(
        f"stdio://private/{secret} config={{'token': '{secret}'}}"
    )

    class LeakyClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            raise transport_external

    client = LeakyClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        with (
            caplog.at_level("WARNING", logger=isolated_runtime.__name__),
            pytest.raises(HostIngressError) as transport_error,
        ):
            await feature.call_host_ingress("telegram-webhook", {"token": secret})
        assert str(transport_error.value) == "host ingress failed"
        _assert_host_ingress_error_is_detached(
            transport_error.value,
            secret=secret,
            external=transport_external,
        )
        assert secret not in str(transport_error.value)
        assert "stdio://" not in str(transport_error.value)
        assert secret not in caplog.text

        with pytest.raises(HostIngressError) as invalid_payload_error:
            await feature.call_host_ingress("telegram-webhook", {"token": object()})
        assert str(invalid_payload_error.value) == "host ingress failed"
        _assert_host_ingress_error_is_detached(
            invalid_payload_error.value,
            secret=secret,
        )
        assert client.ingress_calls == [("telegram-webhook", {"token": secret})]
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_redacts_capability_getter_and_response_errors(
    monkeypatch, tmp_path, caplog
):
    """HostIngressError from an external facade never bypasses redaction."""

    secret = "super-secret-webhook-token"
    getter_external = RuntimeError(f"private capability token={secret}")

    class GetterErrorClient(_HostIngressClient):
        @property
        def host_ingress_capabilities(self):
            raise getter_external

    getter_client = GetterErrorClient()
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: getter_client
    )
    try:
        caplog.clear()
        with (
            caplog.at_level("WARNING", logger=isolated_runtime.__name__),
            pytest.raises(HostIngressError) as getter_error,
        ):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

        assert type(getter_error.value) is HostIngressError
        assert str(getter_error.value) == "host ingress failed"
        _assert_host_ingress_error_is_detached(
            getter_error.value,
            secret=secret,
            external=getter_external,
        )
        assert secret not in caplog.text
        assert getter_client.ingress_calls == []
    finally:
        await feature.shutdown()

    class SecretResponse:
        def __repr__(self):
            return secret

    class InvalidResponseClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            return {"token": SecretResponse()}

    response_client = InvalidResponseClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: response_client
    )
    try:
        caplog.clear()
        with (
            caplog.at_level("WARNING", logger=isolated_runtime.__name__),
            pytest.raises(HostIngressError) as response_error,
        ):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

        assert type(response_error.value) is HostIngressError
        assert str(response_error.value) == "host ingress failed"
        _assert_host_ingress_error_is_detached(
            response_error.value,
            secret=secret,
        )
        assert secret not in caplog.text
        assert response_client.ingress_calls == [
            ("telegram-webhook", {"update_id": 7})
        ]
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_uses_exact_detached_json_request_and_response_snapshots(
    monkeypatch, tmp_path
):
    """Ingress cannot dispatch subclasses or observe post-validation mutation."""

    entered = asyncio.Event()
    release = asyncio.Event()
    original_request = {"outer": {"state": "before"}, "items": [1, 2]}
    original_response = {"outer": {"accepted": True}, "items": [3, 4]}

    class SnapshotClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            entered.set()
            await release.wait()
            self.ingress_calls.append((name, payload))
            return original_response

    client = SnapshotClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    call = None
    try:
        call = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", original_request)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)

        # The client has not consumed its argument yet. Mutating the original
        # after admission must not rewrite the already-snapshotted RPC payload.
        original_request["outer"]["state"] = "after"
        original_request["items"].append(99)
        release.set()
        result = await call

        assert client.ingress_calls == [
            ("telegram-webhook", {"outer": {"state": "before"}, "items": [1, 2]})
        ]
        assert result == original_response
        original_response["outer"]["accepted"] = False
        original_response["items"].append(99)
        assert result == {"outer": {"accepted": True}, "items": [3, 4]}

        class DictSubclass(dict):
            pass

        class ListSubclass(list):
            pass

        class StringSubclass(str):
            pass

        invalid_payloads = (
            DictSubclass({"token": "subclass"}),
            {"items": ListSubclass([1])},
            {"token": StringSubclass("subclass")},
            {StringSubclass("token"): "subclass"},
        )
        for invalid_payload in invalid_payloads:
            with pytest.raises(HostIngressError, match="host ingress failed"):
                await feature.call_host_ingress("telegram-webhook", invalid_payload)
        assert len(client.ingress_calls) == 1

        class ResponseSubclassClient(_HostIngressClient):
            async def call_host_ingress(self, name, payload=None):
                self.ingress_calls.append((name, payload))
                return {"token": StringSubclass("response-subclass")}

        response_client = ResponseSubclassClient(
            ingress_capabilities=HostIngressCapabilities(
                names=("telegram-webhook",)
            )
        )
        response_feature, _ = await _initialized_host_ingress_proxy(
            monkeypatch, tmp_path / "response", lambda **_: response_client
        )
        try:
            with pytest.raises(HostIngressError, match="host ingress failed"):
                await response_feature.call_host_ingress(
                    "telegram-webhook", {"update_id": 1}
                )
            assert response_client.ingress_calls == [
                ("telegram-webhook", {"update_id": 1})
            ]
        finally:
            await response_feature.shutdown()
    finally:
        release.set()
        if call is not None and not call.done():
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_preflights_exact_json_size_before_sdk_serialization(
    monkeypatch, tmp_path
):
    """Dense valid JSON reaches the SDK; oversized escaped values never do."""

    # ``[0,0,...]`` is the densest non-empty JSON tree: 32,767 scalar leaves
    # plus brackets/separators encode to 65,535 bytes, one below the SDK cap.
    dense_payload = [0] * ((MAX_HOST_INGRESS_PAYLOAD_BYTES - 1) // 2)
    dense_response = list(dense_payload)
    oversized_astral = "\U0001f642" * (MAX_HOST_INGRESS_PAYLOAD_BYTES // 12 + 1)
    validator_payloads = []
    sdk_validator = isolated_runtime.validate_host_ingress_payload

    def recording_validator(value):
        validator_payloads.append(value)
        return sdk_validator(value)

    monkeypatch.setattr(
        isolated_runtime,
        "validate_host_ingress_payload",
        recording_validator,
    )

    class DenseClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            return dense_response

    client = DenseClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        assert await feature.call_host_ingress("telegram-webhook", dense_payload) == dense_response
        assert client.ingress_calls == [("telegram-webhook", dense_payload)]

        # Both a scalar and a key exceed 64 KiB only after the SDK's required
        # ASCII escaping.  They must fail before the SDK validator or facade
        # can materialize/receive them.
        validator_count = len(validator_payloads)
        for oversized_request in (
            {"payload": oversized_astral},
            {oversized_astral: "payload"},
        ):
            with pytest.raises(HostIngressError, match="host ingress failed") as raised:
                await feature.call_host_ingress("telegram-webhook", oversized_request)
            _assert_host_ingress_error_is_detached(
                raised.value,
                secret=oversized_astral,
            )
            _assert_host_ingress_tracebacks_are_secret_free(
                raised.value,
                secret=oversized_astral,
            )
        assert len(validator_payloads) == validator_count
        assert client.ingress_calls == [("telegram-webhook", dense_payload)]

        class OversizedResponseClient(_HostIngressClient):
            async def call_host_ingress(self, name, payload=None):
                self.ingress_calls.append((name, payload))
                return {"payload": oversized_astral}

        response_client = OversizedResponseClient(
            ingress_capabilities=HostIngressCapabilities(
                names=("telegram-webhook",)
            )
        )
        response_feature, _ = await _initialized_host_ingress_proxy(
            monkeypatch, tmp_path / "response", lambda **_: response_client
        )
        try:
            with pytest.raises(HostIngressError, match="host ingress failed") as raised:
                await response_feature.call_host_ingress(
                    "telegram-webhook", {"update_id": 1}
                )
            _assert_host_ingress_error_is_detached(
                raised.value,
                secret=oversized_astral,
            )
            _assert_host_ingress_tracebacks_are_secret_free(
                raised.value,
                secret=oversized_astral,
            )
            assert response_client.ingress_calls == [
                ("telegram-webhook", {"update_id": 1})
            ]
            # The response reaches the host worker, but its oversized value
            # never reaches the SDK validator's allocating serializer.
            assert not any(
                value == {"payload": oversized_astral}
                for value in validator_payloads
            )
        finally:
            await response_feature.shutdown()
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_accepts_shared_alias_dags_as_detached_json_trees(
    monkeypatch, tmp_path
):
    """SDK-valid aliases are charged and copied independently per branch."""

    client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    shared = {"items": ["first"]}
    payload = {"left": shared, "right": shared}
    validator_payloads = []
    sdk_validator = isolated_runtime.validate_host_ingress_payload

    def recording_validator(value):
        validator_payloads.append(value)
        return sdk_validator(value)

    monkeypatch.setattr(isolated_runtime, "validate_host_ingress_payload", recording_validator)
    try:
        assert await feature.call_host_ingress("telegram-webhook", payload) == {
            "accepted": True
        }
        dispatched = client.ingress_calls[0][1]
        assert dispatched == payload
        assert dispatched is not payload
        assert dispatched["left"] is not dispatched["right"]
        assert dispatched["left"]["items"] is not dispatched["right"]["items"]
        assert json.dumps(dispatched, ensure_ascii=True, separators=(",", ":")) == json.dumps(
            payload, ensure_ascii=True, separators=(",", ":")
        )
        shared["items"].append("source mutation")
        assert dispatched == {
            "left": {"items": ["first"]},
            "right": {"items": ["first"]},
        }
        assert dispatched in validator_payloads
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_rejects_container_cycles_and_excessive_alias_expansion(
    monkeypatch, tmp_path
):
    """Cycles and aliases that exceed the expanded JSON budget fail closed."""

    client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    validator_called = False

    def unexpected_validator(_value):
        nonlocal validator_called
        validator_called = True
        raise AssertionError("invalid graph reached SDK validation")

    monkeypatch.setattr(isolated_runtime, "validate_host_ingress_payload", unexpected_validator)
    recursive_list = []
    recursive_list.append(recursive_list)
    recursive_dict = {}
    recursive_dict["self"] = recursive_dict
    expanded: list[object] = []
    for _ in range(16):
        expanded = [expanded, expanded]
    try:
        for invalid_payload in (
            {"tree": recursive_list},
            {"tree": recursive_dict},
            {"tree": expanded},
        ):
            with pytest.raises(HostIngressError, match="host ingress failed"):
                await feature.call_host_ingress("telegram-webhook", invalid_payload)
        assert validator_called is False
        assert client.ingress_calls == []
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_generic_failures_have_no_secret_traceback_locals(
    monkeypatch, tmp_path
):
    """Validation/facade/RPC failures expose neither chains nor frame locals."""

    secret = "TRACEBACK-ONLY-SECRET-2755"

    class SecretValue:
        def __repr__(self):
            return secret

    capability_error = RuntimeError(f"capability={secret}")
    descriptor_error = RuntimeError(f"descriptor={secret}")
    transport_error = RuntimeError(f"transport={secret}")

    class GetterFailureClient(_HostIngressClient):
        @property
        def host_ingress_capabilities(self):
            raise capability_error

    class DescriptorFailureClient(_HostIngressClient):
        @property
        def call_host_ingress(self):
            raise descriptor_error

    class TransportFailureClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            raise transport_error

    class ResponseFailureClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            return {"secret": SecretValue()}

    clients_and_payloads = (
        (
            _HostIngressClient(
                ingress_capabilities=HostIngressCapabilities(
                    names=("telegram-webhook",)
                )
            ),
            {"secret": SecretValue()},
        ),
        (GetterFailureClient(), {"secret": secret}),
        (
            DescriptorFailureClient(
                ingress_capabilities=HostIngressCapabilities(
                    names=("telegram-webhook",)
                )
            ),
            {"secret": secret},
        ),
        (
            TransportFailureClient(
                ingress_capabilities=HostIngressCapabilities(
                    names=("telegram-webhook",)
                )
            ),
            {"secret": secret},
        ),
        (
            ResponseFailureClient(
                ingress_capabilities=HostIngressCapabilities(
                    names=("telegram-webhook",)
                )
            ),
            {"secret": secret},
        ),
    )

    for index, (client, payload) in enumerate(clients_and_payloads):
        feature, _ = await _initialized_host_ingress_proxy(
            monkeypatch,
            tmp_path / str(index),
            lambda client=client, **_: client,
        )
        try:
            with pytest.raises(HostIngressError) as raised:
                await feature.call_host_ingress("telegram-webhook", payload)
            assert type(raised.value) is HostIngressError
            assert str(raised.value) == "host ingress failed"
            _assert_host_ingress_error_is_detached(raised.value, secret=secret)
            _assert_host_ingress_tracebacks_are_secret_free(
                raised.value, secret=secret
            )
            _assert_runtime_traceback_locals_are_detached(
                raised.value,
                forbidden_values=(secret, client),
            )
        finally:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_caller_cancelled_host_ingress_discards_successful_secret_outcome(
    monkeypatch, tmp_path
):
    """A cancellation traceback cannot reach a response that won the RPC race."""

    request_secret = "HOST-INGRESS-REQUEST-SECRET-2755"
    response_secret = "HOST-INGRESS-RESPONSE-SECRET-2755"
    client_secret = "HOST-INGRESS-CLIENT-SECRET-2755"

    class CancelAfterSuccessClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cancel_target = None
            self.client_secret = client_secret

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            # Deliver cancellation after this task has reached its successful
            # response path but before the public caller can return it.
            self.cancel_target.cancel("first caller cancellation")
            return {"response": response_secret}

    client = CancelAfterSuccessClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    call = None
    try:
        call = asyncio.create_task(
            feature.call_host_ingress(
                "telegram-webhook", {"request": request_secret}
            )
        )
        client.cancel_target = call
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await call

        assert cancelled.value.args == ("first caller cancellation",)
        assert client.ingress_calls == [
            ("telegram-webhook", {"request": request_secret})
        ]
        assert not _traceback_locals_reach_any(
            cancelled.value,
            (request_secret, response_secret, client_secret, client),
        )
        _assert_runtime_traceback_locals_are_detached(
            cancelled.value,
            forbidden_values=(request_secret, response_secret, client_secret, client),
            operation=call,
        )
    finally:
        if call is not None and not call.done():
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_worker_tasks_drop_private_results_on_all_terminal_paths(
    monkeypatch, tmp_path
):
    """Success, failure, and caller cancellation leave no payload in worker tasks."""

    request_secret = "WORKER-REQUEST-SECRET-2755"
    response_secret = "WORKER-RESPONSE-SECRET-2755"
    delayed_started = asyncio.Event()
    release_delayed = asyncio.Event()
    worker_tasks = []

    class TerminalPathClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mode = "success"
            self.client_secret = "WORKER-CLIENT-SECRET-2755"

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if self.mode == "failure":
                raise RuntimeError(response_secret)
            if self.mode == "delayed":
                delayed_started.set()
                await release_delayed.wait()
            return {"response": response_secret}

    client = TerminalPathClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    real_create_task = asyncio.create_task

    def capture_host_ingress_worker(coro, *, name=None, **kwargs):
        task = real_create_task(coro, name=name, **kwargs)
        if name and name.startswith("isolated-host-ingress:"):
            worker_tasks.append(task)
        return task

    monkeypatch.setattr(
        isolated_runtime.asyncio, "create_task", capture_host_ingress_worker
    )
    delayed_call = None
    try:
        assert await feature.call_host_ingress(
            "telegram-webhook", {"request": request_secret}
        ) == {"response": response_secret}

        client.mode = "failure"
        with pytest.raises(HostIngressError, match="host ingress failed") as failure:
            await feature.call_host_ingress(
                "telegram-webhook", {"request": request_secret}
            )
        _assert_host_ingress_error_is_detached(failure.value, secret=response_secret)

        client.mode = "delayed"
        delayed_call = real_create_task(
            feature.call_host_ingress(
                "telegram-webhook", {"request": request_secret}
            )
        )
        await asyncio.wait_for(delayed_started.wait(), timeout=1)
        delayed_call.cancel("private caller cancellation")
        await asyncio.sleep(0)
        assert delayed_call.done() is False
        release_delayed.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await delayed_call
        assert cancelled.value.args == ("private caller cancellation",)

        assert len(worker_tasks) == 3
        for worker in worker_tasks:
            assert worker.done()
            assert worker.cancelled() is False
            # The one-shot slot is consumed before delivery, so the task's
            # native result cannot retain request, response, or client data.
            assert worker.result() is None
            assert worker.get_coro().cr_frame is None
    finally:
        release_delayed.set()
        if delayed_call is not None and not delayed_call.done():
            delayed_call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await delayed_call
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_preserves_sdk_cancellation(monkeypatch, tmp_path):
    """A child-side cancellation is never redacted as a host ingress error."""

    class CancellingClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            raise asyncio.CancelledError()

    client = CancellingClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        assert client.ingress_calls == [("telegram-webhook", {"update_id": 7})]
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_cancellation_drains_remote_call_before_reload(
    monkeypatch, tmp_path
):
    """Caller cancellation cannot release traffic beneath an active RPC."""

    started = asyncio.Event()
    blocker = asyncio.Event()
    clients = []

    class CancellableClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if len(clients) == 1:
                started.set()
                await blocker.wait()
            return {"client": len(clients)}

    def client_factory(**kwargs):
        client = CancellableClient(
            ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",)),
            **kwargs,
        )
        clients.append(client)
        return client

    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, client_factory
    )
    active = reload_task = None
    try:
        active = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        active.cancel()
        await asyncio.sleep(0)
        assert not active.done()

        reload_task = asyncio.create_task(feature.reload())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed is True
        assert not reload_task.done()
        assert clients[0].stopped is False

        blocker.set()
        with pytest.raises(asyncio.CancelledError):
            await active
        await asyncio.wait_for(reload_task, timeout=1)
        assert clients[0].stopped is True
        assert await feature.call_host_ingress("telegram-webhook", {"sequence": 2}) == {
            "client": 2
        }
    finally:
        blocker.set()
        if reload_task is not None and not reload_task.done():
            reload_task.cancel()
            try:
                await reload_task
            except asyncio.CancelledError:
                pass
        if active is not None and not active.done():
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
        await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_stop_failure_retains_exact_client_for_bounded_retry(
    monkeypatch, tmp_path
):
    """A failed terminal stop neither loses its client handle nor drains forever."""

    ingress_started = asyncio.Event()
    terminate_ingress = asyncio.Event()

    class TerminalStopFailure(BaseException):
        pass

    class FirstStopFailsClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            ingress_started.set()
            await terminate_ingress.wait()
            return {"accepted": True}

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise TerminalStopFailure("first terminal stop failed")
            self.stopped = True
            terminate_ingress.set()

    client = FirstStopFailsClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    ingress = None
    supervisor = feature._supervision_task
    try:
        assert supervisor is not None
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(ingress_started.wait(), timeout=1)

        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ) as raised:
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert type(raised.value) is RuntimeError
        assert "first terminal stop failed" not in str(raised.value)
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None

        assert feature._client is None
        assert feature._terminal_retirement_clients == [client]
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 1
        assert ingress.done() is False

        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert await ingress == {"accepted": True}
        assert client.stop_calls == 2
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate._active == 0
        assert feature._client is None
        assert feature._supervision_task is None
        assert feature._terminal_cleanup_task is None
        assert supervisor.done() is True
        assert supervisor.cancelled() is True
    finally:
        terminate_ingress.set()
        if ingress is not None and not ingress.done():
            ingress.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ingress
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_best_effort_terminal_cleanup_cannot_make_concurrent_shutdown_succeed(
    monkeypatch, tmp_path
):
    """A cached best-effort stop never lends its policy to explicit shutdown."""

    ingress_started = asyncio.Event()
    stop_started = asyncio.Event()
    release_first_stop = asyncio.Event()
    terminate_ingress = asyncio.Event()

    class HostileTerminalStop(BaseException):
        pass

    class ConcurrentStopClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.stops_in_flight = 0
            self.maximum_stops_in_flight = 0

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            ingress_started.set()
            await terminate_ingress.wait()
            return {"accepted": True}

        async def stop(self):
            self.stop_calls += 1
            self.stops_in_flight += 1
            self.maximum_stops_in_flight = max(
                self.maximum_stops_in_flight, self.stops_in_flight
            )
            try:
                if self.stop_calls == 1:
                    stop_started.set()
                    await release_first_stop.wait()
                    raise HostileTerminalStop("hostile terminal stop detail")
                self.stopped = True
                terminate_ingress.set()
            finally:
                self.stops_in_flight -= 1

    client = ConcurrentStopClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    ingress = quarantine = shutdown_task = None
    supervisor = feature._supervision_task
    try:
        assert supervisor is not None
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(ingress_started.wait(), timeout=1)

        # Quarantine owns the cached shared task and is entitled to return to
        # its originating lifecycle failure after its neutral attempt settles.
        quarantine = asyncio.create_task(feature._quarantine_unreconciled_client())
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        assert feature._terminal_cleanup_task is not None
        assert not quarantine.done()

        # An explicit shutdown joins the in-flight neutral work. It must make
        # its own success decision once the hostile stop has settled.
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        assert client.maximum_stops_in_flight == 1

        release_first_stop.set()
        await asyncio.wait_for(quarantine, timeout=1)
        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ) as raised:
            await asyncio.wait_for(shutdown_task, timeout=1)
        assert "hostile terminal stop detail" not in str(raised.value)
        assert raised.value.__cause__ is None

        # The exact facade is still retained and the old ingress is counted,
        # but the sealed proxy makes it unreachable to all new host traffic.
        assert client.stop_calls == 1
        assert client.maximum_stops_in_flight == 1
        assert feature._terminal_retirement_clients == [client]
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 1
        assert not ingress.done()
        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 8})

        # A later explicit caller creates a fresh neutral attempt. It retries
        # the exact retained facade, then may truthfully drain and finish.
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert await ingress == {"accepted": True}
        assert client.stop_calls == 2
        assert client.maximum_stops_in_flight == 1
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate._active == 0
        assert feature._client is None
        assert feature._supervision_task is None
        assert feature._terminal_cleanup_task is None
        assert supervisor.done() is True
        assert supervisor.cancelled() is True
    finally:
        release_first_stop.set()
        terminate_ingress.set()
        for task in (shutdown_task, quarantine, ingress):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_initialize_reports_generic_failure_for_retained_terminal_client(
    monkeypatch, tmp_path
):
    """Explicit initialize cannot start beside an unsuccessfully stopped child."""

    stop_started = asyncio.Event()
    release_first_stop = asyncio.Event()

    class HostileTerminalStop(BaseException):
        pass

    class NeverStopsClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                stop_started.set()
                await release_first_stop.wait()
            raise HostileTerminalStop("terminal stop must remain private")

    client = NeverStopsClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    quarantine = initialize_task = None
    try:
        quarantine = asyncio.create_task(feature._quarantine_unreconciled_client())
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.sleep(0)
        assert not initialize_task.done()

        release_first_stop.set()
        await asyncio.wait_for(quarantine, timeout=1)
        assert client.stop_calls == 1
        assert feature._terminal_retirement_clients == [client]

        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ) as raised:
            await asyncio.wait_for(initialize_task, timeout=1)
        assert type(raised.value) is RuntimeError
        assert "terminal stop must remain private" not in str(raised.value)
        assert raised.value.__cause__ is None
        assert client.stop_calls == 1
        assert feature._terminal_retirement_clients == [client]
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
    finally:
        release_first_stop.set()
        for task in (initialize_task, quarantine):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ):
            await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cancellation_preserves_caller_cancel_after_hostile_stop(
    monkeypatch, tmp_path
):
    """A caller cancellation wins over an incomplete neutral stop attempt."""

    stop_started = asyncio.Event()
    release_first_stop = asyncio.Event()

    class HostileTerminalStop(BaseException):
        pass

    class CancelledStopClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                stop_started.set()
                await release_first_stop.wait()
                raise HostileTerminalStop("do not replace caller cancellation")
            self.stopped = True

    client = CancelledStopClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    shutdown_task = None
    try:
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        shutdown_task.cancel("explicit shutdown cancellation")
        await asyncio.sleep(0)
        assert not shutdown_task.done()

        release_first_stop.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(shutdown_task, timeout=1)
        assert cancelled.value.args == ("explicit shutdown cancellation",)
        assert client.stop_calls == 1
        assert feature._terminal_retirement_clients == [client]
        assert feature._traffic_gate.sealed is True

        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 2
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_cleanup_task is None
    finally:
        release_first_stop.set()
        if shutdown_task is not None and not shutdown_task.done():
            shutdown_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await shutdown_task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_owned_terminal_cleanup_and_shutdown_serialize_exact_client_stop(
    monkeypatch, tmp_path
):
    """A reload owner and shared shutdown never overlap one facade stop."""

    first_stop_started = asyncio.Event()
    release_first_stop = asyncio.Event()

    class HostileTerminalStop(BaseException):
        pass

    class OwnedRaceClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.stops_in_flight = 0
            self.maximum_stops_in_flight = 0

        async def stop(self):
            self.stop_calls += 1
            self.stops_in_flight += 1
            self.maximum_stops_in_flight = max(
                self.maximum_stops_in_flight, self.stops_in_flight
            )
            try:
                if self.stop_calls == 1:
                    first_stop_started.set()
                    await release_first_stop.wait()
                if self.stop_calls < 3:
                    raise HostileTerminalStop("owned cleanup stop failed")
                self.stopped = True
            finally:
                self.stops_in_flight -= 1

    client = OwnedRaceClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    owned = shutdown_task = None
    lock_held = False
    try:
        await feature._reload_lock.acquire()
        lock_held = True
        owned = asyncio.create_task(
            feature._quarantine_unreconciled_client(lifecycle_lock_held=True)
        )
        await asyncio.wait_for(first_stop_started.wait(), timeout=1)

        # The shared shutdown begins while the owned cleanup still awaits its
        # exact client's first stop. It must wait on retirement ownership,
        # rather than start a second facade stop beside it.
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        assert client.maximum_stops_in_flight == 1
        assert not shutdown_task.done()

        release_first_stop.set()
        await asyncio.wait_for(owned, timeout=1)
        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ):
            await asyncio.wait_for(shutdown_task, timeout=1)
        assert client.stop_calls == 2
        assert client.maximum_stops_in_flight == 1
        assert feature._terminal_retirement_clients == [client]
        assert feature._traffic_gate.sealed is True

        feature._reload_lock.release()
        lock_held = False
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 3
        assert client.maximum_stops_in_flight == 1
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_cleanup_task is None
    finally:
        release_first_stop.set()
        if lock_held:
            feature._reload_lock.release()
        for task in (shutdown_task, owned):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_waits_for_reload_then_uses_replacement_child(
    monkeypatch, tmp_path
):
    """Reload drains a running ingress call before retiring its subprocess."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()
    clients = []

    class BlockingClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if payload == {"sequence": 1}:
                active_started.set()
                await release_active.wait()
            return {"client": len(clients), "sequence": payload["sequence"]}

    def client_factory(**kwargs):
        client = BlockingClient(
            ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",)),
            **kwargs,
        )
        clients.append(client)
        return client

    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, client_factory
    )
    active = reload_task = queued = None
    try:
        active = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(active_started.wait(), timeout=1)
        reload_task = asyncio.create_task(feature.reload())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed is True

        queued = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 2})
        )
        await asyncio.sleep(0)
        assert not queued.done()
        assert clients[0].ingress_calls == [("telegram-webhook", {"sequence": 1})]

        release_active.set()
        assert await active == {"client": 1, "sequence": 1}
        await reload_task
        assert clients[0].stopped is True
        assert await queued == {"client": 2, "sequence": 2}
        assert clients[1].ingress_calls == [("telegram-webhook", {"sequence": 2})]
    finally:
        release_active.set()
        for task in (queued, reload_task, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_waits_for_config_transition_before_delivery(
    monkeypatch, tmp_path
):
    """A config transition drains ingress before its lifecycle hook starts."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class TransitionClient(_HostIngressClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared = []

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if payload == {"sequence": 1}:
                active_started.set()
                await release_active.wait()
            return {"sequence": payload["sequence"]}

        async def prepare_config_transition(self, config):
            self.prepared.append(dict(config))
            return ConfigTransitionResult.applied()

    client = TransitionClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    active = update = queued = None
    try:
        active = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(active_started.wait(), timeout=1)
        update = asyncio.create_task(feature.set_config({"enabled": False}))
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed is True
        assert client.prepared == []

        queued = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 2})
        )
        await asyncio.sleep(0)
        assert not queued.done()

        release_active.set()
        assert await active == {"sequence": 1}
        await update
        assert client.prepared == [{"enabled": False}]
        assert await queued == {"sequence": 2}
        assert client.ingress_calls == [
            ("telegram-webhook", {"sequence": 1}),
            ("telegram-webhook", {"sequence": 2}),
        ]
    finally:
        release_active.set()
        for task in (queued, update, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_shutdown_drains_active_call_and_rejects_new_work(
    monkeypatch, tmp_path
):
    """Shutdown cannot stop a child while it serves an admitted ingress RPC."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            active_started.set()
            await release_active.wait()
            return {"accepted": True}

    client = BlockingClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    active = shutdown_task = None
    try:
        active = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(active_started.wait(), timeout=1)
        shutdown_task = asyncio.create_task(feature.shutdown())
        for _ in range(100):
            if feature._traffic_gate.sealed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.sealed is True

        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await feature.call_host_ingress("telegram-webhook", {"sequence": 2})
        assert client.ingress_calls == [("telegram-webhook", {"sequence": 1})]
        # Terminal cleanup stops the selected child before it drains the
        # admitted ingress. A cooperative double still needs the explicit
        # release below, but a production wrapper uses this stop to terminate
        # a permanently wedged subprocess.
        for _ in range(100):
            if client.stopped:
                break
            await asyncio.sleep(0)
        assert client.stopped is True
        assert not shutdown_task.done()

        release_active.set()
        assert await active == {"accepted": True}
        await shutdown_task
        assert client.stopped is True
    finally:
        release_active.set()
        for task in (shutdown_task, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_drain_timeout_keeps_dishonest_stop_sealed_and_owned(
    monkeypatch, tmp_path
):
    """A stop that lies about a wedged RPC cannot hang or report success."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class DishonestStopClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            active_started.set()
            await release_active.wait()
            return {"accepted": True}

        async def stop(self):
            # Deliberately claim success without settling the admitted RPC.
            self.stop_calls += 1
            self.stopped = True

    client = DishonestStopClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    monkeypatch.setattr(isolated_runtime, "_TERMINAL_TRAFFIC_DRAIN_TIMEOUT", 0.02)
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    ingress = None
    try:
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(active_started.wait(), timeout=1)

        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)

        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 1
        assert feature._terminal_traffic_drain_task is not None
        assert ingress.done() is False

        release_active.set()
        assert await ingress == {"accepted": True}
        for _ in range(100):
            if feature._terminal_traffic_drain_task is None:
                break
            await asyncio.sleep(0)
        assert feature._terminal_traffic_drain_task is None

        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert feature._traffic_gate._active == 0
        assert feature._traffic_gate.sealed is True
    finally:
        release_active.set()
        if ingress is not None and not ingress.done():
            ingress.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ingress
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_drain_timeout_seals_before_recovery_finalizer_reopens(
    monkeypatch, tmp_path
):
    """A stopped facade cannot admit a queued ingress during drain-timeout unwind."""

    unhealthy = asyncio.Event()
    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class DishonestStopClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def health(self):
            await unhealthy.wait()
            return False

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if payload == {"sequence": 1}:
                active_started.set()
                await release_active.wait()
            return {"sequence": payload["sequence"]}

        async def stop(self):
            # Claim success without interrupting the already admitted request.
            self.stop_calls += 1
            self.stopped = True

    client = DishonestStopClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(isolated_runtime, "_TERMINAL_TRAFFIC_DRAIN_TIMEOUT", 0.02)
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    active = queued = None
    supervisor = feature._supervision_task
    try:
        assert supervisor is not None
        active = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(active_started.wait(), timeout=1)
        unhealthy.set()
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await real_sleep(0)
        assert feature._traffic_gate.closed is True

        queued = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 2})
        )
        with pytest.raises(isolated_runtime._TerminalTrafficDrainTimedOut):
            await asyncio.wait_for(supervisor, timeout=1)
        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await asyncio.wait_for(queued, timeout=1)

        assert client.stop_calls == 1
        assert client.ingress_calls == [("telegram-webhook", {"sequence": 1})]
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate.closed is True
    finally:
        release_active.set()
        for task in (queued, active):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        for _ in range(100):
            if feature._terminal_traffic_drain_task is None:
                break
            await real_sleep(0)
        if feature._terminal_traffic_drain_task is None:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_revokes_initializer_waiting_for_reload_lock(monkeypatch, tmp_path):
    """A shutdown queued before inner initialize owns the terminal generation."""

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    started_clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        started_clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    initialize_task = shutdown_task = None
    reload_lock_held = False
    try:
        await feature._reload_lock.acquire()
        reload_lock_held = True
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.sleep(0)

        # The outer initialize has created its inner task, but that task has
        # not yet acquired lifecycle ownership.  Shutdown now creates its
        # cleanup transaction before the lock is released.
        shutdown_task = asyncio.create_task(feature.shutdown())
        for _ in range(100):
            if feature._terminal_lifecycle_latched and feature._terminal_cleanup_task:
                break
            await asyncio.sleep(0)
        assert feature._terminal_lifecycle_latched is True
        assert feature._terminal_cleanup_task is not None

        feature._reload_lock.release()
        reload_lock_held = False
        await asyncio.wait_for(shutdown_task, timeout=1)
        with pytest.raises(
            RuntimeError, match="terminal lifecycle changed during initialize"
        ):
            await asyncio.wait_for(initialize_task, timeout=1)

        assert started_clients == []
        assert feature._client is None
        assert feature._supervision_task is None
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate.closed is True
    finally:
        if reload_lock_held:
            feature._reload_lock.release()
        for task in (initialize_task, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_during_initialize_precleanup_revokes_enable_permit(
    monkeypatch, tmp_path
):
    """A newer shutdown cannot be absorbed while re-enable retires old state."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()
    started_clients = []

    class SlowRetirementClient(FakeIsolatedClient):
        async def stop(self):
            stop_started.set()
            await release_stop.wait()
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        started_clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    retired_client = SlowRetirementClient()
    feature._terminal_lifecycle_latched = True
    feature._terminal_retirement_clients = [retired_client]
    initialize_task = shutdown_task = None
    try:
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.wait_for(stop_started.wait(), timeout=1)

        # This terminal request arrives after initialize has begun its old
        # cleanup but before it may acquire the fresh-start ownership lock.
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        release_stop.set()

        with pytest.raises(
            RuntimeError, match="terminal lifecycle changed during initialize"
        ):
            await asyncio.wait_for(initialize_task, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)

        assert started_clients == []
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
    finally:
        release_stop.set()
        for task in (initialize_task, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_cancellation_seals_waiting_ingress_before_terminal_finally(
    monkeypatch, tmp_path
):
    """A waiter cannot slip through recovery's cancellation-finally boundary."""

    close_admission_entered = asyncio.Event()

    class UnhealthyIngressClient(_HostIngressClient):
        async def health(self):
            return False

    client = UnhealthyIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    original_close_admission = feature._close_traffic_gate_admission

    async def block_after_close_admission():
        await original_close_admission()
        close_admission_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(feature, "_close_traffic_gate_admission", block_after_close_admission)
    supervisor = queued = None
    try:
        supervisor = asyncio.create_task(feature._supervise())
        feature._supervision_task = supervisor
        await asyncio.wait_for(close_admission_entered.wait(), timeout=1)
        assert feature._traffic_gate.closed is True

        queued = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.sleep(0)
        assert queued.done() is False

        supervisor.cancel("close-admission cancellation")
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(supervisor, timeout=1)
        assert cancelled.value.args == ("close-admission cancellation",)
        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await asyncio.wait_for(queued, timeout=1)

        assert client.ingress_calls == []
        assert client.stopped is True
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
    finally:
        for task in (queued, supervisor):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_recovery_replays_first_of_repeated_cancellations(
    monkeypatch, tmp_path
):
    """The shared recovery drain retains the caller's first cancellation args."""

    recovery_started = asyncio.Event()
    release_recovery = asyncio.Event()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)

    async def controlled_recovery(_transition):
        recovery_started.set()
        await release_recovery.wait()

    monkeypatch.setattr(
        feature,
        "_recover_fenced_transition_uninterrupted",
        controlled_recovery,
    )
    recovery = None
    try:
        recovery = asyncio.create_task(
            feature._recover_fenced_transition(object(), RuntimeError("original"))
        )
        await asyncio.wait_for(recovery_started.wait(), timeout=1)
        recovery.cancel("first recovery cancellation")
        await asyncio.sleep(0)
        recovery.cancel("second recovery cancellation")
        release_recovery.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(recovery, timeout=1)
        assert cancelled.value.args == ("first recovery cancellation",)
    finally:
        release_recovery.set()
        if recovery is not None and not recovery.done():
            recovery.cancel()
            with pytest.raises(asyncio.CancelledError):
                await recovery


@pytest.mark.asyncio
async def test_supervisor_does_not_stop_terminal_client_after_retirement_lock_wait(
    monkeypatch, tmp_path
):
    """Shutdown ownership won while a health restart waited for the stop lock."""

    class UnhealthyClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            return False

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

        async def start(self):
            self.start_calls += 1
            self.started = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = UnhealthyClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    supervisor = shutdown_task = None
    retirement_lock_held = False
    try:
        await feature._terminal_retirement_lock.acquire()
        retirement_lock_held = True
        supervisor = asyncio.create_task(feature._supervise())
        for _ in range(100):
            if feature._reload_lock.locked() and feature._traffic_gate.closed:
                break
            await real_sleep(0)
        assert feature._reload_lock.locked()
        assert feature._traffic_gate.closed is True

        # Keep this manually-driven supervisor alive long enough to acquire
        # the stop lock after shutdown has latched and unpublished the client.
        feature._supervision_task = None
        shutdown_task = asyncio.create_task(feature.shutdown())
        for _ in range(100):
            if feature._stopping and feature._client is None:
                break
            await real_sleep(0)
        assert feature._stopping is True
        assert feature._terminal_lifecycle_latched is True
        assert feature._client is None

        feature._terminal_retirement_lock.release()
        retirement_lock_held = False
        await asyncio.wait_for(supervisor, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)

        assert client.stop_calls == 1
        assert client.start_calls == 0
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        if retirement_lock_held:
            feature._terminal_retirement_lock.release()
        for task in (supervisor, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_does_not_start_client_after_terminal_backoff_race(
    monkeypatch, tmp_path
):
    """A terminal latch during supervisor backoff prevents the stale restart."""

    backoff_started = asyncio.Event()
    release_backoff = asyncio.Event()

    class UnhealthyClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            return False

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

        async def start(self):
            self.start_calls += 1
            self.started = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = UnhealthyClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep
    sleep_calls = 0

    async def controlled_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            backoff_started.set()
            await release_backoff.wait()
            return
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", controlled_sleep)
    supervisor = shutdown_task = None
    try:
        supervisor = asyncio.create_task(feature._supervise())
        await asyncio.wait_for(backoff_started.wait(), timeout=1)
        assert client.stop_calls == 1

        # Shutdown can retire the stopped facade while the supervisor still
        # owns the reload lock.  Do not cancel the manually-driven task: it
        # must reach the immediate pre-start ownership revalidation itself.
        feature._supervision_task = None
        shutdown_task = asyncio.create_task(feature.shutdown())
        for _ in range(100):
            if feature._stopping and feature._client is None:
                break
            await real_sleep(0)
        assert feature._stopping is True
        assert feature._client is None

        release_backoff.set()
        await asyncio.wait_for(supervisor, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)

        # The supervisor had already stopped this exact facade before the
        # terminal transaction unpublished it during backoff.
        assert client.stop_calls == 1
        assert client.start_calls == 0
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        release_backoff.set()
        for task in (supervisor, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_cancellation_replays_original_args_after_exact_stop(
    monkeypatch, tmp_path
):
    """Cancellation after an owned exact stop is replayed without a second stop."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    class SlowStoppingClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            return False

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                stop_started.set()
                await release_stop.wait()
            self.stopped = True

        async def start(self):
            self.start_calls += 1
            self.started = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = SlowStoppingClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    supervisor = None
    try:
        supervisor = asyncio.create_task(feature._supervise())
        feature._supervision_task = supervisor
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        supervisor.cancel("agent shutdown cancellation")
        await asyncio.sleep(0)
        assert supervisor.done() is False
        release_stop.set()

        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(supervisor, timeout=1)
        assert cancelled.value.args == ("agent shutdown cancellation",)
        assert client.stop_calls == 1
        assert client.start_calls == 0
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_stop_completed_client_markers == []
        assert feature._traffic_gate.sealed is True
    finally:
        release_stop.set()
        if supervisor is not None and not supervisor.done():
            supervisor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await supervisor
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_fences_nonreturning_facade_stop_without_terminal_hang(
    monkeypatch, tmp_path
):
    """A timed-out legacy stop is sealed and retained, never reported as success."""

    stop_started = asyncio.Event()
    cancellation_observed = asyncio.Event()
    release_stop = asyncio.Event()

    class TimeoutSuppressingClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            return False

        async def stop(self):
            self.stop_calls += 1
            stop_started.set()
            while not release_stop.is_set():
                try:
                    await release_stop.wait()
                except asyncio.CancelledError:
                    # A genuinely hostile facade can consume the timeout
                    # cancellation and keep waiting. The proxy must still
                    # return at its bounded fence with this task explicitly
                    # owned, rather than orphaned in the event loop.
                    cancellation_observed.set()
            self.stopped = True

        async def start(self):
            self.start_calls += 1

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = TimeoutSuppressingClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.02)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", 0.02)
    supervisor = None
    try:
        supervisor = asyncio.create_task(feature._supervise())
        feature._supervision_task = supervisor
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        await asyncio.wait_for(supervisor, timeout=1)

        assert cancellation_observed.is_set()
        assert client.stop_calls == 1
        assert client.start_calls == 0
        assert feature._client is None
        assert feature._terminal_retirement_clients == [client]
        assert feature._terminal_cleanup_uncertain is True
        assert len(feature._terminal_lifecycle_tasks) == 1
        assert feature._traffic_gate.sealed is True
        assert feature._terminal_lifecycle_tasks[0].task.get_name().startswith(
            "isolated-supervisor-stop:"
        )
        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

        # The retained task's eventual outcome is consumed and released; it
        # never becomes an orphaned background exception. Its first timeout
        # remains uncertain, so an explicit cleanup still performs the one
        # fresh exact retirement attempt before it can report success.
        release_stop.set()
        for _ in range(100):
            if not feature._terminal_lifecycle_tasks:
                break
            await real_sleep(0)
        assert feature._terminal_lifecycle_tasks == []
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 2
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_lifecycle_tasks == []
        assert feature._traffic_gate.sealed is True
    finally:
        release_stop.set()
        if supervisor is not None and not supervisor.done():
            supervisor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await supervisor


@pytest.mark.asyncio
async def test_timed_out_terminal_stop_stays_owned_and_blocks_exact_retry(
    monkeypatch, tmp_path
):
    """A bounded caller never detaches or races the still-running exact stop."""

    stop_started = asyncio.Event()
    cancellation_observed = asyncio.Event()
    release_stop = asyncio.Event()

    class CancellationHostileClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.secret = "TERMINAL-STOP-SECRET-2755"

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                stop_started.set()
                while not release_stop.is_set():
                    try:
                        await release_stop.wait()
                    except asyncio.CancelledError:
                        cancellation_observed.set()
                return
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", 0.01)
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = CancellationHostileClient()
    feature._client = client
    try:
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        assert cancellation_observed.is_set()
        assert client.stop_calls == 1
        assert feature._terminal_retirement_clients == [client]
        assert len(feature._terminal_lifecycle_tasks) == 1

        # The still-running first stop owns this exact facade. A fresh caller
        # remains fail-closed instead of starting a competing stop coroutine.
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 1

        release_stop.set()
        for _ in range(100):
            if not feature._terminal_lifecycle_tasks:
                break
            await asyncio.sleep(0)
        assert feature._terminal_lifecycle_tasks == []

        # The original timeout remains uncertain, so the first honest success
        # path makes one fresh, serialized exact-client retirement attempt.
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 2
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
    finally:
        release_stop.set()
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_cancellation_resistant_health_probe_is_owned_until_terminal_retry(
    monkeypatch, tmp_path
):
    """Shutdown never succeeds or re-enables beside a stale health coroutine."""

    health_started = asyncio.Event()
    cancellation_observed = asyncio.Event()
    release_health = asyncio.Event()
    started_clients = []

    class CancellationResistantHealthClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.credential = "HEALTH-PROBE-CREDENTIAL-2755"

        async def health(self):
            health_started.set()
            while not release_health.is_set():
                try:
                    await release_health.wait()
                except asyncio.CancelledError:
                    cancellation_observed.set()
            return True

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_TIMEOUT", 0.5)
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_CANCELLATION_GRACE", 0.01)
    client = CancellationResistantHealthClient()
    client_ref = weakref.ref(client)

    def factory(**kwargs):
        replacement = FakeIsolatedClient(**kwargs)
        started_clients.append(replacement)
        return replacement

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=factory)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    supervisor = asyncio.create_task(feature._supervise())
    feature._supervision_task = supervisor
    try:
        await asyncio.wait_for(health_started.wait(), timeout=1)

        # Cancelling the supervisor through the direct shutdown path must not
        # turn the still-running probe into an unowned task or false success.
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        await asyncio.wait_for(cancellation_observed.wait(), timeout=1)
        probe = feature._terminal_health_probe_task
        assert probe is not None and not probe.done()
        assert probe.get_name().startswith("isolated-health-probe:")
        assert client.stop_calls == 0
        assert feature._client is None
        assert feature._traffic_gate.sealed is True

        # Explicit re-enable must first settle the exact old probe; it cannot
        # construct a new facade alongside credentials retained by that task.
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.initialize(), timeout=1)
        assert started_clients == []

        release_health.set()
        for _ in range(100):
            if feature._terminal_health_probe_task is None:
                break
            await real_sleep(0)
        assert feature._terminal_health_probe_task is None

        # Once the callback consumed the probe outcome, a fresh terminal retry
        # can retire the exact facade and a later explicit initialize is safe.
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 1
        await asyncio.wait_for(feature.initialize(), timeout=1)
        assert len(started_clients) == 1
        assert feature._client is started_clients[0]

        await feature.shutdown()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(supervisor, timeout=1)
        del probe
        del client
        gc.collect()
        assert client_ref() is None
    finally:
        release_health.set()
        if not supervisor.done():
            supervisor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await supervisor
        if not feature._stopping:
            await feature.shutdown()


def test_terminal_stop_completion_marker_is_weak_and_drops_dead_identity(tmp_path):
    """Stopped facades and their secrets are never retained by a completion mark."""

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = FakeIsolatedClient()
    client.secret = "STOPPED-FACADE-SECRET-2755"
    client_ref = weakref.ref(client)

    feature._mark_terminal_stop_completed(client)
    assert len(feature._terminal_stop_completed_client_markers) == 1
    del client
    gc.collect()

    assert client_ref() is None
    # A dead weak marker cannot match a later object, even if Python later
    # reuses the old object's id(). Consuming it also removes the stale entry.
    assert feature._forget_terminal_stop_completion(FakeIsolatedClient()) is False
    assert feature._terminal_stop_completed_client_markers == []


@pytest.mark.asyncio
@pytest.mark.parametrize("nonweakrefable", [False, True])
async def test_terminal_cleanup_claims_inflight_supervisor_stop_completion_once(
    tmp_path, nonweakrefable
):
    """A late supervisor callback cannot recreate a retired-cycle marker."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    if nonweakrefable:

        class Client:
            __slots__ = ("stop_calls", "secret")

            def __init__(self):
                self.stop_calls = 0
                self.secret = "NONWEAK-STOP-SECRET-2755"

            async def stop(self):
                self.stop_calls += 1
                stop_started.set()
                await release_stop.wait()

    else:

        class Client:
            def __init__(self):
                self.stop_calls = 0
                self.secret = "WEAK-STOP-SECRET-2755"

            async def stop(self):
                self.stop_calls += 1
                stop_started.set()
                await release_stop.wait()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = Client()
    marker = feature._begin_terminal_stop_completion(client)
    assert marker is not None

    async def supervisor_stop():
        await client.stop()
        feature._mark_terminal_stop_completed(marker)

    owner = asyncio.create_task(supervisor_stop())
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        # This is the terminal-cleanup ordering: it has consumed no completed
        # marker yet, so it claims the in-flight callback and retains the
        # exact facade until that callback proves the one stop completed.
        assert feature._forget_terminal_stop_completion(
            client, terminal_retirement=True
        ) is False
        feature._retain_terminal_retirement_client(client)

        release_stop.set()
        await asyncio.wait_for(owner, timeout=1)

        assert client.stop_calls == 1
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_stop_completed_client_markers == []
        # A later lifecycle reuse cannot consume the retired completion.
        assert feature._forget_terminal_stop_completion(client) is False
    finally:
        release_stop.set()
        if not owner.done():
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner


@pytest.mark.asyncio
async def test_nonweakrefable_stop_marker_avoids_duplicate_stop_and_is_consumed(
    monkeypatch, tmp_path
):
    """A supervisor-proven stop survives the shutdown/backoff race exactly once."""

    class NonWeakrefableClient:
        __slots__ = ("stop_calls", "stopped")

        def __init__(self):
            self.stop_calls = 1  # Supervisor already completed this exact stop.
            self.stopped = True

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = NonWeakrefableClient()
    feature._client = client
    feature._mark_terminal_stop_completed(client)
    assert len(feature._terminal_stop_completed_client_markers) == 1

    await feature.shutdown()

    assert client.stop_calls == 1
    assert feature._terminal_stop_completed_client_markers == []
    assert feature._terminal_retirement_clients == []
    assert feature._client is None


@pytest.mark.asyncio
async def test_unhealthy_supervisor_stops_wedged_host_ingress_before_draining(
    monkeypatch, tmp_path
):
    """Health recovery reaches the wrapper stop path before gate drain."""

    permit_unhealthy_probe = asyncio.Event()
    ingress_started = asyncio.Event()
    terminate_ingress = asyncio.Event()
    restarted = asyncio.Event()

    class WedgedIngressClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.start_calls = 0
            self.stop_calls = 0
            self.active_when_stopped = None
            self.active_when_restarted = None

        async def start(self):
            self.start_calls += 1
            if self.start_calls > 1:
                self.active_when_restarted = feature._traffic_gate._active
                restarted.set()

        async def health(self):
            await permit_unhealthy_probe.wait()
            return self.start_calls > 1

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            ingress_started.set()
            await terminate_ingress.wait()
            return {"accepted": True}

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True
            self.active_when_stopped = feature._traffic_gate._active
            terminate_ingress.set()

    client = WedgedIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    ingress = None
    try:
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(ingress_started.wait(), timeout=1)
        permit_unhealthy_probe.set()

        await asyncio.wait_for(restarted.wait(), timeout=1)
        assert await ingress == {"accepted": True}
        assert client.stop_calls == 1
        assert client.active_when_stopped == 1
        assert client.active_when_restarted == 0
        # The owned facade operation returns before the supervisor's matching
        # gate-finally reopens admission; wait for that public boundary rather
        # than treating the client-side start callback as a host-ready signal.
        for _ in range(100):
            if feature._traffic_gate.closed is False:
                break
            await real_sleep(0)
        assert feature._traffic_gate.closed is False
        assert feature._traffic_gate.sealed is False
        assert feature._client is client
    finally:
        terminate_ingress.set()
        if ingress is not None and not ingress.done():
            ingress.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ingress
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_cancelled_supervisor_uses_terminal_stop_before_draining_host_ingress(
    monkeypatch, tmp_path
):
    """Agent-side supervisor cancellation composes through terminal cleanup."""

    ingress_started = asyncio.Event()
    terminate_ingress = asyncio.Event()

    class WedgedIngressClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            ingress_started.set()
            await terminate_ingress.wait()
            return {"accepted": True}

        async def stop(self):
            self.stopped = True
            terminate_ingress.set()

    client = WedgedIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    ingress = None
    supervisor = feature._supervision_task
    try:
        assert supervisor is not None
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(ingress_started.wait(), timeout=1)
        ingress.cancel("caller cancellation")
        await asyncio.sleep(0)
        assert ingress.done() is False
        assert feature._traffic_gate._active == 1

        supervisor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(supervisor, timeout=1)
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(ingress, timeout=1)
        assert cancelled.value.args == ("caller cancellation",)
        assert client.stopped is True
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 0
        assert feature._client is None
        assert feature._supervision_task is None
    finally:
        terminate_ingress.set()
        if ingress is not None and not ingress.done():
            ingress.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ingress
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_kind", ("future", "task"))
async def test_owned_facade_lifecycle_accepts_completed_future_and_task(
    awaitable_kind, monkeypatch
):
    """A valid facade awaitable always runs through the named host task."""

    async def completed_task():
        return "stopped"

    loop = asyncio.get_running_loop()
    if awaitable_kind == "future":
        operation = loop.create_future()
        operation.set_result("stopped")
    else:
        operation = asyncio.create_task(completed_task(), name="facade-stop-task")
        await operation

    class FutureReturningFacade:
        def stop(self):
            return operation

    created_tasks = []
    real_create = isolated_runtime._create_host_owned_facade_task

    def capture_task(awaitable, *, name):
        task = real_create(awaitable, name=name)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(isolated_runtime, "_create_host_owned_facade_task", capture_task)
    assert await isolated_runtime._await_owned_facade_lifecycle_operation(
        FutureReturningFacade().stop(),
        name="test-facade-stop-awaitable",
        on_late_task=lambda _task: pytest.fail("completed facade must not detach"),
    ) == "stopped"
    assert [task.get_name() for task in created_tasks] == ["test-facade-stop-awaitable"]


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_kind", ("future", "task"))
async def test_owned_facade_lifecycle_cancellation_keeps_pending_future_and_task_owned(
    awaitable_kind, monkeypatch
):
    """Caller cancellation waits for the host owner instead of detaching it."""

    release = asyncio.Event()

    async def pending_task():
        await release.wait()
        return "stopped"

    loop = asyncio.get_running_loop()
    operation = (
        loop.create_future()
        if awaitable_kind == "future"
        else asyncio.create_task(pending_task(), name="facade-pending-stop-task")
    )

    class FutureReturningFacade:
        def stop(self):
            return operation

    created_tasks = []
    real_create = isolated_runtime._create_host_owned_facade_task

    def capture_task(awaitable, *, name):
        task = real_create(awaitable, name=name)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(isolated_runtime, "_create_host_owned_facade_task", capture_task)
    owner = asyncio.create_task(
        isolated_runtime._await_owned_facade_lifecycle_operation(
            FutureReturningFacade().stop(),
            name="test-pending-facade-stop-awaitable",
            on_late_task=lambda _task: pytest.fail("pending facade must settle"),
        )
    )
    await asyncio.sleep(0)
    owner.cancel("caller cancellation")
    await asyncio.sleep(0)
    assert owner.done() is False
    if awaitable_kind == "future":
        operation.set_result("stopped")
    else:
        release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await owner
    assert cancelled.value.args == ("caller cancellation",)
    assert created_tasks[0].get_name() == "test-pending-facade-stop-awaitable"


@pytest.mark.asyncio
async def test_owned_facade_lifecycle_timeout_retains_task_returned_by_facade(monkeypatch):
    """A cancellation-resistant Task return remains attached to its host owner."""

    cancellation_observed = asyncio.Event()
    release = asyncio.Event()

    async def pending_task():
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_observed.set()

    operation = asyncio.create_task(pending_task(), name="facade-hostile-stop-task")

    class TaskReturningFacade:
        def stop(self):
            return operation

    late_tasks = []
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", 0.01)
    try:
        with pytest.raises(isolated_runtime._FacadeLifecycleOperationTimedOut):
            await isolated_runtime._await_owned_facade_lifecycle_operation(
                TaskReturningFacade().stop(),
                name="test-hostile-facade-stop-task",
                on_late_task=late_tasks.append,
            )
        assert cancellation_observed.is_set()
        assert len(late_tasks) == 1
        assert late_tasks[0].get_name() == "test-hostile-facade-stop-task"
    finally:
        release.set()
        await asyncio.wait_for(late_tasks[0] if late_tasks else operation, timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_kind", ("future", "task"))
async def test_terminal_shutdown_owns_future_and_task_returned_by_facade_once(
    awaitable_kind, monkeypatch, tmp_path
):
    """Terminal retirement never retries a successfully owned facade stop."""

    async def completed_task():
        return None

    loop = asyncio.get_running_loop()
    if awaitable_kind == "future":
        operation = loop.create_future()
        operation.set_result(None)
    else:
        operation = asyncio.create_task(completed_task(), name="terminal-facade-stop-task")
        await operation

    class FutureReturningStopClient(FakeIsolatedClient):
        def __init__(self):
            super().__init__()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            return operation

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = FutureReturningStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    await feature.shutdown()
    await feature.shutdown()
    assert client.stop_calls == 1
    assert feature._terminal_retirement_clients == []
    assert feature._terminal_lifecycle_tasks == []


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_kind", ("future", "task"))
async def test_owned_health_probe_accepts_completed_future_and_task(awaitable_kind):
    """Completed facade probes are healthy rather than TypeError failures."""

    async def completed_task():
        return {"healthy": True}

    loop = asyncio.get_running_loop()
    if awaitable_kind == "future":
        operation = loop.create_future()
        operation.set_result({"healthy": True})
    else:
        operation = asyncio.create_task(completed_task(), name="facade-health-task")
        await operation

    class FutureReturningFacade:
        def health(self):
            return operation

    started_tasks = []
    assert await isolated_runtime._await_owned_health_probe(
        FutureReturningFacade().health(),
        name="test-facade-health-awaitable",
        on_started=started_tasks.append,
        on_late_task=lambda _task: pytest.fail("completed health must not detach"),
    ) == {"healthy": True}
    assert [task.get_name() for task in started_tasks] == ["test-facade-health-awaitable"]


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_kind", ("future", "task"))
async def test_owned_health_probe_accepts_pending_future_and_task(awaitable_kind):
    """Pending valid facade probes settle normally through host ownership."""

    release = asyncio.Event()

    async def pending_task():
        await release.wait()
        return {"healthy": True}

    loop = asyncio.get_running_loop()
    operation = (
        loop.create_future()
        if awaitable_kind == "future"
        else asyncio.create_task(pending_task(), name="facade-pending-health-task")
    )

    class FutureReturningFacade:
        def health(self):
            return operation

    started_tasks = []
    owner = asyncio.create_task(
        isolated_runtime._await_owned_health_probe(
            FutureReturningFacade().health(),
            name="test-pending-facade-health-awaitable",
            on_started=started_tasks.append,
            on_late_task=lambda _task: pytest.fail("pending health must settle"),
        )
    )
    await asyncio.sleep(0)
    if awaitable_kind == "future":
        operation.set_result({"healthy": True})
    else:
        release.set()
    assert await owner == {"healthy": True}
    assert started_tasks[0].get_name() == "test-pending-facade-health-awaitable"


@pytest.mark.asyncio
async def test_host_owned_facade_callbacks_deliver_each_registration_once():
    """Settlement has one callback path across registration timing and reentry."""

    loop = asyncio.get_running_loop()
    source = loop.create_future()
    operation = isolated_runtime._create_host_owned_facade_task(
        source, name="callback-once"
    )
    delivered = []

    def duplicate(completed):
        delivered.append(("duplicate", completed))

    def reentrant(completed):
        delivered.append(("reentrant", completed))
        completed.add_done_callback(lambda settled: delivered.append(("nested", settled)))

    # Multiple registrations of the same callback retain asyncio's one-call
    # per registration contract; none may be redelivered by _notify_settled.
    operation.add_done_callback(duplicate)
    operation.add_done_callback(duplicate)
    operation.add_done_callback(reentrant)
    source.set_result(None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert [name for name, _ in delivered] == [
        "duplicate",
        "duplicate",
        "reentrant",
        "nested",
    ]
    assert all(completed is operation for _, completed in delivered)
    assert operation._done_callbacks == []

    # Registration after notification is queued once and not retained.
    operation.add_done_callback(lambda completed: delivered.append(("post", completed)))
    await asyncio.sleep(0)
    assert [name for name, _ in delivered].count("post") == 1
    assert operation._done_callbacks == []


@pytest.mark.asyncio
async def test_host_owned_facade_pre_settled_callback_is_not_delivered_twice():
    """A source settled before facade construction still has one callback path."""

    source = asyncio.get_running_loop().create_future()
    source.set_result(None)
    operation = isolated_runtime._create_host_owned_facade_task(
        source, name="pre-settled-callback"
    )
    delivered = []
    operation.add_done_callback(delivered.append)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert delivered == [operation]
    assert operation._done_callbacks == []


@pytest.mark.asyncio
async def test_foreign_facade_stopped_or_closed_loop_remains_durably_fenced(
    monkeypatch,
):
    """No cross-thread observer/cancel is queued onto stopped or closed loops."""

    stopped_loop = asyncio.new_event_loop()
    stopped_source = stopped_loop.create_future()
    stopped_dispatches = []
    monkeypatch.setattr(
        stopped_loop,
        "call_soon_threadsafe",
        lambda *args: stopped_dispatches.append(args),
    )
    stopped = isolated_runtime._create_host_owned_facade_task(
        stopped_source, name="stopped-foreign-facade"
    )
    stopped.cancel()
    assert stopped.done() is False
    assert stopped_dispatches == []

    # Even a terminal Future stays fenced until its owner loop consumes it.
    # Future subclasses may require that ``result()`` runs in that loop.
    completed_source = stopped_loop.create_future()
    completed_source.set_result(None)
    completed = isolated_runtime._create_host_owned_facade_task(
        completed_source, name="completed-stopped-foreign-facade"
    )
    completed.cancel()
    assert completed.done() is False
    assert completed.foreign_settlement_disposition is None
    assert completed._source is completed_source
    assert stopped_dispatches == []

    closed_loop = asyncio.new_event_loop()
    closed_source = closed_loop.create_future()
    closed_loop.close()
    closed = isolated_runtime._create_host_owned_facade_task(
        closed_source, name="closed-foreign-facade"
    )
    closed.cancel()
    assert closed.done() is False
    stopped_loop.close()


@pytest.mark.asyncio
async def test_precompleted_foreign_future_is_consumed_only_by_owner_after_resume():
    """A stopped pre-settled source fences, then settles on its owner thread."""

    foreign_loop = asyncio.new_event_loop()
    state = {"result_threads": []}
    running = threading.Event()

    class OwnerThreadFuture(asyncio.Future):
        def result(self):
            state["result_threads"].append(threading.get_ident())
            assert threading.get_ident() == state["owner_thread"]
            return super().result()

    source = OwnerThreadFuture(loop=foreign_loop)
    source.set_result("stopped")
    operation = isolated_runtime._create_host_owned_facade_task(
        source, name="precompleted-stopped-foreign-owner-thread"
    )
    notified = asyncio.Event()
    operation.add_done_callback(lambda _completed: notified.set())
    assert operation.done() is False
    assert state["result_threads"] == []

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)
        state["owner_thread"] = threading.get_ident()
        running.set()
        foreign_loop.run_forever()
        foreign_loop.close()

    foreign_thread = threading.Thread(target=run_foreign_loop)
    foreign_thread.start()
    assert running.wait(timeout=1)
    try:
        # Retrying the retained operation asks the now-running owner loop to
        # acknowledge the exact source; it does not issue another facade call.
        operation.cancel()
        await asyncio.wait_for(notified.wait(), timeout=1)
        assert operation.done() is True
        assert operation.foreign_settlement_disposition == "succeeded"
        assert state["result_threads"] == [state["owner_thread"]]
    finally:
        if foreign_thread.is_alive():
            foreign_loop.call_soon_threadsafe(foreign_loop.stop)
            await asyncio.to_thread(foreign_thread.join)


@pytest.mark.asyncio
async def test_terminal_retirement_retries_stopped_foreign_operation_after_owner_restarts(
    monkeypatch, tmp_path
):
    """A retained foreign stop is retried on restart without another facade stop."""

    foreign_loop = asyncio.new_event_loop()
    foreign_running = threading.Event()
    state = {"result_threads": []}

    class OwnerThreadFuture(asyncio.Future):
        def result(self):
            state["result_threads"].append(threading.get_ident())
            assert threading.get_ident() == state["owner_thread"]
            return super().result()

    foreign_stop = OwnerThreadFuture(loop=foreign_loop)

    class CrossLoopStopClient(FakeIsolatedClient):
        def __init__(self):
            super().__init__()
            self.stop_calls = 0
            self.stopped = False

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                return foreign_stop
            self.stopped = True
            return None

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)
        state["owner_thread"] = threading.get_ident()
        foreign_running.set()
        foreign_loop.run_forever()
        foreign_loop.close()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = CrossLoopStopClient()
    feature._retain_terminal_retirement_client(client)

    foreign_thread = None
    try:
        # Initial retirement fences the exact source because its owner loop is
        # stopped.  The source's result must not be consumed from this thread.
        assert await feature._retire_terminal_clients() is False
        assert client.stop_calls == 1
        assert len(feature._terminal_lifecycle_tasks) == 1
        operation = feature._terminal_lifecycle_tasks[0].task
        assert operation.done() is False
        assert state["result_threads"] == []

        foreign_thread = threading.Thread(target=run_foreign_loop)
        foreign_thread.start()
        assert foreign_running.wait(timeout=1)

        # The next cleanup retries cancellation plus observation on the same
        # retained operation.  It remains fenced for this pass, never issuing
        # a duplicate facade stop while owner-loop acknowledgement is pending.
        assert await feature._retire_terminal_clients() is False
        assert client.stop_calls == 1

        for _ in range(100):
            if not feature._terminal_lifecycle_tasks:
                break
            await asyncio.sleep(0)
        assert feature._terminal_lifecycle_tasks == []
        assert operation.foreign_settlement_disposition == "cancelled"
        assert state["result_threads"] == [state["owner_thread"]]
        assert feature._terminal_retirement_clients == [client]

        # A cancelled foreign stop leaves the client fail-closed, but after
        # acknowledgement a bounded later retirement may make the one retry.
        assert await feature._retire_terminal_clients() is True
        assert client.stop_calls == 2
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
    finally:
        if foreign_thread is not None and foreign_thread.is_alive():
            foreign_loop.call_soon_threadsafe(foreign_loop.stop)
            await asyncio.to_thread(foreign_thread.join)
        elif not foreign_loop.is_closed():
            foreign_loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("result", "exception"))
async def test_foreign_terminal_source_is_detached_when_host_delivery_is_stranded(
    monkeypatch, caplog, outcome
):
    """A host callback failure cannot retain a foreign tenant terminal value."""

    foreign_loop = asyncio.new_event_loop()
    secret = "STRANDED-FOREIGN-TERMINAL-SECRET-2755"

    class TenantValue:
        pass

    class TenantError(RuntimeError):
        pass

    source = foreign_loop.create_future()
    value = TenantValue()
    value.secret = secret
    value_ref = weakref.ref(value)
    operation = isolated_runtime._create_host_owned_facade_task(
        source, name=f"stranded-foreign-{outcome}"
    )
    # The foreign owner loop is stopped, so construction cannot queue an
    # observer. Model its eventual owner-loop consumption below while the host
    # is already unable to accept the resulting callback.
    monkeypatch.setattr(
        operation._host_loop,
        "call_soon_threadsafe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("closing")),
    )
    if outcome == "result":
        source.set_result(value)
        error = None
    else:
        error = TenantError(secret)
        error.value = value
        source.set_exception(error)

    # The owner-loop disposition is terminal, but the closing host loop cannot
    # deliver its release callback. Detachment must precede that attempt.
    operation._observe_foreign_source_settlement(source)

    assert operation.done() is True
    assert operation._source is None
    del source, value, error
    gc.collect()
    assert value_ref() is None
    assert secret not in caplog.text
    foreign_loop.close()


@pytest.mark.asyncio
async def test_foreign_facade_dispatch_race_fails_closed_without_escaping(monkeypatch):
    """A loop that closes between inspection and dispatch cannot lose ownership."""

    foreign_loop = asyncio.new_event_loop()
    source = foreign_loop.create_future()
    original_is_running = foreign_loop.is_running
    monkeypatch.setattr(foreign_loop, "is_running", lambda: True)
    monkeypatch.setattr(
        foreign_loop,
        "call_soon_threadsafe",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("loop is closing")),
    )
    operation = isolated_runtime._create_host_owned_facade_task(
        source, name="racing-foreign-facade"
    )

    operation.cancel()
    assert operation.done() is False
    monkeypatch.setattr(foreign_loop, "is_running", original_is_running)
    foreign_loop.close()


@pytest.mark.asyncio
async def test_foreign_facade_owner_loop_consumes_secret_error_before_release(caplog):
    """Foreign failure consumption happens in its owner loop with no warning leak."""

    foreign_loop = asyncio.new_event_loop()
    ready = threading.Event()
    state = {}
    secret = "FOREIGN-FACADE-SECRET-2755"

    class ForeignFacadeError(RuntimeError):
        pass

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)

        def create_source():
            state["source"] = foreign_loop.create_future()
            ready.set()

        foreign_loop.call_soon(create_source)
        foreign_loop.run_forever()
        foreign_loop.close()

    foreign_thread = threading.Thread(target=run_foreign_loop)
    foreign_thread.start()
    assert ready.wait(timeout=1)
    operation = isolated_runtime._create_host_owned_facade_task(
        state["source"], name="foreign-secret-consumption"
    )
    notified = asyncio.Event()
    operation.add_done_callback(lambda _completed: notified.set())

    def fail_source():
        state["source"].set_exception(ForeignFacadeError(secret))
        foreign_loop.call_soon(foreign_loop.stop)

    foreign_loop.call_soon_threadsafe(fail_source)
    try:
        await asyncio.wait_for(notified.wait(), timeout=1)
        await asyncio.wait_for(asyncio.to_thread(foreign_thread.join), timeout=1)
        assert operation.done() is True
    finally:
        if foreign_thread.is_alive():
            foreign_loop.call_soon_threadsafe(foreign_loop.stop)
            await asyncio.to_thread(foreign_thread.join)

    state.clear()
    del operation
    gc.collect()
    assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_disposition"),
    (("success", "succeeded"), ("error", "failed"), ("cancel", "cancelled")),
)
async def test_foreign_pre_settled_source_is_observed_before_owner_loop_stops(
    outcome, expected_disposition, caplog
):
    """A pre-settled foreign source cannot strand its observer behind stop."""

    foreign_loop = asyncio.new_event_loop()
    ready = threading.Event()
    state = {}
    secret = "FOREIGN-PRESETTLED-SECRET-2755"

    class ForeignPresettledError(RuntimeError):
        pass

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)

        def create_source():
            source = foreign_loop.create_future()
            if outcome == "success":
                source.set_result({"secret": secret})
            elif outcome == "error":
                source.set_exception(ForeignPresettledError(secret))
            else:
                source.cancel()
            state["source"] = source
            ready.set()

        foreign_loop.call_soon(create_source)
        foreign_loop.run_forever()
        foreign_loop.close()

    foreign_thread = threading.Thread(target=run_foreign_loop)
    foreign_thread.start()
    assert ready.wait(timeout=1)
    operation = isolated_runtime._create_host_owned_facade_task(
        state["source"], name=f"foreign-pre-settled-{outcome}"
    )
    notified = asyncio.Event()
    operation.add_done_callback(lambda _completed: notified.set())
    # The observer registration is already queued ahead of this stop.  Its
    # owner-loop inline path must consume a completed source in that same turn
    # rather than enqueueing one more callback which this stop would strand.
    foreign_loop.call_soon_threadsafe(foreign_loop.stop)
    try:
        await asyncio.wait_for(notified.wait(), timeout=1)
        await asyncio.wait_for(asyncio.to_thread(foreign_thread.join), timeout=1)
        assert operation.done() is True
        assert operation.foreign_settlement_disposition == expected_disposition
        assert state["source"]._log_traceback is False
    finally:
        if foreign_thread.is_alive():
            foreign_loop.call_soon_threadsafe(foreign_loop.stop)
            await asyncio.to_thread(foreign_thread.join)

    state.clear()
    del operation
    gc.collect()
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_owned_health_probe_child_cancellation_is_an_ordinary_failure():
    """A self-cancelled facade health task must not cancel its supervisor."""

    async def cancelled_health():
        asyncio.current_task().cancel()
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="health probe was cancelled") as failed:
        await isolated_runtime._await_owned_health_probe(
            cancelled_health(),
            name="self-cancelled-health",
            on_started=lambda _task: None,
            on_late_task=lambda _task: pytest.fail("settled child must not detach"),
        )

    assert failed.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", (False, True))
async def test_lifecycle_argless_first_cancellation_preserves_count_without_redelivery(
    repeat,
):
    """An arg-less first cancel is authoritative and leaves no synthetic wakeup."""

    owner_ref = {}

    async def child():
        owner = owner_ref["owner"]
        owner.cancel()
        if repeat:
            owner.cancel("later cancellation reason")
        asyncio.current_task().cancel("child cancellation")
        await asyncio.sleep(0)

    async def owner():
        task = asyncio.current_task()
        try:
            await isolated_runtime._await_owned_facade_lifecycle_operation(
                child(),
                name="argless-first-lifecycle-cancellation",
                on_late_task=lambda _task: pytest.fail("child must settle promptly"),
            )
        except asyncio.CancelledError as cancelled:
            args = cancelled.args
            count = task.cancelling()
            while task.cancelling():
                task.uncancel()
            await asyncio.sleep(0)
            return args, count

    owner_task = asyncio.create_task(owner())
    owner_ref["owner"] = owner_task
    args, count = await asyncio.wait_for(owner_task, timeout=1)
    assert args == ()
    assert count == (2 if repeat else 1)


@pytest.mark.asyncio
async def test_stale_cancellation_count_does_not_reclassify_self_cancelled_health():
    """A caught historical cancel is not a parent cancel for a later probe."""

    owner = asyncio.current_task()
    owner.cancel("historical cancellation")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.sleep(0)
    assert owner.cancelling() == 1

    async def self_cancelled_health():
        asyncio.current_task().cancel()
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="health probe was cancelled"):
        await isolated_runtime._await_owned_health_probe(
            self_cancelled_health(),
            name="stale-cancellation-health",
            on_started=lambda _task: None,
            on_late_task=lambda _task: pytest.fail("settled child must not detach"),
        )

    assert owner.cancelling() == 1
    owner.uncancel()


@pytest.mark.asyncio
async def test_stale_cancellation_count_does_not_reclassify_lifecycle_success():
    """Historical cancellation state cannot turn a settled facade into cancel."""

    owner = asyncio.current_task()
    owner.cancel("historical lifecycle cancellation")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.sleep(0)
    assert owner.cancelling() == 1

    source = asyncio.get_running_loop().create_future()
    source.set_result("stopped")
    assert await isolated_runtime._await_owned_facade_lifecycle_operation(
        source,
        name="stale-cancellation-lifecycle-success",
        on_late_task=lambda _task: pytest.fail("settled facade must not detach"),
    ) == "stopped"
    assert owner.cancelling() == 1
    owner.uncancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stale_cancellation_count_does_not_reclassify_lifecycle_timeout_ack(
    monkeypatch,
):
    """A historical count cannot replace the timeout during child acknowledgement."""

    owner = asyncio.current_task()
    owner.cancel("historical lifecycle cancellation")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.sleep(0)
    assert owner.cancelling() == 1

    async def stop():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return None

    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", 0.1)
    with pytest.raises(isolated_runtime._FacadeLifecycleOperationTimedOut):
        await isolated_runtime._await_owned_facade_lifecycle_operation(
            stop(),
            name="stale-cancellation-lifecycle-timeout-ack",
            on_late_task=lambda _task: pytest.fail("stop should settle promptly"),
        )
    assert owner.cancelling() == 1
    owner.uncancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_terminal_health_failure_after_cancel_does_not_abort_retirement(
    monkeypatch, tmp_path, caplog
):
    """A terminal probe's ordinary post-cancel error still permits one stop."""

    probe_started = asyncio.Event()
    secret = "TERMINAL-HEALTH-SECRET-2755"

    async def health():
        probe_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError(secret)

    class CountingStopClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = CountingStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    probe = isolated_runtime._create_host_owned_facade_task(
        health(), name="terminal-health-post-cancel-error"
    )
    feature._own_health_probe_task(probe)
    await asyncio.wait_for(probe_started.wait(), timeout=1)

    await asyncio.wait_for(feature.shutdown(), timeout=1)

    assert client.stop_calls == 1
    assert feature._terminal_health_probe_task is None
    assert feature._terminal_retirement_clients == []
    assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("child_cancels_first", (False, True))
@pytest.mark.parametrize("cancel_count", (1, 2))
async def test_lifecycle_parent_cancellation_payload_wins_same_turn_as_child(
    child_cancels_first, cancel_count
):
    """A child CancelledError never supplies a same-turn parent reason."""

    owner_ref = {}

    async def child():
        if child_cancels_first:
            asyncio.current_task().cancel()
        owner_ref["owner"].cancel("first parent cancellation")
        if cancel_count == 2:
            owner_ref["owner"].cancel("later parent cancellation")
        if not child_cancels_first:
            asyncio.current_task().cancel()
        await asyncio.sleep(0)

    owner = asyncio.create_task(
        isolated_runtime._await_owned_facade_lifecycle_operation(
            child(),
            name="same-turn-parent-lifecycle-cancellation",
            on_late_task=lambda _task: pytest.fail("child must settle promptly"),
        )
    )
    owner_ref["owner"] = owner

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(owner, timeout=1)

    assert cancelled.value.args == ("first parent cancellation",)
    assert owner.cancelled() is True
    assert owner.cancelling() == cancel_count


@pytest.mark.asyncio
@pytest.mark.parametrize("child_cancels_first", (False, True))
@pytest.mark.parametrize("cancel_count", (1, 2))
async def test_health_parent_cancellation_payload_wins_same_turn_as_child(
    child_cancels_first, cancel_count
):
    """Health retains ordinary child classification absent an accepted parent."""

    owner_ref = {}

    async def child():
        if child_cancels_first:
            asyncio.current_task().cancel()
        owner_ref["owner"].cancel("first health parent cancellation")
        if cancel_count == 2:
            owner_ref["owner"].cancel("later health parent cancellation")
        if not child_cancels_first:
            asyncio.current_task().cancel()
        await asyncio.sleep(0)

    owner = asyncio.create_task(
        isolated_runtime._await_owned_health_probe(
            child(),
            name="same-turn-parent-health-cancellation",
            on_started=lambda _task: None,
            on_late_task=lambda _task: pytest.fail("child must settle promptly"),
        )
    )
    owner_ref["owner"] = owner

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(owner, timeout=1)

    assert cancelled.value.args == ("first health parent cancellation",)
    assert owner.cancelled() is True
    assert owner.cancelling() == cancel_count


@pytest.mark.asyncio
async def test_supervisor_restarts_after_child_cancelled_health_probe(monkeypatch, tmp_path):
    """A child cancellation follows normal health recovery rather than terminal unwind."""

    first_health = asyncio.Event()
    restarted = asyncio.Event()
    real_sleep = asyncio.sleep

    class ChildCancelledHealthClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.health_calls = 0
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            self.health_calls += 1
            if self.health_calls == 1:
                first_health.set()
                asyncio.current_task().cancel()
                await real_sleep(0)
            return True

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

        async def start(self):
            self.start_calls += 1
            self.started = True
            restarted.set()

    async def immediate_sleep(_delay):
        await real_sleep(0)

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    client = ChildCancelledHealthClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    supervisor = asyncio.create_task(feature._supervise())
    feature._supervision_task = supervisor
    try:
        await asyncio.wait_for(first_health.wait(), timeout=1)
        await asyncio.wait_for(restarted.wait(), timeout=1)
        assert client.stop_calls == 1
        assert client.start_calls == 1
        assert feature._traffic_gate.sealed is False
    finally:
        feature._stopping = True
        if not supervisor.done():
            supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)


@pytest.mark.asyncio
async def test_supervisor_logs_neutral_health_and_restart_failures(monkeypatch, tmp_path, caplog):
    """Compatible facade failures never put tenant exception text in logs."""

    health_attempted = asyncio.Event()
    restart_attempted = asyncio.Event()
    secret = "ISOLATED-FACADE-TENANT-SECRET-2755"
    real_sleep = asyncio.sleep

    class SecretFailingClient(FakeIsolatedClient):
        def health(self):
            health_attempted.set()
            raise RuntimeError(secret)

        async def stop(self):
            return None

        def start(self):
            restart_attempted.set()
            raise RuntimeError(secret)

    async def immediate_sleep(_delay):
        await real_sleep(0)

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = SecretFailingClient()
    supervisor = asyncio.create_task(feature._supervise())
    feature._supervision_task = supervisor
    try:
        await asyncio.wait_for(health_attempted.wait(), timeout=1)
        await asyncio.wait_for(restart_attempted.wait(), timeout=1)
        assert "health check failed" in caplog.text
        assert "restart failed" in caplog.text
        assert secret not in caplog.text
    finally:
        feature._stopping = True
        if not supervisor.done():
            supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)


@pytest.mark.asyncio
async def test_owned_health_probe_timeout_retains_task_returned_by_facade(monkeypatch):
    """A timed-out Task-returning health facade cannot become an unowned probe."""

    cancellation_observed = asyncio.Event()
    release = asyncio.Event()

    async def pending_task():
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_observed.set()

    operation = asyncio.create_task(pending_task(), name="facade-hostile-health-task")

    class TaskReturningFacade:
        def health(self):
            return operation

    late_tasks = []
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_CANCELLATION_GRACE", 0.01)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await isolated_runtime._await_owned_health_probe(
                TaskReturningFacade().health(),
                name="test-hostile-facade-health-task",
                on_started=lambda _task: None,
                on_late_task=late_tasks.append,
            )
        assert cancellation_observed.is_set()
        assert len(late_tasks) == 1
        assert late_tasks[0].get_name() == "test-hostile-facade-health-task"
    finally:
        release.set()
        await asyncio.wait_for(late_tasks[0] if late_tasks else operation, timeout=1)


@pytest.mark.asyncio
async def test_same_loop_health_parent_cancellation_claims_original_before_wrapper_runs():
    """Health cancellation reaches the facade Task, never only an observer."""

    started = asyncio.Event()
    release = asyncio.Event()
    cancellation_observed = asyncio.Event()

    async def hostile_health():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_observed.set()

    original = asyncio.create_task(hostile_health(), name="facade-original-health")
    started_tasks = []
    late_tasks = []
    owner = asyncio.create_task(
        isolated_runtime._await_owned_health_probe(
            original,
            name="host-owned-prestart-health",
            on_started=started_tasks.append,
            on_late_task=late_tasks.append,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        owner.cancel("health parent cancellation")
        await asyncio.wait_for(cancellation_observed.wait(), timeout=1)

        assert original.done() is False
        assert len(started_tasks) == 1
        assert late_tasks == started_tasks
        assert late_tasks[0].get_name() == "host-owned-prestart-health"

        release.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(owner, timeout=1)
        assert cancelled.value.args == ("health parent cancellation",)
    finally:
        release.set()
        if not original.done():
            original.cancel()
        await asyncio.gather(original, return_exceptions=True)


@pytest.mark.asyncio
async def test_cross_loop_health_task_fails_closed_and_consumes_late_error():
    """A foreign health Task cannot trigger an ordinary retry while it runs."""

    foreign_loop = asyncio.new_event_loop()
    foreign_ready = threading.Event()
    release_foreign = threading.Event()
    cancellation_observed = threading.Event()
    foreign_state = {}

    class ForeignHealthError(RuntimeError):
        pass

    async def foreign_health():
        while not release_foreign.is_set():
            try:
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                cancellation_observed.set()
        raise ForeignHealthError("FOREIGN-HEALTH-SECRET")

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)
        task = foreign_loop.create_task(foreign_health(), name="foreign-facade-health")
        foreign_state["task"] = task
        foreign_ready.set()
        try:
            foreign_loop.run_until_complete(task)
        except ForeignHealthError:
            # The host observer has already retrieved this private outcome.
            pass
        finally:
            foreign_loop.close()

    foreign_thread = threading.Thread(target=run_foreign_loop)
    foreign_thread.start()
    assert foreign_ready.wait(timeout=1)

    started_tasks = []
    late_tasks = []
    try:
        with pytest.raises(isolated_runtime._CrossLoopFacadeOperationError):
            await isolated_runtime._await_owned_health_probe(
                foreign_state["task"],
                name="host-owned-foreign-health",
                on_started=started_tasks.append,
                on_late_task=late_tasks.append,
            )
        assert late_tasks == started_tasks
        assert len(late_tasks) == 1
        assert late_tasks[0].foreign_loop is True
        assert foreign_state["task"].done() is False
        assert cancellation_observed.wait(timeout=1)

        release_foreign.set()
        await asyncio.wait_for(asyncio.to_thread(foreign_thread.join), timeout=1)
        for _ in range(100):
            if late_tasks[0].done():
                break
            await asyncio.sleep(0)
        assert late_tasks[0].done() is True
    finally:
        release_foreign.set()
        if foreign_thread.is_alive():
            await asyncio.to_thread(foreign_thread.join)


@pytest.mark.asyncio
async def test_terminal_cleanup_never_stops_beside_unacknowledged_foreign_health(
    monkeypatch, tmp_path
):
    """A cross-loop health fence remains terminally incomplete, never retried."""

    foreign_loop = asyncio.new_event_loop()
    foreign_probe = foreign_loop.create_future()

    class StopCountingClient(FakeIsolatedClient):
        def __init__(self):
            super().__init__()
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = StopCountingClient()
    feature._client = client
    probe = isolated_runtime._create_host_owned_facade_task(
        foreign_probe, name="unacknowledged-foreign-terminal-health"
    )
    feature._own_health_probe_task(probe)

    try:
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 0
        assert feature._terminal_cleanup_uncertain is True
        assert feature._terminal_health_probe_task is probe
        assert probe.done() is False
    finally:
        foreign_loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("nonweak_facade", (False, True))
@pytest.mark.parametrize("foreign_outcome", ("success", "error", "cancel"))
async def test_cross_loop_facade_task_is_retained_fenced_and_released_before_retry(
    nonweak_facade, foreign_outcome, monkeypatch, tmp_path
):
    """Only a foreign success releases its exact retained stop ownership."""

    foreign_loop = asyncio.new_event_loop()
    foreign_ready = threading.Event()
    release_foreign = threading.Event()
    cancellation_observed = threading.Event()
    foreign_state = {}

    class ForeignStopError(RuntimeError):
        pass

    async def foreign_stop():
        while not release_foreign.is_set():
            try:
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                cancellation_observed.set()
        if foreign_outcome == "error":
            raise ForeignStopError("FOREIGN-STOP-SECRET-2755")
        if foreign_outcome == "cancel":
            raise asyncio.CancelledError()

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)
        task = foreign_loop.create_task(foreign_stop(), name="foreign-facade-stop")
        foreign_state["task"] = task
        foreign_ready.set()
        try:
            foreign_loop.run_until_complete(task)
        except (ForeignStopError, asyncio.CancelledError):
            # The host's owner-loop observer has already consumed the private
            # disposition; the loop runner only needs to terminate cleanly.
            pass
        finally:
            foreign_loop.close()

    foreign_thread = threading.Thread(target=run_foreign_loop)
    foreign_thread.start()
    assert foreign_ready.wait(timeout=1)

    if nonweak_facade:
        class CrossLoopStopClient:
            __slots__ = ("stop_calls", "stopped")

            def __init__(self):
                self.stop_calls = 0
                self.stopped = False

            def stop(self):
                self.stop_calls += 1
                if self.stop_calls == 1:
                    return foreign_state["task"]
                self.stopped = True
                return None
    else:
        class CrossLoopStopClient(FakeIsolatedClient):
            def __init__(self):
                super().__init__()
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1
                if self.stop_calls == 1:
                    return foreign_state["task"]
                self.stopped = True
                return None

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = CrossLoopStopClient()
    feature._client = client
    try:
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 1
        assert len(feature._terminal_lifecycle_tasks) == 1
        assert feature._terminal_lifecycle_tasks[0].task.foreign_loop is True
        assert foreign_state["task"].done() is False
        assert cancellation_observed.wait(timeout=1)

        # A fresh terminal caller must not retry while the foreign operation
        # still owns the facade's lifecycle handle.
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 1

        release_foreign.set()
        await asyncio.wait_for(asyncio.to_thread(foreign_thread.join), timeout=1)
        for _ in range(100):
            if not feature._terminal_lifecycle_tasks:
                break
            await asyncio.sleep(0)
        assert feature._terminal_lifecycle_tasks == []

        # A successful foreign source releases its exact retained client. A
        # failed/cancelled source remains fail-closed, but only after its first
        # source has settled may a later shutdown attempt a fresh ``stop()``.
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == (1 if foreign_outcome == "success" else 2)
        assert client.stopped is (foreign_outcome != "success")
        assert feature._terminal_retirement_clients == []
    finally:
        release_foreign.set()
        if foreign_thread.is_alive():
            await asyncio.to_thread(foreign_thread.join)
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_foreign_success_keeps_terminal_stop_owned_until_host_disposition_runs(
    monkeypatch, tmp_path
):
    """A host cleanup between owner acknowledgement and callback cannot re-stop."""

    foreign_loop = asyncio.new_event_loop()
    foreign_stop = foreign_loop.create_future()

    class DeferredForeignStopClient(FakeIsolatedClient):
        def __init__(self):
            super().__init__()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                return foreign_stop
            pytest.fail("terminal cleanup issued a duplicate foreign stop")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = DeferredForeignStopClient()
    feature._retain_terminal_retirement_client(client)

    try:
        with pytest.raises(isolated_runtime._CrossLoopFacadeOperationError):
            await isolated_runtime._await_owned_facade_lifecycle_operation(
                client.stop(),
                name="foreign-success-host-disposition-gap",
                on_completed=lambda: feature._release_terminal_retirement_client(client),
                on_timeout=lambda: feature._fence_terminal_retirement_timeout(client),
                on_late_task=lambda task: feature._retain_terminal_lifecycle_task(
                    task, client
                ),
            )
        operation = feature._terminal_lifecycle_tasks[0].task

        # Model the source loop consuming its successful result just before
        # the host handles its queued disposition callback.  No host turn is
        # allowed between these two assertions.
        foreign_stop.set_result(None)
        operation._observe_foreign_source_settlement(foreign_stop)
        assert operation.done() is True
        assert operation.settlement_delivery_complete is False

        assert await feature._retire_terminal_clients() is False
        assert client.stop_calls == 1

        for _ in range(4):
            await asyncio.sleep(0)
        assert feature._terminal_lifecycle_tasks == []
        assert feature._terminal_retirement_clients == []
        assert client.stop_calls == 1
    finally:
        foreign_loop.close()


@pytest.mark.asyncio
async def test_owned_facade_stop_replays_cancellation_before_late_secret_error():
    """A late facade error cannot replace a remembered caller cancellation."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    class SecretStopError(RuntimeError):
        pass

    async def stop():
        stop_started.set()
        await release_stop.wait()
        # Keep the owner in its next shielded await while this facade fails.
        # Without that scheduling edge, the owner can observe task.done() at
        # the top of its loop and miss the path this test protects.
        await asyncio.sleep(0)
        raise SecretStopError("TENANT-STOP-SECRET")

    owner = asyncio.create_task(
        isolated_runtime._await_owned_facade_lifecycle_operation(
            stop(),
            name="test-late-secret-stop",
            on_late_task=lambda _task: pytest.fail("stop should settle promptly"),
        )
    )
    await asyncio.wait_for(stop_started.wait(), timeout=1)
    owner.cancel("reload cancellation")
    await asyncio.sleep(0)
    assert owner.done() is False
    release_stop.set()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(owner, timeout=1)

    assert cancelled.value.args == ("reload cancellation",)
    assert "TENANT-STOP-SECRET" not in repr(cancelled.value)
    assert cancelled.value.__context__ is None
    _assert_runtime_traceback_locals_are_detached(
        cancelled.value,
        forbidden_values=(),
    )


@pytest.mark.asyncio
async def test_owned_facade_cancellation_traceback_drops_late_secret_result():
    """Cancellation does not retain a completed facade task's result."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()
    secret_result = {"token": "TENANT-STOP-RESULT-SECRET"}

    async def stop():
        stop_started.set()
        await release_stop.wait()
        return secret_result

    owner = asyncio.create_task(
        isolated_runtime._await_owned_facade_lifecycle_operation(
            stop(),
            name="test-late-secret-stop-result",
            on_late_task=lambda _task: pytest.fail("stop should settle promptly"),
        )
    )
    await asyncio.wait_for(stop_started.wait(), timeout=1)
    owner.cancel("reload result cancellation")
    await asyncio.sleep(0)
    release_stop.set()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(owner, timeout=1)

    assert cancelled.value.args == ("reload result cancellation",)
    assert not _traceback_locals_reach_any(
        cancelled.value,
        (secret_result,),
    )


@pytest.mark.asyncio
async def test_owned_facade_timeout_acknowledges_child_cancellation_as_timeout(
    monkeypatch,
):
    """The helper's own task.cancel() never becomes a caller cancellation."""

    stop_started = asyncio.Event()
    secret = {"credential": "TIMED-OUT-STOP-SECRET-2755"}

    async def stop():
        stop_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The acknowledgement receives the child's cancellation first.
            # A late secret failure must stay private behind the timeout.
            await asyncio.sleep(0)
            raise RuntimeError(secret)

    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", 0.1)
    with pytest.raises(isolated_runtime._FacadeLifecycleOperationTimedOut) as timed_out:
        await isolated_runtime._await_owned_facade_lifecycle_operation(
            stop(),
            name="test-child-cancellation-timeout",
            on_late_task=lambda _task: pytest.fail("stop should settle promptly"),
        )

    assert stop_started.is_set()
    assert "TIMED-OUT-STOP-SECRET-2755" not in repr(timed_out.value)
    assert timed_out.value.__context__ is None
    _assert_runtime_traceback_locals_are_detached(
        timed_out.value,
        forbidden_values=(secret,),
    )


def test_facade_lifecycle_timeout_covers_locked_sdk_stop_sequence():
    """The host deadline tracks every sequential SDK 0.35.1 stop observation."""

    assert isolated_runtime._SDK_SUBPROCESS_STOP_BUDGET == 33.0
    assert isolated_runtime._FACADE_LIFECYCLE_OPERATION_TIMEOUT == 36.0


@pytest.mark.asyncio
async def test_health_timeout_acknowledgement_does_not_swallow_supervisor_cancellation(
    monkeypatch, tmp_path
):
    """A cancellation concurrent with probe acknowledgement never restarts."""

    health_started = asyncio.Event()
    supervisor: asyncio.Task[None] | None = None

    class TimeoutAcknowledgingClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            health_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                assert supervisor is not None
                supervisor.cancel("supervisor health cancellation")
                raise

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

        async def start(self):
            self.start_calls += 1
            self.started = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_TIMEOUT", 0.01)
    client = TimeoutAcknowledgingClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    try:
        supervisor = asyncio.create_task(feature._supervise())
        feature._supervision_task = supervisor
        await asyncio.wait_for(health_started.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(supervisor, timeout=1)

        assert cancelled.value.args == ("supervisor health cancellation",)
        assert client.start_calls == 0
        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
    finally:
        if supervisor is not None and not supervisor.done():
            supervisor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await supervisor
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_health_timeout_acknowledgement_consumes_late_facade_secret(
    monkeypatch,
):
    """A post-timeout health failure stays private behind TimeoutError."""

    probe_started = asyncio.Event()
    secret = {"credential": "TENANT-HEALTH-SECRET"}

    async def health():
        probe_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Force the helper to await the task after cancellation before it
            # fails, rather than observing the completed task at loop entry.
            await asyncio.sleep(0)
            raise RuntimeError(secret)

    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_CANCELLATION_GRACE", 0.1)
    with pytest.raises(asyncio.TimeoutError) as timed_out:
        await isolated_runtime._await_owned_health_probe(
            health(),
            name="test-late-secret-health",
            on_started=lambda _task: None,
            on_late_task=lambda _task: pytest.fail("health should settle promptly"),
        )

    assert probe_started.is_set()
    assert "TENANT-HEALTH-SECRET" not in repr(timed_out.value)
    _assert_runtime_traceback_locals_are_detached(
        timed_out.value,
        forbidden_values=(secret,),
    )


@pytest.mark.asyncio
async def test_reload_cancellation_after_successful_stop_never_restores_or_retires_twice(
    monkeypatch, tmp_path
):
    """Reload keeps a stop proven during cancellation out of quarantine retry."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    class SlowStopClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            stop_started.set()
            await release_stop.wait()
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = SlowStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    reload_task = asyncio.create_task(feature.reload())
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        reload_task.cancel("reload stop cancellation")
        await asyncio.sleep(0)
        assert reload_task.done() is False
        release_stop.set()

        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(reload_task, timeout=1)

        assert cancelled.value.args == ("reload stop cancellation",)
        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        release_stop.set()
        if not reload_task.done():
            reload_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reload_task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_recovery_later_failure_does_not_retry_successfully_stopped_facade(
    monkeypatch, tmp_path
):
    """Fenced recovery quarantines later failure without a duplicate old stop."""

    class CountingStopClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

    async def promotion_failure(_transition):
        raise RuntimeError("later promotion failure")

    async def clear_pending(_transition):
        return None

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = CountingStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    monkeypatch.setattr(feature, "_promote_config", promotion_failure)
    monkeypatch.setattr(feature, "_clear_owned_pending_before_quarantine", clear_pending)
    try:
        with pytest.raises(RuntimeError, match="later promotion failure"):
            await feature._recover_fenced_transition_uninterrupted(Mock())

        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_recovery_nonweak_facade_does_not_retry_successful_stop(
    monkeypatch, tmp_path
):
    """The proven-stop fence is identity-safe for non-weakrefable facades."""

    class NonWeakCountingStopClient:
        __slots__ = ("stop_calls", "stopped")

        def __init__(self):
            self.stop_calls = 0
            self.stopped = False

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

    async def promotion_failure(_transition):
        raise RuntimeError("later promotion failure")

    async def clear_pending(_transition):
        return None

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = NonWeakCountingStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    monkeypatch.setattr(feature, "_promote_config", promotion_failure)
    monkeypatch.setattr(feature, "_clear_owned_pending_before_quarantine", clear_pending)
    try:
        with pytest.raises(RuntimeError, match="later promotion failure"):
            await feature._recover_fenced_transition_uninterrupted(Mock())

        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("nonweakrefable", [False, True])
async def test_fenced_recovery_cancellation_does_not_restore_proven_stopped_facade(
    monkeypatch, tmp_path, nonweakrefable
):
    """Cancellation after the exact stop settles cannot schedule a second stop."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    if nonweakrefable:

        class SlowStopClient:
            __slots__ = ("stop_calls", "stopped")

            def __init__(self):
                self.stop_calls = 0
                self.stopped = False

            async def stop(self):
                self.stop_calls += 1
                stop_started.set()
                await release_stop.wait()
                self.stopped = True

    else:

        class SlowStopClient(FakeIsolatedClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.stop_calls = 0

            async def stop(self):
                self.stop_calls += 1
                stop_started.set()
                await release_stop.wait()
                self.stopped = True

    async def clear_pending(_transition):
        return None

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = SlowStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    monkeypatch.setattr(feature, "_clear_owned_pending_before_quarantine", clear_pending)
    recovery = asyncio.create_task(
        feature._recover_fenced_transition_uninterrupted(Mock())
    )
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        recovery.cancel("fenced recovery cancellation")
        await asyncio.sleep(0)
        assert recovery.done() is False
        release_stop.set()

        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(recovery, timeout=1)

        assert cancelled.value.args == ("fenced recovery cancellation",)
        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        release_stop.set()
        if not recovery.done():
            recovery.cancel()
            with pytest.raises(asyncio.CancelledError):
                await recovery
        if not feature._stopping:
            await feature.shutdown()
