"""Tests for isolated feature runtime proxy behavior."""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.isolated_runtime import (
    ProxyFeature,
    SchedulerExecutionContextUnavailable,
)
from kestrel_sovereign.features.scheduler.runner import (
    SchedulerExecution,
    SchedulerRunner,
    ScheduledTask,
    _SchedulerExecutionScope,
    _current_execution,
    get_current_scheduler_execution,
)


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
    agent = Mock()
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

    agent = Mock()
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
        async def _renew_lease(self, task):
            await asyncio.Future()

        async def _finalize(self, *args, **kwargs):
            return None

    agent = Mock()
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
    agent = Mock()
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
    agent = Mock()
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

    agent = Mock()
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
    agent = Mock()
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
    agent = Mock()
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
    agent = Mock()
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
    agent = Mock()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}

    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")
    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=client_factory)

    async def fake_load():
        return {"provider": "web", "allowed_senders": ["+13035551234"]}

    feature.load_persisted_config = fake_load  # type: ignore[assignment]
    await feature.initialize()

    assert captured["config"] == {
        "provider": "web",
        "allowed_senders": ["+13035551234"],
    }
    await feature.shutdown()


@pytest.mark.asyncio
async def test_proxy_bridges_channel_capability_into_registry(monkeypatch, tmp_path):
    """A service advertising a channel capability is registered as a forwarding adapter,
    and channels_send-style routing reaches the service tool."""
    channel_feature = FakeChannelFeature()
    agent = Mock()
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
async def test_shutdown_does_not_evict_replacement_adapter(monkeypatch, tmp_path):
    """If another adapter replaced our channel_type, shutdown must not remove it."""
    channel_feature = FakeChannelFeature()
    agent = Mock()
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
    agent = Mock()
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
    from kestrel_sovereign.features.isolated_runtime import _delivery_receipt_from_result

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
    agent = Mock()
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
    import json
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

    agent = Mock()
    agent.features = {}
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=client_factory)
    feature._venv_path = tmp_path / "svc-venv"
    feature._bin_path = None

    feature._build_client()

    assert "env" in captured
    assert "PYTHONPATH" not in captured["env"]


class _FakeStorage:
    """Minimal graph store double: records add_node, serves get_node."""

    def __init__(self):
        self.nodes = {}

    async def add_node(self, node):
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)


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
    agent = Mock()
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
    node = agent.storage.nodes.get("feature_config:TestFeature")
    assert node is not None, "set_config did not persist the feature_config node"
    assert node.properties["config"]["allowed_senders"] == ["8825903191"]

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
async def test_persisted_config_survives_restart(monkeypatch, tmp_path):
    """A config set on one ProxyFeature is loaded by a fresh one (restart)."""
    agent = Mock()
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
    agent = Mock(storage_path=str(tmp_path / "a" / "db.db"), features={})
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
