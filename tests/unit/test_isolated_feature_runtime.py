"""Tests for isolated feature runtime proxy behavior."""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from kestrel_sdk.isolated_feature import ConfigTransitionError, ConfigTransitionResult
from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features import isolated_runtime
from kestrel_sovereign.features.isolated_runtime import (
    ProxyFeature,
    SchedulerExecutionContextUnavailable,
)
from kestrel_sovereign.features.scheduler.runner import (
    ScheduledTask,
    SchedulerExecution,
    SchedulerRunner,
    _current_execution,
    _SchedulerExecutionScope,
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
            self.fail_reads = True

        async def get_node(self, node_id):
            if self.fail_reads:
                raise OSError("storage temporarily unavailable")
            return await super().get_node(node_id)

    agent = Mock()
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

    assert agent.storage.nodes["feature_config:TestFeature"].properties["config"] == {
        "enabled": False,
        "token": stored_config["token"],
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

    agent = Mock(features={})
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
        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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

    agent = Mock(features={})
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
        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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

    agent = Mock()
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

    agent = Mock()
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
    assert agent.storage.nodes["feature_config:TestFeature"].properties["config"] == old_config


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
            if self.cas_calls == 2:
                raise OSError("storage offline during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class FailingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            raise ConfigTransitionError("config transition failed")

    agent = Mock()
    agent.features = {}
    agent.storage = RollbackFailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FailingTransitionClient)
    await feature.persist_config(old_config)
    await feature.initialize()

    with pytest.raises(ConfigTransitionError, match="config transition failed"):
        await feature.set_config(next_config)

    node = agent.storage.nodes["feature_config:TestFeature"]
    assert node.properties["config"] == old_config
    assert "pending_config" not in node.properties
    assert "_isolated_pending_generation" not in node.properties
    assert agent.storage.cas_calls == 3

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

    agent = Mock(features={})
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

        properties = agent.storage.nodes["feature_config:TestFeature"].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert "_isolated_pending_generation" not in properties

        await feature.set_config(retry_config)

        assert clients[0].preparations == 2
        assert feature._host_config == retry_config
        assert agent.storage.nodes["feature_config:TestFeature"].properties["config"] == retry_config
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

    agent = Mock(features={})
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

        staged = agent.storage.nodes["feature_config:TestFeature"].properties
        assert staged["pending_config"] == pending_config
        assert isinstance(staged["_isolated_pending_owner"], str)
        assert isinstance(staged["_isolated_pending_lease_expires_at"], str)

        update.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update

        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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

    agent = Mock(features={})
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
        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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
    first_agent = Mock(features={})
    first_agent.storage = storage
    first_agent.storage_path = str(tmp_path / "first" / "kestrel_prime.db")
    second_agent = Mock(features={})
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
        staged = dict(storage.nodes["feature_config:TestFeature"].properties)

        await second.initialize()
        with pytest.raises(RuntimeError, match="already in progress"):
            await second.set_config(second_config)

        assert storage.nodes["feature_config:TestFeature"].properties == staged
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
    agent = Mock(features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    seed = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await seed.persist_config(old_config)
    storage.nodes["feature_config:TestFeature"].properties.update(
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

        properties = storage.nodes["feature_config:TestFeature"].properties
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
    agent = Mock(features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    seed = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await seed.persist_config(old_config)
    storage.nodes["feature_config:TestFeature"].properties.update(
        {
            "pending_config": malformed_config,
            "_isolated_pending_generation": "malformed-generation",
            "_isolated_pending_owner": "malformed-owner",
            "_isolated_pending_lease_expires_at": "not-a-timestamp",
        }
    )
    original_properties = dict(storage.nodes["feature_config:TestFeature"].properties)
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    try:
        await feature.initialize()
        with pytest.raises(RuntimeError, match="stored pending config lease is invalid"):
            await feature.set_config(next_config)

        assert storage.nodes["feature_config:TestFeature"].properties == original_properties
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

    agent = Mock()
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

    properties = agent.storage.nodes["feature_config:TestFeature"].properties
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

    agent = Mock()
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
    agent = Mock()
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

    agent = Mock()
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
            # Stage candidate, then reject its promotion.
            if self.cas_calls == 2:
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

    agent = Mock()
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
        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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
            if self.cas_calls == 2:
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

    agent = Mock()
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

    agent = Mock(features={})
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
        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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

    agent = Mock(features={})
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
        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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
    winner_agent = Mock(features={})
    winner_agent.storage = storage
    winner_agent.storage_path = str(tmp_path / "winner" / "kestrel_prime.db")
    loser_agent = Mock(features={})
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
        properties = storage.nodes["feature_config:TestFeature"].properties
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

    agent = Mock(features={})
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
        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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

    agent = Mock(features={})
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
        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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
            if self.cas_calls == 2:
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

    agent = Mock()
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
        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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

    agent = Mock()
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
            # The second transition write (promotion) fails.
            if self.cas_calls == 2:
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

    agent = Mock()
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
    properties = agent.storage.nodes["feature_config:TestFeature"].properties
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

    agent = Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db"), features={})
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
    agent = Mock(features={"ChannelFeature": channel_feature})
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
        assert agent.storage.nodes["feature_config:TestFeature"].properties["config"] == old_config

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
    agent = Mock(features={"ChannelFeature": channel_feature})
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

    agent = Mock(features={})
    agent.storage = NoCASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=TransitionClient)
    try:
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
                elif lease != self.initial_lease.isoformat():
                    lease_renewed.set()
            return await super().compare_and_swap_node(node_id, expected, new_node)

    storage = RenewalObservingStorage()
    first_agent = Mock(features={})
    first_agent.storage = storage
    first_agent.storage_path = str(tmp_path / "first" / "kestrel_prime.db")
    second_agent = Mock(features={})
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

        with pytest.raises(RuntimeError, match="already in progress"):
            await second.set_config(second_config)
        assert storage.nodes["feature_config:TestFeature"].properties["pending_config"] == first_config

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

    agent = Mock(features={})
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
        assert "pending_config" in agent.storage.nodes["feature_config:TestFeature"].properties

        # It is still waiting on the first transition's reload lock, so this
        # cancellation must prevent a second generation from being staged once
        # cleanup releases the lock.
        second_update.cancel()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await update
        assert "pending_config" not in agent.storage.nodes["feature_config:TestFeature"].properties

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
    stale_agent = Mock(features={})
    stale_agent.storage = storage
    stale_agent.storage_path = str(tmp_path / "stale" / "kestrel_prime.db")
    writer_agent = Mock(features={})
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

        properties = storage.nodes["feature_config:TestFeature"].properties
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

    winner_agent = Mock(features={})
    winner_agent.storage = storage
    winner_agent.storage_path = str(tmp_path / "winner" / "kestrel_prime.db")
    stale_agent = Mock(features={})
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
            storage.nodes["feature_config:TestFeature"].properties["config"]
            == next_config
        )
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
            # The stage succeeds.  The promotion response is ambiguous, and
            # its required durable re-read then fails, latching quarantine.
            if self.cas_calls == 2:
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
    agent = Mock(features={})
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
        properties = storage.nodes["feature_config:TestFeature"].properties
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
    agent = Mock(features={})
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

        properties = storage.nodes["feature_config:TestFeature"].properties
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
    agent = Mock(features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=StopFailingFencedClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        with pytest.raises(OSError, match="old child would not stop"):
            await feature.set_config(next_config)

        properties = storage.nodes["feature_config:TestFeature"].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert feature._client is None
        assert feature._stopping is True
    finally:
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

    agent = Mock(features={})
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
        assert agent.storage.nodes["feature_config:TestFeature"].properties["config"] == {
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

    agent = Mock(features={})
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

    agent = Mock(features={})
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
    agent = Mock(features={"ChannelFeature": channel_feature})
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

    agent = Mock(features={})
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
            scheduled_result = await feature.call_isolated_tool("ping", {})
        finally:
            _current_execution.reset(token)
        assert scheduled_result["error"] == "isolated feature traffic is unavailable"

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

    agent = Mock(features={})
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

        shutdown_task.cancel()
        shutdown_task.cancel()
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        release_active.set()
        await active
        with pytest.raises(asyncio.CancelledError):
            await shutdown_task
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

    agent = Mock(features={})
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

    agent = Mock(features={})
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

    agent = Mock(features={})
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

        properties = agent.storage.nodes["feature_config:TestFeature"].properties
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
    agent = Mock(features={"ChannelFeature": channel_feature})
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
    agent = Mock(features={"ChannelFeature": channel_feature})
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
    agent = Mock(features={"ChannelFeature": channel_feature})
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
    agent = Mock(features={"ChannelFeature": channel_feature})
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
    agent = Mock(features={"ChannelFeature": channel_feature})
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
    agent = Mock(features={"ChannelFeature": channel_feature})
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
    agent = Mock(features={"ChannelFeature": channel_feature})
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
