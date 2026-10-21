"""Every actual proxy ingress owner retains its first native control receipt."""

import asyncio

import pytest
from kestrel_sdk.isolated_feature import HostIngressCapabilities

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ProcessRuntimeExecutionFence,
    bind_execution_custody,
    is_execution_control_error,
)
from kestrel_sovereign.features.channels.feature import ChannelFeature
from kestrel_sovereign.features.isolated_runtime import ProxyFeature
from kestrel_sovereign.signals import SourceRegistry
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review30 import carrier
from tests.unit.test_isolated_feature_runtime import (
    FakeIsolatedClient,
    _cfg_runtime,
    _TELEGRAM_ATTEMPT_TOKEN,
)

KINDS = ["error", "unknown", "cancel", "cancel_unknown", "group", "ordinary"]


async def owner(
    monkeypatch,
    tmp_path,
    failure,
    *,
    completion_failure=False,
    completion_success=False,
    native_database=None,
    native_effect=None,
):
    agent, db = await native_owner(tmp_path)
    agent.storage.agent_id = agent.did
    await agent.storage.initialize()
    agent.signal_registry = SourceRegistry()
    entered, release = asyncio.Event(), asyncio.Event()
    effects_db = db if native_database is None else native_database

    async def effect():
        if native_effect is None:
            await effects_db.execute("INSERT INTO effects (id) VALUES (1)")
        else:
            await native_effect()

    class NativeChannel(ChannelFeature):
        async def handle_inbound(self, message):
            await effect()
            entered.set()
            await release.wait()
            raise failure

    class Client(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {"channel_type": "telegram", "send_tool": "telegram_send"}
            }

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack", "telegram-polling-nack")
            )
            self.completions = []

        async def call_host_ingress(self, name, payload=None):
            self.completions.append(name)
            if completion_failure:
                await effect()
                entered.set()
                await release.wait()
                raise failure
            if completion_success:
                entered.set()
                await release.wait()
            return {
                "status": "ok",
                "http_status": 200,
                "state": "acknowledged" if name.endswith("-ack") else "retrying",
            }

    channel = NativeChannel(agent)
    agent.features = {"ChannelFeature": channel}
    await channel.initialize()
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=Client)
    agent.features["TestFeature"] = feature
    await feature.initialize()
    key = "telegram:v2:bot:42:update:101"
    message = {
        "channel_type": "telegram",
        "direction": "inbound",
        "sender": "555",
        "recipient": "42",
        "content": "native",
        "id": key,
        "metadata": {"dedupe_key": key},
    }
    event = {
        "type": "channel.inbound",
        "payload": {
            "message": message,
            "_host_ingress_ack": {
                "name": "telegram-polling-ack",
                "payload": {
                    "dedupe_key": key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            },
            "_host_ingress_retry": {
                "name": "telegram-polling-nack",
                "payload": {
                    "dedupe_key": key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            },
        },
    }
    return agent, effects_db, feature, feature._client, event, entered, release


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", ["deferred", "legacy"])
async def test_all_route_workers_retain_native_control_before_retry_or_discard(
    monkeypatch, tmp_path, kind, route
):
    failure = OSError("ordinary route failure") if kind == "ordinary" else carrier(kind)
    agent, db, feature, client, event, entered, release = await owner(
        monkeypatch, tmp_path, failure
    )
    task = None
    completed = asyncio.Event()
    try:
        if route == "deferred":
            await feature._traffic_gate.close()
            await client.event_handler(event)
            # The actual registered SDK reader callback hands this cursor
            # event directly to the finite gate's separate deferred owner.
            (task,) = tuple(feature._deferred_acknowledged_event_tasks)
            task.add_done_callback(lambda done: completed.set())
            await feature._traffic_gate.reopen()
        else:
            await client.event_handler(
                {"type": "channel.inbound", "payload": event["payload"]["message"]}
            )
            (task,) = tuple(feature._event_ingress_tasks)
            task.add_done_callback(lambda done: completed.set())
        await asyncio.wait_for(entered.wait(), 5)
        release.set()
        await asyncio.wait_for(completed.wait(), 5)
        outcome = await await_owned_task(task)
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        if kind == "ordinary":
            assert outcome.error is None
            for completion in tuple(feature._event_ack_tasks):
                await await_owned_task(completion)
            assert client.completions == (
                ["telegram-polling-nack"] if route == "deferred" else []
            )
            await feature.shutdown()
        else:
            assert outcome.error is failure
            assert client.completions == [], "control authorized ACK/retry"
            with pytest.raises(BaseException) as later:
                await feature._route_inbound(event["payload"]["message"])
            assert later.value is failure
            assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
            with pytest.raises(BaseException) as caught:
                await feature.shutdown()
            assert caught.value is failure
        assert feature._client is None
        assert not feature._event_ingress_tasks
        assert not feature._deferred_acknowledged_event_tasks
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("completion", ["ack", "retry"])
async def test_completed_provider_rpc_control_survives_removal_and_proxy_shutdown(
    monkeypatch, tmp_path, kind, completion
):
    failure = (
        OSError("ordinary provider failure") if kind == "ordinary" else carrier(kind)
    )
    agent, db, feature, client, event, entered, release = await owner(
        monkeypatch, tmp_path, failure, completion_failure=True
    )
    task = None
    completed = asyncio.Event()
    try:
        data = event["payload"]
        request = (
            feature._split_inbound_event_acknowledgement(data)[1]
            if completion == "ack"
            else feature._inbound_event_retry_completion(data)
        )
        # One ordinary attempt proves legacy retry behavior without allowing
        # this fixture's PRIMARY KEY guard to become an unrelated retry error.
        if kind == "ordinary":
            monkeypatch.setattr(
                "kestrel_sovereign.features.isolated_runtime._EVENT_INGRESS_ACK_ATTEMPTS",
                1,
            )
        feature._schedule_event_ingress_completion(
            client,
            request,
            expected_state="acknowledged" if completion == "ack" else "retrying",
            kind=completion,
        )
        (task,) = tuple(feature._event_ack_tasks)
        task.add_done_callback(lambda done: completed.set())
        await asyncio.wait_for(entered.wait(), 5)
        release.set()
        await asyncio.wait_for(completed.wait(), 5)
        # Harvesting it a second time must not replace the cancelled original
        # after the proxy's callback already retrieved its first receipt.
        result = await await_owned_task(task)
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        assert not feature._event_ack_tasks
        assert not feature._event_ack_clients
        if kind == "ordinary":
            # Exhausting ordinary RPC attempts fences the source and may
            # cancel this owner as part of its normal terminal latch.
            assert result.error is None or isinstance(
                result.error, asyncio.CancelledError
            )
            await feature.shutdown()
        else:
            assert result.error is failure
            with pytest.raises(BaseException) as later:
                await feature._await_event_ingress_ack_attempt(
                    client.call_host_ingress, request, client
                )
            assert later.value is failure
            assert len(client.completions) == 1
            with pytest.raises(BaseException) as caught:
                await feature.shutdown()
            assert caught.value is failure
        assert feature._client is None
        assert client.stopped
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["ack", "retry"])
@pytest.mark.parametrize("retirement", ["retired", "unknown"])
async def test_successful_rpc_rechecks_original_custody_and_retains_its_receipt(
    monkeypatch, tmp_path, completion, retirement
):
    agent, db, feature, client, event, entered, release = await owner(
        monkeypatch, tmp_path, None, completion_success=True
    )
    task = None
    completed = asyncio.Event()
    try:
        # A native effect has already happened. The provider completion itself
        # succeeds, but its exact original admission retires before the response.
        await db.execute("INSERT INTO effects VALUES (1)")
        data = event["payload"]
        request = (
            feature._split_inbound_event_acknowledgement(data)[1]
            if completion == "ack"
            else feature._inbound_event_retry_completion(data)
        )
        with bind_execution_custody(ProcessRuntimeExecutionFence(agent.did)) as custody:
            feature._schedule_event_ingress_completion(
                client,
                request,
                expected_state="acknowledged" if completion == "ack" else "retrying",
                kind=completion,
            )
            (task,) = tuple(feature._event_ack_tasks)
            task.add_done_callback(lambda done: completed.set())
            await asyncio.wait_for(entered.wait(), 5)
            if retirement == "unknown":
                custody.preserve_commit_uncertainty("unknown")
        release.set()
        await asyncio.wait_for(completed.wait(), 5)
        result = await await_owned_task(task)
        assert is_execution_control_error(result.error)
        assert not feature._event_ack_tasks
        assert not feature._event_ack_clients
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        with pytest.raises(BaseException) as later:
            await feature._route_inbound(data["message"])
        assert later.value is result.error
        with pytest.raises(BaseException) as terminal:
            await feature.shutdown()
        assert terminal.value is result.error
        assert feature._client is None
        assert client.stopped
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        await retire(agent)
