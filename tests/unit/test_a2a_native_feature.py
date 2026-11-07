"""Real decorated feature tools produce native task values without wire models."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_task, new_text_artifact, new_text_message
from kestrel_sdk.native_a2a.types import TaskState
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.a2a.projections import embedded_json_value
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.effect_journal import bind_tool_effect_turn
from tests.unit.test_tool_effect_execution import admit, owner


class NativeFeature(Feature):
    @property
    def tool_description(self):
        return "Exercise native feature task results"

    async def initialize(self):
        pass

    @tool(name="native_echo", description="Return structured values unchanged")
    async def native_echo(self, value: dict) -> dict:
        return value

    @tool(name="native_failure", description="Return a typed tool failure")
    async def native_failure(self) -> ToolResult:
        return ToolResult.failed("Deliberate tool rejection")


def feature():
    return NativeFeature(SimpleNamespace())


@pytest.fixture
def hosted_feature(owner):
    """Open the original journal lifetime in the invoking test's own Task."""
    @asynccontextmanager
    async def open_feature():
        turn = await admit(owner)
        with bind_tool_effect_turn((turn, None)):
            try:
                yield NativeFeature(owner)
            except BaseException as error:
                await turn.close(error)
            else:
                await turn.close()

    return open_feature


def task(**metadata):
    result = new_task(
        "native-task",
        "native-context",
        TaskState.TASK_STATE_SUBMITTED,
        history=[new_text_message("request")],
    )
    result.metadata.update(metadata)
    return result


async def test_native_feature_preserves_history_and_existing_artifacts_and_produces_native_ids(hosted_feature):
    original = task(skill="native_echo", args={"value": {"nested": [None, {}, []]}})
    before = original.history[0].SerializeToString()
    existing = new_text_artifact("earlier", "Keep this")
    original.artifacts.append(existing)
    async with hosted_feature() as native:
        result = await native.handle_task(original)
    assert result.status.state == TaskState.TASK_STATE_COMPLETED
    assert result.history[0].SerializeToString() == before
    assert result.artifacts[0] == existing
    assert result.artifacts[1].artifact_id and result.status.message.message_id
    assert embedded_json_value(result.artifacts[1].parts[0])[1]["result"] == {
        "nested": [None, {}, []]
    }


@pytest.mark.parametrize(
    "metadata",
    [{}, {"skill": 1}, {"skill": "missing"}, {"skill": "native_echo", "args": []}],
)
async def test_missing_unknown_or_malformed_skills_are_rejected_without_execution(
    metadata,
):
    result = await feature().handle_task(task(**metadata))
    assert result.status.state == TaskState.TASK_STATE_REJECTED
    assert result.status.message.message_id
    assert list(result.artifacts) == []


async def test_native_feature_does_not_report_a_typed_tool_error_as_completed(hosted_feature):
    async with hosted_feature() as native:
        result = await native.handle_task(task(skill="native_failure"))
    assert result.status.state == TaskState.TASK_STATE_FAILED
    assert embedded_json_value(result.artifacts[0].parts[0])[1]["status"] == "error"


async def test_native_feature_refuses_private_task_objects():
    with pytest.raises(TypeError, match="official A2A Task"):
        await feature().handle_task({"sessionId": "legacy"})


def test_dynamic_tools_no_longer_carry_a_second_cached_skill_model():
    exposed = feature().get_tools()
    assert {item.name for item in exposed} == {"native_echo", "native_failure"}
    assert all(not hasattr(item, "agent_skill") for item in exposed)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"skill": "missing"},
        {"skill": "native_echo", "args": {"value": {"nested": [None, {}]}}},
        {"skill": "native_echo", "args": {}},
        {"skill": "native_failure"},
    ],
)
async def test_all_feature_status_messages_identify_their_parent_task(metadata, hosted_feature):
    async with hosted_feature() as native:
        result = await native.handle_task(task(**metadata))
    assert result.status.message.task_id == result.id
    assert result.status.message.context_id == result.context_id


async def test_missing_required_native_arguments_are_rejected_before_effect_claim(hosted_feature, owner):
    async with hosted_feature() as native:
        result = await native.handle_task(task(skill="native_echo", args={}))
        assert result.status.state == TaskState.TASK_STATE_REJECTED, "missing required input was admitted as native execution"
        assert result.status.message.task_id == result.id
        assert result.status.message.context_id == result.context_id
        assert await owner.tool_effect_store.database.fetchone(
            "SELECT COUNT(*) FROM tool_effect_claims") == (0,), "invalid native input claimed an effect before binding"


@pytest.mark.parametrize('lane', ['direct', 'manager'])
@pytest.mark.parametrize('withdrawal', ['host', 'disabled'])
async def test_native_a2a_claim_retains_original_feature_registration(
    owner, native_tool_effect_agent, tmp_path, monkeypatch, lane, withdrawal,
):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from tests.utils.native_a2a import manager_for
    from tests.utils.native_tool_effects import drive_native

    foreign = await native_tool_effect_agent(did=owner.did)
    owner.native_effect_target = tmp_path / 'original-host-effect'
    foreign.native_effect_target = tmp_path / 'foreign-host-effect'

    class HostFeature(Feature):
        tool_description = 'Write only through the original admitted host'

        async def initialize(self):
            pass

        @tool(name='host_bound_effect', description='Write the selected host marker')
        async def host_bound_effect(self):
            self.agent.native_effect_target.write_text('native A2A effect')
            return {'success': True}

    native = HostFeature(owner)
    owner.features[native.name] = native
    manager = manager_for(owner.tool_effect_store.database.backend,
        NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()
    manager.register_provider('feature', native.get_agent_skills(), native)
    claim = owner.tool_effect_store.claim

    async def withdraw_after_physical_ack(*args, **kwargs):
        receipt = await claim(*args, **kwargs)
        assert owner.tool_effect_store.database.owns_open_transaction is False
        if withdrawal == 'host':
            native.agent = foreign
        else:
            native.enabled = False
        return receipt

    monkeypatch.setattr(owner.tool_effect_store, 'claim', withdraw_after_physical_ack)
    async def operation():
        if lane == 'manager':
            return await manager.execute_skill('feature', 'host_bound_effect', {})
        return await native.handle_task(task(skill='host_bound_effect', args={}))
    try:
        result = await drive_native(owner, operation)
        assert not foreign.native_effect_target.exists(), 'A2A claim executed through a replacement host'
        assert not owner.native_effect_target.exists(), 'A2A claim executed a withdrawn feature'
        assert result.status.state == TaskState.TASK_STATE_FAILED
        assert await owner.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_claims') == [('NOT_EXECUTED',)]
    finally:
        await manager.close()


@pytest.mark.parametrize('lane', ['direct', 'manager'])
async def test_a2a_deferred_original_audit_joins_before_cancellation_retires_its_owner(
    native_tool_effect_agent, tmp_path, monkeypatch, lane,
):
    import asyncio
    from kestrel_sdk.tools.deferred import DeferredToolResult
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from tests.unit.test_tool_effect_invocation import governed
    from tests.utils.native_a2a import manager_for

    host = await native_tool_effect_agent(did='did:test:a2a-original-audit')
    witness = tmp_path / 'original-a2a-deferred-effect'
    published = []
    class DeferredFeature(Feature):
        tool_description = 'Retain the original A2A deferred audit'
        async def initialize(self):
            pass
        @tool(name='a2a_deferred_effect', description='Write before required audit')
        async def a2a_deferred_effect(self):
            with witness.open('a') as stream:
                stream.write('once\n')
            return DeferredToolResult(ToolResult.ok('Complete').to_dict(),
                publish=lambda: published.append('published') or [])

    native = DeferredFeature(host)
    host.features[native.name] = native
    manager = manager_for(host.tool_effect_store.database.backend,
        NativeRecipientScope(host.did), owns_backend=False)
    await manager.initialize()
    manager.register_provider('feature', native.get_agent_skills(), native)
    entered, release = asyncio.Event(), asyncio.Event()
    attempts = []
    write = host.observability_store.log_tool_dispatch
    async def pending_write(entry, **kwargs):
        attempts.append(entry.dispatch_id)
        entered.set()
        await release.wait()
        return await write(entry, **kwargs)
    monkeypatch.setattr(host.observability_store, 'log_tool_dispatch', pending_write)

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        if lane == 'manager':
            return await manager.execute_skill('feature', 'a2a_deferred_effect', {})
        return await native.handle_task(task(skill='a2a_deferred_effect', args={}))

    calling = asyncio.create_task(invoke(host, 'one original deferred A2A input',
        invocation_id='original-a2a-deferred'))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert witness.read_text().splitlines() == ['once']
        for _ in range(2):
            calling.cancel()
            await asyncio.sleep(0.02)
            assert not calling.done(), 'A2A deferred completion abandoned its original required audit'
        release.set()
        with pytest.raises((asyncio.CancelledError, ResourceOutcomeUnknown, BaseExceptionGroup)) as caught:
            await calling
        assert any(isinstance(error, asyncio.CancelledError)
            for error in execution_evidence_carriers(caught.value))
        assert len(attempts) == 1 and attempts[0]
        assert await host.tool_effect_store.database.fetchone(
            'SELECT result_status FROM a2a_tool_dispatches') == ('success',)
        assert published == []
    finally:
        release.set()
        await asyncio.gather(calling, return_exceptions=True)
        await manager.close()
