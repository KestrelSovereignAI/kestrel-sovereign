"""A subagent's own model failure is a known failure, not an uncertain effect (#4036).

When a feature subagent's own model call fails after every child dispatch it
ran was confirmed, nothing it did has an uncertain outcome: the parent gets
the ordinary ``{"success": False, "error": ...}`` envelope and its turn's
effect journal holds no unconfirmed entry. That covers a refusal, a provider
400 ``prompt is too long`` and the pre-send context-budget refusal alike.

Uncertainty still propagates where it was observed: a child that raises
``ResourceOutcomeUnknown``, or a child whose outcome the subagent's own
dispatch journal never confirmed.

Each case drives a real native turn (``drive_native``) through the
orchestrator's dispatch, on real SQLite effect and dispatch journals, with a
scripted model in place of the provider.
"""

import httpx
import openai
import pytest
import pytest_asyncio
from uuid import uuid4

from kestrel_sdk.features.base import Feature as SDKFeature
from kestrel_sdk.llm.response import LLMResponse, ToolCall
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.agent.parts import part_collector
from kestrel_sovereign.execution_custody import execution_evidence_carriers
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.subagent_dispatch import ensure_subagent_dispatch
from kestrel_sovereign.llm.generation_gate import UnfinishedStopError
from kestrel_sovereign.llm.service import LLMServiceError
from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn
from tests.unit.test_tool_effect_bound_operation import bound_operation as bound_operation
from tests.unit.test_tool_effect_execution import owner as owner
from tests.utils.codex_inline import OwnedCallbackReader
from tests.utils.native_tool_effects import drive_native

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]

CHILDREN = 3
SESSION = 'subagent-model-failure'
# A model that exists only here, so the budget under test is pinned.
SMALL_MODEL = 'test-model-4036-small'
SMALL_LIMIT = 4_000
# One read result large enough to cross the SMALL_MODEL budget on its own:
# about 6,000 tokens against a 3,400-token ceiling. Nothing larger is built.
BULK_VALUE = 'record ' * 6_000


def _refusal():
    # What the gate raises for the 2026-11-12 failure (task c0b384e6).
    return UnfinishedStopError(
        provider='anthropic:plan', model='claude-opus-5-5', stop_reason='refusal')


def _refusal_through_the_service():
    # The same refusal as LLMService's route walk reports it.
    refusal = _refusal()
    error = LLMServiceError(
        f'All providers failed for generate_with_messages (last: anthropic:plan): {refusal}')
    error.__cause__ = refusal
    return error


def _prompt_too_long():
    # What LLMService raises for a provider 400 (2026-11-13, Meridian's turn).
    rejected = openai.BadRequestError(
        'Error code: 400 - prompt is too long: 1158906 tokens > 1000000 maximum',
        response=httpx.Response(400, request=httpx.Request(
            'POST', 'https://provider.invalid/v1/messages')),
        body=None)
    error = LLMServiceError(f'Request rejected by anthropic:plan: {rejected}')
    error.__cause__ = rejected
    return error


@pytest.fixture
def small_model_limit(monkeypatch):
    from kestrel_sovereign.agent.token_counter import TokenCounter

    real = TokenCounter.resolved_context_limit

    def pinned(self):
        return SMALL_LIMIT if self.model == SMALL_MODEL else real(self)

    monkeypatch.setattr(TokenCounter, 'resolved_context_limit', pinned)


@pytest_asyncio.fixture(params=[Feature, SDKFeature], ids=['core', 'sdk'])
async def reading_subagent(bound_operation, request):
    host, _, _ = bound_operation
    reads = []

    class RecordsFeature(request.param):
        tool_description = 'Read stored records'

        async def initialize(self):
            pass

        @tool(name='read_record', description='Read one stored record')
        async def read_record(self, key: str):
            reads.append(key)
            if key == self.failing_key:
                raise self.child_error
            return {'success': True, 'key': key,
                    'value': BULK_VALUE if key == self.bulk_key else f'value-{key}'}

    feature = ensure_subagent_dispatch(RecordsFeature)(host)
    feature.failing_key = None
    feature.child_error = None
    feature.bulk_key = None
    host.features[feature.name] = feature

    reader = OwnedCallbackReader()
    await reader.start()

    def call(index):
        response = LLMResponse(content=None, tool_calls=[ToolCall(
            id=f'read-{index}', name='read_record', arguments={'key': f'k{index}'})])
        response.model = SMALL_MODEL
        return response

    class Model:
        """Reads ``CHILDREN`` records, one per model call, then fails.

        ``inline`` runs the reads inside the first call on an owned reader,
        as the Codex app-server does, and fails that call. ``swallow`` names
        the errors that transport returns to the model as ordinary failed
        results instead of raising; it then calls the executor itself.
        """

        inline = False
        swallow = None
        failure = None
        continuations = 0

        async def generate(self, **kwargs):
            if not self.inline:
                return call(1)
            executor = kwargs['tool_executor']
            for index in range(1, CHILDREN + 1):
                args = {'key': f'k{index}'}
                if self.swallow is None:
                    await reader.dispatch(executor, 'read_record', args)
                    continue
                try:
                    await executor('read_record', args)
                except self.swallow:
                    pass
            raise self.failure

        async def generate_with_messages(self, **kwargs):
            self.continuations += 1
            if self.continuations < CHILDREN:
                return call(self.continuations + 1)
            raise self.failure

    model = host.llm_service = Model()

    async def dispatch(route):
        if route == 'named-inline':
            return await host.execute_named_tool(
                feature.tool_name, {'task': 'read the records'},
                session_id=SESSION, source='owned-inline', _capture={})
        return await host._dispatch_tool_call(
            ToolCall(id=uuid4().hex, name=feature.tool_name,
                     arguments={'task': 'read the records'}),
            {feature.tool_name: feature}, {feature.tool_name}, [], 0,
            'read the records', session_id=SESSION)

    try:
        yield host, feature, model, reads, dispatch
    finally:
        await reader.stop()


def _envelope(result):
    """The parent's result, whichever shape its dispatch path returns."""
    if isinstance(result, tuple):
        result = result[-1]
    assert isinstance(result, dict), f'Not a result envelope: {result!r}'
    return result


async def _run_known_failure(host, dispatch, route):
    seen = {}

    async def operation():
        with part_collector(host=host, session_id=SESSION):
            seen['result'] = _envelope(await dispatch(route))
            seen['unconfirmed'] = tuple(current_tool_effect_turn(host).unconfirmed)

    # A turn whose journal holds an unconfirmed effect fails here, at close.
    await drive_native(host, operation)
    return seen


async def _assert_confirmed_children_only(host, feature, children):
    database = host.tool_effect_store.database
    assert await database.fetchall('SELECT state FROM tool_effect_claims') == (
        [('RETURNED_OK',)] * children)
    assert await database.fetchall(
        "SELECT result_status FROM a2a_tool_dispatches WHERE tool_name='read_record'"
    ) == [('success',)] * children
    assert await database.fetchone(
        'SELECT result_status FROM a2a_tool_dispatches WHERE tool_name=?',
        (feature.tool_name,)) == ('error',)
    assert await database.fetchone(
        'SELECT state, latched FROM tool_effect_operations') == ('CLOSED', 0), (
        'The known failure latched the parent operation as an uncertain effect')


@pytest.mark.parametrize('route', ['chat', 'named-inline'])
@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
@pytest.mark.parametrize('shape', ['raw', 'service'])
async def test_refusal_after_confirmed_children_is_a_known_failure(
    reading_subagent, route, inline, shape,
):
    """Regression 1: N confirmed reads, then the next model call refuses."""
    host, feature, model, reads, dispatch = reading_subagent
    model.inline = inline
    model.failure = _refusal() if shape == 'raw' else _refusal_through_the_service()
    seen = await _run_known_failure(host, dispatch, route)
    assert seen['result']['success'] is False
    assert 'refusal' in str(seen['result']), seen['result']
    assert seen['unconfirmed'] == (), 'The refusal left an unconfirmed effect on the parent turn'
    assert reads == [f'k{i}' for i in range(1, CHILDREN + 1)]
    await _assert_confirmed_children_only(host, feature, CHILDREN)


@pytest.mark.parametrize('route', ['chat', 'named-inline'])
@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
async def test_child_outcome_unknown_still_fences_the_parent_turn(
    reading_subagent, route, inline,
):
    """Regression 2: a child's own unknown outcome reaches the parent unchanged."""
    host, feature, model, reads, dispatch = reading_subagent
    original = ResourceOutcomeUnknown('read_record lost its reply')
    feature.failing_key, feature.child_error = 'k2', original
    model.inline = inline
    model.failure = _refusal()
    seen = {}

    async def operation():
        with part_collector(host=host, session_id=SESSION):
            try:
                seen['result'] = await dispatch(route)
            except ResourceOutcomeUnknown as error:
                seen['error'] = error
            seen['unconfirmed'] = tuple(current_tool_effect_turn(host).unconfirmed)

    with pytest.raises(ResourceOutcomeUnknown) as fenced:
        await drive_native(host, operation)
    assert 'result' not in seen, f'The parent received an envelope: {seen.get("result")!r}'
    assert original in execution_evidence_carriers(seen['error'])
    assert any(original in execution_evidence_carriers(error) for error in seen['unconfirmed']), (
        "The child's unknown outcome never reached the parent turn's journal")
    assert original in execution_evidence_carriers(fenced.value)
    assert reads == ['k1', 'k2']
    assert model.continuations == (0 if inline else 1)


@pytest.mark.parametrize('route', ['chat', 'named-inline'])
@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
async def test_prompt_too_long_after_confirmed_children_is_a_known_failure(
    reading_subagent, route, inline,
):
    """Regression 3: N confirmed reads, then the provider rejects the request."""
    host, feature, model, reads, dispatch = reading_subagent
    model.inline = inline
    model.failure = _prompt_too_long()
    seen = await _run_known_failure(host, dispatch, route)
    assert seen['result']['success'] is False
    assert 'prompt is too long' in str(seen['result']), seen['result']
    assert seen['unconfirmed'] == (), 'The rejection left an unconfirmed effect on the parent turn'
    assert reads == [f'k{i}' for i in range(1, CHILDREN + 1)]
    await _assert_confirmed_children_only(host, feature, CHILDREN)


@pytest.mark.parametrize('route', ['chat', 'named-inline'])
async def test_budget_refusal_after_confirmed_children_is_a_known_failure(
    reading_subagent, small_model_limit, route,
):
    """Regression 4: N confirmed reads, then the pre-send budget check refuses.

    The last read's result pushes the array past a small pinned budget, so
    ``_subagent_context_overflow`` refuses before the next request is sent.
    Only the tool loop runs that check, so there is no owned-reader case.
    """
    host, feature, model, reads, dispatch = reading_subagent
    feature.bulk_key = f'k{CHILDREN}'
    model.failure = AssertionError('the budget refusal must come before this request')
    seen = await _run_known_failure(host, dispatch, route)
    assert seen['result']['success'] is False
    assert 'exceeded its context budget' in str(seen['result']), seen['result']
    assert seen['unconfirmed'] == (), 'The budget refusal left an unconfirmed effect on the parent turn'
    assert reads == [f'k{i}' for i in range(1, CHILDREN + 1)]
    assert model.continuations == CHILDREN - 1, 'The over-budget request was sent'
    await _assert_confirmed_children_only(host, feature, CHILDREN)


async def test_a_child_its_transport_returned_unconfirmed_makes_the_failure_uncertain(
    reading_subagent,
):
    """The subagent's own journal, not the failure's cause graph, decides.

    The transport returns the second read's unknown outcome to the model as an
    ordinary failed result, so the model call that then refuses carries no
    unknown outcome. The journal holds that read unconfirmed, so the subagent
    raises ``ResourceOutcomeUnknown`` rather than returning a known failure.
    """
    host, feature, model, reads, _ = reading_subagent
    original = ResourceOutcomeUnknown('read_record lost its reply')
    feature.failing_key, feature.child_error = 'k2', original
    model.inline = True
    model.swallow = (ResourceOutcomeUnknown,)
    model.failure = _refusal()
    seen = {}

    async def operation():
        with part_collector(host=host, session_id=SESSION):
            try:
                seen['result'] = await feature.execute_as_subagent(task='read the records')
            except ResourceOutcomeUnknown as error:
                seen['error'] = error

    with pytest.raises(ResourceOutcomeUnknown):
        await drive_native(host, operation)
    assert 'result' not in seen, f'An unconfirmed child became a known failure: {seen.get("result")!r}'
    assert any(isinstance(error, UnfinishedStopError)
               for error in execution_evidence_carriers(seen['error']))
    # The turn admits no effect after an uncertain one, so the third read
    # was refused before it ran.
    assert reads == ['k1', 'k2']


@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
async def test_model_failure_with_commit_debt_is_uncertain(reading_subagent, inline):
    """Commit debt is observed uncertainty, whoever raised it.

    The model call fails with an error whose commit outcome is unknown, after
    every read was confirmed. That error is no unknown tool outcome, but the
    write it carries may have committed, so the subagent raises
    ``ResourceOutcomeUnknown`` and the parent turn is fenced.
    """
    host, _, model, reads, dispatch = reading_subagent
    original = RuntimeError('a2a_llm_calls write outcome unknown')
    original._kestrel_unresolved_commit_outcome = 'unknown'
    model.inline = inline
    model.failure = original
    seen = {}

    async def operation():
        with part_collector(host=host, session_id=SESSION):
            try:
                seen['result'] = await dispatch('chat')
            except ResourceOutcomeUnknown as error:
                seen['error'] = error

    with pytest.raises(ResourceOutcomeUnknown):
        await drive_native(host, operation)
    assert 'result' not in seen, f'Commit debt became a known failure: {seen.get("result")!r}'
    assert original in execution_evidence_carriers(seen['error'])
    assert reads == [f'k{i}' for i in range(1, CHILDREN + 1)]


@pytest.mark.parametrize('route', ['chat', 'named-inline'])
async def test_a_child_known_conflict_returned_to_the_model_stays_a_known_failure(
    reading_subagent, route,
):
    """A child the effect journal settled as a known conflict is confirmed.

    The second read raises ``ResourceConflict`` from its body. The effect
    journal settles it ``RETURNED_ERROR``, and the transport returns it to the
    model as a failed result, as the Codex handler does for a non-terminal
    error. The model call that then refuses is a known failure, the same as
    the effect journal's own reading of that child.
    """
    host, feature, model, reads, dispatch = reading_subagent
    feature.failing_key, feature.child_error = 'k2', ResourceConflict('record k2 moved')
    model.inline = True
    model.swallow = (ResourceConflict,)
    model.failure = _refusal()
    seen = await _run_known_failure(host, dispatch, route)
    assert seen['result']['success'] is False
    assert 'refusal' in str(seen['result']), seen['result']
    assert seen['unconfirmed'] == ()
    assert reads == [f'k{i}' for i in range(1, CHILDREN + 1)]
    database = host.tool_effect_store.database
    assert await database.fetchall('SELECT state FROM tool_effect_claims') == [
        ('RETURNED_OK',), ('RETURNED_ERROR',), ('RETURNED_OK',)]
    assert await database.fetchone(
        'SELECT state, latched FROM tool_effect_operations') == ('CLOSED', 0)
