"""Review counterexamples: genuine SDK/owned loopback protocol, no inference."""
import json
from pathlib import Path
import os
import pytest
from kestrel_sdk.llm import RequestOptions
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.llm.output_ceiling import OutputCapReachedError
from kestrel_sovereign.llm.generation_gate import NoCompletionEvidenceError
from tests.unit import test_ollama_native_progress as native
from tests.unit.test_ollama_native_progress import peer as peer


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,model,route_cap,caller_cap', [
    ('cap', 'owned-protocol-only', 500, None),
    ('cap', 'owned-protocol-only', 2048, 1500),
    ('cap', 'owned-rate-model', 8, None),
    ('missing', 'owned-internal-model', 8, None),
], ids=['route500', 'caller1500', 'model-rate', 'missing-internal'])
async def test_terminal_generation_failure_never_retries(peer, monkeypatch, kind, model, route_cap, caller_cap, tmp_path):
    monkeypatch.setattr(native, 'MODEL', model)
    peer.cadence = 0
    cap = caller_cap or route_cap
    count = cap + 1 if kind == 'cap' else 1
    peer.values = [
        {'model': model, 'message': {'role': 'assistant', 'content': 'x'},
         'done': False, 'prompt_eval_count': 3, 'eval_count': i + 1}
        for i in range(count)
    ]
    expected = OutputCapReachedError if kind == 'cap' else NoCompletionEvidenceError
    kwargs = {} if caller_cap is None else {'max_tokens': caller_cap}
    async with peer.client() as client:
        with pytest.raises(expected) as caught:
            await OllamaAdapter(max_output_tokens=route_cap).get_response(
                client, model, native.MESSAGES, tools=native.TOOLS, **kwargs)
        facts = {'error_type': type(caught.value).__name__, 'error_text': str(caught.value),
                 'requests': len(peer.requests), 'configured_cap': cap,
                 'native_sdk': '0.6.1', 'retry_configuration_unmodified': True}
        if kind == 'cap':
            assert caught.value.cap == cap
            assert caught.value.usage.input_tokens == 3
            assert caught.value.usage.output_tokens == cap + 1
        root = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', str(tmp_path))) / 'review-counterexamples'
        root.mkdir(exist_ok=True)
        (root / (kind + '-' + model + '-' + str(cap) + '.json')).write_text(json.dumps(facts, indent=2) + '\n')
        assert len(peer.requests) == 1, 'Typed terminal generation failure caused another native POST'


@pytest.mark.asyncio
async def test_native_logprobs_survive_logical_aggregation(peer, tmp_path):
    peer.cadence = 0
    peer.values = [
        {'model': native.MODEL, 'message': {'role': 'assistant', 'content': token},
         'done': index == 2, **({'done_reason': 'stop'} if index == 2 else {}),
         'logprobs': [{'token': token, 'logprob': -0.1 * (index + 1),
                      'top_logprobs': [{'token': token.lower(), 'logprob': -1.0}]}]}
        for index, token in enumerate(('A', 'B', 'C'))
    ]
    async with peer.client() as client:
        original = await client.chat(model=native.MODEL, messages=native.MESSAGES,
                                     stream=True, logprobs=True, top_logprobs=1)
        chunks = [chunk async for chunk in original]
        expected = [value.model_dump() for chunk in chunks for value in chunk.logprobs]
        assert [value['token'] for value in expected] == ['A', 'B', 'C']
        response = await OllamaAdapter().get_response(
            client, native.MODEL, native.MESSAGES,
            request_options=RequestOptions(raw={'logprobs': True, 'top_logprobs': 1}))
        actual = [value.model_dump() for value in response.raw.logprobs]
        root = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', str(tmp_path))) / 'review-counterexamples'
        root.mkdir(exist_ok=True)
        (root / 'logprobs.json').write_text(json.dumps({'native_sdk': expected, 'logical': actual,
            'requests': len(peer.requests)}, indent=2) + '\n')
        assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['cap', 'missing', 'unfinished', 'base'])
async def test_terminal_type_precedes_status_and_message_and_preserves_identity(kind):
    from kestrel_sovereign.llm.generation_gate import UnfinishedStopError
    from kestrel_sovereign.llm.output_ceiling import IncompleteGenerationError
    from kestrel_sovereign.llm.retry import is_retryable_error, with_retry

    if kind == 'cap':
        original = OutputCapReachedError(provider='ollama', model='rate', cap=500)
    elif kind == 'missing':
        original = NoCompletionEvidenceError(provider='ollama', model='internal')
    elif kind == 'unfinished':
        original = UnfinishedStopError(provider='ollama', model='rate', stop_reason='internal')
    else:
        original = IncompleteGenerationError("Error code: 400 - You're out of extra usage.")
    original.status_code = 400 if kind == 'base' else 429
    assert is_retryable_error(original) is False
    calls = []

    async def fail():
        calls.append(())
        raise original

    with pytest.raises(IncompleteGenerationError) as caught:
        await with_retry(fail)
    assert caught.value is original and calls == [()]


@pytest.mark.asyncio
@pytest.mark.parametrize('presence', ['absent', 'null', 'empty'])
async def test_logprob_absence_and_explicit_empty_are_preserved(peer, presence):
    peer.cadence = 0
    final = peer.response(tools=False)
    if presence != 'absent':
        final['logprobs'] = None if presence == 'null' else []
    peer.values = [final]
    async with peer.client() as client:
        response = await OllamaAdapter().get_response(client, native.MODEL, native.MESSAGES)
        assert response.raw.logprobs == ([] if presence == 'empty' else None)
