"""Canonical Input prepare/apply on native databases, including linked-row refusal."""

from contextlib import asynccontextmanager
from dataclasses import asdict, replace
import hashlib
import hmac
import json

import pytest

from kestrel_sdk.security.aead import AEADCipher
from kestrel_sdk.security.exceptions import DecryptionError
from kestrel_sovereign.a2a.media import encode_inline_json
from kestrel_sovereign.mcp_input_store import MCPInputConflict, MCPInputIntegrityError
from tests.integration.test_mcp_input_store import custody as custody, binding, checkpoint, create


async def linked_create(store):
    prepared = await store._prepare_create(binding(custody='original-task-handle'), checkpoint(),
        now=100, expires_at=1000)
    async with store.write_transaction():
        result = await store._apply_create(prepared)
    return result


async def linked_claim(store, record):
    prepared = await store._prepare_response(record, {'form': {'action': 'accept'}}, now=101)
    async with store.write_transaction():
        result = await store._apply_response(prepared)
    return result


@pytest.mark.asyncio
async def test_ordinary_binding_digest_and_plaintext_keep_exact_schema_two_bytes(custody):
    store, _, _ = custody
    record = await create(store)
    legacy = asdict(binding())
    del legacy['custody']
    expected = hmac.new(store._root, encode_inline_json(legacy, sort_keys=True), hashlib.sha256).hexdigest()
    assert store._binding_digest(binding()) == expected
    assert record._payload == encode_inline_json({'binding': legacy, 'checkpoint': checkpoint(),
        'responses': None, 'continuation': None})
    assert await store.database.fetchone('SELECT revision FROM mcp_input_schema') == (2,)
    assert record.binding.custody is None and 'custody' not in record._payload.decode()


@pytest.mark.asyncio
async def test_public_creation_refuses_task_link_before_writing(custody):
    store, _, _ = custody
    with pytest.raises(MCPInputConflict, match='joint custody'):
        await create(store, custody='original-task-handle')
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize('forged', ['genuine', 'stripped-binding', 'stripped-payload'])
async def test_ordinary_response_cannot_consume_sql_authenticated_task_link(custody, forged):
    store, reopen, _ = custody
    record = await linked_create(store)
    expected = record
    if forged == 'stripped-binding':
        expected = replace(record, binding=replace(record.binding, custody=None))
    elif forged == 'stripped-payload':
        payload = json.loads(record._payload)
        del payload['binding']['custody']
        expected = replace(record, _payload=encode_inline_json(payload))
    restarted = await reopen()
    with pytest.raises(MCPInputConflict):
        await restarted.claim_response(expected, {}, now=101)
    actual = await restarted.read(record.binding, record.request_id)
    assert actual.state == 'WAITING' and actual.revision == 0
    assert actual.binding.custody == 'original-task-handle'


@pytest.mark.asyncio
@pytest.mark.parametrize('stripped', [False, True])
async def test_ordinary_successor_cannot_launder_a_linked_predecessor(custody, stripped):
    store, _, _ = custody
    record = await linked_create(store)
    claim = await linked_claim(store, record)
    if stripped:
        claim = replace(claim, record=replace(claim.record,
            binding=replace(claim.record.binding, custody=None)))
    with pytest.raises(MCPInputConflict):
        await store.create(binding(operation_id='new-original-operation'), checkpoint(),
            now=102, expires_at=1000, predecessor=claim)
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (1,)


@pytest.mark.asyncio
async def test_prepared_writes_require_their_actual_owner_and_an_open_transaction(custody):
    store, reopen, _ = custody
    prepared = await store._prepare_create(binding(), checkpoint(), now=100, expires_at=1000)
    with pytest.raises(MCPInputConflict, match='owning transaction'):
        await store._apply_create(prepared)
    other = await reopen()
    async with other.write_transaction():
        with pytest.raises(MCPInputConflict, match='owning transaction'):
            await other._apply_create(prepared)
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (0,)
    async with store.write_transaction():
        with pytest.raises(MCPInputConflict, match='owning transaction'):
            await store._apply_response(prepared)
        with pytest.raises(MCPInputConflict, match='own durable transaction'):
            await store._prepare_create(binding(), checkpoint(), now=100, expires_at=1000)
        record = await store._apply_create(prepared)
    response = await store._prepare_response(record, {}, now=101)
    with pytest.raises(MCPInputConflict, match='owning transaction'):
        await store._apply_response(response)
    async with other.write_transaction():
        with pytest.raises(MCPInputConflict, match='owning transaction'):
            await other._apply_response(response)
    assert (await store.read(record.binding, record.request_id)).state == 'WAITING'
    claim = await store.claim_response(record, {}, now=101)
    settlement = await store._prepare_settlement(claim, state='NON_RESUMABLE')
    with pytest.raises(MCPInputConflict, match='owning transaction'):
        await store._apply_settlement(settlement)
    async with other.write_transaction():
        with pytest.raises(MCPInputConflict, match='owning transaction'):
            await other._apply_settlement(settlement)
    assert (await store.read(record.binding, record.request_id)).state == 'RESPONDING'


@pytest.mark.asyncio
async def test_apply_halves_do_not_enter_crypto_or_ordinary_read_helpers(custody, monkeypatch):
    store, _, _ = custody
    for name in ('_binding_digest', '_reader_digest', '_encode', '_read_for'):
        original = getattr(store, name)

        def checked(*args, _original=original, **kwargs):
            assert store.database.owns_open_transaction is False, 'Input preparation must precede the transaction'
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, name, checked)
    record = await create(store)
    claim = await store.claim_response(record, {}, now=101)
    successor = await store._prepare_create(binding(operation_id='next-original-operation'),
        checkpoint(), now=102, expires_at=1000, predecessor=claim)
    async with store.write_transaction():
        await store._apply_create(successor)
    assert (await store.settle(claim, state='NON_RESUMABLE')).state == 'NON_RESUMABLE'


@pytest.mark.asyncio
async def test_response_cas_cannot_overwrite_a_different_authenticated_before_image(custody):
    store, _, _ = custody
    record = await create(store)
    prepared = await store._prepare_response(record, {'form': {'action': 'accept'}}, now=101)
    replacement = {**checkpoint(), 'changed-authenticated-checkpoint': True}
    ciphertext = store._encode(record.binding, record.request_id, 0, 'WAITING',
        record.expires_at, None, replacement, reservation_id=record.reservation_id,
        predecessor_id=record.predecessor_id)
    await store.database.execute('UPDATE mcp_inputs SET payload=?', (ciphertext,))
    with pytest.raises(MCPInputConflict, match='already consumed'):
        async with store.write_transaction():
            await store._apply_response(prepared)
    actual = await store.read(record.binding, record.request_id)
    assert actual.state == 'WAITING' and actual.payload == replacement


@pytest.mark.asyncio
async def test_successor_lock_cannot_overwrite_a_different_authenticated_parent(custody):
    store, _, _ = custody
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    parent = claim.record
    prepared = await store._prepare_create(binding(operation_id='new-original-operation'),
        checkpoint(), now=102, expires_at=1000, predecessor=claim)
    replacement = {**checkpoint(), 'changed-authenticated-checkpoint': True}
    ciphertext = store._encode(parent.binding, parent.request_id, parent.revision, 'RESPONDING',
        parent.expires_at, hashlib.sha256(claim.secret.encode()).hexdigest(), replacement,
        parent.responses, reservation_id=parent.reservation_id, predecessor_id=parent.predecessor_id)
    await store.database.execute('UPDATE mcp_inputs SET payload=?', (ciphertext,))
    with pytest.raises(MCPInputConflict, match='original live response claim'):
        async with store.write_transaction():
            await store._apply_create(prepared)
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (1,)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['create', 'response'])
async def test_prepared_write_known_rollback_never_consumes_the_original_revision(custody, operation):
    store, _, _ = custody
    if operation == 'create':
        prepared = await store._prepare_create(binding(), checkpoint(), now=100, expires_at=1000)
        apply = store._apply_create
    else:
        record = await create(store)
        prepared = await store._prepare_response(record, {}, now=101)
        apply = store._apply_response
    with pytest.raises(MCPInputConflict, match='owned rollback'):
        async with store.write_transaction():
            await apply(prepared)
            raise MCPInputConflict('owned rollback')
    if operation == 'create':
        assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (0,)
    else:
        assert (await store.read(record.binding, record.request_id)).state == 'WAITING'


@pytest.mark.asyncio
async def test_custody_link_is_authenticated_in_both_binding_and_original_payload(custody):
    store, _, _ = custody
    record = await linked_create(store)
    assert store._binding_digest(record.binding) != store._binding_digest(replace(record.binding, custody=None))
    row = await store.database.fetchone('SELECT binding_digest,revision,state,expires_at,claim_hash,reservation_id,predecessor_id,payload FROM mcp_inputs')
    cipher = AEADCipher(store._root)
    aad = store._aad(record.request_id, *row[:7])
    value = json.loads(cipher.decrypt(bytes(row[7]), aad=aad))
    value['binding']['custody'] = 'substituted-task-handle'
    await store.database.execute('UPDATE mcp_inputs SET payload=?',
        (cipher.encrypt(encode_inline_json(value), aad=aad),))
    with pytest.raises(MCPInputIntegrityError, match='reader binding'):
        await store.read_for(record.binding.principal_id, record.binding.session_id, record.request_id)


@pytest.mark.asyncio
async def test_original_creation_and_response_bytes_are_detached_before_write_await(custody, monkeypatch):
    store, _, _ = custody
    payload, responses = checkpoint(), {'form': {'action': 'accept'}}
    original = store.write_transaction

    @asynccontextmanager
    async def mutate_caller_values():
        payload.clear()
        responses.clear()
        async with original():
            yield

    monkeypatch.setattr(store, 'write_transaction', mutate_caller_values)
    record = await store.create(binding(), payload, now=100, expires_at=1000)
    assert record.payload == checkpoint()
    responses['form'] = {'action': 'accept'}
    claim = await store.claim_response(record, responses, now=101)
    assert claim.record.responses == {'form': {'action': 'accept'}}


@pytest.mark.asyncio
async def test_public_producers_use_both_canonical_preparation_and_transaction_apply(custody, monkeypatch):
    store, _, _ = custody
    calls = []
    for name in ('_prepare_create', '_apply_create', '_prepare_response', '_apply_response',
                 '_prepare_settlement', '_apply_settlement'):
        original = getattr(store, name)

        async def observed(*args, _name=name, _original=original, **kwargs):
            calls.append((_name, store.database.owns_open_transaction))
            return await _original(*args, **kwargs)

        monkeypatch.setattr(store, name, observed)
    record = await create(store)
    assert calls == [('_prepare_create', False), ('_apply_create', True)], 'Public creation must apply its canonical prepared write'
    claim = await store.claim_response(record, {}, now=101)
    assert calls == [('_prepare_create', False), ('_apply_create', True),
        ('_prepare_response', False), ('_apply_response', True)], 'Public writes must use the canonical prepare/apply pair'
    await store.settle(claim, state='NON_RESUMABLE')
    assert calls[-2:] == [('_prepare_settlement', False), ('_apply_settlement', True)]


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['RESOLVED', 'NON_RESUMABLE', 'UNKNOWN'])
async def test_settlement_cas_cannot_overwrite_a_different_authenticated_before_image(custody, state):
    store, _, _ = custody
    record = await create(store)
    claim = await store.claim_response(record, {'form': {'action': 'accept'}}, now=101)
    prepared = await store._prepare_settlement(claim, state=state)
    actual = claim.record
    replacement = {**actual.payload, 'changed-authenticated-checkpoint': True}
    ciphertext = store._encode(actual.binding, actual.request_id, actual.revision, 'RESPONDING',
        actual.expires_at, hashlib.sha256(claim.secret.encode()).hexdigest(), replacement,
        actual.responses, reservation_id=actual.reservation_id, predecessor_id=actual.predecessor_id)
    await store.database.execute('UPDATE mcp_inputs SET payload=?', (ciphertext,))
    with pytest.raises(MCPInputConflict, match='no longer current'):
        async with store.write_transaction():
            await store._apply_settlement(prepared)
    unchanged = await store.read(actual.binding, actual.request_id)
    assert unchanged.state == 'RESPONDING' and unchanged.payload == replacement


@pytest.mark.asyncio
async def test_settlement_detaches_expected_claim_and_continuation_before_await(custody, monkeypatch):
    store, _, _ = custody
    record = await create(store)
    claim = await store.claim_response(record, {}, now=101)
    continuation = {'owned-local-result': {'value': 'original'}}
    read = store._read_for
    async def mutate_after_capture(*args, **kwargs):
        continuation.clear()
        object.__setattr__(claim.record, 'revision', 999)
        return await read(*args, **kwargs)
    monkeypatch.setattr(store, '_read_for', mutate_after_capture)
    result = await store.settle(claim, state='RESOLVED', continuation=continuation)
    assert result.revision == 2
    assert result.continuation == {'owned-local-result': {'value': 'original'}}


@pytest.mark.asyncio
async def test_successor_preparation_detaches_binding_and_checkpoint_before_parent_read(custody, monkeypatch):
    store, _, _ = custody
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    supplied, target = checkpoint(), binding(operation_id='successor-original-operation')
    original_read = store._read_for

    async def mutate_during_parent_read(*args):
        result = await original_read(*args)
        supplied.clear()
        object.__setattr__(target, 'operation_id', 'substituted-during-parent-read')
        return result

    monkeypatch.setattr(store, '_read_for', mutate_during_parent_read)
    child = None
    try:
        child = await store.create(target, supplied, now=102, expires_at=1000, predecessor=claim)
    except DecryptionError:
        pass
    assert child is not None, 'Caller mutation must not change prepared binding encryption'
    opened = None
    try:
        opened = await store.read_for(child.binding.principal_id, child.binding.session_id, child.request_id)
    except MCPInputIntegrityError:
        pass
    assert opened is not None and opened.payload == checkpoint(), 'Successor evidence must detach before the predecessor read'
    assert opened.binding.operation_id == 'successor-original-operation'


@pytest.mark.asyncio
async def test_response_preparation_detaches_answers_before_original_sql_read(custody, monkeypatch):
    store, _, _ = custody
    first = await create(store)
    supplied = {'form': {'action': 'accept'}}
    original_read = store._read_for

    async def mutate_during_original_read(*args):
        result = await original_read(*args)
        supplied.clear()
        return result

    monkeypatch.setattr(store, '_read_for', mutate_during_original_read)
    claim = await store.claim_response(first, supplied, now=101)
    assert claim.record.responses == {'form': {'action': 'accept'}}, 'Validated answers must detach before the original SQL read'


@pytest.mark.asyncio
async def test_predecessor_expectation_is_captured_before_original_read(custody, monkeypatch):
    store, _, _ = custody
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    original = store._read_for

    async def mutate_parent_expectation(*args):
        result = await original(*args)
        object.__setattr__(claim.record, 'revision', claim.record.revision + 1)
        return result

    monkeypatch.setattr(store, '_read_for', mutate_parent_expectation)
    child = None
    try:
        child = await store.create(binding(operation_id='successor-original-operation'), checkpoint(),
            now=102, expires_at=1000, predecessor=claim)
    except MCPInputConflict:
        pass
    assert child is not None, 'Caller mutation cannot replace the admitted predecessor expectation'
    assert child.predecessor_id == first.request_id


@pytest.mark.asyncio
async def test_response_binding_mismatch_is_a_domain_refusal_despite_caller_mutation(custody, monkeypatch):
    store, _, _ = custody
    first = await create(store)
    second = await create(store, operation_id='another-original-operation')
    supplied = replace(first, request_id=second.request_id)
    original = store._read_for

    async def mutate_expected_binding(*args):
        result = await original(*args)
        object.__setattr__(supplied.binding, 'operation_id', second.binding.operation_id)
        return result

    monkeypatch.setattr(store, '_read_for', mutate_expected_binding)
    observed = None
    try:
        await store.claim_response(supplied, {}, now=101)
    except (MCPInputConflict, DecryptionError) as error:
        observed = error
    assert type(observed) is MCPInputConflict, 'Original binding mismatch must remain a domain refusal'


@pytest.mark.asyncio
async def test_private_linked_successor_keeps_the_original_authenticated_reservation(custody):
    store, reopen, _ = custody
    first = await linked_create(store)
    claim = await linked_claim(store, first)
    prepared = await store._prepare_create(replace(first.binding, operation_id='new-linked-operation'),
        checkpoint(), now=102, expires_at=1000, predecessor=claim)
    async with store.write_transaction():
        child = await store._apply_create(prepared)
    actual = await (await reopen()).read(child.binding, child.request_id)
    assert actual == child and actual.binding.custody == first.binding.custody
    assert actual.reservation_id == first.reservation_id and actual.predecessor_id == first.request_id
    with pytest.raises(MCPInputConflict, match='joint custody'):
        await store.claim_response(actual, {}, now=103)


@pytest.mark.asyncio
async def test_wrong_public_binding_type_is_distinct_from_a_task_link(custody):
    store, _, _ = custody
    with pytest.raises(MCPInputConflict, match='original binding'):
        await store.create(None, checkpoint(), now=100, expires_at=1000)
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (0,)


@pytest.mark.asyncio
async def test_settlement_seals_sql_expiry_even_if_equivalent_caller_number_has_another_json_type(custody):
    store, _, _ = custody
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    supplied = replace(claim, record=replace(claim.record, expires_at=int(claim.record.expires_at)))
    opened = None
    try:
        opened = await store.settle(supplied, state='RESOLVED', continuation={'result': 'original'})
    except MCPInputIntegrityError:
        pass
    assert opened is not None and opened.state == 'RESOLVED', 'Settlement must seal original SQL metadata rather than equivalent caller JSON'
    assert type(opened.expires_at) is float


@pytest.mark.asyncio
async def test_settlement_detaches_continuation_before_sql_read(custody, monkeypatch):
    store, _, _ = custody
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    supplied = {'result': {'private-original': True}}
    original = store._read_for

    async def mutate_continuation(*args):
        result = await original(*args)
        supplied.clear()
        return result

    monkeypatch.setattr(store, '_read_for', mutate_continuation)
    settled = await store.settle(claim, state='RESOLVED', continuation=supplied)
    assert settled.continuation == {'result': {'private-original': True}}, 'Settlement must detach continuation before awaiting'
