"""Original submission custody on independent native SQLite/PostgreSQL connections."""

from dataclasses import replace
import hashlib

import pytest

from kestrel_sovereign.auth import CallerContext, caller_context_scope
from kestrel_sdk.security.aead import AEADCipher
from kestrel_sovereign.mcp_task_store import MCPTaskConflict, MCPTaskIntegrityError
from kestrel_sovereign.mcp_task_submission import (
    MCPTaskDelivery, MCPTaskInteractiveOrigin, MCPTaskServiceOrigin, MCPTaskSubmission,
)
from kestrel_sovereign.storage.db.interface import TransactionError
from kestrel_sovereign.tools.effect_invocation import current_service_operation_attribution
from tests.shared.mcp_task_custody import (
    custody as custody, binding, submission, attach, observation,
)


async def sealed_columns(store, handle):
    return await store.database.fetchone(
        'SELECT submission_limit,submission_digest,submission FROM mcp_task_records WHERE handle=?', (handle,))


@pytest.mark.asyncio
async def test_original_submission_is_private_and_durable_before_any_native_request(custody):
    store = custody.store
    supplied = {'private-answer': {'city': 'private-original-city'}}
    original = submission(arguments=supplied)
    reserved = await store.reserve(binding(), original)
    supplied['private-answer']['city'] = 'changed-after-admission'
    restarted = await custody.reopen()
    opened = await restarted.open_submission(reserved.record)
    assert opened == original
    assert opened.arguments == {'private-answer': {'city': 'private-original-city'}}
    opened.arguments.clear()
    assert (await restarted.open_submission(reserved.record)).arguments == original.arguments
    assert hashlib.sha256(opened.contract).hexdigest() == reserved.record.binding.contract_digest
    assert not (await restarted.reserve(binding(), original)).fresh
    raw_rows = repr(await store.database.fetchall('SELECT * FROM mcp_task_records'))
    for private in ('private-original-city', 'private-answer', 'private-native-tool', reserved.issuance.secret):
        assert private not in raw_rows
    assert 'private-original-city' not in repr(opened)


@pytest.mark.asyncio
async def test_submission_has_independent_headroom_and_is_never_rewritten_by_transitions(custody, monkeypatch):
    store = await custody.reopen(max_payload_bytes=2048, max_submission_bytes=8192)
    statements = []
    actual_execute = store.database.execute

    async def observed_execute(query, *args, **kwargs):
        statements.append(query)
        return await actual_execute(query, *args, **kwargs)

    monkeypatch.setattr(store.database, 'execute', observed_execute)
    original = submission(arguments={'private-large-answer': 'x' * 5000})
    assert len(original.canonical_json) > store.max_payload_bytes
    reserved = await store.reserve(binding(), original)
    before = await sealed_columns(store, reserved.record.handle)
    current = await attach(store, reserved.issuance)
    pending = await store.reserve_mutation(current, operation_id='fresh-cancel', kind='cancel')
    acknowledged = await store.acknowledge_mutation(pending.issuance, {'resultType': 'complete'})
    terminal = await store.observe(acknowledged, observation('completed'))
    assert terminal.phase == 'SETTLED'
    assert await sealed_columns(store, terminal.handle) == before
    assert b'private-large-answer' not in terminal._payload
    lowered = await custody.reopen(max_payload_bytes=1, max_submission_bytes=1)
    assert await lowered.open_submission(terminal) == original
    assert not (await lowered.reserve(binding(), original)).fresh
    assert await lowered.list_remote_tasks() == (terminal,)
    for query in statements:
        if query.startswith('UPDATE mcp_task_records SET '):
            assignments = query.split(' SET ', 1)[1].split(' WHERE ', 1)[0]
            assert 'submission' not in assignments, 'Every submission column must remain immutable'


@pytest.mark.asyncio
@pytest.mark.parametrize('disposition', ['SETTLED', 'NOT_SENT'])
async def test_confirmed_initial_result_or_refusal_preserves_original_submission(custody, disposition):
    store = custody.store
    original = submission()
    reserved = await store.reserve(binding(), original)
    before = await sealed_columns(store, reserved.record.handle)
    envelope = ({'version': 1, 'native': {'resultType': 'complete', 'content': []}}
        if disposition == 'SETTLED' else {'version': 1, 'refusal': {'reason': 'proven-not-sent'}})
    result = await store.settle_initial(reserved.issuance, disposition=disposition, envelope=envelope)
    assert await store.open_submission(result) == original
    assert await sealed_columns(store, result.handle) == before
    assert await store.list_remote_tasks() == ()  # No remote Task was created.


@pytest.mark.asyncio
async def test_repeat_reservation_cannot_replace_original_arguments_contract_or_delivery(custody):
    store = custody.store
    original = submission()
    reserved = await store.reserve(binding(), original)
    variants = [
        (binding(), submission(arguments={'private-answer': 'substituted'})),
        (binding(), MCPTaskSubmission.seal(arguments=original.arguments, contract=original.contract,
            delivery=MCPTaskDelivery('app', 'private-session', 'another-view'),
            authorization=original.authorization)),
        (binding(), submission(authorization=replace(original.authorization, slot_id='d' * 64))),
    ]
    other = submission(contract=b'another-original-descriptor')
    variants.append((replace(binding(), contract_digest=hashlib.sha256(other.contract).hexdigest()), other))
    for target, capsule in variants:
        with pytest.raises(MCPTaskConflict, match='another binding or submission'):
            await store.reserve(target, capsule)
    assert await store.open_submission(reserved.record) == original
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_task_records') == (1,)


@pytest.mark.asyncio
async def test_service_origin_uses_real_ambient_attribution_without_inventing_a_principal_or_session(custody):
    with caller_context_scope(None):
        kind, digest = current_service_operation_attribution()
    origin = MCPTaskServiceOrigin(kind, digest)
    target = binding(origin=origin)
    original = submission(origin=origin)
    reserved = await custody.store.reserve(target, original)
    restarted = await custody.reopen()
    actual = await restarted.read(target, reserved.record.handle)
    assert actual.binding.origin == origin and origin.session_id is None
    assert (await restarted.open_submission(actual)).delivery == MCPTaskDelivery('internal')
    assert (await restarted.open_submission(actual)).authorization == original.authorization
    assert not (await restarted.reserve(target, original)).fresh


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [
    'contract', 'internal-interactive', 'app-service', 'wrong-session',
    'wrong-connection', 'wrong-owner', 'wrong-principal',
])
async def test_submission_mismatch_refuses_before_database_admission(custody, failure):
    store = custody.store
    target = binding()
    original = submission()
    if failure == 'contract':
        original = submission(contract=b'private-wrong-descriptor')
    elif failure == 'wrong-connection':
        original = submission(authorization=replace(original.authorization, connection_name='another-connection'))
    elif failure == 'wrong-owner':
        original = submission(authorization=replace(original.authorization, owner='another-owner'))
    elif failure == 'wrong-principal':
        original = submission(authorization=replace(original.authorization, principal_digest='d' * 64))
    else:
        delivery = MCPTaskDelivery('internal') if failure == 'internal-interactive' else (
            MCPTaskDelivery('chat', 'different-session') if failure == 'wrong-session'
            else MCPTaskDelivery('app', 'private-session', 'original-view'))
        if failure == 'app-service':
            target = binding(origin=MCPTaskServiceOrigin('internal', 'd' * 64, 'private-session'))
        original = MCPTaskSubmission.seal(arguments=original.arguments, contract=original.contract,
            delivery=delivery, authorization=original.authorization)
    before = await store.database.fetchone('SELECT revision FROM mcp_task_keys')
    with pytest.raises(ValueError, match='original'):
        await store.reserve(target, original)
    assert await store.database.fetchone('SELECT revision FROM mcp_task_keys') == before
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_task_records') == (0,)


@pytest.mark.asyncio
async def test_submission_budget_refuses_new_work_but_never_reclaims_existing_custody(custody):
    limited = await custody.reopen(max_submission_bytes=1)
    with pytest.raises((ValueError, TransactionError), match='submission exceeded its storage budget'):
        await limited.reserve(binding(), submission())
    assert await limited.database.fetchone('SELECT COUNT(*) FROM mcp_task_records') == (0,)
    reserved = await custody.store.reserve(binding(), submission())
    assert await limited.open_submission(reserved.record) == submission()
    assert not (await limited.reserve(binding(), submission())).fresh


@pytest.mark.asyncio
@pytest.mark.parametrize('column,value', [
    ('submission_limit', 10000), ('submission_digest', 'd' * 64),
    ('input_lineage_digest', 'e' * 64),
])
async def test_submission_routing_metadata_is_authenticated_before_any_new_admission(custody, column, value):
    store = custody.store
    reserved = await store.reserve(binding(), submission())
    await store.database.execute(f'UPDATE mcp_task_records SET {column}=?', (value,))
    with pytest.raises(MCPTaskIntegrityError):
        await store.open_submission(reserved.record)
    with pytest.raises((MCPTaskIntegrityError, TransactionError)):
        await store.reserve(binding(operation_id='must-not-hide-corruption'), submission())
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_task_records') == (1,)


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ['ciphertext', 'oversize'])
async def test_private_submission_open_authenticates_its_bytes_and_stored_budget(custody, damage):
    store = custody.store
    reserved = await store.reserve(binding(), submission())
    value = b'damaged' if damage == 'ciphertext' else b'x' * (4 * reserved.record.submission_limit + 129)
    await store.database.execute('UPDATE mcp_task_records SET submission=?', (value,))
    message = 'cannot be opened' if damage == 'ciphertext' else 'storage budget'
    with pytest.raises(MCPTaskIntegrityError, match=message):
        await store.open_submission(reserved.record)
    assert not (await store.reserve(binding(), submission())).fresh  # No secret recovery on a metadata lookup.


@pytest.mark.asyncio
async def test_untrusted_record_fields_cannot_replace_the_sql_opened_submission(custody):
    store = custody.store
    reserved = await store.reserve(binding(), submission())
    forged = replace(reserved.record, _payload=b'private-forged-reader', submission_digest='d' * 64, submission_limit=1)
    assert await store.open_submission(forged) == submission()
    foreign = replace(reserved.record, binding=binding(
        origin=MCPTaskInteractiveOrigin('d' * 64, 'private-session')))
    with pytest.raises(MCPTaskConflict, match='another binding'):
        await store.open_submission(foreign)


@pytest.mark.asyncio
@pytest.mark.parametrize('revision', [1, 2, 3, 4, 5])
async def test_prototype_revision_refuses_without_migrating_original_rows(custody, revision):
    store = custody.store
    await store.reserve(binding(), submission())
    before = await store.database.fetchall('SELECT * FROM mcp_task_records')
    await store.database.execute('UPDATE mcp_task_schema SET revision=?', (revision,))
    with pytest.raises(MCPTaskIntegrityError, match='Unsupported MCP Task schema revision'):
        await custody.reopen()
    assert await store.database.fetchall('SELECT * FROM mcp_task_records') == before
    assert await store.database.fetchone('SELECT revision FROM mcp_task_schema') == (revision,)


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ['contract', 'digest', 'plaintext-budget'])
async def test_authenticated_submission_still_requires_original_contract_digest_and_budget(custody, damage):
    store = custody.store
    original = submission()
    reserved = await store.reserve(binding(), original)
    columns = 'scope,handle,operation_digest,binding_digest,remote_digest,revision,phase,claim_hash,payload_limit,payload,submission_limit,submission_digest,input_lineage_digest'
    row = list(await store.database.fetchone(f'SELECT {columns} FROM mcp_task_records'))
    cipher = AEADCipher(store._root)
    payload = cipher.decrypt(bytes(row[9]), aad=store._aad((*row[:9], *row[10:13])))
    if damage == 'contract':
        altered = submission(contract=b'authenticated-but-wrong-original-descriptor')
        row[11] = store._digest('mcp-task-submission-v3',
            [store._binding_digest(binding()), altered.canonical_json.decode('ascii')])
    elif damage == 'digest':
        altered = submission(arguments={'private-answer': 'authenticated-substitution'})
    else:
        altered = original
        row[10] = len(original.canonical_json) - 1
    # A valid native AEAD frame alone cannot make false semantic evidence true.
    routing = cipher.encrypt(payload, aad=store._aad((*row[:9], *row[10:13])))
    private = cipher.encrypt(altered.canonical_json,
        aad=store._submission_aad(row[1], row[3], row[10], row[11]))
    await store.database.execute('UPDATE mcp_task_records SET payload=?,submission_limit=?,submission_digest=?,submission=?',
        (routing, row[10], row[11], private))
    with pytest.raises(MCPTaskIntegrityError, match='cannot be opened'):
        await store.open_submission(reserved.record)


@pytest.mark.asyncio
async def test_authenticated_missing_origin_is_a_custody_integrity_error(custody):
    import json

    store = custody.store
    reserved = await store.reserve(binding(), submission())
    row = await store.database.fetchone('SELECT scope,handle,operation_digest,binding_digest,remote_digest,revision,phase,claim_hash,payload_limit,payload,submission_limit,submission_digest,input_lineage_digest FROM mcp_task_records')
    cipher = AEADCipher(store._root)
    aad = store._aad((*row[:9], *row[10:13]))
    value = json.loads(cipher.decrypt(bytes(row[9]), aad=aad))
    del value['binding']['origin']
    await store.database.execute('UPDATE mcp_task_records SET payload=?',
        (cipher.encrypt(json.dumps(value).encode(), aad=aad),))
    observed = None
    try:
        await store.open_submission(reserved.record)
    except (MCPTaskIntegrityError, KeyError) as error:
        observed = error
    assert type(observed) is MCPTaskIntegrityError, 'Malformed stored provenance must be classified as custody integrity'


@pytest.mark.asyncio
@pytest.mark.parametrize('custody', ['sqlite'], indirect=True)
async def test_sqlite_nonbinary_submission_refuses_before_crypto(custody):
    store = custody.store
    reserved = await store.reserve(binding(), submission())
    await store.database.execute('UPDATE mcp_task_records SET submission=?', (7,))
    observed = None
    try:
        await store.open_submission(reserved.record)
    except (MCPTaskIntegrityError, TypeError) as error:
        observed = error
    assert type(observed) is MCPTaskIntegrityError, 'Nonbinary original bytes must fail as custody integrity'


@pytest.mark.asyncio
async def test_identical_submissions_in_different_operations_do_not_reveal_equality(custody):
    first = await custody.store.reserve(binding(), submission())
    second = await custody.store.reserve(binding(operation_id='another-original-operation'), submission())
    assert first.record.submission_digest != second.record.submission_digest
    assert await custody.store.open_submission(first.record) == await custody.store.open_submission(second.record)


@pytest.mark.asyncio
async def test_reservation_owns_original_bytes_before_transaction_awaits(custody, monkeypatch):
    store = custody.store
    original = submission()
    expected = MCPTaskSubmission(original.canonical_json)
    actual_lock = store._lock

    async def change_caller_object_during_lock():
        await actual_lock()
        object.__setattr__(original, 'canonical_json', submission(arguments={'changed': True}).canonical_json)

    monkeypatch.setattr(store, '_lock', change_caller_object_during_lock)
    reserved = await store.reserve(binding(), original)
    opened = None
    try:
        opened = await store.open_submission(reserved.record)
    except MCPTaskIntegrityError:
        pass
    assert opened == expected, 'Reservation must seal its own original bytes before awaiting'


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', ['operation', 'origin'])
async def test_reservation_owns_original_binding_before_transaction_awaits(custody, monkeypatch, changed):
    store = custody.store
    target = binding()
    expected = binding()
    original_lock = store._lock

    async def change_caller_binding_during_lock():
        await original_lock()
        if changed == 'operation':
            object.__setattr__(target, 'operation_id', 'substituted-during-await')
        else:
            object.__setattr__(target.origin, 'session_id', 'substituted-during-await')

    monkeypatch.setattr(store, '_lock', change_caller_binding_during_lock)
    reserved = await store.reserve(target, submission())
    opened = None
    try:
        opened = await store.open_submission(reserved.record)
    except MCPTaskIntegrityError:
        pass
    assert opened == submission() and reserved.record.binding == expected, 'Reservation must detach the entire original binding before awaiting'


@pytest.mark.asyncio
async def test_service_conversation_retains_real_caller_namespace_and_chat_delivery(custody):
    with caller_context_scope(CallerContext.authenticated('owned-service-subject')):
        kind, digest = current_service_operation_attribution()
    origin = MCPTaskServiceOrigin(kind, digest, 'genuine-service-conversation')
    target = binding(origin=origin)
    original = submission(origin=origin)
    reserved = await custody.store.reserve(target, original)
    opened = await (await custody.reopen()).open_submission(reserved.record)
    assert opened == original
    assert opened.delivery == MCPTaskDelivery('chat', origin.session_id)


@pytest.mark.asyncio
async def test_submission_ciphertexts_cannot_move_between_original_operations(custody):
    store = custody.store
    first = await store.reserve(binding(), submission())
    second = await store.reserve(binding(operation_id='another-original-operation'), submission())
    first_blob = (await sealed_columns(store, first.record.handle))[2]
    await store.database.execute('UPDATE mcp_task_records SET submission=? WHERE handle=?',
        (first_blob, second.record.handle))
    with pytest.raises(MCPTaskIntegrityError, match='cannot be opened'):
        await store.open_submission(second.record)
    assert await store.open_submission(first.record) == submission()


@pytest.mark.asyncio
@pytest.mark.parametrize('custody', ['sqlite'], indirect=True)
async def test_nonbinary_routing_payload_is_classified_as_custody_integrity(custody):
    store = custody.store
    await store.reserve(binding(), submission())
    await store.database.execute('UPDATE mcp_task_records SET payload=?', (7,))
    observed = None
    try:
        await store.list_remote_tasks()
    except (MCPTaskIntegrityError, TypeError) as error:
        observed = error
    assert type(observed) is MCPTaskIntegrityError, 'Nonbinary routing bytes must fail as custody integrity'
