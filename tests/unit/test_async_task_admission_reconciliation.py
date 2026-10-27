"""Exact native Task acceptance when a factory withholds publication."""
import asyncio
import inspect

import pytest
from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, await_owned_task, create_owned_task,
)
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, execution_cleanup_debts, is_execution_control_error,
)

EAGER = [False, True] if hasattr(asyncio, 'eager_task_factory') else [False]


@pytest.mark.asyncio
@pytest.mark.parametrize('eager', EAGER)
@pytest.mark.parametrize('private', [False, True])
@pytest.mark.parametrize('task_kind', ['native', 'lying-get-coro', 'readonly-metadata'])
async def test_accepted_factory_failure_recovers_original_native_task_without_payload(
    eager, private, task_kind,
):
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    admission = OwnedTaskAdmission()
    accepted, entered = [], []
    original_error = RuntimeError('factory failed after native acceptance')
    metadata_error = RuntimeError('accepted metadata refused')

    async def original():
        entered.append(True)

    class NativeSubclass(asyncio.Task):
        def get_coro(self):
            if task_kind == 'lying-get-coro':
                raise AssertionError('native reconciliation called a user descriptor')
            return super().get_coro()

        def __setattr__(self, name, value):
            if task_kind == 'readonly-metadata' and name.startswith('_kestrel'):
                raise metadata_error
            return super().__setattr__(name, value)

    def factory(loop, coro, **kwargs):
        options = dict(kwargs)
        if eager:
            options['eager_start'] = True
        task = (asyncio.Task if task_kind == 'native' else NativeSubclass)(
            coro, loop=loop, **options)
        accepted.append(task)
        raise original_error

    payload = original()
    try:
        loop.set_task_factory(factory)
        if private:
            task = create_owned_task(payload)
        else:
            with pytest.raises(BaseException) as opening:
                create_owned_task(payload, admission=admission)
            assert original_error in tuple(_explicit_control_chain(opening.value))
            task = admission.task
        loop.set_task_factory(previous)
        assert task is accepted[0]
        assert asyncio.Task.get_coro(task) is not payload
        assert not admission._acceptance_unknown
        outcome = await await_owned_task(task)
        assert original_error in tuple(_explicit_control_chain(outcome.error))
        if task_kind == 'readonly-metadata':
            assert metadata_error in tuple(_explicit_control_chain(outcome.error))
        assert not entered
        assert inspect.getcoroutinestate(payload) == inspect.CORO_CLOSED
        assert task.done()
    finally:
        loop.set_task_factory(previous)
        for task in accepted:
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['zero', 'opaque', 'multiple', 'query'])
async def test_ambiguous_factory_acceptance_is_explicit_debt_never_absence(
    monkeypatch, failure,
):
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    admission = OwnedTaskAdmission()
    accepted, entered = [], []
    original_error = RuntimeError('unpublished factory outcome')

    async def original():
        entered.append(True)

    def factory(loop, coro, **kwargs):
        if failure == 'opaque':
            async def outer():
                return await coro
            accepted.append(asyncio.Task(outer(), loop=loop, **kwargs))
        elif failure != 'zero':
            accepted.append(asyncio.Task(coro, loop=loop, **kwargs))
            if failure == 'multiple':
                accepted.append(asyncio.Task(coro, loop=loop, **kwargs))
        raise original_error

    payload = original()
    try:
        loop.set_task_factory(factory)
        with monkeypatch.context() as patch:
            if failure == 'query':
                def refused_registry(*args):
                    raise RuntimeError('original native registry query failed')
                patch.setattr(asyncio, 'all_tasks', refused_registry)
            with pytest.raises(BaseException) as opening:
                create_owned_task(payload, admission=admission)
        loop.set_task_factory(previous)
        assert original_error in tuple(_explicit_control_chain(opening.value))
        assert admission.task is None and admission._acceptance_unknown
        assert is_execution_control_error(opening.value)
        assert execution_cleanup_debts(opening.value)
        assert not entered
        assert inspect.getcoroutinestate(payload) == inspect.CORO_CLOSED
        if failure == 'multiple':
            assert set(admission._reconciled_tasks) == set(accepted)
    finally:
        loop.set_task_factory(previous)
        # These exact native Tasks were accepted by THIS fixture; production
        # did not claim their join. No unrelated Task is adopted or cleaned.
        for task in accepted:
            await await_owned_task(task)
        assert not entered


@pytest.mark.asyncio
async def test_factory_source_cancellation_survives_native_acceptance_and_join():
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    admission, accepted = OwnedTaskAdmission(), []
    original = asyncio.CancelledError('original factory cancellation')
    entered = []

    async def payload():
        entered.append(True)

    def factory(loop, coro, **kwargs):
        task = asyncio.Task(coro, loop=loop, **kwargs)
        accepted.append(task)
        raise original

    try:
        loop.set_task_factory(factory)
        with pytest.raises(asyncio.CancelledError) as opening:
            create_owned_task(payload(), admission=admission, retain_originals=True)
        loop.set_task_factory(previous)
        assert opening.value is original and admission.task is accepted[0]
        result = await await_owned_task(admission.task)
        assert original in tuple(_explicit_control_chain(result.error))
        assert not entered
    finally:
        loop.set_task_factory(previous)
        for task in accepted:
            await await_owned_task(task)


@pytest.mark.asyncio
async def test_opaque_constructor_override_cannot_certify_nonacceptance(monkeypatch):
    loop = asyncio.get_running_loop()
    assert loop.get_task_factory() is None
    admission, accepted, entered = OwnedTaskAdmission(), [], []
    refusal = RuntimeError('owned constructor wrapper failed after acceptance')
    async def payload():
        entered.append(True)
    def constructor(coroutine, **kwargs):
        async def outer():
            return await coroutine
        accepted.append(asyncio.Task(outer(), loop=loop))
        raise refusal
    operation = payload()
    with monkeypatch.context() as patch:
        patch.setattr(asyncio, 'create_task', constructor)
        with pytest.raises(BaseException) as caught:
            create_owned_task(operation, admission=admission)
    assert refusal in _explicit_control_chain(caught.value)
    assert admission._acceptance_unknown and admission.task is None
    assert execution_cleanup_debts(caught.value)
    for task in accepted:
        await await_owned_task(task)
    assert not entered and inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_constructor_identity_descriptor_failure_retains_original_and_debt(monkeypatch):
    loop = asyncio.get_running_loop()
    admission, wrappers, entered = OwnedTaskAdmission(), [], []
    original = RuntimeError('original native construction refused')
    descriptor = RuntimeError('constructor identity inspection refused')
    async def payload():
        entered.append(True)
    class RefusedConstructor:
        def __call__(self, coroutine, **kwargs):
            wrappers.append(coroutine)
            raise original
        @property
        def __func__(self):
            raise descriptor
    operation = payload()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(loop, 'create_task', RefusedConstructor())
            with pytest.raises(BaseException) as caught:
                create_owned_task(operation, admission=admission)
        assert original in _explicit_control_chain(caught.value)
        assert descriptor in _explicit_control_chain(caught.value)
        assert admission._acceptance_unknown and execution_cleanup_debts(caught.value)
        assert admission.task is None and not entered
        assert inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
    finally:
        # This fixture's constructor accepted NO task. It alone can close its
        # captured unaccepted wrapper; product cannot infer that from failure.
        for wrapper in wrappers:
            wrapper.close()
        operation.close()
