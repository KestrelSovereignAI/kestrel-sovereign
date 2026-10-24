"""Admission metadata must never invoke a concurrently installed factory."""
import asyncio
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime
from kestrel_sovereign.storage.db.native_journal_programs import (
    NativeJournalConfigurationError, _NativeJournalSQLiteConnection,
)
from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
from tests.unit.test_signals_dispatcher import (
    db_path as _db_path_fixture, dispatcher_components as _dispatcher_fixture,
    _action_reg, _signal,
)

db_path = _db_path_fixture
dispatcher_components = _dispatcher_fixture


@pytest.mark.asyncio
async def test_admission_snapshot_atomically_fences_actual_text_factory(dispatcher_components, monkeypatch, tmp_path):
    c = dispatcher_components
    connection = c.backend._connection
    native = connection._conn
    worker_id = await connection._execute(threading.get_ident)
    loop = asyncio.get_running_loop()
    checked, continue_worker, proceed_setter, callback_called = (threading.Event() for _ in range(4))
    ready = asyncio.Event()
    thread_errors, callbacks, effects = [], [], []
    independent = sqlite3.connect(tmp_path / 'actual-factory-effect.db', check_same_thread=False)
    independent.execute('CREATE TABLE forbidden_effect(value INTEGER)')
    independent.commit()
    scope = ExecutionCustody(_OriginalWork())
    supported = _NativeJournalSQLiteConnection._original_journal_programs_supported

    def observe_native_check(actual):
        result = supported(actual)
        if actual is native and threading.get_ident() == worker_id and not checked.is_set():
            checked.set()
            if not continue_worker.wait(5):
                raise TimeoutError('test did not release the original native metadata check')
        return result

    def factory(value):
        callbacks.append(threading.get_ident())
        if not callback_called.is_set():
            independent.execute('INSERT INTO forbidden_effect VALUES (1)')
            independent.commit()
            callback_called.set()
        return value.decode('utf-8')

    def configure():
        installed = False
        try:
            if not checked.wait(5):
                raise TimeoutError('actual admission metadata never reached its native check')
            # Probe the actual lock, not a fake factory/admission result. The
            # old reader holds no lock; the fixed atomic check may hold it.
            available = native._journal_configuration_lock.acquire(blocking=False)
            if available:
                try:
                    connection.text_factory = factory
                    installed = True
                finally:
                    native._journal_configuration_lock.release()
            loop.call_soon_threadsafe(ready.set)
            if not proceed_setter.wait(5):
                raise TimeoutError('test did not release the actual configuration setter')
            if not available:
                try:
                    connection.text_factory = factory
                    installed = True
                except NativeJournalConfigurationError:
                    pass
            if installed:
                callback_called.wait(1)
                connection.text_factory = str
        except BaseException as error:
            thread_errors.append(error)
            loop.call_soon_threadsafe(ready.set)

    async def handler(payload):
        effects.append(True)
        return {'genuine': True}

    c.registry.register(_action_reg('sqlite.admission.factory', handler=handler))
    monkeypatch.setattr(_NativeJournalSQLiteConnection, '_original_journal_programs_supported', observe_native_check)
    setter = threading.Thread(target=configure, name='actual-sqlite-admission-setter')
    task = None
    setter.start()
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            task = asyncio.create_task(c.dispatcher.dispatch_signal(_signal('sqlite.admission.factory')))
        await asyncio.wait_for(ready.wait(), 5)
        assert checked.is_set() and not thread_errors
        scope.revoke('actual cutoff before original admission metadata reads')
        continue_worker.set()
        proceed_setter.set()
        outcome = await await_owned_task(task)
        setter.join(timeout=5)
        assert not setter.is_alive() and not thread_errors
        assert outcome.error is not None and not effects
        assert not callbacks, 'admission metadata invoked an application text factory after actual Work cutoff'
        assert independent.execute('SELECT COUNT(*) FROM forbidden_effect').fetchone() == (0,)
    finally:
        continue_worker.set()
        proceed_setter.set()
        if task is not None:
            await await_owned_task(task)
        setter.join(timeout=5)
        assert not setter.is_alive(), 'original configuration thread remained unjoined'
        connection.text_factory = str
        independent.close()
