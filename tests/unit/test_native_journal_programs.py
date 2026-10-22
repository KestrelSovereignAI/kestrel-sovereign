"""Real native connection program registration and transaction-lifetime fences."""
import sqlite3

import pytest

from kestrel_sovereign.storage.db.native_journal_programs import (
    _NativeJournalSQLiteConnection, NativeJournalConfigurationError,
)


@pytest.mark.parametrize('method,arguments', [
    ('create_function', ('journal_identity', 1, lambda value: value)),
    ('create_aggregate', ('journal_aggregate', 1, None)),
    ('create_window_function', ('journal_window', 1, None)),
    ('create_collation', ('journal_order', lambda left, right: 0)),
    ('set_authorizer', (lambda *args: sqlite3.SQLITE_OK,)),
    ('set_progress_handler', (lambda: 0, 1)),
    ('set_trace_callback', (lambda query: None,)),
    ('enable_load_extension', (False,)),
    ('load_extension', ('must-not-load',)),
    ('deserialize', (b'not-a-native-database',)),
    ('setlimit', (sqlite3.SQLITE_LIMIT_LENGTH, 1000000)),
])
def test_native_sqlite_program_registration_is_fenced_until_actual_rollback(method, arguments):
    native = _NativeJournalSQLiteConnection(':memory:')
    try:
        assert native._original_journal_programs_supported()
        native._begin_original_journal()
        assert native.in_transaction
        with pytest.raises(NativeJournalConfigurationError):
            getattr(native, method)(*arguments)
        assert native._original_journal_programs_supported()
        native.rollback()
        assert native._journal_program_seal is None and not native.in_transaction
        # Removing a callback is still configuration, even if it is harmless.
        native.set_trace_callback(None)
        assert not native._original_journal_programs_supported()
    finally:
        native.close()


def test_native_sqlite_dbconfig_is_fenced_when_supported():
    if not hasattr(sqlite3.Connection, 'setconfig'):
        pytest.skip('Native dbconfig was added in Python 3.12')
    native = _NativeJournalSQLiteConnection(':memory:')
    try:
        native._begin_original_journal()
        with pytest.raises(NativeJournalConfigurationError):
            native.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
        native.rollback()
        native.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
        assert not native._original_journal_programs_supported()
    finally:
        native.close()


def test_native_postgres_proof_targets_only_original_physical_relation():
    from kestrel_sovereign.signals.store import SignalLogStore
    assert 'FROM ONLY "owned".signal_log WHERE ' in SignalLogStore._outcome_match_query('"owned".signal_log', postgres=True)
    assert 'FROM main.signal_log WHERE ' in SignalLogStore._outcome_match_query('main.signal_log', postgres=False)


def test_sqlite_native_commit_rechecks_durability_in_same_worker_operation():
    native = _NativeJournalSQLiteConnection(':memory:')
    try:
        native.execute('CREATE TABLE original_effect(value TEXT)')
        native.commit()
        native._begin_original_journal()
        native.execute("INSERT INTO original_effect VALUES('genuine')")
        current = native.execute('PRAGMA fullfsync').fetchone()[0]
        native.execute('PRAGMA fullfsync=' + str(1 - current)).close()
        with pytest.raises(NativeJournalConfigurationError):
            native.commit()
        assert native.in_transaction, 'configuration refusal must happen before actual COMMIT'
        native.rollback()
        assert native.execute('SELECT COUNT(*) FROM original_effect').fetchone() == (0,)
    finally:
        native.close()


def test_native_connect_recipe_detaches_nested_values_without_application_hooks():
    from kestrel_sovereign.signals.outcome_custody import _freeze_native_connect_configuration, _restore_native_connect_configuration
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    hosts = ['owned-original-endpoint']
    settings = {'search_path': 'owned_original_schema'}
    recipe = (('owned-native-dsn',), {'host': hosts, 'server_settings': settings})
    frozen = _freeze_native_connect_configuration(recipe)
    hosts[0] = 'replacement-endpoint'
    settings['search_path'] = 'replacement_schema'
    assert _freeze_native_connect_configuration(recipe) != frozen
    args, restored = _restore_native_connect_configuration(frozen)
    assert args == ('owned-native-dsn',)
    assert restored == {'host': ['owned-original-endpoint'], 'server_settings': {'search_path': 'owned_original_schema'}}
    restored['host'].append('another-endpoint')
    assert _restore_native_connect_configuration(frozen)[1]['host'] == ['owned-original-endpoint']
    hooks = []
    class ForeignMeta(type):
        def __eq__(self, other):
            hooks.append('metaclass equality')
            return False
    class ForeignData(dict, metaclass=ForeignMeta):
        def __iter__(self):
            hooks.append('iteration')
            return super().__iter__()
    with pytest.raises(ExecutionAuthorityError):
        _freeze_native_connect_configuration({'server_settings': ForeignData()})
    assert not hooks
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ExecutionAuthorityError):
        _freeze_native_connect_configuration(cyclic)


@pytest.mark.parametrize('name,value', [('text_factory', bytes), ('isolation_level', None), ('autocommit', True)])
def test_native_sqlite_text_configuration_is_fenced_through_real_commit(name, value):
    native = _NativeJournalSQLiteConnection(':memory:')
    try:
        if name == 'autocommit' and not hasattr(native, name):
            pytest.skip('Native autocommit property was added in Python 3.12')
        native.execute('CREATE TABLE actual_effect(value TEXT)')
        native.commit()
        native._begin_original_journal()
        native.execute("INSERT INTO actual_effect VALUES('genuine')")
        with pytest.raises(NativeJournalConfigurationError):
            setattr(native, name, value)
        native.commit()
        assert native._journal_program_seal is None and not native.in_transaction
        assert native.execute('SELECT value FROM actual_effect').fetchone() == ('genuine',)
    finally:
        native.close()
