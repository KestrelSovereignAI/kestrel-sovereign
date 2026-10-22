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


def test_native_sqlite_text_configuration_is_fenced_through_real_commit():
    native = _NativeJournalSQLiteConnection(':memory:')
    try:
        native.execute('CREATE TABLE actual_effect(value TEXT)')
        native.commit()
        native._begin_original_journal()
        native.execute("INSERT INTO actual_effect VALUES('genuine')")
        with pytest.raises(NativeJournalConfigurationError):
            native.text_factory = bytes
        native.commit()
        assert native._journal_program_seal is None and not native.in_transaction
        assert native.execute('SELECT value FROM actual_effect').fetchone() == ('genuine',)
    finally:
        native.close()
