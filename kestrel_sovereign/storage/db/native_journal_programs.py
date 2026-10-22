"""Native connections whose program configuration can be fenced through exit.

Ordinary database customization is unchanged. A customized connection cannot
carry the original terminal journal, and its configuration cannot change while
that journal owns its native transaction/reset. These are configuration fences,
not SQL executors or grants of work authority.
"""
import asyncio
import sqlite3


class NativeJournalConfigurationError(RuntimeError):
    """The native connection cannot enter/change an original journal profile."""


def _register_program(connection):
    if connection._journal_program_seal is not None:
        raise NativeJournalConfigurationError('Native program configuration is fenced by an original journal')
    # Monotonic, including failed registration and later removal. Reconnect is
    # the supported way to obtain another demonstrably plain native session.
    connection._journal_programs_configured = True


class _NativeJournalSQLiteCursor(sqlite3.Cursor):
    def execute(self, query, parameters=()):
        native = self.connection
        if native._journal_program_seal is not None:
            if (not native._original_journal_programs_supported()
                    or type(parameters) is not tuple
                    or any(type(value) not in (type(None), int, float, str, bytes) for value in parameters)
                    or any((type(value), sqlite3.PrepareProtocol) in sqlite3.adapters for value in parameters)):
                raise NativeJournalConfigurationError('Original journal binds only plain native scalar parameters')
        return super().execute(query, parameters)


class _NativeJournalSQLiteConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        self._journal_programs_configured = False
        self._journal_program_seal = None
        super().__init__(*args, **kwargs)
        # Native autocommit getters are thread-affine. Mirror only supported
        # configuration changes here/on their actual successful native setter;
        # final transaction completion is still checked on the native worker.
        self._journal_autocommit_legacy = (not hasattr(sqlite3.Connection, 'autocommit')
            or self.autocommit == sqlite3.LEGACY_TRANSACTION_CONTROL)

    def __setattr__(self, name, value):
        if name in ('text_factory', 'autocommit', 'isolation_level') and self._journal_program_seal is not None:
            raise NativeJournalConfigurationError('Original journal native configuration is fenced')
        super().__setattr__(name, value)
        if name == 'autocommit':
            self._journal_autocommit_legacy = self.autocommit == sqlite3.LEGACY_TRANSACTION_CONTROL

    def _program_call(self, name, *args, **kwargs):
        _register_program(self)
        return getattr(super(), name)(*args, **kwargs)

    def create_function(self, *args, **kwargs):
        return self._program_call('create_function', *args, **kwargs)

    def create_aggregate(self, *args, **kwargs):
        return self._program_call('create_aggregate', *args, **kwargs)

    def create_window_function(self, *args, **kwargs):
        return self._program_call('create_window_function', *args, **kwargs)

    def create_collation(self, *args, **kwargs):
        return self._program_call('create_collation', *args, **kwargs)

    def set_authorizer(self, *args, **kwargs):
        return self._program_call('set_authorizer', *args, **kwargs)

    def set_progress_handler(self, *args, **kwargs):
        return self._program_call('set_progress_handler', *args, **kwargs)

    def set_trace_callback(self, *args, **kwargs):
        return self._program_call('set_trace_callback', *args, **kwargs)

    def enable_load_extension(self, *args, **kwargs):
        return self._program_call('enable_load_extension', *args, **kwargs)

    def load_extension(self, *args, **kwargs):
        return self._program_call('load_extension', *args, **kwargs)

    def _original_journal_programs_supported(self):
        return (not self._journal_programs_configured and self.text_factory is str
                and self._journal_autocommit_legacy)

    def cursor(self, factory=None):
        if factory is None:
            factory = _NativeJournalSQLiteCursor
        elif self._journal_program_seal is not None and factory is not _NativeJournalSQLiteCursor:
            raise NativeJournalConfigurationError('Original journal cursor factory is fenced')
        return super().cursor(factory=factory)

    def execute(self, query, parameters=()):
        cursor = self.cursor()
        try:
            return cursor.execute(query, parameters)
        except BaseException as original:
            try:
                cursor.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup('Original native cursor and cleanup failures', [original, cleanup]) from None
            raise

    def _begin_original_journal(self):
        # This operation is queued by the original writer under the existing
        # write guard. Check+seal+BEGIN run in one native worker operation.
        if not self._original_journal_programs_supported() or self._journal_program_seal is not None:
            raise NativeJournalConfigurationError('Original journal requires a plain native SQLite connection')
        self._journal_program_seal = object()
        try:
            cursor = super().execute('BEGIN IMMEDIATE')
            cursor.close()
        except BaseException:
            if not self.in_transaction:
                self._journal_program_seal = None
            raise

    def commit(self):
        result = super().commit()
        if self._journal_program_seal is not None and self.in_transaction:
            raise NativeJournalConfigurationError('Original native journal COMMIT did not end its transaction')
        self._journal_program_seal = None
        return result

    def rollback(self):
        result = super().rollback()
        if self._journal_program_seal is not None and self.in_transaction:
            raise NativeJournalConfigurationError('Original native journal rollback did not end its transaction')
        self._journal_program_seal = None
        return result


try:
    import asyncpg
except ImportError:
    NativeJournalPostgresConnection = None
else:
    class NativeJournalPostgresConnection(asyncpg.Connection):
        """Pass as asyncpg.create_pool(connection_class=...) for native journals."""
        __slots__ = ('_journal_programs_configured', '_journal_program_seal')

        def __init__(self, *args, **kwargs):
            self._journal_programs_configured = False
            self._journal_program_seal = None
            super().__init__(*args, **kwargs)

        async def set_type_codec(self, *args, **kwargs):
            _register_program(self)
            return await super().set_type_codec(*args, **kwargs)

        async def reset_type_codec(self, *args, **kwargs):
            _register_program(self)
            return await super().reset_type_codec(*args, **kwargs)

        async def set_builtin_type_codec(self, *args, **kwargs):
            _register_program(self)
            return await super().set_builtin_type_codec(*args, **kwargs)

        async def add_listener(self, *args, **kwargs):
            _register_program(self)
            return await super().add_listener(*args, **kwargs)

        async def remove_listener(self, *args, **kwargs):
            _register_program(self)
            return await super().remove_listener(*args, **kwargs)

        def add_query_logger(self, *args, **kwargs):
            _register_program(self)
            return super().add_query_logger(*args, **kwargs)

        def remove_query_logger(self, *args, **kwargs):
            _register_program(self)
            return super().remove_query_logger(*args, **kwargs)

        def add_log_listener(self, *args, **kwargs):
            _register_program(self)
            return super().add_log_listener(*args, **kwargs)

        def remove_log_listener(self, *args, **kwargs):
            _register_program(self)
            return super().remove_log_listener(*args, **kwargs)

        def add_termination_listener(self, *args, **kwargs):
            _register_program(self)
            return super().add_termination_listener(*args, **kwargs)

        def remove_termination_listener(self, *args, **kwargs):
            _register_program(self)
            return super().remove_termination_listener(*args, **kwargs)

        def _original_journal_programs_supported(self):
            if (self.is_closed() or self._journal_programs_configured or self._query_loggers or self._listeners
                    or self._log_listeners or self._termination_listeners):
                return False
            settings = self._protocol.get_settings()
            return all(settings.get_data_codec(oid, format=fmt) is
                       settings.get_data_codec(oid, format=fmt, ignore_custom_codec=True)
                       for oid in asyncpg.protocol.protocol.BUILTIN_TYPE_OID_MAP
                       for fmt in (-1, 0, 1))

        async def _wait_original_journal_retirement(self):
            # Pool return precedes its old writer's final fence release. A
            # successor borrows observation of that exact original task; it
            # neither clears its seal nor cancels/adopts its native cleanup.
            seal = self._journal_program_seal
            if seal is None:
                return
            owner, _ = seal
            if owner is asyncio.current_task():
                raise NativeJournalConfigurationError('Original journal cannot reenter its native session')
            from kestrel_sovereign._async_ownership import await_owned_task
            await await_owned_task(owner, defer_cancellation=False)
            if self._journal_program_seal is not None:
                raise NativeJournalConfigurationError('Previous original journal has unresolved native retirement')

        def _begin_original_journal(self, native_state):
            if not self._original_journal_programs_supported() or self._journal_program_seal is not None:
                raise NativeJournalConfigurationError('Original journal requires a plain native PostgreSQL connection')
            self._journal_program_seal = (asyncio.current_task(), native_state)

        def _end_original_journal(self):
            if self._journal_program_seal is None:
                return False
            owner, native_state = self._journal_program_seal
            if owner is not asyncio.current_task():
                raise NativeJournalConfigurationError('Only the original native journal owner can release its program fence')
            retirement = native_state.retirement.owner
            if not self.is_closed() and (retirement is None or not retirement.done()):
                return False  # Keep an unresolved original retirement sealed.
            self._journal_program_seal = None
            return True

        def get_reset_query(self):
            # The stock reset runs AFTER COMMIT/local search_path restoration.
            # Its advisory-unlock function must not resolve in an app schema.
            return super().get_reset_query().replace(
                'SELECT pg_advisory_unlock_all();',
                'SELECT pg_catalog.pg_advisory_unlock_all();')
