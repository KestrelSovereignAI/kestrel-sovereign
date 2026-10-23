"""Native connections whose program configuration can be fenced through exit.

Ordinary database customization is unchanged. A customized connection cannot
carry the original terminal journal, and its configuration cannot change while
that journal owns its native transaction/reset. These are configuration fences,
not SQL executors or grants of work authority.
"""
import asyncio
import ipaddress
import math
import sqlite3
import threading
from contextvars import ContextVar
from dataclasses import dataclass

_ORIGINAL_POSTGRES_CONNECT = ContextVar('original_postgres_journal_connect', default=None)


def _has_exact_native_type(value, *supported):
    """Type identity only: membership/equality can execute a metaclass."""
    return any(type(value) is native for native in supported)


class NativeJournalConfigurationError(RuntimeError):
    """The native connection cannot enter/change an original journal profile."""


class NativeJournalDurabilityRefusal(NativeJournalConfigurationError):
    """Native preflight refused BEFORE the original sqlite COMMIT call."""


def _native_tcp_endpoint(peer):
    if (type(peer) is not tuple or len(peer) not in (2, 4)
            or type(peer[0]) is not str or type(peer[1]) is not int
            or not 0 < peer[1] < 65536):
        raise NativeJournalConfigurationError('Original native socket peer is unsupported')
    host = peer[0]
    if len(peer) == 4 and peer[3]:
        if type(peer[3]) is not int or peer[3] < 0:
            raise NativeJournalConfigurationError('Original native IPv6 scope is unsupported')
        host += '%' + str(peer[3])
    try:
        host = ipaddress.ip_address(host).compressed
    except ValueError as error:
        raise NativeJournalConfigurationError('Original native socket peer must be numeric') from error
    return host, peer[1]


class _OriginalJournalConnectLoop:
    """Original native loop, fixed socket peer and independently pinned TLS name.

    asyncpg has no separate public hostaddr argument. This private connection-
    local delegate keeps dialing the admitted numeric peer while its native TLS
    handshake still verifies/sends the original hostname. It changes no global
    resolver or event loop and carries no work admission.
    """
    __slots__ = ('_loop', '_endpoint', '_tls_server_hostname')

    def __init__(self, loop, endpoint, tls_server_hostname):
        self._loop = loop
        self._endpoint = _native_tcp_endpoint(endpoint)
        self._tls_server_hostname = tls_server_hostname

    def __getattr__(self, name):
        return getattr(self._loop, name)

    async def create_connection(self, protocol_factory, host=None, port=None, **kwargs):
        if (host, port) != self._endpoint:
            raise NativeJournalConfigurationError('Original native socket peer differs')
        if kwargs.get('ssl'):
            kwargs['server_hostname'] = self._tls_server_hostname
        return await self._loop.create_connection(protocol_factory, host, port, **kwargs)

    async def start_tls(self, transport, protocol, sslcontext, *, server_hostname=None, **kwargs):
        peer = transport.get_extra_info('peername')
        if _native_tcp_endpoint(peer) != self._endpoint:
            raise NativeJournalConfigurationError('Original native TLS socket peer differs')
        return await self._loop.start_tls(transport, protocol, sslcontext,
            server_hostname=self._tls_server_hostname, **kwargs)


def _register_program(connection):
    if connection._journal_program_seal is not None:
        raise NativeJournalConfigurationError('Native program configuration is fenced by an original journal')
    # Monotonic before actual registration, including failed/removal calls.
    # SQLite's caller holds its configuration lock across this check+mark;
    # asyncpg's supported configuration entry runs on its original event loop.
    connection._journal_programs_configured = True


def _original_sqlite_scalar_adapters_registered():
    # Registry membership can call an application's metaclass __eq__ on hash
    # collisions. Snapshot native keys and compare identities only instead.
    registry = sqlite3.adapters
    if type(registry) is not dict:
        raise NativeJournalConfigurationError('Original journal requires the native adapter registry')
    for key in tuple(dict.keys(registry)):
        if (type(key) is tuple and len(key) == 2 and key[1] is sqlite3.PrepareProtocol
                and any(key[0] is kind for kind in (type(None), bool, int, float, str, bytes))):
            return True
    return False


def _original_sqlite_query(query, parameters, sql_limit, *, database_encoding='UTF-8'):
    """Encode ONLY the Core-owned fixed positional journal SQL as native data.

    CPython's parameter adapter lookup can race global register_adapter while
    prepare releases the GIL. These exact scalar literals leave ZERO native
    binds, so that process-global registry cannot execute a callback here.
    Text is native-database-encoding hex CAST, never application SQL; BLOBs
    are hex too. SQLite interprets CAST(BLOB AS TEXT) in that database encoding.
    Ordinary database execution does not use this private journal compiler.
    """
    if type(query) is not str or type(parameters) is not tuple:
        raise NativeJournalConfigurationError('Original journal requires fixed native SQL and values')
    if type(database_encoding) is not str or database_encoding not in ('UTF-8', 'UTF-16le', 'UTF-16be'):
        raise NativeJournalConfigurationError('Original journal native text encoding is unsupported')
    if not parameters:
        # Trusted parameter-free catalog SQL may contain '?' inside a quoted
        # identifier/literal. With no native binds there is no adapter lookup.
        if len(query.encode('utf-8')) > sql_limit:
            raise sqlite3.DataError('Original journal exceeds native statement length limit')
        return query
    parts = str.split(query, '?')
    if len(parts) != len(parameters) + 1:
        raise NativeJournalConfigurationError('Original journal requires fixed positional placeholders')
    remaining = sql_limit - sum(len(part.encode('utf-8')) for part in parts)
    literals = []
    for value in parameters:
        kind = type(value)
        if value is None:
            literal = 'NULL'
        elif kind is int:
            if not -(2 ** 63) <= value < 2 ** 63:
                raise OverflowError('Python int too large to convert to SQLite INTEGER')
            literal = int.__repr__(value)
        elif kind is float:
            if math.isnan(value):
                literal = 'NULL'  # sqlite3_bind_double maps NaN to NULL.
            elif math.isinf(value):
                literal = '9e999' if value > 0 else '-9e999'
            elif value == 0:
                literal = '-0.0' if math.copysign(1, value) < 0 else '0.0'
            else:
                # SQLite's decimal parser can differ from the original IEEE
                # value by one ULP. Exact 53-bit mantissa and powers of two
                # avoid that parser, with <=19 bounded exact binary steps.
                mantissa, exponent = math.frexp(value)
                coefficient = int(math.ldexp(mantissa, 53))
                exponent -= 53
                terms = ['CAST(' + int.__repr__(coefficient) + ' AS REAL)']
                while exponent:
                    shift = min(abs(exponent), 62)
                    terms.append((' * ' if exponent > 0 else ' / ')
                        + 'CAST(' + int.__repr__(1 << shift) + ' AS REAL)')
                    exponent += -shift if exponent > 0 else shift
                literal = '(' + ''.join(terms) + ')'
        elif kind is str or kind is bytes:
            encoded = str.encode(value, database_encoding) if kind is str else value
            prefix, suffix = ("CAST(X'", "' AS TEXT)") if kind is str else ("X'", "'")
            if len(encoded) * 2 + len(prefix) + len(suffix) > remaining:
                raise sqlite3.DataError('Original journal exceeds native statement length limit')
            literal = prefix + bytes.hex(encoded) + suffix
        else:
            raise NativeJournalConfigurationError('Original journal binds only plain native scalar parameters')
        remaining -= len(literal)
        if remaining < 0:
            raise sqlite3.DataError('Original journal exceeds native statement length limit')
        literals.append(literal)
    return ''.join(part + literal for part, literal in zip(parts, literals)) + parts[-1]


class _NativeJournalSQLiteCursor(sqlite3.Cursor):
    def execute(self, query, parameters=()):
        native = self.connection
        if native._journal_program_seal is not None:
            if (not native._original_journal_programs_supported()
                    or type(parameters) is not tuple
                    or any(not _has_exact_native_type(value, type(None), int, float, str, bytes) for value in parameters)
                    or _original_sqlite_scalar_adapters_registered()):
                raise NativeJournalConfigurationError('Original journal binds only plain native scalar parameters')
            query = _original_sqlite_query(query, parameters,
                sqlite3.Connection.getlimit(native, sqlite3.SQLITE_LIMIT_SQL_LENGTH),
                database_encoding=native._journal_program_seal[-2])
            return super().execute(query)  # Zero native binds: no global adapter dispatch.
        return super().execute(query, parameters)


class _NativeJournalSQLiteConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        self._journal_configuration_lock = threading.RLock()
        self._journal_programs_configured = False
        self._journal_program_seal = None
        super().__init__(*args, **kwargs)
        # Native autocommit getters are thread-affine. Mirror only supported
        # configuration changes here/on their actual successful native setter;
        # final transaction completion is still checked on the native worker.
        self._journal_autocommit_legacy = (not hasattr(sqlite3.Connection, 'autocommit')
            or self.autocommit == sqlite3.LEGACY_TRANSACTION_CONTROL)
        # Core-owned native authorization observes preparation, not SQL text
        # heuristics. A previously cached mutating PRAGMA already retired this
        # profile at its first preparation, so reusing it cannot bypass us.
        super().set_authorizer(self._original_journal_catalog_authorizer)

    def _original_journal_catalog_authorizer(self, action, name, value, database, source):
        if (action == sqlite3.SQLITE_PRAGMA and value is not None
                and name.lower() in ('schema_version', 'writable_schema')):
            self._journal_programs_configured = True
            if self._journal_program_seal is not None:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def __setattr__(self, name, value):
        if name in ('text_factory', 'autocommit', 'isolation_level'):
            with self._journal_configuration_lock:
                if self._journal_program_seal is not None:
                    raise NativeJournalConfigurationError('Original journal native configuration is fenced')
                super().__setattr__(name, value)
                if name == 'autocommit':
                    self._journal_autocommit_legacy = self.autocommit == sqlite3.LEGACY_TRANSACTION_CONTROL
            return
        super().__setattr__(name, value)

    def _program_call(self, method_name, /, *args, **kwargs):
        with self._journal_configuration_lock:
            _register_program(self)
        return getattr(super(), method_name)(*args, **kwargs)

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

    def deserialize(self, *args, **kwargs):
        return self._program_call('deserialize', *args, **kwargs)

    def setlimit(self, *args, **kwargs):
        return self._program_call('setlimit', *args, **kwargs)

    if hasattr(sqlite3.Connection, 'setconfig'):
        def setconfig(self, *args, **kwargs):
            return self._program_call('setconfig', *args, **kwargs)

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
        # Fence application programs before BEGIN can wait for another native
        # writer. The durable configuration belongs to the acquired lock, not
        # to the earlier (possibly stale) external-commit generation.
        with self._journal_configuration_lock:
            if not self._original_journal_programs_supported() or self._journal_program_seal is not None:
                raise NativeJournalConfigurationError('Original journal requires a plain native SQLite connection')
            self._journal_program_seal = object()
        try:
            cursor = super().execute('BEGIN IMMEDIATE')
            cursor.close()
            self._journal_program_seal = self._original_journal_configuration()
        except BaseException:
            if not self.in_transaction:
                self._journal_program_seal = None
            raise

    def _original_journal_configuration(self):
        # Fixed native reads bypass application cursor/row factories. Values
        # are sampled on the SQLite worker, never from a thread-affine getter
        # on the event loop. No SQL text comes from application configuration.
        cursor = super().cursor(factory=sqlite3.Cursor)
        cursor.row_factory = None
        try:
            values = []
            for name in ('journal_mode', 'synchronous', 'fullfsync',
                         'checkpoint_fullfsync', 'wal_autocheckpoint', 'encoding', 'data_version'):
                cursor.execute('PRAGMA main.' + name)
                row = cursor.fetchone()
                if row is None or len(row) != 1 or not _has_exact_native_type(row[0], str, int):
                    raise NativeJournalConfigurationError('Original native durability configuration is unsupported')
                values.append(row[0])
            return tuple(values)
        finally:
            cursor.close()

    def commit(self):
        if (self._journal_program_seal is not None
                and self._original_journal_configuration() != self._journal_program_seal):
            raise NativeJournalDurabilityRefusal('Original journal durability changed before native COMMIT')
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
    asyncpg = None
    NativeJournalPostgresConnection = None
else:
    @dataclass(frozen=True, slots=True, repr=False)
    class _OriginalJournalCancelConnection:
        """Read-only native cancellation view; reuse asyncpg's actual executor.

        A reused session still remembers its DNS alias in _addr. Capture its
        actual socket and independent TLS name before sealing, without changing
        the live connection's ordinary loop/recipe or copying cancellation I/O.
        """
        _loop: object
        _addr: object
        _params: object
        _protocol: object
        _cancellations: set

        @classmethod
        def capture(cls, connection):
            peer = connection._transport.get_extra_info('peername')
            loop = connection._loop
            if type(peer) is tuple:
                address = _native_tcp_endpoint(peer)
                tls = connection._transport.get_extra_info('ssl_object')
                name = tls.server_hostname if tls is not None else None
                loop = _OriginalJournalConnectLoop(loop, address, name)
            elif type(peer) is str:
                address = peer
            else:
                raise NativeJournalConfigurationError('Original native cancellation socket is unsupported')
            return cls(loop, address, connection._params, connection._protocol,
                       connection._cancellations)

    class NativeJournalPostgresConnection(asyncpg.Connection):
        """Pass as asyncpg.create_pool(connection_class=...) for native journals."""
        __slots__ = ('_journal_programs_configured', '_journal_program_seal', '_journal_cancel_connection')

        def __init__(self, *args, **kwargs):
            self._journal_programs_configured = False
            self._journal_program_seal = None
            self._journal_cancel_connection = None
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
            self._journal_cancel_connection = _OriginalJournalCancelConnection.capture(self)
            self._journal_program_seal = (asyncio.current_task(), native_state)

        def _retire_original_journal_refusal(self):
            # Refusing application configuration must not execute it during
            # termination either. Do not invoke public listener-removal hooks
            # or a reset/query program; retire this exact native socket once.
            self._termination_listeners.clear()
            self._listeners.clear()
            self._log_listeners.clear()
            self._query_loggers.clear()
            super().terminate()

        async def _cancel(self, waiter):
            try:
                cancellation = self._journal_cancel_connection
                if cancellation is not None:
                    return await asyncpg.Connection._cancel(cancellation, waiter)
                return await super()._cancel(waiter)
            finally:
                # asyncpg queries current_task(self._loop). Its connection-
                # local TLS delegate still schedules on the actual native
                # loop, so retire that exact physical cancellation Task too.
                self._cancellations.discard(asyncio.current_task())

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
            self._journal_cancel_connection = None
            return True

        def get_reset_query(self):
            # The stock reset runs AFTER COMMIT/local search_path restoration.
            # Its advisory-unlock function must not resolve in an app schema.
            # RESET ALL deliberately excludes these PostgreSQL authorization
            # settings. A returned session must not lend its last user's role
            # or session authorization to another original native operation.
            return 'RESET SESSION AUTHORIZATION;\nRESET ROLE;\n' + super().get_reset_query().replace(
                'SELECT pg_advisory_unlock_all();',
                'SELECT pg_catalog.pg_advisory_unlock_all();')

        async def _original_journal_principal(self):
            row = await self.fetchrow('''SELECT CURRENT_USER, SESSION_USER,
                pg_catalog.current_setting('role'),
                pg_catalog.current_setting('session_authorization')''')
            values = tuple(row)
            if len(values) != 4 or any(type(value) is not str for value in values):
                raise NativeJournalConfigurationError('Original native principal is not canonical')
            return values


async def _native_journal_connect(*args, **kwargs):
    """Check the original recipe at actual reconnect, including queued waits.

    Outside a journal this is the ordinary native connector. Within one, only
    its genuine fixed original recipe may reach asyncpg; detached values cannot
    be changed by a caller retaining a host list or startup-settings dictionary.
    """
    if asyncpg is None:
        raise RuntimeError('Native PostgreSQL driver is unavailable')
    original = _ORIGINAL_POSTGRES_CONNECT.get()
    if original is not None:
        journal, backend = original
        from kestrel_sovereign.signals.outcome_custody import _require_original_signal_outcome
        _require_original_signal_outcome(journal, backend)
        args, kwargs = journal._postgres_pool.parameters_for_connect(args, kwargs)
    return await asyncpg.connection.connect(*args, **kwargs)
