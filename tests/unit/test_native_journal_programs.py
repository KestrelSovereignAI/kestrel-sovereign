"""Real native connection program registration and transaction-lifetime fences."""
import sqlite3

import pytest

from kestrel_sovereign.storage.db.native_journal_programs import (
    _NativeJournalSQLiteConnection, NativeJournalConfigurationError,
)


@pytest.mark.parametrize('value', [None, -(2 ** 63), 2 ** 63 - 1, 1.25, -0.0,
    float('inf'), float('-inf'), float('nan'), '', "quote' and ?; DROP TABLE kept; --\0雪", b'', b"'\x00\xff"])
def test_original_sqlite_literal_values_match_native_binding(value):
    import math
    from kestrel_sovereign.storage.db.native_journal_programs import _original_sqlite_query
    native = _NativeJournalSQLiteConnection(':memory:')
    reference = sqlite3.connect(':memory:')
    try:
        native.execute('CREATE TABLE kept(value TEXT)').close()
        native.commit()
        expected = reference.execute('SELECT typeof(?), ?', (value, value)).fetchone()
        compiled = _original_sqlite_query('SELECT typeof(?), ?', (value, value), 100000)
        assert reference.execute(compiled).fetchone() == expected
        native._begin_original_journal()
        actual = native.execute('SELECT typeof(?), ?', (value, value)).fetchone()
        assert actual == expected
        if type(value) is float and value == 0:
            assert math.copysign(1, actual[1]) == math.copysign(1, expected[1])
        assert native.execute('SELECT COUNT(*) FROM kept').fetchone() == (0,)
        native.rollback()
    finally:
        native.close()
        reference.close()


@pytest.mark.parametrize('value', [2 ** 63, -(2 ** 63) - 1, '\ud800', True, object()])
def test_original_sqlite_literal_compiler_refuses_noncanonical_or_unrepresentable_data(value):
    from kestrel_sovereign.storage.db.native_journal_programs import _original_sqlite_query
    with pytest.raises((OverflowError, UnicodeEncodeError, NativeJournalConfigurationError)):
        _original_sqlite_query('SELECT ?', (value,), 1000)
    native = _NativeJournalSQLiteConnection(':memory:')
    try:
        native._begin_original_journal()
        with pytest.raises((OverflowError, UnicodeEncodeError, NativeJournalConfigurationError)):
            native.execute('SELECT ?', (value,))
        native.rollback()
    finally:
        native.close()


def test_original_sqlite_literal_compiler_respects_native_statement_bound():
    from kestrel_sovereign.storage.db.native_journal_programs import _original_sqlite_query
    with pytest.raises(sqlite3.DataError):
        _original_sqlite_query('SELECT ?', ('too long',), 16)


def test_original_sqlite_parameter_free_literals_keep_quoted_placeholder_text():
    native = _NativeJournalSQLiteConnection(':memory:')
    try:
        native._begin_original_journal()
        assert native.execute("SELECT '?' AS \"native?name\"").fetchone() == ('?',)
        native.rollback()
    finally:
        native.close()


def test_original_sqlite_float_literals_preserve_native_ieee_values():
    import math
    import random
    import struct
    from kestrel_sovereign.storage.db.native_journal_programs import _original_sqlite_query
    generator = random.Random(0)
    values = [struct.unpack('!d', generator.getrandbits(64).to_bytes(8, 'big'))[0] for _ in range(2048)]
    for exponent in (-1074, -1073, -1022, -1000, -1, 0, 1, 1022, 1023):
        value = math.ldexp(1.0, exponent)
        values.extend((value, -value, math.nextafter(value, 0), math.nextafter(value, math.inf)))
    with sqlite3.connect(':memory:') as reference:
        for value in values:
            if not math.isfinite(value):
                continue
            expected = reference.execute('SELECT ?', (value,)).fetchone()[0]
            query = _original_sqlite_query('SELECT ?', (value,), 10000)
            actual = reference.execute(query).fetchone()[0]
            assert struct.pack('!d', actual) == struct.pack('!d', expected), value.hex()


def _owned_tls_contexts(tmp_path, name):
    import ssl
    from datetime import datetime, timedelta, timezone
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(name)]), critical=False)
        .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / 'owned-cert.pem', tmp_path / 'owned-key.pem'
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    client_context = ssl.create_default_context(cafile=str(cert_path))
    assert client_context.check_hostname and client_context.verify_mode == ssl.CERT_REQUIRED
    return server_context, client_context, cert_path


@pytest.mark.asyncio
@pytest.mark.parametrize('upgrade', [False, True])
async def test_original_numeric_peer_preserves_actual_tls_hostname(tmp_path, upgrade):
    import asyncio
    from kestrel_sovereign._async_ownership import await_owned_task
    from kestrel_sovereign.storage.db.native_journal_programs import _OriginalJournalConnectLoop
    name = 'original-journal-peer.invalid'
    server_context, client_context, _ = _owned_tls_contexts(tmp_path, name)
    names, server_transports, upgrades = [], [], []
    server_context.set_servername_callback(lambda obj, servername, context: names.append(servername))
    loop = asyncio.get_running_loop()
    class Protocol(asyncio.Protocol):
        def __init__(self, *, server=False):
            self.server = server
            self.closed = loop.create_future()
        def connection_made(self, transport):
            if self.server:
                server_transports.append(transport)
                if upgrade:
                    upgrades.append(asyncio.create_task(loop.start_tls(transport, self,
                        server_context, server_side=True, ssl_handshake_timeout=3)))
        def connection_lost(self, exc):
            if not self.closed.done():
                self.closed.set_result(exc)
    server = await loop.create_server(lambda: Protocol(server=True), '127.0.0.1', 0,
        ssl=None if upgrade else server_context)
    endpoint = server.sockets[0].getsockname()[:2]
    pinned = _OriginalJournalConnectLoop(loop, endpoint, name)
    client_transport = None
    protocol = Protocol()
    try:
        client_transport, _ = await pinned.create_connection(lambda: protocol, *endpoint,
            **({} if upgrade else {'ssl': client_context, 'ssl_handshake_timeout': 3}))
        if upgrade:
            client_transport = await pinned.start_tls(client_transport, protocol, client_context,
                server_hostname=endpoint[0], ssl_handshake_timeout=3)
            for task in upgrades:
                outcome = await await_owned_task(task)
                assert outcome.error is None
                server_transports.append(outcome.result)
        assert client_transport.get_extra_info('peername')[:2] == endpoint
        assert client_transport.get_extra_info('ssl_object').server_hostname == name
        assert names == [name], 'numeric dialing changed actual TLS SNI / certificate identity'
    finally:
        if client_transport is not None:
            client_transport.close()
            await asyncio.wait_for(protocol.closed, 3)
        server.close()
        await server.wait_closed()
        for transport in server_transports:
            transport.close()
        for task in upgrades:
            if not task.done():
                task.cancel()
            await await_owned_task(task)


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


@pytest.mark.parametrize('pragma', ['schema_version=1', 'writable_schema=OFF'])
def test_sqlite_sql_catalog_reconfiguration_is_monotonic_even_for_cached_pragma(pragma):
    native = _NativeJournalSQLiteConnection(':memory:')
    try:
        native.execute('PRAGMA ' + pragma).close()
        native.execute('PRAGMA ' + pragma).close()  # Same cached statement.
        assert not native._original_journal_programs_supported()
        with pytest.raises(NativeJournalConfigurationError):
            native._begin_original_journal()
        native.execute('CREATE TABLE ordinary(value TEXT)').close()
        native.commit()  # Ordinary configuration remains usable, not journal-privileged.
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
