"""An original journal cannot retain a native lazy external trust lookup."""
import ssl

import pytest
from kestrel_sovereign.storage.db.native_journal_programs import (
    NativeJournalConfigurationError, _adopt_native_tls_context,
    _native_tls_policy_generation, _OriginalNativeTLSPolicy,
)


@pytest.mark.parametrize('loader', ['keyword', 'positional', 'default_paths', 'default_certs'])
def test_lazy_trust_lookup_is_not_an_original_tls_policy(tmp_path, monkeypatch, loader):
    directory = tmp_path / 'owned-empty-ca-directory'
    directory.mkdir()
    monkeypatch.setenv('SSL_CERT_DIR', str(directory))
    monkeypatch.setenv('SSL_CERT_FILE', str(tmp_path / 'absent-default-ca-file'))
    # A driver-created context is adopted before application configuration.
    # Real native calls below remain usable for ordinary TLS, but no complete
    # trust snapshot can be inferred from their generation or current roots.
    context = _adopt_native_tls_context(ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    if loader == 'keyword':
        context.load_verify_locations(capath=str(directory))
    elif loader == 'positional':
        context.load_verify_locations(None, str(directory))
    elif loader == 'default_paths':
        context.set_default_verify_paths()
    else:
        context.load_default_certs()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    with pytest.raises(NativeJournalConfigurationError, match='external trust'):
        with _OriginalNativeTLSPolicy(context, _native_tls_policy_generation(context)):
            pass


def test_failed_lazy_trust_configuration_cannot_be_made_original_again():
    context = _adopt_native_tls_context(ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    with pytest.raises(TypeError):
        context.load_verify_locations(capath=object())
    context.set_ciphers('DEFAULT')
    with pytest.raises(NativeJournalConfigurationError, match='external trust'):
        _native_tls_policy_generation(context)
