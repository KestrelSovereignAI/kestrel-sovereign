"""The plan route must name which credential it bound to.

The keychain can hold several `Claude Code-credentials` items — a stale login
beside the live one. `_resolve_account` picks one and caches it for the life
of the source, and every agent builds its own LLMService and therefore its own
source. So two agents in one process can bind to DIFFERENT Anthropic accounts.

When that happens the endpoint answers with a billing message that names no
account ("You're out of extra usage"), and the divergence is invisible: one
agent fails while another succeeds on the same route, model and host. Observed
live 2026-09-06, and it cost three rounds of investigation to narrow because
nothing recorded the binding.
"""

import time

import pytest

from kestrel_sovereign.llm.anthropic_oauth import (
    ClaudeOAuthTokenManager,
    FileCredentialSource,
    OAuthCredentials,
)

TOKEN = "sk-ant-oat01-AAAAAAAAAAAAAAAAAAAAAAAABBBBBBCCCCCC-ZZZZZZ"


def _creds(access=TOKEN):
    return OAuthCredentials(
        access=access, refresh="refresh-tok", expires_at=time.time() + 3600
    )


class _Source:
    def __init__(self, name):
        self._name = name

    def identity(self):
        return self._name

    def read(self):
        return _creds()

    def write(self, creds):
        return True


def test_identity_names_the_source_item():
    m = ClaudeOAuthTokenManager(_creds(), source=_Source("jason"))
    assert m.credential_identity().startswith("jason#")


def test_identity_distinguishes_two_accounts():
    """The whole point: two managers on different items must not look alike."""
    a = ClaudeOAuthTokenManager(_creds("sk-ant-oat01-" + "A" * 40 + "AAAAAA"),
                                source=_Source("jason"))
    b = ClaudeOAuthTokenManager(_creds("sk-ant-oat01-" + "B" * 40 + "BBBBBB"),
                                source=_Source("jasonschulz"))
    assert a.credential_identity() != b.credential_identity()


def test_identity_distinguishes_two_tokens_on_the_same_item():
    """A refresh that swaps accounts under one item must still be visible."""
    a = ClaudeOAuthTokenManager(_creds("sk-ant-oat01-" + "A" * 40 + "AAAAAA"),
                                source=_Source("jason"))
    b = ClaudeOAuthTokenManager(_creds("sk-ant-oat01-" + "A" * 40 + "ZZZZZZ"),
                                source=_Source("jason"))
    assert a.credential_identity() != b.credential_identity()


def test_identity_never_contains_the_token():
    """Identity is written to logs. A fingerprint must not be usable as a
    credential, and must not expose the token's head, which carries its
    prefix and type."""
    m = ClaudeOAuthTokenManager(_creds(), source=_Source("jason"))
    ident = m.credential_identity()
    assert TOKEN not in ident
    assert TOKEN[:20] not in ident
    assert len(ident) < len(TOKEN)


def test_identity_survives_a_source_that_cannot_answer():
    """Reporting must never break a request."""
    class _Broken:
        def identity(self):
            raise RuntimeError("keychain locked")

    m = ClaudeOAuthTokenManager(_creds(), source=_Broken())
    with pytest.raises(RuntimeError):
        _Broken().identity()          # control: the source really does raise
    assert isinstance(m.credential_identity(), str)


def test_identity_with_no_source_is_still_a_string():
    m = ClaudeOAuthTokenManager(_creds(), source=None)
    assert isinstance(m.credential_identity(), str)


def test_file_source_identity_is_its_path(tmp_path):
    p = tmp_path / ".credentials.json"
    assert FileCredentialSource(p).identity() == str(p)
