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


# ---------------------------------------------------------------------------
# Keychain account resolution must not bind silently to a stale login
# ---------------------------------------------------------------------------

import json
from kestrel_sovereign.llm.anthropic_oauth import KeychainCredentialSource


def _item(exp_offset):
    return json.dumps({"claudeAiOauth": {
        "accessToken": TOKEN, "refreshToken": "r",
        "expiresAt": int((time.time() + exp_offset) * 1000),
    }})


class _FakeKeychain(KeychainCredentialSource):
    """Drives resolution without touching the real keychain."""

    def __init__(self, accounts, items, enumerate_ok=True, default=None):
        super().__init__()
        self._accounts = accounts
        self._items = items
        self._enumerate_ok = enumerate_ok
        self._default = default

    def _list_service_accounts(self):
        return list(self._accounts) if self._enumerate_ok else []

    def _read_account_raw(self, account):
        raw = self._items.get(account)
        return json.loads(raw) if raw else None

    def _run(self, args):
        if args and args[0] == "find-generic-password" and self._default:
            class R:
                returncode = 0
                stdout = f'"acct"<blob>="{self._default}"'
                stderr = ""
            return R()
        return None


def test_picks_the_live_login_over_a_stale_one():
    src = _FakeKeychain(
        ["stale", "live"],
        {"stale": _item(-86400 * 240), "live": _item(3600)},
    )
    assert src._resolve_account() == "live"


def test_warns_when_the_freshest_item_is_itself_expired(caplog):
    src = _FakeKeychain(["stale"], {"stale": _item(-86400 * 240)})
    with caplog.at_level("WARNING"):
        assert src._resolve_account() == "stale"
    assert any("itself expired" in r.message for r in caplog.records), (
        "binding to an expired credential must not be silent"
    )


def test_warns_when_falling_back_to_the_default_item(caplog):
    """The path that can bind a process to a dead credential for its whole
    lifetime — reached whenever enumeration fails, including a `security`
    timeout, since _run returns None then."""
    src = _FakeKeychain([], {}, enumerate_ok=False, default="whichever-is-first")
    with caplog.at_level("WARNING"):
        assert src._resolve_account() == "whichever-is-first"
    assert any("not \nnecessarily" in r.message.replace("  ", " ")
               or "necessarily the live login" in r.message
               for r in caplog.records), "silent fallback to the default item"


def test_a_single_valid_item_resolves_without_warning(caplog):
    """Control: the ordinary one-login case must stay quiet, or the warnings
    above are noise nobody reads."""
    src = _FakeKeychain(["only"], {"only": _item(3600)})
    with caplog.at_level("WARNING"):
        assert src._resolve_account() == "only"
    assert not [r for r in caplog.records if "Claude OAuth" in r.message]
