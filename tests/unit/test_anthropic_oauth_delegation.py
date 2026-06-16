"""Unit tests for Claude Code credential delegation (anthropic:plan).

Covers the credential-source abstraction (Keychain + file), auto-discovery,
from_sources precedence, and the refresh/adopt/write-back lifecycle. All
external effects (the `security` CLI, the OAuth endpoint) are mocked — no real
keychain or token is touched.
"""
import json
import subprocess
from types import SimpleNamespace

import pytest

from kestrel_sovereign.llm import anthropic_oauth as oa
from kestrel_sovereign.llm.anthropic_oauth import (
    ClaudeOAuthTokenManager,
    FileCredentialSource,
    KeychainCredentialSource,
    OAuthCredentials,
    discover_claude_code_source,
)

KEYCHAIN_JSON = {
    "claudeAiOauth": {
        "accessToken": "sk-ant-oat-OLD",
        "refreshToken": "sk-ant-ort-1",
        "expiresAt": 1_700_000_000_000,  # ms (2023-11-14) — long past
        "scopes": ["user:inference"],
        "subscriptionType": "max",
    }
}


def _fake_security(monkeypatch, read_payload=KEYCHAIN_JSON, read_rc=0, captured=None, account="testuser"):
    """Patch subprocess.run to emulate the `security` CLI.

    Models both calls the source makes: account discovery
    (``find-generic-password -s SERVICE`` → attributes incl. ``acct``) and the
    secret read (``... -a ACCOUNT -w`` → the JSON password).
    """

    def fake_run(argv, **kw):
        if argv[:2] == ["security", "find-generic-password"]:
            if "-w" not in argv:  # account discovery (attributes only)
                if read_rc != 0:
                    return SimpleNamespace(returncode=read_rc, stdout="", stderr="")
                return SimpleNamespace(
                    returncode=0,
                    stdout=f'    "acct"<blob>="{account}"\n    "svce"<blob>="{oa._CLAUDE_KEYCHAIN_SERVICE}"\n',
                    stderr="",
                )
            # secret read — must target the discovered account (when discovery
            # succeeded; on a missing item the account falls back to getuser()).
            assert "-a" in argv
            if read_rc == 0:
                assert argv[argv.index("-a") + 1] == account
            out = json.dumps(read_payload) if read_payload is not None else ""
            return SimpleNamespace(returncode=read_rc, stdout=out, stderr="")
        if argv[:2] == ["security", "add-generic-password"]:
            if captured is not None:
                captured["argv"] = argv
                captured["account"] = argv[argv.index("-a") + 1]
                captured["written"] = json.loads(argv[-1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected argv {argv}")

    monkeypatch.setattr(oa.subprocess, "run", fake_run)


# ---------------------------------------------------------------------------
# KeychainCredentialSource
# ---------------------------------------------------------------------------


def test_keychain_read_parses_claude_oauth(monkeypatch):
    _fake_security(monkeypatch)
    creds = KeychainCredentialSource().read()
    assert creds.access == "sk-ant-oat-OLD"
    assert creds.refresh == "sk-ant-ort-1"
    assert creds.expires_at == 1_700_000_000.0  # ms → s


def test_keychain_read_none_on_missing_item(monkeypatch):
    _fake_security(monkeypatch, read_rc=44, read_payload=None)
    assert KeychainCredentialSource().read() is None


def test_keychain_read_none_on_oserror(monkeypatch):
    def boom(*a, **k):
        raise OSError("security not found")

    monkeypatch.setattr(oa.subprocess, "run", boom)
    assert KeychainCredentialSource().read() is None


def test_keychain_write_merges_and_preserves_fields(monkeypatch):
    captured = {}
    _fake_security(monkeypatch, captured=captured)
    ok = KeychainCredentialSource().write(
        OAuthCredentials(access="sk-ant-oat-NEW", refresh="sk-ant-ort-2", expires_at=1_700_000_000.0)
    )
    assert ok is True
    block = captured["written"]["claudeAiOauth"]
    assert block["accessToken"] == "sk-ant-oat-NEW"
    assert block["refreshToken"] == "sk-ant-ort-2"
    assert block["expiresAt"] == 1_700_000_000_000  # written back in ms
    assert block["scopes"] == ["user:inference"]  # preserved
    assert block["subscriptionType"] == "max"  # preserved
    # Write targets the SAME account discovered for the read (not a constant).
    assert captured["account"] == "testuser"


def test_keychain_write_false_when_item_absent(monkeypatch):
    _fake_security(monkeypatch, read_rc=44, read_payload=None)
    assert KeychainCredentialSource().write(OAuthCredentials(access="x")) is False


# ---------------------------------------------------------------------------
# FileCredentialSource + discovery
# ---------------------------------------------------------------------------


def test_file_source_roundtrip_preserves_wrapper(tmp_path):
    f = tmp_path / "creds.json"
    f.write_text(json.dumps({"claudeAiOauth": {"accessToken": "a", "refreshToken": "r", "expiresAt": 1, "scopes": ["s"]}}))
    src = FileCredentialSource(f)
    assert src.read().access == "a"
    src.write(OAuthCredentials(access="a2", refresh="r2", expires_at=1_700_000_000.0))
    block = json.loads(f.read_text())["claudeAiOauth"]
    assert block["accessToken"] == "a2" and block["refreshToken"] == "r2"
    assert block["expiresAt"] == 1_700_000_000_000 and block["scopes"] == ["s"]


def test_discover_prefers_keychain_on_darwin(monkeypatch, tmp_path):
    _fake_security(monkeypatch)
    src = discover_claude_code_source(platform="darwin", home=tmp_path)
    assert isinstance(src, KeychainCredentialSource)


def test_discover_falls_back_to_file_off_darwin(monkeypatch, tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "a", "refreshToken": "r", "expiresAt": 1}})
    )
    src = discover_claude_code_source(platform="linux", home=tmp_path)
    assert isinstance(src, FileCredentialSource)


def test_discover_none_when_nothing(monkeypatch, tmp_path):
    src = discover_claude_code_source(platform="linux", home=tmp_path)
    assert src is None


# ---------------------------------------------------------------------------
# from_sources precedence + delegation gating
# ---------------------------------------------------------------------------


def test_from_sources_static_token_beats_discovery(monkeypatch):
    _fake_security(monkeypatch)  # keychain available
    monkeypatch.setattr(oa.sys, "platform", "darwin")
    mgr = ClaudeOAuthTokenManager.from_sources(
        static_token="sk-ant-oat-static", credentials_path=None, delegate=True
    )
    assert mgr.initial_access_token == "sk-ant-oat-static"
    assert mgr._source is None  # static token is not refreshable


def test_from_sources_delegates_when_no_static_token(monkeypatch):
    _fake_security(monkeypatch)
    monkeypatch.setattr(oa.sys, "platform", "darwin")
    mgr = ClaudeOAuthTokenManager.from_sources(
        static_token=None, credentials_path=None, delegate=True
    )
    assert isinstance(mgr._source, KeychainCredentialSource)
    assert mgr.initial_access_token == "sk-ant-oat-OLD"


def test_from_sources_no_delegation_for_api_route(monkeypatch):
    _fake_security(monkeypatch)  # keychain present but delegate=False
    monkeypatch.setattr(oa.sys, "platform", "darwin")
    mgr = ClaudeOAuthTokenManager.from_sources(
        static_token=None, credentials_path=None, delegate=False
    )
    assert mgr is None


def test_from_sources_delegate_env_disable(monkeypatch):
    _fake_security(monkeypatch)
    monkeypatch.setattr(oa.sys, "platform", "darwin")
    monkeypatch.setenv("KESTREL_ANTHROPIC_OAUTH_DELEGATE", "0")
    mgr = ClaudeOAuthTokenManager.from_sources(static_token=None, credentials_path=None)
    assert mgr is None


def test_env_disable_overrides_explicit_delegate_true(monkeypatch):
    """The operator escape hatch must win even when the caller (plan route)
    passes delegate=True."""
    _fake_security(monkeypatch)
    monkeypatch.setattr(oa.sys, "platform", "darwin")
    monkeypatch.setenv("KESTREL_ANTHROPIC_OAUTH_DELEGATE", "false")
    mgr = ClaudeOAuthTokenManager.from_sources(
        static_token=None, credentials_path=None, delegate=True
    )
    assert mgr is None


# ---------------------------------------------------------------------------
# access_token: adopt-then-refresh + write-back
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_access_token_refreshes_and_writes_back(monkeypatch):
    captured = {}
    _fake_security(monkeypatch, captured=captured)
    src = KeychainCredentialSource()
    mgr = ClaudeOAuthTokenManager(src.read(), source=src)  # expired creds from keychain
    monkeypatch.setattr(oa.time, "time", lambda: 1_900_000_000.0)

    async def fake_refresh(refresh_token, **kw):
        assert refresh_token == "sk-ant-ort-1"
        return OAuthCredentials(access="sk-ant-oat-NEW", refresh="sk-ant-ort-2", expires_at=9_999_999_999.0)

    monkeypatch.setattr(oa, "refresh_anthropic_token", fake_refresh)
    token = await mgr.access_token()
    assert token == "sk-ant-oat-NEW"
    # Rotation was written back to the keychain.
    assert captured["written"]["claudeAiOauth"]["accessToken"] == "sk-ant-oat-NEW"
    assert captured["written"]["claudeAiOauth"]["refreshToken"] == "sk-ant-ort-2"


@pytest.mark.asyncio
async def test_access_token_adopts_concurrent_refresh_without_minting(monkeypatch):
    """If the shared store was already refreshed (e.g. by Claude Code), adopt
    that token instead of minting our own."""
    fresh = {
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat-FROM-CLAUDE-CODE",
            "refreshToken": "sk-ant-ort-x",
            "expiresAt": 9_999_999_999_000,  # ms, far future
        }
    }
    _fake_security(monkeypatch, read_payload=fresh)
    src = KeychainCredentialSource()
    # Seed the manager with a STALE in-memory copy.
    mgr = ClaudeOAuthTokenManager(
        OAuthCredentials(access="stale", refresh="sk-ant-ort-old", expires_at=1.0), source=src
    )
    monkeypatch.setattr(oa.time, "time", lambda: 1_900_000_000.0)

    async def boom(*a, **k):
        raise AssertionError("should not mint a new token when store is already fresh")

    monkeypatch.setattr(oa, "refresh_anthropic_token", boom)
    assert await mgr.access_token() == "sk-ant-oat-FROM-CLAUDE-CODE"
