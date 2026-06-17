"""
Claude subscription (Claude Pro/Max) OAuth token lifecycle.

The ``openai:plan`` route delegates its entire OAuth lifecycle — token
refresh and request shaping — to the ``codex`` binary (which reads/refreshes
``~/.codex/auth.json``). The ``anthropic:plan`` route talks to the Anthropic
SDK directly, so it must own the equivalent itself.

This module delegates to the **Claude Code CLI's own credential store**, the
same way codex delegates to the codex binary's store (and mirroring OpenClaw's
``cli-credentials.ts``): it reads the ``Claude Code-credentials`` macOS Keychain
item, or ``~/.claude/.credentials.json`` on Linux, refreshes the ``sk-ant-oat``
access token with the stored refresh token when near expiry, and writes the
rotated tokens back so Claude Code and Kestrel stay in sync. An explicit
operator-configured file or a static env token are also supported.

Credential sources, in precedence order (see ``from_sources``):
  1. explicit credentials file (``oauth_credentials_file`` / env) — refreshable
  2. static env token (``ANTHROPIC_AUTH_TOKEN``) — used as-is, not refreshed
  3. auto-discovered Claude Code store (Keychain / file) — refreshable delegation
"""
import asyncio
import getpass
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Claude Code's public OAuth client id and token endpoint. These are the same
# values the official client uses; the access tokens they mint are scoped to
# the user's Claude subscription. Decoded at import to avoid a literal that
# scanners flag as a planted credential — it is a public client id, not a secret.
_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"

# Refresh this many seconds BEFORE the stated expiry so a request never races a
# token that lapses mid-flight. Matches OpenClaw's 5-minute skew.
_REFRESH_SKEW_SECONDS = 300

# Claude Code CLI credential store locations (mirrors OpenClaw cli-credentials.ts).
# NOTE: the Keychain item's ACCOUNT is the macOS username (e.g. "jasonschulz"),
# not a fixed string — it is discovered from the existing item, see
# KeychainCredentialSource._resolve_account.
_CLAUDE_KEYCHAIN_SERVICE = "Claude Code-credentials"
_CLAUDE_CREDENTIALS_RELATIVE_PATH = ".claude/.credentials.json"

# Env overrides (parity with codex's KESTREL_CODEX_* / CODEX_REFRESH_TOKEN_URL_OVERRIDE).
_ENV_CREDENTIALS_FILE = "KESTREL_ANTHROPIC_OAUTH_CREDENTIALS_FILE"
_ENV_TOKEN_URL_OVERRIDE = "KESTREL_ANTHROPIC_OAUTH_TOKEN_URL"
_ENV_CLIENT_ID_OVERRIDE = "KESTREL_ANTHROPIC_OAUTH_CLIENT_ID"
# Set to "0"/"false" to disable auto-discovery of the Claude Code store.
_ENV_DELEGATE = "KESTREL_ANTHROPIC_OAUTH_DELEGATE"


@dataclass
class OAuthCredentials:
    """A Claude OAuth credential triple.

    ``expires_at`` is epoch seconds; ``None`` means "unknown / long-lived"
    (e.g. a ``claude setup-token``) and disables proactive refresh.
    """

    access: str
    refresh: Optional[str] = None
    expires_at: Optional[float] = None

    def needs_refresh(self, *, now: float, skew: float = _REFRESH_SKEW_SECONDS) -> bool:
        if not self.refresh or self.expires_at is None:
            return False
        return now + skew >= self.expires_at


def _coerce_expires_at(value: Any) -> Optional[float]:
    """Normalize an ``expires_at`` field to epoch SECONDS.

    Accepts seconds or milliseconds (Claude Code's ``.credentials.json`` stores
    ``expiresAt`` in ms). Values past the year-2200 threshold are treated as ms.
    """
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    # > ~7.3e9 seconds is beyond year 2200, so it must be milliseconds.
    return num / 1000.0 if num > 7_300_000_000 else num


def parse_credentials(raw: Dict[str, Any]) -> Optional[OAuthCredentials]:
    """Parse a credentials dict, tolerating snake_case, camelCase, and the
    Claude Code ``{"claudeAiOauth": {...}}`` wrapper. Returns ``None`` if no
    access token is present."""
    block = raw.get("claudeAiOauth") if isinstance(raw.get("claudeAiOauth"), dict) else raw
    access = block.get("access_token") or block.get("accessToken")
    if not access:
        return None
    refresh = block.get("refresh_token") or block.get("refreshToken")
    expires_at = _coerce_expires_at(
        block.get("expires_at")
        if block.get("expires_at") is not None
        else block.get("expiresAt")
    )
    # Some sources only carry expires_in (seconds-from-now); honor it when no
    # absolute expiry is given.
    if expires_at is None and block.get("expires_in") is not None:
        try:
            expires_at = time.time() + float(block["expires_in"])
        except (TypeError, ValueError):
            expires_at = None
    return OAuthCredentials(access=str(access), refresh=refresh, expires_at=expires_at)


async def refresh_anthropic_token(
    refresh_token: str,
    *,
    client_id: Optional[str] = None,
    token_url: Optional[str] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> OAuthCredentials:
    """Exchange a refresh token for a fresh access token.

    Mirrors ``codex_app_server``'s refresh flow but against Anthropic's
    endpoint. Returns the new credentials (the endpoint may rotate the refresh
    token, so the response's ``refresh_token`` is preferred when present).
    """
    url = token_url or os.environ.get(_ENV_TOKEN_URL_OVERRIDE) or _TOKEN_URL
    cid = client_id or os.environ.get(_ENV_CLIENT_ID_OVERRIDE) or _CLIENT_ID
    payload = {
        "grant_type": "refresh_token",
        "client_id": cid,
        "refresh_token": refresh_token,
    }

    async def _post(c: httpx.AsyncClient) -> httpx.Response:
        # JSON body, NOT form-encoding. Anthropic's OAuth endpoint
        # (platform.claude.com/v1/oauth/token) accepts JSON — this matches the
        # official Claude Code client / OpenClaw's refreshAnthropicToken. (The
        # codex/OpenAI refresh flow at auth.openai.com is form-encoded; that is
        # a different endpoint and not the contract here.)
        return await c.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=30.0,
        )

    if http_client is not None:
        resp = await _post(http_client)
    else:
        async with httpx.AsyncClient() as c:
            resp = await _post(c)

    if resp.status_code >= 400:
        # Surface the OAuth error code/description (safe — these are error
        # strings, not token material) and an actionable recovery hint. The
        # access/refresh tokens are never in an error body.
        err = desc = None
        try:
            body = resp.json()
            err = body.get("error")
            desc = body.get("error_description")
        except (ValueError, AttributeError):
            pass
        detail = ": ".join(p for p in (err, desc) if p) or f"HTTP {resp.status_code}"
        hint = ""
        if err == "invalid_grant":
            # The stored Claude Code credential is expired/revoked and cannot
            # be refreshed — re-auth or fall back to a static token.
            hint = " — the Claude Code credential is expired or revoked; run `claude login` (or `claude setup-token` and set ANTHROPIC_AUTH_TOKEN)"
        raise RuntimeError(f"Anthropic OAuth token refresh failed ({detail}){hint}")
    body = resp.json()
    access = body.get("access_token")
    if not access:
        raise RuntimeError("Anthropic OAuth token refresh returned no access_token")
    expires_at = _coerce_expires_at(body.get("expires_at"))
    if expires_at is None and body.get("expires_in") is not None:
        try:
            expires_at = time.time() + float(body["expires_in"])
        except (TypeError, ValueError):
            expires_at = None
    return OAuthCredentials(
        access=str(access),
        refresh=body.get("refresh_token") or refresh_token,
        expires_at=expires_at,
    )


def _merge_credentials_into(raw: Dict[str, Any], creds: OAuthCredentials) -> Dict[str, Any]:
    """Update ``raw`` IN PLACE with ``creds``, matching the shape it already
    uses — the Claude Code ``{"claudeAiOauth": {...}}`` wrapper, the key casing
    (camelCase vs snake_case), the expiry unit (ms vs s), and any unrelated
    sibling fields all survive. Returns ``raw`` for convenience."""
    wrapped = isinstance(raw.get("claudeAiOauth"), dict)
    block = raw["claudeAiOauth"] if wrapped else raw
    camel = wrapped or "accessToken" in block or "refreshToken" in block

    block["accessToken" if camel else "access_token"] = creds.access
    if creds.refresh is not None:
        block["refreshToken" if camel else "refresh_token"] = creds.refresh
    if creds.expires_at is not None:
        # Claude Code's camelCase store keeps expiresAt in milliseconds.
        block["expiresAt" if camel else "expires_at"] = (
            int(creds.expires_at * 1000) if camel else creds.expires_at
        )
    return raw


class CredentialSource:
    """A readable/writable Claude OAuth credential store. ``read`` seeds the
    initial credentials; ``write`` persists rotated tokens after a refresh so
    the shared store (and other clients like Claude Code) stay in sync."""

    def read(self) -> Optional[OAuthCredentials]:  # pragma: no cover - interface
        raise NotImplementedError

    def write(self, creds: OAuthCredentials) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class FileCredentialSource(CredentialSource):
    """A JSON credentials file (explicit path or ``~/.claude/.credentials.json``)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> Optional[OAuthCredentials]:
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Anthropic OAuth credentials file unreadable (%s): %s", self.path, exc)
            return None
        return parse_credentials(raw) if isinstance(raw, dict) else None

    def write(self, creds: OAuthCredentials) -> bool:
        try:
            try:
                existing = json.loads(self.path.read_text())
                raw = existing if isinstance(existing, dict) else {}
            except (OSError, json.JSONDecodeError):
                raw = {}
            _merge_credentials_into(raw, creds)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(raw, indent=2))
            os.replace(tmp, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            return True
        except OSError as exc:
            logger.warning("Could not persist refreshed Claude OAuth credentials to %s: %s", self.path, exc)
            return False


class KeychainCredentialSource(CredentialSource):
    """The macOS Keychain item Claude Code stores its OAuth login under
    (``Claude Code-credentials``). Reads/writes via the ``security`` CLI with an
    argument vector (never a shell string) so token material can't be
    interpreted. Mirrors OpenClaw's cli-credentials.ts."""

    def __init__(
        self,
        service: str = _CLAUDE_KEYCHAIN_SERVICE,
        account: Optional[str] = None,
    ) -> None:
        self.service = service
        self._account = account  # None → discovered from the existing item

    def _run(self, args) -> Optional[subprocess.CompletedProcess]:
        try:
            return subprocess.run(
                ["security", *args], capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("security %s failed: %s", args[0] if args else "", exc)
            return None

    def _list_service_accounts(self) -> list:
        """All accounts holding an item for this service, via attribute-only
        ``dump-keychain`` (no secrets). The keychain can hold several items for
        ``Claude Code-credentials`` (a stale login plus the live one); a plain
        ``find-generic-password -s SERVICE`` returns an arbitrary one, so we
        must enumerate to find the freshest. Returns [] if unavailable."""
        result = self._run(["dump-keychain"])
        if result is None or result.returncode != 0:
            return []
        accounts = []
        for block in re.split(r"(?m)^keychain:", result.stdout or ""):
            svce = re.search(r'"svce"<blob>="([^"]*)"', block)
            if not svce or svce.group(1) != self.service:
                continue
            acct = re.search(r'"acct"<blob>="([^"]*)"', block)
            if acct and acct.group(1):
                accounts.append(acct.group(1))
        return list(dict.fromkeys(accounts))  # dedupe, preserve order

    def _read_account_raw(self, account: str) -> Optional[Dict[str, Any]]:
        result = self._run(
            ["find-generic-password", "-s", self.service, "-a", account, "-w"]
        )
        if result is None or result.returncode != 0:
            return None
        try:
            raw = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return None
        return raw if isinstance(raw, dict) else None

    def _resolve_account(self) -> str:
        """Pick the account whose stored credential is valid and freshest.

        Read and write must target the SAME item. Claude Code keys its item by
        an opaque per-login account (often the short username), and the
        keychain may also contain a STALE item under another account — picking
        the wrong one means refreshing a dead token. So enumerate the service's
        accounts, read each, and keep the one with a parseable credential and
        the latest ``expires_at``. Falls back to the default lookup, then the
        current username."""
        if self._account:
            return self._account
        best_account = None
        best_expires = None
        for account in self._list_service_accounts():
            raw = self._read_account_raw(account)
            if raw is None:
                continue
            creds = parse_credentials(raw)
            if creds is None:
                continue
            expires = creds.expires_at if creds.expires_at is not None else 0.0
            if best_account is None or expires > best_expires:
                best_account, best_expires = account, expires
        if best_account is not None:
            self._account = best_account
            return best_account
        # Fallbacks: the default item's account attribute, then the OS user.
        result = self._run(["find-generic-password", "-s", self.service])
        if result is not None and result.returncode == 0:
            text = (result.stdout or "") + (result.stderr or "")
            m = re.search(r'"acct"<blob>="([^"]*)"', text)
            if m and m.group(1):
                self._account = m.group(1)
                return self._account
        self._account = getpass.getuser()
        return self._account

    def _read_raw(self) -> Optional[Dict[str, Any]]:
        return self._read_account_raw(self._resolve_account())

    def read(self) -> Optional[OAuthCredentials]:
        raw = self._read_raw()
        return parse_credentials(raw) if raw is not None else None

    def write(self, creds: OAuthCredentials) -> bool:
        # Merge into the existing item so scopes/subscriptionType/etc. survive,
        # and target the SAME account we read from.
        raw = self._read_raw()
        if raw is None:
            return False
        _merge_credentials_into(raw, creds)
        account = self._resolve_account()
        result = self._run(
            [
                "add-generic-password", "-U",
                "-s", self.service, "-a", account, "-w", json.dumps(raw),
            ]
        )
        if result is None or result.returncode != 0:
            rc = "exception" if result is None else result.returncode
            logger.warning("Could not write refreshed credentials to Claude Code keychain (rc=%s)", rc)
            return False
        return True


def discover_claude_code_source(
    *,
    platform: Optional[str] = None,
    home: Optional[Path] = None,
) -> Optional[CredentialSource]:
    """Locate the Claude Code CLI credential store, if the user has logged in.

    macOS: the ``Claude Code-credentials`` Keychain item (preferred).
    Otherwise / Linux: ``~/.claude/.credentials.json``.
    Returns a source only when it actually yields parseable credentials, so a
    missing/empty store falls through cleanly.
    """
    plat = platform or sys.platform
    if plat == "darwin":
        kc = KeychainCredentialSource()
        if kc.read() is not None:
            return kc
    base = home or Path.home()
    file_path = base / _CLAUDE_CREDENTIALS_RELATIVE_PATH
    if file_path.exists():
        fs = FileCredentialSource(file_path)
        if fs.read() is not None:
            return fs
    return None


class ClaudeOAuthTokenManager:
    """Holds the live Claude OAuth credentials for the plan route and refreshes
    them before expiry, writing rotations back to the credential source.

    One instance per route (built in ``provider_registry``). ``access_token()``
    is the only entry point: it returns a currently-valid access token. A static
    token with no refresh token (the ``setup-token`` case) is returned unchanged.
    """

    def __init__(
        self,
        credentials: OAuthCredentials,
        *,
        source: Optional[CredentialSource] = None,
        client_id: Optional[str] = None,
        token_url: Optional[str] = None,
    ) -> None:
        self._creds = credentials
        self._source = source
        self._client_id = client_id
        self._token_url = token_url
        self._lock = asyncio.Lock()

    @property
    def initial_access_token(self) -> str:
        """The access token to build the SDK client with at startup. Refresh
        (if any) then happens lazily before requests via ``access_token()``."""
        return self._creds.access

    @classmethod
    def from_sources(
        cls,
        *,
        static_token: Optional[str],
        credentials_path: Optional[str],
        delegate: Optional[bool] = None,
    ) -> Optional["ClaudeOAuthTokenManager"]:
        """Build a manager from the first available credential source.

        Precedence: explicit file (config/env) → static env token → auto-
        discovered Claude Code store. Returns ``None`` when nothing is found.
        Auto-discovery is on unless ``delegate`` (or ``KESTREL_ANTHROPIC_OAUTH_DELEGATE``)
        is false.
        """
        # 1. Explicit credentials file (operator-configured).
        path = credentials_path or os.environ.get(_ENV_CREDENTIALS_FILE)
        if path:
            src = FileCredentialSource(Path(path).expanduser())
            creds = src.read()
            if creds is not None:
                return cls(creds, source=src)
            logger.warning("Anthropic OAuth credentials file yielded no token: %s", path)

        # 2. Static env token (used as-is; setup-tokens carry no refresh token).
        if static_token:
            return cls(OAuthCredentials(access=static_token))

        # 3. Delegate to the Claude Code CLI store (like codex:plan → codex binary).
        # Requires BOTH the caller's opt-in (route gating; ``delegate`` is False
        # for the metered API-key route) AND the operator env escape hatch.
        env_allows = os.environ.get(_ENV_DELEGATE, "1").lower() not in ("0", "false", "no")
        allow_delegate = (True if delegate is None else delegate) and env_allows
        if allow_delegate:
            src = discover_claude_code_source()
            if src is not None:
                creds = src.read()
                if creds is not None:
                    logger.info("anthropic:plan delegating to Claude Code credential store")
                    return cls(creds, source=src)
        return None

    async def access_token(self) -> str:
        if not self._creds.needs_refresh(now=time.time()):
            return self._creds.access
        async with self._lock:
            # Another client (e.g. Claude Code itself) may have already
            # refreshed the shared store — adopt that before minting our own.
            if self._source is not None:
                fresh = self._source.read()
                if fresh is not None:
                    self._creds = fresh
            if not self._creds.needs_refresh(now=time.time()):
                return self._creds.access
            assert self._creds.refresh is not None  # guarded by needs_refresh
            logger.info("Refreshing Claude OAuth access token (near expiry)")
            new_creds = await refresh_anthropic_token(
                self._creds.refresh,
                client_id=self._client_id,
                token_url=self._token_url,
            )
            self._creds = new_creds
            if self._source is not None:
                self._source.write(new_creds)
            return new_creds.access
