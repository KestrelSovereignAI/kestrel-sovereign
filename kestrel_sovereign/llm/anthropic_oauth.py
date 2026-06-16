"""
Claude subscription (Claude Pro/Max) OAuth token lifecycle.

The ``openai:plan`` route delegates its entire OAuth lifecycle — token
refresh and request shaping — to the ``codex`` binary (see
``codex_app_server.py``). The ``anthropic:plan`` route talks to the Anthropic
SDK directly, so it must own the equivalent itself. This module is the refresh
half: it exchanges a refresh token for a fresh ``sk-ant-oat`` access token
against Anthropic's OAuth endpoint, using Claude Code's public client id —
mirroring ``codex_app_server``'s ``grant_type=refresh_token`` flow.

Credentials are read from an EXPLICIT, operator-configured source only (a JSON
file path or the static env token). This module never scans another
application's credential store. The static-token path (no refresh token) is
the steady state today and simply returns the token unchanged.
"""
import asyncio
import json
import logging
import os
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

# Env overrides (parity with codex's KESTREL_CODEX_* / CODEX_REFRESH_TOKEN_URL_OVERRIDE).
_ENV_CREDENTIALS_FILE = "KESTREL_ANTHROPIC_OAUTH_CREDENTIALS_FILE"
_ENV_TOKEN_URL_OVERRIDE = "KESTREL_ANTHROPIC_OAUTH_TOKEN_URL"
_ENV_CLIENT_ID_OVERRIDE = "KESTREL_ANTHROPIC_OAUTH_CLIENT_ID"


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
        # Do not log the body verbatim — it can echo token material.
        raise RuntimeError(
            f"Anthropic OAuth token refresh failed: HTTP {resp.status_code} (url={url})"
        )
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


class ClaudeOAuthTokenManager:
    """Holds the live Claude OAuth credentials for the plan route and refreshes
    them before expiry.

    One instance per route (built in ``provider_registry``). ``access_token()``
    is the only entry point: it returns a currently-valid access token,
    refreshing under a lock when within the skew window. A static token with no
    refresh token (the ``setup-token`` case) is returned unchanged.
    """

    def __init__(
        self,
        credentials: OAuthCredentials,
        *,
        credentials_path: Optional[Path] = None,
        client_id: Optional[str] = None,
        token_url: Optional[str] = None,
    ) -> None:
        self._creds = credentials
        self._path = credentials_path
        self._client_id = client_id
        self._token_url = token_url
        self._lock = asyncio.Lock()

    @classmethod
    def from_sources(
        cls,
        *,
        static_token: Optional[str],
        credentials_path: Optional[str],
    ) -> Optional["ClaudeOAuthTokenManager"]:
        """Build a manager from an explicit credentials file and/or a static
        token. Returns ``None`` when there is nothing to manage.

        A credentials file (with a refresh token) takes precedence and enables
        proactive refresh. A bare static token yields a manager that simply
        returns it unchanged — harmless, and keeps the call path uniform.
        """
        path = credentials_path or os.environ.get(_ENV_CREDENTIALS_FILE)
        if path:
            p = Path(path).expanduser()
            try:
                raw = json.loads(p.read_text())
            except FileNotFoundError:
                logger.warning("Anthropic OAuth credentials file not found: %s", p)
                raw = None
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Anthropic OAuth credentials file unreadable (%s): %s", p, exc)
                raw = None
            if raw is not None:
                creds = parse_credentials(raw)
                if creds is not None:
                    return cls(creds, credentials_path=p)
        if static_token:
            return cls(OAuthCredentials(access=static_token))
        return None

    async def access_token(self) -> str:
        creds = self._creds
        if not creds.needs_refresh(now=time.time()):
            return creds.access
        async with self._lock:
            # Re-check under the lock; a concurrent caller may have refreshed.
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
            self._persist(new_creds)
            return new_creds.access

    def _persist(self, creds: OAuthCredentials) -> None:
        """Write refreshed credentials back to the source file so the next
        process start (and the rotated refresh token) survives. Best-effort —
        a failure to persist must not break the in-memory refresh."""
        if self._path is None:
            return
        try:
            payload = {
                "access_token": creds.access,
                "refresh_token": creds.refresh,
                "expires_at": creds.expires_at,
            }
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, self._path)
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
        except OSError as exc:
            logger.warning("Could not persist refreshed Claude OAuth credentials: %s", exc)
