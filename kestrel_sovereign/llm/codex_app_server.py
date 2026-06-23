"""Codex app-server JSON-RPC client.

The OpenAI-sanctioned mechanism for using a ChatGPT subscription from a
third-party harness (Sam Altman, 2026-05-02: *"you can sign in to
openclaw with your chatgpt account now and use your subscription
there"*) is to drive the official ``codex app-server`` binary over a
local stdio JSON-RPC channel. The binary owns OAuth (via
``~/.codex/auth.json``), token refresh, account identification and the
chatgpt.com transport. We identify as ourselves in the handshake
(``clientInfo.name = "kestrel"``); no impersonation.

This is a Python port of the OpenClaw bridge at
``kestrel-claw/extensions/codex/src/app-server/``. Wire shapes were
verified against the real binary (``codex 0.131.0-alpha.9``).

Framing: newline-delimited JSON. Each line is one of:
  * response     ``{"id", "result"|"error"}``           (server→client)
  * notification ``{"method", "params"}``  (no ``id``)  (server→client)
  * request      ``{"id", "method", "params"}``         (server→client → we reply)
Client→server messages use the same shapes; request ids are sequential ints.

Server→client requests come in two flavors:

  * **Approval/elicitation** (``item/.../requestApproval``,
    ``mcpServer/elicitation/request``): we decline deterministically —
    kestrel's own approval queue governs tool execution, not the
    app-server's native one.
  * **Tool execution** (``item/tool/call``): if a per-turn handler is
    registered (the bridge wires kestrel's hook-enforcing executor into
    one), invoke it inline; otherwise reply with a clear failure rather
    than silently declining.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from .cancellation import (
    AuthCancellationToken,
    CancelToken,
    await_or_auth_cancelled,
    raise_if_cancelled,
)

logger = logging.getLogger(__name__)

# Hard floor — matches kestrel-claw ``MIN_CODEX_APP_SERVER_VERSION``. The
# app-server JSON-RPC contract changed in incompatible ways below this.
MIN_CODEX_APP_SERVER_VERSION = "0.125.0"

# Resolution order for the binary. It ships inside the Codex desktop app
# bundle and is deliberately NOT on PATH (reference: codex-cli-path),
# which is why ``which codex`` misleads. An explicit env override wins.
_CODEX_BIN_ENV = "KESTREL_CODEX_APP_SERVER_BIN"
_CODEX_BUNDLE_PATH = "/Applications/Codex.app/Contents/Resources/codex"

# Server→client approval/elicitation requests we answer deterministically
# so a turn can complete (mirrors kestrel-claw's
# ``defaultServerRequestResponse``). Kestrel runs its own approval queue;
# the app-server's native approvals are not the gate we care about.
_DEFAULT_APPROVAL_REPLIES = {
    "item/commandExecution/requestApproval": {"decision": "decline"},
    "item/fileChange/requestApproval": {"decision": "decline"},
    "item/permissions/requestApproval": {"permissions": {}, "scope": "turn"},
    "item/tool/requestUserInput": {"answers": {}},
    "mcpServer/elicitation/request": {"action": "decline"},
}

# Type aliases.
ServerRequestHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class CodexAppServerError(RuntimeError):
    """Any app-server failure: spawn, version, protocol error, caller
    config violation, codex-reported turn failure.

    Catch-all supertype. Callers that need to distinguish a transient
    transport stall (which they should surface without rotating to a
    different LLM provider) from a permanent caller-config error
    (which fallback should handle as usual) should check
    :class:`CodexAppServerTransportError` instead.
    """


class CodexAppServerTransportError(CodexAppServerError):
    """A *transient* failure in the codex app-server transport itself:
    RPC timeout, idle stall, websocket dropped, app-server process gone
    mid-turn. These are not evidence the kestrel route is broken — the
    harness owns the transport. The streaming fallback loops in
    ``llm/streaming.py`` use this discriminator to refuse to rotate
    providers on these errors, matching openclaw commit ``3a64dc7623``
    ("keep turn timeouts inside Codex").
    """


class CodexAppServerConnectionClosed(CodexAppServerTransportError):
    """The app-server process is gone (exited / stdin closed). Modeled
    as a transport error: the connection died, which is what the
    fallback escalation rule cares about.
    """


class CodexAppServerCancelled(CodexAppServerError):
    """Interactive Codex auth/login flow was explicitly cancelled."""


def resolve_codex_binary() -> str:
    """Return the path to the codex binary or raise with a fix hint."""
    override = os.environ.get(_CODEX_BIN_ENV)
    if override:
        if not Path(override).exists():
            raise CodexAppServerError(
                f"{_CODEX_BIN_ENV}={override!r} does not exist."
            )
        return override
    if Path(_CODEX_BUNDLE_PATH).exists():
        return _CODEX_BUNDLE_PATH
    found = shutil.which("codex")
    if found:
        return found
    raise CodexAppServerError(
        "codex binary not found. Install the Codex app/CLI, or set "
        f"{_CODEX_BIN_ENV} to its path. (It ships inside Codex.app and is "
        "intentionally off PATH.)"
    )


def _version_tuple(v: str) -> tuple:
    """``0.131.0-alpha.9`` → comparable key.

    Numeric (major, minor, patch) dominates; a prerelease/build suffix
    ranks *below* the same numeric with no suffix (``-1`` sentinel),
    matching kestrel-claw's ``parseVersionForComparison``.
    """
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(.*)$", v.strip())
    if not m:
        return (0, 0, 0, 0)
    major, minor, patch, suffix = m.groups()
    return (int(major), int(minor), int(patch), 0 if not suffix else -1)


def _parse_user_agent_version(user_agent: str) -> Optional[str]:
    """Extract the leading ``<product>/<version>`` semver."""
    m = re.match(
        r"^[^/]+/(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)",
        user_agent or "",
    )
    return m.group(1) if m else None


def _strip_plugin_and_project_sections(toml_text: str) -> str:
    """Return ``toml_text`` with ``[plugins.*]``, ``[marketplaces.*]``,
    and ``[projects.*]`` sections removed; top-level scalar settings
    (``model``, ``model_reasoning_effort``, ``notify``, …) and other
    non-plugin tables are preserved verbatim.

    Used to seed the isolated kestrel ``CODEX_HOME``'s ``config.toml``
    from the user's real ``~/.codex/config.toml`` while:
    1. dropping the plugins that hang codex's session_loop, and
    2. replacing the trusted-projects list with kestrel's own entry.

    Naive line-based scanner: TOML section headers are at column 0 in
    practice (codex never indents headers), so we walk lines and drop
    everything between a stripped header and the next one. Multi-line
    arrays inside a stripped section are dropped with the section.
    """
    out_lines: list[str] = []
    skip = False
    for line in toml_text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("["):
            inner = stripped.lstrip("[").split("]", 1)[0]
            head = inner.split(".", 1)[0].strip().strip('"').lower()
            skip = head in ("plugins", "marketplaces", "projects")
        if not skip:
            out_lines.append(line)
    body = "".join(out_lines)
    # Trim trailing whitespace so the appended trusted-projects block
    # lands on a clean line boundary.
    return body.rstrip() + ("\n" if body.strip() else "")


class CodexAppServerClient:
    """One managed ``codex app-server`` process + JSON-RPC multiplexer."""

    def __init__(self, binary: Optional[str] = None, *, client_version: str = "0.1.0"):
        self._binary = binary or resolve_codex_binary()
        self._client_version = client_version
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._next_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        # Notification sinks, keyed by threadId. Notifications carry
        # ``params.threadId`` and are routed only to the owning turn so
        # concurrent turns never cross-contaminate. Global events
        # (without threadId) broadcast.
        self._turn_sinks: Dict[Any, "asyncio.Queue[dict]"] = {}
        # Server-request handlers, keyed by ``(method, thread_id_or_None)``.
        # Concurrent turns on different threads can each register a
        # thread-scoped ``item/tool/call`` handler — dispatch tries the
        # thread-scoped registration first (matched on ``params.threadId``)
        # and falls back to the unscoped (None) entry, then defaults.
        self._server_request_handlers: Dict[
            Tuple[str, Optional[str]], ServerRequestHandler
        ] = {}
        # Per-thread count of in-flight server→client requests
        # (``item/tool/call``, sandbox-approval RPCs, token refresh).
        # While one is in flight the app-server is *correctly* silent —
        # it is blocked awaiting OUR reply, not stalled upstream. The
        # idle watchdog (``iter_turn_events``) consults this so a
        # legitimately long tool callback (e.g. a Talon job that runs
        # for minutes, or an approval queued on a human) doesn't trip
        # the 300s local watchdog. Keyed by ``threadId`` (None for
        # global/thread-less requests). See ``iter_turn_events``.
        self._inflight_server_requests: Dict[Optional[str], int] = {}
        # #1581: agent reference for typed-event audit on default
        # declines (``_DEFAULT_APPROVAL_REPLIES`` path). Set via
        # ``attach_audit_agent`` from the adapter's
        # ``_ensure_codex_approval_bridge`` so a re-bind on the
        # adapter side also redirects audit here. ``None`` means
        # no decline event is recorded — default for unattached
        # clients (tests, headless callers).
        self._audit_agent: Any = None
        self._stderr_tail: list[str] = []
        # Resolved CODEX_HOME path used by ``_spawn`` (#1410). Captured at
        # spawn time so accessors (``recent_codex_log``) can query the
        # codex-rs internal log DB at ``<CODEX_HOME>/logs_2.sqlite``
        # without recomputing the path resolution.
        self._codex_home: Optional[Path] = None
        self._closed_error: Optional[BaseException] = None
        self._start_lock = asyncio.Lock()
        self._initialized = False
        self._auth_cancellation_token: Optional[AuthCancellationToken] = None

    # ---------------------------------------------------------------- lifecycle
    def _auth_cancelled(self) -> CodexAppServerCancelled:
        return CodexAppServerCancelled("codex login cancelled")

    async def ensure_started(
        self,
        *,
        cancellation_token: Optional[AuthCancellationToken] = None,
    ) -> None:
        if self._initialized:
            return
        async with self._start_lock:
            if self._initialized:
                return
            previous_token = self._auth_cancellation_token
            self._auth_cancellation_token = cancellation_token
            try:
                await self._spawn()
                await self._handshake(cancellation_token=cancellation_token)
                self._initialized = True
            finally:
                self._auth_cancellation_token = previous_token

    async def _spawn(self) -> None:
        # Isolate ``CODEX_HOME`` to a kestrel-managed directory so codex
        # does not auto-load the user's globally-configured plugins
        # (computer-use, codex_apps, browser-use, …). Without isolation,
        # those plugins are mounted as MCP servers on every thread; some
        # (notably ``codex_apps``) hang on ``status=starting`` and block
        # the session_loop's upstream Responses API call indefinitely.
        # Same mechanism kestrel-claw uses
        # (extensions/codex/src/app-server/auth-bridge.ts —
        # ``withAgentCodexHomeEnvironment``): point CODEX_HOME at an
        # isolated dir, then bridge the user's ChatGPT auth across by
        # symlinking auth.json so the subscription sign-in still works.
        kestrel_codex_home = Path.home() / ".kestrel" / "codex-home"
        kestrel_codex_home.mkdir(parents=True, exist_ok=True)
        # Capture for diagnostic accessors (#1410). Read on error paths
        # to surface the codex-rs internal log alongside our stderr tail.
        self._codex_home = kestrel_codex_home
        # Source the user's REAL codex home from the environment when set
        # (e.g. operator overrode ``CODEX_HOME``), not the default
        # ``~/.codex``. Without this an operator on a non-default codex
        # home would have the app-server start unauthenticated because
        # we'd be linking from an empty ``~/.codex``. Codex review
        # #1394 P2.
        user_codex_home_env = os.environ.get("CODEX_HOME", "").strip()
        user_codex_home = (
            Path(user_codex_home_env) if user_codex_home_env
            else Path.home() / ".codex"
        )
        # Bridge auth.json + installation_id from the user's real codex
        # home so codex sees the same ChatGPT identity it would see when
        # run normally. Symlink rather than copy so token refreshes
        # propagate to the source.
        # Re-point the symlink every spawn — a previous bridge created
        # for a different ``CODEX_HOME`` would otherwise stale-point at
        # the old source and authenticate as the wrong account. Codex
        # review #1394 P2.
        # Guard: if the operator points ``CODEX_HOME`` at our managed
        # dir, source == dest and the unlink+symlink dance below would
        # destroy the real auth file. Skip bridging entirely in that
        # case — the file is already where codex expects it. Codex
        # review #1394 P2.
        same_home = (
            kestrel_codex_home.resolve() == user_codex_home.resolve()
        )
        if not same_home:
            for fname in ("auth.json", "installation_id"):
                user_file = user_codex_home / fname
                bridged_file = kestrel_codex_home / fname
                if not user_file.exists():
                    # Source absent: clear any stale bridge from a previous
                    # spawn so the API-key gating below sees the current
                    # state (not stale OAuth). Codex review #1394 P2.
                    if bridged_file.is_symlink() or bridged_file.exists():
                        try:
                            bridged_file.unlink()
                        except OSError:
                            pass
                    continue
                try:
                    if bridged_file.is_symlink() or bridged_file.exists():
                        if (
                            bridged_file.is_symlink()
                            and bridged_file.readlink() == user_file
                        ):
                            continue
                        bridged_file.unlink()
                    bridged_file.symlink_to(user_file)
                except OSError:
                    pass
        # Minimal config.toml — trust the kestrel workspace + cwd so
        # codex doesn't refuse to run, but ship NO ``[plugins.*]`` blocks
        # so the user's globally-enabled MCP plugins
        # (computer-use, codex_apps, etc.) stay un-mounted in our
        # sessions. Kestrel runs its own computer-use feature; codex's
        # bundled one would be a duplicate-mount anyway.
        # Rewrite every spawn so a cwd change (different KESTREL_CODEX_CWD
        # or process cwd) is reflected in the trusted-projects list —
        # otherwise codex rejects/blocks the workspace until the user
        # manually deletes the stale config. Codex review #1394 P2.
        cwd_for_codex = os.environ.get("KESTREL_CODEX_CWD") or str(Path.cwd())
        # Escape TOML basic-string special characters in the cwd before
        # interpolating into the ``[projects."..."]`` table key. A
        # workspace with ``"`` or ``\`` in the path would otherwise
        # produce invalid TOML and codex would either refuse to parse
        # the config or trust a different project than intended. Codex
        # review #1394 P2.
        cwd_escaped = cwd_for_codex.replace("\\", "\\\\").replace('"', '\\"')
        bridged_config = kestrel_codex_home / "config.toml"
        # Build the isolated config by COPYING the user's
        # ~/.codex/config.toml with ``[plugins.*]``, ``[marketplaces.*]``,
        # and ``[projects.*]`` sections stripped, then appending our
        # own trusted-project block. This preserves the user's
        # ``model``, ``model_reasoning_effort``, etc. defaults so the
        # adapter's ``_model_param`` ``auto``/``default`` path still
        # gets a configured default to fall through to — otherwise
        # codex defaults to its own built-in choice and the user's
        # configured default is lost. Codex review #1394 P2.
        trusted_block = (
            f'\n[projects."{cwd_escaped}"]\n'
            f'trust_level = "trusted"\n'
        )
        user_config_path = user_codex_home / "config.toml"
        sanitized_body = ""
        if not same_home and user_config_path.exists():
            try:
                sanitized_body = _strip_plugin_and_project_sections(
                    user_config_path.read_text(encoding="utf-8")
                )
            except OSError:
                sanitized_body = ""
        try:
            bridged_config.write_text(sanitized_body + trusted_block)
        except OSError:
            pass
        # Strip OPENAI_API_KEY / CODEX_API_KEY from the spawn env ONLY
        # when ``_load_chatgpt_login_params()`` returns usable tokens —
        # not just on file presence. A stale/corrupt ``auth.json`` with
        # no ``access_token`` would otherwise leave codex with neither
        # OAuth (login RPC returns None) nor API-key (we stripped them)
        # credentials. Matches claw's
        # ``shouldClearOpenAiApiKeyForCodexAuthProfile`` gating in
        # ``auth-bridge.ts``. Codex review #1394 P2.
        usable_oauth_login = self._load_chatgpt_login_params() is not None
        spawn_env = {**os.environ, "CODEX_HOME": str(kestrel_codex_home)}
        if usable_oauth_login:
            spawn_env.pop("OPENAI_API_KEY", None)
            spawn_env.pop("CODEX_API_KEY", None)
        try:
            # ``--disable apps``: codex's bundled ``codex_apps`` MCP
            # is built-in and tries to fetch ChatGPT app metadata on
            # every thread. For our session it consistently exceeds
            # the 30s MCP startup_timeout, leaving session_loop
            # blocked. Kestrel has its own app/tool surface — we
            # don't need codex's bundled directory.
            #
            # ``limit=16 MiB``: codex emits JSON-RPC frames as single
            # newline-terminated lines on stdout. Streaming events for a
            # turn carrying our typical ~20K-input-token prompt routinely
            # exceed asyncio's default 64 KiB StreamReader buffer (e.g.
            # a single Item-finished event echoing the full assistant
            # text or a snapshot containing the cumulative reasoning
            # trace). The default raises ``ValueError('Separator is
            # found, but chunk is longer than limit')`` from
            # ``StreamReader.__anext__``, the read loop dies before any
            # frame is parsed, and ``_read_loop``'s finally reports
            # ``codex app-server exited (rc=None)`` — silently masking
            # a 64 KiB cap as a process death. 16 MiB matches the
            # codex-rs internal frame budget; if anything ever exceeds
            # it we want a real protocol error, not a quiet crash. See
            # #1438.
            self._proc = await asyncio.create_subprocess_exec(
                self._binary, "--disable", "apps",
                "app-server", "--listen", "stdio://",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=spawn_env,
                limit=16 * 1024 * 1024,
            )
        except OSError as e:
            raise CodexAppServerError(f"Failed to spawn {self._binary}: {e}") from e
        # Clear the prior-instance closed error so callers don't see
        # the OLD process's exit reported on the NEW process's first
        # request. Paired with the ``_initialized = False`` reset at
        # the end of ``_read_loop`` — together they enable recovery
        # from an involuntary app-server exit without restarting
        # kestrel.
        self._closed_error = None
        self._stderr_tail = []
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _handshake(
        self,
        *,
        cancellation_token: Optional[AuthCancellationToken] = None,
    ) -> None:
        result = await self._request_unguarded(
            "initialize",
            {
                "clientInfo": {
                    "name": "kestrel",
                    "title": "Kestrel",
                    "version": self._client_version,
                },
                "capabilities": {"experimentalApi": True},
            },
            timeout=30,
            cancellation_token=cancellation_token,
        )
        ua = (result or {}).get("userAgent", "")
        detected = _parse_user_agent_version(ua)
        if detected and _version_tuple(detected) < _version_tuple(
            MIN_CODEX_APP_SERVER_VERSION
        ):
            await self.aclose()
            raise CodexAppServerError(
                f"codex app-server {MIN_CODEX_APP_SERVER_VERSION}+ required, "
                f"detected {detected}. Update the Codex app/CLI."
            )
        self.notify("initialized")
        # Match kestrel-claw's auth-bridge.ts: after initialize +
        # initialized notification, explicitly drive
        # ``account/login/start`` with the user's ChatGPT tokens.
        # Without this the app-server's session_loop hangs before
        # making the upstream Responses API call — auth.json on disk
        # in CODEX_HOME is consulted by ``codex exec`` but the
        # app-server path requires an explicit login RPC to wire the
        # account into the session (claw's
        # ``applyCodexAppServerAuthProfile`` in
        # ``extensions/codex/src/app-server/shared-client.ts``).
        login_params = self._load_chatgpt_login_params()
        if login_params is not None:
            try:
                await self._request_unguarded(
                    "account/login/start",
                    login_params,
                    timeout=30,
                    cancellation_token=cancellation_token,
                )
            except CodexAppServerCancelled:
                raise
            except CodexAppServerError as e:
                logger.warning(
                    "codex account/login/start failed (continuing — auth "
                    "may still resolve from auth.json): %s", e,
                )

    def _load_chatgpt_login_params(self) -> Optional[dict]:
        """Read the user's codex auth.json and shape it as the
        ``account/login/start`` RPC param for the ChatGPT subscription
        path. Returns ``None`` when no chatgpt OAuth tokens are present
        (e.g. an API-key-only setup) — the caller proceeds without the
        login RPC in that case."""
        # Source from the user's real codex home — ``CODEX_HOME`` env var
        # overrides ``~/.codex`` when set. Matches the same resolution in
        # ``_spawn``; without this, an operator on a non-default codex
        # home gets an unauthenticated app-server.
        user_codex_home_env = os.environ.get("CODEX_HOME", "").strip()
        user_codex_home = (
            Path(user_codex_home_env) if user_codex_home_env
            else Path.home() / ".codex"
        )
        auth_path = user_codex_home / "auth.json"
        if not auth_path.exists():
            return None
        try:
            import json
            data = json.loads(auth_path.read_text())
        except (OSError, ValueError):
            return None
        # Respect the operator's recorded auth mode. An auth.json with
        # ``auth_mode: "apikey"`` means the user explicitly chose the
        # API-key lane — old ChatGPT tokens may still be present in
        # the file from a prior session but they should not be used.
        # Without this gate, my code would route through OAuth with
        # stale tokens AND strip the API-key env vars in _spawn,
        # silently breaking the operator's selected lane. Codex
        # review #1394 P2.
        auth_mode = (data.get("auth_mode") or "").strip().lower()
        if auth_mode and auth_mode != "chatgpt":
            return None
        tokens = data.get("tokens") or {}
        access = (tokens.get("access_token") or "").strip()
        account_id = (tokens.get("account_id") or "").strip()
        if not access:
            return None
        # Plan type lives inside the id_token JWT claims under the
        # OpenAI ``auth.chatgpt_plan_type`` namespace. Decode the
        # payload portion only (signature unchecked — we trust the
        # local file). Falls back to ``None`` when absent; codex
        # accepts a null plan_type.
        plan_type: Optional[str] = None
        id_token = (tokens.get("id_token") or "").strip()
        if id_token:
            try:
                import base64
                parts = id_token.split(".")
                if len(parts) >= 2:
                    payload_b64 = parts[1]
                    payload_b64 += "=" * (-len(payload_b64) % 4)
                    claims = json.loads(base64.urlsafe_b64decode(payload_b64))
                    auth_claim = claims.get(
                        "https://api.openai.com/auth"
                    ) or {}
                    plan_type = (
                        auth_claim.get("chatgpt_plan_type")
                        or claims.get("chatgpt_plan_type")
                        or None
                    )
            except (ValueError, OSError, KeyError):
                plan_type = None
        # ``chatgptAccountId`` is REQUIRED by the schema for
        # ``account/login/start`` (LoginAccountParams::chatgptAuthTokens).
        # If auth.json is missing it, treat this as unusable OAuth
        # state — falling through here without it would cause codex
        # to reject login AND leave us having stripped the API-key
        # env-var fallback. Codex review #1394 P2.
        if not account_id:
            return None
        params: Dict[str, Any] = {
            "type": "chatgptAuthTokens",
            "accessToken": access,
            "chatgptAccountId": account_id,
        }
        if plan_type:
            params["chatgptPlanType"] = plan_type
        return params

    async def _refresh_chatgpt_tokens(
        self,
        *,
        cancellation_token: Optional[AuthCancellationToken] = None,
    ) -> Optional[dict]:
        """Drive the OAuth refresh-token flow against the OpenAI auth
        endpoint, persist the new tokens back to ``auth.json``, and
        return the same shape as :meth:`_load_chatgpt_login_params`.

        Codex CLI runs this same refresh on its own schedule when it
        is the active process. In a long-running kestrel host that is
        the only app-server client, nobody is refreshing for us — so
        re-reading ``auth.json`` from
        ``account/chatgptAuthTokens/refresh`` would return the same
        expired token and codex would 401 again. This method runs the
        OAuth grant_type=refresh_token exchange so we can hand codex
        a real fresh token. Codex review #1394 P1.

        Returns ``None`` when there are no usable OAuth credentials on
        disk or the refresh exchange fails — the caller should error
        out cleanly in that case rather than send an empty token.
        """
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled(self._auth_cancelled)
        user_codex_home_env = os.environ.get("CODEX_HOME", "").strip()
        user_codex_home = (
            Path(user_codex_home_env) if user_codex_home_env
            else Path.home() / ".codex"
        )
        auth_path = user_codex_home / "auth.json"
        if not auth_path.exists():
            return None
        try:
            import json
            data = json.loads(auth_path.read_text())
        except (OSError, ValueError):
            return None
        tokens = data.get("tokens") or {}
        refresh_token = (tokens.get("refresh_token") or "").strip()
        if not refresh_token:
            return None
        token_url = os.environ.get(
            "CODEX_REFRESH_TOKEN_URL_OVERRIDE",
            "https://auth.openai.com/oauth/token",
        )
        # Codex's own client_id — visible as an unredacted string in
        # the codex binary. Required by OpenAI's OAuth endpoint to
        # accept the refresh; using kestrel's own client_id would be
        # rejected as that client isn't registered with OpenAI auth.
        client_id = "app_EMoamEEZ73f0CkXaXp7hrann"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=20) as client:
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled(self._auth_cancelled)
                # OAuth token endpoints (including OpenAI's) require
                # ``application/x-www-form-urlencoded``, not JSON. Use
                # httpx's ``data=`` so the body is form-encoded with
                # the right Content-Type. Codex review #1394 P1.
                resp = await await_or_auth_cancelled(
                    client.post(
                        token_url,
                        data={
                            "grant_type": "refresh_token",
                            "refresh_token": refresh_token,
                            "client_id": client_id,
                            "scope": "openid profile email offline_access",
                        },
                    ),
                    cancellation_token,
                    self._auth_cancelled,
                )
                if resp.status_code != 200:
                    logger.warning(
                        "codex token refresh failed: %s %s",
                        resp.status_code, resp.text[:200],
                    )
                    return None
                body = resp.json()
        except (httpx.HTTPError, ValueError, OSError) as e:
            logger.warning("codex token refresh exception: %s", e)
            return None
        new_access = (body.get("access_token") or "").strip()
        new_id_token = (body.get("id_token") or "").strip()
        new_refresh = (body.get("refresh_token") or refresh_token).strip()
        if not new_access:
            return None
        # Persist the rotated tokens back to auth.json so future
        # spawns (including codex's own CLI) see the same state. Mirror
        # the on-disk shape codex writes: top-level ``tokens`` object
        # with ``access_token`` / ``id_token`` / ``refresh_token`` /
        # ``account_id`` (the account_id stays put — refresh doesn't
        # rotate the workspace identifier).
        tokens["access_token"] = new_access
        if new_id_token:
            tokens["id_token"] = new_id_token
        tokens["refresh_token"] = new_refresh
        data["tokens"] = tokens
        data["last_refresh"] = (
            __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat()
        )
        try:
            auth_path.write_text(json.dumps(data))
        except OSError as e:
            logger.warning(
                "codex token refresh: wrote new token in-memory but "
                "failed to persist to %s: %s — next host restart will "
                "still see the expired token.", auth_path, e,
            )
        # Return the in-memory refreshed state directly. Re-reading
        # auth.json here would replay the OLD token whenever
        # persistence failed (read-only mount, permissions, etc.),
        # causing codex to 401 again immediately. Derive the chatgpt
        # plan type from the new id_token claims using the same logic
        # as ``_load_chatgpt_login_params``. Codex review #1394 P2.
        account_id = (tokens.get("account_id") or "").strip()
        if not account_id:
            return None
        plan_type: Optional[str] = None
        if new_id_token:
            try:
                import base64
                parts = new_id_token.split(".")
                if len(parts) >= 2:
                    payload_b64 = parts[1]
                    payload_b64 += "=" * (-len(payload_b64) % 4)
                    claims = json.loads(base64.urlsafe_b64decode(payload_b64))
                    auth_claim = claims.get(
                        "https://api.openai.com/auth"
                    ) or {}
                    plan_type = (
                        auth_claim.get("chatgpt_plan_type")
                        or claims.get("chatgpt_plan_type")
                        or None
                    )
            except (ValueError, OSError, KeyError):
                plan_type = None
        out: Dict[str, Any] = {
            "type": "chatgptAuthTokens",
            "accessToken": new_access,
            "chatgptAccountId": account_id,
        }
        if plan_type:
            out["chatgptPlanType"] = plan_type
        return out

    async def aclose(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (asyncio.TimeoutError, Exception):
            try:
                proc.kill()
            except Exception:
                pass
        for t in (self._reader_task, self._stderr_task):
            if t:
                t.cancel()
        self._fail_all(CodexAppServerConnectionClosed("codex app-server closed"))
        self._initialized = False
        self._proc = None

    # ----------------------------------------------------------------- io loops
    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        try:
            async for raw in self._proc.stderr:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    self._stderr_tail.append(line)
                    del self._stderr_tail[:-40]
                    logger.debug("codex app-server stderr: %s", line[:500])
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            async for raw in self._proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    import json

                    msg = json.loads(line)
                except ValueError:
                    logger.warning("codex app-server: non-JSON line: %s", line[:200])
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            pass
        except ValueError as e:
            # ``StreamReader.__anext__`` raises ValueError when a single
            # line exceeds the configured ``limit=`` on the subprocess
            # spawn (asyncio's "Separator is found, but chunk is longer
            # than limit" — see #1438). Pre-#1438 this fell through to
            # the finally and the call site saw a generic "app-server
            # exited (rc=None)" with no clue why. Surface the buffer
            # cap explicitly so a future regression points at the spawn
            # ``limit=`` argument instead of looking like an upstream
            # crash. Server-log only; the call-site exception text stays
            # neutral to avoid the cross-session leak boundary set in
            # #1410/#1412.
            logger.error(
                "codex app-server read_loop hit asyncio StreamReader limit "
                "(spawn `limit=` argument too low for current frame): %s",
                e,
            )
        except Exception as e:
            # Don't let an unexpected reader exception look like a clean
            # exit — log the actual error so it survives the
            # ``CodexAppServerConnectionClosed`` thrown in the finally
            # block. The pre-#1438 implementation buried these inside
            # asyncio's "Task exception was never retrieved" warning and
            # operators chasing 500s had nothing to grep for.
            logger.error(
                "codex app-server _read_loop crashed: %s: %s",
                type(e).__name__, e,
                exc_info=True,
            )
        finally:
            # Wait briefly for the OS to report the real returncode.
            # Without this, ``returncode`` would be None when stdout
            # closes BEFORE the process is reaped — the error message
            # would hide whether codex was killed (signal) vs exited
            # normally vs crashed. Critical for diagnosing
            # CodexAppServerConnectionClosed reports (#1399).
            rc_value: Any = "?"
            if self._proc is not None:
                try:
                    rc_value = await asyncio.wait_for(
                        self._proc.wait(), timeout=1.0
                    )
                except (asyncio.TimeoutError, ProcessLookupError):
                    rc_value = self._proc.returncode
            # Diagnostic tail goes to server logs only (#1412). Mirrors
            # the leak boundary established for the idle-timeout path in
            # #1410: ``CodexAppServerConnectionClosed`` propagates to
            # chat callers via ``endpoints/agent.py`` and
            # ``_fail_all`` -> pending futures, so the stderr buffer can
            # carry content from prior turns / other agents sharing
            # CODEX_HOME and must not leak into the user-visible
            # exception text.
            tail = "\n".join(self._stderr_tail[-10:])
            if tail:
                logger.error(
                    "codex app-server exit stderr tail (rc=%s):\n%s",
                    rc_value, tail,
                )
            self._fail_all(
                CodexAppServerConnectionClosed(
                    f"codex app-server exited (rc={rc_value})"
                )
            )
            # Reset connection state so the NEXT request can spawn a
            # fresh process. Without this, an unexpected app-server
            # exit (e.g. duplicate-tool-handler panic, segfault,
            # OOM-kill) left ``_initialized`` stuck at True with a
            # dead ``_proc``; ``ensure_started`` short-circuited and
            # every subsequent request raised CONNECTION_CLOSED until
            # kestrel itself was restarted. ``aclose()`` does this
            # same reset for the explicit-shutdown path; mirror it for
            # the involuntary-exit path.
            self._initialized = False
            self._proc = None

    def _dispatch(self, msg: dict) -> None:
        mid = msg.get("id")
        method = msg.get("method")
        if mid is not None and method is None:
            # Response to one of our requests.
            fut = self._pending.pop(mid, None)
            if fut and not fut.done():
                if "error" in msg and msg["error"] is not None:
                    err = msg["error"]
                    fut.set_exception(
                        CodexAppServerError(
                            f"{err.get('message','rpc error')}"
                            f" (code={err.get('code')})"
                        )
                    )
                else:
                    fut.set_result(msg.get("result"))
            return
        if mid is not None and method is not None:
            # Server→client request. Dispatch in a task so the read loop
            # keeps pumping concurrent notifications/requests while a
            # long-running tool handler is running.
            asyncio.create_task(self._handle_server_request(mid, method, msg.get("params") or {}))
            return
        # Notification — route to the OWNING turn by threadId so
        # concurrent turns never cross-contaminate. Thread-less global
        # events broadcast (iter_turn_events ignores methods it doesn't
        # act on).
        params = msg.get("params") or {}
        tid = params.get("threadId")
        if tid is not None:
            q = self._turn_sinks.get(tid)
            if q is not None:
                q.put_nowait(msg)
            # else: belongs to a turn this client isn't tracking — drop.
        else:
            for q in list(self._turn_sinks.values()):
                q.put_nowait(msg)

    async def _handle_server_request(
        self,
        mid: Any,
        method: str,
        params: dict,
        *,
        cancellation_token: Optional[AuthCancellationToken] = None,
    ) -> None:
        tid = (params or {}).get("threadId")
        if cancellation_token is None:
            cancellation_token = getattr(self, "_auth_cancellation_token", None)
        # Mark this thread as awaiting our reply for the whole lifetime
        # of the handler. The idle watchdog treats a positive count as
        # "not an upstream stall" and keeps waiting rather than tripping
        # at 300s. Decremented in the ``finally`` so a raised handler,
        # an unknown-method default, or a send failure all release it.
        self._inflight_server_requests[tid] = (
            self._inflight_server_requests.get(tid, 0) + 1
        )
        try:
            # Prefer a handler scoped to the owning thread; fall back to
            # a global (None) registration; then to safe defaults.
            handler = self._server_request_handlers.get((method, tid))
            if handler is None:
                handler = self._server_request_handlers.get((method, None))
            if handler is not None:
                result = await handler(params)
            elif method == "account/chatgptAuthTokens/refresh":
                # Codex enters external-token mode when we drove
                # ``account/login/start`` with ``type=chatgptAuthTokens``.
                # Once the access token expires it asks us (the client)
                # for fresh tokens via this server→client RPC. Drive
                # the actual OAuth refresh against OpenAI's token
                # endpoint and persist the rotated tokens back to
                # ``auth.json`` — otherwise long-running sessions would
                # 401 again immediately because nothing else is
                # refreshing the file. Codex review #1394 P1.
                refreshed = await self._refresh_chatgpt_tokens(
                    cancellation_token=cancellation_token
                )
                if refreshed is not None:
                    result = {
                        "accessToken": refreshed["accessToken"],
                        "chatgptAccountId": refreshed.get(
                            "chatgptAccountId", ""
                        ),
                        "chatgptPlanType": refreshed.get("chatgptPlanType"),
                    }
                else:
                    # Refresh failed — surface a structured error so
                    # codex stops and a higher layer can prompt the
                    # operator to re-authenticate, rather than
                    # silently replaying an expired token.
                    self._send({
                        "id": mid,
                        "error": {
                            "code": -32603,
                            "message": (
                                "kestrel could not refresh chatgpt auth "
                                "tokens (run `codex login` to "
                                "re-authenticate)"
                            ),
                        },
                    })
                    return
            elif method in _DEFAULT_APPROVAL_REPLIES:
                result = _DEFAULT_APPROVAL_REPLIES[method]
                # #1581: surface the auto-default decline as a typed
                # event in the agent's next-turn operational state
                # block. Without this, declines via the hardcoded
                # default table (the elicitation/permissions/userInput
                # RPCs that #1575's bridge intentionally doesn't cover)
                # are completely invisible to the agent. Best-effort —
                # never breaks dispatch on an audit-store failure.
                #
                # Two reply shapes carry a decline today:
                #   {"decision": "decline"}  — commandExecution, fileChange
                #   {"action":   "decline"}  — mcpServer/elicitation/request
                # Codex review #1581 round 1 caught that the second
                # shape was being silently dropped from the audit.
                if isinstance(result, dict):
                    decline_value = (
                        result.get("decision")
                        or result.get("action")
                    )
                    if str(decline_value or "").lower() == "decline":
                        await self._record_default_decline(method, params or {})
            elif method == "item/tool/call":
                # No bridge registered (e.g. text-only turn). Reply with
                # an explicit failure rather than silently saying ok.
                result = {
                    "contentItems": [{
                        "type": "inputText",
                        "text": "Kestrel did not register an item/tool/call handler for this turn.",
                    }],
                    "success": False,
                }
            else:
                result = {}
            self._send({"id": mid, "result": result})
        except CodexAppServerCancelled:
            raise
        except Exception as e:
            logger.warning("codex server-request %s failed: %s", method, e)
            try:
                self._send({"id": mid, "error": {"message": str(e)}})
            except Exception:
                pass
        finally:
            n = self._inflight_server_requests.get(tid, 0) - 1
            if n > 0:
                self._inflight_server_requests[tid] = n
            else:
                # Drop the key at zero so the map doesn't accumulate one
                # entry per finished thread over a long-lived connection.
                self._inflight_server_requests.pop(tid, None)
            # Wake any idle watchdog so the app-server gets a fresh full
            # idle budget for its NEXT event now that we've replied — a
            # callback that finished late in its window must not raise at
            # the next boundary (see ``iter_turn_events``). Thread-scoped
            # request → that turn's sink; thread-less (approvals omit
            # threadId) → broadcast to every turn, same as notifications.
            marker = {"__inflight_done__": True}
            if tid is not None:
                q = self._turn_sinks.get(tid)
                if q is not None:
                    q.put_nowait(marker)
            else:
                for q in list(self._turn_sinks.values()):
                    q.put_nowait(marker)

    def attach_audit_agent(self, agent: Any) -> None:
        """Bind the agent so default-decline RPCs (the
        ``_DEFAULT_APPROVAL_REPLIES`` path) can write typed events
        into the agent's audit store (#1581). Idempotent; passing
        ``None`` detaches.

        Called from the adapter's
        ``_ensure_codex_approval_bridge`` so a rebind on the adapter
        side also redirects audit here.
        """
        self._audit_agent = agent

    async def _record_default_decline(
        self, method: str, params: Dict[str, Any],
    ) -> None:
        """Persist an auto-default decline as a typed event for the
        attached agent's next-turn operational state block. No-op
        when no agent is attached or the audit store isn't reachable
        (tests, headless callers, transient DB failure)."""
        # getattr — older fixtures construct the client without going
        # through __init__, so the attr may not exist.
        agent = getattr(self, "_audit_agent", None)
        if agent is None:
            return
        try:
            from kestrel_sovereign.features.storage_access import (
                resolve_feature_database,
            )
            db = resolve_feature_database(agent)
            if db is None:
                return
            from kestrel_sovereign.llm.codex_decline_events import (
                ensure_codex_decline_events_table, record_decline,
            )
            await ensure_codex_decline_events_table(db)
            agent_id = (
                getattr(agent, "did", None)
                or getattr(agent, "_agent_name", None)
                or "anonymous"
            )
            # Surface SOMETHING for ``tool`` so the operational block
            # can render it. The default-decline RPCs (elicitation /
            # permissions / requestUserInput) don't carry a single
            # canonical "tool" field, so use the method as a label.
            await record_decline(
                db,
                agent_id=str(agent_id),
                request=method,
                tool=method.rsplit("/", 1)[0],
                reason="auto_default",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "codex_decline_events: default-decline record failed "
                "(%s) — %s", method, exc,
            )

    def register_server_request_handler(
        self, method: str, handler: ServerRequestHandler,
        *, thread_id: Optional[str] = None,
    ) -> Callable[[], None]:
        """Register an async handler for a server→client request method.

        Scope the registration to ``thread_id`` when the handler is
        meaningful only for one turn (the codex tool-execution bridge
        does this — concurrent turns each get their own
        ``item/tool/call`` handler keyed by their thread). An unscoped
        (``thread_id=None``) registration acts as a fallback for any
        thread without a specific handler.

        Returns an ``unregister`` callable; safe to call once.
        """
        key = (method, thread_id)
        self._server_request_handlers[key] = handler

        def _unregister() -> None:
            if self._server_request_handlers.get(key) is handler:
                self._server_request_handlers.pop(key, None)

        return _unregister

    def _fail_all(self, exc: BaseException) -> None:
        self._closed_error = exc
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()
        for q in self._turn_sinks.values():
            q.put_nowait({"__closed__": True})

    # ------------------------------------------------------------------- rpc io
    def _send(self, obj: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.is_closing():
            raise CodexAppServerConnectionClosed("codex app-server stdin closed")
        import json

        proc.stdin.write((json.dumps(obj) + "\n").encode())

    async def request(
        self, method: str, params: Optional[dict] = None, *, timeout: float = 120
    ) -> Any:
        await_started = method != "initialize"
        if await_started:
            await self.ensure_started()
        if self._closed_error and method != "initialize":
            raise self._closed_error
        return await self._request_unguarded(method, params, timeout=timeout)

    async def _request_unguarded(
        self,
        method: str,
        params: Optional[dict] = None,
        *,
        timeout: float = 120,
        cancellation_token: Optional[AuthCancellationToken] = None,
    ) -> Any:
        """Send a request without going through ``ensure_started``.
        For use from inside ``_handshake`` where the start lock is
        already held — calling ``request`` from there would deadlock.
        """
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled(self._auth_cancelled)
        mid = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[mid] = fut
        self._send({"id": mid, "method": method, "params": params or {}})
        try:
            return await asyncio.wait_for(
                await_or_auth_cancelled(
                    fut,
                    cancellation_token,
                    self._auth_cancelled,
                ),
                timeout=timeout,
            )
        except CodexAppServerCancelled:
            self._pending.pop(mid, None)
            raise
        except asyncio.TimeoutError as e:
            self._pending.pop(mid, None)
            raise CodexAppServerTransportError(
                f"{method} timed out after {timeout}s"
            ) from e
        except asyncio.CancelledError:
            # Ctrl-C / outer task cancel (including ``asyncio.timeout``
            # firing on the agent turn) during a pending RPC. Drop the
            # mid from ``_pending`` so a late-arriving response from
            # the app-server doesn't try to resolve a dead future and
            # so the dict doesn't grow for the life of the process.
            # Re-raise the original ``CancelledError`` to preserve
            # cooperative cancellation — converting to a typed error
            # here would prevent ``asyncio.timeout`` from rewriting
            # the cancel into ``TimeoutError``. The user-facing
            # "openai:plan login cancelled" message is the outer
            # agent-error formatter's job, not this layer's. See #1421.
            self._pending.pop(mid, None)
            raise

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        self._send({"method": method, "params": params or {}})

    # ------------------------------------------------------- turn event streaming
    def open_turn_sink(self, key: Any) -> "asyncio.Queue[dict]":
        q: "asyncio.Queue[dict]" = asyncio.Queue()
        self._turn_sinks[key] = q
        return q

    def close_turn_sink(self, key: Any) -> None:
        self._turn_sinks.pop(key, None)

    # ------------------------------------------------------- diagnostics (#1410)
    def recent_stderr(self, n: int = 10) -> list[str]:
        """Return the last ``n`` lines of captured codex-rs stderr.

        Drained live into a 40-line ring buffer by ``_drain_stderr``;
        this accessor exposes a snapshot for error-path callers (e.g.
        the idle-timeout branch of ``iter_turn_events``). Returns an
        empty list when nothing has been captured.
        """
        if not self._stderr_tail:
            return []
        return list(self._stderr_tail[-n:])

    def recent_codex_log(self, n: int = 30) -> list[str]:
        """Tail the last ``n`` rows of codex-rs's internal log DB (#1410).

        codex-rs writes structured logs to ``<CODEX_HOME>/logs_2.sqlite``
        via sqlx — schema: ``logs(id, ts, ts_nanos, level, target,
        feedback_log_body, module_path, file, line, thread_id)``. On
        any ``CodexAppServerError`` this is the most reliable place to
        find the codex-side root cause (upstream RPC error, auth
        refresh failure, websocket close, etc.) — our stderr tail only
        sees what codex-rs explicitly prints, not what it logs.

        Defensive on every failure mode: missing CODEX_HOME, missing
        DB file, schema drift, locked DB, oversize DB, anything else
        — returns ``[]`` rather than raising. This is an error-path
        helper; it must never compound the failure it's reporting on.
        """
        if not self._codex_home:
            return []
        db_path = self._codex_home / "logs_2.sqlite"
        if not db_path.is_file():
            return []
        try:
            import sqlite3
            # Open read-only with a 1s busy timeout so we never block the
            # error path on a long-held codex-rs writer lock. URI mode
            # gives us ``mode=ro`` which won't create the DB if missing.
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, timeout=1.0,
            )
            try:
                cur = conn.execute(
                    "SELECT ts, level, target, feedback_log_body "
                    "FROM logs ORDER BY id DESC LIMIT ?",
                    (max(1, n),),
                )
                rows = cur.fetchall()
            finally:
                conn.close()
        except Exception as e:
            logger.debug("recent_codex_log query failed: %s", e)
            return []
        # Restore chronological order (oldest → newest) for human
        # readability and format compactly. Body may be NULL.
        lines: list[str] = []
        for ts, level, target, body in reversed(rows):
            piece = (body or "").strip().replace("\n", " ")
            lines.append(f"[{ts} {level} {target}] {piece}"[:500])
        return lines

    async def iter_turn_events(
        self,
        sink: "asyncio.Queue[dict]",
        *,
        idle_timeout: float = 300,
        thread_id: Optional[str] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> "asyncio.AsyncIterator[dict]":
        """Yield notifications until ``turn/completed`` / failure / close.

        ``idle_timeout`` (default 300s) bounds how long we wait for the
        app-server to produce the NEXT event. It exists to catch an
        upstream codex / ChatGPT stall — the app-server opened the
        websocket but nothing comes back.

        It must NOT fire while the app-server is legitimately blocked
        awaiting a reply to a server→client request it sent us
        (``item/tool/call``, a sandbox-approval RPC, a token refresh).
        During such a call the app-server is *supposed* to be silent,
        and the callback may run far longer than 300s — e.g. a Talon
        coordinator job that dispatches a feature-workspace build and
        waits minutes for the terminal result, or an approval parked on
        a human. ``thread_id`` lets us consult
        ``_inflight_server_requests``: while a request is in flight for
        this turn's thread — or any thread-less (``None``-keyed) request,
        e.g. a sandbox approval whose params omit ``threadId`` — the
        watchdog re-arms instead of raising, deferring the "tool took
        too long" decision to the app-server's own server-request
        timeout (the correct owner of that bound).
        Callers that don't pass ``thread_id`` keep the legacy
        unconditional-watchdog behavior (tests, thread-less turns).

        On idle-timeout (#1410) codex-rs stderr + the internal sqlite
        log tail are logged at ERROR level for server-side diagnosis,
        but DELIBERATELY KEPT OUT of the raised ``CodexAppServerError``
        message. The exception text propagates to chat callers (e.g.
        ``endpoints/agent.py`` yields ``Error: {e}`` to the streaming
        response), and codex-rs's structured log carries content from
        prior turns / other agents on the same CODEX_HOME — surfacing
        it to whichever user triggers the timeout would be a
        cross-session data leak (codex round-1 P1).
        """
        while True:
            try:
                started = asyncio.get_running_loop().time()
                while True:
                    raise_if_cancelled(cancel_token)
                    elapsed = asyncio.get_running_loop().time() - started
                    remaining = idle_timeout - elapsed
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    try:
                        msg = await asyncio.wait_for(
                            sink.get(), timeout=min(remaining, 0.1)
                        )
                        break
                    except asyncio.TimeoutError:
                        if (
                            asyncio.get_running_loop().time() - started
                            >= idle_timeout
                        ):
                            raise
            except asyncio.TimeoutError as e:
                # Not a stall if the app-server is blocked awaiting our
                # reply to a server-request it sent (long tool callback,
                # approval, token refresh). Re-arm and keep waiting; the
                # app-server's own server-request timeout bounds a truly
                # stuck callback.
                #
                # Count thread-less (``None``-keyed) requests too:
                # sandbox-approval RPCs don't carry ``threadId`` (see
                # ``_make_codex_approval_handler``), so a long human
                # approval would otherwise still trip the watchdog. A
                # global request can't be attributed to a specific turn,
                # so every active turn re-arms on it — re-arming is the
                # safe direction (worst case we wait for codex's own
                # server-request timeout instead of raising falsely).
                if thread_id is not None and (
                    self._inflight_server_requests.get(thread_id, 0) > 0
                    or self._inflight_server_requests.get(None, 0) > 0
                ):
                    logger.debug(
                        "codex idle %ss elapsed but a server-request is "
                        "in flight for thread %s — re-arming watchdog",
                        idle_timeout, thread_id,
                    )
                    continue
                base = f"codex turn idle for {idle_timeout}s with no completion"
                # Log diagnostic tails server-side so operators can see
                # codex-side root cause via ``kestrel logs``; do not
                # attach to the user-visible exception.
                stderr_tail = self.recent_stderr(10)
                if stderr_tail:
                    logger.error(
                        "codex app-server idle-timeout: codex stderr (last lines): %s",
                        " | ".join(stderr_tail),
                    )
                log_tail = self.recent_codex_log(30)
                if log_tail:
                    logger.error(
                        "codex app-server idle-timeout: codex-rs log (last entries): %s",
                        " | ".join(log_tail),
                    )
                raise CodexAppServerTransportError(base) from e
            if msg.get("__inflight_done__"):
                # A server-request just finished and woke us (pushed by
                # ``_handle_server_request``). The ``get()`` returning
                # restarts the idle budget for the next iteration, so the
                # app-server gets a FULL fresh window to emit its next
                # event after our reply — not just whatever was left of
                # the in-flight interval. Without this, a callback that
                # finishes late in a window (t=599 under a 300s budget)
                # would raise almost immediately at the next boundary
                # even though the app-server only just regained control.
                continue
            if msg.get("__closed__"):
                raise self._closed_error or CodexAppServerConnectionClosed(
                    "codex app-server closed mid-turn"
                )
            yield msg
            if msg.get("method") in ("turn/completed", "turn/failed"):
                return
