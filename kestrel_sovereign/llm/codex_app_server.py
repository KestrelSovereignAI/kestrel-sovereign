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
    """Any app-server failure: spawn, version, RPC error, exit, timeout."""


class CodexAppServerConnectionClosed(CodexAppServerError):
    """The app-server process is gone (exited / stdin closed)."""


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
        self._stderr_tail: list[str] = []
        self._closed_error: Optional[BaseException] = None
        self._start_lock = asyncio.Lock()
        self._initialized = False

    # ---------------------------------------------------------------- lifecycle
    async def ensure_started(self) -> None:
        if self._initialized:
            return
        async with self._start_lock:
            if self._initialized:
                return
            await self._spawn()
            await self._handshake()
            self._initialized = True

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
        bridged_config = kestrel_codex_home / "config.toml"
        try:
            bridged_config.write_text(
                f'model = "gpt-5.5"\n\n'
                f'[projects."{cwd_for_codex}"]\n'
                f'trust_level = "trusted"\n'
            )
        except OSError:
            pass
        # Strip OPENAI_API_KEY / CODEX_API_KEY from the spawn env ONLY
        # when an OAuth ChatGPT auth was actually bridged across —
        # otherwise codex would route via the chatgpt subscription path
        # and find no tokens. API-key-only deployments (no
        # ``auth.json`` to bridge) need the env vars to remain
        # available so codex can authenticate via API key. Matches
        # claw's ``shouldClearOpenAiApiKeyForCodexAuthProfile`` gating
        # in ``auth-bridge.ts``. Codex review #1394 P2.
        bridged_oauth_present = (kestrel_codex_home / "auth.json").is_symlink() or (
            kestrel_codex_home / "auth.json"
        ).exists()
        spawn_env = {**os.environ, "CODEX_HOME": str(kestrel_codex_home)}
        if bridged_oauth_present:
            spawn_env.pop("OPENAI_API_KEY", None)
            spawn_env.pop("CODEX_API_KEY", None)
        try:
            # ``--disable apps``: codex's bundled ``codex_apps`` MCP
            # is built-in and tries to fetch ChatGPT app metadata on
            # every thread. For our session it consistently exceeds
            # the 30s MCP startup_timeout, leaving session_loop
            # blocked. Kestrel has its own app/tool surface — we
            # don't need codex's bundled directory.
            self._proc = await asyncio.create_subprocess_exec(
                self._binary, "--disable", "apps",
                "app-server", "--listen", "stdio://",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=spawn_env,
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

    async def _handshake(self) -> None:
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
                    "account/login/start", login_params, timeout=30,
                )
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
        params: Dict[str, Any] = {
            "type": "chatgptAuthTokens",
            "accessToken": access,
        }
        if account_id:
            params["chatgptAccountId"] = account_id
        if plan_type:
            params["chatgptPlanType"] = plan_type
        return params

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
        finally:
            tail = "\n".join(self._stderr_tail[-10:])
            self._fail_all(
                CodexAppServerConnectionClosed(
                    f"codex app-server exited (rc={self._proc.returncode if self._proc else '?'})"
                    + (f": {tail}" if tail else "")
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
        self, mid: Any, method: str, params: dict
    ) -> None:
        try:
            tid = (params or {}).get("threadId")
            # Prefer a handler scoped to the owning thread; fall back to
            # a global (None) registration; then to safe defaults.
            handler = self._server_request_handlers.get((method, tid))
            if handler is None:
                handler = self._server_request_handlers.get((method, None))
            if handler is not None:
                result = await handler(params)
            elif method in _DEFAULT_APPROVAL_REPLIES:
                result = _DEFAULT_APPROVAL_REPLIES[method]
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
        except Exception as e:
            logger.warning("codex server-request %s failed: %s", method, e)
            try:
                self._send({"id": mid, "error": {"message": str(e)}})
            except Exception:
                pass

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
        self, method: str, params: Optional[dict] = None, *, timeout: float = 120,
    ) -> Any:
        """Send a request without going through ``ensure_started``.
        For use from inside ``_handshake`` where the start lock is
        already held — calling ``request`` from there would deadlock.
        """
        mid = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[mid] = fut
        self._send({"id": mid, "method": method, "params": params or {}})
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError as e:
            self._pending.pop(mid, None)
            raise CodexAppServerError(f"{method} timed out after {timeout}s") from e

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        self._send({"method": method, "params": params or {}})

    # ------------------------------------------------------- turn event streaming
    def open_turn_sink(self, key: Any) -> "asyncio.Queue[dict]":
        q: "asyncio.Queue[dict]" = asyncio.Queue()
        self._turn_sinks[key] = q
        return q

    def close_turn_sink(self, key: Any) -> None:
        self._turn_sinks.pop(key, None)

    async def iter_turn_events(
        self, sink: "asyncio.Queue[dict]", *, idle_timeout: float = 300
    ) -> "asyncio.AsyncIterator[dict]":
        """Yield notifications until ``turn/completed`` / failure / close.

        ``idle_timeout`` defaults to 300s so a long-running synchronous
        tool callback (e.g. one waiting on kestrel's approval queue)
        doesn't trip the local watchdog before the app-server's own
        server-request timeout fires.
        """
        while True:
            try:
                msg = await asyncio.wait_for(sink.get(), timeout=idle_timeout)
            except asyncio.TimeoutError as e:
                raise CodexAppServerError(
                    f"codex turn idle for {idle_timeout}s with no completion"
                ) from e
            if msg.get("__closed__"):
                raise self._closed_error or CodexAppServerConnectionClosed(
                    "codex app-server closed mid-turn"
                )
            yield msg
            if msg.get("method") in ("turn/completed", "turn/failed"):
                return
