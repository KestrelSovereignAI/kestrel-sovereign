"""Host-supervised Arize Phoenix subprocess + same-origin embed helpers (#2570).

Part of the OTel-native pivot (kestrel-feature-observability#32). The deployed
host (``kestrel_sovereign.server:app``) supervises **one** Phoenix instance per
deployment the same way it supervises agents: a detached subprocess whose
stdout/stderr are redirected to a log file, its PID tracked on disk, and a
bounded health wait after spawn.

Three responsibilities live here so ``server.py`` stays thin:

1. :class:`PhoenixSupervisor` — spawn/stop the ``phoenix serve`` subprocess bound
   to ``127.0.0.1`` only (sovereign; never exposed). SQLite storage lives under
   the host's data-dir convention (:func:`kestrel_sovereign.paths.project_dir`).
2. Embed session cookie — :func:`issue_embed_cookie` / :func:`verify_embed_cookie`
   mint and validate a short-lived, signed, ``HttpOnly`` ``SameSite=Lax`` cookie
   scoped to ``/phoenix``. This is what lets the console iframe load the Phoenix
   UI same-origin with **no credentials in the URL**.
3. Reverse proxy — :func:`proxy_to_phoenix` streams an authenticated
   ``/phoenix/{path}`` request through to the local Phoenix, rewriting the path
   to Phoenix's configured host-root-path.

Graceful degrade is a hard requirement: if ``arize-phoenix`` is not importable,
:func:`phoenix_available` returns ``False`` and the host is unaffected — one log
line, ``/phoenix/`` returns a clear 503. Opt out even when installed with
``KESTREL_PHOENIX_ENABLED=0``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import itsdangerous
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)

# Phoenix is bound to loopback ONLY. It is sovereign infrastructure — the trace
# backend must never be reachable except through the host's authenticated
# reverse proxy. This is deliberately not configurable.
PHOENIX_BIND_HOST = "127.0.0.1"

#: Default Phoenix HTTP/UI port (also serves HTTP OTLP at ``/v1/traces``).
DEFAULT_PHOENIX_PORT = 6006
#: Default Phoenix gRPC OTLP collector port (what ``OTEL_EXPORTER_OTLP_ENDPOINT``
#: points at — the framework's exporter in ``telemetry.py`` is gRPC).
DEFAULT_PHOENIX_GRPC_PORT = 4317

#: Sub-path the UI is served under for same-origin embedding.
PHOENIX_ROOT_PATH = "/phoenix"

# --- Embed session cookie ------------------------------------------------
#: Cookie the console mints via ``POST /api/host/phoenix/session`` and the
#: browser then auto-attaches to iframe ``/phoenix`` requests.
EMBED_COOKIE_NAME = "kestrel_phoenix_embed"
#: Scope the cookie to ``/phoenix`` so it is never sent to any other endpoint.
EMBED_COOKIE_PATH = "/phoenix"
#: Short-lived by design (relative to the 7-day main session cookie).
EMBED_TTL_SECONDS = 8 * 3600
_EMBED_SALT = "kestrel-phoenix-embed-v1"

# Proxy timeouts — Phoenix is local, but the UI holds long-poll/streaming
# connections, so the read timeout is generous.
_PROXY_CONNECT_TIMEOUT = 5.0
_PROXY_READ_TIMEOUT = 300.0

_HOP_BY_HOP = {"host", "transfer-encoding", "connection", "keep-alive"}


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _falsy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("0", "false", "no", "off")


def phoenix_available() -> bool:
    """Return whether ``arize-phoenix`` is importable (the ``phoenix`` package)."""
    try:
        return importlib.util.find_spec("phoenix") is not None
    except (ImportError, ValueError):
        return False


def phoenix_enabled() -> bool:
    """Return whether the host should supervise Phoenix.

    Enabled by default whenever ``arize-phoenix`` is installed. Opt out with
    ``KESTREL_PHOENIX_ENABLED=0`` even when installed.

    This is the *installed-and-not-opted-out* predicate. Whether the host
    lifespan should actually spawn the subprocess right now is
    :func:`should_supervise_phoenix`, which additionally suppresses auto-start
    inside the automated test suite.
    """
    if _falsy(os.environ.get("KESTREL_PHOENIX_ENABLED")):
        return False
    return phoenix_available()


def _running_under_pytest() -> bool:
    """True when executing inside the pytest test suite.

    ``pytest`` is always imported into ``sys.modules`` for a test run and never
    in a real ``uvicorn``/``gunicorn`` host process, so this cleanly separates
    the two. ``PYTEST_CURRENT_TEST`` is also honoured for defence in depth.
    """
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def should_supervise_phoenix() -> bool:
    """Whether the host lifespan should actually launch a supervised Phoenix.

    This is :func:`phoenix_enabled` **and** not running under pytest. The host
    must never spawn a *real* Phoenix subprocess during automated tests — issue
    #2570 acceptance is explicit: "no real Phoenix needed in CI", and the
    dedicated tests stub the process. Without this guard, every integration
    test that starts the app through ``TestClient`` (triggering the lifespan)
    would spawn a heavyweight Phoenix subprocess and then pay its
    SIGTERM/SIGKILL teardown on shutdown — mirroring how agent subprocesses are
    gated out of the test path (``KESTREL_MULTI_AGENT``).

    Direct :meth:`PhoenixSupervisor.start` calls are intentionally *not* guarded
    here, so unit tests can still exercise the (stubbed) spawn/stop lifecycle.
    """
    if _running_under_pytest():
        return False
    return phoenix_enabled()


def phoenix_port() -> int:
    """Phoenix HTTP/UI port (``KESTREL_PHOENIX_PORT``, default 6006)."""
    try:
        return int(os.environ.get("KESTREL_PHOENIX_PORT", DEFAULT_PHOENIX_PORT))
    except (TypeError, ValueError):
        return DEFAULT_PHOENIX_PORT


def phoenix_grpc_port() -> int:
    """Phoenix gRPC OTLP port (``KESTREL_PHOENIX_GRPC_PORT``, default 4317)."""
    try:
        return int(
            os.environ.get("KESTREL_PHOENIX_GRPC_PORT", DEFAULT_PHOENIX_GRPC_PORT)
        )
    except (TypeError, ValueError):
        return DEFAULT_PHOENIX_GRPC_PORT


def phoenix_working_dir() -> Path:
    """Directory Phoenix stores its SQLite DB in, under the data-dir convention.

    Uses :func:`kestrel_sovereign.paths.project_dir` (KESTREL_HOME / marker
    walk-up / ``~/.kestrel``) so the trace store lives alongside the agents'
    data, not in a scratch CWD.
    """
    from kestrel_sovereign.paths import project_dir

    return project_dir() / "phoenix"


def phoenix_otlp_endpoint() -> str:
    """The gRPC OTLP endpoint the host + agents export spans to."""
    return f"http://{PHOENIX_BIND_HOST}:{phoenix_grpc_port()}"


def autowire_otlp_endpoint(env: Optional[dict] = None) -> Optional[str]:
    """Zero-config wiring (INV-SOLO): default ``OTEL_EXPORTER_OTLP_ENDPOINT``.

    Point the OTLP exporter at the local Phoenix collector, but ONLY when the
    operator has not already set the endpoint. Mutates ``env`` in place (defaults
    to ``os.environ``, so both the host's own tracing and env inherited by
    spawned agents pick it up) and returns the endpoint it set, or ``None`` when
    it left an operator-provided value untouched.
    """
    target = env if env is not None else os.environ
    if target.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    endpoint = phoenix_otlp_endpoint()
    target["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    return endpoint


def supports_host_root_path() -> bool:
    """Best-effort probe: does the installed Phoenix honour ``PHOENIX_HOST_ROOT_PATH``?

    Newer Phoenix versions expose the config via
    ``phoenix.config.get_env_host_root_path`` (and the
    ``PHOENIX_HOST_ROOT_PATH`` env constant). If neither is present we fall back
    to root-path serving and the proxy strips the ``/phoenix`` prefix instead
    (see :meth:`PhoenixSupervisor.upstream_url`).
    """
    try:
        import phoenix.config as pc  # type: ignore
    except Exception:  # noqa: BLE001 - any import failure ⇒ assume unsupported
        return False
    return (
        hasattr(pc, "get_env_host_root_path")
        or hasattr(pc, "PHOENIX_HOST_ROOT_PATH")
        or "PHOENIX_HOST_ROOT_PATH" in getattr(pc, "__dict__", {})
    )


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class PhoenixSupervisor:
    """Supervise a single Phoenix subprocess, mirroring the agent process model.

    The subprocess is spawned detached (its own session/process group) with
    stdout+stderr redirected straight to ``phoenix.log`` in the working dir, and
    its PID written to ``phoenix.pid``. This matches
    :class:`~kestrel_sovereign.multi_agent.process_manager.ProcessManager`'s
    ``_spawn_detached`` pattern.
    """

    def __init__(
        self,
        *,
        host: str = PHOENIX_BIND_HOST,
        port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        working_dir: Optional[Path] = None,
        root_path: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port if port is not None else phoenix_port()
        self.grpc_port = grpc_port if grpc_port is not None else phoenix_grpc_port()
        self.working_dir = (
            Path(working_dir) if working_dir is not None else phoenix_working_dir()
        )
        # ``None`` means "auto-detect from the installed Phoenix". Resolved on
        # start() so tests can force a value.
        self._root_path_override = root_path
        self.root_path = root_path if root_path is not None else PHOENIX_ROOT_PATH
        self.process: Optional[subprocess.Popen] = None
        self._client: Optional[httpx.AsyncClient] = None

    # -- paths -----------------------------------------------------------
    @property
    def pid_file(self) -> Path:
        return self.working_dir / "phoenix.pid"

    @property
    def log_file(self) -> Path:
        return self.working_dir / "phoenix.log"

    # -- state -----------------------------------------------------------
    @property
    def running(self) -> bool:
        """True if the supervised process is alive."""
        if self.process is None:
            return False
        return self.process.poll() is None

    # -- launch ----------------------------------------------------------
    def build_command(self) -> list[str]:
        """The argv used to launch Phoenix as a subprocess."""
        return [sys.executable, "-m", "phoenix.server.main", "serve"]

    def build_env(self, base_env: Optional[dict] = None) -> dict:
        """Compose the child env: loopback bind, port, working dir, root-path."""
        env = dict(base_env if base_env is not None else os.environ)
        env["PHOENIX_HOST"] = self.host
        env["PHOENIX_PORT"] = str(self.port)
        env["PHOENIX_GRPC_PORT"] = str(self.grpc_port)
        env["PHOENIX_WORKING_DIR"] = str(self.working_dir)
        # SQLite storage lives in the working dir.
        env.setdefault(
            "PHOENIX_SQL_DATABASE_URL",
            f"sqlite:///{self.working_dir / 'phoenix.db'}",
        )
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self.root_path:
            env["PHOENIX_HOST_ROOT_PATH"] = self.root_path
        else:
            # Fallback (root-path serving) — make sure a stale value can't leak in.
            env.pop("PHOENIX_HOST_ROOT_PATH", None)
        return env

    def start(self, *, wait_for_health: bool = True, timeout: float = 30.0) -> bool:
        """Spawn Phoenix. Returns True if the subprocess was launched.

        Never raises on a Phoenix-side problem: a failure to launch degrades to
        a logged warning and a ``False`` return so the host keeps booting.
        """
        if not phoenix_enabled():
            logger.info(
                "Phoenix trace backend not enabled "
                "(arize-phoenix not installed or KESTREL_PHOENIX_ENABLED=0) — "
                "skipping Phoenix supervision; /phoenix will return 503."
            )
            return False

        # Resolve the effective root-path against the installed Phoenix unless a
        # value was pinned by the caller/tests.
        if self._root_path_override is None:
            if supports_host_root_path():
                self.root_path = PHOENIX_ROOT_PATH
            else:
                logger.warning(
                    "Installed Phoenix does not support PHOENIX_HOST_ROOT_PATH — "
                    "falling back to root-path serving; the proxy will strip the "
                    "/phoenix prefix. Some absolute UI links may not embed cleanly."
                )
                self.root_path = ""

        try:
            self.working_dir.mkdir(parents=True, exist_ok=True)
            log_fd = os.open(
                self.log_file,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o644,
            )
            try:
                kwargs: dict = dict(
                    env=self.build_env(),
                    stdout=log_fd,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                )
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    kwargs["start_new_session"] = True
                self.process = subprocess.Popen(self.build_command(), **kwargs)
            finally:
                os.close(log_fd)
        except Exception as exc:  # noqa: BLE001 - never block host startup
            logger.warning("Failed to launch Phoenix subprocess: %s", exc)
            self.process = None
            return False

        try:
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            self.pid_file.write_text(str(self.process.pid))
        except OSError:
            pass

        logger.info(
            "Phoenix supervised on %s:%s (PID %s, root_path=%r, working_dir=%s)",
            self.host,
            self.port,
            self.process.pid,
            self.root_path or "/",
            self.working_dir,
        )

        if wait_for_health:
            if self.wait_for_health(timeout=timeout):
                logger.info("Phoenix is healthy on %s:%s", self.host, self.port)
            else:
                logger.warning(
                    "Phoenix did not report healthy within %.0fs — continuing; "
                    "the UI/collector may still be coming up.",
                    timeout,
                )
        return True

    def wait_for_health(self, timeout: float = 30.0) -> bool:
        """Poll Phoenix's UI root until it answers, or the process dies."""
        url = f"http://{self.host}:{self.port}{self.root_path}/"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.running:
                return False
            try:
                resp = httpx.get(url, timeout=2.0, follow_redirects=False)
                # Any HTTP answer (even a redirect/401) means the server is up.
                if resp.status_code < 500:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        return False

    def stop(self, *, timeout: float = 10.0) -> None:
        """Terminate Phoenix gracefully, escalating to SIGKILL after ``timeout``."""
        proc = self.process
        if proc is None:
            self._clear_pid()
            return
        if proc.poll() is not None:
            self.process = None
            self._clear_pid()
            return

        logger.info("Stopping Phoenix (PID %s)...", proc.pid)
        try:
            if sys.platform == "win32":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.25)

        if proc.poll() is None:
            logger.warning("Phoenix did not stop gracefully — sending SIGKILL")
            try:
                proc.kill()
            except (OSError, ProcessLookupError):
                pass
            try:
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, Exception):  # noqa: BLE001
                pass

        self.process = None
        self._clear_pid()
        logger.info("Phoenix stopped")

    def _clear_pid(self) -> None:
        try:
            self.pid_file.unlink(missing_ok=True)
        except OSError:
            pass

    async def aclose(self) -> None:
        """Close the proxy httpx client, if opened."""
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    # -- proxy helpers ---------------------------------------------------
    def _proxy_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=_PROXY_CONNECT_TIMEOUT,
                    read=_PROXY_READ_TIMEOUT,
                    write=_PROXY_READ_TIMEOUT,
                    pool=_PROXY_CONNECT_TIMEOUT,
                ),
                follow_redirects=False,
            )
        return self._client

    def upstream_url(self, path: str, query: str = "") -> str:
        """Map an incoming ``/phoenix/{path}`` to the Phoenix upstream URL.

        When Phoenix serves under ``PHOENIX_HOST_ROOT_PATH=/phoenix`` (the normal
        case) the full ``/phoenix/...`` path is forwarded verbatim so Phoenix's
        own routing matches. In the fallback (root-path serving) the ``/phoenix``
        prefix is dropped.
        """
        rel = path.lstrip("/")
        prefix = self.root_path or ""
        full = f"{prefix}/{rel}" if prefix else f"/{rel}"
        url = f"http://{self.host}:{self.port}{full}"
        if query:
            url = f"{url}?{query}"
        return url


# ---------------------------------------------------------------------------
# Embed session cookie (no credentials in URLs)
# ---------------------------------------------------------------------------


def _serializer(secret: str) -> "itsdangerous.URLSafeTimedSerializer":
    return itsdangerous.URLSafeTimedSerializer(secret, salt=_EMBED_SALT)


def mint_embed_token(secret: str, *, identity: str = "sovereign") -> str:
    """Return a signed, timestamped embed token."""
    return _serializer(secret).dumps({"sub": identity})


def issue_embed_cookie(
    response: Response,
    secret: str,
    *,
    identity: str = "sovereign",
    secure: bool = False,
) -> str:
    """Attach the short-lived embed cookie to ``response``; return the token.

    ``HttpOnly`` (JS can't read it — it only needs to be *sent*), ``SameSite=Lax``
    and scoped to ``/phoenix`` so it is never attached to any other endpoint.
    """
    token = mint_embed_token(secret, identity=identity)
    response.set_cookie(
        EMBED_COOKIE_NAME,
        token,
        max_age=EMBED_TTL_SECONDS,
        path=EMBED_COOKIE_PATH,
        httponly=True,
        samesite="lax",
        secure=secure,
    )
    return token


def verify_embed_cookie(request: Request, secret: str) -> bool:
    """True iff the request carries a valid, unexpired embed cookie."""
    token = request.cookies.get(EMBED_COOKIE_NAME)
    if not token:
        return False
    try:
        _serializer(secret).loads(token, max_age=EMBED_TTL_SECONDS)
        return True
    except itsdangerous.BadData:
        return False


# ---------------------------------------------------------------------------
# Reverse proxy
# ---------------------------------------------------------------------------


def _phoenix_disabled_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Phoenix not enabled",
            "hint": (
                "Install the extra (pip install 'kestrel-sovereign[phoenix]') and "
                "ensure KESTREL_PHOENIX_ENABLED is not 0."
            ),
        },
    )


async def proxy_to_phoenix(
    request: Request,
    supervisor: Optional[PhoenixSupervisor],
    path: str,
) -> Response:
    """Stream an authenticated ``/phoenix/{path}`` request to the local Phoenix.

    Auth is enforced upstream by the host's ``auth_middleware`` (session cookie,
    API key, or the minted embed cookie) — by the time we get here the caller is
    authorized. Returns a clear 503 when Phoenix is not supervised.
    """
    if supervisor is None or not supervisor.running:
        return _phoenix_disabled_response()

    client = supervisor._proxy_client()
    target_url = supervisor.upstream_url(path, request.url.query)
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    body = await request.body()

    try:
        upstream_req = client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body if body else None,
        )
        upstream = await client.send(upstream_req, stream=True)
    except httpx.ConnectError:
        logger.warning("Phoenix is unreachable at %s", target_url)
        return JSONResponse(
            status_code=502,
            content={"detail": "Phoenix is unreachable"},
        )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"detail": "Timeout proxying to Phoenix"},
        )

    resp_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP | {"content-length"}
    }

    async def _stream():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        _stream(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


__all__ = [
    "PHOENIX_BIND_HOST",
    "PHOENIX_ROOT_PATH",
    "EMBED_COOKIE_NAME",
    "EMBED_COOKIE_PATH",
    "EMBED_TTL_SECONDS",
    "PhoenixSupervisor",
    "phoenix_available",
    "phoenix_enabled",
    "should_supervise_phoenix",
    "phoenix_port",
    "phoenix_grpc_port",
    "phoenix_working_dir",
    "phoenix_otlp_endpoint",
    "autowire_otlp_endpoint",
    "supports_host_root_path",
    "mint_embed_token",
    "issue_embed_cookie",
    "verify_embed_cookie",
    "proxy_to_phoenix",
]
