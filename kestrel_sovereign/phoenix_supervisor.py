"""Host-supervised Arize Phoenix subprocess + same-origin embed helpers (#2570).

Part of the OTel-native pivot (kestrel-feature-observability#32). The deployed
host (``kestrel_sovereign.server:app``) supervises **one** Phoenix instance per
deployment the same way it supervises agents: a detached subprocess whose
stdout/stderr are redirected to a log file, its PID tracked on disk, and a
bounded health wait after spawn.

Three responsibilities live here so ``server.py`` stays thin:

1. :class:`PhoenixSupervisor` — spawn/stop the ``phoenix serve`` subprocess bound
   to ``127.0.0.1`` only (sovereign; never exposed). SQLite storage lives in a
   dedicated private host-data directory, never an implicit source checkout.
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

import errno
import importlib.util
import logging
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
import itsdangerous
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from kestrel_sovereign.private_storage import (
    PRIVATE_DIRECTORY_MODE,
    PRIVATE_FILE_MODE,
    PrivateStorageError as PhoenixStorageError,
    absolute_without_following_leaf as _absolute_without_following_leaf,
    ensure_private_directory as _ensure_private_directory,
    open_private_file as _open_private_file,
    path_exists as _path_exists,
)

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

#: Explicit operator override for the trace-store working directory. This is
#: deliberately separate from Phoenix's own environment variables: Kestrel is
#: the custody owner and must validate the directory before it starts Phoenix.
PHOENIX_WORKING_DIR_ENV = "KESTREL_PHOENIX_WORKING_DIR"

PRIVATE_CHILD_UMASK = 0o077

#: Default Phoenix project the host + every agent it spawns group their traces
#: under when the operator hasn't pinned one — a single ``kestrel-fleet`` project
#: instead of Phoenix's ``default`` (obs#32). The SDK's tracing bootstrap reads
#: ``KESTREL_OTEL_PROJECT`` (SDK >= 0.30.2); autowiring it here on ``os.environ``
#: means the host process AND subprocess agents that inherit the env pick it up.
DEFAULT_OTEL_PROJECT = "kestrel-fleet"

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


def phoenix_host_data_root() -> Path:
    """Private host-runtime root used for Phoenix when no override is set.

    An explicit ``KESTREL_HOME`` is an operator custody decision and is
    honoured. Without one, source-checkout discovery is intentionally ignored:
    trace data belongs under ``~/.kestrel/host-data``, never in a repository
    merely because Kestrel was launched from that repository.
    """
    from kestrel_sovereign.paths import host_data_dir

    return host_data_dir()


def phoenix_working_dir() -> Path:
    """Resolve the validated Phoenix working-directory destination.

    ``KESTREL_PHOENIX_WORKING_DIR`` is the explicit per-service override.
    Otherwise the store is isolated under :func:`phoenix_host_data_root`.
    Resolution has no filesystem side effects; :meth:`PhoenixSupervisor.prepare_storage`
    owns secure creation, hardening, and migration.
    """
    override = os.environ.get(PHOENIX_WORKING_DIR_ENV)
    if override:
        return _absolute_without_following_leaf(Path(override))
    return phoenix_host_data_root() / "phoenix"


def legacy_phoenix_working_dir() -> Path:
    """Return the pre-#2609 project-relative trace-store location."""
    from kestrel_sovereign.paths import project_dir

    return _absolute_without_following_leaf(project_dir() / "phoenix")


def phoenix_otlp_endpoint() -> str:
    """The OTLP/HTTP endpoint the host + agents export spans to.

    Every Kestrel emitter uses ``opentelemetry-exporter-otlp-proto-http``
    (POST ``{endpoint}/v1/traces``), which Phoenix serves on its main HTTP
    port — NOT the gRPC collector port (4317). Pointing the HTTP exporter at
    the gRPC port yields ``BadStatusLine`` errors (HTTP/2 frames in an
    HTTP/1.1 response) and no spans.
    """
    return f"http://{PHOENIX_BIND_HOST}:{phoenix_port()}"


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


def autowire_otel_project(env: Optional[dict] = None) -> Optional[str]:
    """Zero-config wiring (INV-SOLO): default ``KESTREL_OTEL_PROJECT``.

    Group the host's own traces and every span its spawned agents emit under a
    single ``kestrel-fleet`` Phoenix project rather than Phoenix's ``default``,
    but ONLY when the operator has not already pinned a project. Mutates ``env``
    in place (defaults to ``os.environ``, so both this process and env inherited
    by spawned agents pick it up) and returns the project it set, or ``None``
    when it left an operator-provided value untouched.
    """
    target = env if env is not None else os.environ
    if target.get("KESTREL_OTEL_PROJECT"):
        return None
    target["KESTREL_OTEL_PROJECT"] = DEFAULT_OTEL_PROJECT
    return DEFAULT_OTEL_PROJECT


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
# Private storage custody
# ---------------------------------------------------------------------------


def _secure_storage_tree(root: Path) -> None:
    """Reject links/special files and restrict an existing store recursively."""
    if not _path_exists(root):
        return
    _ensure_private_directory(root)
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise PhoenixStorageError(
            f"cannot enumerate Phoenix storage {root}: {exc}"
        ) from exc

    for entry in entries:
        path = Path(entry.path)
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise PhoenixStorageError(
                f"cannot inspect Phoenix storage entry {path}: {exc}"
            ) from exc
        if stat.S_ISLNK(st.st_mode):
            raise PhoenixStorageError(
                f"Phoenix storage contains a symbolic link; refusing custody: {path}"
            )
        if stat.S_ISDIR(st.st_mode):
            _secure_storage_tree(path)
            continue
        if not stat.S_ISREG(st.st_mode):
            raise PhoenixStorageError(
                f"Phoenix storage contains a special file; refusing custody: {path}"
            )
        # ``DirEntry.stat()`` above is the fast cached path (Win32
        # FindFirstFile/FindNextFile data on Windows) and does not reliably
        # populate ``st_nlink`` there — it reports 0 for perfectly normal,
        # single-link files. A full ``os.stat()`` call forces the real
        # syscall (GetFileInformationByHandle) and reports the correct
        # count. Without this, the hard-link check below always raises on
        # Windows, permanently refusing Phoenix custody regardless of the
        # file's actual state.
        try:
            nlink = os.stat(path, follow_symlinks=False).st_nlink
        except OSError as exc:
            raise PhoenixStorageError(
                f"cannot inspect Phoenix storage entry {path}: {exc}"
            ) from exc
        if nlink != 1:
            raise PhoenixStorageError(
                f"Phoenix storage file has {nlink} hard links; exclusive "
                f"custody cannot be established: {path}"
            )
        try:
            path.chmod(PRIVATE_FILE_MODE)
        except OSError as exc:
            raise PhoenixStorageError(
                f"cannot restrict Phoenix file {path} to mode 0600: {exc}"
            ) from exc


def _read_pid_file(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return None


def _copy_store_across_filesystems(source: Path, destination: Path) -> None:
    """Copy a stopped, already-hardened store through a private staging dir."""
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=".phoenix-migrate-", dir=destination.parent)
        )
        staging.chmod(PRIVATE_DIRECTORY_MODE)
    except OSError as exc:
        raise PhoenixStorageError(
            f"cannot create private Phoenix migration staging directory: {exc}"
        ) from exc

    try:
        for entry in source.iterdir():
            target = staging / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, copy_function=shutil.copy2)
            else:
                shutil.copy2(entry, target)
        _secure_storage_tree(staging)
        os.replace(staging, destination)
        _secure_storage_tree(destination)
        shutil.rmtree(source)
    except (OSError, shutil.Error) as exc:
        try:
            if _path_exists(staging):
                shutil.rmtree(staging)
        except OSError:
            pass
        raise PhoenixStorageError(
            f"cannot safely migrate Phoenix storage from {source} to "
            f"{destination}: {exc}"
        ) from exc


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
        self._uses_default_working_dir = (
            working_dir is None and not os.environ.get(PHOENIX_WORKING_DIR_ENV)
        )
        self.working_dir = (
            _absolute_without_following_leaf(Path(working_dir))
            if working_dir is not None
            else phoenix_working_dir()
        )
        self._storage_prepared = False
        # ``None`` means "auto-detect from the installed Phoenix". Resolved on
        # start() so tests can force a value.
        self._root_path_override = root_path
        self.root_path = root_path if root_path is not None else PHOENIX_ROOT_PATH
        self.process: Optional[subprocess.Popen] = None
        # PID of an orphaned Phoenix this supervisor *adopted* rather than spawned
        # (#2589). When set, ``self.process`` is ``None`` — there is no Popen
        # handle for a child a previous, hard-killed host leaked.
        self._adopted_pid: Optional[int] = None
        self._client: Optional[httpx.AsyncClient] = None

    # -- paths -----------------------------------------------------------
    @property
    def pid_file(self) -> Path:
        return self.working_dir / "phoenix.pid"

    @property
    def log_file(self) -> Path:
        return self.working_dir / "phoenix.log"

    # -- storage custody -------------------------------------------------
    def prepare_storage(self) -> None:
        """Secure the trace store before OTLP wiring or Phoenix launch.

        New stores are created as ``0700``. Existing stores are recursively
        hardened before any new sensitive bytes can be written. Supervisors
        using the default resolver also migrate the legacy project-relative
        ``phoenix/`` directory when the destination is unambiguous and stopped.

        Raises:
            PhoenixStorageError: custody cannot be established safely. Callers
                must leave tracing disabled in this case.
        """
        if self._storage_prepared:
            return

        if self._uses_default_working_dir:
            _ensure_private_directory(self.working_dir.parent)
            self._migrate_legacy_storage()

        _ensure_private_directory(self.working_dir)
        _secure_storage_tree(self.working_dir)
        self._storage_prepared = True

    def _migrate_legacy_storage(self) -> None:
        legacy = legacy_phoenix_working_dir()
        if legacy == self.working_dir or not _path_exists(legacy):
            return

        # First contain the disclosure in place. If anything after this point
        # fails, the legacy data is still private even though tracing remains
        # disabled until the operator resolves the migration error.
        _secure_storage_tree(legacy)

        legacy_pid = _read_pid_file(legacy / "phoenix.pid")
        if legacy_pid and self._pid_alive(legacy_pid):
            raise PhoenixStorageError(
                f"legacy Phoenix store {legacy} belongs to live PID {legacy_pid}; "
                "stop that Phoenix process, then restart Kestrel to migrate it"
            )

        if _path_exists(self.working_dir):
            _secure_storage_tree(self.working_dir)
            raise PhoenixStorageError(
                f"both legacy Phoenix storage {legacy} and destination "
                f"{self.working_dir} contain state; tracing is disabled to avoid "
                "an unsafe database merge. Back up both directories, choose the "
                "authoritative store, then remove or relocate the other."
            )

        try:
            os.replace(legacy, self.working_dir)
            _secure_storage_tree(self.working_dir)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise PhoenixStorageError(
                    f"cannot migrate Phoenix storage from {legacy} to "
                    f"{self.working_dir}: {exc}"
                ) from exc
            _copy_store_across_filesystems(legacy, self.working_dir)

        logger.warning(
            "Migrated legacy Phoenix trace storage from %s to private host-data "
            "directory %s.",
            legacy,
            self.working_dir,
        )

    # -- state -----------------------------------------------------------
    @property
    def running(self) -> bool:
        """True if the tracked process (spawned or adopted) is alive.

        Process liveness only. The session-mint / UI health gate uses
        :meth:`is_reachable` instead — health is *reachability*, not child
        liveness (#2589): gating on a tracked child's ``poll()`` ties the mint to
        a possibly-zombie process rather than whatever is actually serving the
        port.
        """
        if self.process is not None:
            return self.process.poll() is None
        if self._adopted_pid is not None:
            return self._pid_alive(self._adopted_pid)
        return False

    # -- reachability (health = can the iframe reach Phoenix) ------------
    def is_healthy(self, timeout: float = 2.0) -> bool:
        """Synchronous reachability probe of the Phoenix port (#2589).

        ``True`` when Phoenix answers an HTTP request on its port, regardless of
        whether *this* supervisor spawned it. This is what lets a restarted host
        detect (and adopt) an orphaned child that is still serving the port.
        """
        url = f"http://{self.host}:{self.port}{self.root_path}/"
        try:
            resp = httpx.get(url, timeout=timeout, follow_redirects=False)
            return resp.status_code < 500
        except httpx.HTTPError:
            return False

    async def is_reachable(self, timeout: float = 2.0) -> bool:
        """Async reachability probe — "can the iframe reach Phoenix" (#2589).

        The mint endpoint (and any supervisor health) gates on this rather than
        ``process.poll()`` so it tracks the process actually serving the port,
        not the tracked-but-possibly-zombie child.
        """
        url = f"http://{self.host}:{self.port}{self.root_path}/"
        try:
            client = self._proxy_client()
            resp = await client.get(url, timeout=timeout, follow_redirects=False)
            return resp.status_code < 500
        except httpx.HTTPError:
            return False

    # -- pid helpers (adopt-or-reap orphans across restarts) -------------
    def _read_pid(self) -> Optional[int]:
        """Read the tracked PID from the pidfile, or ``None`` if missing/invalid."""
        try:
            return int(self.pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    def _write_pid(self, pid: int) -> None:
        """Persist ``pid`` to the pidfile so a hard-killed host's successor can
        reap or adopt the leak (mirrors ``ProcessManager`` pid tracking)."""
        self.prepare_storage()
        fd = _open_private_file(
            self.pid_file,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        )
        try:
            os.write(fd, str(pid).encode("ascii"))
            os.fsync(fd)
        except OSError as exc:
            raise PhoenixStorageError(
                f"cannot persist private Phoenix PID file {self.pid_file}: {exc}"
            ) from exc
        finally:
            os.close(fd)

    @staticmethod
    def _pid_alive(pid: Optional[int]) -> bool:
        """Whether a process with ``pid`` is alive (best-effort, cross-platform)."""
        if not pid:
            return False
        if sys.platform == "win32":
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            except Exception:  # noqa: BLE001
                return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _legacy_store_pid(self) -> Optional[int]:
        """PID recorded in the legacy store's pidfile, or ``None``.

        Returns a PID only when this supervisor uses the default resolver
        (the only configuration that migrates) AND a distinct legacy
        ``phoenix/`` store is actually present with a pidfile. This is the
        same legacy-store signal :meth:`_migrate_legacy_storage` consults, so
        adopt-or-reap and the pre-storage reconcile route on it without an
        ``lsof`` dependency (#2690).
        """
        if not self._uses_default_working_dir:
            return None
        legacy = legacy_phoenix_working_dir()
        if legacy == self.working_dir or not _path_exists(legacy):
            return None
        return _read_pid_file(legacy / "phoenix.pid")

    def _port_listener_pids(self) -> list[int]:
        """Live PIDs actually listening on the Phoenix port (cross-platform, #2690).

        Delegates to the shared :meth:`ProcessManager.find_pids_on_port` lookup
        (``psutil`` first, then ``netstat``/``lsof``), so port ownership is
        established the SAME way on every platform — Windows and hosts without
        ``lsof`` included. A pidfile PID is **never** treated as proof of
        ownership: it can name a zombie that lost the race for the port, a
        still-live but non-listening child, or a reused PID (all observed on
        #2589/#2690). When no listener can be identified this returns ``[]``,
        which callers treat as "ownership unknown" and fail closed rather than
        trusting the pidfile.
        """
        from kestrel_sovereign.multi_agent.process_manager import ProcessManager

        try:
            pids = ProcessManager.find_pids_on_port(self.port)
        except Exception:  # noqa: BLE001 - discovery must never raise into custody
            return []
        return [p for p in pids if self._pid_alive(p)]

    def reconcile_storage_conflict(self) -> None:
        """Reap a legacy-store (or unidentified) Phoenix holding our port (#2690).

        Custody-aware reconcile that MUST run *before* :meth:`prepare_storage`
        (in both the host lifespan and :meth:`start`). A leaked pre-migration
        Phoenix serving the legacy repo-relative store can keep the port across
        a restart. If it survives into ``prepare_storage``, the legacy
        migration refuses to move a *live* store (custody stays read-only,
        #2627) — yet adopt-or-reap would otherwise adopt that same live
        Phoenix, leaving the whole boot on the divergent legacy store with the
        OTLP autowiring unset and tracing silently off (#2690).

        So, before storage is prepared, identify what holds the port and, if it
        is not our private-store child, reap it (SIGTERM → bounded wait →
        SIGKILL). The legacy migration then sees a *stopped* store and a
        fresh/adopted child serves the private store.

        Ownership is resolved through the shared cross-platform listener lookup
        (:meth:`_port_listener_pids` → ``ProcessManager.find_pids_on_port``); a
        pidfile PID is never treated as proof of ownership. Routing:

        * listener PID matches the PRIVATE pidfile → leave it (adoptable later)
        * listener PID matches the LEGACY pidfile → reap it (a legacy-store
          Phoenix is never adoptable)
        * matches neither → reap conservatively (an unidentifiable Phoenix
          squatting our port is never trustworthy to adopt — a known
          private-store child is spawned instead)
        * port is healthy but NO listener PID can be identified → **fail closed**
          (:class:`PhoenixStorageError`): migrating now would strand the boot on
          a divergent store and an unverifiable listener is never safe to adopt,
          so leave the legacy store untouched for the operator to resolve.

        Scoped to the default resolver with a legacy store actually present —
        the only configuration that migrates — so an explicit
        ``KESTREL_PHOENIX_WORKING_DIR`` never triggers a reap. Idempotent:
        after the reap the port is free, so a repeat boot is a no-op here and
        converges through adopt-or-reap instead.

        Raises:
            PhoenixStorageError: a Phoenix holds the port with an
                unidentifiable owner while a legacy store is still present;
                callers must leave tracing disabled (fail closed).
        """
        if not self._uses_default_working_dir:
            return
        legacy = legacy_phoenix_working_dir()
        if legacy == self.working_dir or not _path_exists(legacy):
            return
        if not self.is_healthy():
            return

        private_pid = self._read_pid()
        legacy_pid = _read_pid_file(legacy / "phoenix.pid")
        listeners = self._port_listener_pids()
        if not listeners:
            # Something is serving the port but its owning PID cannot be
            # established while a legacy store is still present. Migrating now
            # would strand the boot on a divergent store, and an unidentifiable
            # listener is never safe to adopt. Fail custody CLOSED (#2690) —
            # leave the legacy store untouched and let the operator free the
            # port rather than treating a pidfile PID as proof of ownership.
            raise PhoenixStorageError(
                f"a Phoenix is serving {self.host}:{self.port} but its owning "
                f"PID cannot be identified while the legacy trace store {legacy} "
                "is still present; refusing to migrate or adopt. Stop the "
                "process holding the port, then restart Kestrel."
            )
        for pid in listeners:
            if private_pid and pid == private_pid:
                # Already serving the PRIVATE store — adoptable by adopt-or-reap.
                continue
            descriptor = (
                "legacy-store" if (legacy_pid and pid == legacy_pid) else "unidentified"
            )
            logger.warning(
                "Reaping %s Phoenix (PID %s) holding %s:%s before migration so "
                "the legacy trace store can move to private host-data and a "
                "private-store child can serve (#2690).",
                descriptor,
                pid,
                self.host,
                self.port,
            )
            self._terminate_pid(pid)

    def _terminate_pid(self, pid: int, *, timeout: float = 10.0) -> None:
        """Reap a leaked/zombie Phoenix child by PID (SIGTERM, then SIGKILL)."""
        logger.info("Reaping leaked Phoenix child (PID %s)", pid)
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    check=False,
                )
                return
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._pid_alive(pid):
                return
            time.sleep(0.25)
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    def _adopt_or_reap(self) -> bool:
        """Reconcile any Phoenix already bound to our port before spawning (#2589).

        A hard host stop (launcher SIGKILL path) skips lifespan shutdown and
        leaks the Phoenix child, which keeps serving the port across the restart.
        Spawning a *second* child into the held port just idles a zombie, while
        the mint/health gate can end up pointing at the wrong process (the
        split-brain observed on #2589). So, before spawning, when the port is
        healthy we route strictly on **verified port ownership** (#2690):

        * Listener PID matches the PRIVATE pidfile → **adopt** it (a child we
          spawned, still serving our private store). Reap any *other* tracked
          child (the split-brain zombie that lost the race). Returns ``True`` so
          the caller does **not** spawn a second child.
        * Listener PID does NOT match the private pidfile → it is **not**
          trustworthy to adopt (it could be a legacy-store Phoenix, an external
          process, or a reused PID — adopting it would strand the whole boot on a
          divergent/untrusted trace store). Reap it (and any stale private PID),
          clear the pidfile, and return ``False`` so a fresh private-store child
          is spawned.
        * Healthy but NO owning PID can be identified (no ``psutil``/``lsof``, or
          the listener belongs to another user) → **fail closed**: we cannot
          verify it is our private child and cannot reap it to spawn our own, so
          raise :class:`PhoenixStorageError` rather than adopt an unverifiable
          store. A pidfile PID is never treated as proof of port ownership.

        When nothing healthy holds the port, reap a leaked/hung child the pidfile
        names (a bound-failed leak) so a fresh child can bind. Returns ``False``.
        """
        prior_pid = self._read_pid()
        if self.is_healthy():
            listeners = self._port_listener_pids()
            # Adopt ONLY a listener whose PID matches our PRIVATE pidfile — the
            # PID a child WE spawned wrote there. Anything else serving the port
            # is never adoptable as our private-store child (#2690).
            if prior_pid and prior_pid in listeners:
                # Reap any OTHER serving child (the split-brain zombie that lost
                # the race for the port), then adopt the verified private child.
                for pid in listeners:
                    if pid != prior_pid and self._pid_alive(pid):
                        self._terminate_pid(pid)
                self._adopted_pid = prior_pid
                self.process = None
                self._write_pid(prior_pid)
                logger.info(
                    "Adopted already-serving private-store Phoenix (PID %s) on "
                    "%s:%s — not spawning a second child.",
                    prior_pid,
                    self.host,
                    self.port,
                )
                return True

            if listeners:
                # Healthy, but the listener is NOT our private-store child. Reap
                # it (and any stale private PID the pidfile still names) and let
                # the caller spawn a fresh private child rather than adopt an
                # untrusted/divergent trace store (#2690).
                legacy_pid = self._legacy_store_pid()
                for pid in {prior_pid, *listeners}:
                    if not pid or not self._pid_alive(pid):
                        continue
                    descriptor = (
                        "legacy-store" if (legacy_pid and pid == legacy_pid)
                        else "untrusted"
                    )
                    logger.warning(
                        "Reaping %s Phoenix (PID %s) holding %s:%s — it is not "
                        "our private-store child; spawning a fresh private-store "
                        "child instead of adopting a divergent trace store "
                        "(#2690).",
                        descriptor,
                        pid,
                        self.host,
                        self.port,
                    )
                    self._terminate_pid(pid)
                self._clear_pid()
                return False

            # Healthy but the owning PID cannot be identified. We cannot verify
            # it is our private child, and cannot reap it to spawn our own — fail
            # closed rather than adopt an unverifiable trace store (#2690).
            raise PhoenixStorageError(
                f"a Phoenix is serving {self.host}:{self.port} but its owning "
                "PID cannot be identified; refusing to adopt an unverifiable "
                "trace store. Stop the process holding the port, then restart "
                "Kestrel."
            )

        # Nothing healthy on the port. Reap a leaked/hung child the pidfile names
        # (it could not bind and will never serve) so our fresh child can, then
        # drop the stale pidfile.
        if prior_pid and self._pid_alive(prior_pid):
            logger.info(
                "Reaping stale Phoenix child (PID %s) — not serving %s:%s.",
                prior_pid,
                self.host,
                self.port,
            )
            self._terminate_pid(prior_pid)
        self._clear_pid()
        return False

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
        # Disable Phoenix's external telemetry pixels (FullStory + Scarf.sh).
        # The UI otherwise loads a scarf.sh tracking pixel that phones home and
        # trips ad/tracker blockers, so the embedded console logs
        # ERR_BLOCKED_BY_CLIENT noise. ``PHOENIX_TELEMETRY_ENABLED`` is the
        # master toggle (phoenix.config.get_env_telemetry_enabled) — verified in
        # the installed 17.x config; it MUST be the literal "true"/"false"
        # Phoenix parses, anything else asserts at Phoenix startup. ``setdefault``
        # so an operator who wants the pixels can still opt back in explicitly.
        env.setdefault("PHOENIX_TELEMETRY_ENABLED", "false")
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

        # Custody-aware reconcile BEFORE storage prep (#2690): reap a leaked
        # legacy-store (or unidentified) Phoenix holding the port so the legacy
        # migration below sees a *stopped* store.
        try:
            self.reconcile_storage_conflict()
        except PhoenixStorageError as exc:
            # Custody could not be established safely (e.g. an unidentifiable
            # Phoenix holds the port while a legacy store is present). Fail
            # CLOSED rather than migrate/adopt onto a divergent store (#2690).
            logger.error(
                "Phoenix tracing disabled because storage-conflict reconcile "
                "could not establish safe custody: %s",
                exc,
            )
            return False
        except Exception as exc:  # noqa: BLE001 - other reconcile errors degrade to spawn
            logger.warning("Phoenix storage-conflict reconcile failed: %s", exc)

        try:
            self.prepare_storage()
        except PhoenixStorageError as exc:
            logger.error(
                "Phoenix tracing disabled because private storage custody could "
                "not be established: %s",
                exc,
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

        # Adopt-or-reap: never spawn a second child into a port an orphaned
        # Phoenix already holds (#2589 split-brain). Adopt a still-serving child,
        # or reap a non-serving leak so our fresh child can bind. Never blocks
        # host startup — a reconcile failure degrades to a normal spawn.
        try:
            if self._adopt_or_reap():
                return True
        except PhoenixStorageError as exc:
            logger.error(
                "Phoenix tracing disabled because private PID custody failed: %s",
                exc,
            )
            return False
        except Exception as exc:  # noqa: BLE001 - reconcile must never block startup
            logger.warning("Phoenix adopt-or-reap reconcile failed: %s", exc)

        try:
            log_fd = _open_private_file(
                self.log_file,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
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
                    # Popen applies this in the child before exec. Phoenix's DB,
                    # WAL, SHM, and any future files are private from their first
                    # inode; no post-write repair window exists.
                    kwargs["umask"] = PRIVATE_CHILD_UMASK
                self.process = subprocess.Popen(self.build_command(), **kwargs)
            finally:
                os.close(log_fd)
        except (
            OSError,
            ValueError,
            PhoenixStorageError,
            subprocess.SubprocessError,
        ) as exc:
            logger.warning("Failed to launch Phoenix subprocess: %s", exc)
            self.process = None
            return False

        # Freshly spawned: this supervisor owns a real Popen, not an adopted PID.
        self._adopted_pid = None
        try:
            self._write_pid(self.process.pid)
        except PhoenixStorageError as exc:
            logger.error(
                "Phoenix tracing disabled because its PID file could not be "
                "secured: %s",
                exc,
            )
            proc = self.process
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except (OSError, subprocess.SubprocessError):
                    pass
            except (OSError, subprocess.SubprocessError):
                pass
            self.process = None
            return False

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
            # No Popen handle — but we may have ADOPTED an orphaned child by PID
            # (#2589). Reap it so a graceful host shutdown leaves no orphan.
            if self._adopted_pid is not None:
                self._terminate_pid(self._adopted_pid, timeout=timeout)
                self._adopted_pid = None
            self._clear_pid()
            return
        if proc.poll() is not None:
            self.process = None
            self._adopted_pid = None
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
        self._adopted_pid = None
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

        ASGI ``root_path`` semantics: ``PHOENIX_HOST_ROOT_PATH`` tells Phoenix
        to *generate* URLs under ``/phoenix`` (the SPA basename), but its
        routing still matches the UNPREFIXED path — the reverse proxy is
        expected to strip the prefix before forwarding. Forwarding the prefix
        verbatim makes every asset/API path miss and fall back to the SPA
        index (200 ``text/html`` for ``.js`` → the browser's strict-MIME
        module errors). Verified live on 17.7.0: ``/assets/x.js`` → JS,
        ``/phoenix/assets/x.js`` → index fallback.
        """
        rel = path.lstrip("/")
        url = f"http://{self.host}:{self.port}/{rel}"
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

    Gating is by *reachability*, not child liveness (#2589): an adopted orphan
    has no local ``Popen`` handle, and during a non-blocking first boot Phoenix
    may still be coming up. So we forward to the port and let the connection be
    the reachability test — an unreachable Phoenix returns 503 ("not ready yet"),
    not 502.
    """
    if supervisor is None:
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
        # Not reachable yet (still starting on first boot, or crashed) — 503 so
        # the console retries rather than treating it as a hard proxy error.
        logger.warning("Phoenix is unreachable at %s", target_url)
        return _phoenix_disabled_response()
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
    "PHOENIX_WORKING_DIR_ENV",
    "DEFAULT_OTEL_PROJECT",
    "EMBED_COOKIE_NAME",
    "EMBED_COOKIE_PATH",
    "EMBED_TTL_SECONDS",
    "PhoenixSupervisor",
    "PhoenixStorageError",
    "phoenix_available",
    "phoenix_enabled",
    "should_supervise_phoenix",
    "phoenix_port",
    "phoenix_grpc_port",
    "phoenix_working_dir",
    "phoenix_host_data_root",
    "legacy_phoenix_working_dir",
    "phoenix_otlp_endpoint",
    "autowire_otlp_endpoint",
    "autowire_otel_project",
    "supports_host_root_path",
    "mint_embed_token",
    "issue_embed_cookie",
    "verify_embed_cookie",
    "proxy_to_phoenix",
]
