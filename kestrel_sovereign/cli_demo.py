"""``kestrel demo run <name>`` CLI command — sub-PR 3.1 of epic #1050
(bash-to-Python port of ``demos/run.sh``).

Runs a Kestrel demo against an isolated demo agent. Never talks to
the live server on port 8888.

The runner:

1. Validates ``demos/<name>/config.cjs`` exists (refuses unknown demos,
   listing the available set).
2. Refuses ``DEMO_PORT=8888`` and refuses an already-bound port — both
   are convention-layer rails against the 2026-04-24 incident
   (``#766``).
3. Spawns ``scripts/setup_demo_agent.py`` to create a fresh
   ``agent_data/demo/`` DB.
4. Starts a dedicated uvicorn on ``DEMO_PORT`` (default 8900) against
   the demo DB; ``KESTREL_MULTI_AGENT_CONFIG`` is forced to a
   non-existent path so the server skips multi_agent loading and the
   ``KESTREL_DEMO_SERVER=1`` flag re-asserts the same intent at the
   feature layer (``#868``).
5. Polls ``/health`` until 200, then sanity-checks
   ``/api/agents`` — every loaded agent must report ``is_demo=true``,
   else we refuse to run (this is the routing precondition that wiped
   Meridian; both preventatives failing simultaneously is the only
   case where the rail catches it).
6. ``cd demos/<name> && npx playwright test --config=config.cjs`` with
   ``KESTREL_URL`` pointing at the isolated server, ``KESTREL_API_KEY``
   STRIPPED so the demo fetches the demo agent's key via
   ``/api/auth/key``, and LLM-provider keys preserved.
7. Tears the server down on exit (``finally``-block trap), unless
   ``--keep-server`` was passed.

``kestrel demo smoke`` (#2682) drives the same isolated-server lifecycle for
the Sovereign Console Playwright smoke — the subset pull-request CI runs. It
differs from a demo wherever a demo would let host state in:

* The instance lives in a fresh home (``--home``, which must be absent or
  empty, or a new temp dir). That home is the server's ``KESTREL_HOME`` and
  working directory. The server also reads no ``.env`` file at all
  (``KESTREL_SKIP_DOTENV=1``): not its project home's, not its launch
  directory's, and not a legacy one next to the package source, any of which
  would refill the variables the smoke removed. It gets a throwaway
  ``KESTREL_DATA_KEY``.
* No ``KESTREL_*`` setting, provider key, or credential-shaped variable is
  inherited, and the agent is configured with only the local Ollama route, so
  no paid LLM can be selected. The smoke sends only ``!status``, a
  non-cognitive command.
* Before Playwright runs, it proves the instance's origin: the server
  resolves ``kestrel_sovereign`` from this runner's package, and the one agent
  it serves carries the DID minted by this run's inception. Those facts go to
  the spec as ``console-smoke-instance.json``; the spec re-proves them through
  the browser.
* The server PID is written to ``<home>/server.pid`` while it runs, so a
  supervisor (the CI job's ``always()`` step) can tear it down even when this
  process is killed before its own ``finally`` runs.

Cross-platform: ``npx`` works on Windows; ``uvicorn`` boots the same
way; ``subprocess.Popen(..., start_new_session=True)`` /
``CREATE_NEW_PROCESS_GROUP`` are wrapped by
:mod:`kestrel_sovereign._subprocess_helpers`. The ``lsof`` portability
gap (Windows lacks ``lsof``) is bridged by attempting a TCP connection
on the port: if anything answers, the port is busy.

Usage::

    kestrel demo run technical
    kestrel demo run spawn --port 9001
    kestrel demo run voice --keep-server   # leave the demo server up
    kestrel demo smoke                     # Console smoke in a fresh temp home
    kestrel demo smoke --home "$RUNNER_TEMP/console-smoke" --port 8910
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple

from kestrel_sovereign._subprocess_helpers import (
    run_streaming,
    start_background_process,
    stop_process,
    wait_for_health,
)
from kestrel_sovereign.paths import SKIP_DOTENV_ENV


# Ports the runner refuses to use. ``8888`` is the live server in the
# default multi_agent config; using it as DEMO_PORT would collide with
# (and tempt destructive ops against) live agents. Same intent as the
# bash predecessor's explicit refusal.
_FORBIDDEN_PORTS = frozenset({8888})

# Default demo port — must not collide with the live server's 8888.
# Matches the bash predecessor.
_DEFAULT_DEMO_PORT = 8900

# Default Console-smoke port: off 8888, and off 8900 so a smoke can run
# beside a demo.
_DEFAULT_SMOKE_PORT = 8910

# Seconds to wait for /health. A cold CI runner boots the server slower than
# a workstation, so the smoke allows more than a demo.
_DEMO_HEALTH_TIMEOUT = 60.0
_SMOKE_HEALTH_TIMEOUT = 120.0

# Files the smoke writes in its home. The manifest and PID file are the
# runner's contract with the Playwright spec and the CI teardown step.
SMOKE_MANIFEST_NAME = "console-smoke-instance.json"
SMOKE_PID_NAME = "server.pid"
SMOKE_LOG_NAME = "server.log"
_SMOKE_INCEPTION_NAME = "inception.json"

# The Playwright project that is the CI smoke subset.
_SMOKE_PLAYWRIGHT_CMD = (
    "npx", "playwright", "test",
    "--config", "tests/e2e/playwright.config.cjs",
    "--project", "console-smoke",
)

# Provider-key env vars preserved through to the demo's ``npx playwright``
# subprocess. ``KESTREL_API_KEY`` is deliberately NOT in this list — it's
# the production server's key and must not auth against the demo DB.
_PROVIDER_KEY_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "REPLICATE_API_TOKEN",
    "TAVILY_API_KEY",
    "RUNPOD_API_KEY",
    "OLLAMA_HOST",
)

# Variable names that look like a credential. The smoke strips every one of
# them, not only the providers listed above: an instance that must never
# use a production key is safer refusing all keys than enumerating them.
_CREDENTIAL_ENV_NAME = re.compile(r"(?:_KEY|_TOKEN|_SECRET|_PASSWORD)$")

# Resolves ``kestrel_sovereign`` without importing it, as the
# LIVE_AGENT_DOGFOODING runbook's origin check does.
_MODULE_ORIGIN_PROBE = (
    "import importlib.util, pathlib; "
    "print(pathlib.Path(importlib.util.find_spec('kestrel_sovereign').origin)"
    ".resolve().parent)"
)


class _SmokeSetupError(Exception):
    """A precondition the Console smoke refuses to run without."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Repo root (one level up from this package). Mirrors
    ``cli_verify_install._repo_root``."""
    return Path(__file__).resolve().parent.parent


def _list_demos(repo: Path) -> List[str]:
    """Return the names of every demo dir that has a ``config.cjs``.

    A demo is "runnable" iff its directory contains ``config.cjs`` —
    matches the bash predecessor's existence check.
    """
    demos_dir = repo / "demos"
    if not demos_dir.is_dir():
        return []
    out: List[str] = []
    for child in sorted(demos_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "config.cjs").is_file():
            out.append(child.name)
    return out


def _port_is_busy(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if ``host:port`` already has a listener.

    The bash predecessor used ``lsof -nP -iTCP:$DEMO_PORT -sTCP:LISTEN``
    which is not available on Windows. A bind-attempt to the port works
    everywhere: if SO_REUSEADDR-aware bind succeeds the port is free,
    if it raises ``OSError`` something already owns it.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return True
        return False
    finally:
        s.close()


def _port_refusal(port: int) -> Optional[str]:
    """Why ``port`` cannot host an isolated server, or None if it can."""
    if port in _FORBIDDEN_PORTS:
        return (
            f"refusing DEMO_PORT={port} — that's the live server. "
            "Pick another port."
        )
    if _port_is_busy(port):
        return (
            f"port {port} already in use; free it or pass "
            "--port <free-port>"
        )
    return None


def _load_dotenv_for_demo(repo: Path) -> dict:
    """Load ``<repo>/.env`` and return its values as a dict.

    The bash predecessor did ``source "$ROOT/.env"`` so LLM provider
    keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, etc.)
    flowed into both the demo server process and the Playwright runner.
    Codex review on PR #1071 caught that the Python port skipped this,
    so demos that depend on a key only set in .env failed silently
    until the operator manually exported it.

    ``interpolate=False`` matches the Tier 1.2 secrets-sync rule —
    keys are values, not templates; ``${VAR}`` shouldn't expand.
    Returns ``{}`` if .env isn't present (legitimate — operator may
    have keys exported in their shell already).
    """
    env_file = repo / ".env"
    if not env_file.exists():
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover — python-dotenv is a dep
        return {}
    raw = dotenv_values(str(env_file), interpolate=False)
    return {k: v for k, v in raw.items() if v is not None}


def _pin_demo_database_env(env: dict, demo_db: Path) -> None:
    """Keep every demo subprocess inside one disposable SQLite custody root."""

    env.pop("KESTREL_DATABASE_URL", None)
    env.pop("KESTREL_HOLD_EVIDENCE_DATABASE_URL", None)
    env.pop("KESTREL_HOLD_BACKEND", None)
    env["KESTREL_DB_BACKEND"] = "sqlite"
    env["KESTREL_DB_PATH"] = str(demo_db)
    env["KESTREL_HOST_DB_PATH"] = str(
        demo_db / "host-data" / "host-features.db"
    )


def _build_demo_env(parent_env: dict, demo_db: Path, repo: Path) -> dict:
    """Build the env for the ``uvicorn`` demo-server subprocess.

    Strips ``KESTREL_API_KEY`` (production key must not auth against
    the demo DB). Forces ``KESTREL_DB_PATH``, the multi_agent override,
    and the demo-server flag — both belt AND braces for ``#868``:

    * ``KESTREL_MULTI_AGENT_CONFIG`` points at a non-existent path so
      the server skips multi_agent loading (``server.py:201``).
    * ``KESTREL_DEMO_SERVER=1`` makes the security feature default to
      ALLOW (Playwright can't click modals) AND lets ``server.py``
      refuse multi_agent auto-load even if someone removes the override
      above.

    ``<repo>/.env`` is loaded UNDERNEATH the parent env (parent wins on
    collisions) so operators who already exported a key in their shell
    don't have it overridden by a stale .env value, but operators who
    only have keys in .env still get them through. Mirrors the bash's
    ``source "$ROOT/.env"`` behaviour without trampling explicit shell
    state.
    """
    env: dict = {}
    env.update(_load_dotenv_for_demo(repo))
    env.update(parent_env)
    env.pop("KESTREL_API_KEY", None)
    _pin_demo_database_env(env, demo_db)
    env["KESTREL_MULTI_AGENT_CONFIG"] = str(
        demo_db / "multi_agent-disabled.toml"
    )
    env["KESTREL_DEMO_SERVER"] = "1"
    return env


def _build_playwright_env(parent_env: dict, demo_url: str, repo: Path, demo_db: Path) -> dict:
    """Build the env for ``npx playwright test``.

    Strips ``KESTREL_API_KEY`` (the demo fetches its own key via
    ``/api/auth/key``); sets ``KESTREL_URL`` to the isolated demo
    server; sets ``KESTREL_DEMO_SERVER=1`` (some demo helpers branch
    on this — same flag the server reads); and sets ``KESTREL_DB_PATH``
    to the isolated demo sandbox so the demo can prove which directory
    is safe to reset (issue #1973 — ``resetDemoAgentDatabases`` refuses
    any path outside it). Provider keys come from ``<repo>/.env`` (loaded
    under parent_env so shell exports win on collision), preserving the
    bash predecessor's behaviour.
    """
    env: dict = {}
    env.update(_load_dotenv_for_demo(repo))
    env.update(parent_env)
    env.pop("KESTREL_API_KEY", None)
    # Make sure we explicitly carry every provider key forward, even if
    # the parent process scrubbed PATH-style enrichment. The dict is
    # already populated above; this list documents the intent for
    # future readers.
    for key in _PROVIDER_KEY_ENV:
        if key in parent_env and key not in env:
            env[key] = parent_env[key]
    env["KESTREL_URL"] = demo_url
    env["KESTREL_DEMO_SERVER"] = "1"
    _pin_demo_database_env(env, demo_db)
    return env


def _build_smoke_env(
    parent_env: dict, home: Path, data_dir: Path, data_key: str,
) -> dict:
    """Build the env shared by every Console-smoke subprocess.

    Nothing that configures another Kestrel install crosses over: every
    ``KESTREL_*`` variable is dropped and only the instance's own settings
    are set. Provider keys and every credential-shaped variable are dropped
    too, so the instance cannot reach a paid LLM or a production service.

    ``KESTREL_HOME`` is the fresh home, so the project resolver stays inside
    the instance. Removing a variable here is not enough on its own: the
    server loads its ``.env`` files with ``override=False``, which fills in
    exactly the variables that are absent, and one of those files sits next to
    the package source rather than in the home. ``KESTREL_SKIP_DOTENV`` tells
    the server to read none of them, so this mapping is the instance's whole
    environment.
    ``KESTREL_DATA_KEY`` is a throwaway key minted for this run, and the
    born-hybrid identity's did:web domain is ``localhost``, the same default
    ``kestrel setup --quickstart`` uses.
    ``KESTREL_SKIP_REACHABILITY_PROBE`` lets the local-only agent boot with no
    Ollama daemon, as the clean-install job does, and Phoenix is disabled so
    the instance neither starts a trace store nor binds its ports.
    """
    env = {
        name: value
        for name, value in parent_env.items()
        if not name.startswith("KESTREL_")
        and name not in _PROVIDER_KEY_ENV
        and not _CREDENTIAL_ENV_NAME.search(name)
    }
    _pin_demo_database_env(env, data_dir)
    env["KESTREL_HOME"] = str(home)
    env[SKIP_DOTENV_ENV] = "1"
    env["KESTREL_DATA_KEY"] = data_key
    env["KESTREL_DID_WEB_DOMAIN"] = "localhost"
    env["KESTREL_MULTI_AGENT_CONFIG"] = str(home / "multi_agent-disabled.toml")
    env["KESTREL_DEMO_SERVER"] = "1"
    env["KESTREL_SKIP_REACHABILITY_PROBE"] = "1"
    env["KESTREL_PHOENIX_ENABLED"] = "0"
    return env


def _ephemeral_data_key() -> str:
    """A fresh ``KESTREL_DATA_KEY`` that encrypts only this run's identity."""
    from kestrel_sdk.security.aead import AEADCipher

    return AEADCipher.generate_key().decode("ascii")


def _prepare_smoke_home(home: Optional[str]) -> Path:
    """Return a fresh, private instance home.

    With no ``home`` a new temp dir is created. A supplied path must be
    absent or empty: the smoke proves it is serving a *freshly created*
    instance, and it never deletes an operator's directory to make one.
    """
    if home is None:
        return Path(tempfile.mkdtemp(prefix="kestrel-console-smoke-")).resolve()
    path = Path(home).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise _SmokeSetupError(
            f"--home {path} already exists and is not empty; the smoke only "
            "runs in a freshly created home."
        )
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _load_inception_manifest(path: Path, data_dir: Path) -> dict:
    """Read the setup script's manifest and check it describes ``data_dir``."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise _SmokeSetupError(f"cannot read inception manifest {path}: {e}")
    did = manifest.get("agent_did")
    if not isinstance(did, str) or not did.startswith("did:"):
        raise _SmokeSetupError(f"inception manifest {path} has no agent DID")
    db_path = Path(str(manifest.get("db_path", ""))).resolve()
    if data_dir.resolve() not in db_path.parents:
        raise _SmokeSetupError(
            f"inception wrote its database to {db_path}, outside the smoke "
            f"data dir {data_dir}"
        )
    return manifest


def _server_module_origin(cwd: Path, env: dict) -> Path:
    """Where the server will import ``kestrel_sovereign`` from.

    Resolved by the same interpreter, working directory, and environment the
    server is launched with, so module search resolves identically.
    """
    result = subprocess.run(
        [sys.executable, "-c", _MODULE_ORIGIN_PROBE],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise _SmokeSetupError(
            "cannot resolve the server's kestrel_sovereign origin: "
            f"{result.stderr.strip() or 'no output'}"
        )
    return Path(result.stdout.strip().splitlines()[-1])


def _fetch_demo_api_key(demo_url: str) -> Optional[str]:
    """Mint/return the demo server's API key via the public ``/api/auth/key``
    endpoint (in ``server.py``'s ``public_paths``). ``/api/agents`` requires it."""
    try:
        with urllib.request.urlopen(f"{demo_url}/api/auth/key", timeout=5) as resp:
            return json.loads(resp.read()).get("key")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return None


def _fetch_agents(demo_url: str) -> Tuple[Optional[list], Optional[str]]:
    """Return ``(agents, None)`` from ``/api/agents``, or ``(None, error)``.

    ``/api/agents`` is authenticated, so mint the demo key first
    (``X-API-Key``); without it the server returns 401 and no check
    built on this could ever pass.
    """
    api_key = _fetch_demo_api_key(demo_url)
    request = urllib.request.Request(f"{demo_url}/api/agents")
    if api_key:
        request.add_header("X-API-Key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=5) as resp:
            body = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        return None, f"!!fetch-error: {e}"
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        return None, f"!!parse-error: {e}; body[:400]={body[:400]!r}"
    return data.get("agents") or [], None


def _verify_only_demo_agents(demo_url: str) -> Optional[str]:
    """Sanity-check ``/api/agents``: every loaded agent must report
    ``is_demo=true``. Returns None on success, else a string with
    the list of non-demo agents.

    This is acceptance-criterion #3 from ``#868``. The two upstream
    defences (``KESTREL_MULTI_AGENT_CONFIG`` override + the demo flag)
    should make this unreachable in practice, but the cost of a false
    negative is wiping a live agent — re-check at the boundary.
    """
    agents, error = _fetch_agents(demo_url)
    if error is not None:
        return error
    live = [
        a.get("name") or a.get("id") or "<unnamed>"
        for a in agents
        if a.get("is_demo") is not True
    ]
    if not live:
        return None
    return ",".join(live)


def _verify_served_identity(demo_url: str, expected_did: str) -> Optional[str]:
    """Check the server serves exactly the agent this run created.

    A DID is minted per inception, so a match proves the server is reading
    the freshly created database rather than any other. Returns None on
    success, else a description of what it serves instead.
    """
    agents, error = _fetch_agents(demo_url)
    if error is not None:
        return error
    served = [a.get("id") for a in agents]
    if served != [expected_did]:
        return (
            f"expected exactly the freshly created agent {expected_did}; "
            f"server reports {served}"
        )
    return None


@contextlib.contextmanager
def _sigterm_raises_exit() -> Iterator[None]:
    """Turn SIGTERM into ``SystemExit`` so ``finally`` teardown runs.

    SIGTERM (CI cancellation, ``kill``) otherwise ends the process without
    unwinding, leaving the isolated server running. Only the main thread
    may install a handler; anywhere else this is a no-op.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _exit(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, _exit)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _run_isolated_playwright(
    *,
    prefix: str,
    port: int,
    data_dir: Path,
    server_cwd: Path,
    server_env: dict,
    server_log: Path,
    health_timeout: float,
    playwright_cmd: List[str],
    playwright_cwd: Path,
    playwright_env: dict,
    keep_server: bool,
    pid_file: Optional[Path] = None,
    verify_instance: Optional[Callable[[str, Any], Optional[str]]] = None,
) -> int:
    """Start an isolated server, prove it, run Playwright, tear it down.

    The lifecycle ``kestrel demo run`` and ``kestrel demo smoke`` share:
    start uvicorn on ``port``, wait for ``/health``, refuse any non-demo
    agent (#868), run the caller's ``verify_instance`` (an error string
    refuses), then run Playwright. The server is stopped on every exit —
    success, failure, an exception, or SIGTERM — unless ``keep_server``.
    """
    demo_url = f"http://127.0.0.1:{port}"
    # Use sys.executable -m uvicorn so we don't depend on whether the
    # operator has ``uvicorn`` on PATH — matches the in-process startup
    # idiom in cli.py:_start_inprocess_mode.
    server_cmd = [
        sys.executable, "-m", "uvicorn", "kestrel_sovereign.server:app",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    with _sigterm_raises_exit():
        log_fd = open(server_log, "wb")
        print(
            f"[{prefix}] Starting isolated server on {demo_url} "
            f"(DB={data_dir}) ..."
        )
        proc = start_background_process(
            server_cmd,
            cwd=server_cwd,
            env=server_env,
            stdout=log_fd,
            stderr=log_fd,
        )
        exit_code = 1
        try:
            if pid_file is not None:
                pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")

            print(f"[{prefix}] Waiting for {demo_url}/health ...", flush=True)
            if not wait_for_health(port, timeout=health_timeout, proc=proc):
                print(
                    "error: server did not become healthy within "
                    f"{health_timeout:.0f}s. Log: {server_log}",
                    file=sys.stderr,
                )
                return exit_code

            # Routing precondition (#868 AC#3).
            print(f"[{prefix}] Verifying every loaded agent is is_demo=true ...")
            bad = _verify_only_demo_agents(demo_url)
            if bad is not None:
                print(
                    "error: refusing to run — server reports non-demo "
                    f"agent(s): {bad}\n"
                    "       This is the routing precondition that wiped "
                    "Meridian (#867/#868).",
                    file=sys.stderr,
                )
                return exit_code

            if verify_instance is not None:
                problem = verify_instance(demo_url, proc)
                if problem is not None:
                    print(f"error: refusing to run — {problem}", file=sys.stderr)
                    return exit_code

            print(f"[{prefix}] Running {' '.join(playwright_cmd)} ...")
            exit_code = run_streaming(
                playwright_cmd,
                cwd=playwright_cwd,
                env=playwright_env,
            )
            print(f"[{prefix}] Done (exit={exit_code}).")
            return exit_code
        finally:
            log_fd.close()
            if not keep_server:
                print(
                    f"[{prefix}] Stopping server (PID {proc.pid}) ...",
                    flush=True,
                )
                stop_process(proc)
                if pid_file is not None:
                    pid_file.unlink(missing_ok=True)
                if exit_code != 0:
                    print(
                        f"[{prefix}] Server log: {server_log}",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"[{prefix}] --keep-server set; leaving uvicorn at "
                    f"PID {proc.pid} ({demo_url}). Stop it manually when "
                    "done.",
                    file=sys.stderr,
                )


# ---------------------------------------------------------------------------
# Subverb handlers
# ---------------------------------------------------------------------------

def _cmd_demo_run(args) -> int:
    """``kestrel demo run <name>`` — full demo lifecycle."""
    repo = _repo_root()
    demos = _list_demos(repo)
    name: str = args.name
    port: int = int(args.port) if args.port is not None else _DEFAULT_DEMO_PORT
    keep_server: bool = bool(getattr(args, "keep_server", False))

    if name not in demos:
        print(
            f"error: demo {name!r} not found; available: "
            f"{demos or '(none)'}",
            file=sys.stderr,
        )
        return 2

    demo_dir = repo / "demos" / name
    if not (demo_dir / "config.cjs").is_file():
        # Defensive — _list_demos only returns dirs that already had
        # config.cjs, but keep the error path so a race (someone
        # deleted config.cjs mid-run) still produces a clean message.
        print(
            f"error: demos/{name}/config.cjs missing",
            file=sys.stderr,
        )
        return 2

    refusal = _port_refusal(port)
    if refusal is not None:
        print(f"error: {refusal}", file=sys.stderr)
        return 2

    demo_url = f"http://127.0.0.1:{port}"
    demo_db = repo / "agent_data" / "demo"

    # 1. Setup demo agent DB.
    print(f"[demo-runner] Creating fresh demo agent DB at {demo_db} ...")
    rc = run_streaming(
        [sys.executable, str(repo / "scripts" / "setup_demo_agent.py")],
        cwd=repo,
    )
    if rc != 0:
        print(
            "error: scripts/setup_demo_agent.py failed; aborting.",
            file=sys.stderr,
        )
        return 1

    # 2-5. Isolated server → health → isolation check → demo → teardown.
    rc = _run_isolated_playwright(
        prefix="demo-runner",
        port=port,
        data_dir=demo_db,
        server_cwd=repo,
        server_env=_build_demo_env(os.environ.copy(), demo_db, repo),
        server_log=Path(tempfile.gettempdir()) / f"kestrel-demo-server-{port}.log",
        health_timeout=_DEMO_HEALTH_TIMEOUT,
        playwright_cmd=["npx", "playwright", "test", "--config=config.cjs"],
        playwright_cwd=demo_dir,
        playwright_env=_build_playwright_env(os.environ.copy(), demo_url, repo, demo_db),
        keep_server=keep_server,
    )
    print(f"[demo-runner] Artifacts in demos/{name}/demo-output/")
    return rc


def _cmd_demo_smoke(args) -> int:
    """``kestrel demo smoke`` — the Sovereign Console Playwright smoke."""
    repo = _repo_root()
    port: int = int(args.port) if args.port is not None else _DEFAULT_SMOKE_PORT
    keep_server: bool = bool(getattr(args, "keep_server", False))

    refusal = _port_refusal(port)
    if refusal is not None:
        print(f"error: {refusal}", file=sys.stderr)
        return 2
    try:
        home = _prepare_smoke_home(getattr(args, "home", None))
    except _SmokeSetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    data_dir = home / "agent_data" / "console-smoke"
    inception_path = home / _SMOKE_INCEPTION_NAME
    manifest_path = home / SMOKE_MANIFEST_NAME
    demo_url = f"http://127.0.0.1:{port}"
    env = _build_smoke_env(os.environ.copy(), home, data_dir, _ephemeral_data_key())

    # 1. A fresh demo agent, configured with the local LLM route only.
    print(f"[console-smoke] Creating a fresh smoke agent in {home} ...")
    rc = run_streaming(
        [
            sys.executable, str(repo / "scripts" / "setup_demo_agent.py"),
            "--data-dir", str(data_dir),
            "--manifest", str(inception_path),
            "--local-llm-only",
        ],
        cwd=home,
        env=env,
    )
    if rc != 0:
        print(
            "error: scripts/setup_demo_agent.py failed; aborting.",
            file=sys.stderr,
        )
        return 1

    # 2. Prove the server will run this checkout's code.
    try:
        inception = _load_inception_manifest(inception_path, data_dir)
        module_origin = _server_module_origin(home, env)
    except _SmokeSetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    expected_origin = Path(__file__).resolve().parent
    if module_origin != expected_origin:
        print(
            "error: refusing to run — the server would import "
            f"kestrel_sovereign from {module_origin}, not from this "
            f"runner's {expected_origin}",
            file=sys.stderr,
        )
        return 1
    print(f"[console-smoke] Server module origin: {module_origin}")

    def _record_instance(url: str, proc: Any) -> Optional[str]:
        # 3. Prove the server serves this run's database, then hand the
        # proven facts to the spec.
        problem = _verify_served_identity(url, inception["agent_did"])
        if problem is not None:
            return problem
        manifest = {
            "base_url": url,
            "port": port,
            "home": str(home),
            "data_dir": str(data_dir),
            "db_path": inception["db_path"],
            "agent_did": inception["agent_did"],
            "agent_name": inception.get("agent_name"),
            "module_origin": str(module_origin),
            "server_pid": proc.pid,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[console-smoke] Serving {inception['agent_did']} from "
            f"{inception['db_path']}"
        )
        return None

    playwright_env = {
        name: value for name, value in env.items() if name != "KESTREL_DATA_KEY"
    }
    playwright_env["KESTREL_URL"] = demo_url
    playwright_env["KESTREL_CONSOLE_SMOKE_MANIFEST"] = str(manifest_path)

    rc = _run_isolated_playwright(
        prefix="console-smoke",
        port=port,
        data_dir=data_dir,
        server_cwd=home,
        server_env=env,
        server_log=home / SMOKE_LOG_NAME,
        health_timeout=_SMOKE_HEALTH_TIMEOUT,
        playwright_cmd=list(_SMOKE_PLAYWRIGHT_CMD),
        playwright_cwd=repo,
        playwright_env=playwright_env,
        keep_server=keep_server,
        pid_file=home / SMOKE_PID_NAME,
        verify_instance=_record_instance,
    )
    print(f"[console-smoke] Instance home: {home}")
    return rc


# ---------------------------------------------------------------------------
# Argparse subcommand wiring
# ---------------------------------------------------------------------------

def add_demo_subcommand(
    subparsers: "argparse._SubParsersAction",
) -> None:
    """Register ``kestrel demo {run,smoke}`` under the parent subparsers.

    Called from :func:`kestrel_sovereign.cli.build_parser`. Mirrors the
    ``cli_release`` / ``cli_deploy`` locality pattern so an operator
    who never runs demos doesn't pay for the import.
    """
    demo_p = subparsers.add_parser(
        "demo",
        help="Run a Kestrel demo against an isolated demo agent — "
             "port of demos/run.sh (epic #1050 tier 3)",
    )
    demo_sub = demo_p.add_subparsers(dest="demo_command")

    run_p = demo_sub.add_parser(
        "run",
        help="Run a demo by name (e.g. `kestrel demo run technical`)",
    )
    run_p.add_argument(
        "name",
        help="Demo name — must match a directory under demos/ that "
             "contains config.cjs (e.g. technical, spawn, trash)",
    )
    run_p.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port for the isolated demo server (default: "
             f"{_DEFAULT_DEMO_PORT}; refuses 8888)",
    )
    run_p.add_argument(
        "--keep-server",
        action="store_true",
        help="Skip the EXIT-trap teardown so the operator can poke "
             "around the demo server afterward. Default: stop server "
             "on exit.",
    )

    smoke_p = demo_sub.add_parser(
        "smoke",
        help="Run the Sovereign Console Playwright smoke (the CI subset) "
             "against a freshly created isolated instance",
    )
    smoke_p.add_argument(
        "--home",
        default=None,
        help="Instance home; must be absent or empty (default: a new temp "
             "dir). Holds the database, server.log, and the instance "
             "manifest.",
    )
    smoke_p.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port for the isolated server (default: "
             f"{_DEFAULT_SMOKE_PORT}; refuses 8888)",
    )
    smoke_p.add_argument(
        "--keep-server",
        action="store_true",
        help="Leave the server running after the smoke. Default: stop it.",
    )


# ---------------------------------------------------------------------------
# Top-level handler
# ---------------------------------------------------------------------------

def cmd_demo(args) -> int:
    """Dispatch ``kestrel demo ...``.

    Exit codes:
        0 — demo/smoke passed
        1 — demo/smoke failed (setup failed, server unhealthy, isolation or
            origin check failed, playwright reported failure)
        2 — argument error (unknown demo, forbidden port, port busy,
            non-empty smoke home)
    """
    sub = getattr(args, "demo_command", None)
    if sub == "run":
        return _cmd_demo_run(args)
    if sub == "smoke":
        return _cmd_demo_smoke(args)
    print(
        "Usage: kestrel demo run <name> [--port PORT] [--keep-server]\n"
        "       kestrel demo smoke [--home DIR] [--port PORT] [--keep-server]",
        file=sys.stderr,
    )
    return 1


__all__ = [
    "SMOKE_LOG_NAME",
    "SMOKE_MANIFEST_NAME",
    "SMOKE_PID_NAME",
    "add_demo_subcommand",
    "cmd_demo",
]
