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

Cross-platform: ``npx`` works on Windows; ``uvicorn`` boots the same
way; ``subprocess.Popen(..., start_new_session=True)`` /
``CREATE_NEW_PROCESS_GROUP`` are wrapped by
:mod:`kestrel_sovereign._subprocess_helpers`. The ``lsof`` portability
gap (Windows lacks ``lsof``) is bridged by attempting a TCP connection
on the port: if anything answers, the port is busy.

Usage::

    kestrel demo run technical
    kestrel demo run spawn --port 9001
    kestrel demo run trash --keep-server   # leave the demo server up
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from kestrel_sovereign._subprocess_helpers import (
    run_streaming,
    start_background_process,
    stop_process,
    wait_for_health,
)


# Ports the runner refuses to use. ``8888`` is the live server in the
# default multi_agent config; using it as DEMO_PORT would collide with
# (and tempt destructive ops against) live agents. Same intent as the
# bash predecessor's explicit refusal.
_FORBIDDEN_PORTS = frozenset({8888})

# Default demo port — must not collide with the live server's 8888.
# Matches the bash predecessor.
_DEFAULT_DEMO_PORT = 8900

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


def _build_demo_env(parent_env: dict, demo_db: Path) -> dict:
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
    """
    env = dict(parent_env)
    env.pop("KESTREL_API_KEY", None)
    env["KESTREL_DB_PATH"] = str(demo_db)
    env["KESTREL_MULTI_AGENT_CONFIG"] = str(
        demo_db / "multi_agent-disabled.toml"
    )
    env["KESTREL_DEMO_SERVER"] = "1"
    return env


def _build_playwright_env(parent_env: dict, demo_url: str) -> dict:
    """Build the env for ``npx playwright test``.

    Strips ``KESTREL_API_KEY`` (the demo fetches its own key via
    ``/api/auth/key``); sets ``KESTREL_URL`` to the isolated demo
    server; sets ``KESTREL_DEMO_SERVER=1`` (some demo helpers branch
    on this — same flag the server reads). Provider keys explicitly
    survive so the demo agent can actually call an LLM.
    """
    env = {k: v for k, v in parent_env.items() if k != "KESTREL_API_KEY"}
    # Make sure we explicitly carry every provider key forward, even if
    # the parent process scrubbed PATH-style enrichment. ``parent_env``
    # already has them post-``os.environ.copy()`` but this list documents
    # the intent for future readers.
    for key in _PROVIDER_KEY_ENV:
        if key in parent_env:
            env[key] = parent_env[key]
    env["KESTREL_URL"] = demo_url
    env["KESTREL_DEMO_SERVER"] = "1"
    return env


def _verify_only_demo_agents(demo_url: str) -> Optional[str]:
    """Sanity-check ``/api/agents``: every loaded agent must report
    ``is_demo=true``. Returns None on success, else a string with
    the list of non-demo agents.

    This is acceptance-criterion #3 from ``#868``. The two upstream
    defences (``KESTREL_MULTI_AGENT_CONFIG`` override + the demo flag)
    should make this unreachable in practice, but the cost of a false
    negative is wiping a live agent — re-check at the boundary.
    """
    url = f"{demo_url}/api/agents"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        return f"!!fetch-error: {e}"
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        return f"!!parse-error: {e}; body[:400]={body[:400]!r}"
    agents = data.get("agents") or []
    live = [
        a.get("name") or a.get("id") or "<unnamed>"
        for a in agents
        if a.get("is_demo") is not True
    ]
    if not live:
        return None
    return ",".join(live)


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

    if port in _FORBIDDEN_PORTS:
        print(
            f"error: refusing DEMO_PORT={port} — that's the live "
            "server. Pick another port.",
            file=sys.stderr,
        )
        return 2

    if _port_is_busy(port):
        print(
            f"error: port {port} already in use; free it or pass "
            f"--port <free-port>",
            file=sys.stderr,
        )
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

    # 2. Spawn isolated uvicorn against demo DB; logs go to a tempfile
    # we can ``tail``-print on health-check failure.
    server_log = Path(tempfile.gettempdir()) / f"kestrel-demo-server-{port}.log"
    log_fd = open(server_log, "wb")
    server_env = _build_demo_env(os.environ.copy(), demo_db)
    # Use sys.executable -m uvicorn so we don't depend on whether the
    # operator has ``uvicorn`` on PATH — matches the in-process startup
    # idiom in cli.py:_start_inprocess_mode.
    server_cmd = [
        sys.executable, "-m", "uvicorn", "server:app",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    print(
        f"[demo-runner] Starting isolated server on {demo_url} "
        f"(DB={demo_db}) ..."
    )
    proc = start_background_process(
        server_cmd,
        cwd=repo,
        env=server_env,
        stdout=log_fd,
        stderr=log_fd,
    )

    exit_code = 0
    try:
        # 3. Wait for /health.
        print(
            f"[demo-runner] Waiting for {demo_url}/health ...",
            flush=True,
        )
        if not wait_for_health(port, timeout=60.0, proc=proc):
            print(
                "error: server did not become healthy within 60s. "
                f"Log: {server_log}",
                file=sys.stderr,
            )
            return 1

        # 4. Verify routing precondition (#868 AC#3).
        print(
            "[demo-runner] Verifying every loaded agent is is_demo=true ..."
        )
        bad = _verify_only_demo_agents(demo_url)
        if bad is not None:
            print(
                "error: refusing to run — server reports non-demo "
                f"agent(s): {bad}\n"
                "       This is the routing precondition that wiped "
                "Meridian (#867/#868).",
                file=sys.stderr,
            )
            return 1

        # 5. Run the demo via npx.
        print(f"[demo-runner] Running demos/{name} ...")
        playwright_env = _build_playwright_env(os.environ.copy(), demo_url)
        rc = run_streaming(
            ["npx", "playwright", "test", "--config=config.cjs"],
            cwd=demo_dir,
            env=playwright_env,
        )
        exit_code = rc
        print(
            f"[demo-runner] Done (exit={rc}). Artifacts in "
            f"demos/{name}/demo-output/"
        )
        return rc
    finally:
        log_fd.close()
        if not keep_server:
            print(
                f"[demo-runner] Stopping server (PID {proc.pid}) ...",
                flush=True,
            )
            stop_process(proc)
            if exit_code != 0:
                print(
                    f"[demo-runner] Server log: {server_log}",
                    file=sys.stderr,
                )
        else:
            print(
                f"[demo-runner] --keep-server set; leaving uvicorn at "
                f"PID {proc.pid} ({demo_url}). Stop it manually when "
                "done.",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Argparse subcommand wiring
# ---------------------------------------------------------------------------

def add_demo_subcommand(
    subparsers: "argparse._SubParsersAction",
) -> None:
    """Register ``kestrel demo {run}`` under the parent subparsers.

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


# ---------------------------------------------------------------------------
# Top-level handler
# ---------------------------------------------------------------------------

def cmd_demo(args) -> int:
    """Dispatch ``kestrel demo ...``.

    Exit codes:
        0 — demo passed
        1 — demo failed (server unhealthy, sanity-check failed,
            playwright reported failure)
        2 — argument error (unknown demo, forbidden port, port busy)
    """
    sub = getattr(args, "demo_command", None)
    if sub == "run":
        return _cmd_demo_run(args)
    print(
        "Usage: kestrel demo run <name> [--port PORT] [--keep-server]",
        file=sys.stderr,
    )
    return 1


__all__ = [
    "add_demo_subcommand",
    "cmd_demo",
]
