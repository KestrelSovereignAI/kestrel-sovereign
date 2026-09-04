"""Kestrel CLI — host/agent lifecycle commands.

``start``, ``stop``, ``restart``, ``update``, ``status``, ``logs``. Extracted
from ``cli.py`` (#1678) following the ``cli_<group>.py`` convention. Shared,
test-patched helpers (``_get_project_dir``, ``MultiAgentConfig``, and the
``update`` git/uv helpers patched via ``cli.<name>``) are reached through the
``cli`` module at call time so existing ``patch("kestrel_sovereign.cli.*")``
seams keep working unchanged.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import quote

from kestrel_sovereign.multi_agent.config import MULTI_AGENT_CONFIG_FILENAME
from kestrel_sovereign.multi_agent.process_manager import (
    DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS,
    PidStatus,
    ProcessManager,
)


def _positive_seconds(value: str) -> float:
    """Argparse type for a strictly-positive duration in seconds."""
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return seconds


def _startup_timeout(args) -> float:
    """Resolve the shared lifecycle readiness timeout from parsed args."""
    return float(
        getattr(
            args,
            "startup_timeout",
            DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS,
        )
    )


def _format_seconds(seconds: float) -> str:
    return f"{seconds:g}s"


# Reanchor runbook surfaced when a readiness timeout traces back to
# constitution safe mode (#2616/#2618).
_REANCHOR_RUNBOOK = "docs/architecture/security/SOVEREIGN_TRUST_ROOT.md"


def _probe_health_status(port: int) -> Optional[int]:
    """One-shot GET /health. Returns the HTTP status code when anything is
    answering on the port (200, 503, ...), or ``None`` when nothing is."""
    import urllib.error
    import urllib.request

    url = f"http://localhost:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _fetch_detailed_health(port: int, api_key: str) -> Optional[dict]:
    """Authenticated GET /health/detailed. Returns the parsed JSON payload,
    or ``None`` when the key is rejected or the request/parse fails.

    A restricted host answers 503 with the diagnostic body, which urllib
    raises as ``HTTPError`` — the body still has to be read from it.
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"http://localhost:{port}/health/detailed",
        headers={"X-API-Key": api_key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return None
        try:
            body = exc.read()
        except (OSError, ValueError):
            return None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _diagnose_unready_server(port: int, env: dict) -> Optional[list]:
    """Explain a readiness-poll timeout when the server IS responding.

    After ``wait_for_health`` gives up, the port may still be serving
    ``/health`` — most notably held at 503 because an agent booted into
    constitution safe mode (#2616/#2618). The public payload deliberately
    hides the cause (#2629 anti-fingerprinting), so the cause is read from
    the authenticated ``/health/detailed`` using the locally-resolved
    ``KESTREL_API_KEY`` (the CLI runs as the operator and already loads the
    project ``.env``). Returns display lines, or ``None`` when nothing is
    answering on the port — the caller keeps the generic timeout message.
    """
    status_code = _probe_health_status(port)
    if status_code is None:
        return None
    if status_code == 200:
        return [
            "/health now reports HTTP 200 — the server finished "
            "initializing just after the deadline. Check `kestrel status`.",
        ]
    api_key = (env.get("KESTREL_API_KEY") or "").strip()
    if not api_key:
        return [
            f"/health is responding (HTTP {status_code}) — the server "
            "process is up but held not-ready.",
            "No KESTREL_API_KEY resolved locally; query the authenticated "
            "/health/detailed for the cause (e.g. constitution safe mode). "
            f"Reanchor runbook: {_REANCHOR_RUNBOOK}",
        ]
    payload = _fetch_detailed_health(port, api_key)
    if payload is None:
        return [
            f"/health is responding (HTTP {status_code}) — the server "
            "process is up but held not-ready.",
            "/health/detailed could not be read with the local "
            "KESTREL_API_KEY (rejected or errored) — query it manually. "
            f"Reanchor runbook: {_REANCHOR_RUNBOOK}",
        ]
    lines = [
        f"/health is responding (HTTP {status_code}); /health/detailed "
        f"reports status: {payload.get('status', 'unknown')}",
    ]
    records = payload.get("constitution_safe_mode") or []
    if records:
        lines.append("Constitution safe mode:")
        for record in records:
            if not isinstance(record, dict):
                continue
            lines.append(
                f"  - {record.get('agent', '?')}: "
                f"{record.get('failure', '?')} "
                f"({record.get('error_code', '?')})"
            )
        lines.append(f"Reanchor runbook: {_REANCHOR_RUNBOOK}")
    elif payload.get("error"):
        lines.append(f"Detail: {payload['error']}")
    return lines


def _add_startup_timeout_argument(parser: argparse.ArgumentParser) -> None:
    """Add the canonical readiness deadline to a lifecycle subcommand."""
    parser.add_argument(
        "--startup-timeout",
        type=_positive_seconds,
        default=DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "Seconds to wait for /health after start/restart "
            f"(default: {DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS:g})"
        ),
    )


def _format_uptime(pid: int) -> str:
    """Get uptime string for a running process."""
    try:
        import psutil
        p = psutil.Process(pid)
        elapsed = time.time() - p.create_time()
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    except Exception:
        return "-"


# ---------------------------------------------------------------------------
# Host PID / log file paths
# ---------------------------------------------------------------------------

def _host_pid_file(project_dir: Optional[Path] = None) -> Path:
    """PID file for the host process."""
    if project_dir is None:
        project_dir = cli._get_project_dir()
    logs_dir = project_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / ".host.pid"


def _host_log_file(project_dir: Optional[Path] = None) -> Path:
    """Log file for the host process."""
    if project_dir is None:
        project_dir = cli._get_project_dir()
    logs_dir = project_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "host.log"


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_start(args) -> int:
    """Start host and/or agents."""
    project_dir = cli._get_project_dir()

    first_run_rc = cli._maybe_first_run_setup(project_dir)
    if first_run_rc is not None:
        return first_run_rc

    multi_agent = cli.MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)
    pm = ProcessManager(project_dir)
    startup_timeout = _startup_timeout(args)

    if args.name:
        # Start a single agent by name
        local_agents = multi_agent.get_local_agents()
        if args.name not in local_agents:
            print(f"Agent '{args.name}' not found in multi_agent config")
            print(f"Available agents: {', '.join(local_agents.keys()) or '(none)'}")
            return 1

        agent_cfg = local_agents[args.name]
        print(f"   Starting {args.name} on :{agent_cfg.port}...", end="", flush=True)
        try:
            pm.start_agent(args.name, agent_cfg, multi_agent.host.bind, standalone=True)
        except RuntimeError as e:
            print("          \u274c")
            print(f"   {e}")
            return 1

        if pm.wait_for_health(agent_cfg.port, timeout=startup_timeout):
            print("          \u2705")
        else:
            print("          \u274c")
            log_file = ProcessManager.agent_log_file(
                (project_dir / agent_cfg.data_dir).resolve()
            )
            detail_lines = _diagnose_unready_server(
                agent_cfg.port, pm._load_env()
            )
            if detail_lines is None:
                print(
                    f"   {args.name} did not become healthy within "
                    f"{_format_seconds(startup_timeout)}; the process may "
                    "still be initializing."
                )
            else:
                print(
                    f"   {args.name} did not become healthy within "
                    f"{_format_seconds(startup_timeout)}:"
                )
                for line in detail_lines:
                    print(f"   {line}")
            print(f"   Check log: {log_file}")
            return 1
        return 0

    return _start_inprocess_mode(
        project_dir,
        multi_agent,
        pm,
        startup_timeout=startup_timeout,
    )


def _start_inprocess_mode(
    project_dir: Path,
    multi_agent,
    pm: ProcessManager,
    *,
    startup_timeout: float = DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS,
) -> int:
    """Start all agents in a single server process (default mode)."""
    autostart = multi_agent.get_autostart_agents()
    manual = {
        name: cfg for name, cfg in multi_agent.get_local_agents().items()
        if not cfg.autostart
    }

    # Credential material below governs only a future launch. A verified live
    # PID already identifies the serving process; reinterpreting it through a
    # rotated or removed project key would make idempotent ``start`` fail or
    # print a credential that the running host cannot accept.
    host_pid_file = _host_pid_file(project_dir)
    existing = pm.read_pid_record(host_pid_file)
    if existing.is_running:
        print(f"   Server already running (PID: {existing.pid})")
        return 0

    # Keyed on the status, not on whether a PID came back. ``read_pid``
    # withholds a stale number by design, so testing its result for
    # truthiness silently stopped clearing exactly the records that most need
    # clearing — and a leftover legacy file flips from stale to undecidable
    # the moment its number is reused, after which stop would signal the
    # replacement (#2995).
    if existing.needs_cleanup:
        pm.clear_pid(host_pid_file)

    # An API-key fleet cannot mint a sovereign credential at runtime: every
    # managed child can reach the same loopback bootstrap route as a browser.
    # Setup normally provisions this value in the project .env. OAuth-required
    # fleets already have a complete operator-authentication lane and do not
    # need a parallel API key.
    env = pm._load_env()
    from kestrel_sovereign.auth import (
        normalize_api_key,
        required_oauth_is_configured,
    )

    host_api_key = normalize_api_key(env.get("KESTREL_API_KEY")) or ""
    oauth_required = env.get("KESTREL_REQUIRE_OAUTH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    keyless_oauth_configured = required_oauth_is_configured(env)
    if not host_api_key.strip() and not keyless_oauth_configured:
        if oauth_required:
            print(
                "❌ Keyless multi-agent OAuth requires configured Google OAuth "
                "credentials (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET). "
                "Otherwise run `kestrel setup keys` and retry."
            )
            return 1
        print(
            "❌ Multi-agent host requires a stable KESTREL_API_KEY in the "
            "project .env. Run `kestrel setup keys` and retry."
        )
        return 1
    browser_url = f"http://localhost:{multi_agent.host.port}/"
    if host_api_key:
        browser_url += f"#key={quote(host_api_key, safe='')}"

    print("\U0001F985 Kestrel MultiAgent starting (in-process)...")
    # The fragment is never sent in an HTTP request or access log. app.js
    # consumes it into sessionStorage and strips it from the address bar.
    print(f"   URL:      {browser_url}")

    if autostart or manual:
        print("   Agents:")
        for name, cfg in autostart.items():
            resolved = (project_dir / cfg.data_dir).resolve()
            print(f"     {name:12} {resolved}/       autostart")
        for name, cfg in manual.items():
            resolved = (project_dir / cfg.data_dir).resolve()
            print(f"     {name:12} {resolved}/       manual")
    print()

    if pm.is_port_in_use(multi_agent.host.port, multi_agent.host.bind):
        orphans = pm.find_pids_on_port(multi_agent.host.port)
        print(f"   Port {multi_agent.host.port} already in use"
              + (f" by PID(s) {orphans}" if orphans else ""))
        print("   Run: kestrel terminate   (add --force if it doesn't die)")
        return 1

    env["PORT"] = str(multi_agent.host.port)
    env["KESTREL_MULTI_AGENT"] = "true"
    env["KESTREL_SERVE_UI"] = "true"

    log_file = _host_log_file(project_dir)
    cmd = [sys.executable, "-m", "uvicorn", "kestrel_sovereign.server:app",
           "--host", multi_agent.host.bind, "--port", str(multi_agent.host.port)]

    print(f"   Starting server on :{multi_agent.host.port}...", end="", flush=True)
    # In-process host: ``kestrel start`` is fire-and-exit (returns
    # after wait_for_health). The pipe+pump model in ``_spawn``
    # requires the parent to keep running for the daemon pump
    # thread to survive — so this launcher uses the detached
    # variant that hands the log file's fd straight to the child.
    # Without this, runtime INFO/WARNING/ERROR lines from the
    # uvicorn host hit a closed pipe (EPIPE) once the launcher
    # exits and are silently swallowed; host.log appears to freeze
    # after startup and runtime tracebacks vanish. See #1461.
    pm._spawn_detached(
        cmd, env, log_file, host_pid_file, port=multi_agent.host.port
    )

    if pm.wait_for_health(multi_agent.host.port, timeout=startup_timeout):
        print("          \u2705")
    else:
        print("          \u274c")
        detail_lines = _diagnose_unready_server(multi_agent.host.port, env)
        if detail_lines is None:
            print(
                "   Server did not become healthy within "
                f"{_format_seconds(startup_timeout)}; the process may still "
                "be initializing."
            )
        else:
            print(
                "   Server did not become healthy within "
                f"{_format_seconds(startup_timeout)}:"
            )
            for line in detail_lines:
                print(f"   {line}")
        print(f"   Check log: {log_file}")
        return 1

    print(f"\n\U0001F985 MultiAgent ready: {browser_url}")
    return 0


class PortReapResult(Enum):
    """Outcome of reaping untracked listeners on a port.

    Three outcomes rather than a bool, because the caller asks two different
    questions — "was anything listening?" and "is the port free now?" — and a
    single flag can only answer one of them. Collapsing them is what let
    ``stop`` report a clean shutdown over a listener it never dislodged
    (#2990).
    """

    NOTHING_FOUND = "nothing_found"
    RELEASED = "released"
    STILL_HELD = "still_held"


def _await_port_release(port: int, bind: str, attempts: int = 10) -> bool:
    """Poll until `bind:port` can be bound, returning whether it was released."""
    for _ in range(attempts):
        if not ProcessManager.is_port_in_use(port, bind):
            return True
        time.sleep(0.3)
    return not ProcessManager.is_port_in_use(port, bind)


def _reap_orphans_on_port(
    port: int, label: str, force: bool, bind: str = "0.0.0.0"
) -> PortReapResult:
    """Kill untracked listeners on `port` and report whether it was freed.

    The result is decided by re-probing the port, never by the fact that a
    signal was sent: a listener owned by another user cannot be signalled at
    all (``os.kill`` raises ``PermissionError``), and a delivered signal is not
    proof the listener honoured it. The operator's next ``kestrel start`` needs
    the port, so the port is the postcondition worth checking.
    """
    orphans = ProcessManager.find_pids_on_port(port)
    if not orphans:
        # An empty list is NOT proof the port is free. ``find_pids_on_port``
        # returns [] on any discovery failure — psutil absent, a parse error,
        # or a listener owned by another user this process cannot enumerate —
        # which is precisely the unkillable listener this function exists to
        # catch. Ask the port before concluding there was nothing here.
        if ProcessManager.is_port_in_use(port, bind):
            return PortReapResult.STILL_HELD
        return PortReapResult.NOTHING_FOUND
    print(f"   {label}: orphan listener(s) on :{port} {orphans} — killing")
    for opid in orphans:
        ProcessManager.kill_process(opid, force=force)
    if _await_port_release(port, bind):
        return PortReapResult.RELEASED
    for opid in orphans:
        ProcessManager.kill_process(opid, force=True)
    if _await_port_release(port, bind):
        return PortReapResult.RELEASED
    return PortReapResult.STILL_HELD


def _report_port_still_held(label: str, port: int) -> None:
    """Explain a port that survived SIGKILL, and how to find its owner."""
    print(
        f"   {label}: port :{port} is still in use — "
        f"not reporting {label} as terminated"
    )
    # A remediation naming a tool the platform does not ship is not a
    # remediation. Windows is a supported target and has no lsof.
    if sys.platform == "win32":
        print(f"   Identify the owner with: netstat -ano | findstr :{port}")
    else:
        print(f"   Identify the owner with: lsof -nP -iTCP:{port} -sTCP:LISTEN")


def cmd_terminate(args) -> int:
    """Terminate host and/or agent processes."""
    project_dir = cli._get_project_dir()
    multi_agent = cli.MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)
    pm = ProcessManager(project_dir)
    force = getattr(args, "force", False)

    if args.name:
        # Terminate a single agent process.
        local_agents = multi_agent.get_local_agents()
        if args.name not in local_agents:
            print(f"Agent '{args.name}' not found in multi_agent config")
            return 1

        agent_cfg = local_agents[args.name]
        pm.register_agent(args.name, agent_cfg)
        ap = pm._agents.get(args.name)
        if ap and ap.pid:
            print(f"   Terminating {args.name} (PID: {ap.pid})...")
            if not pm.terminate_agent(args.name):
                print(
                    f"   {args.name}: PID {ap.pid} is still running after "
                    f"SIGKILL — not reporting {args.name} as terminated"
                )
                return 1
            # The tracked PID being gone does not mean the port is free: a
            # supervisor may already have rebound it under a new PID. Same
            # two-fact rule the host below uses.
            if ProcessManager.is_port_in_use(agent_cfg.port, multi_agent.host.bind):
                _report_port_still_held(args.name, agent_cfg.port)
                return 1
            print(f"   {args.name} terminated")
            return 0

        reaped = _reap_orphans_on_port(
            agent_cfg.port, args.name, force, multi_agent.host.bind
        )
        if reaped is PortReapResult.RELEASED:
            print(f"   {args.name} terminated (orphan)")
        elif reaped is PortReapResult.NOTHING_FOUND:
            print(f"   {args.name} is not running")
        else:
            _report_port_still_held(args.name, agent_cfg.port)
            return 1
        return 0

    # Terminate everything: agent processes first, then the host process.
    print("\U0001F6D1 Terminating Kestrel MultiAgent...")

    # Anything that outlived the stop, named so the summary cannot claim a
    # clean shutdown over the top of it.
    unterminated: list[str] = []

    for name, cfg in multi_agent.get_local_agents().items():
        pm.register_agent(name, cfg)
        ap = pm._agents.get(name)
        if ap and ap.pid:
            print(f"   Terminating {name} (PID: {ap.pid})...")
            if not pm.terminate_agent(name):
                print(
                    f"   {name}: PID {ap.pid} is still running after SIGKILL"
                )
                unterminated.append(name)
            elif ProcessManager.is_port_in_use(cfg.port, multi_agent.host.bind):
                _report_port_still_held(name, cfg.port)
                unterminated.append(name)
            else:
                print(f"   {name} terminated")
        elif _reap_orphans_on_port(
            cfg.port, name, force, multi_agent.host.bind
        ) is PortReapResult.STILL_HELD:
            _report_port_still_held(name, cfg.port)
            unterminated.append(name)

    # Terminate host process.
    host_pid_file = _host_pid_file(project_dir)
    # The verified read, so "nothing is there" and "something else is there
    # now" are different answers. A number alone could not tell them apart,
    # and both used to arrive as the same PID (#2995).
    host_record = pm.read_pid_record(host_pid_file)
    host_pid = host_record.pid if host_record.is_running else None
    if host_pid:
        print(f"   Terminating host (PID: {host_pid})...")
        # The instant from the FILE, not a fresh lookup. Looking it up again
        # would read whatever holds the number now, so a PID reused between
        # the read and the kill would be validated against itself and
        # signalled — the exact protection being asked for. None for a legacy
        # record, which signals as before.
        started_at = host_record.started_at
        pm.kill_process(host_pid, force=force, started_at=started_at)
        for _ in range(10):
            if not pm.is_process_running(host_pid):
                break
            time.sleep(0.5)
        if pm.is_process_running(host_pid):
            pm.kill_process(host_pid, force=True, started_at=started_at)
            time.sleep(0.5)
        # Two independent facts, both required before claiming a stop: the
        # process is gone, and the port it served is free. Neither implies the
        # other — a host that failed to bind leaves the port free while still
        # running, and ``is_process_running`` cannot see a process owned by
        # another user (#2995), which the port probe can.
        host_alive = pm.is_process_running(host_pid)
        port_held = ProcessManager.is_port_in_use(
            multi_agent.host.port, multi_agent.host.bind
        )
        if not host_alive:
            # The PID file is worth keeping only while it names something
            # real. Once that process is gone the record is stale, and a
            # stale record is worse than none: the PID can be reused, and the
            # next lifecycle command would signal an unrelated process. Clear
            # it on its own facts, independent of who holds the port.
            pm.clear_pid(host_pid_file)
        if host_alive or port_held:
            if host_alive:
                print(
                    f"   host: PID {host_pid} is still running after SIGKILL"
                )
            if port_held:
                _report_port_still_held("host", multi_agent.host.port)
            unterminated.append("host")
        else:
            print("   host terminated")
    else:
        if host_record.status is PidStatus.STALE:
            # Known to name a process that is gone, or one that is not the
            # host. Keeping that record is what lets a later command signal
            # whatever inherited the number, so it goes now — this is the one
            # place that can say so with evidence rather than a guess.
            print(f"   host: clearing stale PID record ({host_record.detail})")
            pm.clear_pid(host_pid_file)
        if _reap_orphans_on_port(
            multi_agent.host.port, "host", force, multi_agent.host.bind
        ) is PortReapResult.STILL_HELD:
            _report_port_still_held("host", multi_agent.host.port)
            unterminated.append("host")

    if unterminated:
        print(
            "\u274c MultiAgent termination incomplete — still running: "
            + ", ".join(unterminated)
        )
        return 1

    print("\u2705 MultiAgent terminated")
    return 0


def cmd_restart(args) -> int:
    """Restart host and/or agents (terminate then start)."""
    # Through ``cli.`` so test patches of cli.cmd_terminate / cli.cmd_start apply.
    rc = cli.cmd_terminate(args)
    if rc != 0:
        return rc
    print()
    return cli.cmd_start(args)


def _resolve_source_checkout() -> Optional[Path]:
    """Find the editable source checkout of ``kestrel_sovereign``, if any.

    Returns the directory containing ``pyproject.toml`` + ``.git`` for
    the running package, or ``None`` when running from a PyPI install
    (no accessible source tree).

    Distinct from ``cli._get_project_dir()`` — that returns the runtime
    data root (honoring ``KESTREL_HOME``), which is the wrong place
    for ``git pull`` and ``uv pip install -e .`` (codex review round
    2 P1). On a configured deployment with ``KESTREL_HOME=/data`` set,
    the data root is NOT a git checkout and has no pyproject; we have
    to introspect the actual installed package's location instead.
    """
    try:
        import kestrel_sovereign as _ks_pkg
    except Exception:  # noqa: BLE001
        return None
    pkg_file = getattr(_ks_pkg, "__file__", None)
    if not pkg_file:
        return None
    # In an editable install, pkg_dir is e.g.
    # /Volumes/.../kestrel-sovereign/kestrel_sovereign — and its
    # parent is the source checkout root.
    candidate = Path(pkg_file).resolve().parent.parent
    if not candidate.exists():
        return None
    if (
        (candidate / "pyproject.toml").exists()
        and (candidate / ".git").exists()
    ):
        return candidate
    return None


class _GitFailedError(RuntimeError):
    """Raised by ``_git_working_tree_dirty`` when the project IS a git
    checkout but ``git status`` itself failed (missing git binary,
    dubious-ownership refusal, etc.). Distinct from
    "not a git checkout" so the caller can surface the real cause
    instead of silently skipping the pull (codex review round 1 P2)."""


def _project_dir_is_git(project_dir: Path) -> bool:
    """Filesystem-only check: does the project look like a git checkout?

    A separate function from ``_git_working_tree_dirty`` so the caller
    can distinguish "pip-installed user, no checkout, silently skip the
    pull" from "real checkout but git itself failed, surface it" —
    codex review #1 P2.
    """
    return (project_dir / ".git").exists()


def _git_working_tree_dirty(project_dir: Path) -> Tuple[bool, str]:
    """Return ``(dirty, summary)`` for the TRACKED files in the working
    tree.

    ``--untracked-files=no`` so untracked files (stale
    ``kestrel.toml.backup-*`` from ``kestrel setup`` rewrites, ad-hoc
    scratch files, etc.) don't block a perfectly-safe ``git pull
    --ff-only``. Only modified, staged, or unmerged TRACKED files
    count as dirty — those are what an FF pull can collide with.
    Followup to feat/kestrel-update-command after the initial roll-out
    refused on a tree whose only "dirt" was untracked backup files.

    ``summary`` is the first few porcelain lines (or the empty
    string when clean) so the refusal message can surface what's
    actually wrong instead of forcing the operator to re-run
    ``git status`` by hand.

    Raises :class:`_GitFailedError` when git itself fails — the caller
    must NOT continue silently with the rest of the update pipeline
    because a real checkout with a broken git is a different
    condition from a pip-installed user with no checkout. Use
    :func:`_project_dir_is_git` upstream to detect the latter.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        raise _GitFailedError(f"git not available: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or (
            f"git status exited {result.returncode}"
        )
        raise _GitFailedError(detail)
    porcelain = result.stdout.rstrip()
    if not porcelain:
        return False, ""
    lines = porcelain.splitlines()
    head = lines[:5]
    tail = (
        f"\n    (+{len(lines) - 5} more)"
        if len(lines) > 5 else ""
    )
    summary = "\n".join(f"    {ln}" for ln in head) + tail
    return True, summary


def _run_git_pull(project_dir: Path) -> Tuple[int, str]:
    """Run ``git pull --ff-only`` in ``project_dir``. Returns
    ``(returncode, combined_output)``."""
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return 1, f"git not available: {exc}"
    out = (result.stdout or "") + (result.stderr or "")
    return result.returncode, out


def _git_reattach_if_safely_detached(project_dir: Path) -> Optional[str]:
    """Reattach a detached-HEAD checkout to its default branch, but ONLY when
    doing so loses no commits. Returns the branch name if it reattached, else
    ``None``.

    Why this exists: in a multi-agent setup, ``gh pr merge --delete-branch`` run
    from a *sibling* git worktree silently flips this PRIMARY checkout off its
    branch into detached HEAD at the same commit. The next ``kestrel update``
    then aborts because ``git pull --ff-only`` refuses on a detached HEAD. The
    detached commit is just the old branch tip (no work to lose), so reattaching
    is safe and lets the update proceed instead of demanding manual recovery.

    Safety: only reattaches when HEAD is an ANCESTOR of ``origin/<branch>`` (a
    pure fast-forward — nothing committed on the detached HEAD would be
    orphaned). If the detached HEAD has diverged (someone committed on it), this
    returns ``None`` and leaves it untouched so a human decides. Call AFTER a
    fetch (e.g. a failed ``git pull``, which fetches before the merge) so
    ``origin/<branch>`` is current.
    """
    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=str(project_dir),
            capture_output=True, text=True, check=False,
        )

    try:
        # symbolic-ref HEAD fails (non-zero) exactly when HEAD is detached.
        if _git("symbolic-ref", "-q", "HEAD").returncode == 0:
            return None  # already on a branch — nothing to do

        # Resolve the remote's default branch (origin/HEAD → e.g. "main");
        # fall back to "main" if the symbolic ref isn't configured.
        head_ref = _git("symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD")
        if head_ref.returncode == 0 and head_ref.stdout.strip():
            ref = head_ref.stdout.strip()
            branch = ref.split("/", 1)[1] if "/" in ref else ref
        else:
            branch = "main"

        # Reattach ONLY if the detached commit is already contained in
        # origin/<branch> — i.e. a no-loss fast-forward.
        if _git("merge-base", "--is-ancestor", "HEAD", f"origin/{branch}").returncode != 0:
            return None

        if _git("checkout", branch).returncode != 0:
            return None
        return branch
    except (FileNotFoundError, OSError):
        return None


def _run_uv_sync(project_dir: Path) -> Tuple[int, str]:
    """Run ``uv sync`` against the source checkout, targeting the
    venv that owns this process.

    Followup to feat/kestrel-update-command: an editable + uv.lock
    workflow refreshes deps via ``uv sync``, NOT ``uv pip install -e
    .`` (which only reinstalls the project package). ``uv sync``
    also prunes anything not in the lock — including
    kestrel-feature-* packages installed out-of-tree — which is
    why ``kestrel feature sync`` runs immediately after to restore
    them.

    Codex review round 1 P1: bare ``uv sync`` targets the project's
    default ``.venv``, not the venv the operator is currently in.
    Codex review round 2 P1: a venv-installed ``kestrel`` invoked
    WITHOUT shell activation (systemd / cron / direct path) has no
    ``VIRTUAL_ENV`` exported, so we can't rely on the env var alone.
    Detect "running inside a venv" via ``sys.prefix !=
    sys.base_prefix``, then SEED ``VIRTUAL_ENV=sys.prefix`` for the
    subprocess and pass ``--active`` so uv picks the right env. This
    mirrors the ``--python sys.executable`` pin on
    ``_run_uv_pip_install_editable``.
    """
    cmd = ["uv", "sync"]
    env = os.environ.copy()
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        # We're inside a venv per sys.prefix. OVERWRITE VIRTUAL_ENV
        # (not ``setdefault``) so a stale or mismatched inherited
        # value can't redirect uv to a different env than the one
        # owning the kestrel binary. Codex review round 3 P1 caught
        # the setdefault footgun.
        env["VIRTUAL_ENV"] = sys.prefix
        cmd.append("--active")
    elif env.get("VIRTUAL_ENV"):
        # Not strictly in a venv per sys.prefix (e.g. system Python)
        # but the operator has VIRTUAL_ENV set — honor it.
        cmd.append("--active")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except (FileNotFoundError, OSError) as exc:
        return 1, f"uv not available: {exc}"
    out = (result.stdout or "") + (result.stderr or "")
    return result.returncode, out


def _run_uv_pip_install_editable(project_dir: Path, no_deps: bool) -> Tuple[int, str]:
    """Run ``uv pip install -e .`` against the venv that owns this
    process — NOT just whatever uv resolves on its own.

    Pinning ``--python sys.executable`` mirrors what
    ``_extension_install_run`` (used by ``feature install/upgrade``)
    does and prevents a footgun where uv installs into a different
    environment than the one running ``kestrel``, leaving the
    subsequent ``kestrel restart`` with the OLD install (codex review
    round 1 P1).
    """
    cmd = ["uv", "pip", "install", "--python", sys.executable]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(["-e", "."])
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return 1, f"uv not available: {exc}"
    out = (result.stdout or "") + (result.stderr or "")
    return result.returncode, out


def _editable_git_pull(checkout: Path, allow_dirty: bool) -> Tuple[int, str]:
    """``git pull --ff-only`` an editable feature checkout (issue #1788).

    The checkout IS the running code for an editable install, so pulling it
    updates the feature. A non-fast-forward or a dirty tracked tree is REPORTED
    clearly (returncode 2) rather than auto-merged or silently skipped — the
    operator must resolve it. A directory that isn't a git checkout is a no-op
    (returncode 0): an editable install from a plain source dir has nothing to
    pull.
    """
    if not checkout.exists():
        return 1, f"editable checkout does not exist: {checkout}"
    if not _project_dir_is_git(checkout):
        return 0, f"(not a git checkout — nothing to pull: {checkout})"
    try:
        dirty, summary = _git_working_tree_dirty(checkout)
    except _GitFailedError as exc:
        return 1, f"git status errored: {exc}"
    if dirty and not allow_dirty:
        msg = (
            "REFUSED — checkout has modified tracked files; commit/stash or "
            "pass --allow-dirty"
        )
        if summary:
            msg += "\n" + summary
        return 2, msg
    rc, out = _run_git_pull(checkout)
    if rc != 0:
        # Same sibling-worktree detached-HEAD flip that hits the main source
        # checkout can hit an editable feature checkout too. The failed pull
        # already fetched, so try a no-loss reattach to the default branch and
        # retry once; a diverged HEAD is left untouched (the failure stands).
        reattached = _git_reattach_if_safely_detached(checkout)
        if reattached:
            rc2, out2 = _run_git_pull(checkout)
            if rc2 == 0:
                return rc2, (
                    f"detached HEAD detected — reattached to "
                    f"'{reattached}' (no commits lost)\n" + out2
                )
            return rc2, out2
    return rc, out


def _run_feature_reconcile(
    project_dir: Path,
    *,
    manifest_override: Optional[str],
    dry_run: bool,
    allow_dirty: bool,
    continue_on_error: bool,
    prefer: Optional[str],
) -> int:
    """Reconcile the host venv against ``union(per-agent feature allowlists)``.

    Implements the provisioning half of the per-agent allowlist mechanism
    (issue #1788): the allowlist is a *filter*, this makes the host hold the
    *set of features the agents actually need*.

      required = union(all [agents.*].features) + MANDATORY_FEATURES
      for each required class -> package (via the registry):
        - missing  -> install from its source-map entry
        - present  -> update (git pull --ff-only if editable; pip --upgrade if pypi)
      - a required class with no resolvable package -> ERROR (no blind fallback)

    Returns 0 on success, non-zero on any hard error or per-package failure
    (unless ``continue_on_error`` downgrades package failures to warnings).
    """
    from kestrel_sovereign import feature_reconcile as fr
    from kestrel_sovereign.cli_features import CoreInstallGuard, core_state_refusal
    from kestrel_sovereign.feature_registry import load_registry
    import importlib.metadata as md

    # 1. Derive WHAT the host needs from multi_agent.toml.
    ma_path = project_dir / MULTI_AGENT_CONFIG_FILENAME
    multi_agent = cli.MultiAgentConfig.load(ma_path)
    required_classes, load_all = fr.compute_required_classes(multi_agent)

    registry = load_registry()
    # Live discovery is the source of truth: resolve allowlist classes against
    # what the venv ACTUALLY provides (installed entry-point packages + in-tree
    # bundled classes) before falling back to the static catalog. Otherwise a
    # real, loadable feature the catalog simply hasn't listed yet would be
    # mis-reported as an unresolvable error (issue #1788).
    try:
        from kestrel_sovereign.features import (
            discover_entrypoint_feature_dists,
            discover_local_feature_class_names,
        )
        entrypoint_dists = discover_entrypoint_feature_dists()
        local_core_classes = discover_local_feature_class_names()
    except Exception as exc:  # noqa: BLE001
        print(
            f"  note: live feature discovery failed ({exc}); "
            "resolving against the static registry only"
        )
        entrypoint_dists, local_core_classes = {}, set()

    pkg_infos, class_to_pkg, unresolved = fr.resolve_packages(
        required_classes, registry,
        entrypoint_dists=entrypoint_dists,
        local_core_classes=local_core_classes,
    )

    if unresolved:
        print(
            "• reconcile: ERROR — feature class(es) in an agent allowlist have "
            "no resolvable package in the registry:",
            file=sys.stderr,
        )
        for cls in unresolved:
            print(f"    {cls}", file=sys.stderr)
        print(
            "    Add the owning package to the feature registry, or fix the "
            "allowlist. (No blind fallback — aborting.)",
            file=sys.stderr,
        )
        return 1

    if load_all:
        print(
            "  note: agent(s) with no `features` allowlist load every installed "
            f"feature; reconcile only guarantees the named union: {', '.join(load_all)}"
        )

    # 2. WHERE each comes from — the source map.
    manifest_path = cli._host_manifest_path(
        argparse.Namespace(manifest=manifest_override)
    )
    manifest_entries = []
    if manifest_path.exists():
        try:
            manifest_entries = cli._load_host_manifest(manifest_path)
        except (ValueError, OSError) as exc:
            print(
                f"• reconcile: ERROR — bad source map {manifest_path}: {exc}",
                file=sys.stderr,
            )
            return 1
    source_index = fr.build_source_index(manifest_entries, registry)

    # 3. Current state of each required package in this venv.
    installed_versions = {}
    editable_paths = {}
    for pkg in pkg_infos:
        try:
            installed_versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            installed_versions[pkg] = None
        editable_paths[pkg] = cli._editable_install_path(pkg)

    actions, no_source = fr.plan_reconcile(
        pkg_infos,
        source_index,
        installed_versions,
        editable_paths,
        class_to_pkg,
        prefer=prefer,
    )

    if no_source:
        print(
            "• reconcile: ERROR — required feature package(s) have no known "
            "install source (no source-map entry and not resolvable):",
            file=sys.stderr,
        )
        for pkg in no_source:
            print(f"    {pkg}", file=sys.stderr)
        return 1

    if not actions:
        print("• reconcile: host venv already satisfies the agent allowlists.")
        return 0

    # Keyed the way every package identity in the plan is keyed
    # (``fr.canonical_package``) — an action carrying the canonical name must
    # not miss a registry row the catalog happened to spell differently.
    git_urls = {
        fr.canonical_package(info.package): info.git
        for info in registry.values() if info.package
    }

    # 4. Guard the core install across the whole batch.
    #
    # Every kestrel-feature-* depends on kestrel-sovereign. `uv pip` is
    # project-blind (see _extension_install_run), so a feature whose core pin the
    # checkout fails resolves core from the index and replaces the operator's
    # core with a wheel copy — invisibly, because cwd=checkout keeps shadowing
    # site-packages for anything started from inside it (issue #2949).
    #
    # Same guard object as `feature install` / `upgrade` / `sync`, holding core
    # to the SAME source-map policy: reconcile never installs core itself (core
    # classes are bundled, so they are excluded from the plan), so there is no
    # core entry to apply first here — only a policy to hold everything else to.
    guard = CoreInstallGuard.snapshot(source_index)

    print(f"  {'PACKAGE':<34} {'CURRENT':<10} {'ACTION'}")
    print(f"  {'-' * 62}")

    rc = 0
    for action in actions:
        label = action.package
        current = action.current_version or "-"
        # A `present` action is installed + loadable with no managed source —
        # nothing to do, just report it (never executed).
        if action.op == "present":
            print(f"  {label:<34} {current:<10} present (no managed source)")
            continue
        how = (
            f"-e {action.source}" if action.mode == "editable"
            else f"pip {action.source}"
        )
        if dry_run:
            print(f"  {label:<34} {current:<10} would {action.op} ({how})")
            continue

        ok, detail = _execute_reconcile_action(
            action, git_urls, allow_dirty, guard=guard,
        )
        if ok:
            print(f"  {label:<34} {current:<10} {action.op}d ({how})")
            if detail:
                for line in detail.splitlines():
                    print(f"      {line}")
        else:
            rc = 1
            print(f"  {label:<34} {current:<10} FAILED")
            for line in (detail or "").splitlines()[-5:]:
                print(f"      {line}")
            if guard.constraints:
                print(
                    f"      note: core is pinned to {guard.constraints[0]} for "
                    "this install so a feature cannot silently replace the "
                    "declared core. If this is a version conflict, update "
                    "the checkout to satisfy the feature — do not remove the pin."
                )
            bound_note = guard.manifest_bound_note()
            if bound_note:
                print(f"      note: {bound_note}")
            if not continue_on_error:
                print(
                    "• reconcile: FAILED — aborting before restart. "
                    "Re-run with --continue-on-error to proceed anyway.",
                    file=sys.stderr,
                )
                break

    # 5. Assert core survived. Runs even when the loop aborted early: a failing
    # install can be the very thing that broke the link.
    #
    # ``CORE_UNSAFE`` outranks a package failure and is returned verbatim: it
    # says the venv is running a core the manifest does not declare, which no
    # caller may continue past. Collapsing it into ``1`` here is what let
    # ``--continue-on-error`` restart the fleet onto an undeclared core.
    if not dry_run:
        core_rc = guard.verify()
        if core_state_refusal(core_rc):
            return core_rc
        if core_rc:
            rc = 1

    return rc


def _execute_reconcile_action(action, git_urls: dict, allow_dirty: bool, *, guard):
    """Execute one :class:`ReconcileAction`. Returns ``(ok, detail)``.

    Editable -> ``git pull --ff-only`` the checkout (plus ``pip install -e`` when
    not yet installed). PyPI -> ``pip install [--upgrade] spec`` with a git-URL
    fallback (mirrors ``feature sync``/``upgrade``).

    *guard* is required: every install here is a feature install, and a feature
    install without the core guard is the #2949 defect. Callers with genuinely
    nothing to protect pass ``CoreInstallGuard.unguarded()`` and say so.
    """
    from kestrel_sovereign.cli_features import _pip_spec

    if action.mode == "editable":
        checkout = Path(action.source).expanduser()
        rc, out = _editable_git_pull(checkout, allow_dirty)
        if rc != 0:
            return False, out
        detail = out.strip()
        # Re-link into the venv when the package is absent or currently linked
        # elsewhere (a non-editable PyPI build, or a different checkout). When
        # it is already editable-linked to this checkout, the pull alone is the
        # update. `relink` is decided by plan_reconcile (codex round 3 P2).
        if action.relink:
            result = guard.run(["-e", _pip_spec(str(checkout), action.extras)])
            if result.returncode != 0:
                return False, (result.stderr or result.stdout or "").strip()
        return True, detail

    # PyPI mode. action.source is ALREADY a full pip requirement with extras
    # rendered before the version spec (``pkg[extra]>=x``) by plan_reconcile,
    # so it is passed verbatim — re-applying _pip_spec here would misplace the
    # extras after the spec and break pip.
    spec = action.source
    pip_args = []
    if action.op == "update":
        pip_args.append("--upgrade")
    pip_args.append(spec)
    # Replacing an editable link with the PyPI wheel needs a force reinstall;
    # otherwise pip treats an already-satisfying editable version as done and
    # leaves the checkout linked (codex round 9 P2). Scoped to THIS package —
    # a bare --force-reinstall cascades to every resolved dependency, and core
    # is one for every feature, so it would pull a same-version core wheel over
    # the editable link that the version pin cannot exclude (issue #2949).
    reinstall = action.package if action.force_reinstall else None
    result = guard.run(pip_args, reinstall=reinstall)
    # The git-URL fallback installs the repo HEAD from a DIFFERENT source with
    # NO version constraint, so it is only available to an entry that declared
    # no source of its own — substituting it for a declared one moves the
    # feature outside the operator's window, or off the index they named
    # (codex round 7 P2; ``ReconcileAction.source_declared``).
    if (
        result.returncode != 0
        and not action.source_declared
        and git_urls.get(action.package)
    ):
        git_ref = f"git+{git_urls[action.package]}"
        git_spec = (
            f"{_pip_spec(action.package, action.extras)} @ {git_ref}"
            if action.extras else git_ref
        )
        fallback = (
            ["--upgrade", git_spec] if action.op == "update" else [git_spec]
        )
        result = guard.run(fallback, reinstall=reinstall)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "").strip()
    return True, ""


def cmd_update(args) -> int:
    """One-shot ``git pull`` + ``uv pip install -e .`` +
    ``kestrel feature sync`` + ``kestrel restart``.

    Each step can be skipped via a ``--no-<step>`` flag, and
    ``--dry-run`` previews without mutating anything.

    Safety:
      - A dirty working tree refuses the pull unless ``--allow-dirty``.
      - ``git pull --ff-only`` so a non-fast-forward upstream aborts
        cleanly instead of producing a surprise merge commit.
      - Any failure short-circuits the remaining steps so a
        half-applied update doesn't get restarted into.
      - ``--continue-on-error`` lets ``feature sync`` fail without
        skipping the restart (useful when a single optional feature
        package is temporarily unreachable). It does NOT cover an
        unrepaired core drift: that returns ``CORE_UNSAFE`` and always
        aborts before the restart, because continuing would bring the
        agents up on a core the manifest does not declare (#2949).
    """
    from kestrel_sovereign.cli_features import core_state_refusal

    # cli._get_project_dir() returns the RUNTIME data root (honors
    # KESTREL_HOME) which is the wrong place for git pull and
    # uv pip install -e . — use the actual editable source checkout
    # the package lives in (codex review round 2 P1). The runtime
    # data root only matters for the feature-sync manifest lookup.
    project_dir = cli._get_project_dir()
    source_checkout = cli._resolve_source_checkout()
    dry_run = bool(getattr(args, "dry_run", False))
    pull = bool(getattr(args, "pull", True))
    install = bool(getattr(args, "install", True))
    features = bool(getattr(args, "features", True))
    restart = bool(getattr(args, "restart", True))
    allow_dirty = bool(getattr(args, "allow_dirty", False))
    no_deps = bool(getattr(args, "no_deps", False))
    continue_on_error = bool(getattr(args, "continue_on_error", False))
    # --prefer-source / --prefer-pypi bulk-override the per-feature reconcile
    # update mode (issue #1788). Mutually exclusive at the parser level.
    prefer = None
    if bool(getattr(args, "prefer_source", False)):
        prefer = "source"
    elif bool(getattr(args, "prefer_pypi", False)):
        prefer = "pypi"

    target = getattr(args, "name", None)
    target_label = target if target else "all agents"

    print(f"kestrel update → project: {project_dir}")
    if source_checkout:
        print(f"  source checkout: {source_checkout}")
    else:
        print("  source checkout: (none — PyPI-installed, "
              "pull/install will be skipped)")
    print(
        "  steps: "
        f"pull={pull} install={install} "
        f"features={features} restart={restart}"
    )
    print(f"  target: {target_label}{' [DRY-RUN]' if dry_run else ''}")
    print()

    # Step 1: git pull --ff-only against the source checkout (NOT the
    # data root, which may be a non-git KESTREL_HOME).
    if pull:
        if source_checkout is None:
            print(
                "• pull: skipped (no editable source checkout — "
                "kestrel_sovereign is installed from PyPI)"
            )
        elif not cli._project_dir_is_git(source_checkout):
            # Defensive: _resolve_source_checkout already checked for
            # .git, but if it disappeared between resolution and now
            # treat as non-checkout rather than erroring.
            print("• pull: skipped (source checkout has no .git)")
        else:
            try:
                dirty, dirty_summary = cli._git_working_tree_dirty(source_checkout)
            except _GitFailedError as exc:
                print(
                    f"• pull: FAILED — git status errored ({exc}). "
                    "Aborting before install/sync/restart.",
                    file=sys.stderr,
                )
                return 1
            if dirty and not allow_dirty:
                print(
                    "• pull: REFUSED — working tree has modified tracked "
                    "files. Commit/stash first, or pass --allow-dirty.",
                    file=sys.stderr,
                )
                if dirty_summary:
                    print(dirty_summary, file=sys.stderr)
                return 2
            if dry_run:
                print(
                    f"• pull: would run `git pull --ff-only` "
                    f"in {source_checkout}"
                )
            else:
                print(f"• pull: git pull --ff-only ({source_checkout})")
                rc, out = cli._run_git_pull(source_checkout)
                if rc != 0:
                    # A sibling worktree's `gh pr merge --delete-branch` can flip
                    # this checkout into detached HEAD, which makes `git pull
                    # --ff-only` refuse. The failed pull already fetched, so try
                    # a no-loss reattach to the default branch and retry once.
                    reattached = cli._git_reattach_if_safely_detached(source_checkout)
                    if reattached:
                        print(
                            f"    detached HEAD detected — reattached to "
                            f"'{reattached}' (no commits lost); retrying pull"
                        )
                        rc, out = cli._run_git_pull(source_checkout)
                    if rc != 0:
                        print(out.rstrip(), file=sys.stderr)
                        print(
                            "• pull: FAILED — aborting before "
                            "install/sync/restart.",
                            file=sys.stderr,
                        )
                        return rc
                if out.strip():
                    for line in out.rstrip().splitlines():
                        print(f"    {line}")
    else:
        print("• pull: skipped (--no-pull)")

    # Step 2: refresh the install from the source checkout.
    #
    # Two flavors depending on the workflow:
    #   - ``uv sync`` for editable + lockfile setups (the modern
    #     uv-managed workflow). Refreshes the full env from uv.lock,
    #     which also PRUNES anything not in the lock — that's why
    #     ``kestrel feature sync`` runs immediately after, to restore
    #     out-of-tree feature packages.
    #   - ``uv pip install -e .`` for the simpler case where only the
    #     kestrel-sovereign package needs reinstalling.
    #
    # Default: auto-detect by checking for ``uv.lock`` at the source
    # root. ``--uv-sync`` / ``--no-uv-sync`` lets the operator pin
    # explicitly. PyPI installs (no source checkout) skip the step
    # entirely — there's no editable env to refresh.
    explicit_uv_sync = getattr(args, "uv_sync", None)
    if install:
        if source_checkout is None:
            print(
                "• install: skipped (no editable source checkout — "
                "use `pip install --upgrade kestrel-sovereign` "
                "to update a PyPI install)"
            )
        else:
            if explicit_uv_sync is True:
                use_uv_sync = True
            elif explicit_uv_sync is False:
                use_uv_sync = False
            else:
                # Auto-detect: an editable + uv.lock workflow uses
                # `uv sync` to refresh deps.
                use_uv_sync = (source_checkout / "uv.lock").exists()

            if use_uv_sync:
                cmd_label = "uv sync"
                if dry_run:
                    print(
                        f"• install: would run `{cmd_label}` "
                        f"(detected uv.lock) in {source_checkout}"
                    )
                else:
                    print(
                        f"• install: {cmd_label} "
                        f"(detected uv.lock) ({source_checkout})"
                    )
                    rc, out = cli._run_uv_sync(source_checkout)
                    if rc != 0:
                        print(out.rstrip(), file=sys.stderr)
                        print(
                            "• install: FAILED — aborting before "
                            "sync/restart.",
                            file=sys.stderr,
                        )
                        return rc
                    tail = [
                        ln for ln in out.rstrip().splitlines()
                        if ln.strip()
                    ][-3:]
                    for line in tail:
                        print(f"    {line}")
            else:
                cmd_label = (
                    "uv pip install"
                    + (" --no-deps" if no_deps else "")
                    + " -e ."
                )
                if dry_run:
                    print(
                        f"• install: would run `{cmd_label}` "
                        f"in {source_checkout}"
                    )
                else:
                    print(f"• install: {cmd_label} ({source_checkout})")
                    rc, out = cli._run_uv_pip_install_editable(
                        source_checkout, no_deps=no_deps,
                    )
                    if rc != 0:
                        print(out.rstrip(), file=sys.stderr)
                        print(
                            "• install: FAILED — aborting before "
                            "sync/restart.",
                            file=sys.stderr,
                        )
                        return rc
                    tail = [
                        ln for ln in out.rstrip().splitlines()
                        if ln.strip()
                    ][-3:]
                    for line in tail:
                        print(f"    {line}")
    else:
        print("• install: skipped (--no-install)")

    # Step 2.5: kestrel feature sync — restore the host's out-of-tree feature
    # packages that `uv sync` just pruned (anything not in sovereign's lock).
    #
    # This MUST run before reconcile: a feature package describes itself via its
    # ``kestrel_sovereign.features`` entry points, and reconcile resolves
    # allowlist classes from that LIVE discovery (the static catalog is only a
    # fallback for not-yet-installed packages — see feature_reconcile.
    # resolve_packages). If reconcile ran first, it would discover against the
    # just-pruned venv, fail to resolve any editable feature not also hand-listed
    # in the static catalog, and abort. Restoring first means the features are
    # present to self-declare, so reconcile validates against reality.
    # (Reordered from the original #1788 placement, which checked before the
    # restore and made the static catalog load-bearing.)
    if features:
        manifest_path = cli._host_manifest_path(
            argparse.Namespace(manifest=getattr(args, "manifest", None))
        )
        if not manifest_path.exists():
            # No host manifest: nothing recorded to restore. Skip the restore
            # (cmd_feature_sync errors on a missing manifest) but DON'T abort —
            # reconcile still runs so registry-backed allowlist features get
            # provisioned, and it hard-fails on any NAMED allowlist class that
            # is now unresolvable. Warn because `uv sync` may have pruned
            # out-of-tree packages a load-all agent relied on (which reconcile
            # can't enumerate); capturing a manifest makes them survive updates.
            print(
                f"• features: no host manifest at {manifest_path} — skipping "
                "restore. If this host has out-of-tree feature packages, "
                "capture one with `kestrel feature sync --capture` so they "
                "survive `uv sync`. Continuing to reconcile.",
                file=sys.stderr,
            )
        elif dry_run:
            print("• features: would run `kestrel feature sync`")
        else:
            print("• features: kestrel feature sync")
            sync_args = argparse.Namespace(
                manifest=getattr(args, "manifest", None),
                capture=False,
                dry_run=False,
                # Forwarded, or `kestrel update --allow-dirty` still refuses to
                # pull a dirty declared core checkout during sync — the flag
                # advertised on THIS command silently not reaching the step that
                # honours it.
                allow_dirty=allow_dirty,
            )
            rc = cli.cmd_feature_sync(sync_args)
            # A core state is a safety failure, not an optional-package
            # failure, so --continue-on-error does not reach any of them. The
            # sentence comes from the one table that knows them all, rather
            # than a branch per code that a new code can be added without.
            refusal = core_state_refusal(rc)
            if refusal:
                print(
                    f"• features: FAILED — {refusal}; --continue-on-error does "
                    "not cover this.",
                    file=sys.stderr,
                )
                return rc
            if rc != 0 and not continue_on_error:
                print(
                    "• features: FAILED — aborting before reconcile/restart. "
                    "Re-run with --continue-on-error to proceed anyway.",
                    file=sys.stderr,
                )
                return rc
    else:
        print("• features: skipped (--no-features)")

    # Step 2.6: reconcile the host venv against union(agent allowlists).
    #
    # The per-agent `features` allowlist is a FILTER, not an INSTALLER: a class
    # named in an allowlist but missing from the venv silently never loads.
    # Reconcile validates union(allowlists) ⊆ host venv and installs any union
    # member still missing (e.g. catalogued-but-not-yet-installed packages),
    # resolving class→package from live entry-point discovery of the features
    # restored by the sync above. Gated by the same `features` flag.
    if features:
        if dry_run:
            print("• reconcile: union(agent allowlists) → host venv [preview]")
        else:
            print("• reconcile: union(agent allowlists) → host venv")
        rc = _run_feature_reconcile(
            project_dir,
            manifest_override=getattr(args, "manifest", None),
            dry_run=dry_run,
            allow_dirty=allow_dirty,
            continue_on_error=continue_on_error,
            prefer=prefer,
        )
        # Not continuable, by design: restarting here would bring every agent
        # up on a core the manifest does not declare — or one it cannot load —
        # and returning 0 would report that as a successful update.
        refusal = core_state_refusal(rc)
        if refusal:
            print(
                f"• reconcile: FAILED — {refusal}; --continue-on-error does "
                "not cover this.",
                file=sys.stderr,
            )
            return rc
        if rc != 0 and not continue_on_error:
            print(
                "• reconcile: FAILED — aborting before restart. "
                "Re-run with --continue-on-error to proceed anyway.",
                file=sys.stderr,
            )
            return rc
    else:
        print("• reconcile: skipped (--no-features)")

    # Step 4: kestrel restart.
    if restart:
        if dry_run:
            label = f"would run `kestrel restart {target or ''}`".rstrip()
            print(f"• restart: {label}")
        else:
            print(f"• restart: kestrel restart {target or ''}".rstrip())
            restart_args = argparse.Namespace(
                name=target,
                force=bool(getattr(args, "force", False)),
                startup_timeout=_startup_timeout(args),
            )
            rc = cli.cmd_restart(restart_args)
            if rc != 0:
                return rc
    else:
        print("• restart: skipped (--no-restart)")

    if dry_run:
        print("\nkestrel update: dry-run complete; no changes made.")
    else:
        print("\nkestrel update: done.")
    return 0


def cmd_status(args) -> int:
    """Show status of host and all agents."""
    project_dir = cli._get_project_dir()
    multi_agent = cli.MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)

    # Host/server status
    # Same verified read as ``stop`` and the reanchor guard: status must not
    # report an agent the stop path cannot see, or vice versa (#2995).
    host_record = ProcessManager.read_pid_record(_host_pid_file(project_dir))
    host_running = host_record.is_running
    host_pid = host_record.pid
    host_pid_str = str(host_pid) if host_running else "-"
    host_uptime = _format_uptime(host_pid) if host_running else "-"

    # Detect mode: check if any agent has its own PID file (subprocess mode)
    local_agents = multi_agent.get_local_agents()
    any_agent_pid = any(
        ProcessManager.read_pid_record(
            ProcessManager.agent_pid_file((project_dir / cfg.data_dir).resolve())
        ).is_running
        for cfg in local_agents.values()
    )

    if any_agent_pid:
        # Subprocess mode: show per-agent PID status
        print(f"  {'NAME':12} {'PORT':>6}   {'STATUS':10} {'PID':>7}   {'UPTIME':>8}")
        host_status = "online" if host_running else "offline"
        print(f"  {'host':12} {multi_agent.host.port:>6}   {host_status:10} {host_pid_str:>7}   {host_uptime:>8}")

        for name, cfg in local_agents.items():
            resolved_dir = (project_dir / cfg.data_dir).resolve()
            agent_record = ProcessManager.read_pid_record(
                ProcessManager.agent_pid_file(resolved_dir)
            )
            pid = agent_record.pid
            running = agent_record.is_running
            status_str = "online" if running else "offline"
            pid_str = str(pid) if running else "-"
            uptime = _format_uptime(pid) if running else "-"
            print(f"  {name:12} {cfg.port:>6}   {status_str:10} {pid_str:>7}   {uptime:>8}")
    else:
        # In-process mode: query server API for agent status
        print(f"  {'NAME':12} {'STATUS':10} {'PID':>7}   {'UPTIME':>8}")
        server_status = "online" if host_running else "offline"
        print(f"  {'server':12} {server_status:10} {host_pid_str:>7}   {host_uptime:>8}")

        if host_running:
            agents = _query_agents_api(multi_agent.host.port)
            if agents is not None:
                for agent_info in agents:
                    name = agent_info.get("name", "?")
                    print(f"  {name:12} {'in-process':10} {host_pid_str:>7}   {host_uptime:>8}")
            else:
                # API unavailable, show config-based list
                for name in local_agents:
                    print(f"  {name:12} {'in-process':10} {host_pid_str:>7}   {host_uptime:>8}")
        else:
            for name in local_agents:
                print(f"  {name:12} {'offline':10} {'-':>7}   {'-':>8}")

    return 0


def _query_agents_api(port: int):
    """Query the running server's /api/agents endpoint. Returns list or None."""
    try:
        import urllib.request
        import json
        url = f"http://localhost:{port}/api/agents"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return data.get("agents", [])
    except Exception:
        return None


def cmd_logs(args) -> int:
    """Tail agent or host logs."""
    project_dir = cli._get_project_dir()

    if args.name == "host" or args.name == "server":
        log_file = _host_log_file(project_dir)
    else:
        multi_agent = cli.MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)
        local_agents = multi_agent.get_local_agents()
        if args.name not in local_agents:
            print(f"Agent '{args.name}' not found in multi_agent config")
            print("Use 'host' for host/server logs")
            return 1

        agent_cfg = local_agents[args.name]
        resolved_dir = (project_dir / agent_cfg.data_dir).resolve()
        log_file = ProcessManager.agent_log_file(resolved_dir)

        # In-process mode: agent has no separate log, fall back to server log
        if not log_file.exists():
            log_file = _host_log_file(project_dir)

    if not log_file.exists():
        print(f"No log file found: {log_file}")
        return 1

    # Build tail command
    tail_args = ["tail"]
    tail_args.extend(["-n", str(args.lines)])
    if args.follow:
        tail_args.append("-f")
    tail_args.append(str(log_file))

    try:
        return subprocess.call(tail_args)
    except KeyboardInterrupt:
        return 0




def add_lifecycle_subparsers(subparsers) -> None:
    """Register start/terminate/restart/update/status/logs on ``subparsers``."""
    # kestrel start [name]
    start_p = subparsers.add_parser("start", help="Start host and/or agents")
    start_p.add_argument("name", nargs="?", help="Agent name (omit for all)")
    _add_startup_timeout_argument(start_p)

    # kestrel terminate [name] [--force]
    terminate_p = subparsers.add_parser(
        "terminate",
        help="Terminate host and/or agent processes",
    )
    terminate_p.add_argument("name", nargs="?", help="Agent name (omit for all)")
    terminate_p.add_argument(
        "--force", action="store_true",
        help="Send SIGKILL instead of SIGTERM (also used when reaping orphans)",
    )

    # kestrel restart [name] [--force]
    restart_p = subparsers.add_parser("restart", help="Restart host and/or agents")
    restart_p.add_argument("name", nargs="?", help="Agent name (omit for all)")
    restart_p.add_argument(
        "--force", action="store_true",
        help="Force-kill existing processes during the stop phase",
    )
    _add_startup_timeout_argument(restart_p)

    # kestrel update [agent]
    update_p = subparsers.add_parser(
        "update",
        help=(
            "One-shot: git pull + uv pip install -e . + "
            "kestrel feature sync + kestrel restart"
        ),
    )
    update_p.add_argument(
        "name", nargs="?",
        help="Agent name (omit for all) — only the restart step honors this",
    )
    update_p.add_argument(
        "--no-pull", dest="pull", action="store_false",
        help="Skip the `git pull --ff-only` step",
    )
    update_p.add_argument(
        "--no-install", dest="install", action="store_false",
        help=(
            "Skip the install step "
            "(`uv sync` or `uv pip install -e .` — auto-detected)"
        ),
    )
    update_p.add_argument(
        "--no-features", dest="features", action="store_false",
        help=(
            "Skip the feature-provisioning steps "
            "(reconcile against agent allowlists + `kestrel feature sync`)"
        ),
    )
    prefer_grp = update_p.add_mutually_exclusive_group()
    prefer_grp.add_argument(
        "--prefer-source", action="store_true",
        help=(
            "Reconcile: update every feature with a known checkout via "
            "`git pull` (editable), overriding its source-map mode"
        ),
    )
    prefer_grp.add_argument(
        "--prefer-pypi", action="store_true",
        help=(
            "Reconcile: update every feature via `pip install --upgrade`, "
            "overriding its source-map mode (no editable git pulls)"
        ),
    )
    update_p.add_argument(
        "--no-restart", dest="restart", action="store_false",
        help="Skip the final `kestrel restart` step",
    )
    update_p.add_argument(
        "--allow-dirty", action="store_true",
        help=(
            "Pull even when the working tree has modified TRACKED files. "
            "Untracked files (e.g. kestrel.toml.backup-*) never block."
        ),
    )
    update_p.add_argument(
        "--no-deps", action="store_true",
        help=(
            "Pass --no-deps to `uv pip install` "
            "(ignored when the install step resolves to `uv sync`)"
        ),
    )
    # Tri-state: None = auto-detect by uv.lock presence; True = force
    # ``uv sync``; False = force ``uv pip install -e .``. argparse's
    # ``BooleanOptionalAction`` gives both ``--uv-sync`` and
    # ``--no-uv-sync`` with a default of ``None`` so the auto-detect
    # branch stays the implicit behaviour.
    update_p.add_argument(
        "--uv-sync", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Use `uv sync` (true) or `uv pip install -e .` (false) "
            "for the install step. Default: auto-detect by uv.lock "
            "presence."
        ),
    )
    update_p.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue to the restart step if `feature sync` fails",
    )
    update_p.add_argument(
        "--dry-run", action="store_true",
        help="Preview the steps that would run; mutate nothing",
    )
    update_p.add_argument(
        "--manifest",
        help=(
            "Path to a non-default host-features manifest "
            "(forwarded to `kestrel feature sync`)"
        ),
    )
    update_p.add_argument(
        "--force", action="store_true",
        help="Forwarded to `kestrel restart` (force-kill stale processes)",
    )
    _add_startup_timeout_argument(update_p)

    # kestrel status
    subparsers.add_parser("status", help="Show status of host and agents")

    # kestrel logs <name>
    logs_p = subparsers.add_parser("logs", help="Tail agent or host logs")
    logs_p.add_argument("name", help="Agent name or 'host'")
    logs_p.add_argument("-n", "--lines", type=int, default=50)
    logs_p.add_argument("-f", "--follow", action="store_true")


# Imported at module bottom (not top) to break the cli <-> cli_lifecycle import
# cycle: cli.py re-exports the public handlers above, so when this module is
# imported first it must finish defining them before `cli` (which imports
# them back) loads. Every `cli.<name>` reference above resolves at call time,
# so binding `cli` here at end-of-module is safe.
from kestrel_sovereign import cli  # noqa: E402
