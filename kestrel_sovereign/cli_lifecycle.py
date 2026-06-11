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
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from kestrel_sovereign.multi_agent.config import MULTI_AGENT_CONFIG_FILENAME
from kestrel_sovereign.multi_agent.process_manager import ProcessManager


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
            print(f"          \u274c")
            print(f"   {e}")
            return 1

        if pm.wait_for_health(agent_cfg.port, timeout=30):
            print("          \u2705")
        else:
            print("          \u274c")
            return 1
        return 0

    if getattr(args, "subprocess", False):
        return _start_subprocess_mode(project_dir, multi_agent, pm)
    return _start_inprocess_mode(project_dir, multi_agent, pm)


def _start_inprocess_mode(project_dir: Path, multi_agent, pm: ProcessManager) -> int:
    """Start all agents in a single server process (default mode)."""
    autostart = multi_agent.get_autostart_agents()
    manual = {
        name: cfg for name, cfg in multi_agent.get_local_agents().items()
        if not cfg.autostart
    }

    print("\U0001F985 Kestrel MultiAgent starting (in-process)...")
    print(f"   URL:      http://localhost:{multi_agent.host.port}")

    if autostart or manual:
        print("   Agents:")
        for name, cfg in autostart.items():
            resolved = (project_dir / cfg.data_dir).resolve()
            print(f"     {name:12} {resolved}/       autostart")
        for name, cfg in manual.items():
            resolved = (project_dir / cfg.data_dir).resolve()
            print(f"     {name:12} {resolved}/       manual")
    print()

    host_pid_file = _host_pid_file(project_dir)
    existing_host = pm.read_pid(host_pid_file)
    if existing_host and pm.is_process_running(existing_host):
        print(f"   Server already running (PID: {existing_host})")
        return 0

    if existing_host:
        pm.clear_pid(host_pid_file)

    if pm.is_port_in_use(multi_agent.host.port):
        orphans = pm.find_pids_on_port(multi_agent.host.port)
        print(f"   Port {multi_agent.host.port} already in use"
              + (f" by PID(s) {orphans}" if orphans else ""))
        print(f"   Run: kestrel stop   (add --force if it doesn't die)")
        return 1

    env = pm._load_env()
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
    pm._spawn_detached(cmd, env, log_file, host_pid_file)

    if pm.wait_for_health(multi_agent.host.port, timeout=30):
        print("          \u2705")
    else:
        print("          \u274c")
        print(f"   Check log: {log_file}")
        return 1

    print(f"\n\U0001F985 MultiAgent ready: http://localhost:{multi_agent.host.port}")
    return 0


def _start_subprocess_mode(project_dir: Path, multi_agent, pm: ProcessManager) -> int:
    """Start host + separate agent processes (legacy --subprocess mode)."""
    # Start the full multi_agent (host + autostart agents)
    print("\U0001F985 Kestrel MultiAgent starting (subprocess)...")
    print(f"   Host:     http://localhost:{multi_agent.host.port}")

    autostart = multi_agent.get_autostart_agents()
    manual = {
        name: cfg for name, cfg in multi_agent.get_local_agents().items()
        if not cfg.autostart
    }

    if autostart or manual:
        print("   Agents:")
        for name, cfg in autostart.items():
            resolved = (project_dir / cfg.data_dir).resolve()
            print(f"     {name:12} port {cfg.port}  {resolved}/       autostart")
        for name, cfg in manual.items():
            resolved = (project_dir / cfg.data_dir).resolve()
            print(f"     {name:12} port {cfg.port}  {resolved}/       manual (skipped)")
    print()

    # Start host
    host_pid_file = _host_pid_file(project_dir)
    existing_host = pm.read_pid(host_pid_file)
    if existing_host and pm.is_process_running(existing_host):
        print(f"   Host already running (PID: {existing_host})")
    else:
        if existing_host:
            pm.clear_pid(host_pid_file)

        if pm.is_port_in_use(multi_agent.host.port):
            orphans = pm.find_pids_on_port(multi_agent.host.port)
            print(f"   Host port {multi_agent.host.port} already in use"
                  + (f" by PID(s) {orphans}" if orphans else ""))
            print(f"   Run: kestrel stop   (add --force if it doesn't die)")
            return 1

        env = pm._load_env()
        env["PORT"] = str(multi_agent.host.port)
        # Host is NOT an agent — no DB path, no KESTREL_SERVE_UI

        log_file = _host_log_file(project_dir)
        # Use the fully-qualified package path. Pre-move this said
        # ``host:app``, which only resolved when CWD happened to contain
        # ``host.py`` — i.e. only on source clones. Pip-installed users
        # got ``ModuleNotFoundError: No module named 'host'`` on the
        # legacy ``--subprocess`` path. Same fix shape as #1097.
        cmd = [sys.executable, "-m", "uvicorn", "kestrel_sovereign.host:app",
               "--host", multi_agent.host.bind, "--port", str(multi_agent.host.port)]

        print(f"   Starting host on :{multi_agent.host.port}...", end="", flush=True)
        pm._spawn(cmd, env, log_file, host_pid_file)

        if pm.wait_for_health(multi_agent.host.port, timeout=30):
            print("          \u2705")
        else:
            print("          \u274c")
            print(f"   Check log: {log_file}")
            return 1

    # Start autostart agents
    for name, cfg in autostart.items():
        print(f"   Starting {name} on :{cfg.port}...", end="", flush=True)
        try:
            pm.start_agent(name, cfg, multi_agent.host.bind)
        except RuntimeError as e:
            print(f"          \u274c")
            print(f"   {e}")
            continue

        if pm.wait_for_health(cfg.port, timeout=30):
            print("          \u2705")
        else:
            print("          \u274c")

    print(f"\n\U0001F985 MultiAgent ready: http://localhost:{multi_agent.host.port}")
    return 0


def _reap_orphans_on_port(port: int, label: str, force: bool) -> bool:
    """Kill untracked listeners on `port`. Returns True if any were killed."""
    orphans = ProcessManager.find_pids_on_port(port)
    if not orphans:
        return False
    print(f"   {label}: orphan listener(s) on :{port} {orphans} — killing")
    for opid in orphans:
        ProcessManager.kill_process(opid, force=force)
    for _ in range(10):
        if not ProcessManager.is_port_in_use(port):
            return True
        time.sleep(0.3)
    if ProcessManager.is_port_in_use(port):
        for opid in orphans:
            ProcessManager.kill_process(opid, force=True)
        time.sleep(0.3)
    return True


def cmd_stop(args) -> int:
    """Stop host and/or agents."""
    project_dir = cli._get_project_dir()
    multi_agent = cli.MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)
    pm = ProcessManager(project_dir)
    force = getattr(args, "force", False)

    if args.name:
        # Stop a single agent
        local_agents = multi_agent.get_local_agents()
        if args.name not in local_agents:
            print(f"Agent '{args.name}' not found in multi_agent config")
            return 1

        agent_cfg = local_agents[args.name]
        pm.register_agent(args.name, agent_cfg)
        ap = pm._agents.get(args.name)
        if ap and ap.pid:
            print(f"   Stopping {args.name} (PID: {ap.pid})...")
            pm.stop_agent(args.name)
            print(f"   {args.name} stopped")
        elif _reap_orphans_on_port(agent_cfg.port, args.name, force):
            print(f"   {args.name} stopped (orphan)")
        else:
            print(f"   {args.name} is not running")
        return 0

    # Stop everything: agents first, then host
    print("\U0001F6D1 Stopping Kestrel MultiAgent...")

    for name, cfg in multi_agent.get_local_agents().items():
        pm.register_agent(name, cfg)
        ap = pm._agents.get(name)
        if ap and ap.pid:
            print(f"   Stopping {name} (PID: {ap.pid})...")
            pm.stop_agent(name)
            print(f"   {name} stopped")
        else:
            _reap_orphans_on_port(cfg.port, name, force)

    # Stop host
    host_pid_file = _host_pid_file(project_dir)
    host_pid = pm.read_pid(host_pid_file)
    if host_pid and pm.is_process_running(host_pid):
        print(f"   Stopping host (PID: {host_pid})...")
        pm.kill_process(host_pid, force=force)
        for _ in range(10):
            if not pm.is_process_running(host_pid):
                break
            time.sleep(0.5)
        if pm.is_process_running(host_pid):
            pm.kill_process(host_pid, force=True)
            time.sleep(0.5)
        pm.clear_pid(host_pid_file)
        print("   host stopped")
    else:
        if host_pid:
            pm.clear_pid(host_pid_file)
        _reap_orphans_on_port(multi_agent.host.port, "host", force)

    print("\u2705 MultiAgent stopped")
    return 0


def cmd_restart(args) -> int:
    """Restart host and/or agents (stop then start)."""
    # Through ``cli.`` so test patches of cli.cmd_stop / cli.cmd_start apply.
    rc = cli.cmd_stop(args)
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
        package is temporarily unreachable).
    """
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

    # Step 3: kestrel feature sync.
    if features:
        if dry_run:
            print("• features: would run `kestrel feature sync`")
        else:
            print("• features: kestrel feature sync")
            sync_args = argparse.Namespace(
                manifest=getattr(args, "manifest", None),
                capture=False,
                dry_run=False,
            )
            rc = cli.cmd_feature_sync(sync_args)
            if rc != 0 and not continue_on_error:
                print(
                    "• features: FAILED — aborting before restart. "
                    "Re-run with --continue-on-error to restart anyway.",
                    file=sys.stderr,
                )
                return rc
    else:
        print("• features: skipped (--no-features)")

    # Step 4: kestrel restart.
    if restart:
        if dry_run:
            label = f"would run `kestrel restart {target or ''}`".rstrip()
            print(f"• restart: {label}")
        else:
            print(f"• restart: kestrel restart {target or ''}".rstrip())
            restart_args = argparse.Namespace(
                name=target,
                subprocess=bool(getattr(args, "subprocess", False)),
                force=bool(getattr(args, "force", False)),
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
    host_pid = ProcessManager.read_pid(_host_pid_file(project_dir))
    host_running = host_pid is not None and ProcessManager.is_process_running(host_pid)
    host_pid_str = str(host_pid) if host_running else "-"
    host_uptime = _format_uptime(host_pid) if host_running else "-"

    # Detect mode: check if any agent has its own PID file (subprocess mode)
    local_agents = multi_agent.get_local_agents()
    any_agent_pid = any(
        ProcessManager.read_pid(
            ProcessManager.agent_pid_file((project_dir / cfg.data_dir).resolve())
        ) is not None
        for cfg in local_agents.values()
    )

    if any_agent_pid:
        # Subprocess mode: show per-agent PID status
        print(f"  {'NAME':12} {'PORT':>6}   {'STATUS':10} {'PID':>7}   {'UPTIME':>8}")
        host_status = "online" if host_running else "offline"
        print(f"  {'host':12} {multi_agent.host.port:>6}   {host_status:10} {host_pid_str:>7}   {host_uptime:>8}")

        for name, cfg in local_agents.items():
            resolved_dir = (project_dir / cfg.data_dir).resolve()
            pid = ProcessManager.read_pid(ProcessManager.agent_pid_file(resolved_dir))
            running = pid is not None and ProcessManager.is_process_running(pid)
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
    """Register start/stop/restart/update/status/logs on ``subparsers``."""
    # kestrel start [name] [--subprocess]
    start_p = subparsers.add_parser("start", help="Start host and/or agents")
    start_p.add_argument("name", nargs="?", help="Agent name (omit for all)")
    start_p.add_argument(
        "--subprocess", action="store_true",
        help="Run each agent as a separate process (legacy mode)",
    )

    # kestrel stop [name] [--force]
    stop_p = subparsers.add_parser("stop", help="Stop host and/or agents")
    stop_p.add_argument("name", nargs="?", help="Agent name (omit for all)")
    stop_p.add_argument(
        "--force", action="store_true",
        help="Send SIGKILL instead of SIGTERM (also used when reaping orphans)",
    )

    # kestrel restart [name] [--subprocess] [--force]
    restart_p = subparsers.add_parser("restart", help="Restart host and/or agents")
    restart_p.add_argument("name", nargs="?", help="Agent name (omit for all)")
    restart_p.add_argument(
        "--subprocess", action="store_true",
        help="Run each agent as a separate process (legacy mode)",
    )
    restart_p.add_argument(
        "--force", action="store_true",
        help="Force-kill existing processes during the stop phase",
    )

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
        help="Skip the `kestrel feature sync` step",
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
        "--subprocess", action="store_true",
        help="Forwarded to `kestrel restart` (run agents as subprocesses)",
    )
    update_p.add_argument(
        "--force", action="store_true",
        help="Forwarded to `kestrel restart` (force-kill stale processes)",
    )

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

