"""``kestrel serve`` — manage local llama.cpp model servers on this host.

A thin launcher/registry in front of ``llama-server``. It does NOT reimplement
inference or the LLM service; it just starts/stops/switches the big local GGUF
models (Kimi, GLM-5.2, DeepSeek V4-Flash, …) that share one port and cannot
co-reside in RAM (150–470GB each on the M3 Ultra 512GB).

Why this lives here (and not in core or a feature): it must cooperate with the
memory budget that ``scripts/workload_manager.py`` already arbitrates — two
independent managers of port 8001 / 512GB would OOM the box on a switch. Core
must never import llama.cpp; a separate feature is premature until a 2nd host
exists. So: a kestrel CLI command group that reuses ``MemoryGuard``.

Subcommands::

    kestrel serve list                 # registry + which model is running
    kestrel serve up <name>            # pre-flight RAM check, launch detached, wait for /health
    kestrel serve down                 # stop the running model
    kestrel serve switch <name>        # down current, then up new (RAM-safe one-at-a-time)
    kestrel serve status               # running model + /health + memory pressure

Registry (TOML). Resolution order: ``$KESTREL_SERVE_REGISTRY`` →
``./serve_models.toml`` → ``~/.kestrel/serve_models.toml``. See
``serve_models.example.toml`` for the schema. ``gguf`` may be a glob; for a
sharded model it auto-selects the ``*-00001-of-*.gguf`` first shard (so shard
counts never need hand-patching).
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path
from typing import Any, Optional

logger = __import__("logging").getLogger(__name__)

DEFAULT_PORT = 8001
STATE_DIR = Path(os.environ.get("KESTREL_SERVE_STATE_DIR", str(Path.home() / ".kestrel")))
STATE_FILE = STATE_DIR / "serve_state.json"
LOG_DIR = STATE_DIR / "logs"
# Keep a margin so we never fill RAM to the brim and trigger swap thrash.
RAM_MARGIN_GB = 24.0


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def resolve_registry_path(explicit: Optional[str] = None) -> Optional[Path]:
    """Find the registry TOML. Returns None if no registry exists anywhere."""
    candidates = [
        explicit,
        os.environ.get("KESTREL_SERVE_REGISTRY"),
        "serve_models.toml",
        str(STATE_DIR / "serve_models.toml"),
    ]
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser()
    return None


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    """Load the registry. Returns {model_name: entry}. Raises on malformed TOML."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"{path}: [models] must be a table of model entries")
    # Surface a registry-wide default port onto each entry unless overridden.
    default_port = int(data.get("default_port", DEFAULT_PORT))
    for entry in models.values():
        entry.setdefault("port", default_port)
    return models


def resolve_gguf(pattern: str) -> Path:
    """Expand a gguf path/glob to a single first-shard file.

    - exact file -> that file
    - one match -> it
    - many matches -> the ``*-00001-of-*.gguf`` first shard (sharded model)
    """
    p = Path(pattern).expanduser()
    if p.is_file():
        return p
    matches = sorted(Path(m) for m in _glob.glob(str(p)))
    if not matches:
        raise FileNotFoundError(f"no GGUF matches: {pattern}")
    if len(matches) == 1:
        return matches[0]
    firsts = [m for m in matches if "00001-of-" in m.name]
    if len(firsts) == 1:
        return firsts[0]
    raise ValueError(
        f"{pattern} matched {len(matches)} files and the first shard is ambiguous: "
        + ", ".join(m.name for m in matches)
    )


def build_command(name: str, entry: dict[str, Any], port: int) -> list[str]:
    """Construct the llama-server argv for a registry entry."""
    binary = entry.get("binary", "llama-server")
    # Bare name -> resolve on PATH (Homebrew); absolute/relative path -> use as-is.
    if os.sep not in binary:
        resolved = shutil.which(binary)
        if not resolved:
            raise FileNotFoundError(f"{name}: binary {binary!r} not found on PATH")
        binary = resolved
    elif not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"{name}: binary not executable: {binary}")

    model = resolve_gguf(entry["gguf"])
    cmd = [
        binary,
        "--model", str(model),
        "--ctx-size", str(entry.get("ctx_size", 131072)),
        "--port", str(port),
        "--kv-unified",
    ]
    kv = entry.get("kv_cache_type")
    if kv:
        cmd += ["--cache-type-k", str(kv), "--cache-type-v", str(kv)]
    rf = entry.get("reasoning_format")
    if rf:
        cmd += ["--reasoning-format", str(rf)]
    extra = entry.get("extra_args", [])
    if extra:
        cmd += [str(a) for a in extra]
    return cmd


# --------------------------------------------------------------------------- #
# State / process helpers
# --------------------------------------------------------------------------- #
def read_state() -> Optional[dict[str, Any]]:
    if not STATE_FILE.is_file():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def clear_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_model() -> Optional[dict[str, Any]]:
    """Return live state ({name,pid,port,...}) if a managed model is running, else None."""
    st = read_state()
    if st and pid_alive(int(st.get("pid", -1))):
        return st
    if st:  # stale pidfile — process is gone
        clear_state()
    return None


def health_ok(port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Memory pre-flight (reuses workload_manager's MemoryGuard when importable)
# --------------------------------------------------------------------------- #
def memory_status() -> tuple[float, float, Optional[str]]:
    """Return (total_gb, available_gb, pressure_label_or_None)."""
    try:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from scripts.memory_guard import MemoryGuard  # type: ignore

        s = MemoryGuard().get_status()
        return s.total_gb, s.available_gb, s.pressure.name
    except Exception:
        # Degrade visibly to a direct macOS read — the hard RAM gate still works,
        # we just lose the richer kernel pressure level.
        total = avail = 0.0
        try:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                       capture_output=True, text=True).stdout) / (1024 ** 3)
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = 4096
            free = inactive = spec = purge = 0
            for line in vm.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split("bytes")[0].strip())
                elif line.startswith("Pages free:"):
                    free = int(line.rsplit(maxsplit=1)[1].rstrip("."))
                elif line.startswith("Pages inactive:"):
                    inactive = int(line.rsplit(maxsplit=1)[1].rstrip("."))
                elif line.startswith("Pages speculative:"):
                    spec = int(line.rsplit(maxsplit=1)[1].rstrip("."))
                elif line.startswith("Pages purgeable:"):
                    purge = int(line.rsplit(maxsplit=1)[1].rstrip("."))
            avail = (free + inactive + spec + purge) * page / (1024 ** 3)
        except Exception:
            pass
        return total, avail, None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def _get_entry(models: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    if name not in models:
        raise SystemExit(
            f"unknown model {name!r}. Known: {', '.join(sorted(models)) or '(none)'}"
        )
    return models[name]


def cmd_list(models: dict[str, dict[str, Any]]) -> int:
    live = running_model()
    live_name = live["name"] if live else None
    if not models:
        print("(registry empty)")
        return 0
    width = max(len(n) for n in models)
    print(f"{'MODEL':<{width}}  STATE    EST_RAM  BINARY")
    for name in sorted(models):
        e = models[name]
        state = "RUNNING" if name == live_name else "-"
        binary = e.get("binary", "llama-server")
        print(f"{name:<{width}}  {state:<7}  {str(e.get('est_ram_gb','?'))+'GB':>7}  {binary}")
    return 0


def cmd_status(models: dict[str, dict[str, Any]]) -> int:
    total, avail, pressure = memory_status()
    print(f"Memory: {avail:.0f}GB available / {total:.0f}GB total"
          + (f"  pressure={pressure}" if pressure else ""))
    live = running_model()
    if not live:
        print("No managed model running.")
        return 0
    ok = health_ok(int(live["port"]))
    print(f"Running: {live['name']}  pid={live['pid']}  port={live['port']}  "
          f"health={'OK' if ok else 'NOT READY'}")
    print(f"  log: {live.get('log', '?')}")
    return 0


def _start(name: str, entry: dict[str, Any], port: int, *, timeout: int,
           wait: bool, force: bool) -> int:
    est = float(entry.get("est_ram_gb", 0) or 0)
    total, avail, pressure = memory_status()
    if est and not force and est + RAM_MARGIN_GB > avail:
        print(f"REFUSING: {name} needs ~{est:.0f}GB (+{RAM_MARGIN_GB:.0f}GB margin) "
              f"but only {avail:.0f}GB available. Use --force to override.", file=sys.stderr)
        return 1

    cmd = build_command(name, entry, port)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"serve-{name}.log"
    env = os.environ.copy()
    for k, v in (entry.get("env") or {}).items():
        env[str(k)] = str(v)

    print(f"Starting {name} on :{port} (est ~{est:.0f}GB, {avail:.0f}GB free)…")
    log_fh = open(log_path, "ab")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
        start_new_session=True,  # detach so the server outlives this CLI call
    )
    write_state({
        "name": name, "pid": proc.pid, "port": port,
        "log": str(log_path), "cmd": cmd, "started_at": int(time.time()),
    })
    if not wait:
        print(f"  pid={proc.pid}  log={log_path}  (not waiting; check `kestrel serve status`)")
        return 0

    print(f"  pid={proc.pid}  waiting for /health (up to {timeout}s; large models take minutes)…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"FAILED: server exited (code {proc.returncode}). Tail of {log_path}:",
                  file=sys.stderr)
            _tail(log_path)
            clear_state()
            return 1
        if health_ok(port):
            print(f"  HEALTHY: {name} serving on http://localhost:{port}/v1")
            return 0
        time.sleep(3)
    print(f"  TIMEOUT after {timeout}s (still loading?). Process is up; "
          f"check `kestrel serve status`.", file=sys.stderr)
    return 1


def _tail(path: Path, n: int = 20) -> None:
    try:
        lines = path.read_text(errors="replace").splitlines()[-n:]
        print("\n".join(lines), file=sys.stderr)
    except OSError:
        pass


def _stop(timeout: int = 30) -> int:
    live = running_model()
    if not live:
        print("No managed model running.")
        return 0
    pid, name = int(live["pid"]), live["name"]
    print(f"Stopping {name} (pid {pid})…")
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        clear_state()
        return 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            clear_state()
            print(f"  stopped {name}.")
            return 0
        time.sleep(1)
    print(f"  did not exit in {timeout}s; sending SIGKILL.", file=sys.stderr)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    clear_state()
    return 0


def cmd_up(models: dict[str, dict[str, Any]], args: argparse.Namespace) -> int:
    entry = _get_entry(models, args.name)
    live = running_model()
    if live:
        print(f"REFUSING: {live['name']} is already running (pid {live['pid']}). "
              f"Use `kestrel serve switch {args.name}` or `kestrel serve down` first.",
              file=sys.stderr)
        return 1
    port = args.port or int(entry.get("port", DEFAULT_PORT))
    return _start(args.name, entry, port, timeout=args.timeout,
                  wait=not args.no_wait, force=args.force)


def cmd_switch(models: dict[str, dict[str, Any]], args: argparse.Namespace) -> int:
    entry = _get_entry(models, args.name)
    rc = _stop(timeout=args.timeout)
    if rc != 0:
        return rc
    port = args.port or int(entry.get("port", DEFAULT_PORT))
    return _start(args.name, entry, port, timeout=args.timeout,
                  wait=not args.no_wait, force=args.force)


# --------------------------------------------------------------------------- #
# argparse wiring
# --------------------------------------------------------------------------- #
def add_serve_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``serve`` command group under the top-level CLI parser."""
    parser = subparsers.add_parser(
        "serve",
        help="Manage local llama.cpp model servers (Kimi/GLM/V4-Flash) on this host",
    )
    sub = parser.add_subparsers(dest="serve_command")

    sub.add_parser("list", help="List registry models and which one is running")
    sub.add_parser("status", help="Show running model, /health, and memory")
    sub.add_parser("down", help="Stop the running model").add_argument(
        "--timeout", type=int, default=30, help="Seconds to await graceful stop")

    for verb, helptext in (("up", "Start a model"), ("switch", "Stop current, start another")):
        p = sub.add_parser(verb, help=helptext)
        p.add_argument("name", help="Model name from the registry")
        p.add_argument("--port", type=int, default=None, help="Override port")
        p.add_argument("--timeout", type=int, default=300,
                       help="Seconds to await /health (default 300)")
        p.add_argument("--no-wait", action="store_true",
                       help="Return immediately after launch")
        p.add_argument("--force", action="store_true",
                       help="Start even if estimated RAM exceeds available")

    parser.set_defaults(_handler=run)


def run(args: argparse.Namespace) -> int:
    """Dispatch ``kestrel serve <subcommand>``."""
    if not getattr(args, "serve_command", None):
        print("usage: kestrel serve {list,up,down,switch,status}", file=sys.stderr)
        return 1

    reg_path = resolve_registry_path()
    if reg_path is None:
        print("No registry found. Create serve_models.toml (see serve_models.example.toml) "
              "or set $KESTREL_SERVE_REGISTRY.", file=sys.stderr)
        return 1
    try:
        models = load_registry(reg_path)
    except (tomllib.TOMLDecodeError, ValueError, OSError) as e:
        print(f"Failed to load registry {reg_path}: {e}", file=sys.stderr)
        return 1

    cmd = args.serve_command
    if cmd == "list":
        return cmd_list(models)
    if cmd == "status":
        return cmd_status(models)
    if cmd == "down":
        return _stop(timeout=args.timeout)
    if cmd == "up":
        return cmd_up(models, args)
    if cmd == "switch":
        return cmd_switch(models, args)
    print(f"unknown serve subcommand: {cmd}", file=sys.stderr)
    return 1
