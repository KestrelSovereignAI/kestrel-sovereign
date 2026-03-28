#!/usr/bin/env python3
"""
Workload Manager - Orchestrates all Mac Studio workloads.

Single daemon that manages Kimi health, talon processing, LoRA batch
training, embedding computation, and Emma self-improvement — all with
memory-pressure-aware scheduling.

Usage:
    python scripts/workload_manager.py
    python scripts/workload_manager.py --status   # Print current status and exit
    python scripts/workload_manager.py --verbose

Exposes HTTP status at http://localhost:8099/status
"""

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
from datetime import datetime, time as dtime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure logs directory exists
Path("/Volumes/data2/projects/kestrel-sovereign/logs").mkdir(exist_ok=True)

from scripts.memory_guard import MemoryGuard, PressureLevel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            "/Volumes/data2/projects/kestrel-sovereign/logs/workload_manager.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=7,
        ),
    ],
)
logger = logging.getLogger("workload_manager")

# Schedule: when to run what
SLEEP_START = dtime(0, 0)   # 00:00
SLEEP_END = dtime(2, 0)     # 02:00
LORA_START = dtime(2, 0)    # 02:00
LORA_END = dtime(6, 0)      # 06:00


class WorkloadState:
    """Tracks state of all workloads."""

    def __init__(self):
        self.started_at = datetime.now(timezone.utc)
        self.kimi_healthy = False
        self.kimi_last_check: Optional[datetime] = None
        self.talon_running = False
        self.talon_pid: Optional[int] = None
        self.talon_issues_today = 0
        self.talon_current_issue: Optional[str] = None
        self.lora_running = False
        self.lora_pid: Optional[int] = None
        self.embeddings_running = False
        self.embeddings_pid: Optional[int] = None
        self.emma_mode = "unknown"  # "work", "sleep"
        self.memory_pressure = "unknown"
        self.memory_used_gb = 0.0
        self.memory_total_gb = 512.0

    def to_dict(self) -> dict:
        return {
            "uptime_hours": round(
                (datetime.now(timezone.utc) - self.started_at).total_seconds() / 3600, 1
            ),
            "kimi": {
                "healthy": self.kimi_healthy,
                "last_check": self.kimi_last_check.isoformat() if self.kimi_last_check else None,
                "port": 8001,
            },
            "talon": {
                "running": self.talon_running,
                "pid": self.talon_pid,
                "issues_today": self.talon_issues_today,
                "current_issue": self.talon_current_issue,
            },
            "lora": {
                "running": self.lora_running,
                "pid": self.lora_pid,
            },
            "embeddings": {
                "running": self.embeddings_running,
                "pid": self.embeddings_pid,
            },
            "emma": {
                "mode": self.emma_mode,
            },
            "memory": {
                "total_gb": self.memory_total_gb,
                "used_gb": round(self.memory_used_gb, 1),
                "pressure": self.memory_pressure,
            },
        }


# Global state for HTTP handler
_state = WorkloadState()


class StatusHandler(BaseHTTPRequestHandler):
    """HTTP handler for /status endpoint."""

    def do_GET(self):
        if self.path == "/status" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_state.to_dict(), indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


def start_status_server(port: int = 8099):
    """Start HTTP status server in background thread."""
    try:
        server = HTTPServer(("127.0.0.1", port), StatusHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Status server: http://localhost:{port}/status")
    except OSError as e:
        logger.warning(f"Could not start status server on port {port}: {e}")


def check_kimi_health() -> bool:
    """Check if Kimi llama-server is responding."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8001/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_in_time_window(start: dtime, end: dtime) -> bool:
    """Check if current time is within a window (handles midnight crossover)."""
    now = datetime.now().time()
    if start <= end:
        return start <= now < end
    else:
        return now >= start or now < end


def is_sleep_time() -> bool:
    return is_in_time_window(SLEEP_START, SLEEP_END)


def is_lora_time() -> bool:
    return is_in_time_window(LORA_START, LORA_END)


class SubprocessManager:
    """Manages a single long-running subprocess."""

    def __init__(self, name: str, cmd: list[str], cwd: str | Path, env: Optional[dict] = None):
        self.name = name
        self.cmd = cmd
        self.cwd = str(cwd)
        self.env = env or os.environ.copy()
        self.process: Optional[asyncio.subprocess.Process] = None

    async def start(self) -> Optional[int]:
        """Start the subprocess. Returns PID or None."""
        if self.process and self.process.returncode is None:
            logger.debug(f"{self.name} already running (PID {self.process.pid})")
            return self.process.pid

        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            logger.info(f"Started {self.name} (PID {self.process.pid})")
            return self.process.pid
        except Exception as e:
            logger.error(f"Failed to start {self.name}: {e}")
            return None

    async def stop(self, timeout: int = 30):
        """Stop the subprocess gracefully."""
        if not self.process or self.process.returncode is not None:
            return

        logger.info(f"Stopping {self.name} (PID {self.process.pid})")
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Force-killing {self.name}")
            self.process.kill()
            await self.process.wait()

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None


async def run_workload_manager(verbose: bool = False):
    """Main orchestration loop."""
    global _state

    guard = MemoryGuard()

    # Get GH token once
    try:
        result = subprocess.run(
            ["gh", "auth", "token", "--user", "UncleSaurus"],
            capture_output=True, text=True, timeout=10,
        )
        gh_token = result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        gh_token = os.environ.get("GITHUB_TOKEN", "")

    talon_env = os.environ.copy()
    talon_env["GH_TOKEN"] = gh_token
    talon_env["GITHUB_TOKEN"] = gh_token
    talon_env.pop("ANTHROPIC_API_KEY", None)

    # Subprocess managers
    talon = SubprocessManager(
        name="talon-daemon",
        cmd=[
            sys.executable, str(project_root / "scripts/talon_daemon.py"),
            "--config", str(project_root / "scripts/talon_daemon.toml"),
        ] + (["--verbose"] if verbose else []),
        cwd=project_root,
        env=talon_env,
    )

    embeddings = SubprocessManager(
        name="embedding-worker",
        cmd=[sys.executable, str(project_root / "scripts/embedding_worker.py")],
        cwd=project_root,
    )

    lora_batch = SubprocessManager(
        name="lora-batch",
        cmd=[sys.executable, str(project_root / "scripts/frinz_lora_batch.py")],
        cwd=project_root,
    )

    start_status_server()

    logger.info("Workload Manager started")
    logger.info(f"  Schedule: sleep={SLEEP_START}-{SLEEP_END}, lora={LORA_START}-{LORA_END}")

    while True:
        try:
            # Update memory status
            mem_status = guard.get_status()
            _state.memory_pressure = mem_status.pressure.value
            _state.memory_used_gb = mem_status.used_gb
            _state.memory_total_gb = mem_status.total_gb

            # Check Kimi health
            _state.kimi_healthy = check_kimi_health()
            _state.kimi_last_check = datetime.now(timezone.utc)
            if not _state.kimi_healthy:
                logger.warning("Kimi not responding on port 8001")

            # Determine Emma mode
            _state.emma_mode = "sleep" if is_sleep_time() else "work"

            # --- Workload decisions based on pressure and schedule ---

            if mem_status.pressure == PressureLevel.RED:
                # RED: stop everything except Kimi
                logger.warning("RED pressure — stopping heavy workloads")
                await lora_batch.stop()
                await talon.stop()
                # Keep embeddings if possible (very lightweight)
                _state.talon_running = False
                _state.lora_running = False

            elif is_sleep_time():
                # Sleep window: stop talon and LoRA, let Emma reflect
                if talon.is_running:
                    logger.info("Sleep window — pausing talon")
                    await talon.stop()
                if lora_batch.is_running:
                    await lora_batch.stop()
                _state.talon_running = False
                _state.lora_running = False

            else:
                # Normal operation

                # Talon — always run during work hours if GREEN
                if mem_status.pressure == PressureLevel.GREEN and not talon.is_running:
                    pid = await talon.start()
                    _state.talon_pid = pid
                _state.talon_running = talon.is_running

                # LoRA batch — only during designated hours, GREEN pressure
                if is_lora_time() and mem_status.pressure == PressureLevel.GREEN:
                    if not lora_batch.is_running:
                        pid = await lora_batch.start()
                        _state.lora_pid = pid
                elif lora_batch.is_running and not is_lora_time():
                    logger.info("LoRA window ended, stopping batch")
                    await lora_batch.stop()
                _state.lora_running = lora_batch.is_running

            # Embeddings — always run unless RED
            if mem_status.pressure != PressureLevel.RED and not embeddings.is_running:
                pid = await embeddings.start()
                _state.embeddings_pid = pid
            elif mem_status.pressure == PressureLevel.RED and embeddings.is_running:
                await embeddings.stop()
            _state.embeddings_running = embeddings.is_running

            # Check for crashed subprocesses and restart
            if talon.process and talon.process.returncode is not None and not is_sleep_time():
                logger.warning(f"Talon exited (code {talon.process.returncode}), will restart next cycle")
                talon.process = None

            if embeddings.process and embeddings.process.returncode is not None:
                logger.warning(f"Embeddings exited (code {embeddings.process.returncode}), will restart")
                embeddings.process = None

            await asyncio.sleep(30)  # Check every 30 seconds

        except asyncio.CancelledError:
            logger.info("Workload Manager shutting down")
            await talon.stop()
            await lora_batch.stop()
            await embeddings.stop()
            break
        except Exception as e:
            logger.error(f"Error in workload loop: {e}", exc_info=True)
            await asyncio.sleep(60)

    logger.info("Workload Manager stopped")


def print_status():
    """Print current status from the running daemon."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8099/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            print(json.dumps(data, indent=2))
    except Exception:
        print("Workload Manager not running (no response on localhost:8099)")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Workload Manager")
    parser.add_argument("--status", action="store_true", help="Print status and exit")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    loop = asyncio.new_event_loop()
    task = None

    def shutdown(sig, frame):
        logger.info(f"Received {signal.Signals(sig).name}")
        if task:
            task.cancel()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        task = loop.create_task(run_workload_manager(verbose=args.verbose))
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
