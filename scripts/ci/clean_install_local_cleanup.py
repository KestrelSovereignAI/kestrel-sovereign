"""Fail-closed cleanup for the disposable local clean-install rehearsal.

Only a live PID record fenced to this checkout and its assigned port may be
signalled. An occupied port with no such record is ambiguous: retain the
temporary venv and host database instead of deleting files a server may use.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.multi_agent.process_manager import PidStatus, ProcessManager


def _wait_for_exit(pid_file: Path, timeout_seconds: float = 5) -> bool:
    """Wait for a recorded process to become absent/stale, never guess."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = ProcessManager.read_pid_record(pid_file).status
        if status in (PidStatus.ABSENT, PidStatus.STALE):
            return True
        if status is not PidStatus.LIVE:
            return False
        time.sleep(0.1)
    return ProcessManager.read_pid_record(pid_file).status in (
        PidStatus.ABSENT, PidStatus.STALE
    )


def stop_owned_process(pid_file: Path, root: Path, port: int) -> bool:
    """Stop only the process proven by this checkout's fenced PID record."""
    record = ProcessManager.read_pid_record(pid_file)
    if record.status is PidStatus.LIVE:
        if (
            record.pid is None
            or record.started_at is None
            or record.root != str(root)
            or record.port != port
        ):
            return False
        ProcessManager.kill_process(record.pid, started_at=record.started_at)
        if not _wait_for_exit(pid_file):
            if ProcessManager.read_pid_record(pid_file).status is not PidStatus.LIVE:
                return False
            ProcessManager.kill_process(
                record.pid, force=True, started_at=record.started_at
            )
            if not _wait_for_exit(pid_file):
                return False
    elif record.status in (PidStatus.UNDECIDABLE, PidStatus.UNREADABLE):
        return False

    return not ProcessManager.is_port_in_use(port, "127.0.0.1")


def cleanup_safe(root: Path, *, check_ports: bool) -> bool:
    """Return whether the harness temp directory can now be removed."""
    if not check_ports:
        return True  # No server has been launched by this script yet.
    config = MultiAgentConfig.load(root / "multi_agent.toml")
    if set(config.agents) != {"Kestrel"}:
        return False
    agent = config.agents["Kestrel"]
    if not isinstance(agent, LocalAgentConfig):
        return False
    agent_dir = agent.resolve_data_dir(root)
    if not agent_dir.is_relative_to(root):
        return False
    # Stop the host first: it may run the configured agent in-process.
    host_ok = stop_owned_process(
        root / "logs" / ".host.pid", root, config.host.port
    )
    agent_ok = stop_owned_process(
        ProcessManager.agent_pid_file(agent_dir), root, agent.port
    )
    return host_ok and agent_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check-ports", action="store_true")
    args = parser.parse_args()
    try:
        safe = cleanup_safe(args.root.resolve(), check_ports=args.check_ports)
    except Exception as exc:  # noqa: BLE001 — preserve resources on uncertainty
        print(f"cleanup inconclusive: {exc}")
        return 1
    if not safe:
        print("cleanup inconclusive: a recorded process or assigned port remains")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
