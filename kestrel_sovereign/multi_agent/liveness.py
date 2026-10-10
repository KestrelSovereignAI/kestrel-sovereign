"""Whether a live process is serving an agent, however it was launched (#3522).

``kestrel start`` records what it launches: ``logs/.host.pid`` for the
in-process host, ``<data_dir>/agent.pid`` for an agent process of its own. A
server started any other way writes neither: the documented direct launch
(``python -m kestrel_sovereign.server``) and a container's Uvicorn entrypoint
both serve agents no launcher record names. A guard that read only those files
reported such an agent stopped, so ``kestrel update --no-restart`` could
replace the constitution beneath it and its next integrity audit put it in
Safe Mode, and an offline reanchor could write a database it was serving.

So the process serving an agent records itself. Every ``KestrelAgent`` writes
a :class:`ServingRecord` into its data directory as it boots and removes it
once it has shut down. There is one file per agent instance, so two holders of
one directory, or an in-process restart whose old and new agent overlap, never
remove each other's record.

A record is evidence of liveness, never of absence. An agent is stopped only
when no launcher record and no serving record names a process that may still
be running. A record whose process cannot be identified from here counts as
running: one written on another host or in another PID namespace (a container,
or a machine sharing the data directory under another hostname), one that
cannot be read, and one naming a PID whose start time cannot be checked.
Waving a guard past a live agent costs more than a refusal an operator can
clear (#2995). A record is attributed by hostname and PID namespace only, so
two machines sharing one data directory under the same hostname cannot be
told apart.

Not a port probe. A direct launch listens wherever its operator said, and
whatever answers on a registered port need not serve this project: the public
``/health`` deliberately names no agent.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Directory under an agent's data directory holding its serving records.
SERVING_RECORD_DIRNAME = ".serving"


@dataclass(frozen=True)
class AgentHolder:
    """A process serving an agent, or one that may be: the evidence, and the fix.

    ``command`` is the command that stops it, when one is known to work from
    here. ``remedy`` says how to stop it either way. ``verified`` is False
    when the evidence cannot establish that anything is running, only that
    nothing can be proven stopped.
    """

    evidence: str
    remedy: str
    command: Optional[str] = None
    verified: bool = True


def _launcher_holder(command: str, evidence: str) -> AgentHolder:
    return AgentHolder(
        evidence=evidence, remedy=f"`{command}` stops it", command=command
    )


def _stop_command(pid: int) -> str:
    if sys.platform == "win32":
        return f"taskkill /PID {pid}"
    return f"kill {pid}"


def _pid_namespace() -> Optional[str]:
    """This process's PID namespace, where the platform has them (Linux)."""
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def _here() -> tuple[str, Optional[str]]:
    """Where a recorded PID means the process this one would probe."""
    return socket.gethostname(), _pid_namespace()


class ServingRecord:
    """This process's record that it serves the agent in one data directory."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def acquire(cls, data_dir: Path) -> "ServingRecord":
        """Record that this process serves the agent in ``data_dir``.

        The record is published whole (written aside, then renamed), so a
        reader never mistakes a half-written file for an unreadable one.
        Records this host can prove stale are removed first.

        Raises:
            OSError: The record cannot be written.
        """
        from kestrel_sovereign.multi_agent.process_manager import ProcessManager

        directory = Path(data_dir) / SERVING_RECORD_DIRNAME
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _remove_stale_records(directory)
        pid = os.getpid()
        host, pid_namespace = _here()
        payload = {
            "pid": pid,
            "started_at": ProcessManager.process_start_time(pid),
            "host": host,
            "pid_namespace": pid_namespace,
            "data_dir": str(data_dir),
        }
        path = directory / f"{pid}-{secrets.token_hex(8)}.json"
        staging = directory / f".{path.name}.tmp"
        fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staging, path)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
        return cls(path)

    def release(self) -> None:
        """Remove the record: this process no longer serves the agent.

        Raises:
            OSError: The record cannot be removed.
        """
        self.path.unlink(missing_ok=True)


def _remove_stale_records(directory: Path) -> None:
    """Delete records this host can prove name a process that has exited."""
    from kestrel_sovereign.multi_agent.process_manager import PidStatus, ProcessManager

    for path in directory.glob("*.json"):
        if _foreign_origin(_read_payload(path)) is not None:
            continue
        if ProcessManager.read_pid_record(path).status is PidStatus.STALE:
            path.unlink(missing_ok=True)


def _read_payload(path: Path):
    """The record's JSON object, or the reason it cannot be read."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return exc
    if not isinstance(payload, dict):
        return ValueError("not a JSON object")
    return payload


def _foreign_origin(payload) -> Optional[str]:
    """Where a record was written, when that is not somewhere this host can probe."""
    if not _well_formed(payload):
        return None
    host, pid_namespace = _here()
    if payload.get("host") != host:
        return f"on host {payload.get('host')!r}"
    if payload.get("pid_namespace") != pid_namespace:
        return f"in PID namespace {payload.get('pid_namespace')!r}"
    return None


def _well_formed(payload) -> bool:
    return (
        isinstance(payload, dict)
        and type(payload.get("pid")) is int
        and isinstance(payload.get("host"), str)
    )


def _unprovable(path: Path, evidence: str) -> AgentHolder:
    return AgentHolder(
        evidence=evidence,
        remedy=(
            "stop the process serving it; if none is, delete "
            f"{path} and retry"
        ),
        verified=False,
    )


def _serving_record_holder(path: Path) -> Optional[AgentHolder]:
    """What one serving record establishes, or None when it names nothing live."""
    from kestrel_sovereign.multi_agent.process_manager import PidStatus, ProcessManager

    payload = _read_payload(path)
    if isinstance(payload, Exception):
        if isinstance(payload, FileNotFoundError):
            # Released between listing and reading.
            return None
        return _unprovable(
            path,
            f"its serving record {path} cannot be read ({payload}), so "
            "whether it is served cannot be told",
        )
    if not _well_formed(payload):
        return _unprovable(
            path,
            f"{path} is not a serving record this version can read (it "
            "names no PID or host), so whether it is served cannot be told",
        )
    origin = _foreign_origin(payload)
    if origin is not None:
        return _unprovable(
            path,
            f"{path} records PID {payload.get('pid')} serving it {origin}, "
            "which cannot be checked from here",
        )
    record = ProcessManager.read_pid_record(path)
    if record.status is PidStatus.LIVE:
        return AgentHolder(
            evidence=(
                f"PID {record.pid} serves it, started without `kestrel start`; "
                f"recorded in {path}"
            ),
            remedy=f"`{_stop_command(record.pid)}` stops it",
            command=_stop_command(record.pid),
        )
    if record.status in (PidStatus.STALE, PidStatus.ABSENT):
        return None
    if record.status is PidStatus.UNDECIDABLE:
        return _unprovable(
            path,
            f"{path} records PID {record.pid} serving it, and that process "
            "exists but its identity cannot be established from here",
        )
    return _unprovable(path, f"{record.detail}; recorded in {path}")


def serving_holder(data_dir: Path) -> Optional[AgentHolder]:
    """The process a serving record says serves the agent in ``data_dir``.

    A record this host can verify wins over one it cannot, so the remedy names
    a command whenever there is one.
    """
    directory = Path(data_dir) / SERVING_RECORD_DIRNAME
    try:
        paths = sorted(
            entry.path
            for entry in os.scandir(directory)
            if entry.name.endswith(".json") and not entry.name.startswith(".")
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        return AgentHolder(
            evidence=(
                f"its serving records in {directory} cannot be listed ({exc}), "
                "so whether it is served cannot be told"
            ),
            remedy=f"make {directory} readable and retry",
            verified=False,
        )
    holders = [
        holder
        for holder in (_serving_record_holder(Path(path)) for path in paths)
        if holder is not None
    ]
    verified = [holder for holder in holders if holder.verified]
    return (verified or holders or [None])[0]


def agent_holder(
    project_dir: Path, agent_name: str, data_dir: Path
) -> Optional[AgentHolder]:
    """Which process is serving this agent, or may be; None when none is.

    The launcher's records come first, because their remedy is the one a
    managed host needs: in the default in-process mode ``kestrel start``
    writes ONLY ``logs/.host.pid``, and ``kestrel terminate <agent>`` cannot
    stop an agent with no process of its own (#2920). A server no launcher
    record names is found by its own serving record.

    A launcher record that cannot be read counts as running too: it was
    written by a launch whose process may still be there. It is reported only
    when nothing proves an agent running, so it never hides a process that a
    working command stops.
    """
    from kestrel_sovereign.multi_agent.process_manager import PidStatus, ProcessManager

    unreadable: list[AgentHolder] = []
    for pid_file, command, label in (
        (
            ProcessManager.agent_pid_file(data_dir),
            f"kestrel terminate {agent_name}",
            "agent",
        ),
        (project_dir / "logs" / ".host.pid", "kestrel terminate", "host"),
    ):
        # The same verified read ``kestrel terminate`` uses, so the guard and
        # the remedy it prescribes cannot disagree about whether a launched
        # process is up. ``is_running`` counts an undecidable record as
        # running: it names a process that IS alive (#2995).
        record = ProcessManager.read_pid_record(pid_file)
        if record.is_running:
            return _launcher_holder(
                command,
                f"{label} PID {record.pid}, launched by `kestrel start`",
            )
        if record.status is PidStatus.UNREADABLE:
            unreadable.append(
                AgentHolder(
                    evidence=(
                        f"{record.detail}, so whether the {label} process it "
                        "records is still running cannot be told"
                    ),
                    remedy=(
                        f"`{command}` stops what `kestrel start` launched on "
                        f"its registered port; if nothing is running, delete "
                        f"{pid_file} and retry"
                    ),
                    verified=False,
                )
            )
    serving = serving_holder(data_dir)
    if serving is not None and serving.verified:
        return serving
    return (unreadable or [serving])[0]
