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
a :class:`ServingRecord` into its data directory as it boots, and removes it
only once every resource it acquired has a confirmed release (see
:mod:`kestrel_sovereign.agent.custody`); until then the record outlives the
agent and goes stale when its process exits. There is one file per agent
instance, so two holders of one directory, or an in-process restart whose old
and new agent overlap, never remove each other's record.

A record is evidence of liveness, never of absence. An agent is stopped only
when no launcher record and no serving record names a process that may still
be running. A serving record names nothing running only on positive proof
(:func:`_provably_stale`): it reads, every field it identifies its process by
parses, it was written on this host and in this PID namespace, and nothing
runs as its PID here or what does started at another instant. Any other record
counts as running, and is never deleted: one written on another host or in
another PID namespace (a container, or a machine sharing the data directory
under another hostname), one that cannot be read, one missing a field or
holding a value that is not one (a PID that is not a positive integer, a start
time that is ``NaN``, infinite or not a number), and one naming a PID whose
start time cannot be checked. Waving a guard past a live agent costs more than
a refusal an operator can clear (#2995). A record is attributed by hostname
and PID namespace only, so two machines sharing one data directory under the
same hostname cannot be told apart.

Not a port probe. A direct launch listens wherever its operator said, and
whatever answers on a registered port need not serve this project: the public
``/health`` deliberately names no agent.
"""

from __future__ import annotations

import errno
import json
import os
import secrets
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

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


class ProcessIdentityUnavailable(RuntimeError):
    """The operating system cannot identify this original physical owner."""


def _kernel_session_id() -> str:
    """The kernel's random boot UUID, never a wall-clock or host-name proxy.

    Linux exposes its boot_id through procfs; Darwin exposes bootsessionuuid
    through sysctlbyname. An unavailable interface is an explicit capability
    refusal. No weaker identity substitutes for it.
    """
    if sys.platform.startswith("linux"):
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    elif sys.platform == "darwin":
        import ctypes

        sysctl = ctypes.CDLL(None, use_errno=True).sysctlbyname
        sysctl.argtypes = (
            ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p, ctypes.c_size_t,
        )
        sysctl.restype = ctypes.c_int
        size = ctypes.c_size_t()
        key = b"kern.bootsessionuuid"
        if sysctl(key, None, ctypes.byref(size), None, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        buffer = ctypes.create_string_buffer(size.value)
        if sysctl(key, buffer, ctypes.byref(size), None, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        value = buffer.value.decode("ascii")
    else:
        raise OSError(errno.ENOTSUP, "Physical process retirement requires a supported kernel boot UUID")
    return f"{sys.platform}:{UUID(value)}"


def process_identity(*, require_kernel_session: bool = False) -> dict[str, Any]:
    """This process, as a record names it: PID, start instant, host, namespace.

    Every field :func:`recorded_process_exited` checks. The start instant is
    None when it cannot be read, and a record holding None never proves its
    process exited.

    Strict mode additionally requires the kernel boot UUID and a raw kernel
    birth identity. Its epoch value is diagnostic and never establishes an
    exited process or a changed current owner.
    """
    from kestrel_sovereign.multi_agent.process_manager import ProcessManager

    pid = os.getpid()
    host, pid_namespace = _here()
    identity = {
        "pid": pid,
        "started_at": ProcessManager.process_start_time(pid),
        "host": host,
        "pid_namespace": pid_namespace,
    }
    if require_kernel_session:
        if _parse_record(identity) is None or (
            sys.platform.startswith("linux") and not pid_namespace
        ):
            raise ProcessIdentityUnavailable("Original process PID, start time or namespace is unavailable")
        try:
            identity["kernel_session"] = _kernel_session_id()
            identity["birth_id"] = ProcessManager.process_birth_identity(pid)
        except (OSError, ValueError) as error:
            raise ProcessIdentityUnavailable("Original kernel process identity is unavailable") from error
    return identity


class ServingRecord:
    """This process's record that it serves the agent in one data directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._database_records: list[ServingRecord] = []

    @classmethod
    def acquire(
        cls, data_dir: Path, *, storage_path: Path | None = None
    ) -> ServingRecord:
        """Publish profile and canonical database ownership before boot.

        A database symlink may point outside the profile. Both directories
        must name the serving process so offline tools can select either path.
        """
        primary = cls._acquire_one(data_dir)
        try:
            if storage_path is not None:
                database_dir = Path(storage_path).expanduser().resolve().parent
                if database_dir != Path(data_dir).resolve():
                    primary._database_records.append(cls._acquire_one(database_dir))
        except BaseException as error:
            try:
                primary.release()
            except OSError as cleanup_error:
                from kestrel_sovereign.execution_custody import execution_terminal_error

                raise execution_terminal_error(
                    error, cleanup_error, retain_originals=True
                )
            raise
        return primary

    @classmethod
    def _acquire_one(cls, data_dir: Path) -> ServingRecord:
        """Record that this process serves the agent in ``data_dir``.

        The record is published whole (written aside, then renamed), so a
        reader never mistakes a half-written file for an unreadable one.
        Records this host can prove stale are removed first.

        Raises:
            OSError: The record cannot be written.
        """
        directory = Path(data_dir) / SERVING_RECORD_DIRNAME
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _remove_stale_records(directory)
        payload = {**process_identity(), "data_dir": str(data_dir)}
        path = directory / f"{payload['pid']}-{secrets.token_hex(8)}.json"
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
        # Retain the profile record if an alias record could not be released;
        # a caller may retry, and no guard may infer successful cleanup.
        for record in self._database_records:
            record.release()
        self.path.unlink(missing_ok=True)


def _record_paths(directory: Path) -> list[Path]:
    """The serving records in ``directory``, as the guard and the sweep see them.

    Staging files (``.<name>.tmp``) are not records.

    Raises:
        OSError: ``directory`` exists but cannot be listed.
    """
    try:
        return sorted(
            Path(entry.path)
            for entry in os.scandir(directory)
            if entry.name.endswith(".json") and not entry.name.startswith(".")
        )
    except FileNotFoundError:
        return []


def _released(path: Path, payload) -> bool:
    """Whether a listed record was removed before it could be read.

    A directory entry that is still there but reads as missing (a symbolic
    link to nothing) was not released: it is a record that cannot be read.
    """
    return isinstance(payload, FileNotFoundError) and not os.path.lexists(path)


def _remove_stale_records(directory: Path) -> None:
    """Delete the records :func:`_provably_stale` proves name an exited process.

    A directory that cannot be listed deletes nothing.
    """
    try:
        paths = _record_paths(directory)
    except OSError:
        return
    for path in paths:
        payload = _read_payload(path)
        if _released(path, payload):
            continue
        if _provably_stale(path, payload):
            path.unlink(missing_ok=True)


def _read_payload(path: Path):
    """The record's JSON object, or the reason it cannot be read."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        # ValueError includes JSON with an integer too long to convert.
        return exc
    if not isinstance(payload, dict):
        return ValueError("not a JSON object")
    return payload


@dataclass(frozen=True)
class _RecordedProcess:
    """The fields a serving record identifies the process it names by."""

    pid: int
    started_at: float
    host: str
    pid_namespace: Optional[str]
    birth_id: str | None = None


def _parse_record(payload: dict[str, Any]) -> Optional[_RecordedProcess]:
    """Every identifying field of a record, or None unless each one parses.

    ``pid`` and ``started_at`` through the same validators a launcher record
    is read with (:func:`recorded_pid`, :func:`recorded_start_time`). ``host``
    is a non-empty string. ``pid_namespace`` must be present, as every record
    carries it: None where the platform has no PID namespaces, a non-empty
    namespace name where it does. A record whose own start time could not be
    read when it was written holds None there, so it never parses: nothing
    could ever prove it stale.
    """
    from kestrel_sovereign.multi_agent.process_manager import (
        recorded_birth_identity,
        recorded_pid,
        recorded_start_time,
    )

    if "pid_namespace" not in payload:
        return None
    pid = recorded_pid(payload.get("pid"))
    started_at = recorded_start_time(payload.get("started_at"))
    host = payload.get("host")
    pid_namespace = payload["pid_namespace"]
    if pid is None or started_at is None:
        return None
    if not isinstance(host, str) or not host:
        return None
    if pid_namespace is not None and (
        not isinstance(pid_namespace, str) or not pid_namespace
    ):
        return None
    birth_id = recorded_birth_identity(payload.get("birth_id"))
    if "birth_id" in payload and birth_id is None:
        return None
    return _RecordedProcess(pid, started_at, host, pid_namespace, birth_id)


def same_process_identity(first: Any, second: Any) -> bool:
    """Match strict kernel identities; hostname and epoch are diagnostics."""
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    before, after = _parse_record(first), _parse_record(second)
    if before is None or after is None or before.birth_id is None or after.birth_id is None:
        return False
    kernel = first.get("kernel_session")
    return (
        isinstance(kernel, str) and bool(kernel)
        and kernel == second.get("kernel_session")
        and (before.pid, before.pid_namespace, before.birth_id)
        == (after.pid, after.pid_namespace, after.birth_id)
    )


def _unprovable(path: Path, evidence: str) -> AgentHolder:
    return AgentHolder(
        evidence=evidence,
        remedy=(f"stop the process serving it; if none is, delete {path} and retry"),
        verified=False,
    )


def _record_holder(path: Path, payload, *, require_kernel_session: bool = False) -> Optional[AgentHolder]:
    """What one serving record establishes: None only on proof that it is stale.

    The record must read; every identifying field must parse
    (:func:`_parse_record`); it must have been written on this host and in
    this PID namespace, since a PID means a process only in the table it was
    issued from; and then nothing may run as its PID here, or what does must
    have started at another instant. Each of those is checked here, once, so
    the guard (:func:`_serving_record_holder`) and the stale-record sweep
    (:func:`_remove_stale_records`) cannot disagree. Any record not proven
    stale comes back as a holder, verified only when the process it names is
    running and is the one it recorded.
    """
    from kestrel_sovereign.multi_agent.process_manager import PidStatus, ProcessManager

    if isinstance(payload, Exception):
        return _unprovable(
            path,
            f"its serving record {path} cannot be read ({payload}), so "
            "whether it is served cannot be told",
        )
    recorded = _parse_record(payload)
    if recorded is None:
        return _unprovable(
            path,
            f"{path} does not record a valid PID, start time, host and PID "
            "namespace, so whether its process still serves it cannot be told",
        )
    if require_kernel_session:
        kernel = payload.get("kernel_session")
        if not isinstance(kernel, str) or not kernel:
            return _unprovable(path, "original kernel session identity is unavailable")
        if sys.platform.startswith("linux") and not recorded.pid_namespace:
            return _unprovable(path, "original kernel PID namespace is unavailable")
        try:
            if kernel != _kernel_session_id():
                return _unprovable(path, "original process belongs to a different kernel session")
        except (OSError, ValueError) as error:
            return _unprovable(path, f"original kernel session cannot be observed ({error})")
    host, pid_namespace = _here()
    # Kernel session UUID + PID namespace identify strict containment. The OS
    # hostname can change during a live process and carries no strict authority.
    host_mismatch = not require_kernel_session and recorded.host != host
    if host_mismatch or recorded.pid_namespace != pid_namespace:
        where = (
            f"on host {recorded.host!r}"
            if host_mismatch
            else f"in PID namespace {recorded.pid_namespace!r}"
        )
        return _unprovable(
            path,
            f"{path} records PID {recorded.pid} serving it {where}, which "
            "cannot be checked from here",
        )
    if require_kernel_session:
        status, detail = ProcessManager.identify_recorded_process(
            recorded.pid, recorded.started_at, path.name,
            recorded_birth=recorded.birth_id, require_birth=True,
        )
    else:
        status, detail = ProcessManager.identify_recorded_process(
            recorded.pid, recorded.started_at, path.name
        )
    if status is PidStatus.STALE:
        return None
    if status is PidStatus.LIVE:
        command = _stop_command(recorded.pid)
        return AgentHolder(
            evidence=(
                f"PID {recorded.pid} serves it, started without `kestrel "
                f"start`; recorded in {path}"
            ),
            remedy=f"`{command}` stops it",
            command=command,
        )
    return _unprovable(path, f"{detail}; recorded in {path}")


def _provably_stale(path: Path, payload) -> bool:
    """Whether a serving record may be deleted: positive proof it is stale.

    The sweep deletes exactly what this proves. The guard reads the same
    judgement: :func:`_record_holder` returns None exactly when this is True,
    and reports every other record as a process that may serve the agent.
    """
    return _record_holder(path, payload) is None


def recorded_process_exited(
    path: Path, payload: Any, *, require_kernel_session: bool = False
) -> bool:
    """Whether the process ``payload`` names has provably exited.

    ``payload`` carries the fields :func:`process_identity` writes, read from
    the record at ``path`` (named in diagnostics only). The same judgement a
    serving record is swept by (:func:`_provably_stale`), for any record that
    names the process owning something: True only when the record parses, was
    written on this host and in this PID namespace, and nothing runs as its
    PID here or what does started at another instant.

    Strict kernel mode uses the same boot UUID, PID namespace and original
    birth identity. Hostname and epoch changes cannot prove or hide an exit.
    """
    if not isinstance(payload, dict):
        return False
    return _record_holder(path, payload, require_kernel_session=require_kernel_session) is None


def _serving_record_holder(path: Path) -> Optional[AgentHolder]:
    """What one serving record establishes, or None when it names nothing live."""
    payload = _read_payload(path)
    if _released(path, payload):
        return None
    return _record_holder(path, payload)


def serving_holder(data_dir: Path) -> Optional[AgentHolder]:
    """The process a serving record says serves the agent in ``data_dir``.

    A record this host can verify wins over one it cannot, so the remedy names
    a command whenever there is one.
    """
    directory = Path(data_dir) / SERVING_RECORD_DIRNAME
    try:
        paths = _record_paths(directory)
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
        for holder in (_serving_record_holder(path) for path in paths)
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
