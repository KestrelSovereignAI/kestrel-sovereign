"""Fixed two-process TLS/Unix byte bridge; no RPC parser or action authority."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import socket
import ssl
import stat
import struct
import sys
import tempfile

from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, await_owned_operation, await_owned_task, create_owned_task,
    raise_owned_outcome,
)
from kestrel_sovereign.execution_custody import (
    execution_commit_outcome, execution_evidence_carriers,
    execution_terminal_error, is_execution_control_error,
)
from .worker_profile import IPC, MAX_REQUEST, MAX_RESPONSE, PROFILE, TOKEN
from .worker_stream import join_original_writer_close


class LaneRole(StrEnum):
    NORMAL = "normal"
    PRIORITY = "priority"

    @property
    def capacity(self):
        return 4 if self is LaneRole.NORMAL else 2

    @property
    def port(self):
        return 41401 if self is LaneRole.NORMAL else 41402

    @property
    def socket_name(self):
        return "worker.sock" if self is LaneRole.NORMAL else "control.sock"


def validate_configuration(value, *, role=None):
    required = {"profile", "identity", "browser_incarnation", "broker_incarnation",
                "profile_sha256", "expires_at", "client_ca", "client_sha256", "roles"}
    if (type(value) is not dict or set(value) != required or value["profile"] != PROFILE
            or any(type(value[k]) is not str or TOKEN.fullmatch(value[k]) is None
                   for k in ("identity", "browser_incarnation", "broker_incarnation", "profile_sha256", "client_sha256"))
            or type(value["roles"]) is not dict
            or set(value["roles"]) != ({role.value} if type(role) is LaneRole else {x.value for x in LaneRole})
            or type(value["expires_at"]) is not str
            or type(value["client_ca"]) is not str or len(value["client_ca"]) > 16384):
        raise ValueError("invalid_fixed_broker_configuration")
    expiry = datetime.fromisoformat(value["expires_at"])
    if expiry.utcoffset() is None or not 0 < (expiry - datetime.now(UTC)).total_seconds() <= 3600:
        raise ValueError("invalid_fixed_broker_expiry")
    for name, item in value["roles"].items():
        if (type(item) is not dict or set(item) != {"certificate", "key", "sha256"}
                or any(type(item[k]) is not str or len(item[k]) > 16384 for k in ("certificate", "key"))
                or type(item["sha256"]) is not str or TOKEN.fullmatch(item["sha256"]) is None):
            raise ValueError("invalid_fixed_broker_role")
    if role is not None and type(role) is not LaneRole:
        raise ValueError("exact_fixed_lane_role_required")
    return expiry


def server_context(value, role):
    validate_configuration(value, role=role)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cadata=value["client_ca"])
    # Original role secrets never enter argv/env/shared IPC. Files disappear
    # only after the native context has loaded them; no zeroization claim.
    with tempfile.TemporaryDirectory(prefix="worker-role-", dir="/tmp") as directory:
        files = []
        for key in ("certificate", "key"):
            p = Path(directory) / key
            with open(p, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
                stream.write(value["roles"][role.value][key])
            files.append(str(p))
        context.load_cert_chain(*files)
    return context


def require_peer(sock, uid, gid):
    if not hasattr(socket, "SO_PEERCRED"):
        raise RuntimeError("linux_original_peer_credentials_required")
    _pid, actual_uid, actual_gid = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    if (actual_uid, actual_gid) != (uid, gid):
        raise ValueError("original_unix_peer_differs")


def _node_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _original_mount(info):
    # Fixed read-only broker mount, not a path/UID assertion standing in for
    # the actual kernel mount. No arbitrary mount name or PID lookup is used.
    with open("/proc/self/mountinfo", "rb") as stream:
        raw = stream.read(262145)
    if len(raw) > 262144:
        raise ValueError("bounded_original_mount_required")
    found = []
    for line in raw.decode().splitlines():
        left, separator, right = line.partition(" - ")
        fields, filesystem = left.split(), right.split()
        if separator and len(fields) >= 6 and fields[4] == IPC:
            if (len(filesystem) < 3 or filesystem[0] != "tmpfs" or "ro" not in fields[5].split(",")
                    or fields[2] != str(os.major(info.st_dev))+":"+str(os.minor(info.st_dev))):
                raise ValueError("original_readonly_ipc_mount_differs")
            found.append(line)
    if len(found) != 1:
        raise ValueError("one_original_ipc_mount_required")
    return found[0]


def _current_ipc_root(fd):
    actual, anchored = os.lstat(IPC), os.fstat(fd)
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)
    return identity(actual), identity(anchored), _original_mount(anchored)


class _OriginalIPC:
    """One anchored mount/component/socket observation; never authority."""
    def __init__(self, value, role):
        self.fd = None
        self.value, self.role = value, role
        try:
            # Broker has search-only group permission on this 0710 root.
            # O_PATH anchors the original inode without requiring read/list.
            self.fd = os.open(IPC, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(self.fd)
            if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o710
                    or (info.st_uid, info.st_gid) != (10002, 10003)):
                raise ValueError("original_ipc_directory_differs")
            self.root = (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)
            self.mount = _original_mount(info)
            self.facts = self._facts()
            self.socket = self._socket()
            self.require_original()
        except BaseException:
            self.close()
            raise

    def _facts(self):
        observed = []
        for name, expected in (("incarnation", self.value["browser_incarnation"]), ("profile", self.value["profile_sha256"])):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
            try:
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or (info.st_uid, info.st_gid) != (10002, 10003)
                        or stat.S_IMODE(info.st_mode) != 0o640 or os.read(fd, 66).decode() != expected):
                    raise ValueError("original_ipc_identity_differs")
                observed.append(_node_identity(info))
            finally:
                os.close(fd)
        return tuple(observed)

    def _socket(self):
        info = os.stat(self.role.socket_name, dir_fd=self.fd, follow_symlinks=False)
        if (not stat.S_ISSOCK(info.st_mode) or (info.st_uid, info.st_gid) != (10002, 10003)
                or stat.S_IMODE(info.st_mode) != 0o660):
            raise ValueError("original_ipc_socket_differs")
        return _node_identity(info)

    def require_original(self):
        if self.fd is None:
            raise ValueError("original_ipc_descriptor_closed")
        actual, anchored, mount = _current_ipc_root(self.fd)
        if (actual != self.root or anchored != self.root
                or mount != self.mount or self._facts() != self.facts
                or self._socket() != self.socket):
            raise ValueError("original_ipc_component_replaced")

    @property
    def path(self):
        self.require_original()
        return "/proc/self/fd/"+str(self.fd)+"/"+self.role.socket_name

    def require_same(self, original):
        self.require_original()
        original.require_original()
        if (self.root, self.mount, self.facts, self.socket) != (original.root, original.mount, original.facts, original.socket):
            raise ValueError("original_lane_ipc_identity_changed")

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)


def require_ipc(value, role):
    return _OriginalIPC(value, role)


@dataclass(eq=False)
class _Slot:
    lane: object
    admission: OwnedTaskAdmission = field(default_factory=OwnedTaskAdmission)
    raw: object = None
    unix: object = None
    transport: object = None
    writer: object = None
    unix_writer: object = None
    ipc: object = None
    cancelled: bool = False
    started: bool = False
    retired: bool = False
    _cancel_token: object = field(default_factory=object, repr=False)
    failure: BaseException | None = None

    def cancel(self):
        task = self.admission.task
        if task is not None and not task.done() and not self.cancelled:
            self.cancelled = True
            task.cancel(self._cancel_token)

    def _own_revocation(self, error):
        """Only this slot's delivered cancellation can be a neutral drain.

        Its ordinary exchange cause is diagnostic; native/control evidence or
        any other BaseException still prevents a retirement acknowledgment.
        """
        return (self.cancelled and type(error) is asyncio.CancelledError
            and len(error.args) == 1 and error.args[0] is self._cancel_token
            and all(key in {"__notes__", "_kestrel_retained_traceback"} for key in vars(error))
            and not is_execution_control_error(error)
            and execution_commit_outcome(error) is None
            and all(node is error or isinstance(node, Exception)
                    for node in execution_evidence_carriers(error)))

    async def join(self):
        """Join the accepted Task and close a socket whose payload never ran."""
        outcome = await await_owned_task(self.admission.task)
        close_error = None
        if not self.started:
            self.failure = outcome.error
            try:
                if self.raw is not None:
                    self.raw.close()
                self.retired = True
            except BaseException as error:
                close_error = error
        return execution_terminal_error(_original_slot_error(self, outcome.error),
            outcome.cancellation, close_error, retain_originals=True)

    async def exchange(self):
        self.started = True
        failure = None
        try:
            async with asyncio.timeout(20):
                reader = asyncio.StreamReader(limit=MAX_REQUEST + 1)
                protocol = asyncio.StreamReaderProtocol(reader)
                async def upgrade():
                    self.transport, actual = await self.lane.loop.connect_accepted_socket(
                        lambda: protocol, self.raw, ssl=self.lane.context,
                        ssl_handshake_timeout=1, ssl_shutdown_timeout=1)
                    self.writer = asyncio.StreamWriter(self.transport, actual, reader, self.lane.loop)
                raise_owned_outcome(await await_owned_operation(upgrade(), retirement_resource=self),
                                    operation="original broker TLS open", retain_originals=True)
                self.lane.check()
                peer = self.writer.get_extra_info("ssl_object")
                if (peer is None or peer.version() != "TLSv1.3"
                        or hashlib.sha256(peer.getpeercert(binary_form=True)).hexdigest() != self.lane.value["client_sha256"]):
                    raise ValueError("original_broker_client_differs")
                async with asyncio.timeout(1 if self.lane.role is LaneRole.PRIORITY else 20):
                    request = await reader.readline()
                if not request.endswith(b"\n") or len(request) > MAX_REQUEST:
                    raise ValueError("bounded_original_worker_frame_required")
                self.ipc = require_ipc(self.lane.value, self.lane.role)
                self.ipc.require_same(self.lane.original_ipc)
                self.unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.unix.setblocking(False)
                async def connect():
                    await self.lane.loop.sock_connect(self.unix, self.ipc.path)
                    require_peer(self.unix, 10002, 10003)
                    self.ipc.require_original()
                    self.ipc.require_same(self.lane.original_ipc)
                    result = await asyncio.open_unix_connection(sock=self.unix, limit=MAX_RESPONSE + 1)
                    self.unix_writer = result[1]
                    return result[0]
                local = raise_owned_outcome(await await_owned_operation(connect(), retirement_resource=self),
                                           operation="original broker Unix open", retain_originals=True)
                self.lane.check()
                self.ipc.require_original()
                self.unix_writer.write(request)
                await self.unix_writer.drain()
                response = await local.readline()
                if not response.endswith(b"\n") or len(response) > MAX_RESPONSE:
                    raise ValueError("original_worker_response_unknown")
                self.lane.check()
                self.ipc.require_original()
                self.writer.write(response)
                await self.writer.drain()
        except BaseException as error:
            failure = error
        finally:
            # Capture every original closer before any can settle another.
            writers = tuple(x for x in (self.unix_writer, self.writer) if x is not None)
            closes = []
            for writer in writers:
                try:
                    writer.close()
                except BaseException as error:
                    closes.append(error)
            for writer in writers:
                outcome = await await_owned_operation(join_original_writer_close(writer),
                    retirement_resource=writer, retain_originals=True)
                if outcome.result is not None:
                    failure = execution_terminal_error(failure, outcome.result, retain_originals=True)
                if outcome.error is not None:
                    closes.append(outcome.error)
                if outcome.cancellation is not None:
                    if self._own_revocation(outcome.cancellation):
                        failure = execution_terminal_error(failure, outcome.cancellation,
                                                           retain_originals=True)
                    else:
                        closes.append(outcome.cancellation)
            for raw in (self.unix, self.raw):
                if raw is not None:
                    try:
                        raw.close()
                    except BaseException as error:
                        closes.append(error)
            if self.ipc is not None:
                try:
                    self.ipc.close()
                except BaseException as error:
                    closes.append(error)
            self.failure = failure
            if closes:
                raise execution_terminal_error(failure, *closes, retain_originals=True)
            self.retired = True
            if failure is not None and (is_execution_control_error(failure)
                    or execution_commit_outcome(failure) is not None
                    or (not isinstance(failure, Exception) and not self._own_revocation(failure))):
                raise failure
            # A lost exchange has no success response. Its original exception
            # is retained on this finite slot through the actual owning join.


def _original_slot_error(slot, error):
    # A lane's own single cancellation drains an UNKNOWN exchange; it never
    # fabricates an application response. Native cleanup/control debt still
    # prevents a successful process-retirement ACK.
    if slot.retired and slot._own_revocation(error):
        return None
    return error


class _BrokerLifetimeRetired(RuntimeError):
    """The original lifetime check reached normal admission withdrawal."""


class _BrokerLifetime:
    """One admitted wall deadline with a monotonic upper bound.

    A wall-clock step may shorten admission, but cannot renew its original
    monotonic budget. Timers and early-child decisions use this same boundary.
    """
    def __init__(self, expiry):
        self.expiry = expiry
        self.loop = asyncio.get_running_loop()
        self.deadline = self.loop.time() + max(0, (expiry - datetime.now(UTC)).total_seconds())
        self.retired = False

    def expired(self):
        self.retired = self.retired or datetime.now(UTC) >= self.expiry or self.loop.time() >= self.deadline
        return self.retired


class WorkerLane:
    """Reserve before accept; fixed live slots, no rejected-task/history pool."""
    def __init__(self, value, role, context):
        self.expiry = validate_configuration(value, role=role)
        self.lifetime = _BrokerLifetime(self.expiry)
        self.value, self.role, self.context = value, role, context
        self.loop = asyncio.get_running_loop()
        self.listener = None
        self.slots = [None] * role.capacity
        self.changed = asyncio.Event()
        self.active = True
        self.terminal = None
        self.last_exchange_failure = None
        self.original_ipc = None

    def check(self):
        if not self.active or self.lifetime.expired():
            raise _BrokerLifetimeRetired("original_broker_lifetime_retired")

    def _arm(self):
        if self.active and None in self.slots:
            self.loop.add_reader(self.listener.fileno(), self._accept_one)
        elif self.listener is not None:
            self.loop.remove_reader(self.listener.fileno())

    def _accept_one(self):
        slot = None
        try:
            try:
                self.check()
            except _BrokerLifetimeRetired:
                self.revoke()
                return
            if None not in self.slots:
                self._arm()
                return
            index = self.slots.index(None)
            slot = self.slots[index] = _Slot(self)  # Reserve BEFORE native accept.
            slot.raw, _address = self.listener.accept()
            slot.raw.setblocking(False)
            task = create_owned_task(slot.exchange(), admission=slot.admission,
                retirement_resource=slot, retain_originals=True, name="original-worker-lane-slot")
            task.add_done_callback(lambda _task: self.changed.set())
        except BlockingIOError:
            self.slots[index] = None
        except BaseException as error:
            self.terminal = execution_terminal_error(self.terminal, error, retain_originals=True)
            self.revoke()
        self._arm()
        self.changed.set()

    def revoke(self):
        self.active = False
        if self.listener is not None:
            self.loop.remove_reader(self.listener.fileno())
        for slot in self.slots:
            if slot is not None:
                slot.cancel()
        self.changed.set()

    async def _close_originals(self, failure=None):
        # Capture before a fallible listener close; neither that failure nor
        # one failed writer may abandon another accepted original join.
        joins = tuple((s, s.admission.task) for s in self.slots if s is not None)
        try:
            self.revoke()
        except BaseException as error:
            self.terminal = execution_terminal_error(self.terminal, error, retain_originals=True)
        if self.listener is not None:
            try:
                self.listener.close()
            except BaseException as error:
                self.terminal = execution_terminal_error(self.terminal, error, retain_originals=True)
        for slot, task in joins:
            if task is not None:
                self.terminal = execution_terminal_error(self.terminal, await slot.join(), retain_originals=True)
            elif slot.raw is not None:
                try:
                    slot.raw.close()
                except BaseException as error:
                    self.terminal = execution_terminal_error(self.terminal, error, retain_originals=True)
        if self.original_ipc is not None:
            try:
                self.original_ipc.close()
            except BaseException as error:
                self.terminal = execution_terminal_error(self.terminal, error, retain_originals=True)
        if failure is not None or self.terminal is not None:
            raise execution_terminal_error(failure, self.terminal, retain_originals=True)

    async def serve(self):
        if sys.platform != "linux" or not hasattr(socket, "SO_PEERCRED"):
            raise RuntimeError("supported_linux_worker_broker_required")
        failure = None
        try:
            self.original_ipc = require_ipc(self.value, self.role)
            self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.listener.setblocking(False)
            self.listener.bind(("0.0.0.0", self.role.port))
            self.listener.listen(self.role.capacity)
            self._arm()
            (Path("/tmp/kestrel-broker")/(self.role.value+".ready")).write_text("ready")
            timer = self.loop.call_at(self.lifetime.deadline, self.revoke)
            try:
                while self.active or any(x is not None for x in self.slots):
                    await self.changed.wait()
                    self.changed.clear()
                    for index, slot in enumerate(self.slots):
                        if slot is None:
                            continue
                        task = slot.admission.task
                        if task is not None and task.done():
                            error = await slot.join()
                            if error is not None:
                                self.terminal = execution_terminal_error(self.terminal, error, retain_originals=True)
                                self.revoke()
                            self.last_exchange_failure = slot.failure
                            self.slots[index] = None
                        elif task is None:
                            # An unpublished/unknown native admission is debt;
                            # this lane cannot recycle its accepted descriptor.
                            self.revoke()
                            raise RuntimeError("original_lane_admission_unresolved")
                    self._arm()
            finally:
                timer.cancel()
        except BaseException as error:
            failure = error
        finally:
            await self._close_originals(failure)


async def _wait_roles_ready(children, retirement):
    async with asyncio.timeout(30):
        while not all((Path("/tmp/kestrel-broker")/(role.value+".ready")).is_file() for role in LaneRole):
            if retirement.is_set():
                return
            if any(child.returncode is not None for child in children):
                raise RuntimeError("original_broker_child_ended_before_ready")
            await asyncio.sleep(.05)
    (Path("/tmp/kestrel-broker")/"ready").write_text("ready")


async def supervise(value, program, *, _retirement=None):
    """Two fixed original Process handles; no restart/replacement or RPC logic."""
    lifetime = _BrokerLifetime(validate_configuration(value))
    expected = Path(__file__).with_name("_worker_programs") / "broker_worker.py"
    if Path(program).resolve() != expected.resolve():
        raise ValueError("original_installed_broker_program_required")
    children, waiters = [], []
    loop = asyncio.get_running_loop()
    if _retirement is not None and type(_retirement) is not asyncio.Event:
        raise ValueError("original_broker_lifecycle_event_required")
    shutdown = _retirement if _retirement is not None else asyncio.Event()
    def withdraw():
        shutdown.set()
    failure = None
    timer = None
    try:
        timer = loop.call_at(lifetime.deadline, withdraw)
        for role in LaneRole:
            if shutdown.is_set() or lifetime.expired():
                shutdown.set()
                break
            child = None
            async def launch(role=role):
                nonlocal child
                child = await asyncio.create_subprocess_exec(sys.executable, "-B", str(Path(__file__).with_name("worker_entry.py")),
                    "broker", "--lane", role.value,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL, env={"HOME": os.environ.get("HOME", ""), "PATH": os.defpath,
                        "PYTHON_DOTENV_DISABLED": "1"}, close_fds=True)
                children.append(child)  # Before fallible result publication.
            raise_owned_outcome(await await_owned_operation(launch(), retirement_resource=children),
                                operation="original broker child launch", retain_originals=True)
            child_value = {**value, "roles": {role.value: value["roles"][role.value]}}
            child.stdin.write(json.dumps(child_value, separators=(",", ":")).encode())
            await child.stdin.drain()
            child.stdin.close()
        if shutdown.is_set():
            return
        for child in children:
            admission = OwnedTaskAdmission()
            waiters.append(admission)  # Before Task result publication.
            create_owned_task(child.wait(), admission=admission, retirement_resource=child,
                              retain_originals=True, name="original-broker-child-wait")
        await _wait_roles_ready(children, shutdown)
        stop = OwnedTaskAdmission()
        waiters.append(stop)
        create_owned_task(shutdown.wait(), admission=stop, retirement_resource=shutdown,
                          retain_originals=True, name="original-broker-lifecycle-withdrawal")
        done, _pending = await asyncio.wait(tuple(x.task for x in waiters), return_when=asyncio.FIRST_COMPLETED)
        # A dead lane ends the whole broker, including a priority lane which
        # exited zero early. It cannot leave its peer accepting until expiry.
        early = not lifetime.expired()
        if not shutdown.is_set() and (early or any(task.result() != 0 for task in done)):
            raise RuntimeError("original_broker_child_ended")
    except BaseException as error:
        failure = error
    finally:
        shutdown.set()
        errors = []
        if timer is not None:
            try:
                timer.cancel()
            except BaseException as error:
                errors.append(error)
        for child in children:
            try:
                child.stdin.close()
            except BaseException as error:
                errors.append(error)
            try:
                if child.returncode is None:
                    child.terminate()
            except ProcessLookupError:
                pass
            except BaseException as error:
                errors.append(error)
        for child in children:
            outcome = await await_owned_operation(child.wait(), retirement_resource=child)
            errors.extend(x for x in (outcome.error, outcome.cancellation) if x is not None)
            if outcome.error is None and outcome.result != 0:
                errors.append(RuntimeError("original_broker_child_native_retirement_unknown"))
        for admission in waiters:
            if admission.task is not None:
                outcome = await await_owned_task(admission.task)
                errors.extend(x for x in (outcome.error, outcome.cancellation) if x is not None)
        if failure is not None or errors:
            raise execution_terminal_error(failure, *errors, retain_originals=True)
