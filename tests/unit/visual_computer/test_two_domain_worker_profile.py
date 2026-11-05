"""Pure profile and scripted broker-fault proofs, not native conformance.

The scripted Process/listener/writer objects below have no native PID/socket.
Only their original asyncio Tasks are real. Linux mount/UID/TLS/container and
physical process proofs belong to the separately authorized native phase.
"""
import asyncio
import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest
from kestrel_sdk.resources import ResourceUnsupported
from kestrel_sovereign._async_ownership import OwnedTaskAdmission, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.visual_computer import worker_broker as broker
from kestrel_sovereign.visual_computer.worker_profile import (
    CONTRACT, DATA, DRIVER, LABEL, OPERATIONS, PROFILE, LocalWorkerProfile,
    container_arguments, host_configuration, verify_container,
)


def settings():
    return {"profile": PROFILE, "identity": "a"*64, "browser_incarnation": "b"*64,
        "broker_incarnation": "c"*64, "profile_sha256": "d"*64,
        "expires_at": (datetime.now(UTC)+timedelta(minutes=10)).isoformat(),
        "client_ca": "scripted certificate, never used by TLS", "client_sha256": "e"*64,
        "roles": {role: {"certificate": "scripted leaf", "key": "scripted key", "sha256": "f"*64}
                  for role in ("normal", "priority")}}


def child_settings(role):
    value = settings()
    return {**value, "roles": {role.value: value["roles"][role.value]}}


def test_closed_roles_profiles_and_absent_extensions():
    profile = LocalWorkerProfile("browser@sha256:"+"a"*64, "broker@sha256:"+"b"*64)
    assert "files" not in OPERATIONS and "navigate" not in OPERATIONS
    assert DRIVER != "local-docker-browser"
    browser = container_arguments(profile, "browser", "c"*64)
    priority = host_configuration("broker", "c"*64)
    assert browser[browser.index("--network")+1] == "none"
    assert "--publish" not in browser
    assert "--pull=never" in browser
    assert priority["NetworkMode"] == "kestrel-worker-net-"+"c"*64
    assert priority["ReadonlyRootfs"] and priority["Mounts"][0]["ReadOnly"]
    assert broker.LaneRole.NORMAL.capacity == 4
    assert broker.LaneRole.PRIORITY.capacity == 2
    with pytest.raises(ResourceUnsupported):
        container_arguments(profile, "arbitrary-role", "c"*64)


@pytest.mark.parametrize("changed", ["roles", "expiry", "identity", "extra"])
def test_broker_bootstrap_refuses_noncanonical_shapes(changed):
    value = settings()
    if changed == "roles":
        value["roles"] = ["normal", "priority"]
    elif changed == "expiry":
        value["expires_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
    elif changed == "identity":
        value["identity"] = "bad"
    else:
        value["caller_checker"] = "not a grant"
    with pytest.raises((ValueError, TypeError)):
        broker.validate_configuration(value)


def test_lane_child_receives_only_its_original_role_key():
    value = settings()
    for role in broker.LaneRole:
        child = {**value, "roles": {role.value: value["roles"][role.value]}}
        broker.validate_configuration(child, role=role)
        with pytest.raises(ValueError):
            broker.validate_configuration(value, role=role)


@pytest.mark.parametrize("replacement", ["root", "descriptor", "mount", "fact-inode", "socket"])
def test_anchored_ipc_replacement_refuses_before_returning_connect_path(monkeypatch, replacement):
    # Scripted kernel observations only; no directory descriptor or Unix
    # socket is opened and no Linux identity is claimed by this unit test.
    original = object.__new__(broker._OriginalIPC)
    original.fd, original.role = 17, broker.LaneRole.PRIORITY
    original.root, original.mount = (1, 2, 3, 10002, 10003), "original read-only mount"
    original.facts, original.socket = (("original fact inode",),), ("original socket inode",)
    current = [original.root, original.root, original.mount]
    original._facts = lambda: original.facts
    original._socket = lambda: original.socket
    monkeypatch.setattr(broker, "_current_ipc_root", lambda fd: tuple(current))
    assert original.path == "/proc/self/fd/17/control.sock"
    if replacement in {"root", "descriptor"}:
        current[0 if replacement == "root" else 1] = (1, 9, 3, 10002, 10003)
    elif replacement == "mount":
        current[2] = "replacement mount"
    elif replacement == "fact-inode":
        original._facts = lambda: (("replacement fact inode, same content",),)
    else:
        original._socket = lambda: ("replacement socket inode",)
    with pytest.raises(ValueError, match="original_ipc_component_replaced"):
        _ = original.path


def test_search_only_ipc_anchor_uses_opath_without_directory_read(monkeypatch):
    """Scripted flags; actual Linux permission result is a separate proof."""
    calls, closed = [], []
    info = SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFDIR | 0o710, st_uid=10002, st_gid=10003)
    identity = (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)
    def opened(path, flags):
        calls.append((path, flags))
        return 17
    flags = SimpleNamespace(O_RDONLY=0, O_PATH=0x200000, O_DIRECTORY=0x10000, O_NOFOLLOW=0x20000,
                            open=opened, fstat=lambda _fd: info, close=closed.append)
    monkeypatch.setattr(broker, "os", flags)
    monkeypatch.setattr(broker, "_original_mount", lambda _info: "original ro tmpfs")
    monkeypatch.setattr(broker, "_current_ipc_root", lambda _fd: (identity, identity, "original ro tmpfs"))
    monkeypatch.setattr(broker._OriginalIPC, "_facts", lambda _self: ("original fact",))
    monkeypatch.setattr(broker._OriginalIPC, "_socket", lambda _self: "original socket")
    anchor = broker._OriginalIPC(settings(), broker.LaneRole.NORMAL)
    try:
        assert calls == [(broker.IPC, flags.O_PATH | flags.O_DIRECTORY | flags.O_NOFOLLOW)]
    finally:
        anchor.close()
    assert closed == [17]


@pytest.mark.asyncio
async def test_full_lane_never_calls_accept_or_allocates_another_slot():
    lane = broker.WorkerLane(child_settings(broker.LaneRole.PRIORITY), broker.LaneRole.PRIORITY, object())
    calls = []
    lane.slots = [object(), object()]
    lane.listener = SimpleNamespace(fileno=lambda: -1, accept=lambda: calls.append("accept"))
    lane._arm = lambda: calls.append("disarm")
    lane._accept_one()
    assert calls == ["disarm"]
    assert len(lane.slots) == 2


class ScriptedCloseError(Exception):
    pass


@pytest.mark.asyncio
async def test_listener_close_failure_still_joins_every_original_slot():
    lane = broker.WorkerLane(child_settings(broker.LaneRole.PRIORITY), broker.LaneRole.PRIORITY, object())
    entered, releases, joined, tasks = [], [asyncio.Event(), asyncio.Event()], [], []
    for index in range(2):
        admission = OwnedTaskAdmission()
        async def original(index=index):
            entered.append(index)
            await releases[index].wait()
            joined.append(index)
        task = create_owned_task(original(), admission=admission, name="scripted-original-slot")
        tasks.append(task)
        # No native resources in this test. Revoke closes admission; the
        # scripted listener supplies the actual gates before raising.
        lane.slots[index] = broker._Slot(lane, admission=admission, started=True)
    await asyncio.sleep(0)
    failure = ScriptedCloseError("original listener close")
    def close():
        for event in releases:
            event.set()
        raise failure
    lane.listener = SimpleNamespace(close=close)
    lane.revoke = lambda: None
    try:
        with pytest.raises(BaseException) as caught:
            await lane._close_originals()
        assert failure in tuple(_explicit_control_chain(caught.value))
        assert sorted(joined) == sorted(entered) == [0, 1]
        assert all(task.done() for task in tasks)
    finally:
        for event in releases:
            event.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_writer_close_failure_keeps_other_original_waits():
    waits, closes = [], []
    class Writer:
        def __init__(self, name, fails=False):
            self.name, self.fails = name, fails
        def close(self):
            closes.append(self.name)
            if self.fails:
                raise ScriptedCloseError("original writer close")
        async def wait_closed(self):
            waits.append(self.name)
    class Loop:
        async def connect_accepted_socket(self, *args, **kwargs):
            raise ValueError("scripted original TLS open refusal")
    slot = broker._Slot(SimpleNamespace(loop=Loop(), context=object()))
    slot.writer, slot.unix_writer = Writer("tls", True), Writer("unix")
    with pytest.raises(BaseException) as caught:
        await slot.exchange()
    assert any(type(item) is ScriptedCloseError and str(item) == "original writer close"
               for item in _explicit_control_chain(caught.value))
    assert sorted(closes) == sorted(waits) == ["tls", "unix"]
    assert str(slot.failure) == "scripted original TLS open refusal"


class ScriptedProcess:
    """No OS Process: exercise original handle retention/joins only."""
    def __init__(self, role):
        self.role, self.returncode = role, None
        self.exited, self.joined = asyncio.Event(), 0
        self.stdin = SimpleNamespace(write=lambda raw: None, close=lambda: None, drain=self.drain)
        self.termination = 0
    async def drain(self):
        return None
    async def wait(self):
        if self.role == "normal" and self.returncode is None:
            self.returncode = 7
            self.exited.set()
        await self.exited.wait()
        self.joined += 1
        return self.returncode
    def terminate(self):
        self.termination += 1
        self.returncode = -15
        self.exited.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("publication_failure", [False, True])
async def test_original_supervisor_retires_and_joins_peer_on_first_exit_or_publication_error(monkeypatch, publication_failure):
    children, tasks = [], []
    async def launch(*argv, **kwargs):
        child = ScriptedProcess(argv[-1])
        children.append(child)
        return child
    async def ready(originals, retirement):
        assert originals == children
    real_raise = broker.raise_owned_outcome
    def publish(outcome, **kwargs):
        if publication_failure and kwargs["operation"] == "original broker child launch":
            raise ScriptedCloseError("accepted child publication failed")
        return real_raise(outcome, **kwargs)
    real_create = broker.create_owned_task
    def capture(*args, **kwargs):
        task = real_create(*args, **kwargs)
        tasks.append(task)
        return task
    monkeypatch.setattr(broker.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(broker, "_wait_roles_ready", ready)
    monkeypatch.setattr(broker, "raise_owned_outcome", publish)
    monkeypatch.setattr(broker, "create_owned_task", capture)
    program = Path(broker.__file__).with_name("_worker_programs")/"broker_worker.py"
    with pytest.raises(BaseException):
        await broker.supervise(settings(), program)
    assert len(children) == (1 if publication_failure else 2)
    assert all(child.joined > 0 and child.returncode is not None for child in children)
    assert all(task.done() for task in tasks)
    if not publication_failure:
        assert children[1].termination == 1


@pytest.mark.asyncio
async def test_original_supervisor_stdin_close_error_cannot_skip_peer_retirement(monkeypatch):
    children, tasks, terminated = [], [], asyncio.Event()
    failure = ScriptedCloseError("scripted original child stdin close")
    async def launch(*argv, **kwargs):
        child = ScriptedProcess(argv[-1])
        children.append(child)
        closes = 0
        def close():
            nonlocal closes
            closes += 1
            if child.role == "priority" and closes == 2:
                raise failure
        original_terminate = child.terminate
        def terminate():
            terminated.set()
            original_terminate()
        child.stdin.close, child.terminate = close, terminate
        return child
    async def ready(*_args):
        return None
    real_create = broker.create_owned_task
    def capture(*args, **kwargs):
        task = real_create(*args, **kwargs)
        tasks.append(task)
        return task
    monkeypatch.setattr(broker.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(broker, "_wait_roles_ready", ready)
    monkeypatch.setattr(broker, "create_owned_task", capture)
    program = Path(broker.__file__).with_name("_worker_programs")/"broker_worker.py"
    owner_admission = OwnedTaskAdmission()
    owner = create_owned_task(broker.supervise(settings(), program), admission=owner_admission,
                              name="scripted-original-broker-owner")
    observed = False
    try:
        try:
            await asyncio.wait_for(terminated.wait(), .1)
            observed = True
        except TimeoutError:
            pass
    finally:
        # Release only the fixture's scripted children, then physically join
        # every actual Task. This is not native process cleanup evidence.
        for child in children:
            child.returncode = -15 if child.returncode is None else child.returncode
            child.exited.set()
        outcome = await broker.await_owned_task(owner)
        await asyncio.gather(*tasks, return_exceptions=True)
    assert observed
    assert failure in tuple(_explicit_control_chain(outcome.error))
    assert all(child.joined > 0 for child in children)


# Native04's safe browser-created projection: every HostConfig field and the
# complete daemon record mount shape. The redacted seccomp item is replaced
# below by the original packaged policy; image Env/labels are SCRIPTED data.
# This is portable parser/profile coverage, never native session conformance.
_MEASURED_BROWSER_HOST_JSON = r'''{
  "Binds": null,
  "ContainerIDFile": "",
  "LogConfig": {
    "Type": "none",
    "Config": {}
  },
  "NetworkMode": "none",
  "PortBindings": {},
  "RestartPolicy": {
    "Name": "no",
    "MaximumRetryCount": 0
  },
  "AutoRemove": false,
  "VolumeDriver": "",
  "VolumesFrom": null,
  "ConsoleSize": [
    0,
    0
  ],
  "CapAdd": null,
  "CapDrop": [
    "ALL"
  ],
  "CgroupnsMode": "private",
  "Dns": null,
  "DnsOptions": [],
  "DnsSearch": [],
  "ExtraHosts": null,
  "GroupAdd": null,
  "IpcMode": "private",
  "Cgroup": "",
  "Links": null,
  "OomScoreAdj": 0,
  "PidMode": "",
  "Privileged": false,
  "PublishAllPorts": false,
  "ReadonlyRootfs": true,
  "SecurityOpt": [
    "no-new-privileges",
    {
      "type": "str",
      "serialized_bytes": 10504,
      "sha256": "2d209e671c50fbb73ebb51cf6aa3711e07dd30c724bebf209949213bde34dacc"
    }
  ],
  "Tmpfs": {
    "/tmp": "rw,nosuid,nodev,size=256m,mode=1777"
  },
  "UTSMode": "",
  "UsernsMode": "",
  "ShmSize": 268435456,
  "Runtime": "runc",
  "Isolation": "",
  "CpuShares": 0,
  "Memory": 1073741824,
  "NanoCpus": 2000000000,
  "CgroupParent": "",
  "BlkioWeight": 0,
  "BlkioWeightDevice": [],
  "BlkioDeviceReadBps": [],
  "BlkioDeviceWriteBps": [],
  "BlkioDeviceReadIOps": [],
  "BlkioDeviceWriteIOps": [],
  "CpuPeriod": 0,
  "CpuQuota": 0,
  "CpuRealtimePeriod": 0,
  "CpuRealtimeRuntime": 0,
  "CpusetCpus": "",
  "CpusetMems": "",
  "Devices": [],
  "DeviceCgroupRules": null,
  "DeviceRequests": null,
  "MemoryReservation": 0,
  "MemorySwap": 1073741824,
  "MemorySwappiness": null,
  "OomKillDisable": false,
  "PidsLimit": 256,
  "Ulimits": [
    {
      "Name": "rtprio",
      "Hard": 0,
      "Soft": 0
    }
  ],
  "CpuCount": 0,
  "CpuPercent": 0,
  "IOMaximumIOps": 0,
  "IOMaximumBandwidth": 0,
  "Mounts": [
    {
      "Type": "volume",
      "Source": "kestrel-worker-ipc-f543581994d261c8d3a0a8a5755c5adaa985eaf1bf4974e09cb6a86011683a0a",
      "Target": "/run/kestrel-worker-ipc"
    }
  ],
  "MaskedPaths": [
    "/proc/acpi",
    "/proc/asound",
    "/proc/interrupts",
    "/proc/kcore",
    "/proc/keys",
    "/proc/latency_stats",
    "/proc/sched_debug",
    "/proc/scsi",
    "/proc/timer_list",
    "/proc/timer_stats",
    "/sys/devices/virtual/powercap",
    "/sys/firmware"
  ],
  "ReadonlyPaths": [
    "/proc/bus",
    "/proc/fs",
    "/proc/irq",
    "/proc/sys",
    "/proc/sysrq-trigger"
  ],
  "Init": true
}'''
_MEASURED_BROWSER_MOUNTS_JSON = r'''[
  {
    "Type": "volume",
    "Name": "kestrel-worker-ipc-f543581994d261c8d3a0a8a5755c5adaa985eaf1bf4974e09cb6a86011683a0a",
    "Source": "/var/lib/docker/volumes/kestrel-worker-ipc-f543581994d261c8d3a0a8a5755c5adaa985eaf1bf4974e09cb6a86011683a0a/_data",
    "Destination": "/run/kestrel-worker-ipc",
    "Driver": "local",
    "Mode": "z",
    "RW": true,
    "Propagation": ""
  }
]'''


@pytest.fixture
def measured_created_browser_profile():
    identity = 'f543581994d261c8d3a0a8a5755c5adaa985eaf1bf4974e09cb6a86011683a0a'
    profile = LocalWorkerProfile('native-browser@sha256:6cf61d3ac66c895f6e62834e606ccf0104d9d4ddd03f8c482d2798dcebe156a2', 'native-broker@sha256:f2b399eb6efa00dbf440079923afa5ed81de996d9d5a518c9ed654a5a0177fff')
    host = json.loads(_MEASURED_BROWSER_HOST_JSON)
    host["SecurityOpt"][1] = "seccomp=" + (DATA / "browser-seccomp.json").read_text()
    image = {"Id": profile.image("browser"), "Os": "linux", "Architecture": "arm64",
             "Config": {"Env": ["SCRIPTED_PROFILE_FIXTURE=1"],
                        "Labels": {LABEL + ".contract": CONTRACT}}}
    record = {"HostConfig": host, "Image": image["Id"],
              "Mounts": json.loads(_MEASURED_BROWSER_MOUNTS_JSON), "RestartCount": 0,
              "State": {"Status": "created", "Pid": 0},
              "Config": {"User": "10002:10003", "Env": list(image["Config"]["Env"]),
                         "Labels": {LABEL: identity, LABEL + ".profile": profile.digest()},
                         "Cmd": ["python", "-B", "/opt/kestrel-worker-entry.py", "browser"],
                         "Entrypoint": None, "OpenStdin": False, "Tty": False}}
    return record, image, profile, identity


def test_measured_browser_created_profile_accepts_original_mount_serialization(measured_created_browser_profile):
    record, image, profile, identity = measured_created_browser_profile
    assert set(record["HostConfig"]["Mounts"][0]) == {"Type", "Source", "Target"}
    assert record["Mounts"][0]["Mode"] == "z" and record["Mounts"][0]["RW"] is True
    original_record, original_image = copy.deepcopy(record), copy.deepcopy(image)
    verify_container(record, image, profile, "browser", identity)
    assert record == original_record and image == original_image


@pytest.mark.parametrize("changed", [
    "explicit-readonly-false", "readonly-host", "extra-host-mount-field", "unknown-host-field",
    "missing-host-field", "readonly-record", "empty-mode", "unmeasured-mode",
    "foreign-host-volume", "foreign-record-volume", "network", "root-user", "policy", "image",
])
def test_measured_browser_profile_still_refuses_unmeasured_or_unsafe_changes(measured_created_browser_profile, changed):
    record, image, profile, identity = measured_created_browser_profile
    host, mount = record["HostConfig"], record["Mounts"][0]
    if changed == "explicit-readonly-false":
        host["Mounts"][0]["ReadOnly"] = False
    elif changed == "readonly-host":
        host["Mounts"][0]["ReadOnly"] = True
    elif changed == "extra-host-mount-field":
        host["Mounts"][0]["VolumeOptions"] = {}
    elif changed == "unknown-host-field":
        host["UnmeasuredDaemonField"] = False
    elif changed == "missing-host-field":
        del host["Init"]
    elif changed == "readonly-record":
        mount["RW"] = False
    elif changed == "empty-mode":
        mount["Mode"] = ""
    elif changed == "unmeasured-mode":
        mount["Mode"] = "Z"
    elif changed == "foreign-host-volume":
        host["Mounts"][0]["Source"] += "-foreign"
    elif changed == "foreign-record-volume":
        mount["Name"] += "-foreign"
    elif changed == "network":
        host["NetworkMode"] = "bridge"
    elif changed == "root-user":
        record["Config"]["User"] = "0:0"
    elif changed == "policy":
        policy = json.loads(host["SecurityOpt"][1].partition("=")[2])
        policy["defaultAction"] = "SCMP_ACT_ALLOW"
        host["SecurityOpt"][1] = "seccomp=" + json.dumps(policy)
    else:
        assert changed == "image"
        record["Image"] = "sha256:" + "0" * 64
    with pytest.raises(ResourceUnsupported, match="fixed_two_domain_container_profile_differs"):
        verify_container(record, image, profile, "browser", identity)


# Native05's complete safe broker-created projection, independent of the
# expected host_configuration(). Packaged broker seccomp is inserted below;
# image Env/contract/profile labels are SCRIPTED unrelated-equality data.
# This vector does not establish native bootstrap, IO, session or authority.
_MEASURED_BROKER_HOST_JSON = r'''{
  "Binds": null,
  "ContainerIDFile": "",
  "LogConfig": {
    "Type": "none",
    "Config": {}
  },
  "NetworkMode": "kestrel-worker-net-a4e85b08fc88d5e3b7ca17be1f69ab09bcde908f6dea565d419695b96b59e844",
  "PortBindings": {
    "41401/tcp": [
      {
        "HostIp": "127.0.0.1",
        "HostPort": ""
      }
    ],
    "41402/tcp": [
      {
        "HostIp": "127.0.0.1",
        "HostPort": ""
      }
    ]
  },
  "RestartPolicy": {
    "Name": "no",
    "MaximumRetryCount": 0
  },
  "AutoRemove": false,
  "VolumeDriver": "",
  "VolumesFrom": null,
  "ConsoleSize": [
    0,
    0
  ],
  "CapAdd": null,
  "CapDrop": [
    "ALL"
  ],
  "CgroupnsMode": "private",
  "Dns": null,
  "DnsOptions": [],
  "DnsSearch": [],
  "ExtraHosts": null,
  "GroupAdd": null,
  "IpcMode": "private",
  "Cgroup": "",
  "Links": null,
  "OomScoreAdj": 0,
  "PidMode": "",
  "Privileged": false,
  "PublishAllPorts": false,
  "ReadonlyRootfs": true,
  "SecurityOpt": [
    "no-new-privileges",
    {
      "type": "str",
      "serialized_bytes": 1770,
      "sha256": "14296af642674696b6f8fee74dd0fb2f2fc62213b637023d4c39281ec0a0c564"
    }
  ],
  "Tmpfs": {
    "/tmp": "rw,nosuid,nodev,size=16m,mode=700,uid=10001,gid=10003"
  },
  "UTSMode": "",
  "UsernsMode": "",
  "ShmSize": 16777216,
  "Runtime": "runc",
  "Isolation": "",
  "CpuShares": 0,
  "Memory": 536870912,
  "NanoCpus": 1000000000,
  "CgroupParent": "",
  "BlkioWeight": 0,
  "BlkioWeightDevice": [],
  "BlkioDeviceReadBps": [],
  "BlkioDeviceWriteBps": [],
  "BlkioDeviceReadIOps": [],
  "BlkioDeviceWriteIOps": [],
  "CpuPeriod": 0,
  "CpuQuota": 0,
  "CpuRealtimePeriod": 0,
  "CpuRealtimeRuntime": 0,
  "CpusetCpus": "",
  "CpusetMems": "",
  "Devices": [],
  "DeviceCgroupRules": null,
  "DeviceRequests": null,
  "MemoryReservation": 0,
  "MemorySwap": 536870912,
  "MemorySwappiness": null,
  "OomKillDisable": false,
  "PidsLimit": 64,
  "Ulimits": [
    {
      "Name": "rtprio",
      "Hard": 0,
      "Soft": 0
    }
  ],
  "CpuCount": 0,
  "CpuPercent": 0,
  "IOMaximumIOps": 0,
  "IOMaximumBandwidth": 0,
  "Mounts": [
    {
      "Type": "volume",
      "Source": "kestrel-worker-ipc-a4e85b08fc88d5e3b7ca17be1f69ab09bcde908f6dea565d419695b96b59e844",
      "Target": "/run/kestrel-worker-ipc",
      "ReadOnly": true
    }
  ],
  "MaskedPaths": [
    "/proc/acpi",
    "/proc/asound",
    "/proc/interrupts",
    "/proc/kcore",
    "/proc/keys",
    "/proc/latency_stats",
    "/proc/sched_debug",
    "/proc/scsi",
    "/proc/timer_list",
    "/proc/timer_stats",
    "/sys/devices/virtual/powercap",
    "/sys/firmware"
  ],
  "ReadonlyPaths": [
    "/proc/bus",
    "/proc/fs",
    "/proc/irq",
    "/proc/sys",
    "/proc/sysrq-trigger"
  ],
  "Init": true
}'''
_MEASURED_BROKER_MOUNTS_JSON = r'''[
  {
    "Type": "volume",
    "Name": "kestrel-worker-ipc-a4e85b08fc88d5e3b7ca17be1f69ab09bcde908f6dea565d419695b96b59e844",
    "Source": "/var/lib/docker/volumes/kestrel-worker-ipc-a4e85b08fc88d5e3b7ca17be1f69ab09bcde908f6dea565d419695b96b59e844/_data",
    "Destination": "/run/kestrel-worker-ipc",
    "Driver": "local",
    "Mode": "z",
    "RW": false,
    "Propagation": ""
  }
]'''


@pytest.fixture
def measured_created_broker_profile():
    identity = 'a4e85b08fc88d5e3b7ca17be1f69ab09bcde908f6dea565d419695b96b59e844'
    profile = LocalWorkerProfile('native-browser@sha256:8ed11c9eca0c40c6d97f768c5d2670abc06ab0c762e35c8e3508e057248d6e50', 'native-broker@sha256:b8a101b46bf596c7afd4395c2e0070f2f98fb606c48c1c5de2cde8dd3da6067b')
    host = json.loads(_MEASURED_BROKER_HOST_JSON)
    host["SecurityOpt"][1] = "seccomp=" + (DATA / "broker-seccomp.json").read_text()
    image = {"Id": profile.image("broker"), "Os": "linux", "Architecture": "arm64",
             "Config": {"Env": ["SCRIPTED_BROKER_PROFILE_FIXTURE=1"],
                        "Labels": {LABEL + ".contract": CONTRACT}}}
    record = {"HostConfig": host, "Image": image["Id"],
              "Mounts": json.loads(_MEASURED_BROKER_MOUNTS_JSON), "RestartCount": 0,
              "State": {"Status": "created", "Pid": 0},
              "Config": {"User": "10001:10003", "Env": list(image["Config"]["Env"]),
                         "Labels": {LABEL: identity, LABEL + ".profile": profile.digest()},
                         "Cmd": ["python", "-B", "/opt/kestrel-worker-entry.py", "broker"],
                         "Entrypoint": None, "OpenStdin": False, "Tty": False}}
    return record, image, profile, identity


def test_measured_broker_created_profile_accepts_original_mount_serialization(measured_created_broker_profile):
    record, image, profile, identity = measured_created_broker_profile
    assert record["HostConfig"]["Mounts"][0]["ReadOnly"] is True
    assert record["Mounts"][0]["Mode"] == "z" and record["Mounts"][0]["RW"] is False
    original_record, original_image = copy.deepcopy(record), copy.deepcopy(image)
    verify_container(record, image, profile, "broker", identity)
    assert record == original_record and image == original_image


@pytest.mark.parametrize("changed", [
    "empty-mode", "changed-mode", "writable-record", "missing-readonly", "false-readonly",
    "foreign-host-volume", "foreign-record-volume", "foreign-host-target", "unknown-host-field",
    "missing-host-field", "extra-host-mount-field", "wrong-role-user", "unsafe-network", "policy", "image",
])
def test_measured_broker_profile_still_refuses_unmeasured_or_unsafe_changes(measured_created_broker_profile, changed):
    record, image, profile, identity = measured_created_broker_profile
    host, mount = record["HostConfig"], record["Mounts"][0]
    if changed == "empty-mode":
        mount["Mode"] = ""
    elif changed == "changed-mode":
        mount["Mode"] = "ro,z"
    elif changed == "writable-record":
        mount["RW"] = True
    elif changed == "missing-readonly":
        del host["Mounts"][0]["ReadOnly"]
    elif changed == "false-readonly":
        host["Mounts"][0]["ReadOnly"] = False
    elif changed == "foreign-host-volume":
        host["Mounts"][0]["Source"] += "-foreign"
    elif changed == "foreign-record-volume":
        mount["Name"] += "-foreign"
    elif changed == "foreign-host-target":
        host["Mounts"][0]["Target"] += "-foreign"
    elif changed == "unknown-host-field":
        host["UnmeasuredDaemonField"] = False
    elif changed == "missing-host-field":
        del host["Init"]
    elif changed == "extra-host-mount-field":
        host["Mounts"][0]["VolumeOptions"] = {}
    elif changed == "wrong-role-user":
        record["Config"]["User"] = "0:0"
    elif changed == "unsafe-network":
        host["NetworkMode"] = "host"
    elif changed == "policy":
        policy = json.loads(host["SecurityOpt"][1].partition("=")[2])
        policy["defaultAction"] = ("SCMP_ACT_ALLOW" if policy["defaultAction"] != "SCMP_ACT_ALLOW"
                                   else "SCMP_ACT_KILL_PROCESS")
        host["SecurityOpt"][1] = "seccomp=" + json.dumps(policy)
    else:
        assert changed == "image"
        record["Image"] = "sha256:" + "0" * 64
    with pytest.raises(ResourceUnsupported, match="fixed_two_domain_container_profile_differs"):
        verify_container(record, image, profile, "broker", identity)


def test_broker_create_arguments_use_packaged_listener_policy():
    """Pure CLI/policy wiring; Linux syscall admission is a separate proof."""
    profile = LocalWorkerProfile("browser@sha256:" + "a" * 64, "broker@sha256:" + "b" * 64)
    arguments = container_arguments(profile, "broker", "c" * 64)
    policies = [arguments[index + 1].partition("=")[2]
                for index, value in enumerate(arguments[:-1])
                if value == "--security-opt" and arguments[index + 1].startswith("seccomp=")]
    assert policies == [str(DATA / "broker-seccomp.json")]
    policy = json.loads(Path(policies[0]).read_bytes())
    allowed = {name for rule in policy["syscalls"]
               if rule["action"] == "SCMP_ACT_ALLOW" and not rule.get("args")
               for name in rule["names"]}
    # The canonical broker directly creates/binds/listens/accepts sockets.
    # This finite contract is not a complete Python/OpenSSL/Linux ABI claim.
    assert {"socket", "bind", "listen", "accept", "accept4"} <= allowed
    assert policy["defaultAction"] == "SCMP_ACT_ERRNO"
    assert policy["defaultErrnoRet"] == 1


def test_broker_profile_refuses_daemon_policy_without_listen(measured_created_broker_profile):
    """A scripted daemon map cannot substitute the old listener-denying policy."""
    record, image, profile, identity = measured_created_broker_profile
    options = record["HostConfig"]["SecurityOpt"]
    indices = [index for index, value in enumerate(options) if value.startswith("seccomp=")]
    assert len(indices) == 1
    index = indices[0]
    policy = json.loads(options[index].partition("=")[2])
    rules = [rule for rule in policy["syscalls"] if "listen" in rule["names"]]
    assert len(rules) == 1 and rules[0]["action"] == "SCMP_ACT_ALLOW" and not rules[0].get("args")
    rules[0]["names"].remove("listen")
    options[index] = "seccomp=" + json.dumps(policy)
    original_record = copy.deepcopy(record)
    with pytest.raises(ResourceUnsupported, match="fixed_two_domain_container_profile_differs"):
        verify_container(record, image, profile, "broker", identity)
    assert record == original_record


# SCRIPTED identity/start values and measured normal-bridge structural facts.
# These tests perform no native IO and confer no Host/session authority.
@pytest.fixture
def original_broker_endpoints():
    cid, native = "b" * 64, (37958, "SCRIPTED-original-start")
    record = {"Id": cid, "State": {"Running": True, "Pid": native[0], "StartedAt": native[1]},
              "NetworkSettings": {"Ports": {
                  "41401/tcp": [{"HostIp": "127.0.0.1", "HostPort": "64476"}],
                  "41402/tcp": [{"HostIp": "127.0.0.1", "HostPort": "64477"}],
              }}}
    return record, cid, native


def test_original_loopback_map_preserves_role_order_without_mutation(original_broker_endpoints):
    from kestrel_sovereign.visual_computer.local_worker import _original_loopback_endpoints
    record, cid, native = original_broker_endpoints
    before = copy.deepcopy(record)
    assert _original_loopback_endpoints(record, cid, native) == (64476, 64477)
    assert record == before


@pytest.mark.parametrize("changed", [
    "missing-map", "missing-role", "extra-role", "empty", "multiple", "binding-extra", "binding-missing",
    "wildcard", "ipv6", "hostname", "blank", "unicode", "signed", "leading-zero", "zero", "overflow",
    "integer", "boolean", "duplicate", "foreign-cid", "not-running", "boolean-pid", "foreign-pid",
    "foreign-start", "missing-start",
])
def test_original_loopback_map_refuses_non_original_or_noncanonical_endpoints(original_broker_endpoints, changed):
    from kestrel_sovereign.visual_computer.local_worker import _original_loopback_endpoints
    record, cid, native = original_broker_endpoints
    ports = record["NetworkSettings"]["Ports"]
    binding = ports["41401/tcp"][0]
    if changed == "missing-map":
        del record["NetworkSettings"]["Ports"]
    elif changed == "missing-role":
        del ports["41402/tcp"]
    elif changed == "extra-role":
        ports["443/tcp"] = []
    elif changed == "empty":
        ports["41401/tcp"] = []
    elif changed == "multiple":
        ports["41401/tcp"].append(dict(binding))
    elif changed == "binding-extra":
        binding["Unmeasured"] = True
    elif changed == "binding-missing":
        del binding["HostIp"]
    elif changed in {"wildcard", "ipv6", "hostname"}:
        binding["HostIp"] = {"wildcard": "0.0.0.0", "ipv6": "::1", "hostname": "localhost"}[changed]
    elif changed in {"blank", "unicode", "signed", "leading-zero", "zero", "overflow", "integer", "boolean"}:
        binding["HostPort"] = {"blank": "", "unicode": "６４４７６", "signed": "+64476", "leading-zero": "06447",
                               "zero": "0", "overflow": "65536", "integer": 64476, "boolean": True}[changed]
    elif changed == "duplicate":
        ports["41402/tcp"][0]["HostPort"] = binding["HostPort"]
    elif changed == "foreign-cid":
        record["Id"] = "c" * 64
    elif changed == "not-running":
        record["State"]["Running"] = False
    elif changed in {"boolean-pid", "foreign-pid"}:
        record["State"]["Pid"] = True if changed == "boolean-pid" else native[0] + 1
    elif changed == "foreign-start":
        record["State"]["StartedAt"] += "-replacement"
    else:
        assert changed == "missing-start"
        del record["State"]["StartedAt"]
    before = copy.deepcopy(record)
    with pytest.raises(ResourceUnsupported, match="original_loopback_role_endpoint_required"):
        _original_loopback_endpoints(record, cid, native)
    assert record == before


@pytest.fixture
def measured_normal_network():
    # Structural vector from the saved noninternal network inspect. Identity,
    # name/label and member are SCRIPTED substitutions, not native ownership.
    identity, native_id, cid = "c" * 64, "d" * 64, "b" * 64
    network = {"Id": native_id, "Name": "kestrel-worker-net-" + identity,
               "Labels": {LABEL: identity}, "Driver": "bridge", "Scope": "local",
               "EnableIPv4": True, "EnableIPv6": False, "Internal": False,
               "Attachable": False, "Ingress": False, "ConfigOnly": False,
               "ConfigFrom": {"Network": ""}, "Options": {"com.docker.network.enable_ipv4": "true"},
               "Containers": {cid: {"SCRIPTED_MEMBER": True}}}
    return network, native_id, identity, cid


def test_v2_network_matches_measured_noninternal_shape(measured_normal_network):
    from kestrel_sovereign.visual_computer.local_worker import _verify_broker_network
    network, native_id, identity, cid = measured_normal_network
    before = copy.deepcopy(network)
    _verify_broker_network(network, native_id, identity, {cid})
    assert network == before
    network["Containers"] = {}
    _verify_broker_network(network, native_id, identity, set())


@pytest.mark.parametrize("changed", ["id", "name", "label", "extra-label", "driver", "scope", "ipv4", "ipv6",
    "internal", "attachable", "ingress", "config-only", "config-from", "options", "missing-member", "foreign-member"])
def test_v2_network_refuses_foreign_or_unmeasured_topology(measured_normal_network, changed):
    from kestrel_sovereign.visual_computer.local_worker import _verify_broker_network
    network, native_id, identity, cid = measured_normal_network
    if changed in {"id", "name", "driver", "scope"}:
        field = {"id": "Id", "name": "Name", "driver": "Driver", "scope": "Scope"}[changed]
        network[field] += "-foreign"
    elif changed == "label":
        network["Labels"][LABEL] = "e" * 64
    elif changed == "extra-label":
        network["Labels"]["foreign"] = "1"
    elif changed in {"ipv4", "ipv6", "internal", "attachable", "ingress", "config-only"}:
        field = {"ipv4": "EnableIPv4", "ipv6": "EnableIPv6", "internal": "Internal", "attachable": "Attachable",
                 "ingress": "Ingress", "config-only": "ConfigOnly"}[changed]
        network[field] = not network[field]
    elif changed == "config-from":
        network["ConfigFrom"] = {"Network": "foreign"}
    elif changed == "options":
        network["Options"]["com.docker.network.bridge.enable_icc"] = "true"
    elif changed == "missing-member":
        network["Containers"] = {}
    else:
        assert changed == "foreign-member"
        network["Containers"]["e" * 64] = {}
    with pytest.raises(ResourceUnsupported, match="original_broker_network_changed"):
        _verify_broker_network(network, native_id, identity, {cid})


def test_v2_profile_refuses_v1_image_contract(measured_created_broker_profile):
    record, image, profile, identity = measured_created_broker_profile
    assert CONTRACT == "core-sdk046-two-domain-v2"
    image["Config"]["Labels"][LABEL + ".contract"] = "core-sdk046-two-domain-v1"
    with pytest.raises(ResourceUnsupported, match="fixed_two_domain_container_profile_differs"):
        verify_container(record, image, profile, "broker", identity)
