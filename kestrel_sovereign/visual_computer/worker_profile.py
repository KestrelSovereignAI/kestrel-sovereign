"""Closed local two-domain profile. Configuration facts confer no authority."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from kestrel_sdk.resources import ResourceUnsupported

PROFILE = "local-two-domain-worker-v2"
PROVIDER = "local-two-domain-worker"
DRIVER = "remote-pinned-browser-v1"
LABEL = "io.kestrel.local-two-domain"
CONTRACT = "core-sdk046-two-domain-v2"
IPC = "/run/kestrel-worker-ipc"
PRIVATE = "/tmp/kestrel-browser"
LAUNCHER = "/opt/kestrel-worker-entry.py"
MAX_REQUEST, MAX_RESPONSE = 512 * 1024, 23 * 1024 * 1024
# The pinned Playwright driver allows30s for its original browser close. The
# worker watchdog also covers context/pipe joins; the Host waits beyond it.
BROWSER_CLOSE_SECONDS = 30
BROWSER_RETIRE_SECONDS = BROWSER_CLOSE_SECONDS + 10
WORKER_RETIRE_WAIT_SECONDS = BROWSER_RETIRE_SECONDS + 10
OPERATIONS = frozenset({"screenshot", "click", "type", "key", "scroll", "wait", "takeover"})
DATA = Path(__file__).with_name("worker_profile_data")
TOKEN = re.compile(r"[a-f0-9]{64}\Z")
IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:([a-f0-9]{64})\Z")


@dataclass(frozen=True)
class LocalWorkerProfile:
    browser_image: str
    broker_image: str
    architecture: str = "arm64"

    def __post_init__(self):
        if (type(self) is not LocalWorkerProfile
                or any(type(v) is not str or IMAGE.fullmatch(v) is None
                       for v in (self.browser_image, self.broker_image))
                or self.architecture != "arm64"):
            raise ResourceUnsupported("invalid_fixed_local_worker_profile")

    def image(self, role):
        if role not in {"browser", "broker"}:
            raise ResourceUnsupported("invalid_fixed_worker_role")
        return "sha256:" + IMAGE.fullmatch(getattr(self, role + "_image")).group(1)

    def digest(self):
        value = {"profile": PROFILE, "contract": CONTRACT,
                 "browser": self.browser_image, "broker": self.broker_image,
                 "architecture": self.architecture,
                 "policies": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(DATA.glob("*.json"))}}
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _names(identity):
    if type(identity) is not str or TOKEN.fullmatch(identity) is None:
        raise ResourceUnsupported("invalid_original_worker_identity")
    return {"network": "kestrel-worker-net-" + identity,
            "volume": "kestrel-worker-ipc-" + identity,
            "browser": "kestrel-worker-browser-" + identity,
            "broker": "kestrel-worker-broker-" + identity}


def host_configuration(role, identity):
    names = _names(identity)
    if role not in {"browser", "broker"}:
        raise ResourceUnsupported("invalid_fixed_worker_role")
    raw = json.loads((DATA / (role + "-host.json")).read_bytes())
    def bind(value):
        if isinstance(value, dict):
            return {k: bind(v) for k, v in value.items()}
        if isinstance(value, list):
            return [bind(v) for v in value]
        return names.get(value[1:], value) if type(value) is str and value.startswith("$") else value
    return bind(raw)


def container_arguments(profile, role, identity):
    if type(profile) is not LocalWorkerProfile or role not in {"browser", "broker"}:
        raise ResourceUnsupported("exact_local_worker_profile_required")
    names = _names(identity)
    uid, memory, cpus, pids, shm = (("10002:10003", "1g", "2", "256", "256m")
                                  if role == "browser" else ("10001:10003", "512m", "1", "64", "16m"))
    args = ["create", "--pull=never", "--name", names[role], "--label", LABEL + "=" + identity,
            "--label", LABEL + ".profile=" + profile.digest(), "--init",
            "--user", uid, "--network", "none" if role == "browser" else names["network"],
            "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--security-opt", "seccomp=" + str(DATA / (role + "-seccomp.json")),
            "--memory", memory, "--memory-swap", memory, "--cpus", cpus,
            "--pids-limit", pids, "--shm-size", shm, "--ipc", "private",
            "--cgroupns", "private", "--runtime", "runc", "--ulimit", "rtprio=0:0",
            "--log-driver", "none", "--mount",
            "type=volume,src=" + names["volume"] + ",dst=" + IPC + (",readonly" if role == "broker" else ""),
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=" + ("256m,mode=1777" if role == "browser" else "16m,mode=700,uid=10001,gid=10003")]
    if role == "broker":
        for port in (41401, 41402):
            args += ["--publish", "127.0.0.1::" + str(port)]
    return args + [profile.image(role), "python", "-B", LAUNCHER, role]


def verify_container(record, image, profile, role, identity):
    """Full daemon map equality; unknown fields/defaults are unsupported."""
    expected = host_configuration(role, identity)
    host, config = dict(record["HostConfig"]), record["Config"]
    options = host.get("SecurityOpt") or []
    policies = [x.partition("=")[2] for x in options if x.startswith("seccomp=")]
    try:
        valid_policy = len(policies) == 1 and json.loads(policies[0]) == json.loads((DATA / (role + "-seccomp.json")).read_bytes())
    except (ValueError, TypeError):
        valid_policy = False
    if valid_policy:
        host["SecurityOpt"] = ["seccomp=fixed-worker-policy" if x.startswith("seccomp=") else x for x in options]
    # Same narrowly measured Docker post-start serialization as the old profile.
    if record.get("State", {}).get("Status") in {"running", "exited"} and host.get("OomKillDisable") is None:
        host["OomKillDisable"] = False
    names = _names(identity)
    expected_mount = {"Type": "volume", "Name": names["volume"], "Destination": IPC,
                      "Driver": "local", "Mode": "z",
                      "RW": role == "browser", "Propagation": ""}
    mounts = record.get("Mounts", [])
    mount_ok = len(mounts) == 1 and all(mounts[0].get(k) == v for k, v in expected_mount.items())
    labels = config.get("Labels") or {}
    if (not valid_policy or host != expected or not mount_ok
            or record["Image"] != profile.image(role) or image["Id"] != profile.image(role)
            or image.get("Os") != "linux" or image.get("Architecture") != profile.architecture
            or image.get("Config", {}).get("Labels", {}).get(LABEL + ".contract") != CONTRACT
            or labels.get(LABEL) != identity or labels.get(LABEL + ".profile") != profile.digest()
            or config.get("User") != ("10002:10003" if role == "browser" else "10001:10003")
            or config.get("Env") != image.get("Config", {}).get("Env")
            or config.get("Cmd") != ["python", "-B", LAUNCHER, role]
            or config.get("Entrypoint") is not None or config.get("OpenStdin") is not False
            or config.get("Tty") is not False or record.get("RestartCount") != 0):
        raise ResourceUnsupported("fixed_two_domain_container_profile_differs")
