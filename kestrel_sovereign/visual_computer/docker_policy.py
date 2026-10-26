"""Canonical reviewed offline Docker browser profile, shared with real verification."""
import hashlib
import json
from pathlib import Path

LABEL = "io.kestrel.visual-browser-probe"
POLICY_ROOT = Path(__file__).with_name("docker_profile")
SECCOMP_SHA256 = "38c15179be39d9bc1aadf19b06618bdb0dd9137f531435ff306c8dccf3b59f19"
HOST_CONFIG_SHA256 = "5b9799bdf8e9f9292554007d64778a9941dedd78dee2797dac362e881c06f394"


def verified_seccomp_policy() -> dict:
    content = (POLICY_ROOT / "seccomp.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != SECCOMP_SHA256:
        raise RuntimeError("browser seccomp policy digest differs from the reviewed profile")
    return json.loads(content)



def verified_host_configuration() -> dict:
    content = (POLICY_ROOT / "host-config.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != HOST_CONFIG_SHA256:
        raise RuntimeError("browser host configuration digest differs from the reviewed profile")
    return json.loads(content)

def container_arguments(image: str, identity: str, *, worker: bool = False,
                        label: str = LABEL, name_prefix: str = "kestrel-browser-probe-") -> list[str]:
    return [
        "create", "--name", f"{name_prefix}{identity}", "--label", f"{label}={identity}",
        "--init", "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--security-opt", f"seccomp={POLICY_ROOT / 'seccomp.json'}",
        "--memory", "1g", "--memory-swap", "1g", "--ipc", "private",
        "--cgroupns", "private", "--runtime", "runc", "--cpus", "2", "--pids-limit", "256", "--shm-size", "256m",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
        "--ulimit", "rtprio=0:0", "--log-driver", "none",
        image, "python", "/opt/kestrel-browser/worker.py" if worker else "/opt/kestrel-browser/sandbox_probe.py",
    ]


def verify_container(record: dict, image: str, identity: str, *, label: str = LABEL) -> None:
    host, config = dict(record["HostConfig"]), record["Config"]
    # Docker 29.8.2 on the measured cgroup-v2 host serializes this default
    # false field as null after start. No OOM-disable request was admitted.
    # Restrict normalization to post-create states; all other fields are exact.
    if record.get("State", {}).get("Status") in {"running", "exited"} and host.get("OomKillDisable") is None:
        host["OomKillDisable"] = False
    options = host.get("SecurityOpt") or []
    policies = [option.partition("=")[2] for option in options if option.startswith("seccomp=")]
    expected_policy = verified_seccomp_policy()
    try:
        policy_matches = len(policies) == 1 and json.loads(policies[0]) == expected_policy
    except (TypeError, ValueError):
        policy_matches = False
    if policy_matches:
        host["SecurityOpt"] = ["seccomp=reviewed-policy" if option.startswith("seccomp=") else option
                               for option in options]
    # Compare the complete reviewed map, including default-valued fields. New
    # daemon settings are unsupported until reviewed; they never pass silently.
    checks = (
        host == verified_host_configuration(),
        record["Image"] == image,
        config["Labels"].get(label) == identity,
        config["User"] == "pwuser",
        record.get("Mounts") == [],
        policy_matches,
    )
    if not all(checks):
        raise RuntimeError("Docker did not apply the required isolated browser configuration")

