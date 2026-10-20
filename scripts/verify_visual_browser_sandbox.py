#!/usr/bin/env python3
"""Build and verify the local browser substrate using only owned Docker resources.

This opt-in developer check is separate from host/session/inference acceptance.
It exercises a synthetic offline page; it never invokes a production agent.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path

LABEL = "io.kestrel.visual-browser-probe"
ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "docker" / "visual-browser"


def container_arguments(image: str, identity: str) -> list[str]:
    return [
        "create", "--name", f"kestrel-browser-probe-{identity}", "--label", f"{LABEL}={identity}",
        "--init", "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--security-opt", f"seccomp={CONTEXT / 'seccomp.json'}",
        "--memory", "1g", "--cpus", "2", "--pids-limit", "256", "--shm-size", "256m",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
        image, "python", "/opt/kestrel-browser/sandbox_probe.py",
    ]


def verify_container(record: dict, image: str, identity: str) -> None:
    host, config = record["HostConfig"], record["Config"]
    policies = [option.partition("=")[2] for option in host["SecurityOpt"] if option.startswith("seccomp=")]
    expected_policy = json.loads((CONTEXT / "seccomp.json").read_text())
    policy_matches = len(policies) == 1 and json.loads(policies[0]) == expected_policy
    checks = (
        record["Image"] == image,
        config["Labels"].get(LABEL) == identity,
        config["User"] == "pwuser",
        host["NetworkMode"] == "none",
        host["ReadonlyRootfs"] is True,
        host["Privileged"] is False,
        host["CapDrop"] == ["ALL"],
        not host.get("CapAdd"),
        not host.get("Binds"), not host.get("Mounts"),
        not host.get("Devices"), not host.get("DeviceRequests"),
        not host.get("PortBindings"), host.get("PublishAllPorts") is False,
        all(mount["Type"] == "tmpfs" and mount["Destination"] == "/tmp" for mount in record.get("Mounts", [])),
        host.get("PidMode", "") == "",
        host.get("IpcMode") != "host",
        host["Memory"] == 1024**3,
        host["NanoCpus"] == 2_000_000_000,
        host["PidsLimit"] == 256,
        host["ShmSize"] == 256 * 1024**2,
        any(option in {"no-new-privileges", "no-new-privileges=true"} for option in host["SecurityOpt"]),
        policy_matches,
    )
    if not all(checks):
        raise RuntimeError("Docker did not apply the required isolated browser configuration")


def verify_result(result: dict) -> bytes:
    if (result.get("uid") in (None, 0) or result.get("capabilities") != 0
            or result.get("no_new_privileges") is not True
            or result.get("fixture_result") != "Persistent local browser draft"
            or result.get("refused_destinations") != ["1.1.1.1", "169.254.169.254"]):
        raise RuntimeError("browser probe evidence is incomplete")
    for expected in ("PID namespaces\tYes", "Network namespaces\tYes", "Seccomp-BPF sandbox\tYes"):
        if expected not in result.get("sandbox", ""):
            raise RuntimeError("Chromium sandbox evidence is incomplete")
    png = base64.b64decode(result["png_base64"], validate=True)
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or hashlib.sha256(png).hexdigest() != result["png_sha256"]:
        raise RuntimeError("browser screenshot failed verification")
    return png


def run(output: Path, *, docker: str) -> dict:
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    identity = uuid.uuid4().hex
    tag = f"kestrel-visual-browser-probe:{identity}"
    state = {"probe_id": identity, "tag": tag, "container_name": f"kestrel-browser-probe-{identity}",
             "container_id": None, "image_id": None,
             "cleanup_confirmed": False, "scope": "offline browser substrate; not host/session/Kite acceptance"}
    manifest = output / "custody.json"

    def record():
        manifest.write_text(json.dumps(state, indent=2) + "\n")

    def invoke(arguments, *, timeout=30):
        try:
            return subprocess.run([docker, *arguments], capture_output=True, timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            state["outcome_unknown"] = True; record()
            raise

    def checked(arguments, *, timeout=30):
        result = invoke(arguments, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"Docker {' '.join(arguments[:2])} failed: {result.stderr.decode(errors='replace')[-3000:]}")
        return result.stdout

    record()
    try:
        build = invoke(["build", "--label", f"{LABEL}={identity}", "--tag", tag, str(CONTEXT)], timeout=600)
        (output / "build.log").write_bytes(build.stdout + build.stderr)
        if build.returncode:
            raise RuntimeError("browser image build failed; see build.log")
        image = json.loads(checked(["image", "inspect", tag]))[0]
        if image["Config"]["Labels"].get(LABEL) != identity:
            raise RuntimeError("browser image ownership does not match")
        state["image_id"] = image["Id"]; record()
        cid = checked(container_arguments(image["Id"], identity)).decode().strip()
        if not re.fullmatch(r"[a-f0-9]{64}", cid):
            raise RuntimeError("Docker did not return an exact container identity")
        state["container_id"] = cid; record()
        config = json.loads(checked(["container", "inspect", cid]))[0]
        verify_container(config, image["Id"], identity)
        completed = invoke(["start", "--attach", cid], timeout=120)
        (output / "browser.stderr").write_bytes(completed.stderr)
        if completed.returncode:
            raise RuntimeError("browser sandbox check failed; see browser.stderr")
        result = json.loads(completed.stdout)
        (output / "browser.png").write_bytes(verify_result(result))
        result.pop("png_base64")
        state["result"] = result; record()
    finally:
        # Check labels before touching a resource; never prune unrelated state.
        def inspect_owned(kind, target):
            result = invoke([kind, "inspect", target])
            if result.returncode == 0:
                return json.loads(result.stdout)[0]
            absent = re.compile(rb"(?:Error: |Error response from daemon: )No such (?:object|image|container): "
                                + re.escape(target.encode()) + rb"\s*")
            if absent.fullmatch(result.stderr):
                return None
            raise RuntimeError("Docker cleanup could not verify resource state; retain custody.json")

        owned = inspect_owned("container", state["container_name"])
        if owned is not None:
            if owned["Config"]["Labels"].get(LABEL) != identity:
                raise RuntimeError("refusing cleanup of an unowned container")
            state["container_id"] = owned["Id"]; record()
            checked(["container", "rm", "--force", owned["Id"]])
        owned = inspect_owned("image", tag)
        if owned is not None:
            if (state["image_id"] is not None and owned["Id"] != state["image_id"]
                    or owned["Config"]["Labels"].get(LABEL) != identity):
                raise RuntimeError("refusing cleanup of an unowned image")
            state["image_id"] = owned["Id"]; record()
            checked(["image", "rm", tag])
        state["cleanup_confirmed"] = not state.get("outcome_unknown", False); record()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; must not exist")
    args = parser.parse_args()
    docker = shutil.which("docker")
    if not docker:
        parser.error("Docker CLI is required")
    result = run(args.output.resolve(), docker=docker)
    print(json.dumps({"passed": True, "cleanup_confirmed": result["cleanup_confirmed"], "evidence": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
