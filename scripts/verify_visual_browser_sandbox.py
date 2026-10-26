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
from kestrel_sovereign.visual_computer.docker_policy import (
    POLICY_ROOT, verified_seccomp_policy, verified_host_configuration,
    container_arguments, verify_container,
)

def verify_result(result: dict) -> bytes:
    if (result.get("uid") in (None, 0) or result.get("capabilities") != {name: 0 for name in ("CapEff", "CapBnd", "CapPrm", "CapInh", "CapAmb")}
            or result.get("rtprio_limit") != [0, 0]
            or not isinstance(result.get("groups"), list) or 0 in result["groups"]
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


def source_identity() -> dict:
    """Bind evidence to actual context bytes, including dirty/fault-injected builds."""
    files = {path.relative_to(CONTEXT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(CONTEXT.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    files.update({"reviewed-profile/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(POLICY_ROOT.glob("*.json"))})
    adapter_root = ROOT / "kestrel_sovereign" / "visual_computer"
    files.update({"host-adapter/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(adapter_root.glob("docker*.py"))})
    for name in ("verify_visual_browser_sandbox.py", "verify_visual_browser_worker.py", "verify_visual_docker_adapters.py"):
        path = ROOT / "scripts" / name
        if path.is_file():
            files["verification/" + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    context_digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, check=False)
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, check=False)
    head = revision.stdout.decode(errors="replace").strip()
    return {"revision": head if revision.returncode == 0 and re.fullmatch(r"[a-f0-9]{40}", head) else None,
            "dirty": bool(status.stdout) if status.returncode == 0 else None,
            "build_context_sha256": context_digest, "build_context_files": files}


def run(output: Path, *, docker: str, worker_check=None) -> dict:
    verified_seccomp_policy()
    verified_host_configuration()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    identity = uuid.uuid4().hex
    tag = f"kestrel-visual-browser-probe:{identity}"
    state = {"probe_id": identity, "tag": tag, "container_name": f"kestrel-browser-probe-{identity}",
             "container_id": None, "image_id": None,
             "cleanup_confirmed": False, "scope": "offline browser worker" if worker_check else "offline browser substrate",
             "host_or_kite_acceptance": False, "source": source_identity()}
    manifest = output / "custody.json"

    def record():
        manifest.write_text(json.dumps(state, indent=2) + "\n")

    def invoke(arguments, *, timeout=30, input=None):
        try:
            return subprocess.run([docker, *arguments], input=input, capture_output=True, timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            state["outcome_unknown"] = True; record()
            raise

    def checked(arguments, *, timeout=30, input=None):
        result = invoke(arguments, timeout=timeout, input=input)
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
        cid = checked(container_arguments(image["Id"], identity, worker=worker_check is not None)).decode().strip()
        if not re.fullmatch(r"[a-f0-9]{64}", cid):
            raise RuntimeError("Docker did not return an exact container identity")
        state["container_id"] = cid; record()
        config = json.loads(checked(["container", "inspect", cid]))[0]
        verify_container(config, image["Id"], identity)
        if worker_check is not None:
            checked(["start", cid])
            state["result"] = worker_check(checked, cid, output)
            record()
            return state
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
    parser.add_argument("--worker", action="store_true", help="Exercise persistent authenticated offline worker IPC")
    args = parser.parse_args()
    docker = shutil.which("docker")
    if not docker:
        parser.error("Docker CLI is required")
    worker_check = None
    if args.worker:
        from verify_visual_browser_worker import exercise
        worker_check = exercise
    result = run(args.output.resolve(), docker=docker, worker_check=worker_check)
    print(json.dumps({"passed": True, "cleanup_confirmed": result["cleanup_confirmed"], "evidence": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
