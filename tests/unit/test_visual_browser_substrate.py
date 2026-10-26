"""Substrate admission/cleanup checks; the real browser probe is opt-in."""
import base64
import hashlib
import json
import subprocess

import pytest

from scripts import verify_visual_browser_sandbox as probe
from kestrel_sovereign.visual_computer import docker_policy


def container():
    host = probe.verified_host_configuration()
    host["SecurityOpt"] = ["no-new-privileges", "seccomp=" + (probe.POLICY_ROOT / "seccomp.json").read_text()]
    return {"Id": "c" * 64, "Image": "sha256:test", "Mounts": [],
            "Config": {"Labels": {probe.LABEL: "test"}, "User": "pwuser"}, "HostConfig": host}


def test_container_admission_and_command_agree_on_isolation():
    probe.verify_container(container(), "sha256:test", "test")
    arguments = probe.container_arguments("sha256:test", "test")
    for name, value in [("--network", "none"), ("--cap-drop", "ALL"), ("--memory", "1g"),
                        ("--memory-swap", "1g"), ("--ipc", "private"), ("--cgroupns", "private"),
                        ("--runtime", "runc"), ("--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777"),
                        ("--cpus", "2"), ("--shm-size", "256m"), ("--pids-limit", "256")]:
        assert arguments[arguments.index(name) + 1] == value
    assert arguments[arguments.index("--name") + 1] == "kestrel-browser-probe-test"
    assert arguments[arguments.index("--label") + 1] == f"{probe.LABEL}=test"
    assert arguments[arguments.index("--ulimit") + 1] == "rtprio=0:0"
    assert arguments[arguments.index("--log-driver") + 1] == "none"
    assert [arguments[i + 1] for i, value in enumerate(arguments) if value == "--security-opt"] == [
        "no-new-privileges", f"seccomp={probe.POLICY_ROOT / 'seccomp.json'}"]
    assert "--read-only" in arguments and "--init" in arguments
    assert "--privileged" not in arguments and "--volume" not in arguments and "--publish" not in arguments


@pytest.mark.parametrize("field,value", [
    ("NetworkMode", "host"), ("ReadonlyRootfs", False), ("Privileged", True),
    ("CapDrop", []), ("CapAdd", ["SYS_ADMIN"]), ("Binds", ["/:/host"]),
    ("Devices", [{"PathOnHost": "/dev/kvm"}]), ("PidMode", "host"), ("IpcMode", "host"),
    ("PublishAllPorts", True), ("Memory", 0), ("NanoCpus", 0), ("PidsLimit", 0),
    ("Tmpfs", {"/tmp": "rw,exec,mode=1777", "/home/pwuser": "rw,exec"}),
    ("Ulimits", [{"Name": "rtprio", "Soft": 99, "Hard": 99}]),
    ("GroupAdd", ["0"]), ("OomScoreAdj", -1000), ("RestartPolicy", {"Name": "always", "MaximumRetryCount": 0}),
    ("LogConfig", {"Type": "syslog", "Config": {}}), ("Annotations", {"x": "y"}), ("Cgroup", "host"),
    ("MemorySwap", -1), ("Init", False), ("DeviceCgroupRules", ["a *:* rwm"]),
    ("CgroupParent", "other"), ("VolumesFrom", ["other"]), ("MaskedPaths", []), ("ReadonlyPaths", []),
    ("UTSMode", "host"), ("UsernsMode", "host"), ("CgroupnsMode", "host"),
    ("IpcMode", "container:other"), ("Runtime", "custom"), ("Sysctls", {"net.ipv4.ip_forward": "1"}),
    ("OomKillDisable", True), ("SecurityOpt", container()["HostConfig"]["SecurityOpt"] + ["systempaths=unconfined"]),
    ("SecurityOpt", ["no-new-privileges", 'seccomp={"defaultAction":"SCMP_ACT_ALLOW"}']),
])
def test_refuses_weakened_actual_docker_configuration(field, value):
    record = container(); record["HostConfig"][field] = value
    with pytest.raises(RuntimeError, match="required isolated"):
        probe.verify_container(record, "sha256:test", "test")


def test_refuses_wrong_owner_image_and_implicit_image_volumes():
    for field, value in [("Image", "other-image"), ("Mounts", [{"Type": "volume", "Destination": "/data"}])]:
        record = container(); record[field] = value
        with pytest.raises(RuntimeError): probe.verify_container(record, "sha256:test", "test")
    record = container(); record["Config"]["Labels"][probe.LABEL] = "other"
    with pytest.raises(RuntimeError): probe.verify_container(record, "sha256:test", "test")
    record = container(); record["Config"]["User"] = "root"
    with pytest.raises(RuntimeError): probe.verify_container(record, "sha256:test", "test")


def evidence():
    png = b"\x89PNG\r\n\x1a\nsynthetic"
    return {"uid": 1001, "capabilities": {name: 0 for name in ("CapEff", "CapBnd", "CapPrm", "CapInh", "CapAmb")},
            "rtprio_limit": [0, 0], "groups": [100, 1001], "no_new_privileges": True,
            "fixture_result": "Persistent local browser draft", "refused_destinations": ["1.1.1.1", "169.254.169.254"],
            "sandbox": "PID namespaces\tYes\nNetwork namespaces\tYes\nSeccomp-BPF sandbox\tYes",
            "png_base64": base64.b64encode(png).decode(), "png_sha256": hashlib.sha256(png).hexdigest()}


@pytest.mark.parametrize("change", [{"uid": 0}, {"capabilities": 1}, {"no_new_privileges": False},
                                  {"sandbox": "not sandboxed"}, {"refused_destinations": []},
                                  {"fixture_result": ""}, {"png_sha256": "0" * 64}])
def test_refuses_incomplete_or_unverified_browser_evidence(change):
    with pytest.raises(RuntimeError): probe.verify_result({**evidence(), **change})


def test_seccomp_keeps_upstream_default_deny_and_enables_only_sandbox_namespaces():
    profile = probe.verified_seccomp_policy()
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    assert set(profile["syscalls"][0]["names"]) == {"clone", "chroot", "setns", "unshare"}
    assert profile["syscalls"][0]["action"] == "SCMP_ACT_ALLOW"


@pytest.mark.parametrize("unknown_create,network_mode", [(False, "none"), (True, "none"), (False, "host")])
def test_cleanup_finds_owned_container_even_when_create_response_is_lost(tmp_path, monkeypatch, unknown_create, network_mode):
    calls = []; identity = None; tag = None
    record = container()
    record["HostConfig"]["NetworkMode"] = network_mode
    def fake_run(command, **kwargs):
        nonlocal identity, tag
        args = command[1:]; calls.append(args)
        if args[0] == "build":
            identity = args[args.index("--label")+1].split("=", 1)[1]; tag = args[args.index("--tag")+1]
            record["Config"]["Labels"][probe.LABEL] = identity
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if args[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, json.dumps([{"Id": "sha256:test", "Config": record["Config"]}]).encode(), b"")
        if args[0] == "create":
            if unknown_create: raise subprocess.TimeoutExpired(command, 30)
            return subprocess.CompletedProcess(command, 0, b"c" * 64, b"")
        if args[:2] == ["container", "inspect"]:
            return subprocess.CompletedProcess(command, 0, json.dumps([record]).encode(), b"")
        if args[0] == "start":
            return subprocess.CompletedProcess(command, 0, json.dumps(evidence()).encode(), b"")
        return subprocess.CompletedProcess(command, 0, b"", b"")
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    output = tmp_path / "evidence"
    if unknown_create:
        with pytest.raises(subprocess.TimeoutExpired): probe.run(output, docker="docker")
    elif network_mode != "none":
        with pytest.raises(RuntimeError, match="required isolated"): probe.run(output, docker="docker")
        assert not any(call[0] == "start" for call in calls), "actual admission must run before container start"
    else:
        assert probe.run(output, docker="docker")["cleanup_confirmed"] is True
    state = json.loads((output / "custody.json").read_text())
    assert state["source"]["build_context_sha256"]
    assert state["source"]["build_context_files"]["Dockerfile"]
    assert state["cleanup_confirmed"] is (not unknown_create)
    assert ["container", "rm", "--force", "c" * 64] in calls
    assert ["image", "rm", tag] in calls


def test_nonempty_evidence_directory_is_never_reused(tmp_path):
    existing = tmp_path / "existing"; existing.mkdir(); (existing / "keep").write_text("keep")
    with pytest.raises(FileExistsError): probe.run(existing, docker="must-not-run")
    assert (existing / "keep").read_text() == "keep"


def test_daemon_socket_error_is_not_proof_that_owned_resources_are_absent(tmp_path, monkeypatch):
    def failure(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, b"", b"Cannot connect: No such file or directory")
    monkeypatch.setattr(probe.subprocess, "run", failure)
    output = tmp_path / "evidence"
    with pytest.raises(RuntimeError, match="cleanup could not verify"):
        probe.run(output, docker="docker")
    assert json.loads((output / "custody.json").read_text())["cleanup_confirmed"] is False


def test_cleanup_refuses_a_replaced_container_owner(tmp_path, monkeypatch):
    calls, identity, inspections = [], None, 0
    def engine(command, **kwargs):
        nonlocal identity, inspections
        args = command[1:]; calls.append(args); record = container()
        if args[0] == "build":
            identity = args[args.index("--label") + 1].split("=", 1)[1]
        record["Config"]["Labels"][probe.LABEL] = identity
        if args[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, json.dumps([{"Id": "sha256:test", "Config": record["Config"]}]).encode(), b"")
        if args[0] == "create":
            return subprocess.CompletedProcess(command, 0, b"c" * 64, b"")
        if args[:2] == ["container", "inspect"]:
            inspections += 1
            if inspections > 1: record["Config"]["Labels"][probe.LABEL] = "other"
            return subprocess.CompletedProcess(command, 0, json.dumps([record]).encode(), b"")
        if args[0] == "start":
            return subprocess.CompletedProcess(command, 1, b"", b"synthetic failure")
        return subprocess.CompletedProcess(command, 0, b"", b"")
    monkeypatch.setattr(probe.subprocess, "run", engine)
    with pytest.raises(RuntimeError, match="unowned container"):
        probe.run(tmp_path / "evidence", docker="docker")
    assert not any("rm" in call for call in calls)


def test_modified_seccomp_policy_is_rejected_before_docker_build(tmp_path, monkeypatch):
    context = tmp_path / 'context'; context.mkdir()
    profile = json.loads((probe.POLICY_ROOT / 'seccomp.json').read_text())
    profile['syscalls'].append({'names': ['bpf', 'mount', 'pivot_root', 'umount2', 'keyctl', 'add_key', 'perf_event_open'],
                                'action': 'SCMP_ACT_ALLOW'})
    (context / 'seccomp.json').write_text(json.dumps(profile))
    monkeypatch.setattr(docker_policy, 'POLICY_ROOT', context)
    calls = []
    def engine(command, **kwargs):
        calls.append(command)
        raise AssertionError('modified policy reached Docker')
    monkeypatch.setattr(probe.subprocess, 'run', engine)
    with pytest.raises(RuntimeError, match='seccomp policy digest'):
        probe.run(tmp_path / 'evidence', docker='docker')
    assert calls == []


@pytest.mark.parametrize('change', [
    {'rtprio_limit': [99, 99]}, {'groups': [0, 1001]},
    {'capabilities': {'CapEff': 0, 'CapBnd': 0x2000, 'CapPrm': 0, 'CapInh': 0, 'CapAmb': 0}},
])
def test_refuses_unsafe_measured_process_limits(change):
    with pytest.raises(RuntimeError): probe.verify_result({**evidence(), **change})


def test_refuses_unknown_actual_host_configuration_fields():
    record = container(); record['HostConfig']['FutureDaemonSetting'] = True
    with pytest.raises(RuntimeError, match='required isolated'):
        probe.verify_container(record, 'sha256:test', 'test')


def test_host_configuration_profile_is_pinned_before_docker(tmp_path, monkeypatch):
    context = tmp_path / 'context'; context.mkdir()
    (context / 'seccomp.json').write_bytes((probe.POLICY_ROOT / 'seccomp.json').read_bytes())
    profile = probe.verified_host_configuration(); profile['GroupAdd'] = ['0']
    (context / 'host-config.json').write_text(json.dumps(profile))
    monkeypatch.setattr(docker_policy, 'POLICY_ROOT', context)
    with pytest.raises(RuntimeError, match='host configuration digest'):
        probe.run(tmp_path / 'evidence', docker='must-not-be-called')


def test_source_identity_binds_actual_nested_context_bytes_and_dirty_head(tmp_path, monkeypatch):
    context = tmp_path / "context"
    (context / "nested").mkdir(parents=True)
    (context / "Dockerfile").write_text("COPY nested /opt/worker")
    source = context / "nested" / "worker.py"
    source.write_text("original")
    monkeypatch.setattr(probe, "CONTEXT", context)

    def git(command, **options):
        output = b"a" * 40 if "rev-parse" in command else b" M nested/worker.py"
        return subprocess.CompletedProcess(command, 0, output, b"")

    monkeypatch.setattr(probe.subprocess, "run", git)
    before = probe.source_identity()
    assert before["revision"] == "a" * 40 and before["dirty"] is True
    assert before["build_context_files"]["nested/worker.py"] == hashlib.sha256(b"original").hexdigest()
    source.write_text("changed")
    assert probe.source_identity()["build_context_sha256"] != before["build_context_sha256"]


@pytest.mark.parametrize("status", ["running", "exited"])
def test_measured_docker29_post_start_null_oom_default_is_explicitly_supported(status):
    value = container()
    value["State"] = {"Status": status}
    value["HostConfig"]["OomKillDisable"] = None
    probe.verify_container(value, "sha256:test", "test")
    value["HostConfig"]["OomKillDisable"] = True
    with pytest.raises(RuntimeError):
        probe.verify_container(value, "sha256:test", "test")


def test_created_container_cannot_normalize_away_unreviewed_oom_setting():
    value = container()
    value["State"] = {"Status": "created"}
    value["HostConfig"]["OomKillDisable"] = None
    with pytest.raises(RuntimeError):
        probe.verify_container(value, "sha256:test", "test")
