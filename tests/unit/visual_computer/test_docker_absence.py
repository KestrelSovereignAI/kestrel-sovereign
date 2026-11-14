"""Native Docker absence facts stay exact and never classify transport errors."""
import os
import sys

import pytest
from kestrel_sdk.resources import ResourceNotFound
from kestrel_sovereign.visual_computer.docker import (
    DockerBoundary, DockerCommandError, _docker_resource_absent,
)

TARGET = "owned-volume"
VOLUME_ARGS = ["volume", "inspect", TARGET]
VOLUME_ERROR = b"Error response from daemon: get owned-volume: no such volume\n"


def test_exact_volume_api_refusal_proves_absence():
    assert _docker_resource_absent(VOLUME_ARGS, 1, b"[]\n", VOLUME_ERROR)


@pytest.mark.parametrize("arguments,code,output,error", [
    (["container", "inspect", TARGET], 1, b"[]\n", VOLUME_ERROR),
    (["network", "inspect", TARGET], 1, b"[]\n", VOLUME_ERROR),
    (["image", "inspect", TARGET], 1, b"[]\n", VOLUME_ERROR),
    (["volume", "ls", TARGET], 1, b"[]\n", VOLUME_ERROR),
    (["volume", "inspect", "--", TARGET], 1, b"[]\n", VOLUME_ERROR),
    (VOLUME_ARGS, 0, b"[]\n", VOLUME_ERROR),
    (VOLUME_ARGS, 2, b"[]\n", VOLUME_ERROR),
    (VOLUME_ARGS, -1, b"[]\n", VOLUME_ERROR),
    (VOLUME_ARGS, True, b"[]\n", VOLUME_ERROR),
    (VOLUME_ARGS, 1, b"", VOLUME_ERROR),
    (VOLUME_ARGS, 1, b"[]", VOLUME_ERROR),
    (VOLUME_ARGS, 1, b"[{}]\n", VOLUME_ERROR),
    (VOLUME_ARGS, 1, b"{}\n", VOLUME_ERROR),
    (VOLUME_ARGS, 1, b" []\n", VOLUME_ERROR),
    (VOLUME_ARGS, 1, b"[]\n", b"Error response from daemon: get other-volume: no such volume\n"),
    (VOLUME_ARGS, 1, b"[]\n", b"Error response from daemon: get owned-volume-extra: no such volume\n"),
    (VOLUME_ARGS, 1, b"[]\n", VOLUME_ERROR[:-1]),
    (VOLUME_ARGS, 1, b"[]\n", VOLUME_ERROR+b"\n"),
    (VOLUME_ARGS, 1, b"[]\n", b"Error: get owned-volume: no such volume\n"),
    (VOLUME_ARGS, 1, b"[]\n", b"Error response from daemon: get owned-volume: No such volume\n"),
    (VOLUME_ARGS, 1, b"[]\n", b"Error response from daemon: get owned-volume: permission denied\n"),
    (VOLUME_ARGS, 1, b"[]\n", b"Cannot connect to the Docker daemon\n"),
])
def test_incomplete_foreign_or_transport_refusal_is_not_absence(arguments, code, output, error):
    assert not _docker_resource_absent(arguments, code, output, error)


NETWORK_TARGET = "owned-network"
NETWORK_ARGS = ["network", "inspect", NETWORK_TARGET]
NETWORK_ERROR = b"Error response from daemon: network owned-network not found\n"


def test_exact_network_api_refusal_proves_absence():
    assert _docker_resource_absent(NETWORK_ARGS, 1, b"[]\n", NETWORK_ERROR)


@pytest.mark.parametrize("arguments,code,output,error", [
    (["volume", "inspect", NETWORK_TARGET], 1, b"[]\n", NETWORK_ERROR),
    (["container", "inspect", NETWORK_TARGET], 1, b"[]\n", NETWORK_ERROR),
    (["image", "inspect", NETWORK_TARGET], 1, b"[]\n", NETWORK_ERROR),
    (["network", "inspect", "--", NETWORK_TARGET], 1, b"[]\n", NETWORK_ERROR),
    (NETWORK_ARGS, 0, b"[]\n", NETWORK_ERROR),
    (NETWORK_ARGS, 2, b"[]\n", NETWORK_ERROR),
    (NETWORK_ARGS, True, b"[]\n", NETWORK_ERROR),
    (NETWORK_ARGS, 1, b"", NETWORK_ERROR),
    (NETWORK_ARGS, 1, b"[]", NETWORK_ERROR),
    (NETWORK_ARGS, 1, b"{}\n", NETWORK_ERROR),
    (NETWORK_ARGS, 1, b"[]\n", b"Error response from daemon: network other-network not found\n"),
    (NETWORK_ARGS, 1, b"[]\n", b"Error response from daemon: network owned-network-extra not found\n"),
    (NETWORK_ARGS, 1, b"[]\n", NETWORK_ERROR[:-1]),
    (NETWORK_ARGS, 1, b"[]\n", NETWORK_ERROR+b"\n"),
    (NETWORK_ARGS, 1, b"[]\n", b"Error response from daemon: Network owned-network not found\n"),
    (NETWORK_ARGS, 1, b"[]\n", b"Error: network owned-network not found\n"),
    (NETWORK_ARGS, 1, b"[]\n", b"Error response from daemon: network owned-network permission denied\n"),
    (NETWORK_ARGS, 1, b"[]\n", b"Cannot connect to the Docker daemon\n"),
])
def test_network_wrong_target_kind_incomplete_and_transport_errors_remain_unknown(arguments, code, output, error):
    assert not _docker_resource_absent(arguments, code, output, error)


@pytest.mark.parametrize("kind", ["container", "image", "network", "volume"])
@pytest.mark.parametrize("prefix", [b"Error: ", b"Error response from daemon: "])
@pytest.mark.parametrize("code", [1, 2])
def test_original_exact_daemon_formats_remain_supported(kind, prefix, code):
    assert _docker_resource_absent([kind, "inspect", TARGET], code, b"",
        prefix+b"No such object: owned-volume\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["volume-api", "original", "wrong-target", "network-api", "wrong-network-target"])
async def test_canonical_boundary_wires_native_response_classification(tmp_path, monkeypatch, style):
    # Controlled CLI output double, with REAL original Process/pipes/waits.
    # No daemon/constructor/checker/authority is replaced or contacted here.
    error = (VOLUME_ERROR if style == "volume-api" else
        b"Error response from daemon: No such volume: owned-volume\n" if style == "original" else
        b"Error response from daemon: get other-volume: no such volume\n")
    arguments = VOLUME_ARGS
    if style in {"network-api", "wrong-network-target"}:
        arguments = NETWORK_ARGS
        error = (NETWORK_ERROR if style == "network-api" else
                 b"Error response from daemon: network other-network not found\n")
    program = tmp_path/"docker"
    program.write_text(f"#!{sys.executable} -B\nimport sys\n"
        f"sys.stdin.buffer.read()\nsys.stdout.buffer.write({b'[]' + bytes([10])!r})\n"
        f"sys.stderr.buffer.write({error!r})\nraise SystemExit(1)\n")
    program.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path)+os.pathsep+os.environ.get("PATH", ""))
    boundary = DockerBoundary(str(tmp_path/"not-contacted.sock"))
    assert boundary.executable == str(program)
    expected = DockerCommandError if style in {"wrong-target", "wrong-network-target"} else ResourceNotFound
    with pytest.raises(expected) as caught:
        await boundary.invoke(arguments)
    if style in {"wrong-target", "wrong-network-target"}:
        assert caught.value.diagnostics.get_secret_value() == b"[]\n"+error
