"""Local endpoint discovery never turns a remote Docker context into a test."""
import json
from pathlib import Path
import socket
import subprocess
import tempfile

import pytest

from tests.shared import local_docker as discovery


@pytest.fixture
def daemon_path():
    with tempfile.TemporaryDirectory(prefix="ks-dsock-", dir="/tmp") as directory:
        path=Path(directory)/"d.sock"
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(path))
            yield str(path.resolve())


def test_explicit_test_socket_wins_without_reading_operator_context(monkeypatch, daemon_path):
    monkeypatch.setenv("ARTIFACT_TEST_DOCKER_SOCKET", daemon_path)
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.invalid:2375")
    monkeypatch.setattr(discovery.subprocess,"run",lambda *a,**k:pytest.fail("explicit socket must not read a context"))
    assert discovery.local_docker_socket()==daemon_path


def test_discovers_only_local_context_metadata_with_sanitized_environment(monkeypatch, daemon_path):
    monkeypatch.delenv("ARTIFACT_TEST_DOCKER_SOCKET",raising=False)
    monkeypatch.delenv("DOCKER_HOST",raising=False)
    monkeypatch.setenv("DOCKER_CONTEXT","owned-local")
    monkeypatch.setenv("PRIVATE_TEST_CREDENTIAL","must-not-be-forwarded")
    monkeypatch.setattr(discovery.shutil,"which",lambda name:"/test/docker")
    def context(command,**options):
        assert command==["/test/docker","context","inspect","--format","{{json .Endpoints.docker.Host}}"]
        assert set(options["env"]) <= {"HOME","PATH","DOCKER_CONTEXT"}
        assert options["timeout"]==5 and options["check"] is True
        return subprocess.CompletedProcess(command,0,json.dumps("unix://"+daemon_path).encode(),b"")
    monkeypatch.setattr(discovery.subprocess,"run",context)
    assert discovery.local_docker_socket()==daemon_path


@pytest.mark.parametrize("endpoint",["tcp://remote.invalid:2375","ssh://owner@remote.invalid","unix://remote.invalid/socket"])
def test_remote_endpoint_is_refused_without_connecting(monkeypatch,endpoint):
    monkeypatch.delenv("ARTIFACT_TEST_DOCKER_SOCKET",raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT",raising=False)
    monkeypatch.setenv("DOCKER_HOST",endpoint)
    monkeypatch.setattr(discovery.subprocess,"run",lambda *a,**k:pytest.fail("remote endpoint must not connect"))
    with pytest.raises(discovery.LocalDockerUnavailable,match="local Unix"):
        discovery.local_docker_socket()


def test_missing_or_non_socket_target_is_not_an_implicit_fallback(monkeypatch,tmp_path):
    regular=tmp_path/"regular";regular.write_text("not a socket")
    for target in (str(regular),str(tmp_path/"missing")):
        monkeypatch.setenv("ARTIFACT_TEST_DOCKER_SOCKET",target)
        with pytest.raises(discovery.LocalDockerUnavailable):discovery.local_docker_socket()
    monkeypatch.setenv("ARTIFACT_TEST_DOCKER_SOCKET","relative.sock")
    with pytest.raises(ValueError,match="absolute"):discovery.local_docker_socket()
