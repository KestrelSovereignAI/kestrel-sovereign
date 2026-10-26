"""Read a local test daemon endpoint without personal paths or remote fallback.

Discovery reads Docker context metadata only. The actual adapter still uses an
explicit Unix socket and its own empty Docker client configuration.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import unquote, urlsplit


class LocalDockerUnavailable(RuntimeError):
    pass


def _socket_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("local Docker test socket must be an absolute path")
    if not path.is_socket():
        raise LocalDockerUnavailable("selected local Docker test socket is unavailable")
    return str(path.resolve())


def local_docker_socket():
    explicit = os.environ.get("ARTIFACT_TEST_DOCKER_SOCKET")
    if explicit is not None:
        return _socket_path(explicit)
    endpoint = os.environ.get("DOCKER_HOST") if not os.environ.get("DOCKER_CONTEXT") else None
    if endpoint is None:
        executable = shutil.which("docker")
        if executable is None:
            raise LocalDockerUnavailable("local Docker CLI/context discovery is unavailable")
        environment = {key:os.environ[key] for key in ("HOME", "PATH", "DOCKER_CONTEXT") if key in os.environ}
        try:
            result = subprocess.run([executable, "context", "inspect", "--format", "{{json .Endpoints.docker.Host}}"],
                env=environment, capture_output=True, timeout=5, check=True)
            endpoint = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            raise LocalDockerUnavailable("local Docker context could not be discovered") from None
    if not isinstance(endpoint, str):
        raise LocalDockerUnavailable("Docker context has no local Unix endpoint")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "unix" or parsed.netloc or parsed.query or parsed.fragment:
        raise LocalDockerUnavailable("real Docker tests require a local Unix context")
    return _socket_path(unquote(parsed.path))
