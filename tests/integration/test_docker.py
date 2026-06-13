"""
Docker integration tests for Kestrel.

These tests verify that the Docker image builds correctly and the
container starts up properly with health checks passing.

Run with: pytest -m docker tests/integration/test_docker.py
"""
import os
import shutil
import subprocess
import time
import tempfile
import pytest
import requests
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import shared test utilities for crash-safe resource tracking
try:
    from tests.shared import registry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

IMAGE_TAG = "kestrel:test"
CONTAINER_NAME = "kestrel-docker-test"
RUN_DOCKER_TESTS = os.environ.get("KESTREL_TEST_DOCKER", "").lower() in {
    "1",
    "true",
    "yes",
}


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0


def docker_compose_available() -> bool:
    if not docker_available():
        return False
    result = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def wait_for_health(url: str, timeout: float = 30.0, interval: float = 1.0) -> bool:
    """
    Wait for health endpoint to return 200.

    Returns True if healthy, False if timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
    return False


@pytest.mark.skipif(
    not RUN_DOCKER_TESTS or not docker_available(),
    reason="Docker e2e test; set KESTREL_TEST_DOCKER=1 and run pytest -m docker.",
)
@pytest.mark.docker
@pytest.mark.integration
def test_docker_healthcheck():
    """
    Test that the Docker image builds and starts correctly.

    Uses --cidfile for reliable container ID capture and registers
    with the resource registry for crash recovery.
    """
    # Build the image
    build = subprocess.run(
        ["docker", "build", "-t", IMAGE_TAG, "."],
        capture_output=True,
        text=True
    )
    assert build.returncode == 0, f"docker build failed: {build.stderr}\n{build.stdout}"

    # Create temp files for cidfile and agent data
    with tempfile.TemporaryDirectory() as agent_dir:
        cidfile = os.path.join(agent_dir, "container.cid")

        # Run container with --cidfile for reliable ID capture
        # Remove --rm so we can inspect on failure, we'll clean up manually
        proc = subprocess.Popen([
            "docker", "run",
            "--cidfile", cidfile,
            "--name", CONTAINER_NAME,
            "-p", "8888:8888",
            "-v", f"{agent_dir}:/app/agent_data",
            IMAGE_TAG
        ])

        container_id = None
        resource_id = None

        try:
            # Wait for cidfile to be written (Docker writes it when container starts)
            for _ in range(10):
                if os.path.exists(cidfile) and os.path.getsize(cidfile) > 0:
                    with open(cidfile) as f:
                        container_id = f.read().strip()
                    break
                time.sleep(0.5)

            assert container_id, "Container ID not captured - container may have failed to start"

            # Register with resource registry for crash recovery
            if REGISTRY_AVAILABLE:
                resource_id = registry.track_docker(container_id, "kestrel-docker-test")
                print(f"📝 Registered container with ID: {resource_id}")

            # Wait for health check with condition-based wait
            health_ok = wait_for_health("http://127.0.0.1:8888/health", timeout=30.0)
            assert health_ok, "Health check did not return 200 within timeout"

            # Verify health response content
            resp = requests.get("http://127.0.0.1:8888/health", timeout=5)
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert (Path(agent_dir) / "kestrel_prime.db").exists()

        finally:
            # Clean up container properly
            if container_id:
                # Stop gracefully first
                subprocess.run(
                    ["docker", "stop", container_id],
                    capture_output=True,
                    timeout=10
                )
                # Remove container
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    capture_output=True
                )

            # Also try by name in case cidfile didn't work
            subprocess.run(
                ["docker", "rm", "-f", CONTAINER_NAME],
                capture_output=True
            )

            # Unregister from resource registry
            if resource_id and REGISTRY_AVAILABLE:
                registry.untrack(resource_id)

            # Clean up cidfile
            if os.path.exists(cidfile):
                os.unlink(cidfile)


@pytest.mark.skipif(
    not RUN_DOCKER_TESTS or not docker_compose_available(),
    reason="Docker Compose e2e test; set KESTREL_TEST_DOCKER=1 and run pytest -m docker.",
)
@pytest.mark.docker
@pytest.mark.integration
def test_docker_compose_fresh_boot_initializes_mounted_agent_data():
    """Fresh compose boot writes the DB where the server later reads it."""
    project_name = f"kestrel-dbpath-{os.getpid()}"

    with tempfile.TemporaryDirectory() as temp_root:
        temp_root_path = Path(temp_root)
        agent_dir = temp_root_path / "agent_data"
        agent_dir.mkdir()
        override_path = temp_root_path / "compose.override.yml"
        override_path.write_text(
            "services:\n"
            "  kestrel_app:\n"
            "    volumes:\n"
            f"      - {agent_dir}:/app/agent_data\n",
            encoding="utf-8",
        )

        compose_cmd = [
            "docker",
            "compose",
            "-p",
            project_name,
            "-f",
            str(project_root / "docker-compose.yml"),
            "-f",
            str(override_path),
        ]

        up = subprocess.run(
            [*compose_cmd, "up", "--build", "-d", "kestrel_app"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,
        )
        assert up.returncode == 0, f"docker compose up failed: {up.stderr}\n{up.stdout}"

        try:
            health_ok = wait_for_health("http://127.0.0.1:8888/health", timeout=180.0)
            logs = subprocess.run(
                [*compose_cmd, "logs", "--no-color", "kestrel_app"],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=30,
            )
            assert health_ok, (
                "Compose health check did not return 200 within timeout.\n"
                f"Logs:\n{logs.stdout}\n{logs.stderr}"
            )
            assert (agent_dir / "kestrel_prime.db").exists()
        finally:
            subprocess.run(
                [*compose_cmd, "down", "-v"],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=60,
            )
