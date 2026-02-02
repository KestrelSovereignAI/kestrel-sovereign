"""
Docker integration tests for Kestrel.

These tests verify that the Docker image builds correctly and the
container starts up properly with health checks passing.

Run with: pytest -m docker tests/integration/test_docker.py
"""
import os
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


@pytest.mark.skip(reason="Infrastructure test - takes too long for regular runs. Use pytest -m docker to run explicitly.")
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
            "-p", "7777:7777",
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
            health_ok = wait_for_health("http://127.0.0.1:7777/health", timeout=30.0)
            assert health_ok, "Health check did not return 200 within timeout"

            # Verify health response content
            resp = requests.get("http://127.0.0.1:7777/health", timeout=5)
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data

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







