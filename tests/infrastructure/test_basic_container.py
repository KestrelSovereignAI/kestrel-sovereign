"""
Basic container test for RunPod infrastructure.

Tests that a simple Ubuntu container can start on RunPod GPU infrastructure.
Pod is ALWAYS terminated in finally block to prevent orphaned pods.

Requires: RUNPOD_API_KEY environment variable
Marker: cloud_resource (requires --run-cloud to execute)
"""
import os
import time

import pytest


pytestmark = [
    pytest.mark.cloud_resource,
    pytest.mark.slow,
]


@pytest.fixture
def runpod_client():
    """Initialize RunPod client with API key."""
    import runpod
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        pytest.skip("RUNPOD_API_KEY not set")

    runpod.api_key = api_key
    return runpod


@pytest.fixture
def cleanup_pod(runpod_client):
    """Fixture that ensures pod cleanup after test."""
    pod_ids = []

    yield pod_ids  # Test can append pod_id to this list

    # Cleanup all created pods
    for pod_id in pod_ids:
        if pod_id:
            print(f"Cleaning up pod {pod_id}...")
            try:
                runpod_client.stop_pod(pod_id)
                print("Pod stopped.")
            except Exception as e:
                print(f"Error stopping pod: {e}")
            try:
                runpod_client.terminate_pod(pod_id)
                print("Pod terminated.")
            except Exception as e:
                print(f"Error terminating pod: {e}")


def test_basic_ubuntu_container(runpod_client, cleanup_pod):
    """Test that a basic Ubuntu container can start on RunPod."""
    pod_config = {
        "name": "basic-test",
        "image_name": "ubuntu:20.04",
        "gpu_type_id": "NVIDIA GeForce RTX 3090",
        "container_disk_in_gb": 20,
        "cloud_type": "COMMUNITY",
        "docker_args": "echo 'Container started successfully' && sleep 300",
        "ports": "",
        "volume_in_gb": 0,
        "env": {}
    }

    print("Creating basic Ubuntu test pod...")
    response = runpod_client.create_pod(**pod_config)
    pod_id = response['id']
    cleanup_pod.append(pod_id)  # Ensure cleanup
    print(f"Pod created: {pod_id}")

    # Wait and check status
    print("Waiting for pod...")
    final_status = None
    for i in range(24):  # 2 minutes max
        pod_info = runpod_client.get_pod(pod_id)
        status = pod_info.get('desiredStatus')
        runtime = pod_info.get('runtime')
        print(f"Status: {status}, Runtime: {runtime is not None}")

        if status in ['FAILED', 'CRASHED']:
            final_status = status
            break
        elif status == 'RUNNING' and runtime:
            final_status = 'RUNNING'
            print(f"Runtime: {runtime}")
            break

        time.sleep(5)

    assert final_status == 'RUNNING', f"Pod did not reach RUNNING state: {final_status}"
