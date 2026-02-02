"""
SSH-enabled pod test for RunPod infrastructure.

Tests SSH connectivity to a RunPod pod. Pod is ALWAYS terminated
in finally block to prevent orphaned pods.

Requires: RUNPOD_API_KEY environment variable
Marker: cloud_resource (requires --run-cloud to execute)

Note: This test verifies SSH becomes available, then cleans up.
For interactive SSH debugging, use manual_debug_pod.py instead.
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


def test_ssh_enabled_pod(runpod_client, cleanup_pod):
    """Test that a pod with SSH enabled becomes accessible."""
    pod_config = {
        "name": "ssh-test",
        "image_name": "gcr.io/YOUR_PROJECT_ID/kestrel-gpu:latest",
        "gpu_type_id": "NVIDIA GeForce RTX 3090",
        "container_disk_in_gb": 20,
        "cloud_type": "COMMUNITY",
        "start_ssh": True,  # Enable SSH
        "ports": "22/tcp,8888/http,11434/http",
        "volume_in_gb": 0,
        "env": {}
    }

    print("Creating pod with SSH enabled...")
    response = runpod_client.create_pod(**pod_config)
    pod_id = response['id']
    cleanup_pod.append(pod_id)  # Ensure cleanup
    print(f"Pod created: {pod_id}")

    # Wait for pod to be ready with SSH
    print("Waiting for pod with SSH...")
    ssh_available = False
    final_status = None

    for i in range(60):  # 5 minutes max
        pod_info = runpod_client.get_pod(pod_id)
        status = pod_info.get('desiredStatus')
        runtime = pod_info.get('runtime')

        print(f"Status: {status}, Runtime available: {runtime is not None}")

        if status == 'RUNNING' and runtime:
            print("Pod ready with runtime info!")
            print(f"Runtime: {runtime}")

            # Check if SSH port is available
            if runtime and 'ports' in runtime:
                for port_info in runtime['ports']:
                    if port_info.get('privatePort') == 22:
                        ssh_ip = port_info.get('ip')
                        ssh_port = port_info.get('publicPort')
                        if ssh_ip and ssh_port:
                            print(f"SSH available at: {ssh_ip}:{ssh_port}")
                            ssh_available = True
            final_status = 'RUNNING'
            break
        elif status in ['FAILED', 'CRASHED']:
            final_status = status
            print(f"Pod failed with status: {status}")
            break

        time.sleep(5)

    assert final_status == 'RUNNING', f"Pod did not reach RUNNING state: {final_status}"
    assert ssh_available, "SSH port not available in runtime info"
