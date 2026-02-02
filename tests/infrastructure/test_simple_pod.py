"""
Simple pod test for RunPod infrastructure.

Basic test to verify RunPod API connectivity and pod creation.
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


def test_simple_pod_creation(runpod_client, cleanup_pod):
    """Test basic RunPod API connectivity and pod creation."""
    pod_config = {
        "name": "simple-test",
        "image_name": "gcr.io/YOUR_PROJECT_ID/kestrel-gpu:latest",
        "gpu_type_id": "NVIDIA GeForce RTX 3090",
        "container_disk_in_gb": 20,
        "cloud_type": "COMMUNITY",
        "ports": "8888/http",
        "volume_in_gb": 0,
        "env": {
            "TEST_MODE": "1"
        }
    }

    print("Creating simple test pod...")
    response = runpod_client.create_pod(**pod_config)
    print(f"Raw response: {response}")

    # Handle different response structures
    if 'data' in response and 'podFindAndDeployOnDemand' in response['data']:
        pod = response['data']['podFindAndDeployOnDemand']
    elif 'id' in response:
        pod = response
    else:
        pytest.fail(f"Unexpected response structure: {response}")

    pod_id = pod['id']
    cleanup_pod.append(pod_id)  # Ensure cleanup
    print(f"Pod created: {pod_id}")

    # Wait for it to start
    print("Waiting for pod...")
    final_status = None
    for i in range(60):  # 5 minutes
        pod_info = runpod_client.get_pod(pod_id)
        status = pod_info.get('desiredStatus')
        print(f"Status: {status}")

        if status == 'RUNNING':
            final_status = 'RUNNING'
            print("Pod is running!")
            runtime = pod_info.get('runtime')
            if runtime:
                print(f"Runtime info: {runtime}")
            else:
                print("No runtime info yet")
            break
        elif status in ['FAILED', 'CRASHED']:
            final_status = status
            print(f"Pod failed with status: {status}")
            break

        time.sleep(5)

    print(f"Final pod info: {runpod_client.get_pod(pod_id)}")
    assert final_status == 'RUNNING', f"Pod did not reach RUNNING state: {final_status}"
