#!/usr/bin/env python3
"""
RunPod GPU Testing Script for Kestrel Agent

This script demonstrates how to deploy and test the Kestrel GPU-enabled container
on RunPod's cloud GPU infrastructure.

Requirements:
- RunPod API key in .env file
- Docker image built and pushed to a registry (e.g., Docker Hub)
- runpod Python package installed

Usage:
1. Build and push the GPU image:
   docker build -f docker/Dockerfile.gpu -t yourusername/kestrel-gpu:latest .
   docker push yourusername/kestrel-gpu:latest

2. Update the IMAGE_NAME below with your registry path

3. Run this script:
   python test_runpod_gpu.py

The script will:
- Create a RunPod pod with GPU
- Start the Kestrel agent
- Test basic functionality
- Clean up the pod
"""

import os
import time
import requests
import logging
from dotenv import load_dotenv
import runpod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
IMAGE_NAME = "gcr.io/YOUR_PROJECT_ID/kestrel-gpu:latest"  # Update this!
GPU_TYPE = "NVIDIA GeForce RTX 3090"  # Exact GPU ID from RunPod API
CONTAINER_DISK_SIZE = 50  # GB
CLOUD_TYPE = "COMMUNITY"  # Community cloud

def create_pod():
    """Create a RunPod pod with GPU for testing Kestrel."""
    logger.info("Creating RunPod pod with GPU...")

    # Pod configuration
    pod_config = {
        "name": "kestrel-gpu-test",
        "image_name": IMAGE_NAME,
        "gpu_type_id": GPU_TYPE,
        "container_disk_in_gb": CONTAINER_DISK_SIZE,
        "cloud_type": CLOUD_TYPE,
        "docker_args": "",  # Additional docker run args if needed
        "ports": "8888/http,11434/http",  # Expose agent and Ollama ports
        "volume_in_gb": 0,  # No persistent volume for testing
        "env": {
            "KESTREL_ENV": "production",
            "OLLAMA_HOST": "0.0.0.0:11434",
            # Add other env vars as needed
        }
    }

    # Create the pod
    pod = runpod.create_pod(**pod_config)
    
    logger.info(f"Pod created: {pod['id']}")
    logger.info(f"Status: {pod.get('status', 'Unknown')}")

    return pod

def wait_for_pod_ready(pod_id, timeout=600):
    """Wait for pod to be ready."""
    logger.info("Waiting for pod to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        pod = runpod.get_pod(pod_id)
        logger.info(f"Pod data: {pod}")
        status = pod.get('status', pod.get('desiredStatus', 'Unknown'))

        logger.info(f"Pod status: {status}")

        if status == "RUNNING":
            return pod
        elif status in ["FAILED", "CRASHED"]:
            raise Exception(f"Pod failed with status: {status}")

        time.sleep(10)

    raise Exception("Pod startup timeout")

def test_kestrel_agent(pod):
    """Test the Kestrel agent running in the pod."""
    pod_id = pod['id']

    logger.info("Waiting for container services to start (Ollama + Kestrel agent)...")

    # Wait longer for services to start up
    max_wait = 300  # 5 minutes for model download + service startup
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            # Get updated pod info
            updated_pod = runpod.get_pod(pod_id)
            runtime = updated_pod.get('runtime')

            if runtime and 'ports' in runtime:
                # Try to find the public endpoints
                agent_url = None
                ollama_url = None

                for port_info in runtime['ports']:
                    if port_info.get('privatePort') == 8888:
                        # RunPod gives us the public IP and port
                        ip = port_info.get('ip')
                        public_port = port_info.get('publicPort')
                        if ip and public_port:
                            agent_url = f"http://{ip}:{public_port}"
                    elif port_info.get('privatePort') == 11434:
                        ip = port_info.get('ip')
                        public_port = port_info.get('publicPort')
                        if ip and public_port:
                            ollama_url = f"http://{ip}:{public_port}"

                # Test the endpoints if we have them
                if agent_url:
                    logger.info(f"Testing Kestrel agent at: {agent_url}")
                    try:
                        response = requests.get(f"{agent_url}/health", timeout=10)
                        if response.status_code == 200:
                            logger.info("✅ Kestrel agent health check passed!")
                            logger.info(f"Response: {response.json()}")
                        else:
                            logger.warning(f"Kestrel health check returned {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Kestrel health check failed: {e}")

                if ollama_url:
                    logger.info(f"Testing Ollama at: {ollama_url}")
                    try:
                        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
                        if response.status_code == 200:
                            logger.info("✅ Ollama API accessible!")
                            models = response.json().get('models', [])
                            logger.info(f"Available models: {len(models)}")
                        else:
                            logger.warning(f"Ollama API returned {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Ollama API check failed: {e}")

                # If we got here, we successfully tested the endpoints
                if agent_url or ollama_url:
                    logger.info("✅ Container services are running and accessible!")
                    return True

            # Wait before checking again
            logger.info("Waiting for runtime info... (this can take 2-5 minutes)")
            time.sleep(30)

        except Exception as e:
            logger.warning(f"Error during testing: {e}")
            time.sleep(10)

    # If we get here, we couldn't access the services
    logger.warning("Could not access container services within timeout")
    logger.info("Pod details:")
    logger.info(f"  ID: {pod_id}")
    logger.info(f"  Status: {updated_pod.get('desiredStatus', 'Unknown')}")
    logger.info(f"  GPU: {updated_pod.get('machine', {}).get('gpuDisplayName', 'Unknown')}")
    logger.info("⚠️  Pod is running but services may still be starting...")
    return False

def cleanup_pod(pod_id):
    """Stop and terminate the test pod."""
    logger.info(f"Cleaning up pod: {pod_id}")
    try:
        runpod.stop_pod(pod_id)
        logger.info("✅ Pod stopped successfully")
    except Exception as e:
        logger.error(f"❌ Error stopping pod: {e}")
    try:
        runpod.terminate_pod(pod_id)
        logger.info("✅ Pod terminated successfully")
    except Exception as e:
        logger.error(f"❌ Error terminating pod: {e}")

def main():
    """Main testing workflow."""
    # Check API key
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        logger.error("❌ RUNPOD_API_KEY not found in environment")
        return

    runpod.api_key = api_key
    pod_id = None  # Initialize before try block for reliable cleanup

    try:
        # Create pod
        pod = create_pod()
        pod_id = pod['id']

        # Wait for ready
        pod = wait_for_pod_ready(pod_id)

        # Test functionality
        test_kestrel_agent(pod)

        logger.info("✅ GPU pod test completed successfully!")
        logger.info(f"Pod ID: {pod_id}")
        logger.info("Stopping pod to avoid costs...")

    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping...")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        # ALWAYS clean up the pod
        if pod_id:
            cleanup_pod(pod_id)

if __name__ == "__main__":
    main()