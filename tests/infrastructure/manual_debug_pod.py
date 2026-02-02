#!/usr/bin/env python3
"""
Manual debug helper for RunPod pods.

Gets info about the most recent pod including logs.
This is a utility script, not a test - run directly with python.

Usage:
    python manual_debug_pod.py
"""
import os


def main():
    """Get debug info for the most recent pod."""
    import runpod
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("Error: RUNPOD_API_KEY not set")
        return

    runpod.api_key = api_key

    # Get the most recent pod
    pods = runpod.get_pods()
    if pods:
        latest_pod = pods[0]  # Most recent first
        pod_id = latest_pod['id']
        print(f"Checking pod: {pod_id}")
        print(f"Status: {latest_pod.get('desiredStatus')}")
        print(f"Runtime: {latest_pod.get('runtime')}")

        # Try to get logs
        try:
            logs = runpod.get_pod_logs(pod_id)
            print(f"\nPod logs:\n{logs}")
        except Exception as e:
            print(f"Could not get logs: {e}")
    else:
        print("No pods found")


if __name__ == "__main__":
    main()
