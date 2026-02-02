#!/usr/bin/env python3
"""
Check RunPod pods status.

Lists all pods with their status and running costs.
This is a utility script, not a test - run directly with python.

Usage:
    python check_pods.py
"""
import os


def main():
    """Check and display all RunPod pods."""
    import runpod
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("Error: RUNPOD_API_KEY not set")
        return

    runpod.api_key = api_key

    try:
        pods = runpod.get_pods()
        print('Current pods:')
        running_cost = 0
        for pod in pods:
            status = pod.get('desiredStatus', 'Unknown')
            name = pod.get('name', 'Unnamed')
            cost_per_hr = pod.get('costPerHr', 0)
            print(f'  {pod["id"]}: {status} - {name}')
            if status == 'RUNNING':
                gpu = pod.get('machine', {}).get('gpuDisplayName', 'Unknown')
                print(f'    GPU: {gpu}, Cost/hr: ${cost_per_hr}')
                running_cost += cost_per_hr
        if running_cost > 0:
            print(f'\nRunning pods costing ${running_cost}/hour')
            print('Stop them with: runpod.stop_pod(pod_id)')
        elif not pods:
            print('  (no pods)')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    main()
