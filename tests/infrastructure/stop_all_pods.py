#!/usr/bin/env python3
"""
Stop all running RunPod pods.

Emergency cleanup script to stop all running pods and prevent cost accumulation.
This is a utility script, not a test - run directly with python.

Usage:
    python stop_all_pods.py
"""
import os


def main():
    """Stop all running RunPod pods."""
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
        print('Stopping all running pods...')
        stopped_count = 0
        for pod in pods:
            status = pod.get('desiredStatus', 'Unknown')
            if status == 'RUNNING':
                pod_id = pod['id']
                name = pod.get('name', 'Unnamed')
                print(f'Stopping {pod_id} ({name})...')
                try:
                    runpod.stop_pod(pod_id)
                    print(f'Stopped {pod_id}')
                    stopped_count += 1
                except Exception as e:
                    print(f'Failed to stop {pod_id}: {e}')

        if stopped_count > 0:
            print(f'\nStopped {stopped_count} pods')
        else:
            print('\nNo running pods to stop')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    main()
