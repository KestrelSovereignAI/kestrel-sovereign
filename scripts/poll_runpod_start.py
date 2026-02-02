#!/usr/bin/env python3
"""
Poll RunPod every 5 minutes to try starting a pod.
Stops when the pod starts successfully.

Usage:
    python scripts/poll_runpod_start.py

    # Or with custom interval
    python scripts/poll_runpod_start.py --interval 300 --pod-id l6bcivl0w96gq5
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Force unbuffered output for background execution
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_env():
    """Load environment from .env file."""
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

def main():
    parser = argparse.ArgumentParser(description="Poll RunPod to start a pod")
    parser.add_argument("--pod-id", default="l6bcivl0w96gq5", help="Pod ID to start")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds (default: 300 = 5 min)")
    parser.add_argument("--max-attempts", type=int, default=48, help="Max attempts before giving up (default: 48 = 4 hours)")
    args = parser.parse_args()

    load_env()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not found in environment")
        sys.exit(1)

    import runpod
    runpod.api_key = api_key

    pod_id = args.pod_id
    interval = args.interval
    max_attempts = args.max_attempts

    print(f"🔄 Polling RunPod to start pod {pod_id}")
    print(f"   Interval: {interval} seconds ({interval // 60} minutes)")
    print(f"   Max attempts: {max_attempts} (will give up after {max_attempts * interval // 3600:.1f} hours)")
    print(f"   Press Ctrl+C to stop\n")

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        now = datetime.now().strftime("%H:%M:%S")

        try:
            # Check current pod status first
            pod = runpod.get_pod(pod_id)
            status = pod.get("desiredStatus", "UNKNOWN")

            if status == "RUNNING":
                print(f"\n✅ [{now}] Pod is already running!")
                print(f"   GPU: {pod.get('machine', {}).get('gpuDisplayName', 'unknown')}")
                print(f"   URL: https://{pod_id}-8000.proxy.runpod.net")
                sys.exit(0)

            print(f"[{now}] Attempt {attempt}/{max_attempts}: Status={status}, trying to start...")

            result = runpod.resume_pod(pod_id, gpu_count=1)

            if result.get("desiredStatus") == "RUNNING":
                print(f"\n✅ [{now}] Pod started successfully!")
                print(f"   Result: {result}")
                print(f"   URL: https://{pod_id}-8000.proxy.runpod.net")
                sys.exit(0)
            else:
                print(f"   Unexpected result: {result}")

        except Exception as e:
            error_msg = str(e)
            if "not enough free GPUs" in error_msg:
                print(f"   No GPUs available, waiting {interval}s...")
            else:
                print(f"   Error: {error_msg}")

        if attempt < max_attempts:
            time.sleep(interval)

    print(f"\n❌ Gave up after {max_attempts} attempts")
    print(f"   Try starting manually: https://www.runpod.io/console/pods")
    sys.exit(1)

if __name__ == "__main__":
    main()
