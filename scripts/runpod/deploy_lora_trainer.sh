#!/bin/bash
# Deploy Kestrel LoRA Trainer to RunPod
# Uses RunPodManager from features/runpod/ with template_id for GCR auth
#
# Usage:
#   ./scripts/runpod/deploy_lora_trainer.sh [--test] [--status] [--stop] [--profile=NAME]
#
# Options:
#   --test           Run health check after deployment
#   --status         Show status of any running lora trainer pods
#   --stop           Stop the current training pod (but keep it for resume)
#   --kill           Terminate the pod completely
#   --profile=NAME   Use specific profile (training, training-4090). Default: training
#
# Profiles:
#   training       RTX 3090 ($0.22/hr) - default, best value
#   training-4090  RTX 4090 ($0.44/hr) - use when RTX 3090 unavailable
#
# Requirements:
#   - RUNPOD_API_KEY environment variable
#   - runpod_config.toml with training profile and template_id

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Kestrel LoRA Trainer - RunPod Deployment"
echo ""

# Check for required env var
if [ -z "$RUNPOD_API_KEY" ]; then
    echo -e "${RED}Error: RUNPOD_API_KEY environment variable is required${NC}"
    exit 1
fi

# Parse arguments
ACTION="deploy"
TEST_AFTER=false
PROFILE="training"

for arg in "$@"; do
    case $arg in
        --test)
            TEST_AFTER=true
            ;;
        --status)
            ACTION="status"
            ;;
        --stop)
            ACTION="stop"
            ;;
        --kill)
            ACTION="kill"
            ;;
        --profile=*)
            PROFILE="${arg#*=}"
            ;;
    esac
done

echo "📦 Profile: $PROFILE"

# Python script that uses RunPodManager
run_python() {
    uv run python3 << 'PYTHON_SCRIPT'
import asyncio
import sys
import os
sys.path.insert(0, ".")

from features.runpod.runpod_manager import RunPodManager

async def main():
    action = os.environ.get("RUNPOD_ACTION", "deploy")
    test_after = os.environ.get("TEST_AFTER", "false") == "true"

    manager = RunPodManager()

    # Get training profile info
    profile_name = os.environ.get("RUNPOD_PROFILE", "training")
    profile = manager.profiles.get(profile_name)
    if not profile:
        print(f"❌ Error: No '{profile_name}' profile found in runpod_config.toml")
        print(f"   Available profiles: {list(manager.profiles.keys())}")
        sys.exit(1)

    print("📋 Training Profile:")
    print(f"   GPU: {profile.gpu_type_id}")
    print(f"   Image: {profile.image_name}")
    print(f"   Template: {profile.template_id or 'None (public image)'}")
    print(f"   Network Volume: {profile.network_volume_id or 'None (ephemeral)'}")
    print(f"   Port: {profile.inference_port}")
    print("")

    if action == "status":
        print("🔍 Checking pod status...")
        status = await manager.get_status()
        if status.get("active"):
            print(f"✅ Pod is ACTIVE")
            print(f"   Pod ID: {status.get('pod_id')}")
            print(f"   Status: {status.get('status')}")
            print(f"   Base URL: {status.get('base_url')}")
            print(f"   Expires: {status.get('expires_at')}")
        else:
            print("⚫ No active pod")
            # Check for stopped pods that can be resumed
            stopped = await manager.find_stopped_pod("lora_training", "training")
            if stopped:
                print(f"💤 Found stopped pod {stopped['id']} - can resume")
        return

    if action == "stop":
        print("⏹️  Stopping pod (keeping for resume)...")
        result = await manager.stop_session()
        print(f"   Result: {result}")
        return

    if action == "kill":
        print("💀 Terminating pod completely...")
        status = await manager.get_status()
        if status.get("pod_id"):
            await asyncio.to_thread(manager.provider.terminate_pod, status["pod_id"])
            print("   Terminated")
        else:
            print("   No active pod to terminate")
        return

    # Deploy action
    print("🚀 Starting training pod...")
    print("   (Using template_id for GCR private registry auth)")
    print("")

    try:
        result = await manager.start_session(
            task_profile=profile_name,
            model_name="FLUX.1-dev",
            ttl_seconds=3600,
            metadata={
                "name": "kestrel-lora-trainer",
                "purpose": "lora_training"
            }
        )

        print(f"✅ Pod created successfully!")
        print(f"   Pod ID: {result.get('pod_id')}")
        print(f"   Status: {result.get('status')}")
        print(f"   Base URL: {result.get('base_url')}")
        print(f"   TTL: {result.get('ttl_seconds')}s")
        print("")

        if test_after:
            import httpx
            base_url = result.get('base_url')
            if base_url:
                print("🧪 Testing endpoints...")
                try:
                    health = httpx.get(f"{base_url}/health", timeout=60)
                    print(f"   /health: {health.status_code}")
                    print(f"   {health.text[:200]}")
                except Exception as e:
                    print(f"   /health: {e}")

                try:
                    openapi = httpx.get(f"{base_url}/openapi.json", timeout=30)
                    if openapi.status_code == 200:
                        spec = openapi.json()
                        paths = list(spec.get("paths", {}).keys())
                        print(f"   Available endpoints: {paths}")
                except Exception as e:
                    print(f"   /openapi.json: {e}")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Deployment failed: {error_msg}")

        if "no longer any instances available" in error_msg.lower():
            print("\n📋 GPU Availability Issue:")
            print(f"   Profile '{profile_name}' GPU unavailable in US-KS-2 datacenter.")
            print("   (Network volume requires pods in same datacenter)")
            print("\n   Options:")
            print("   1. Wait and retry later")
            if profile_name == "training":
                print("   2. Try RTX 4090: ./scripts/runpod/deploy_lora_trainer.sh --profile=training-4090")
            else:
                print("   2. Try RTX 3090: ./scripts/runpod/deploy_lora_trainer.sh --profile=training")
            print("   3. Check RunPod dashboard for availability")

        sys.exit(1)

asyncio.run(main())
PYTHON_SCRIPT
}

# Export action for Python script
export RUNPOD_ACTION="$ACTION"
export RUNPOD_PROFILE="$PROFILE"
export TEST_AFTER="$TEST_AFTER"

run_python

echo ""
echo "Done!"
