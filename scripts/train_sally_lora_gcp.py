#!/usr/bin/env python3
"""
Train Sally's LoRA using GCP Compute A100 80GB.

Uses the new GCPComputeManager with the training-a100-80gb profile.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kestrel_sovereign.features.gcp_compute.gcp_compute_manager import GCPComputeManager

# Sally's companion ID and avatar path
SALLY_COMPANION_ID = "8f65aa8d-c574-456e-9953-40408dcac333"
SALLY_AVATAR_PATH = "/tmp/sally_avatar.jpg"  # Downloaded from API


async def main():
    # Check Sally's avatar exists
    if not os.path.exists(SALLY_AVATAR_PATH):
        print(f"ERROR: Sally's avatar not found at {SALLY_AVATAR_PATH}")
        return 1

    print(f"Sally's avatar: {SALLY_AVATAR_PATH}")
    print(f"File size: {os.path.getsize(SALLY_AVATAR_PATH)} bytes")

    # Initialize GCP Compute Manager
    manager = GCPComputeManager()
    print(f"GCP Project: {manager.project_id}")
    print(f"Default Zone: {manager.default_zone}")

    # List available profiles
    print("\nAvailable profiles:")
    for profile_id, profile in manager.profiles.items():
        print(f"  - {profile_id}: {profile.name} (${profile.cost_per_hr_spot}/hr spot)")

    # Use the A100 80GB profile
    profile_name = "training-a100-80gb"
    print(f"\nStarting session with profile: {profile_name}")

    try:
        # Start session
        session_info = await manager.start_session(
            task_profile=profile_name,
            ttl_seconds=3600,  # 1 hour max
            use_spot=True,  # Save money
            metadata={
                "companion_id": SALLY_COMPANION_ID,
                "purpose": "lora_training"
            }
        )

        print(f"\nSession started!")
        print(f"  Instance: {session_info.get('instance_name')}")
        print(f"  External IP: {session_info.get('external_ip')}")
        print(f"  Status: {session_info.get('status')}")
        print(f"  Cost: ${session_info.get('actual_cost_per_hr')}/hr")

        # Wait for instance to be ready
        print("\nWaiting for instance to be ready...")
        session = manager._session

        # Submit training job
        print("\nSubmitting LoRA training job...")
        training_result = await manager.submit_training_job(
            session=session,
            image_path=SALLY_AVATAR_PATH,
            companion_id=SALLY_COMPANION_ID,
            instance_prompt="a photo of sks woman",
            training_steps=500,
            learning_rate=1e-4,
            rank=16,
        )

        print(f"Training job submitted: {training_result}")

        # Poll for completion
        print("\nPolling for training completion...")
        while True:
            status = await manager.poll_training_status(session, training_result.get("job_id", ""))
            print(f"  Status: {status.get('status')} - Progress: {status.get('progress', 0)*100:.0f}%")

            if status.get("status") == "completed":
                print("\n✅ Training completed!")
                break
            elif status.get("status") == "failed":
                print(f"\n❌ Training failed: {status.get('error')}")
                break

            await asyncio.sleep(30)  # Poll every 30 seconds

        # Download LoRA if successful
        if status.get("status") == "completed":
            print("\nDownloading trained LoRA...")
            lora_path = await manager.download_lora(session, training_result.get("job_id", ""))
            print(f"LoRA saved to: {lora_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Terminate session
        print("\nTerminating session...")
        try:
            await manager.terminate_session(manager._session)
            print("Session terminated.")
        except Exception as e:
            print(f"Warning: Failed to terminate session: {e}")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
