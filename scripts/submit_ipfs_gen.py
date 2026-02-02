#!/usr/bin/env python3
"""Submit generation jobs using IPFS CIDs to Vertex AI."""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from kestrel_sovereign.features.vertex_ai.vertex_ai_manager import VertexAIManager


async def main():
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        print(f"HF_TOKEN set: {hf_token[:10]}...")
    else:
        print("ERROR: HF_TOKEN not set!")
        return

    manager = VertexAIManager()
    print(f"Manager HF_TOKEN: {manager.hf_token[:10] if manager.hf_token else 'NOT SET'}...")

    # Companions with verified IPFS CIDs (uploaded via Lighthouse)
    companions = [
        ("Leah", "QmSG1uakkjNdhSJRfYBmGsdSpHwrRbBbYW2vD1F2psrzuZ", "TOK379f40b3"),
        ("Stephanie", "QmSoqtQPqnEsioqZcQFAL54QZ6MvzPTo5K3bEMANqMzmRD", "TOK865ffca5"),
        ("Lila", "QmbS71VE1bnDoMakmMzgQUxDNndVJPurG66sW3mBVKPvcY", "TOKa05d41a4"),
        ("Maria", "QmV1iuMJq5awUXt45obTmPuGGSNZZrTR39hrS4taC6BNKP", "TOKdbcabc51"),
    ]

    jobs = []
    for name, cid, trigger in companions:
        print(f"Submitting IPFS generation job for {name}...")
        prompt = f"professional portrait of {trigger}, friendly nurse in scrubs, warm smile, healthcare setting"

        job = await manager.submit_generation_job(
            prompt=prompt,
            trigger_word=trigger,
            output_gcs_prefix=f"gs://kestrel-training/generation/{name.lower()}_ipfs_test",
            lora_ipfs_cid=cid,
            # Uses default Lighthouse gateway (gateway.lighthouse.storage/ipfs)
            image_tag="ipfs-v1",
        )
        jobs.append((name, job))
        print(f"  Job ID: {job['job_id']}")

    print()
    print("All jobs submitted:")
    for name, job in jobs:
        print(f"  {name}: {job['job_id']}")

    print()
    print("Monitor with:")
    print("  gcloud ai custom-jobs describe <job_id> --region=us-central1 --project=YOUR_PROJECT_ID")


if __name__ == "__main__":
    asyncio.run(main())
