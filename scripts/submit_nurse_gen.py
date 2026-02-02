#!/usr/bin/env python3
"""Submit generation jobs to Vertex AI using GCS LoRAs."""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load .env file
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

    # Companions with GCS paths to their trained LoRAs
    companions = [
        ("Leah", "gs://kestrel-training/training/379f40b3-a808-41a1-bb83-29202dd09f00/20251228_152032/output/379f40b3-a808-41a1-bb83-29202dd09f00/pytorch_lora_weights.safetensors", "TOK379f40b3"),
        ("Stephanie", "gs://kestrel-training/training/865ffca5-7a9c-41fa-aa5c-0f805609cd96/20251227_200509/output/865ffca5-7a9c-41fa-aa5c-0f805609cd96/pytorch_lora_weights.safetensors", "TOK865ffca5"),
        ("Lila", "gs://kestrel-training/training/a05d41a4-965d-4822-853c-a88b0ab8f32d/20251228_191632/output/a05d41a4-965d-4822-853c-a88b0ab8f32d/pytorch_lora_weights.safetensors", "TOKa05d41a4"),
        ("Maria", "gs://kestrel-training/training/dbcabc51-8c46-4cc8-bc2e-713241e82b6d/20251227_190700/output/dbcabc51-8c46-4cc8-bc2e-713241e82b6d/pytorch_lora_weights.safetensors", "TOKdbcabc51"),
    ]

    jobs = []
    for name, gcs_path, trigger in companions:
        print(f"Submitting generation job for {name}...")
        prompt = f"professional portrait of {trigger}, friendly nurse in scrubs, warm smile, healthcare setting"

        job = await manager.submit_generation_job(
            prompt=prompt,
            trigger_word=trigger,
            output_gcs_prefix=f"gs://kestrel-training/generation/{name.lower()}_selfie_gcs",
            lora_gcs_path=gcs_path,
            # Use latest for GCS jobs (ipfs-v1 is for IPFS)
            image_tag="latest",
        )
        jobs.append((name, job))
        print(f"  Job ID: {job['job_id']}")

    print()
    print("All jobs submitted:")
    for name, job in jobs:
        print(f"  {name}: {job['job_id']}")


if __name__ == "__main__":
    asyncio.run(main())
