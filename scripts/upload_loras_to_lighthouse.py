#!/usr/bin/env python3
"""Upload trained LoRAs from GCS to Lighthouse for IPFS gateway testing."""

import asyncio
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from lighthouseweb3 import Lighthouse
import subprocess


async def main():
    api_key = os.getenv("LIGHTHOUSE_API_KEY")
    if not api_key:
        print("ERROR: LIGHTHOUSE_API_KEY not set!")
        return

    print(f"Lighthouse API key: {api_key[:10]}...")

    # Initialize Lighthouse client
    lh = Lighthouse(token=api_key)

    # GCS paths to trained LoRAs
    loras = [
        ("Leah", "gs://kestrel-training/training/379f40b3-a808-41a1-bb83-29202dd09f00/20251228_152032/output/379f40b3-a808-41a1-bb83-29202dd09f00/pytorch_lora_weights.safetensors", "TOK379f40b3"),
        ("Stephanie", "gs://kestrel-training/training/865ffca5-7a9c-41fa-aa5c-0f805609cd96/20251227_200509/output/865ffca5-7a9c-41fa-aa5c-0f805609cd96/pytorch_lora_weights.safetensors", "TOK865ffca5"),
        ("Lila", "gs://kestrel-training/training/a05d41a4-965d-4822-853c-a88b0ab8f32d/20251228_191632/output/a05d41a4-965d-4822-853c-a88b0ab8f32d/pytorch_lora_weights.safetensors", "TOKa05d41a4"),
        ("Maria", "gs://kestrel-training/training/dbcabc51-8c46-4cc8-bc2e-713241e82b6d/20251227_190700/output/dbcabc51-8c46-4cc8-bc2e-713241e82b6d/pytorch_lora_weights.safetensors", "TOKdbcabc51"),
    ]

    results = []

    for name, gcs_path, trigger in loras:
        print(f"\n{'='*60}")
        print(f"Uploading {name} ({trigger})")
        print(f"  GCS: {gcs_path}")

        # Download from GCS to temp file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Download from GCS
            print(f"  Downloading from GCS...")
            result = subprocess.run(
                ["gsutil", "cp", gcs_path, tmp_path],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"  ERROR: gsutil failed: {result.stderr}")
                continue

            # Get file size
            size = os.path.getsize(tmp_path)
            print(f"  Downloaded: {size / 1024 / 1024:.2f} MB")

            # Upload to Lighthouse
            print(f"  Uploading to Lighthouse...")
            upload_response = lh.upload(source=tmp_path, tag=f"kestrel-lora-{name.lower()}")

            # Parse response
            if isinstance(upload_response, dict) and "data" in upload_response:
                cid = upload_response["data"].get("Hash")
            else:
                cid = str(upload_response)

            print(f"  ✅ Uploaded! CID: {cid}")

            # Verify it's accessible
            gateway_url = f"https://files.lighthouse.storage/viewFile/{cid}"
            print(f"  Gateway URL: {gateway_url}")

            results.append((name, cid, trigger))

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    print(f"\n{'='*60}")
    print("SUMMARY - Use these CIDs for IPFS generation:")
    print(f"{'='*60}")
    for name, cid, trigger in results:
        print(f"  {name}: CID={cid}, trigger={trigger}")

    print("\nPython dict format:")
    print("companions = [")
    for name, cid, trigger in results:
        print(f'    ("{name}", "{cid}", "{trigger}"),')
    print("]")


if __name__ == "__main__":
    asyncio.run(main())
