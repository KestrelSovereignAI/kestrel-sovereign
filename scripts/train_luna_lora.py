#!/usr/bin/env python3
"""
Train Luna's LoRA on the persistent RunPod pod.

This script:
1. Fetches Luna's avatar from PostgreSQL
2. Submits training to the SimpleTuner FLUX.2 API on RunPod
3. Polls for completion
4. Stores the trained LoRA weights

Usage:
    set -a && source .env && set +a && uv run python scripts/train_luna_lora.py
"""

import asyncio
import base64
import json
import logging
import os
import sys
from datetime import datetime

import asyncpg
import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Luna's companion ID
LUNA_COMPANION_ID = "d6822ae1-f12c-40d9-8487-a7c4bdc93c35"

# Pod ID from runpod_config.toml
POD_ID = "48csncdfmnoniv"
BASE_URL = f"https://{POD_ID}-8000.proxy.runpod.net"


async def get_db_pool():
    """Create PostgreSQL connection pool."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL not set")
    return await asyncpg.create_pool(database_url)


async def fetch_luna_avatar(pool) -> bytes:
    """Fetch Luna's avatar from PostgreSQL."""
    logger.info(f"Fetching avatar for companion {LUNA_COMPANION_ID}")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT name, avatar_data FROM companions WHERE id = $1",
            LUNA_COMPANION_ID
        )
        if not row:
            raise RuntimeError(f"Companion {LUNA_COMPANION_ID} not found")

        name = row["name"]
        avatar_data = row["avatar_data"]

        if not avatar_data:
            raise RuntimeError(f"Companion {name} has no avatar_data")

        logger.info(f"Found companion: {name}")
        logger.info(f"Avatar size: {len(avatar_data)} bytes")

        return bytes(avatar_data)


# Training configuration - FAST MODE for iteration
TRAINING_STEPS = 100  # 100 steps = ~2-3 min, 1000 steps = ~20 min
LORA_RANK = 8  # Lower rank = faster training, slightly lower quality


async def submit_training(avatar_data: bytes) -> str:
    """Submit training job to SimpleTuner API."""
    train_url = f"{BASE_URL}/train"
    logger.info(f"Submitting training to {train_url}")
    logger.info(f"  Steps: {TRAINING_STEPS} (fast mode)")
    logger.info(f"  LoRA Rank: {LORA_RANK}")

    # Detect content type
    if avatar_data[:8] == b'\x89PNG\r\n\x1a\n':
        content_type = "image/png"
        filename = "avatar.png"
    else:
        content_type = "image/jpeg"
        filename = "avatar.jpg"

    logger.info(f"Image format: {content_type}")

    # Submit as multipart form with fast training settings
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {"image": (filename, avatar_data, content_type)}
        data = {
            "companion_id": LUNA_COMPANION_ID,
            "steps": str(TRAINING_STEPS),
            "lora_rank": str(LORA_RANK),
        }

        response = await client.post(train_url, files=files, data=data)

        if response.status_code != 200:
            logger.error(f"Training submission failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            raise RuntimeError(f"Training submission failed: {response.text}")

        result = response.json()
        job_id = result.get("job_id")
        if not job_id:
            raise RuntimeError(f"No job_id in response: {result}")

        logger.info(f"Training started: {job_id}")
        return job_id


async def poll_training_status(job_id: str) -> dict:
    """Poll training status until complete."""
    status_url = f"{BASE_URL}/status/{job_id}"
    logger.info(f"Polling status at {status_url}")

    last_progress = -1
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                response = await client.get(status_url)
                if response.status_code != 200:
                    logger.warning(f"Status check returned {response.status_code}")
                    await asyncio.sleep(30)
                    continue

                status = response.json()
                current_status = status.get("status", "unknown")
                progress = status.get("progress", 0)

                # Log progress updates
                progress_pct = int(progress * 100)
                if progress_pct != last_progress:
                    logger.info(f"Training progress: {progress_pct}% - {current_status}")
                    last_progress = progress_pct

                if current_status == "completed":
                    logger.info("Training completed!")
                    return status
                elif current_status == "failed":
                    error = status.get("error", "Unknown error")
                    logger.error(f"Training failed: {error}")
                    raise RuntimeError(f"Training failed: {error}")

                await asyncio.sleep(30)  # Poll every 30 seconds

            except httpx.TimeoutException:
                logger.warning("Status check timed out, retrying...")
                await asyncio.sleep(30)


async def download_lora(job_id: str) -> bytes:
    """Download trained LoRA weights."""
    download_url = f"{BASE_URL}/download/{job_id}"
    logger.info(f"Downloading LoRA from {download_url}")

    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 min timeout for large files
        response = await client.get(download_url)
        if response.status_code != 200:
            raise RuntimeError(f"Download failed: {response.status_code}")

        lora_data = response.content
        logger.info(f"Downloaded LoRA: {len(lora_data)} bytes")
        return lora_data


async def store_lora(pool, companion_id: str, lora_data: bytes, trigger_word: str):
    """Store LoRA weights in PostgreSQL."""
    import hashlib

    content_hash = hashlib.sha256(lora_data).hexdigest()
    metadata = {
        "trigger_word": trigger_word,
        "trainer": "simpletuner-flux2",
        "trained_at": datetime.utcnow().isoformat()
    }

    async with pool.acquire() as conn:
        # Store in companion_files
        await conn.execute("""
            INSERT INTO companion_files
            (companion_id, file_type, content_hash, file_data, file_size, metadata)
            VALUES ($1, 'lora_weights', $2, $3, $4, $5::jsonb)
            ON CONFLICT (companion_id, content_hash) DO UPDATE
            SET metadata = $5::jsonb, created_at = NOW()
        """, companion_id, content_hash, lora_data, len(lora_data),
            json.dumps(metadata)
        )

        # Update companion avatar_config
        row = await conn.fetchrow(
            "SELECT avatar_config FROM companions WHERE id = $1",
            companion_id
        )
        if row:
            raw_config = row["avatar_config"]
            if raw_config is None:
                avatar_config = {}
            elif isinstance(raw_config, str):
                try:
                    avatar_config = json.loads(raw_config)
                except json.JSONDecodeError:
                    avatar_config = {}
            else:
                avatar_config = raw_config

            avatar_config.update({
                "lora_training_status": "completed",
                "lora_content_hash": content_hash,
                "lora_trigger_word": trigger_word,
                "lora_trained_at": datetime.utcnow().isoformat()
            })

            await conn.execute(
                "UPDATE companions SET avatar_config = $1 WHERE id = $2",
                json.dumps(avatar_config),
                companion_id
            )

    logger.info(f"Stored LoRA with hash: {content_hash}")
    return content_hash


async def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Luna LoRA Training")
    logger.info("="*60)
    logger.info(f"Companion ID: {LUNA_COMPANION_ID}")
    logger.info(f"Pod ID: {POD_ID}")
    logger.info(f"Base URL: {BASE_URL}")
    logger.info("")

    # Check pod health first
    logger.info("Checking pod health...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code != 200:
                logger.error(f"Pod not healthy: {response.status_code}")
                sys.exit(1)
            health = response.json()
            logger.info(f"Pod health: {health}")
        except Exception as e:
            logger.error(f"Cannot reach pod: {e}")
            sys.exit(1)

    # Connect to PostgreSQL
    logger.info("Connecting to PostgreSQL...")
    pool = await get_db_pool()

    try:
        # Fetch Luna's avatar
        avatar_data = await fetch_luna_avatar(pool)

        # Submit training
        job_id = await submit_training(avatar_data)

        # Poll for completion
        status = await poll_training_status(job_id)

        # Generate trigger word
        trigger_word = f"TOK{LUNA_COMPANION_ID[:8]}"

        # Download trained LoRA
        lora_data = await download_lora(job_id)

        # Store in PostgreSQL
        content_hash = await store_lora(pool, LUNA_COMPANION_ID, lora_data, trigger_word)

        logger.info("")
        logger.info("="*60)
        logger.info("Training Complete!")
        logger.info(f"  Content Hash: {content_hash}")
        logger.info(f"  Trigger Word: {trigger_word}")
        logger.info(f"  LoRA Size: {len(lora_data)} bytes")
        logger.info("="*60)

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
