#!/usr/bin/env python3
"""
Frinz LoRA Batch Pipeline - Pre-render avatars and LoRA models locally.

Queries Frinz PostgreSQL for companions that need LoRA training or
selfie generation, then processes them locally on MPS (free, sovereign).

Usage:
    python scripts/frinz_lora_batch.py                    # Train + generate for all companions
    python scripts/frinz_lora_batch.py --train-only        # Only train LoRA models
    python scripts/frinz_lora_batch.py --generate-only     # Only generate selfies (requires existing LoRA)
    python scripts/frinz_lora_batch.py --companion UUID    # Process specific companion
    python scripts/frinz_lora_batch.py --dry-run           # List companions that need work

Environment Variables:
    DATABASE_URL - Frinz PostgreSQL URL (default: from .env)
    LOCAL_MPS_MODEL_PATH - Path to SDXL model
    DIFFUSERS_PATH - Path to diffusers installation
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")
# Also load Frinz .env for DATABASE_URL
frinz_env = Path("/Volumes/data2/projects/frinz/.env")
if frinz_env.exists():
    load_dotenv(frinz_env, override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("frinz_lora_batch")

# Standard scenes to pre-generate for each companion
STANDARD_SCENES = [
    ("casual", "casual candid photo, relaxed pose, natural lighting, warm smile"),
    ("romantic", "romantic portrait, soft lighting, warm gaze, dreamy atmosphere"),
    ("playful", "playful fun photo, laughing, bright colors, joyful energy"),
    ("professional", "professional headshot, studio lighting, confident pose"),
    ("cozy", "cozy indoor photo, soft sweater, warm blanket, relaxed evening"),
    ("dreamy", "dreamy ethereal portrait, golden hour, bokeh background"),
]


async def get_db_pool():
    """Get asyncpg connection pool."""
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg not installed. Run: pip install asyncpg")
        sys.exit(1)

    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)

    return await asyncpg.create_pool(database_url, min_size=1, max_size=3)


async def get_companions_needing_lora(pool) -> list[dict]:
    """Find companions with avatars but no trained LoRA."""
    async with pool.acquire() as conn:
        # Get companions that have avatar files but no LoRA weights
        rows = await conn.fetch("""
            SELECT DISTINCT c.id, c.name, c.personality_config
            FROM companions c
            JOIN companion_files cf_avatar ON cf_avatar.companion_id = c.id
                AND cf_avatar.file_type = 'avatar'
            WHERE NOT EXISTS (
                SELECT 1 FROM companion_files cf_lora
                WHERE cf_lora.companion_id = c.id
                AND cf_lora.file_type = 'lora_weights'
            )
            ORDER BY c.name
        """)
        return [dict(r) for r in rows]


async def get_companions_needing_selfies(pool, scenes: list[str]) -> list[dict]:
    """Find companions with LoRA but missing scene selfies."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT DISTINCT c.id, c.name
            FROM companions c
            JOIN companion_files cf_lora ON cf_lora.companion_id = c.id
                AND cf_lora.file_type = 'lora_weights'
            ORDER BY c.name
        """)
        result = []
        for row in rows:
            # Check which scenes are missing
            existing = await conn.fetch("""
                SELECT metadata->>'scene' as scene
                FROM companion_files
                WHERE companion_id = $1 AND file_type = 'selfie'
            """, row["id"])
            existing_scenes = {r["scene"] for r in existing if r["scene"]}
            missing = [s for s in scenes if s not in existing_scenes]
            if missing:
                result.append({**dict(row), "missing_scenes": missing})
        return result


async def get_avatar_data(pool, companion_id) -> Optional[bytes]:
    """Get the latest avatar image for a companion."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT file_data FROM companion_files
            WHERE companion_id = $1 AND file_type = 'avatar'
            ORDER BY created_at DESC LIMIT 1
        """, companion_id)
        return row["file_data"] if row else None


async def get_lora_path(pool, companion_id) -> Optional[str]:
    """Get the local LoRA weights path for a companion (save to disk if needed)."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT file_data, metadata FROM companion_files
            WHERE companion_id = $1 AND file_type = 'lora_weights'
            ORDER BY created_at DESC LIMIT 1
        """, companion_id)
        if not row:
            return None

        # Save to local path for inference
        lora_dir = Path(f"/Volumes/data2/models/local-training/lora-weights/{companion_id}")
        lora_dir.mkdir(parents=True, exist_ok=True)
        lora_path = lora_dir / "pytorch_lora_weights.safetensors"
        lora_path.write_bytes(row["file_data"])
        return str(lora_path)


async def store_lora_weights(pool, companion_id: str, weights_bytes: bytes, trigger_word: str):
    """Store trained LoRA weights in Frinz database."""
    content_hash = hashlib.sha256(weights_bytes).hexdigest()
    metadata = json.dumps({
        "trigger_word": trigger_word,
        "provider": "local_mps",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    })

    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO companion_files (companion_id, file_type, content_hash, file_data, metadata)
            VALUES ($1, 'lora_weights', $2, $3, $4::jsonb)
        """, companion_id, content_hash, weights_bytes, metadata)

    logger.info(f"Stored LoRA weights ({len(weights_bytes)} bytes) for {companion_id[:8]}")


async def store_selfie(pool, companion_id: str, image_bytes: bytes, scene: str):
    """Store generated selfie in Frinz database."""
    content_hash = hashlib.sha256(image_bytes).hexdigest()
    metadata = json.dumps({
        "scene": scene,
        "provider": "local_mps",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })

    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO companion_files (companion_id, file_type, content_hash, file_data, metadata)
            VALUES ($1, 'selfie', $2, $3, $4::jsonb)
        """, companion_id, content_hash, image_bytes, metadata)

    logger.info(f"Stored selfie ({scene}) for {companion_id[:8]}")


def check_memory_pressure() -> str:
    """Check macOS memory pressure."""
    import subprocess
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            level = int(result.stdout.strip())
            if level >= 4:
                return "red"
            elif level >= 2:
                return "yellow"
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return "green"


async def train_companion_lora(adapter, pool, companion: dict) -> bool:
    """Train LoRA for a single companion. Returns True on success."""
    companion_id = str(companion["id"])
    name = companion["name"]
    logger.info(f"Training LoRA for {name} ({companion_id[:8]})")

    avatar_data = await get_avatar_data(pool, companion["id"])
    if not avatar_data:
        logger.warning(f"No avatar found for {name}, skipping")
        return False

    from kestrel_sovereign.features.training.types import TrainingConfig, TrainingState

    config = TrainingConfig(
        steps=500,
        learning_rate=1e-4,
        lora_rank=128,
        resolution="512",
        trigger_word=f"TOK{companion_id[:8]}",
    )

    try:
        job = await adapter.start_training(companion_id, avatar_data, config)
        logger.info(f"Training started: job={job.job_id[:8]}, trigger={config.trigger_word}")

        # Poll until complete
        while True:
            await asyncio.sleep(30)

            # Check memory pressure
            pressure = check_memory_pressure()
            if pressure == "red":
                logger.warning("Memory pressure RED — cancelling training")
                await adapter.cancel(job.job_id)
                return False

            status = await adapter.get_status(job.job_id)
            logger.info(f"Training {name}: {status.state.value} ({status.progress:.0%})")

            if status.state == TrainingState.COMPLETED:
                weights = await adapter.download_weights(job.job_id)
                if weights:
                    await store_lora_weights(pool, companion_id, weights, config.trigger_word)
                    await adapter.cleanup(job.job_id)
                    return True
                else:
                    logger.error(f"No weights produced for {name}")
                    await adapter.cleanup(job.job_id)
                    return False

            elif status.state == TrainingState.FAILED:
                logger.error(f"Training failed for {name}: {status.error}")
                await adapter.cleanup(job.job_id)
                return False

    except Exception as e:
        logger.error(f"Error training {name}: {e}")
        return False


async def generate_companion_selfies(adapter, pool, companion: dict, scenes: list[tuple[str, str]]) -> int:
    """Generate selfies for a companion. Returns count of successful generations."""
    companion_id = str(companion["id"])
    name = companion["name"]
    trigger_word = f"TOK{companion_id[:8]}"
    success_count = 0

    lora_path = await get_lora_path(pool, companion["id"])
    if not lora_path:
        logger.warning(f"No LoRA weights for {name}, skipping generation")
        return 0

    from kestrel_sovereign.features.training.types import GenerationConfig, GenerationState

    missing_scenes = companion.get("missing_scenes", [s[0] for s in scenes])

    for scene_name, scene_prompt in scenes:
        if scene_name not in missing_scenes:
            continue

        # Check memory pressure
        if check_memory_pressure() != "green":
            logger.warning("Memory pressure — stopping generation")
            break

        full_prompt = f"{trigger_word} {scene_prompt}"
        logger.info(f"Generating {scene_name} selfie for {name}")

        config = GenerationConfig(
            prompt=full_prompt,
            lora_path=lora_path,
            width=1024,
            height=1024,
            num_inference_steps=30,
            guidance_scale=7.5,
        )

        result = await adapter.generate_image(config)

        if result.state == GenerationState.COMPLETED and result.images:
            # Decode base64 data URL
            import base64
            data_url = result.images[0]
            if data_url.startswith("data:"):
                b64_data = data_url.split(",", 1)[1]
            else:
                b64_data = data_url
            image_bytes = base64.b64decode(b64_data)

            await store_selfie(pool, companion_id, image_bytes, scene_name)
            success_count += 1
            logger.info(f"Generated {scene_name} for {name} ({result.elapsed_seconds:.1f}s)")
        else:
            logger.error(f"Failed {scene_name} for {name}: {result.error}")

        # Brief pause between generations
        await asyncio.sleep(5)

    return success_count


async def run_batch(args):
    """Main batch processing loop."""
    pool = await get_db_pool()

    try:
        # Initialize adapter
        from kestrel_sovereign.features.training.adapters.local_mps_adapter import LocalMPSTrainingAdapter

        adapter = LocalMPSTrainingAdapter()
        if not adapter.is_available():
            logger.error(
                "Local MPS adapter not available. Check:\n"
                "  LOCAL_MPS_MODEL_PATH - SDXL model in diffusers format\n"
                "  DIFFUSERS_PATH - diffusers installation with training scripts"
            )
            return

        # Determine what needs doing
        if args.train_only or not args.generate_only:
            if args.companion:
                companions_for_training = [{"id": args.companion, "name": "specified"}]
            else:
                companions_for_training = await get_companions_needing_lora(pool)
        else:
            companions_for_training = []

        if args.generate_only or not args.train_only:
            scene_names = [s[0] for s in STANDARD_SCENES]
            if args.companion:
                companions_for_selfies = [
                    {"id": args.companion, "name": "specified", "missing_scenes": scene_names}
                ]
            else:
                companions_for_selfies = await get_companions_needing_selfies(pool, scene_names)
        else:
            companions_for_selfies = []

        logger.info(
            f"Batch plan: {len(companions_for_training)} to train, "
            f"{len(companions_for_selfies)} need selfies"
        )

        if args.dry_run:
            print("\nCompanions needing LoRA training:")
            for c in companions_for_training:
                print(f"  {c['name']} ({str(c['id'])[:8]})")
            print(f"\nCompanions needing selfies:")
            for c in companions_for_selfies:
                missing = c.get("missing_scenes", [])
                print(f"  {c['name']} ({str(c['id'])[:8]}): {', '.join(missing)}")
            return

        # Phase 1: Train LoRA models
        trained = 0
        failed_training = 0
        for companion in companions_for_training:
            if check_memory_pressure() != "green":
                logger.warning("Memory pressure — pausing training batch")
                break
            success = await train_companion_lora(adapter, pool, companion)
            if success:
                trained += 1
            else:
                failed_training += 1

        # Phase 2: Generate selfies
        generated = 0
        for companion in companions_for_selfies:
            if check_memory_pressure() != "green":
                logger.warning("Memory pressure — pausing generation batch")
                break
            count = await generate_companion_selfies(adapter, pool, companion, STANDARD_SCENES)
            generated += count

        logger.info(
            f"Batch complete: {trained} trained, {failed_training} failed, "
            f"{generated} selfies generated"
        )

        await adapter.close()

    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description="Frinz LoRA Batch Pipeline")
    parser.add_argument("--train-only", action="store_true", help="Only train LoRA models")
    parser.add_argument("--generate-only", action="store_true", help="Only generate selfies")
    parser.add_argument("--companion", type=str, help="Process specific companion UUID")
    parser.add_argument("--dry-run", action="store_true", help="List work without processing")
    args = parser.parse_args()

    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
