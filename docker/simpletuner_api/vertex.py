"""
Vertex AI Batch Mode Functions.

Provides batch execution modes for Vertex AI Custom Jobs.
Handles training and generation in one-shot execution with GCS storage.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DOWNLOAD
from . import config as app_config
from .inference import get_inference_pipeline
from .training import create_simpletuner_config, run_training

logger = logging.getLogger(__name__)


async def run_vertex_batch_generation(args):
    """
    Run generation in batch mode for Vertex AI Custom Jobs.

    This is a one-shot execution that:
    1. Downloads LoRA from IPFS or GCS
    2. Loads FLUX.2-dev with int8-quanto
    3. Generates images
    4. Uploads results to GCS
    5. Exits with success (0) or failure (1)

    Args:
        args: Parsed command-line arguments with:
            - lora_gcs: gs://bucket/path/to/pytorch_lora_weights.safetensors (optional)
            - lora_ipfs: IPFS CID for LoRA weights (optional)
            - ipfs_gateway: IPFS gateway URL (default: https://gateway.lighthouse.storage/ipfs)
            - output_gcs: gs://bucket/path/for/output/
            - prompt: Generation prompt
            - trigger_word: LoRA trigger word
            - num_outputs: Number of images to generate
            - width, height: Image dimensions
    """
    from google.cloud import storage as gcs
    from io import BytesIO
    import requests

    logger.info("=" * 60)
    logger.info("Vertex AI Batch Generation Mode")
    logger.info("=" * 60)
    logger.info(f"LoRA IPFS: {args.lora_ipfs}")
    logger.info(f"LoRA GCS: {args.lora_gcs}")
    logger.info(f"IPFS Gateway: {args.ipfs_gateway}")
    logger.info(f"Output GCS: {args.output_gcs}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Trigger Word: {args.trigger_word}")
    logger.info(f"Num Outputs: {args.num_outputs}")
    logger.info(f"Dimensions: {args.width}x{args.height}")

    try:
        local_lora_path = "/tmp/lora/pytorch_lora_weights.safetensors"
        os.makedirs("/tmp/lora", exist_ok=True)

        # Step 1: Download LoRA from IPFS or GCS
        if args.lora_ipfs:
            # Download from IPFS via gateway
            logger.info(f"Downloading LoRA from IPFS: {args.lora_ipfs}")
            ipfs_url = f"{args.ipfs_gateway}/{args.lora_ipfs}"
            logger.info(f"Fetching from: {ipfs_url}")

            response = requests.get(ipfs_url, timeout=HTTP_TIMEOUT_DOWNLOAD, stream=True)
            response.raise_for_status()

            with open(local_lora_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(local_lora_path)
            logger.info(f"Downloaded LoRA from IPFS to {local_lora_path} ({file_size / 1024 / 1024:.1f} MB)")

        else:
            # Download from GCS
            logger.info("Downloading LoRA from GCS...")
            client = gcs.Client()

            # Parse GCS URI
            if not args.lora_gcs.startswith("gs://"):
                raise ValueError(f"Invalid GCS URI: {args.lora_gcs}")
            gcs_parts = args.lora_gcs[5:].split("/", 1)
            bucket_name = gcs_parts[0]
            blob_path = gcs_parts[1]

            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.download_to_filename(local_lora_path)
            logger.info(f"Downloaded LoRA from GCS to {local_lora_path}")

        # Step 2: Load pipeline and generate
        logger.info("Loading inference pipeline...")
        pipe = await get_inference_pipeline()

        logger.info(f"Loading LoRA from {local_lora_path}")
        pipe.load_lora_weights("/tmp/lora", weight_name="pytorch_lora_weights.safetensors")

        # Ensure trigger word in prompt
        prompt = args.prompt
        if args.trigger_word and args.trigger_word not in prompt:
            prompt = f"a photo of {args.trigger_word}, {prompt}"

        logger.info(f"Generating {args.num_outputs} images: {prompt[:80]}...")

        images = []
        for i in range(args.num_outputs):
            logger.info(f"Generating image {i+1}/{args.num_outputs}...")
            result = pipe(
                prompt=prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=28,
                guidance_scale=4.0,
            )
            images.append(result.images[0])
            logger.info(f"Generated image {i+1}")

        pipe.unload_lora_weights()

        # Step 3: Upload images to GCS
        logger.info("Uploading images to GCS...")

        # Parse output GCS URI
        if not args.output_gcs.startswith("gs://"):
            raise ValueError(f"Invalid output GCS URI: {args.output_gcs}")
        output_parts = args.output_gcs[5:].split("/", 1)
        output_bucket_name = output_parts[0]
        output_prefix = output_parts[1] if len(output_parts) > 1 else ""

        # Create GCS client for output upload (may not have been created if using IPFS input)
        output_client = gcs.Client()
        output_bucket = output_client.bucket(output_bucket_name)

        image_urls = []
        for i, image in enumerate(images):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            blob_name = f"{output_prefix}/image_{i}.png"
            output_blob = output_bucket.blob(blob_name)
            output_blob.upload_from_file(buffer, content_type="image/png")
            image_urls.append(f"gs://{output_bucket_name}/{blob_name}")
            logger.info(f"Uploaded: gs://{output_bucket_name}/{blob_name}")

        # Write metadata
        metadata = {
            "prompt": prompt,
            "trigger_word": args.trigger_word,
            "lora_source": args.lora_ipfs if args.lora_ipfs else args.lora_gcs,
            "lora_ipfs": args.lora_ipfs,
            "lora_gcs": args.lora_gcs,
            "num_outputs": len(images),
            "image_urls": image_urls,
            "width": args.width,
            "height": args.height,
            "completed_at": datetime.utcnow().isoformat(),
        }
        metadata_blob = output_bucket.blob(f"{output_prefix}/metadata.json")
        metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
        logger.info(f"Uploaded metadata to gs://{output_bucket_name}/{output_prefix}/metadata.json")

        logger.info("=" * 60)
        logger.info(f"Generation completed! {len(images)} images")
        logger.info("=" * 60)
        sys.exit(0)

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def run_vertex_batch_training(args):
    """
    Run training in batch mode for Vertex AI Custom Jobs.

    This is a one-shot execution that:
    1. Downloads training image from GCS
    2. Runs SimpleTuner training
    3. Uploads trained LoRA to GCS
    4. Exits with success (0) or failure (1)

    Args:
        args: Parsed command-line arguments with:
            - avatar_gcs: gs://bucket/path/to/avatar.png
            - output_gcs: gs://bucket/path/for/output/
            - companion_id: Companion UUID
            - trigger_word: LoRA trigger word
            - steps: Training steps
            - lora_rank: LoRA rank
    """
    from google.cloud import storage as gcs

    logger.info("=" * 60)
    logger.info("Vertex AI Batch Training Mode")
    logger.info("=" * 60)
    logger.info(f"Companion ID: {args.companion_id}")
    logger.info(f"Trigger Word: {args.trigger_word}")
    logger.info(f"Steps: {args.steps}")
    logger.info(f"LoRA Rank: {args.lora_rank}")
    logger.info(f"Avatar GCS: {args.avatar_gcs}")
    logger.info(f"Output GCS: {args.output_gcs}")

    job_id = args.companion_id
    paths = app_config.get_runtime_paths()

    try:
        # Step 1: Download avatar from GCS
        logger.info("Downloading training image from GCS...")

        # Parse GCS URI: gs://bucket/path/to/file
        if not args.avatar_gcs.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {args.avatar_gcs}")

        gcs_parts = args.avatar_gcs[5:].split("/", 1)
        bucket_name = gcs_parts[0]
        blob_path = gcs_parts[1] if len(gcs_parts) > 1 else ""

        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Create dataset directory - use configured paths
        dataset_path = f"{paths['datasets_path']}/{job_id}"
        os.makedirs(dataset_path, exist_ok=True)

        # Download image
        local_image_path = f"{dataset_path}/image_001.png"
        blob.download_to_filename(local_image_path)
        logger.info(f"Downloaded training image to {local_image_path}")

        # Step 2: Create training config
        output_path = f"{paths['output_path']}/{job_id}"
        cache_path = f"{output_path}/cache"

        config = create_simpletuner_config(
            job_id=job_id,
            trigger_word=args.trigger_word,
            dataset_path=dataset_path,
            output_path=output_path,
            cache_path=cache_path,
            steps=args.steps,
            lora_rank=args.lora_rank,
        )

        # Create job record for monitoring
        app_config.training_jobs[job_id] = {
            "job_id": job_id,
            "companion_id": args.companion_id,
            "trigger_word": args.trigger_word,
            "status": "preparing",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": args.steps,
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "lora_path": None,
            "error": None,
        }

        # Step 3: Run training (synchronously in batch mode)
        logger.info("Starting SimpleTuner training...")
        await run_training(job_id, config, dataset_path)

        job = app_config.training_jobs[job_id]

        if job["status"] == "failed":
            raise RuntimeError(f"Training failed: {job.get('error', 'Unknown error')}")

        # Step 4: Upload LoRA to GCS
        logger.info("Uploading trained LoRA to GCS...")

        lora_path = job.get("lora_path")
        if not lora_path or not os.path.exists(lora_path):
            raise RuntimeError(f"LoRA file not found: {lora_path}")

        # Parse output GCS URI
        if not args.output_gcs.startswith("gs://"):
            raise ValueError(f"Invalid output GCS URI: {args.output_gcs}")

        output_parts = args.output_gcs[5:].split("/", 1)
        output_bucket_name = output_parts[0]
        output_prefix = output_parts[1] if len(output_parts) > 1 else ""

        output_bucket = client.bucket(output_bucket_name)

        # Upload LoRA file
        lora_blob_path = f"{output_prefix}/{job_id}/pytorch_lora_weights.safetensors"
        lora_blob = output_bucket.blob(lora_blob_path)
        lora_blob.upload_from_filename(lora_path)
        logger.info(f"Uploaded LoRA to gs://{output_bucket_name}/{lora_blob_path}")

        # Upload metadata
        metadata = {
            "job_id": job_id,
            "companion_id": args.companion_id,
            "trigger_word": args.trigger_word,
            "steps": args.steps,
            "lora_rank": args.lora_rank,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "lora_gcs_path": f"gs://{output_bucket_name}/{lora_blob_path}",
        }
        metadata_blob_path = f"{output_prefix}/{job_id}/metadata.json"
        metadata_blob = output_bucket.blob(metadata_blob_path)
        metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
        logger.info(f"Uploaded metadata to gs://{output_bucket_name}/{metadata_blob_path}")

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"LoRA: gs://{output_bucket_name}/{lora_blob_path}")
        logger.info("=" * 60)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Batch training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
