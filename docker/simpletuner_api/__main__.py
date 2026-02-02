"""
SimpleTuner FLUX.2 Training API - Entry Point.

Supports three modes:
1. API Server Mode (default): Runs FastAPI server on RunPod
2. Vertex AI Training Mode: One-shot training job with GCS storage
3. Vertex AI Generation Mode: One-shot generation job with GCS storage

Usage:
    # API server mode (RunPod)
    python -m simpletuner_api --port 8000

    # Vertex AI training batch mode
    python -m simpletuner_api --vertex-mode \\
        --avatar-gcs gs://bucket/avatar.png \\
        --output-gcs gs://bucket/output/ \\
        --companion-id <uuid> \\
        --trigger-word TOKabc123

    # Vertex AI generation batch mode
    python -m simpletuner_api --generate-mode \\
        --lora-ipfs QmXxx... \\
        --output-gcs gs://bucket/output/ \\
        --prompt "a photo of TOKabc123"
"""

import argparse
import asyncio
import logging
import sys

import uvicorn

from . import create_app
from .config import setup_paths
from .vertex import run_vertex_batch_generation, run_vertex_batch_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="SimpleTuner FLUX.2 Training - API or Batch Mode"
    )

    # Training mode (Vertex AI batch)
    parser.add_argument(
        "--vertex-mode",
        action="store_true",
        help="Run in Vertex AI training batch mode (download from GCS, train, upload to GCS, exit)"
    )
    parser.add_argument(
        "--avatar-gcs",
        type=str,
        help="GCS URI for training image (gs://bucket/path/to/avatar.png)"
    )
    parser.add_argument(
        "--companion-id",
        type=str,
        help="Companion UUID being trained"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Training steps (default: 1000)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )

    # Generation mode (Vertex AI batch)
    parser.add_argument(
        "--generate-mode",
        action="store_true",
        help="Run in Vertex AI generation batch mode (download LoRA, generate, upload to GCS, exit)"
    )
    parser.add_argument(
        "--lora-gcs",
        type=str,
        help="GCS URI for LoRA weights (gs://bucket/path/to/pytorch_lora_weights.safetensors)"
    )
    parser.add_argument(
        "--lora-ipfs",
        type=str,
        help="IPFS CID for LoRA weights (e.g., QmXxx...)"
    )
    parser.add_argument(
        "--ipfs-gateway",
        type=str,
        default="https://gateway.lighthouse.storage/ipfs",
        help="IPFS gateway URL (default: https://gateway.lighthouse.storage/ipfs)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Generation prompt"
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)"
    )

    # Shared arguments
    parser.add_argument(
        "--output-gcs",
        type=str,
        help="GCS URI prefix for output (gs://bucket/path/for/output/)"
    )
    parser.add_argument(
        "--trigger-word",
        type=str,
        help="LoRA trigger word (default: TOK)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000, only for API mode)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SimpleTuner FLUX.2 Training")
    logger.info("=" * 60)

    # Log GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.warning("CUDA not available!")
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")

    if args.generate_mode:
        # Vertex AI generation batch mode
        if not args.lora_gcs and not args.lora_ipfs:
            parser.error("--lora-gcs or --lora-ipfs is required for --generate-mode")
        if args.lora_gcs and args.lora_ipfs:
            parser.error("Specify either --lora-gcs or --lora-ipfs, not both")
        if not args.output_gcs:
            parser.error("--output-gcs is required for --generate-mode")
        if not args.prompt:
            parser.error("--prompt is required for --generate-mode")

        # Set default trigger word if not provided
        if not args.trigger_word:
            args.trigger_word = "TOK"

        # Configure paths for Vertex AI environment
        setup_paths(is_vertex_mode=True)

        logger.info("Running in VERTEX AI GENERATION MODE")
        asyncio.run(run_vertex_batch_generation(args))

    elif args.vertex_mode:
        # Vertex AI training batch mode
        if not args.avatar_gcs:
            parser.error("--avatar-gcs is required for --vertex-mode")
        if not args.output_gcs:
            parser.error("--output-gcs is required for --vertex-mode")
        if not args.companion_id:
            parser.error("--companion-id is required for --vertex-mode")

        # Set default trigger word if not provided
        if not args.trigger_word:
            clean_id = args.companion_id.replace("-", "")[:8]
            args.trigger_word = f"TOK{clean_id}"

        # Configure paths for Vertex AI environment
        setup_paths(is_vertex_mode=True)

        logger.info("Running in VERTEX AI TRAINING MODE")
        asyncio.run(run_vertex_batch_training(args))

    else:
        # API server mode (RunPod)
        # Configure paths for RunPod environment - WILL FAIL if /workspace not mounted
        setup_paths(is_vertex_mode=False)

        logger.info("Running in API SERVER MODE")
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
