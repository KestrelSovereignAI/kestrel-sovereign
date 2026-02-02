#!/usr/bin/env python3
"""
SDXL Direct Generation Demo - No LoRA (baseline test)

Tests SDXL pipeline on local MPS before adding LoRA training.

Usage:
    uv run python scripts/sdxl_direct_demo.py
"""

import asyncio
import base64
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

# Paths
MODEL_PATH = os.getenv("SDXL_MODEL_PATH", os.path.expanduser("~/models/sdxl/stable-diffusion-xl-base-1.0"))
OUTPUT_DIR = Path(os.getenv("LOCAL_MPS_OUTPUT_DIR", os.path.expanduser("~/models/local-training/demo_output")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("SDXL Direct Generation Demo (No LoRA)")
    print("=" * 70)
    print()

    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: MPS (Apple Silicon)")
    print()

    # Load pipeline
    print("Loading SDXL pipeline...")
    start = datetime.now()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to("mps")

    load_time = (datetime.now() - start).total_seconds()
    print(f"Pipeline loaded in {load_time:.1f}s")
    print()

    # Generate nurse portrait
    print("=" * 70)
    print("Generating nurse portrait...")
    print("=" * 70)

    nurse_prompt = "beautiful young woman wearing white nurse scrubs uniform, " \
                   "standing in modern hospital, professional medical setting, " \
                   "warm smile, natural lighting, photorealistic, 8k quality"

    print(f"Prompt: {nurse_prompt[:60]}...")

    generator = torch.Generator(device="mps").manual_seed(42)

    start = datetime.now()
    image = pipe(
        prompt=nurse_prompt,
        negative_prompt="blurry, ugly, deformed, low quality",
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
        width=1024,
        height=1024,
    ).images[0]

    gen_time = (datetime.now() - start).total_seconds()
    print(f"Generated in {gen_time:.1f}s")

    # Save
    nurse_path = OUTPUT_DIR / "sdxl_nurse_baseline.png"
    image.save(nurse_path)
    print(f"Saved: {nurse_path}")
    print()

    # Generate beach selfie
    print("=" * 70)
    print("Generating beach selfie...")
    print("=" * 70)

    beach_prompt = "beautiful young woman on tropical beach at golden hour sunset, " \
                   "wearing elegant red bikini swimsuit, ocean waves background, " \
                   "relaxed happy pose, professional photography, 8k quality"

    print(f"Prompt: {beach_prompt[:60]}...")

    generator = torch.Generator(device="mps").manual_seed(123)

    start = datetime.now()
    image = pipe(
        prompt=beach_prompt,
        negative_prompt="blurry, ugly, deformed, low quality",
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
        width=1024,
        height=1024,
    ).images[0]

    gen_time = (datetime.now() - start).total_seconds()
    print(f"Generated in {gen_time:.1f}s")

    # Save
    beach_path = OUTPUT_DIR / "sdxl_beach_baseline.png"
    image.save(beach_path)
    print(f"Saved: {beach_path}")
    print()

    # Summary
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print(f"Generated images:")
    print(f"  1. {nurse_path}")
    print(f"  2. {beach_path}")
    print()
    print("Open these files to verify SDXL generation works!")
    print()
    print("Note: These are BASELINE images (no LoRA). They won't have")
    print("a specific identity - each is a random person matching the prompt.")


if __name__ == "__main__":
    main()
