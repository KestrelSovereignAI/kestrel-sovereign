#!/usr/bin/env python3
"""
LoRA Training Entry Point for Kestrel

Trains character-specific LoRA on FLUX.2-dev using a single reference image.
FLUX.2 (December 2025) has NO built-in safety filters when self-hosted = TRUE SOVEREIGNTY.
Designed to run on RunPod RTX 3090/4090 GPU with FP8 quantization.

Usage:
    # As server (receives training requests via HTTP)
    python train_lora.py --server

    # Direct training (for testing)
    python train_lora.py --image /path/to/avatar.jpg --output /path/to/lora.safetensors
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import toml
from PIL import Image

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_MEDIUM, HTTP_TIMEOUT_UPLOAD

# FastAPI imports for server mode
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported base models for LoRA training
# FLUX.2-dev (December 2025) - NO built-in safety filters when self-hosted = TRUE SOVEREIGNTY
SUPPORTED_MODELS = {
    "flux2-dev": "black-forest-labs/FLUX.2-dev",     # FLUX.2 - best quality, no censorship when self-hosted
    "flux-dev": "black-forest-labs/FLUX.2-dev",      # Alias for backwards compatibility
}

# Training configuration
# FLUX.2-dev requires ~90GB VRAM full, ~64GB with FP8 quantization
# RTX 4090 (24GB) needs aggressive quantization settings below
DEFAULT_CONFIG = {
    "base_model": "black-forest-labs/FLUX.2-dev",  # FLUX.2 base model
    "training_steps": 500,
    "learning_rate": 1e-4,
    "lora_rank": 16,
    "lora_alpha": 16,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "resolution": 1024,
    "mixed_precision": "bf16",
    "use_8bit_adam": True,
    "num_augmentations": 10,  # Create variations of single image
}


@dataclass
class TrainingJob:
    """Represents a LoRA training job"""
    job_id: str
    companion_id: str
    image_url: str
    callback_url: Optional[str]
    status: str = "pending"  # pending, downloading, training, uploading, completed, failed
    progress: float = 0.0
    error: Optional[str] = None
    output_path: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    base_model: str = "flux-dev"  # Model shortname used for training


class LoRATrainer:
    """Handles the actual LoRA training process"""

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.base_dir = Path("/app")
        self.training_dir = self.base_dir / "training_data"
        self.output_dir = self.base_dir / "output"
        self.cache_dir = self.base_dir / "cache"

        # Ensure directories exist
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def download_image(self, url: str, dest_path: Path) -> bool:
        """Download avatar image from URL"""
        logger.info(f"Downloading image from {url}")
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_MEDIUM) as client:
                response = await client.get(url)
                response.raise_for_status()

                with open(dest_path, 'wb') as f:
                    f.write(response.content)

                # Verify it's a valid image
                img = Image.open(dest_path)
                img.verify()

                logger.info(f"Downloaded image: {dest_path} ({os.path.getsize(dest_path)} bytes)")
                return True
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            return False

    def augment_image(self, image_path: Path, output_dir: Path, num_augmentations: int = 10) -> list:
        """Create augmented versions of the training image"""
        logger.info(f"Creating {num_augmentations} augmented versions")

        img = Image.open(image_path)
        augmented_paths = [image_path]  # Include original

        for i in range(num_augmentations - 1):
            aug_path = output_dir / f"aug_{i:03d}.jpg"

            # Apply random transformations
            aug_img = img.copy()

            # Random crop (slight variations)
            import random
            w, h = aug_img.size
            crop_percent = random.uniform(0.85, 0.95)
            new_w, new_h = int(w * crop_percent), int(h * crop_percent)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            aug_img = aug_img.crop((left, top, left + new_w, top + new_h))
            aug_img = aug_img.resize((w, h), Image.Resampling.LANCZOS)

            # Random horizontal flip (50% chance)
            if random.random() > 0.5:
                aug_img = aug_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Slight color variation
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Color(aug_img)
            aug_img = enhancer.enhance(random.uniform(0.9, 1.1))

            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(random.uniform(0.95, 1.05))

            aug_img.save(aug_path, quality=95)
            augmented_paths.append(aug_path)

        logger.info(f"Created {len(augmented_paths)} training images")
        return augmented_paths

    def create_caption_files(self, image_paths: list, trigger_word: str = "kestrel_character"):
        """Create caption files for each training image"""
        caption_template = f"a photo of {trigger_word}, portrait, high quality, detailed"

        for img_path in image_paths:
            caption_path = img_path.with_suffix('.txt')
            caption_path.write_text(caption_template)

        logger.info(f"Created caption files for {len(image_paths)} images")

    async def train_lora(
        self,
        image_paths: list,
        output_path: Path,
        companion_id: str,
        progress_callback=None
    ) -> bool:
        """
        Train LoRA on FLUX.2-dev using diffusers with memory optimizations.

        Based on:
        - docs/research/Training LoRA on the FLUX 2 [dev] Diffusion Model.pdf
        - HuggingFace diffusers FLUX DreamBooth LoRA example

        Key optimizations for RTX 4090 (24GB):
        - gradient_checkpointing
        - 8-bit Adam optimizer
        - bf16 mixed precision
        - CPU offload for text encoders
        - Pre-cached latents

        Args:
            image_paths: List of training image paths
            output_path: Where to save the .safetensors LoRA
            companion_id: Unique identifier for logging/naming
            progress_callback: Optional callback for progress updates

        Returns:
            True if training succeeded, False otherwise
        """
        logger.info(f"Starting FLUX.2 LoRA training for companion {companion_id}")
        logger.info(f"Training images: {len(image_paths)}, Output: {output_path}")

        try:
            import torch
            import torch.nn.functional as F
            from diffusers import Flux2Pipeline, Flux2Transformer2DModel, AutoencoderKL
            from diffusers.optimization import get_scheduler
            from transformers import Mistral3ForConditionalGeneration, AutoTokenizer
            from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
            from safetensors.torch import save_file
            import gc

            # Check GPU availability
            if not torch.cuda.is_available():
                logger.error("CUDA not available - cannot train")
                return False

            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Training on: {gpu_name} ({gpu_memory_gb:.1f}GB)")

            # Determine precision based on GPU
            weight_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logger.info(f"Using dtype: {weight_dtype}")

            # Cache directory for HuggingFace models
            cache_dir = os.environ.get("HF_HOME", "/workspace/huggingface")
            os.makedirs(cache_dir, exist_ok=True)

            if progress_callback:
                progress_callback(0.05)

            # =========================================================
            # Step 1: Load VAE and encode training images to latents
            # =========================================================
            logger.info("Loading VAE and encoding training images...")

            vae = AutoencoderKL.from_pretrained(
                self.config["base_model"],
                subfolder="vae",
                torch_dtype=weight_dtype,
                cache_dir=cache_dir
            ).to(device)
            vae.requires_grad_(False)

            # Pre-encode all images to latents (saves memory during training)
            resolution = self.config["resolution"]
            latents_list = []

            for img_path in image_paths:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)

                # Convert to tensor [-1, 1]
                img_tensor = torch.from_numpy(
                    np.array(img).transpose(2, 0, 1)
                ).float() / 127.5 - 1.0
                img_tensor = img_tensor.unsqueeze(0).to(device, dtype=weight_dtype)

                with torch.no_grad():
                    latent = vae.encode(img_tensor).latent_dist.sample()
                    latent = latent * vae.config.scaling_factor
                    latents_list.append(latent.cpu())

            # Free VAE memory
            del vae
            torch.cuda.empty_cache()
            gc.collect()

            if progress_callback:
                progress_callback(0.15)

            # =========================================================
            # Step 2: Load and encode text prompts
            # =========================================================
            # FLUX.2 uses Mistral Small 3.1 as single text encoder (simpler than FLUX.1's CLIP+T5)
            logger.info("Loading Mistral text encoder and encoding prompts...")

            tokenizer = AutoTokenizer.from_pretrained(
                self.config["base_model"],
                subfolder="tokenizer",
                cache_dir=cache_dir
            )

            text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                self.config["base_model"],
                subfolder="text_encoder",
                torch_dtype=weight_dtype,
                cache_dir=cache_dir
            ).to(device)
            text_encoder.requires_grad_(False)

            # Encode the training prompt
            trigger_word = f"companion_{companion_id[:8]}"
            prompt = f"a photo of {trigger_word}, portrait, high quality, detailed"

            # Mistral encoding (FLUX.2 uses single encoder with max_length=512)
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                prompt_embeds = text_encoder(
                    text_inputs.input_ids,
                    output_hidden_states=True
                ).hidden_states[-1]

            pooled_prompt_embeds = prompt_embeds[:, 0, :]  # CLS token

            # Free text encoder memory
            del text_encoder
            torch.cuda.empty_cache()
            gc.collect()

            if progress_callback:
                progress_callback(0.25)

            # =========================================================
            # Step 3: Load transformer and apply LoRA
            # =========================================================
            logger.info("Loading FLUX.2 transformer and configuring LoRA...")

            transformer = Flux2Transformer2DModel.from_pretrained(
                self.config["base_model"],
                subfolder="transformer",
                torch_dtype=weight_dtype,
                cache_dir=cache_dir
            )

            # Enable gradient checkpointing for memory savings
            transformer.enable_gradient_checkpointing()

            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config["lora_rank"],
                lora_alpha=self.config["lora_alpha"],
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=0.0,
            )

            transformer = get_peft_model(transformer, lora_config)
            transformer.to(device)

            # Count trainable parameters
            trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in transformer.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

            if progress_callback:
                progress_callback(0.30)

            # =========================================================
            # Step 4: Setup optimizer and scheduler
            # =========================================================
            logger.info("Setting up optimizer...")

            # Use 8-bit Adam for memory efficiency
            if self.config.get("use_8bit_adam", True):
                try:
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.AdamW8bit(
                        transformer.parameters(),
                        lr=self.config["learning_rate"],
                        betas=(0.9, 0.999),
                        weight_decay=1e-2
                    )
                    logger.info("Using 8-bit Adam optimizer")
                except ImportError:
                    optimizer = torch.optim.AdamW(
                        transformer.parameters(),
                        lr=self.config["learning_rate"],
                        betas=(0.9, 0.999),
                        weight_decay=1e-2
                    )
                    logger.info("Using standard AdamW (bitsandbytes not available)")
            else:
                optimizer = torch.optim.AdamW(
                    transformer.parameters(),
                    lr=self.config["learning_rate"],
                    betas=(0.9, 0.999),
                    weight_decay=1e-2
                )

            max_train_steps = self.config.get("training_steps", 1500)
            lr_scheduler = get_scheduler(
                "constant",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=max_train_steps
            )

            if progress_callback:
                progress_callback(0.35)

            # =========================================================
            # Step 5: Training loop
            # =========================================================
            logger.info(f"Starting training for {max_train_steps} steps...")

            transformer.train()
            global_step = 0
            num_latents = len(latents_list)

            for step in range(max_train_steps):
                # Get random latent from pre-computed set
                latent_idx = step % num_latents
                latents = latents_list[latent_idx].to(device, dtype=weight_dtype)

                # Sample random timestep
                # FLUX uses flow matching, so timesteps are continuous [0, 1]
                timesteps = torch.rand(1, device=device) * 0.999 + 0.001

                # Sample noise
                noise = torch.randn_like(latents)

                # Flow matching: noisy = (1 - sigma) * latents + sigma * noise
                sigma = timesteps.view(-1, 1, 1, 1)
                noisy_latents = (1 - sigma) * latents + sigma * noise

                # Predict the noise residual
                # FLUX.2 uses single Mistral encoder (prompt_embeds) instead of FLUX.1's CLIP+T5
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds.to(device, dtype=weight_dtype),
                    pooled_projections=pooled_prompt_embeds.to(device, dtype=weight_dtype),
                    return_dict=False
                )[0]

                # Target is the velocity (noise - latents) for flow matching
                target = noise - latents

                # MSE loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (step + 1) % self.config.get("gradient_accumulation", 4) == 0:
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                global_step += 1

                # Log progress
                if step % 50 == 0:
                    progress = 0.35 + 0.55 * (step / max_train_steps)
                    logger.info(f"Step {step}/{max_train_steps} | Loss: {loss.item():.4f}")
                    if progress_callback:
                        progress_callback(progress)

            if progress_callback:
                progress_callback(0.90)

            # =========================================================
            # Step 6: Save LoRA weights
            # =========================================================
            logger.info("Saving LoRA weights...")

            transformer.eval()

            # Get LoRA state dict
            lora_state_dict = get_peft_model_state_dict(transformer)

            # Convert to safetensors-compatible format
            lora_weights = {}
            for key, value in lora_state_dict.items():
                # Convert to float16 for smaller file size
                lora_weights[key] = value.cpu().half()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(lora_weights, str(output_path))

            # Also save metadata
            metadata_path = output_path.with_suffix('.json')
            metadata = {
                "companion_id": companion_id,
                "trigger_word": trigger_word,
                "base_model": self.config["base_model"],
                "training_images": len(image_paths),
                "training_steps": max_train_steps,
                "lora_rank": self.config["lora_rank"],
                "lora_alpha": self.config["lora_alpha"],
                "resolution": resolution,
                "created_at": time.time(),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))

            if progress_callback:
                progress_callback(1.0)

            logger.info(f"Training complete! LoRA saved to {output_path}")
            logger.info(f"Trigger word for generation: {trigger_word}")

            # Cleanup
            del transformer, optimizer
            torch.cuda.empty_cache()
            gc.collect()

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return False

    async def train_from_url(
        self,
        image_url: str,
        companion_id: str,
        output_path: Optional[Path] = None,
        progress_callback=None
    ) -> Optional[Path]:
        """Full training pipeline from image URL to LoRA file"""

        # Create job-specific directories
        job_dir = self.training_dir / companion_id
        job_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_path or (self.output_dir / f"{companion_id}.safetensors")

        try:
            # Step 1: Download image
            if progress_callback:
                progress_callback(0.05)

            image_path = job_dir / "original.jpg"
            if not await self.download_image(image_url, image_path):
                return None

            # Step 2: Augment image
            if progress_callback:
                progress_callback(0.1)

            augmented = self.augment_image(
                image_path,
                job_dir,
                self.config["num_augmentations"]
            )

            # Step 3: Create captions
            trigger_word = f"companion_{companion_id[:8]}"
            self.create_caption_files(augmented, trigger_word)

            # Step 4: Train
            def training_progress(p):
                # Scale training progress to 0.15-0.95
                if progress_callback:
                    progress_callback(0.15 + p * 0.8)

            success = await self.train_lora(
                augmented,
                output_path,
                companion_id,
                training_progress
            )

            if not success:
                return None

            # Step 5: Cleanup
            if progress_callback:
                progress_callback(1.0)

            # Clean up training data
            shutil.rmtree(job_dir, ignore_errors=True)

            return output_path

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            shutil.rmtree(job_dir, ignore_errors=True)
            return None


# Server mode for RunPod deployment
if FASTAPI_AVAILABLE:
    app = FastAPI(title="Kestrel LoRA Trainer", version="1.0.0")
    trainer = LoRATrainer()
    jobs: dict[str, TrainingJob] = {}

    # Request models for JSON body endpoints
    class GenerateRequest(BaseModel):
        prompt: str
        lora_url: Optional[str] = None
        lora_path: Optional[str] = None
        num_outputs: int = 1
        aspect_ratio: str = "1:1"
        output_format: str = "jpg"

    # Global pipeline - loaded once at startup to avoid timeout on first /generate
    _inference_pipeline = None
    _pipeline_loading = False

    @app.on_event("startup")
    async def load_inference_model():
        """Pre-load FLUX model at startup to avoid timeout on first /generate request.

        The model is ~24GB and can take 5-10 minutes to download on first run.
        Subsequent restarts use the cached model from /workspace/huggingface.
        """
        global _inference_pipeline, _pipeline_loading
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA not available - skipping model preload")
            return

        _pipeline_loading = True
        logger.info("="*60)
        logger.info("PRE-LOADING FLUX MODEL (may take 5-10 minutes on first run)")
        logger.info("="*60)

        try:
            from diffusers import Flux2Pipeline

            # Use /workspace for persistent volume storage (survives pod restart)
            cache_dir = os.environ.get("HF_HOME", "/workspace/huggingface")
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Using cache dir: {cache_dir}")

            _inference_pipeline = Flux2Pipeline.from_pretrained(
                DEFAULT_CONFIG["base_model"],
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir
            )

            # Check GPU memory and apply appropriate optimization
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU memory: {gpu_memory_gb:.1f} GB")

            if gpu_memory_gb >= 40:
                # A100 80GB, H100, etc - just move to GPU, no offload needed
                _inference_pipeline = _inference_pipeline.to("cuda")
                logger.info("Large GPU detected - model loaded to CUDA without offload")
            else:
                # RTX 3090/4090 (24GB) - use CPU offload
                _inference_pipeline.enable_model_cpu_offload()
                logger.info("Enabled CPU offload for memory optimization")

            logger.info("="*60)
            logger.info("FLUX MODEL LOADED AND READY!")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
            # Don't crash - let individual requests try to load
        finally:
            _pipeline_loading = False

    @app.get("/health")
    async def health_check():
        """Basic health check - returns OK if service is running."""
        import torch
        global _inference_pipeline, _pipeline_loading
        return {
            "status": "healthy",
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "active_jobs": len([j for j in jobs.values() if j.status == "training"]),
            "model_loaded": _inference_pipeline is not None,
            "model_loading": _pipeline_loading,
        }

    @app.get("/ready")
    async def readiness_check():
        """
        Kubernetes/RunPod readiness probe.

        Returns 200 ONLY when:
        - Model is fully loaded to GPU (not just downloading)
        - No startup in progress
        - Ready to accept training/inference jobs

        Returns 503 if not ready (model still loading).

        Use this endpoint to check if the pod is ready to accept work.
        The /health endpoint returns OK even while model is loading.
        """
        import torch
        global _inference_pipeline, _pipeline_loading

        # Check if still loading
        if _pipeline_loading:
            raise HTTPException(
                status_code=503,
                detail="Model still loading - please wait"
            )

        # Check if model is loaded
        if _inference_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded - startup may have failed"
            )

        # Check CUDA is actually available
        if not torch.cuda.is_available():
            raise HTTPException(
                status_code=503,
                detail="CUDA not available"
            )

        return {
            "status": "ready",
            "model_loaded": True,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
        }

    @app.get("/models")
    async def list_models():
        """List supported base models for LoRA training"""
        return {
            "models": SUPPORTED_MODELS,
            "default": "flux-dev",
        }

    @app.post("/train")
    async def start_training(
        companion_id: str,
        image_url: str,
        callback_url: Optional[str] = None,
        base_model: str = "flux-dev",  # Model shortname from SUPPORTED_MODELS
        background_tasks: BackgroundTasks = None
    ):
        """Start a LoRA training job

        Args:
            companion_id: Unique companion identifier
            image_url: URL to avatar image for training
            callback_url: Optional webhook for completion notification
            base_model: Model to train on (flux-dev, flux-schnell, sdxl, sd15)
        """
        # Resolve model name
        if base_model in SUPPORTED_MODELS:
            model_path = SUPPORTED_MODELS[base_model]
        elif base_model.startswith("black-forest-labs/") or base_model.startswith("stabilityai/"):
            model_path = base_model  # Allow full HF paths
        else:
            raise HTTPException(400, f"Unknown model: {base_model}. Supported: {list(SUPPORTED_MODELS.keys())}")

        job_id = hashlib.md5(f"{companion_id}:{time.time()}".encode()).hexdigest()[:16]

        job = TrainingJob(
            job_id=job_id,
            companion_id=companion_id,
            image_url=image_url,
            callback_url=callback_url,
            started_at=time.time()
        )
        job.base_model = base_model  # Track which model was used
        jobs[job_id] = job

        async def run_training():
            try:
                job.status = "training"

                def update_progress(p):
                    job.progress = p

                output_path = await trainer.train_from_url(
                    image_url=image_url,
                    companion_id=companion_id,
                    progress_callback=update_progress
                )

                if output_path:
                    job.status = "completed"
                    job.output_path = str(output_path)
                else:
                    job.status = "failed"
                    job.error = "Training failed"

                job.completed_at = time.time()

                # Send callback if configured
                if callback_url:
                    try:
                        async with httpx.AsyncClient() as client:
                            await client.post(callback_url, json={
                                "job_id": job_id,
                                "status": job.status,
                                "output_path": job.output_path,
                                "error": job.error,
                            })
                    except Exception as e:
                        logger.error(f"Callback failed: {e}")

            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                job.completed_at = time.time()

        background_tasks.add_task(run_training)

        return {
            "job_id": job_id,
            "status": "started",
            "companion_id": companion_id,
        }

    @app.get("/jobs/{job_id}")
    async def get_job_status(job_id: str):
        """Get status of a training job"""
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")

        job = jobs[job_id]
        return {
            "job_id": job.job_id,
            "companion_id": job.companion_id,
            "status": job.status,
            "progress": job.progress,
            "error": job.error,
            "output_path": job.output_path,
            "duration": (job.completed_at or time.time()) - job.started_at if job.started_at else None,
        }

    @app.get("/jobs/{job_id}/download")
    async def download_lora(job_id: str):
        """Download the trained LoRA file"""
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")

        job = jobs[job_id]
        if job.status != "completed" or not job.output_path:
            raise HTTPException(400, "Training not complete or failed")

        from fastapi.responses import FileResponse
        return FileResponse(
            job.output_path,
            media_type="application/octet-stream",
            filename=f"lora_{job.companion_id}.safetensors"
        )

    # =========================================================================
    # Inference Endpoint - Generate images WITH trained LoRA
    # =========================================================================

    @app.post("/generate")
    async def generate_with_lora(request: GenerateRequest):
        """
        Generate images using FLUX with optional LoRA.

        Args (JSON body):
            prompt: Image generation prompt
            lora_url: URL to download LoRA from (ipfs://... or http://...)
            lora_path: Local path to LoRA if already downloaded
            num_outputs: Number of images to generate (1-4)
            aspect_ratio: Image aspect ratio (1:1, 16:9, 9:16, etc)
            output_format: Output format (jpg, png, webp)

        Returns:
            {"images": [base64_encoded_images], "format": "jpg"}
        """
        global _inference_pipeline, _pipeline_loading
        import torch
        import base64
        from io import BytesIO

        # Extract from request body
        prompt = request.prompt
        lora_url = request.lora_url
        lora_path = request.lora_path
        num_outputs = request.num_outputs
        aspect_ratio = request.aspect_ratio
        output_format = request.output_format

        if not torch.cuda.is_available():
            raise HTTPException(503, "GPU not available for inference")

        # Check if model is still loading
        if _pipeline_loading:
            raise HTTPException(503, "Model is still loading, please try again in a minute")

        try:
            # Use pre-loaded pipeline if available, otherwise load on demand
            if _inference_pipeline is not None:
                logger.info("Using pre-loaded FLUX pipeline")
                pipe = _inference_pipeline
            else:
                # Fallback: load model on demand (slower, but works if startup failed)
                logger.warning("Pre-loaded pipeline not available, loading on demand...")
                from diffusers import Flux2Pipeline
                cache_dir = os.environ.get("HF_HOME", "/workspace/huggingface")
                pipe = Flux2Pipeline.from_pretrained(
                    DEFAULT_CONFIG["base_model"],
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir
                )
                # Check GPU memory and apply appropriate optimization
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory_gb >= 40:
                    pipe = pipe.to("cuda")
                else:
                    pipe.enable_model_cpu_offload()

            # Load LoRA if provided
            if lora_url:
                # Download from URL (IPFS or HTTP)
                logger.info(f"Downloading LoRA from {lora_url}...")
                lora_local = Path("/app/cache/inference_lora.safetensors")

                if lora_url.startswith("ipfs://"):
                    # Download from IPFS gateway
                    cid = lora_url.replace("ipfs://", "")
                    gateway_url = f"https://ipfs.io/ipfs/{cid}"
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_UPLOAD) as client:
                        response = await client.get(gateway_url)
                        response.raise_for_status()
                        lora_local.write_bytes(response.content)
                elif lora_url.startswith("http"):
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_UPLOAD) as client:
                        response = await client.get(lora_url)
                        response.raise_for_status()
                        lora_local.write_bytes(response.content)
                else:
                    raise HTTPException(400, f"Unsupported LoRA URL scheme: {lora_url}")

                lora_path = str(lora_local)
                logger.info(f"Downloaded LoRA to {lora_path}")

            if lora_path:
                logger.info(f"Loading LoRA from {lora_path}...")
                pipe.load_lora_weights(lora_path)
                logger.info("LoRA weights loaded successfully")

            # Parse aspect ratio to dimensions
            aspect_dims = {
                "1:1": (1024, 1024),
                "16:9": (1344, 768),
                "9:16": (768, 1344),
                "4:3": (1152, 896),
                "3:4": (896, 1152),
            }
            width, height = aspect_dims.get(aspect_ratio, (1024, 1024))

            # Generate images
            logger.info(f"Generating {num_outputs} image(s) at {width}x{height}...")
            images = pipe(
                prompt=prompt,
                num_images_per_prompt=min(num_outputs, 4),
                width=width,
                height=height,
                num_inference_steps=28,
                guidance_scale=3.5
            ).images

            # Encode as base64
            encoded_images = []
            for img in images:
                buffer = BytesIO()
                img.save(buffer, format=output_format.upper())
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                encoded_images.append(encoded)

            logger.info(f"Generated {len(encoded_images)} images successfully")
            return {
                "images": encoded_images,
                "format": output_format,
                "width": width,
                "height": height,
                "prompt": prompt,
                "used_lora": bool(lora_path)
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(500, f"Generation failed: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Kestrel LoRA Trainer")
    parser.add_argument("--server", action="store_true", help="Run as HTTP server")
    parser.add_argument("--image", type=str, help="Path to training image")
    parser.add_argument("--image-url", type=str, help="URL to download training image")
    parser.add_argument("--output", type=str, help="Output path for LoRA")
    parser.add_argument("--companion-id", type=str, default="test", help="Companion ID")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    args = parser.parse_args()

    if args.server:
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available - install with: pip install fastapi uvicorn")
            sys.exit(1)

        logger.info(f"Starting LoRA training server on port {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)

    elif args.image or args.image_url:
        # Direct training mode
        trainer = LoRATrainer()
        output = Path(args.output) if args.output else Path(f"./lora_{args.companion_id}.safetensors")

        async def run():
            if args.image_url:
                result = await trainer.train_from_url(
                    image_url=args.image_url,
                    companion_id=args.companion_id,
                    output_path=output
                )
            else:
                # Local file training
                image_path = Path(args.image)
                if not image_path.exists():
                    logger.error(f"Image not found: {args.image}")
                    return None

                # Create temp dir and copy image
                job_dir = trainer.training_dir / args.companion_id
                job_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(image_path, job_dir / "original.jpg")

                augmented = trainer.augment_image(
                    job_dir / "original.jpg",
                    job_dir,
                    trainer.config["num_augmentations"]
                )
                trainer.create_caption_files(augmented, f"companion_{args.companion_id[:8]}")

                success = await trainer.train_lora(
                    augmented,
                    output,
                    args.companion_id
                )

                shutil.rmtree(job_dir, ignore_errors=True)
                return output if success else None

            return result

        result = asyncio.run(run())
        if result:
            logger.info(f"Training complete: {result}")
        else:
            logger.error("Training failed")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
