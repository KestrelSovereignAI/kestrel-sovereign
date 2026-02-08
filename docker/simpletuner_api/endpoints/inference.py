"""
SimpleTuner Inference API Endpoints.

Provides REST API for image generation with trained LoRAs.
"""

import base64
import logging
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DOWNLOAD
from kestrel_sovereign.kestrel_config.defaults import get_lighthouse_gateway_url
from .. import config as app_config
from ..inference import (
    QUANTIZED_CACHE_DIR,
    QUANTIZED_MARKER_PATH,
    check_flux2_cached,
    generation_jobs,
    get_inference_pipeline,
    get_inference_pipeline_sync,
    is_pipeline_ready,
    is_quantized_model_cached,
    preload_status,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    lora_path: str = Form(...),
    trigger_word: str = Form("TOK"),
    num_outputs: int = Form(1),
    width: int = Form(1024),
    height: int = Form(1024),
    num_inference_steps: int = Form(20),
    guidance_scale: float = Form(3.5),
):
    """
    Generate images using FLUX.2-dev with a trained LoRA.

    Args:
        prompt: Generation prompt (should include trigger_word)
        lora_path: Path to LoRA safetensors file
        trigger_word: Trigger word used during training
        num_outputs: Number of images to generate (1-4)
        width: Image width (default 1024)
        height: Image height (default 1024)
        num_inference_steps: Denoising steps (default 20)
        guidance_scale: CFG scale (default 3.5)

    Returns:
        {"success": true, "images": [base64_image, ...]}
    """
    paths = app_config.get_runtime_paths()

    # Validate LoRA path
    lora_file = Path(lora_path)
    if not lora_file.exists():
        # Try runtime lora path
        lora_file = Path(f"{paths['lora_path']}/{lora_path}/pytorch_lora_weights.safetensors")
        if not lora_file.exists():
            raise HTTPException(404, f"LoRA not found: {lora_path}")

    try:
        # Get pipeline
        pipe = await get_inference_pipeline()

        # Load LoRA weights
        logger.info(f"Loading LoRA: {lora_file}")
        pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)

        # Ensure trigger word is in prompt
        if trigger_word and trigger_word not in prompt:
            prompt = f"a photo of {trigger_word}, {prompt}"

        logger.info(f"Generating {num_outputs} images: {prompt[:50]}...")

        # Generate
        images = []
        for i in range(min(num_outputs, 4)):
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            image = result.images[0]

            # Convert to base64
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode()
            images.append(f"data:image/png;base64,{b64}")

        # Unload LoRA to free memory for next request
        pipe.unload_lora_weights()

        logger.info(f"Generated {len(images)} images")

        return {
            "success": True,
            "images": images,
            "prompt": prompt,
            "lora_path": str(lora_file),
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")


@router.post("/generate/async")
async def generate_image_async(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    lora_path: str = Form(...),
    trigger_word: str = Form("TOK"),
    num_outputs: int = Form(1),
    width: int = Form(1024),
    height: int = Form(1024),
    num_inference_steps: int = Form(28),
    guidance_scale: float = Form(4.0),
    lora_ipfs_cid: Optional[str] = Form(None),
    ipfs_gateway: Optional[str] = Form(None),
):
    """
    Start async image generation. Returns job_id immediately.
    Poll /generate/status/{job_id} for results.

    This avoids Cloudflare's 100s timeout by running generation in background.

    LoRA Loading Priority:
    1. If lora_ipfs_cid provided: Download from IPFS gateway (preferred)
    2. Otherwise: Use lora_path (local or workspace path)
    """
    job_id = str(uuid.uuid4())
    paths = app_config.get_runtime_paths()

    generation_jobs[job_id] = {
        "status": "pending",
        "prompt": prompt,
        "lora_path": lora_path,
        "lora_ipfs_cid": lora_ipfs_cid,
        "images": [],
        "error": None,
    }

    def run_generation():
        import requests

        try:
            generation_jobs[job_id]["status"] = "loading_model"

            # Resolve LoRA path - IPFS takes priority
            if lora_ipfs_cid:
                # Download from IPFS gateway
                generation_jobs[job_id]["status"] = "downloading_lora"
                local_lora_dir = f"{paths['tmp_path']}/ipfs_lora_{job_id}"
                os.makedirs(local_lora_dir, exist_ok=True)
                local_lora_file = f"{local_lora_dir}/pytorch_lora_weights.safetensors"

                ipfs_url = f"{ipfs_gateway}/{lora_ipfs_cid}"
                logger.info(f"Downloading LoRA from IPFS: {ipfs_url}")

                response = requests.get(ipfs_url, timeout=HTTP_TIMEOUT_DOWNLOAD, stream=True)
                response.raise_for_status()

                with open(local_lora_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = os.path.getsize(local_lora_file)
                logger.info(f"Downloaded LoRA from IPFS ({file_size / 1024 / 1024:.1f} MB)")
                lora_file = Path(local_lora_file)
            else:
                # Use local path
                lora_file = Path(lora_path)
                if not lora_file.exists():
                    lora_file = Path(f"{paths['lora_path']}/{lora_path}/pytorch_lora_weights.safetensors")
                    if not lora_file.exists():
                        generation_jobs[job_id]["status"] = "failed"
                        generation_jobs[job_id]["error"] = f"LoRA not found: {lora_path}"
                        return

            # Get pipeline using sync version (avoids asyncio event loop issues)
            # Background threads must use get_inference_pipeline_sync() not async version
            pipe = get_inference_pipeline_sync()

            generation_jobs[job_id]["status"] = "loading_lora"
            pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)

            # Ensure trigger word in prompt
            gen_prompt = prompt
            if trigger_word and trigger_word not in prompt:
                gen_prompt = f"a photo of {trigger_word}, {prompt}"

            generation_jobs[job_id]["status"] = "generating"

            images = []
            for i in range(min(num_outputs, 4)):
                result = pipe(
                    prompt=gen_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                image = result.images[0]

                buffer = BytesIO()
                image.save(buffer, format="PNG")
                b64 = base64.b64encode(buffer.getvalue()).decode()
                images.append(f"data:image/png;base64,{b64}")

            pipe.unload_lora_weights()

            generation_jobs[job_id]["status"] = "completed"
            generation_jobs[job_id]["images"] = images
            logger.info(f"Async generation {job_id} completed: {len(images)} images")

        except Exception as e:
            logger.error(f"Async generation {job_id} failed: {e}")
            generation_jobs[job_id]["status"] = "failed"
            generation_jobs[job_id]["error"] = str(e)

    background_tasks.add_task(run_generation)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Generation started. Poll /generate/status/{job_id} for results.",
    }


@router.get("/generate/status/{job_id}")
async def get_generation_status(job_id: str):
    """Get status of async generation job."""
    if job_id not in generation_jobs:
        raise HTTPException(404, "Job not found")

    job = generation_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "images": job["images"] if job["status"] == "completed" else [],
        "error": job["error"],
    }


@router.post("/preload")
async def preload_model(background_tasks: BackgroundTasks, quantize: bool = True):
    """
    Pre-download and optionally pre-quantize FLUX.2-dev model.

    Args:
        quantize: If True, also quantize and cache the model (saves ~10 min on future loads)

    Workflow:
    1. Downloads FLUX.2-dev model to HF_HOME (~60GB)
    2. If quantize=True: Loads and quantizes to int8, saves to /workspace/quantized_flux/
    3. Future inference loads from quantized cache (2-3 min vs 10-15 min)
    """
    global preload_status

    if preload_status["status"] == "running":
        return {"status": "already_running", "progress": preload_status["progress"]}

    # Check if quantized cache exists (best case)
    if quantize and is_quantized_model_cached():
        preload_status.update({"status": "complete", "progress": "Quantized model already cached!", "error": None})
        logger.info("Quantized FLUX.2-dev already cached, skipping")
        return {
            "status": "already_cached",
            "message": "Quantized FLUX.2-dev model is already cached on disk",
            "cache_path": QUANTIZED_CACHE_DIR,
        }

    # Check if base model is cached (partial case)
    if check_flux2_cached() and not quantize:
        preload_status.update({"status": "complete", "progress": "Model already cached!", "error": None})
        logger.info("FLUX.2-dev already cached, skipping download")
        return {"status": "already_cached", "message": "FLUX.2-dev model is already cached"}

    def download_and_quantize():
        try:
            # Step 1: Download base model if needed
            if not check_flux2_cached():
                preload_status.update({"status": "running", "progress": "Downloading FLUX.2-dev (~60GB)...", "error": None})
                from huggingface_hub import snapshot_download
                logger.info("Starting FLUX.2-dev download...")
                snapshot_download(
                    "black-forest-labs/FLUX.2-dev",
                    local_dir_use_symlinks=True
                )
                logger.info("FLUX.2-dev download complete!")

            # Step 2: Quantize and cache if requested
            if quantize and not is_quantized_model_cached():
                preload_status.update({"status": "running", "progress": "Quantizing model (10-15 min)...", "error": None})
                logger.info("Starting quantization...")

                # Use asyncio to call the async function
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(get_inference_pipeline(save_quantized=True))
                loop.close()

                preload_status.update({"status": "complete", "progress": "Quantized model cached!", "error": None})
                logger.info("Quantization and caching complete!")
            else:
                preload_status.update({"status": "complete", "progress": "Download complete!", "error": None})

        except Exception as e:
            preload_status.update({"status": "failed", "progress": "", "error": str(e)})
            logger.error(f"Preload failed: {e}")
            import traceback
            traceback.print_exc()

    background_tasks.add_task(download_and_quantize)

    return {
        "status": "started",
        "message": f"Model {'download + quantization' if quantize else 'download'} started in background",
        "quantize": quantize,
    }


@router.post("/preload/quantize")
async def preload_quantize(background_tasks: BackgroundTasks):
    """
    Force quantize and cache the model, even if base model is already downloaded.

    Use this if:
    - Base model is downloaded but not quantized
    - Previous quantization failed
    - You want to refresh the quantized cache
    """
    if preload_status["status"] == "running":
        return {"status": "already_running", "progress": preload_status["progress"]}

    if is_pipeline_ready():
        return {
            "status": "already_loaded",
            "message": "Pipeline already loaded in memory. No need to preload.",
        }

    def quantize_model():
        try:
            preload_status.update({"status": "running", "progress": "Quantizing model (10-15 min)...", "error": None})
            logger.info("Starting forced quantization...")

            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(get_inference_pipeline(save_quantized=True))
            loop.close()

            preload_status.update({"status": "complete", "progress": "Quantized model cached!", "error": None})
            logger.info("Quantization and caching complete!")

        except Exception as e:
            preload_status.update({"status": "failed", "progress": "", "error": str(e)})
            logger.error(f"Quantization failed: {e}")
            import traceback
            traceback.print_exc()

    background_tasks.add_task(quantize_model)

    return {
        "status": "started",
        "message": "Model quantization started in background. Check /preload/status for progress.",
    }


@router.get("/preload/status")
async def get_preload_status():
    """
    Check status of model preload and quantized model cache.

    Returns:
        - status: Current preload status (idle, running, complete, failed)
        - cached: Whether quantized model is cached on disk
        - cache_path: Path to quantized model cache
        - cache_size_gb: Size of cache in GB (if cached)
        - pipeline_loaded: Whether pipeline is loaded in memory
    """
    result = dict(preload_status)

    # Add cache status
    result["cached"] = is_quantized_model_cached()
    result["cache_path"] = QUANTIZED_CACHE_DIR
    result["pipeline_loaded"] = is_pipeline_ready()

    # Add cache size if cached
    if result["cached"]:
        try:
            cache_size_bytes = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, filenames in os.walk(QUANTIZED_CACHE_DIR)
                for f in filenames
            )
            result["cache_size_gb"] = round(cache_size_bytes / (1024**3), 2)

            # Read marker file for metadata
            if os.path.exists(QUANTIZED_MARKER_PATH):
                with open(QUANTIZED_MARKER_PATH, "r") as f:
                    result["cache_metadata"] = f.read().strip()
        except Exception as e:
            result["cache_size_error"] = str(e)

    return result


@router.get("/loras")
async def list_loras():
    """
    List available trained LoRAs.

    Returns:
        {"loras": [{"id": "test-avatar-001", "path": "...", "size_mb": 104}, ...]}
    """
    paths = app_config.get_runtime_paths()
    loras = []
    lora_dir = Path(paths["lora_path"])

    if lora_dir.exists():
        for entry in lora_dir.iterdir():
            if entry.is_dir():
                lora_file = entry / "pytorch_lora_weights.safetensors"
                if lora_file.exists():
                    loras.append({
                        "id": entry.name,
                        "path": str(lora_file),
                        "size_mb": lora_file.stat().st_size / (1024 * 1024),
                    })

    return {"loras": loras, "count": len(loras)}
