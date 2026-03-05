"""
SimpleTuner Training API Endpoints.

Provides REST API for LoRA training management.
"""

import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .. import config as app_config
from ..training import create_simpletuner_config, run_training

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    paths = app_config.get_runtime_paths()
    return {
        "status": "healthy",
        "service": "simpletuner-flux2",
        "current_job": app_config.current_job,
        "jobs_count": len(app_config.training_jobs),
        "workspace_path": paths["base_path"],
    }


@router.get("/debug/logs/{job_id}")
async def get_training_logs(job_id: str, lines: int = 100):
    """
    Get training logs for a job.

    Returns:
        - SimpleTuner debug.log (verbose training output)
        - TensorBoard logs directory listing
        - Last N lines of output
    """
    if job_id not in app_config.training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = app_config.training_jobs[job_id]
    output_path = f"{app_config.get_output_path()}/{job_id}"

    logs = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "debug_log": None,
        "tensorboard_events": [],
        "checkpoints": [],
    }

    # Read SimpleTuner's debug.log (may be in /app or output dir)
    for debug_log_path in [Path("/app/debug.log"), Path(output_path) / "logs" / "debug.log"]:
        if debug_log_path.exists():
            try:
                with open(debug_log_path, "r") as f:
                    all_lines = f.readlines()
                    logs["debug_log"] = "".join(all_lines[-lines:])
                break
            except Exception as e:
                logs["debug_log_error"] = str(e)

    # List TensorBoard event files
    logs_dir = Path(output_path) / "logs"
    if logs_dir.exists():
        for f in logs_dir.glob("events.out.tfevents.*"):
            logs["tensorboard_events"].append({
                "name": f.name,
                "size_kb": f.stat().st_size / 1024,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })

    # List checkpoints
    output_dir = Path(output_path)
    if output_dir.exists():
        for checkpoint in output_dir.glob("checkpoint-*"):
            if checkpoint.is_dir():
                lora_file = checkpoint / "pytorch_lora_weights.safetensors"
                logs["checkpoints"].append({
                    "name": checkpoint.name,
                    "has_lora": lora_file.exists(),
                    "lora_size_mb": lora_file.stat().st_size / (1024*1024) if lora_file.exists() else None,
                })

    return logs


@router.get("/ready")
async def ready_check():
    """
    Readiness check endpoint.

    Returns 200 when the service is ready to accept training jobs.
    Used by the Training API to wait for model loading before submitting jobs.

    Note: SimpleTuner loads the model on-demand during training,
    so we just check if the service is up and not currently training.
    """
    if app_config.current_job is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Training in progress: {app_config.current_job}"
        )

    return {
        "status": "ready",
        "service": "simpletuner-flux2",
        "can_accept_job": True,
    }


@router.post("/train")
async def start_training(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    companion_id: str = Form(...),
    trigger_word: Optional[str] = Form(None),
    steps: int = Form(1000),
    lora_rank: int = Form(16),
    callback_url: Optional[str] = Form(None),
):
    """
    Start LoRA training for a companion.

    Args:
        image: Training image (avatar)
        companion_id: Companion UUID
        trigger_word: Custom trigger word (default: TOK{companion_id[:8]})
        steps: Training steps (default: 1000)
        lora_rank: LoRA rank (default: 16)
        callback_url: Optional webhook for completion

    Returns:
        Job ID and status
    """
    if app_config.current_job is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Training already in progress: {app_config.current_job}"
        )

    job_id = str(uuid.uuid4())

    # Generate trigger word if not provided
    if not trigger_word:
        clean_id = companion_id.replace("-", "")[:8]
        trigger_word = f"TOK{clean_id}"

    # Create dataset directory with the image - use configured paths
    dataset_path = f"{app_config.get_datasets_path()}/{job_id}"
    output_path = f"{app_config.get_output_path()}/{job_id}"
    os.makedirs(dataset_path, exist_ok=True)

    # Save uploaded image
    image_path = f"{dataset_path}/image_001.jpg"
    async with aiofiles.open(image_path, "wb") as f:
        content = await image.read()
        await f.write(content)

    logger.info(f"Saved training image: {image_path} ({len(content)} bytes)")

    # Create job record
    app_config.training_jobs[job_id] = {
        "job_id": job_id,
        "companion_id": companion_id,
        "trigger_word": trigger_word,
        "status": "queued",
        "progress": 0.0,
        "current_step": 0,
        "total_steps": steps,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "lora_path": None,
        "error": None,
        "callback_url": callback_url,
    }

    # Create SimpleTuner config
    cache_path = f"{output_path}/cache"
    config = create_simpletuner_config(
        job_id=job_id,
        trigger_word=trigger_word,
        dataset_path=dataset_path,
        output_path=output_path,
        cache_path=cache_path,
        steps=steps,
        lora_rank=lora_rank,
    )

    # Start training in background
    background_tasks.add_task(run_training, job_id, config, dataset_path)

    return {
        "job_id": job_id,
        "status": "queued",
        "companion_id": companion_id,
        "trigger_word": trigger_word,
        "estimated_minutes": steps // 60 + 10,  # Rough estimate
    }


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get training job status."""
    if job_id not in app_config.training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = app_config.training_jobs[job_id]
    return {
        "job_id": job["job_id"],
        "companion_id": job["companion_id"],
        "trigger_word": job["trigger_word"],
        "status": job["status"],
        "progress": job["progress"],
        "current_step": job["current_step"],
        "total_steps": job["total_steps"],
        "error": job["error"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "lora_path": job.get("lora_path"),
    }


@router.get("/download/{job_id}")
async def download_lora(job_id: str):
    """Download trained LoRA weights."""
    if job_id not in app_config.training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = app_config.training_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Training not complete. Status: {job['status']}"
        )

    if not job["lora_path"] or not os.path.exists(job["lora_path"]):
        raise HTTPException(status_code=404, detail="LoRA file not found")

    return FileResponse(
        job["lora_path"],
        media_type="application/octet-stream",
        filename=f"{job['companion_id']}.safetensors"
    )


@router.get("/current-job")
async def get_current_job():
    """
    Get information about the currently running job.

    Returns:
        Job info if training in progress, or None
    """
    if app_config.current_job is None:
        return {"current_job": None, "status": "idle"}

    job = app_config.training_jobs.get(app_config.current_job, {})
    return {
        "current_job": app_config.current_job,
        "status": job.get("status", "unknown"),
        "progress": job.get("progress", 0.0),
        "current_step": job.get("current_step", 0),
        "total_steps": job.get("total_steps", 0),
        "companion_id": job.get("companion_id"),
        "started_at": job.get("started_at"),
    }


@router.post("/cancel/{job_id}")
async def cancel_training(job_id: str):
    """
    Cancel a running training job.

    Note: This is a best-effort cancellation. SimpleTuner doesn't have
    native cancellation support, so we mark the job as cancelled and
    the next status check will reflect this.

    For truly stopping a stuck job, the pod may need to be restarted.
    """
    if job_id not in app_config.training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = app_config.training_jobs[job_id]

    if job["status"] in ("completed", "failed", "cancelled"):
        return {
            "job_id": job_id,
            "status": job["status"],
            "message": f"Job already {job['status']}, cannot cancel"
        }

    # Mark as cancelled
    job["status"] = "cancelled"
    job["error"] = "Cancelled by user"
    job["completed_at"] = datetime.utcnow().isoformat()

    # Clear current job if this was it
    if app_config.current_job == job_id:
        app_config.current_job = None

    logger.info(f"Training job {job_id} marked as cancelled")

    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job marked as cancelled. Note: The training process may still be running. Restart pod if needed."
    }


@router.post("/clear-current-job")
async def clear_current_job():
    """
    Force-clear the current job lock.

    USE WITH CAUTION: This should only be used when a job is stuck
    and the training process has died or is unresponsive.

    This does NOT kill any running processes - it just clears the lock
    so new jobs can be submitted.
    """
    old_job = app_config.current_job
    app_config.current_job = None

    if old_job:
        if old_job in app_config.training_jobs:
            job = app_config.training_jobs[old_job]
            if job["status"] not in ("completed", "failed", "cancelled"):
                job["status"] = "failed"
                job["error"] = "Force-cleared by admin"
                job["completed_at"] = datetime.utcnow().isoformat()

        logger.warning(f"Force-cleared current job lock: {old_job}")
        return {
            "cleared_job": old_job,
            "message": "Current job lock cleared. New training can be submitted."
        }

    return {
        "cleared_job": None,
        "message": "No job was running."
    }


@router.get("/loras")
async def list_loras():
    """List all trained LoRAs on the network volume."""
    lora_path = app_config.get_lora_path()
    loras = []

    if os.path.exists(lora_path):
        for item in os.listdir(lora_path):
            item_path = os.path.join(lora_path, item)
            if os.path.isdir(item_path):
                lora_file = os.path.join(item_path, "pytorch_lora_weights.safetensors")
                if os.path.exists(lora_file):
                    stat = os.stat(lora_file)
                    loras.append({
                        "companion_id": item,
                        "size_mb": stat.st_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })

    return {"loras": loras, "count": len(loras)}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a training job and its associated files."""
    if job_id not in app_config.training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = app_config.training_jobs[job_id]

    if job["status"] == "training":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a job that is currently training. Cancel it first."
        )

    # Delete associated files
    paths_to_delete = [
        f"{app_config.get_output_path()}/{job_id}",
        f"{app_config.get_datasets_path()}/{job_id}",
    ]

    deleted_paths = []
    for path in paths_to_delete:
        if os.path.exists(path):
            shutil.rmtree(path)
            deleted_paths.append(path)

    # Remove job record
    del app_config.training_jobs[job_id]

    return {
        "job_id": job_id,
        "deleted": True,
        "deleted_paths": deleted_paths,
    }
