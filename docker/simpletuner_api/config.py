"""
SimpleTuner API Configuration and Path Management.

Handles runtime path configuration for different environments:
- RunPod: Requires network volume at /workspace
- Vertex AI: Uses standard container paths
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# RunPod network volume mount point - REQUIRED, no alternatives
WORKSPACE_PATH = "/workspace"

# Global path configuration (set by setup_paths at startup)
_runtime_paths = {
    "base_path": WORKSPACE_PATH,
    "hf_path": f"{WORKSPACE_PATH}/huggingface",
    "tmp_path": f"{WORKSPACE_PATH}/tmp",
    "lora_path": f"{WORKSPACE_PATH}/trained_loras",
    "output_path": f"{WORKSPACE_PATH}/output",
    "datasets_path": f"{WORKSPACE_PATH}/datasets",
}

# Training state
training_jobs: dict = {}
current_job: Optional[str] = None


def get_runtime_paths() -> dict:
    """Get current runtime paths configuration."""
    return _runtime_paths.copy()


def get_lora_path() -> str:
    """Get the path for trained LoRAs."""
    return _runtime_paths["lora_path"]


def get_output_path() -> str:
    """Get the output path."""
    return _runtime_paths["output_path"]


def get_datasets_path() -> str:
    """Get the datasets path."""
    return _runtime_paths["datasets_path"]


def get_hf_path() -> str:
    """Get HuggingFace cache path."""
    return _runtime_paths["hf_path"]


def get_tmp_path() -> str:
    """Get temporary files path."""
    return _runtime_paths["tmp_path"]


def setup_paths(is_vertex_mode: bool = False) -> dict:
    """
    Configure paths based on runtime environment.

    Vertex AI: Uses standard container paths (/app, /tmp)
    RunPod: REQUIRES network volume at /workspace - FAILS if not mounted

    Args:
        is_vertex_mode: True for Vertex AI environment, False for RunPod

    Returns:
        Dictionary of configured paths

    Raises:
        RuntimeError: If RunPod mode and /workspace is not properly mounted
    """
    global _runtime_paths

    if is_vertex_mode:
        # Vertex AI: use standard Linux paths within the container
        base_path = "/app"
        tmp_path = "/tmp"
        logger.info("Configuring paths for VERTEX AI mode")

        _runtime_paths = {
            "base_path": base_path,
            "hf_path": f"{base_path}/huggingface",
            "tmp_path": tmp_path,
            "lora_path": f"{base_path}/trained_loras",
            "output_path": f"{base_path}/output",
            "datasets_path": f"{base_path}/datasets",
        }
    else:
        # RunPod: REQUIRE network volume at /workspace
        # NO FALLBACKS - fail fast if not properly configured
        if not os.path.isdir(WORKSPACE_PATH):
            raise RuntimeError(
                f"FATAL: {WORKSPACE_PATH} does not exist. "
                f"RunPod network volume MUST be mounted at {WORKSPACE_PATH}. "
                f"Check pod configuration."
            )

        if not os.path.ismount(WORKSPACE_PATH):
            raise RuntimeError(
                f"FATAL: {WORKSPACE_PATH} is not a mount point. "
                f"RunPod network volume MUST be mounted at {WORKSPACE_PATH}. "
                f"This is NOT a directory on the container disk. "
                f"Check pod configuration - volume_mount_path must be {WORKSPACE_PATH}."
            )

        logger.info(f"Network volume verified at {WORKSPACE_PATH}")

        # All paths under /workspace for persistence
        _runtime_paths = {
            "base_path": WORKSPACE_PATH,
            "hf_path": f"{WORKSPACE_PATH}/huggingface",
            "tmp_path": f"{WORKSPACE_PATH}/tmp",
            "lora_path": f"{WORKSPACE_PATH}/trained_loras",
            "output_path": f"{WORKSPACE_PATH}/output",
            "datasets_path": f"{WORKSPACE_PATH}/datasets",
        }

        logger.info(f"Configuring paths for RUNPOD mode with base: {WORKSPACE_PATH}")

    # Create all directories
    for key, path in _runtime_paths.items():
        if key != "base_path":  # Don't try to create the base mount point
            os.makedirs(path, exist_ok=True)

    # Set environment variables for HuggingFace/PyTorch
    os.environ["HF_HOME"] = _runtime_paths["hf_path"]
    os.environ["TRANSFORMERS_CACHE"] = _runtime_paths["hf_path"]
    os.environ["TORCH_HOME"] = _runtime_paths["hf_path"]
    os.environ["TMPDIR"] = _runtime_paths["tmp_path"]
    os.environ["TEMP"] = _runtime_paths["tmp_path"]
    os.environ["TMP"] = _runtime_paths["tmp_path"]

    logger.info(f"Paths configured: {_runtime_paths}")
    return _runtime_paths
