"""
RunPod GPU management for Kestrel.

Modular structure for the RunPod GPU instance manager:
- models.py: Data models, enums, exceptions
- providers.py: GPU provider abstractions (direct, managed proxy)
- core.py: Core SDK operations, profile loading, session management
- training.py: LoRA training methods (HTTP API)
- ollama.py: Ollama cloud server methods
- manager.py: Combined RunPodManager class
- feature.py: Kestrel feature integration

Usage:
    from kestrel_sovereign.features.runpod import RunPodManager, RunPodFeature

    # Direct manager usage
    manager = RunPodManager()

    # LoRA Training workflow
    session = await manager.start_training_pod("companion-123")
    job_id = await manager.submit_training_job(session, avatar_data, "companion-123")
    status = await manager.poll_training_status(session, job_id)
    lora_data = await manager.download_lora(session, job_id)

    # Ollama Cloud workflow
    session = await manager.start_ollama_pod(["qwen2.5:7b"])
    base_url = await manager.get_ollama_base_url()

    # Or as a Kestrel feature
    feature = RunPodFeature(agent)
"""

from .feature import RunPodFeature
from .manager import RunPodManager
from .models import (
    GPUProfile,
    PodStatus,
    RunPodManagerError,
    RunPodSession,
)

__all__ = [
    "RunPodFeature",
    "RunPodManager",
    "RunPodManagerError",
    "RunPodSession",
    "PodStatus",
    "GPUProfile",
]
