"""
RunPod Manager - Combined Class.

Combines all RunPod functionality from the mixin classes
into a single manager class.
"""

from typing import Any, Dict, Optional

from .core import RunPodManagerCore
from .ollama import RunPodOllamaMixin
from .training import RunPodTrainingMixin


class RunPodManager(
    RunPodManagerCore,
    RunPodTrainingMixin,
    RunPodOllamaMixin,
):
    """
    Full RunPod GPU instance manager.

    Combines:
    - RunPodManagerCore: SDK operations, profile loading, session management
    - RunPodTrainingMixin: LoRA training methods
    - RunPodOllamaMixin: Ollama cloud server methods

    Usage:
        manager = RunPodManager()

        # LoRA Training
        session = await manager.start_training_pod("companion-123")
        job_id = await manager.submit_training_job(session, avatar_data, "companion-123")
        status = await manager.poll_training_status(session, job_id)
        lora_data = await manager.download_lora(session, job_id)

        # Ollama Cloud
        session = await manager.start_ollama_pod(["phi4"])
        base_url = await manager.get_ollama_base_url()
        # Use base_url with OllamaAdapter

        # Cleanup
        await manager.terminate_session(session)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, mode: Optional[str] = None):
        """Initialize the RunPod manager."""
        super().__init__(config, mode)
