"""
Ollama GPU Manager - Cloud Ollama on RunPod/Vast.ai

This module provides management of Ollama instances running on cloud GPUs
for users without local GPU hardware. It reuses the existing RunPod
infrastructure from the training system.

Key Features:
- Pod lifecycle management (start, stop, resume)
- Persistent model storage via network volumes
- Integration with LLMService.switch_backend()
- Cold start mitigation via persistent pods

Usage:
    from kestrel_sovereign.features.ollama import OllamaGPUManager

    manager = OllamaGPUManager()
    session = await manager.start_session(profile="ollama")

    # LLM service will now route to remote Ollama
    from kestrel_sovereign.llm.service import LLMService
    llm = LLMService()
    llm.switch_backend("remote_gpu", {
        "base_url": session.inference_url,
        "model": "qwen2.5:7b",
    })
"""

from .ollama_manager import OllamaGPUManager, OllamaSession, OllamaGPUManagerError
from .ollama_gpu_adapter import (
    OllamaGPUAdapter,
    start_remote_ollama,
    stop_remote_ollama,
)

__all__ = [
    "OllamaGPUManager",
    "OllamaSession",
    "OllamaGPUManagerError",
    "OllamaGPUAdapter",
    "start_remote_ollama",
    "stop_remote_ollama",
]
