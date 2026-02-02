"""
Training Provider Adapters

Each adapter wraps an existing manager class to implement the
TrainingProvider protocol.

Available Adapters:
- LocalMPSTrainingAdapter: Local Apple Silicon with MPS backend
- VertexAITrainingAdapter: Wraps VertexAIManager (serverless)
- RunPodTrainingAdapter: Wraps RunPodManager (session-based with persistent pods)
- GCPComputeTrainingAdapter: Wraps GCPComputeManager (session-based)
- ReplicateTrainingAdapter: Wraps Replicate API directly (serverless)
- VastAITrainingAdapter: Wraps VastAIManager (session-based)
"""

from .local_mps_adapter import LocalMPSTrainingAdapter
from .vertex_ai_adapter import VertexAITrainingAdapter
from .gcp_compute_adapter import GCPComputeTrainingAdapter
from .replicate_adapter import ReplicateTrainingAdapter
from .vastai_adapter import VastAITrainingAdapter
from .runpod_adapter import RunPodTrainingAdapter

__all__ = [
    "LocalMPSTrainingAdapter",
    "VertexAITrainingAdapter",
    "GCPComputeTrainingAdapter",
    "ReplicateTrainingAdapter",
    "VastAITrainingAdapter",
    "RunPodTrainingAdapter",
]
