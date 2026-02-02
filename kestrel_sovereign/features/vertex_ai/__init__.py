"""
Vertex AI Feature for Kestrel.

Provides serverless GPU training via Google Cloud Vertex AI Custom Jobs.
Key advantages over VM-based approaches (RunPod, GCP Compute):
- No pod/instance lifecycle management
- No TTL issues - jobs run to completion
- Pay only for actual compute time
- Automatic retry and error handling
"""

from .vertex_ai_manager import VertexAIManager, VertexAITrainingJob

__all__ = ["VertexAIManager", "VertexAITrainingJob"]
