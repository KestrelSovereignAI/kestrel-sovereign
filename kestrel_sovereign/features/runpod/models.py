"""
RunPod Data Models and Exceptions.

Contains dataclasses, enums, and exception classes for RunPod integration.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class PodStatus(Enum):
    """Lifecycle states returned by RunPod."""

    OFFLINE = "offline"
    PROVISIONING = "provisioning"
    LOADING = "loading"
    READY = "ready"
    TERMINATING = "terminating"
    ERROR = "error"


@dataclass
class GPUProfile:
    """Represents a GPU profile configuration loaded from runpod_config.toml."""

    id: str
    name: str
    task_type: str
    gpu_type_id: str
    image_name: str
    container_disk_gb: int
    volume_gb: int
    ports: List[str]
    inference_port: int
    inference_protocol: str = "http"
    inference_base_path: str = "/v1"
    image_invoke_path: Optional[str] = None
    default_model: Optional[str] = None
    pod_type: Optional[str] = None
    vram_gb: Optional[int] = None
    cost_per_hr: Optional[float] = None
    max_context_window: Optional[int] = None
    readiness_timeout_seconds: Optional[int] = None
    template_id: Optional[str] = None  # RunPod template with registry auth
    network_volume_id: Optional[str] = None  # Network volume ID for persistent storage
    volume_mount_path: Optional[str] = None  # Mount path for network volume (e.g., /workspace)
    persistent_pod_id: Optional[str] = None  # Use existing pod instead of creating new (resume/pause mode)
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunPodSession:
    """Tracks the currently active RunPod session."""

    pod_id: str
    profile: GPUProfile
    task_profile: str
    model_name: str
    pod_type: Optional[str]
    status: PodStatus
    ttl_seconds: int
    started_at: datetime
    expires_at: datetime
    backend_base_url: Optional[str] = None
    inference_url: Optional[str] = None
    image_endpoint: Optional[str] = None
    runtime: Dict[str, Any] = field(default_factory=dict)
    # Metering context (optional, for usage billing)
    companion_id: Optional[str] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "profile": self.profile.id,
            "task_profile": self.task_profile,
            "model_name": self.model_name,
            "status": self.status.value,
            "backend_base_url": self.backend_base_url,
            "inference_url": self.inference_url,
            "image_endpoint": self.image_endpoint,
            "ttl_seconds": self.ttl_seconds,
            "remaining_ttl_seconds": self.remaining_ttl_seconds,
            "cost_per_hr": self.profile.cost_per_hr,
            "vram_gb": self.profile.vram_gb,
            "runtime": self.runtime,
        }

    @property
    def remaining_ttl_seconds(self) -> int:
        delta = (self.expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(delta))

    @property
    def is_active(self) -> bool:
        return self.status not in {PodStatus.OFFLINE, PodStatus.TERMINATING, PodStatus.ERROR}


class RunPodManagerError(Exception):
    """Custom exception for manager failures."""
