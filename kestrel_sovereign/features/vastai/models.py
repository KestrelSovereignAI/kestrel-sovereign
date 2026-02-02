"""
Vast.ai Data Models and Exceptions.

Contains dataclasses, enums, and exception classes for Vast.ai integration.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class InstanceStatus(Enum):
    """Lifecycle states for Vast.ai instances."""

    OFFLINE = "offline"
    CREATING = "creating"
    LOADING = "loading"
    RUNNING = "running"
    STOPPING = "stopping"
    EXITED = "exited"
    ERROR = "error"


@dataclass
class GPUProfile:
    """GPU profile configuration loaded from vastai_config.toml."""

    id: str
    name: str
    task_type: str
    image_name: str
    disk_gb: int
    gpu_ram_min: int  # Minimum GPU RAM in GB
    num_gpus: int = 1
    reliability_min: float = 0.9
    compute_cap_min: int = 0  # 0 = any, 800 = Ampere+
    cuda_vers_min: float = 11.0
    ports: List[str] = field(default_factory=lambda: ["8888/http"])
    inference_port: int = 8888
    inference_protocol: str = "http"
    inference_base_path: str = "/v1"
    onstart_cmd: Optional[str] = None
    default_model: Optional[str] = None
    cost_per_hr_max: Optional[float] = None  # Max $/hr willing to pay
    env: Dict[str, str] = field(default_factory=dict)
    # Docker registry auth: format "-u <user> -p <pass> <registry>"
    docker_login: Optional[str] = None


@dataclass
class VastAISession:
    """Tracks an active Vast.ai instance session."""

    instance_id: int
    profile: GPUProfile
    task_profile: str
    model_name: str
    status: InstanceStatus
    ttl_seconds: int
    started_at: datetime
    expires_at: datetime
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    backend_base_url: Optional[str] = None
    inference_url: Optional[str] = None
    actual_cost_per_hr: Optional[float] = None
    gpu_name: Optional[str] = None
    runtime: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "profile": self.profile.id,
            "task_profile": self.task_profile,
            "model_name": self.model_name,
            "status": self.status.value,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "backend_base_url": self.backend_base_url,
            "inference_url": self.inference_url,
            "ttl_seconds": self.ttl_seconds,
            "remaining_ttl_seconds": self.remaining_ttl_seconds,
            "actual_cost_per_hr": self.actual_cost_per_hr,
            "gpu_name": self.gpu_name,
            "runtime": self.runtime,
        }

    @property
    def remaining_ttl_seconds(self) -> int:
        delta = (self.expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(delta))

    @property
    def is_active(self) -> bool:
        return self.status in {InstanceStatus.CREATING, InstanceStatus.LOADING, InstanceStatus.RUNNING}


class VastAIManagerError(Exception):
    """Custom exception for Vast.ai manager failures."""
