"""
GCP Compute Engine Data Models and Exceptions.

Contains dataclasses, enums, and exception classes for GCP Compute integration.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional


class InstanceStatus(Enum):
    """Lifecycle states for GCP Compute instances."""

    OFFLINE = "offline"
    PROVISIONING = "provisioning"
    STAGING = "staging"
    RUNNING = "running"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"
    ERROR = "error"

    @classmethod
    def from_gcp_status(cls, status: str) -> "InstanceStatus":
        """Convert GCP instance status to our enum."""
        mapping = {
            "PROVISIONING": cls.PROVISIONING,
            "STAGING": cls.STAGING,
            "RUNNING": cls.RUNNING,
            "STOPPING": cls.STOPPING,
            "TERMINATED": cls.TERMINATED,
            "SUSPENDED": cls.SUSPENDED,
        }
        return mapping.get(status, cls.ERROR)


@dataclass
class GPUProfile:
    """GPU profile configuration loaded from gcp_compute_config.toml."""

    id: str
    name: str
    task_type: str
    machine_type: str
    gpu_type: str
    gpu_count: int
    boot_disk_size_gb: int
    image_name: str
    boot_disk_type: str = "pd-ssd"
    persistent_disk: Optional[str] = None
    inference_port: int = 8000
    inference_protocol: str = "http"
    inference_base_path: str = ""
    default_model: Optional[str] = None
    cost_per_hr_on_demand: Optional[float] = None
    cost_per_hr_spot: Optional[float] = None
    readiness_timeout_seconds: int = 600
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class GCPComputeSession:
    """Tracks an active GCP Compute instance session."""

    instance_name: str
    zone: str
    profile: GPUProfile
    task_profile: str
    model_name: Optional[str]
    status: InstanceStatus
    ttl_seconds: int
    started_at: datetime
    expires_at: datetime
    is_spot: bool = False
    external_ip: Optional[str] = None
    internal_ip: Optional[str] = None
    backend_base_url: Optional[str] = None
    inference_url: Optional[str] = None
    actual_cost_per_hr: Optional[float] = None
    persistent_disk_attached: bool = False
    ssh_host: Optional[str] = None
    ssh_port: int = 22
    runtime: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_name": self.instance_name,
            "zone": self.zone,
            "profile": self.profile.id,
            "task_profile": self.task_profile,
            "model_name": self.model_name,
            "status": self.status.value,
            "is_spot": self.is_spot,
            "external_ip": self.external_ip,
            "internal_ip": self.internal_ip,
            "backend_base_url": self.backend_base_url,
            "inference_url": self.inference_url,
            "ttl_seconds": self.ttl_seconds,
            "remaining_ttl_seconds": self.remaining_ttl_seconds,
            "actual_cost_per_hr": self.actual_cost_per_hr,
            "persistent_disk_attached": self.persistent_disk_attached,
            "ssh_host": self.ssh_host,
            "runtime": self.runtime,
        }

    @property
    def remaining_ttl_seconds(self) -> int:
        delta = (self.expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(delta))

    @property
    def is_active(self) -> bool:
        return self.status in {
            InstanceStatus.PROVISIONING,
            InstanceStatus.STAGING,
            InstanceStatus.RUNNING,
        }


class GCPComputeManagerError(Exception):
    """Custom exception for GCP Compute manager failures."""
