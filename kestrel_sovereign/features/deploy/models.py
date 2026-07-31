"""
Deploy Data Models and Exceptions.

Contains dataclasses, enums, and exception classes for agent self-deployment.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class DeployStatus(Enum):
    """Lifecycle states for deployed agent services."""

    OFFLINE = "offline"
    BUILDING = "building"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    TEARING_DOWN = "tearing_down"
    TERMINATED = "terminated"


class DeployProviderType(Enum):
    """Cloud deployment provider types."""

    CLOUD_RUN = "cloud_run"
    AZURE_CONTAINER_APPS = "azure_container_apps"


@dataclass
class DeploymentProfile:
    """
    Deployment profile configuration.

    Defines how an agent should be deployed to a cloud provider.
    """

    provider: DeployProviderType
    service_name: str
    region: str
    min_instances: int = 0  # Scale to zero by default
    max_instances: int = 10
    memory: str = "2Gi"
    cpu: int = 2
    port: int = 8080
    timeout: int = 300  # seconds
    concurrency: int = 80
    deployment_mode: str = "agent"  # "agent" (single agent) or "multi_agent" (multi-agent host)
    # Cloud Run must make the identity/state lifetime explicit.  The provider
    # rejects its default ``unspecified`` value; it remains the dataclass
    # default so non-Cloud-Run providers keep their existing configuration
    # contract.
    persistence_mode: str = "unspecified"
    dockerfile: str = "docker/Dockerfile.cloudrun"
    env_vars: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)

    # Provider-specific fields
    gcp_project_id: Optional[str] = None
    azure_resource_group: Optional[str] = None

    @property
    def is_scale_to_zero(self) -> bool:
        """Whether this deployment scales to zero when idle."""
        return self.min_instances == 0

    @property
    def is_multi_agent(self) -> bool:
        """Whether this profile deploys a multi-agent multi_agent host."""
        return self.deployment_mode == "multi_agent"

    @property
    def is_ephemeral_demo(self) -> bool:
        """Whether the profile deliberately creates disposable demo identity."""
        return self.persistence_mode == "ephemeral_demo"

    @property
    def is_durable_sovereign(self) -> bool:
        """Whether the profile promises durable sovereign continuity."""
        return self.persistence_mode == "durable_sovereign"


@dataclass
class DeploymentSession:
    """Tracks an active deployment session."""

    service_name: str
    provider: DeployProviderType
    profile: DeploymentProfile
    status: DeployStatus
    started_at: datetime
    service_url: Optional[str] = None
    revision: Optional[str] = None
    health_status: Optional[str] = None
    last_updated: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary representation."""
        return {
            "service_name": self.service_name,
            "provider": self.provider.value,
            "status": self.status.value,
            "service_url": self.service_url,
            "revision": self.revision,
            "health_status": self.health_status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "error_message": self.error_message,
            "deployment_mode": self.profile.deployment_mode,
            "persistence_mode": self.profile.persistence_mode,
            "is_scale_to_zero": self.profile.is_scale_to_zero,
            "min_instances": self.profile.min_instances,
            "max_instances": self.profile.max_instances,
            "metadata": self.metadata,
        }


class DeployManagerError(Exception):
    """Custom exception for deployment failures."""
