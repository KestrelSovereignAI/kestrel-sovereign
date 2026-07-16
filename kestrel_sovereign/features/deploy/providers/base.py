"""
Deploy Provider Abstract Base Class.

Defines the interface that all deployment providers must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ._health import probe_http_health

if TYPE_CHECKING:
    from ..models import DeploymentProfile

logger = logging.getLogger(__name__)


class DeployProvider(ABC):
    """Abstract provider that knows how to deploy agents to cloud platforms."""

    @abstractmethod
    async def deploy(
        self,
        image: str,
        service_name: str,
        profile: "DeploymentProfile",
        env_vars: Optional[Dict[str, str]] = None,
        secrets: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Deploy an agent to the cloud platform.

        Args:
            image: Container image to deploy (e.g., gcr.io/project/image:tag)
            service_name: Name of the service to create/update
            profile: Deployment profile with configuration
            env_vars: Environment variables to set
            secrets: Secrets to mount (name -> secret reference)

        Returns:
            Deployment info dict with keys: service_url, revision, status
        """
        ...

    @abstractmethod
    async def get_status(self, service_name: str) -> Dict[str, Any]:
        """
        Get the current status of a deployed service.

        Args:
            service_name: Name of the service

        Returns:
            Status dict with keys: status, service_url, revision, health
        """
        ...

    @abstractmethod
    async def teardown(self, service_name: str) -> Dict[str, Any]:
        """
        Delete a deployed service.

        Args:
            service_name: Name of the service to delete

        Returns:
            Result dict with keys: status, message
        """
        ...

    @abstractmethod
    async def get_logs(self, service_name: str, lines: int = 100) -> str:
        """
        Get recent logs from a deployed service.

        Args:
            service_name: Name of the service
            lines: Number of log lines to retrieve

        Returns:
            Log output as string
        """
        ...

    @abstractmethod
    async def list_deployments(self) -> List[Dict[str, Any]]:
        """
        List all agent deployments.

        Returns:
            List of deployment dicts with keys: name, status, url, created
        """
        ...

    async def health_check(self, url: str) -> Dict[str, Any]:
        """
        Check health of a deployed service.

        Args:
            url: Service URL to check

        Returns:
            Health result dict with keys: healthy, status_code, response_time
        """
        result = await probe_http_health(url)
        if "error" in result:
            logger.warning("Health check failed: %s", result["error"])
        return dict(result)

    def cleanup(self) -> None:
        """
        Clean up provider resources (temp files, credentials, etc.).

        Called during explicit teardown or when the provider is no longer needed.
        Subclasses should override to clean up provider-specific resources.
        """
