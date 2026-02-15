"""
Azure Container Apps Deployment Provider (Stub).

Stub implementation of DeployProvider for Azure Container Apps.
Full implementation coming soon.
"""

import logging
from typing import Any, Dict, List, Optional

from ..models import DeploymentProfile
from .base import DeployProvider

logger = logging.getLogger(__name__)


class AzureContainerProvider(DeployProvider):
    """
    Provider that deploys to Azure Container Apps (stub).

    Azure Container Apps is a serverless container platform that scales to zero,
    similar to Google Cloud Run. Once implemented, this provider will use:
    - azure-mgmt-appcontainers SDK for container apps management
    - azure-identity SDK for authentication (DefaultAzureCredential)
    - AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP env vars for targeting

    Azure Container Apps Features:
    - Scale to zero with min_replicas=0 (similar to Cloud Run min_instances=0)
    - Automatic HTTPS with managed certificates
    - Built-in ingress and service discovery
    - Container revision management
    - Integrated with Azure Monitor for logging

    Authentication Flow (planned):
    1. DefaultAzureCredential (Azure CLI, Managed Identity, etc.)
    2. AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID env vars
    3. Azure service principal JSON file

    Required Environment Variables:
    - AZURE_SUBSCRIPTION_ID: Azure subscription ID
    - AZURE_RESOURCE_GROUP: Resource group for container apps
    - AZURE_LOCATION: Azure region (e.g., eastus2)
    """

    def __init__(
        self,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
    ):
        """
        Initialize Azure Container Apps provider (stub).

        Args:
            subscription_id: Azure subscription ID
            resource_group: Azure resource group name
        """
        import os

        self.subscription_id = subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = resource_group or os.getenv("AZURE_RESOURCE_GROUP")

        logger.info(
            "Azure Container Apps provider initialized (stub - not yet implemented)"
        )

    async def deploy(
        self,
        image: str,
        service_name: str,
        profile: DeploymentProfile,
        env_vars: Optional[Dict[str, str]] = None,
        secrets: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Deploy an agent to Azure Container Apps.

        Args:
            image: Container image to deploy
            service_name: Name of the container app
            profile: Deployment profile with configuration
            env_vars: Environment variables to set
            secrets: Secrets to mount

        Returns:
            Deployment info dict

        Raises:
            NotImplementedError: Azure Container Apps support coming soon
        """
        raise NotImplementedError(
            "Azure Container Apps support coming soon. "
            "Planned SDK: azure-mgmt-appcontainers with DefaultAzureCredential auth"
        )

    async def get_status(self, service_name: str) -> Dict[str, Any]:
        """
        Get the current status of a deployed container app.

        Args:
            service_name: Name of the container app

        Returns:
            Status dict

        Raises:
            NotImplementedError: Azure Container Apps support coming soon
        """
        raise NotImplementedError(
            "Azure Container Apps support coming soon. "
            "Will use ContainerAppsAPIClient.container_apps.get() for status"
        )

    async def teardown(self, service_name: str) -> Dict[str, Any]:
        """
        Delete a deployed container app.

        Args:
            service_name: Name of the container app to delete

        Returns:
            Result dict

        Raises:
            NotImplementedError: Azure Container Apps support coming soon
        """
        raise NotImplementedError(
            "Azure Container Apps support coming soon. "
            "Will use ContainerAppsAPIClient.container_apps.begin_delete()"
        )

    async def get_logs(self, service_name: str, lines: int = 100) -> str:
        """
        Get recent logs from a deployed container app.

        Args:
            service_name: Name of the container app
            lines: Number of log lines to retrieve

        Returns:
            Log output as string

        Raises:
            NotImplementedError: Azure Container Apps support coming soon
        """
        raise NotImplementedError(
            "Azure Container Apps support coming soon. "
            "Will use Azure Monitor integration for log retrieval"
        )

    async def list_deployments(self) -> List[Dict[str, Any]]:
        """
        List all container app deployments.

        Returns:
            List of deployment dicts

        Raises:
            NotImplementedError: Azure Container Apps support coming soon
        """
        raise NotImplementedError(
            "Azure Container Apps support coming soon. "
            "Will use ContainerAppsAPIClient.container_apps.list_by_resource_group()"
        )

    async def health_check(self, url: str) -> Dict[str, Any]:
        """
        Check health of a deployed container app.

        Args:
            url: Service URL to check

        Returns:
            Health result dict

        Raises:
            NotImplementedError: Azure Container Apps support coming soon
        """
        raise NotImplementedError(
            "Azure Container Apps support coming soon. "
            "Will perform HTTP health check similar to Cloud Run provider"
        )
