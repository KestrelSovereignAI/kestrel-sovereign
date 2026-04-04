"""
Azure Container Apps Deployment Provider.

Implements deployment to Azure Container Apps using the azure-mgmt-appcontainers SDK.
Supports both single-agent and rookery (multi-agent) deployment modes.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from kestrel_sdk.deploy.models import DeployManagerError, DeploymentProfile
from kestrel_sdk.deploy.base import DeployProvider

logger = logging.getLogger(__name__)


class AzureContainerProvider(DeployProvider):
    """
    Provider that deploys to Azure Container Apps.

    Azure Container Apps is a serverless container platform that supports:
    - Scale to zero with min_replicas=0
    - Automatic HTTPS with managed certificates
    - Container revision management
    - Integrated monitoring via Azure Monitor

    Authentication chain:
    1. DefaultAzureCredential (Azure CLI, Managed Identity, env vars)
    2. AZURE_CLIENT_ID + AZURE_CLIENT_SECRET + AZURE_TENANT_ID env vars

    Required environment variables:
    - AZURE_SUBSCRIPTION_ID: Azure subscription ID
    - AZURE_RESOURCE_GROUP: Resource group for container apps
    - AZURE_LOCATION: Azure region (e.g., eastus2) — used only for new environments
    """

    def __init__(
        self,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
    ):
        self.subscription_id = subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = resource_group or os.getenv("AZURE_RESOURCE_GROUP")
        self.location = os.getenv("AZURE_LOCATION", "eastus2")

        if not self.subscription_id:
            raise DeployManagerError(
                "AZURE_SUBSCRIPTION_ID is required for Azure Container Apps deployments"
            )
        if not self.resource_group:
            raise DeployManagerError(
                "AZURE_RESOURCE_GROUP is required for Azure Container Apps deployments"
            )

        self._client = None
        self._credential = None

    def _get_credential(self):
        """Get Azure credential using DefaultAzureCredential."""
        if self._credential is None:
            try:
                from azure.identity import DefaultAzureCredential

                self._credential = DefaultAzureCredential()
            except ImportError as e:
                raise DeployManagerError(
                    "azure-identity not installed. Run: pip install azure-identity"
                ) from e
        return self._credential

    def _get_client(self):
        """Lazy-load the Container Apps management client."""
        if self._client is None:
            try:
                from azure.mgmt.appcontainers import ContainerAppsAPIClient

                self._client = ContainerAppsAPIClient(
                    credential=self._get_credential(),
                    subscription_id=self.subscription_id,
                )
            except ImportError as e:
                raise DeployManagerError(
                    "azure-mgmt-appcontainers not installed. "
                    "Run: pip install azure-mgmt-appcontainers"
                ) from e
        return self._client

    def _get_or_create_environment(self, region: str) -> str:
        """Get existing Container Apps Environment or create one.

        Returns the environment resource ID.
        """
        client = self._get_client()
        env_name = "kestrel-env"

        # Check if environment exists
        try:
            env = client.managed_environments.get(
                resource_group_name=self.resource_group,
                environment_name=env_name,
            )
            return env.id
        except Exception:
            pass

        # Create new environment
        logger.info(f"Creating Container Apps Environment '{env_name}' in {region}...")
        from azure.mgmt.appcontainers.models import ManagedEnvironment

        env_params = ManagedEnvironment(location=region)
        poller = client.managed_environments.begin_create_or_update(
            resource_group_name=self.resource_group,
            environment_name=env_name,
            environment_envelope=env_params,
        )
        env = poller.result()
        logger.info(f"Environment created: {env.id}")
        return env.id

    async def deploy(
        self,
        image: str,
        service_name: str,
        profile: DeploymentProfile,
        env_vars: Optional[Dict[str, str]] = None,
        secrets: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Deploy to Azure Container Apps."""
        try:
            from azure.mgmt.appcontainers.models import (
                Configuration,
                Container,
                ContainerApp,
                CustomScaleRule,
                Dapr,
                EnvironmentVar,
                Ingress,
                Scale,
                ScaleRule,
                Secret,
                Template,
            )

            client = self._get_client()

            # Get or create the managed environment
            env_id = await asyncio.to_thread(
                self._get_or_create_environment, profile.region
            )

            # Build environment variables
            env_list = []
            all_env_vars = {**profile.env_vars, **(env_vars or {})}
            for key, value in all_env_vars.items():
                env_list.append(EnvironmentVar(name=key, value=value))

            # Build secrets and secret env var references
            secret_list = []
            all_secrets = {**profile.secrets, **(secrets or {})}
            for key, secret_value in all_secrets.items():
                # Secret names must be lowercase alphanumeric + hyphens
                secret_name = key.lower().replace("_", "-")
                secret_list.append(Secret(name=secret_name, value=secret_value))
                env_list.append(
                    EnvironmentVar(name=key, secret_ref=secret_name)
                )

            # Build container spec
            container = Container(
                name=service_name,
                image=image,
                env=env_list,
                resources={
                    "cpu": float(profile.cpu),
                    "memory": profile.memory,
                },
            )

            # Build scale rules
            scale = Scale(
                min_replicas=profile.min_instances,
                max_replicas=profile.max_instances,
            )

            # Build ingress (external, HTTPS)
            ingress = Ingress(
                external=True,
                target_port=profile.port,
                transport="auto",
            )

            # Build the container app
            container_app = ContainerApp(
                location=profile.region,
                managed_environment_id=env_id,
                configuration=Configuration(
                    ingress=ingress,
                    secrets=secret_list if secret_list else None,
                ),
                template=Template(
                    containers=[container],
                    scale=scale,
                ),
            )

            # Check if app exists (update vs create)
            try:
                existing = await asyncio.to_thread(
                    client.container_apps.get,
                    resource_group_name=self.resource_group,
                    container_app_name=service_name,
                )
                is_update = True
                logger.info(f"Updating existing Container App: {service_name}")
            except Exception:
                is_update = False
                logger.info(f"Creating new Container App: {service_name}")

            # Create or update
            poller = await asyncio.to_thread(
                client.container_apps.begin_create_or_update,
                resource_group_name=self.resource_group,
                container_app_name=service_name,
                container_app_envelope=container_app,
            )

            # Wait for completion
            logger.info("Waiting for Container App deployment to complete...")
            result = await asyncio.to_thread(poller.result, timeout=600)

            # Get the FQDN
            service_url = None
            if result.configuration and result.configuration.ingress:
                fqdn = result.configuration.ingress.fqdn
                if fqdn:
                    service_url = f"https://{fqdn}"

            logger.info(f"Deployment complete: {service_url}")

            return {
                "service_url": service_url,
                "revision": result.latest_revision_name,
                "status": "active",
            }

        except ImportError as e:
            raise DeployManagerError(f"Missing dependency: {e}") from e
        except Exception as e:
            logger.error(f"Deployment failed: {e}", exc_info=True)
            raise DeployManagerError(f"Deployment failed: {e}") from e

    async def get_status(self, service_name: str) -> Dict[str, Any]:
        """Get Azure Container App status."""
        try:
            client = self._get_client()

            app = await asyncio.to_thread(
                client.container_apps.get,
                resource_group_name=self.resource_group,
                container_app_name=service_name,
            )

            # Determine status from provisioning state
            provisioning_state = (app.provisioning_state or "").lower()
            if provisioning_state == "succeeded":
                status = "active"
            elif provisioning_state in ("inprogress", "updating"):
                status = "deploying"
            elif provisioning_state == "failed":
                status = "failed"
            else:
                status = "unknown"

            # Get URL
            service_url = None
            if app.configuration and app.configuration.ingress:
                fqdn = app.configuration.ingress.fqdn
                if fqdn:
                    service_url = f"https://{fqdn}"

            return {
                "status": status,
                "service_url": service_url,
                "revision": app.latest_revision_name,
                "health": "healthy" if status == "active" else "unknown",
            }

        except ImportError as e:
            raise DeployManagerError(f"Missing dependency: {e}") from e
        except Exception as e:
            # App not found
            if "ResourceNotFound" in str(type(e).__name__) or "not found" in str(e).lower():
                return {
                    "status": "offline",
                    "service_url": None,
                    "revision": None,
                    "health": "unknown",
                }
            logger.error(f"Failed to get status: {e}", exc_info=True)
            raise DeployManagerError(f"Failed to get status: {e}") from e

    async def teardown(self, service_name: str) -> Dict[str, Any]:
        """Delete an Azure Container App."""
        try:
            client = self._get_client()

            # Check if app exists
            try:
                await asyncio.to_thread(
                    client.container_apps.get,
                    resource_group_name=self.resource_group,
                    container_app_name=service_name,
                )
            except Exception:
                return {
                    "status": "not_found",
                    "message": f"Container App {service_name} not found",
                }

            logger.info(f"Deleting Container App: {service_name}")
            poller = await asyncio.to_thread(
                client.container_apps.begin_delete,
                resource_group_name=self.resource_group,
                container_app_name=service_name,
            )

            # Wait for deletion
            await asyncio.to_thread(poller.result, timeout=300)

            logger.info(f"Container App {service_name} deleted")

            return {
                "status": "deleted",
                "message": f"Container App {service_name} deleted successfully",
            }

        except ImportError as e:
            raise DeployManagerError(f"Missing dependency: {e}") from e
        except Exception as e:
            logger.error(f"Teardown failed: {e}", exc_info=True)
            raise DeployManagerError(f"Teardown failed: {e}") from e

    async def get_logs(self, service_name: str, lines: int = 100) -> str:
        """Get recent logs from Azure Container App via system logs.

        Uses the Container Apps system log stream. For richer queries,
        Azure Monitor / Log Analytics workspace integration is recommended.
        """
        try:
            client = self._get_client()

            # List recent revisions to get log sources
            revisions = await asyncio.to_thread(
                client.container_apps_revisions.list_revisions,
                resource_group_name=self.resource_group,
                container_app_name=service_name,
            )

            revision_names = []
            for rev in revisions:
                revision_names.append(rev.name)

            if not revision_names:
                return f"No revisions found for Container App {service_name}"

            # Get replica logs from the latest revision
            # Note: Full log retrieval requires Log Analytics workspace
            # This returns basic info about the revision state
            latest_rev = revision_names[0]
            try:
                replicas = await asyncio.to_thread(
                    client.container_apps_revision_replicas.list_replicas,
                    resource_group_name=self.resource_group,
                    container_app_name=service_name,
                    revision_name=latest_rev,
                )
                replica_info = []
                for replica in replicas:
                    replica_info.append(
                        f"Replica: {replica.name}, Created: {replica.created_time}, "
                        f"Running: {replica.running_state}"
                    )
                if replica_info:
                    return "\n".join(replica_info)
            except Exception as e:
                logger.debug(f"Could not list replicas: {e}")

            return (
                f"Container App {service_name} has {len(revision_names)} revision(s). "
                f"Latest: {latest_rev}. "
                f"For detailed logs, configure Azure Monitor Log Analytics workspace."
            )

        except ImportError as e:
            raise DeployManagerError(
                "azure-mgmt-appcontainers not installed. "
                "Run: pip install azure-mgmt-appcontainers"
            ) from e
        except Exception as e:
            logger.error(f"Failed to get logs: {e}", exc_info=True)
            return f"Error fetching logs: {e}"

    async def list_deployments(self) -> List[Dict[str, Any]]:
        """List all Kestrel Container App deployments in the resource group."""
        try:
            client = self._get_client()

            apps = await asyncio.to_thread(
                client.container_apps.list_by_resource_group,
                resource_group_name=self.resource_group,
            )

            deployments = []
            for app in apps:
                # Only include apps matching kestrel-* pattern
                if not app.name.startswith("kestrel-"):
                    continue

                provisioning_state = (app.provisioning_state or "").lower()
                if provisioning_state == "succeeded":
                    status = "active"
                elif provisioning_state in ("inprogress", "updating"):
                    status = "deploying"
                else:
                    status = provisioning_state

                service_url = None
                if app.configuration and app.configuration.ingress:
                    fqdn = app.configuration.ingress.fqdn
                    if fqdn:
                        service_url = f"https://{fqdn}"

                deployments.append({
                    "name": app.name,
                    "status": status,
                    "url": service_url,
                    "created": (
                        app.system_data.created_at.isoformat()
                        if app.system_data and app.system_data.created_at
                        else None
                    ),
                })

            return deployments

        except ImportError as e:
            raise DeployManagerError(f"Missing dependency: {e}") from e
        except Exception as e:
            logger.error(f"Failed to list deployments: {e}", exc_info=True)
            return []

    async def health_check(self, url: str) -> Dict[str, Any]:
        """Check health of a deployed Container App."""
        import httpx

        try:
            health_url = f"{url.rstrip('/')}/health"
            start_time = time.time()

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(health_url)
                response_time = time.time() - start_time

                return {
                    "healthy": 200 <= response.status_code < 400,
                    "status_code": response.status_code,
                    "response_time": response_time,
                }

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return {
                "healthy": False,
                "status_code": None,
                "response_time": None,
                "error": str(e),
            }
