"""
Cloud Run Deployment Provider.

Implements deployment to Google Cloud Run using the google-cloud-run SDK.
"""

import asyncio
import atexit
import logging
import os
import time
from typing import Any, Dict, List, Optional

from kestrel_sdk.deploy.models import DeployManagerError, DeploymentProfile
from kestrel_sdk.deploy.base import DeployProvider

logger = logging.getLogger(__name__)


class CloudRunProvider(DeployProvider):
    """
    Provider that deploys to Google Cloud Run.

    Uses the google-cloud-run Python SDK for programmatic deployment.
    Follows the same authentication chain as GCP Compute:
    1. GOOGLE_APPLICATION_CREDENTIALS env var
    2. GCP_SERVICE_ACCOUNT_KEY env var (inline JSON)
    3. Project-local credentials/kestrel-agent-admin.json
    4. Application Default Credentials (fallback)
    """

    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize Cloud Run provider.

        Args:
            project_id: GCP project ID (defaults to GCP_PROJECT_ID env var)
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        if not self.project_id:
            raise DeployManagerError(
                "GCP_PROJECT_ID is required for Cloud Run deployments"
            )

        self._services_client = None
        self._logging_client = None
        self._temp_cred_file = None
        self._setup_auth()

    def _setup_auth(self):
        """Setup GCP authentication from environment or project credentials."""
        import tempfile

        # Priority 1: Explicit GOOGLE_APPLICATION_CREDENTIALS env var
        creds_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_file and os.path.exists(creds_file):
            logger.info(
                f"Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {creds_file}"
            )
            return

        # Priority 2: Inline JSON key from GCP_SERVICE_ACCOUNT_KEY
        key_json = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
        if key_json:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                f.write(key_json)
                self._temp_cred_file = f.name
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
                logger.info("Using service account key from GCP_SERVICE_ACCOUNT_KEY")
            atexit.register(self._cleanup_temp_creds)
            return

        # Priority 3: Project-local service account file
        project_creds = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "credentials",
            "kestrel-agent-admin.json",
        )
        if os.path.exists(project_creds):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = project_creds
            logger.info(f"Using project service account: {project_creds}")
            return

        # Fallback: Application Default Credentials
        logger.warning(
            "No explicit credentials found. Using Application Default Credentials. "
            "Set GOOGLE_APPLICATION_CREDENTIALS or place credentials/kestrel-agent-admin.json"
        )

    def _cleanup_temp_creds(self):
        """Remove temporary credentials file created from GCP_SERVICE_ACCOUNT_KEY."""
        if self._temp_cred_file and os.path.exists(self._temp_cred_file):
            try:
                os.unlink(self._temp_cred_file)
                logger.debug(f"Cleaned up temp credentials: {self._temp_cred_file}")
            except OSError:
                pass
            self._temp_cred_file = None

    def cleanup(self) -> None:
        """
        Clean up provider resources.

        Removes temporary credential files created from inline
        GCP_SERVICE_ACCOUNT_KEY. Safe to call multiple times.
        """
        self._cleanup_temp_creds()

    def __del__(self):
        """Safety net: clean up temp files if cleanup() was never called."""
        try:
            self._cleanup_temp_creds()
        except Exception:
            pass

    def _get_services_client(self):
        """Lazy-load the Cloud Run services client."""
        if self._services_client is None:
            try:
                from google.cloud.run_v2 import ServicesClient

                self._services_client = ServicesClient()
            except ImportError as e:
                raise DeployManagerError(
                    "google-cloud-run not installed. Run: pip install google-cloud-run"
                ) from e
        return self._services_client

    def _get_logging_client(self):
        """Lazy-load the Cloud Logging client."""
        if self._logging_client is None:
            try:
                from google.cloud import logging as cloud_logging

                self._logging_client = cloud_logging.Client(project=self.project_id)
            except ImportError as e:
                raise DeployManagerError(
                    "google-cloud-logging not installed. Run: pip install google-cloud-logging"
                ) from e
        return self._logging_client

    async def deploy(
        self,
        image: str,
        service_name: str,
        profile: DeploymentProfile,
        env_vars: Optional[Dict[str, str]] = None,
        secrets: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Deploy to Cloud Run.

        Programmatizes scripts/cloudrun/deploy_dev.sh and deploy_prod.sh.
        """
        try:
            from google.cloud.run_v2 import Service
            from google.cloud.run_v2.types import (
                Container,
                ContainerPort,
                EnvVar,
                EnvVarSource,
                ResourceRequirements,
                SecretKeySelector,
            )

            client = self._get_services_client()

            # Build parent path: projects/{project}/locations/{region}
            parent = f"projects/{self.project_id}/locations/{profile.region}"
            service_path = f"{parent}/services/{service_name}"

            # Check if service exists
            try:
                existing_service = await asyncio.to_thread(
                    client.get_service, name=service_path
                )
                is_update = True
                logger.info(f"Updating existing Cloud Run service: {service_name}")
            except Exception:
                is_update = False
                logger.info(f"Creating new Cloud Run service: {service_name}")

            # Build environment variables
            env_list = []
            all_env_vars = {**profile.env_vars, **(env_vars or {})}
            for key, value in all_env_vars.items():
                env_list.append(EnvVar(name=key, value=value))

            # Build secret references (mounted as environment variables from Secret Manager)
            all_secrets = {**profile.secrets, **(secrets or {})}
            for key, secret_ref in all_secrets.items():
                # secret_ref format: "secret-name:version" (e.g., "kestrel-openai-key:latest")
                env_list.append(
                    EnvVar(
                        name=key,
                        value_source=EnvVarSource(
                            secret_key_ref=SecretKeySelector(
                                secret=secret_ref.split(":")[0],
                                version=secret_ref.split(":")[-1],
                            )
                        ),
                    )
                )

            # Build container spec
            container = Container(
                image=image,
                ports=[ContainerPort(container_port=profile.port)],
                env=env_list,
                resources=ResourceRequirements(
                    limits={
                        "memory": profile.memory,
                        "cpu": str(profile.cpu),
                    }
                ),
            )

            # Build service spec
            service = Service()
            service.template.containers = [container]
            service.template.scaling.min_instance_count = profile.min_instances
            service.template.scaling.max_instance_count = profile.max_instances
            service.template.timeout = f"{profile.timeout}s"
            service.template.max_instance_request_concurrency = profile.concurrency

            if is_update:
                # Update existing service
                service.name = service_path
                operation = await asyncio.to_thread(
                    client.update_service, service=service
                )
            else:
                # Create new service
                service.name = service_name
                operation = await asyncio.to_thread(
                    client.create_service, parent=parent, service=service, service_id=service_name
                )

            # Wait for operation to complete
            logger.info(f"Waiting for Cloud Run operation to complete...")
            result = await asyncio.to_thread(operation.result, timeout=600)

            # Get service URL
            service_url = result.uri

            logger.info(f"Deployment complete: {service_url}")

            return {
                "service_url": service_url,
                "revision": result.latest_ready_revision,
                "status": "active",
            }

        except ImportError as e:
            raise DeployManagerError(f"Missing dependency: {e}") from e
        except Exception as e:
            logger.error(f"Deployment failed: {e}", exc_info=True)
            raise DeployManagerError(f"Deployment failed: {e}") from e

    async def get_status(self, service_name: str) -> Dict[str, Any]:
        """Get Cloud Run service status."""
        try:
            client = self._get_services_client()
            service_path = f"projects/{self.project_id}/locations/*/services/{service_name}"

            # List services to find the exact location
            parent = f"projects/{self.project_id}/locations/-"
            services = await asyncio.to_thread(client.list_services, parent=parent)

            for service in services:
                if service.name.endswith(f"/services/{service_name}"):
                    # Parse status
                    conditions = service.conditions or []
                    status = "active"
                    for condition in conditions:
                        if condition.type == "Ready" and condition.state != "CONDITION_SUCCEEDED":
                            status = "deploying"
                            break

                    return {
                        "status": status,
                        "service_url": service.uri,
                        "revision": service.latest_ready_revision,
                        "health": "healthy" if status == "active" else "unknown",
                    }

            return {
                "status": "offline",
                "service_url": None,
                "revision": None,
                "health": "unknown",
            }

        except ImportError as e:
            raise DeployManagerError(f"Missing dependency: {e}") from e
        except Exception as e:
            logger.error(f"Failed to get status: {e}", exc_info=True)
            raise DeployManagerError(f"Failed to get status: {e}") from e

    async def teardown(self, service_name: str) -> Dict[str, Any]:
        """Delete a Cloud Run service."""
        try:
            client = self._get_services_client()

            # Find service across all regions
            parent = f"projects/{self.project_id}/locations/-"
            services = await asyncio.to_thread(client.list_services, parent=parent)

            service_path = None
            for service in services:
                if service.name.endswith(f"/services/{service_name}"):
                    service_path = service.name
                    break

            if not service_path:
                return {
                    "status": "not_found",
                    "message": f"Service {service_name} not found",
                }

            logger.info(f"Deleting Cloud Run service: {service_name}")
            operation = await asyncio.to_thread(
                client.delete_service, name=service_path
            )

            # Wait for deletion
            await asyncio.to_thread(operation.result, timeout=300)

            logger.info(f"Service {service_name} deleted")

            return {
                "status": "deleted",
                "message": f"Service {service_name} deleted successfully",
            }

        except ImportError as e:
            raise DeployManagerError(f"Missing dependency: {e}") from e
        except Exception as e:
            logger.error(f"Teardown failed: {e}", exc_info=True)
            raise DeployManagerError(f"Teardown failed: {e}") from e

    async def get_logs(self, service_name: str, lines: int = 100) -> str:
        """Get recent logs from Cloud Logging."""
        try:
            client = self._get_logging_client()

            # Query logs for this service
            filter_str = (
                f'resource.type="cloud_run_revision" '
                f'resource.labels.service_name="{service_name}"'
            )

            # Get recent entries
            entries = list(
                client.list_entries(
                    filter_=filter_str,
                    order_by="timestamp desc",
                    max_results=lines,
                )
            )

            if not entries:
                return f"No logs found for service {service_name}"

            # Format logs
            log_lines = []
            for entry in reversed(entries):  # Oldest first
                timestamp = entry.timestamp.isoformat() if entry.timestamp else "unknown"
                payload = entry.payload
                if isinstance(payload, dict):
                    message = payload.get("message", str(payload))
                else:
                    message = str(payload)
                log_lines.append(f"[{timestamp}] {message}")

            return "\n".join(log_lines)

        except ImportError as e:
            raise DeployManagerError(
                f"google-cloud-logging not installed. Run: pip install google-cloud-logging"
            ) from e
        except Exception as e:
            logger.error(f"Failed to get logs: {e}", exc_info=True)
            return f"Error fetching logs: {e}"

    async def list_deployments(self) -> List[Dict[str, Any]]:
        """List all Kestrel agent deployments."""
        try:
            client = self._get_services_client()
            parent = f"projects/{self.project_id}/locations/-"

            services = await asyncio.to_thread(client.list_services, parent=parent)

            deployments = []
            for service in services:
                # Only include services matching kestrel-* pattern
                service_name = service.name.split("/")[-1]
                if service_name.startswith("kestrel-"):
                    conditions = service.conditions or []
                    status = "active"
                    for condition in conditions:
                        if condition.type == "Ready" and condition.state != "CONDITION_SUCCEEDED":
                            status = "deploying"
                            break

                    deployments.append({
                        "name": service_name,
                        "status": status,
                        "url": service.uri,
                        "created": service.create_time.isoformat() if service.create_time else None,
                    })

            return deployments

        except ImportError as e:
            raise DeployManagerError(f"Missing dependency: {e}") from e
        except Exception as e:
            logger.error(f"Failed to list deployments: {e}", exc_info=True)
            return []

    async def health_check(self, url: str) -> Dict[str, Any]:
        """Check health of a deployed service."""
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
