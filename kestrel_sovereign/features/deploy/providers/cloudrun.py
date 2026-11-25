"""
Cloud Run Deployment Provider.

Implements deployment to Google Cloud Run using the google-cloud-run SDK.
"""

import asyncio
import hashlib
import logging
import os
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

from ..models import DeployManagerError, DeploymentProfile
from ..persistence import validate_cloudrun_persistence
from .base import DeployProvider

logger = logging.getLogger(__name__)


def _is_seeded_demo(profile: DeploymentProfile) -> bool:
    return any(
        setting in profile.env_vars or setting in profile.secrets
        for setting in ("KESTREL_DEMO_SEED_DATA_KEY_FILE", "KESTREL_DEMO_SEED_ROSTER_TEMPLATE")
    )


def _apply_managed_template(service: Any, container: Any, profile: DeploymentProfile) -> None:
    """Write the revision-template fields Kestrel manages onto ``service``.

    Preserve unmanaged configuration at every nested level, including mounts,
    probes, command/args, port protocol and resource flags. This single-image
    interface cannot select an ingress container from a sidecar deployment.
    """
    count = len(service.template.containers)
    if count > 1:
        raise DeployManagerError("Cloud Run profile updates require a single-container service")
    if len(container.ports) != 1:
        raise DeployManagerError("Cloud Run profiles require one explicit ingress port")
    if _is_seeded_demo(profile):
        from google.cloud.run_v2.types import ServiceScaling

        scaling = service.scaling
        if (
            profile.max_instances != 1
            or sum(target.percent > 0 for target in service.traffic) > 1
            or (
                scaling.scaling_mode == ServiceScaling.ScalingMode.MANUAL
                and scaling.manual_instance_count != 1
            )
        ):
            raise DeployManagerError("Seeded demo publication requires one instance and an unsplit traffic target")
    current = service.template.containers[0] if count else container
    if count:
        if len(current.ports) > 1:
            raise DeployManagerError("Cloud Run profile updates require one ingress port")
        port = container.ports[0].container_port
        for probe in (current.startup_probe, current.liveness_probe, current.readiness_probe):
            if any(value and value != port for value in (
                    probe.http_get.port, probe.tcp_socket.port, probe.grpc.port)):
                raise DeployManagerError(
                    "Existing Cloud Run probe port conflicts with the configured ingress port")
    volumes, mounts = _plan_secret_files(
        service.template.volumes, current.volume_mounts,
        profile.secret_files, profile.secret_file_adoptions,
    )
    # A fetched template can name an existing immutable revision. Omit that
    # identity so Cloud Run generates a name for the new configuration (#4170).
    service.template.revision = ""
    if count == 1:
        current.image = container.image
        current.env = container.env
        if current.ports:
            current.ports[0].container_port = container.ports[0].container_port
        else:
            current.ports = container.ports
        current.resources.limits.update(container.resources.limits)
    else:
        service.template.containers = [current]
    service.template.volumes = volumes
    service.template.containers[0].volume_mounts = mounts
    service.template.scaling.min_instance_count = profile.min_instances
    service.template.scaling.max_instance_count = profile.max_instances
    service.template.timeout = f"{profile.timeout}s"
    service.template.max_instance_request_concurrency = profile.concurrency


def _manual_secret_mount(mounts, volumes, path: PurePosixPath, volume_name: str, secret: str, version: str):
    """Accept only the single manual mount explicitly named by the profile."""
    directory = str(path.parent)
    related = [mount for mount in mounts if mount.name == volume_name or mount.mount_path == directory]
    if not related:
        return None
    volume = volumes.get(volume_name)
    if len(related) == 1 and volume is not None:
        mount = related[0]
        if (
            mount.name == volume_name and mount.mount_path == directory and not mount.sub_path
            and volume.secret.secret == secret and len(volume.secret.items) == 1
        ):
            item = volume.secret.items[0]
            if (item.path, item.version) == (path.name, version):
                return mount
    raise DeployManagerError(f"Cloud Run secret file adoption at {path} differs from the declared manual mount")


def _plan_secret_files(
    existing_volumes, existing_mounts, files: dict[str, str], adoptions: dict[str, str],
) -> tuple[list, list]:
    from google.cloud.run_v2.types import SecretVolumeSource, VersionToPath, Volume, VolumeMount

    if (
        adoptions.keys() - files.keys()
        or any(not isinstance(name, str) or not name for name in adoptions.values())
    ):
        raise DeployManagerError("Cloud Run secret file adoption requires a declared file and existing volume name")
    original_volumes = {volume.name: volume for volume in existing_volumes}
    owned = {
        (mount.name, mount.mount_path) for mount in existing_mounts
        if (mount.name in original_volumes and original_volumes[mount.name].secret.secret
            and mount.name == _secret_file_volume_name(mount.mount_path))
    }
    mounts = [mount for mount in existing_mounts if (mount.name, mount.mount_path) not in owned]
    referenced = {mount.name for mount in mounts}
    owned_names = {name for name, _ in owned}
    volumes = {
        name: volume for name, volume in original_volumes.items()
        if name not in owned_names or name in referenced
    }
    cloud_sql_volumes = {
        volume.name for volume in volumes.values() if volume.cloud_sql_instance.instances
    }
    directories = {
        "/cloudsql" if not mount.mount_path and mount.name in cloud_sql_volumes else mount.mount_path
        for mount in mounts
    }
    for filename, reference in sorted(files.items()):
        path = PurePosixPath(filename)
        secret, _, version = reference.rpartition(":")
        if (
            not path.is_absolute() or ".." in path.parts or len(path.parts) < 3
            or not secret or not version.isascii() or not version.isdecimal() or int(version) < 1
        ):
            raise DeployManagerError("Cloud Run secret files require an absolute file path and a pinned version")
        directory = str(path.parent)
        name = _secret_file_volume_name(directory)
        previous = original_volumes[name].secret if name in owned_names else SecretVolumeSource()
        if filename in adoptions:
            manual = _manual_secret_mount(mounts, volumes, path, adoptions[filename], secret, version)
            if manual is not None:
                if name in volumes:
                    raise DeployManagerError("Cloud Run secret file adoption conflicts with the managed volume name")
                previous = volumes.pop(manual.name).secret
                mounts.remove(manual)
                directories.remove(directory)
        if any(
            PurePosixPath(other).is_relative_to(directory) or path.parent.is_relative_to(other)
            for other in directories
        ):
            raise DeployManagerError("Cloud Run secret file mount overlaps another volume")
        directories.add(directory)
        mode = next((item.mode for item in previous.items if item.path == path.name), 0)
        volume = Volume(
            name=name, secret=SecretVolumeSource(
                secret=secret, default_mode=previous.default_mode,
                items=[VersionToPath(version=version, path=path.name, mode=mode)],
            ),
        )
        if name in volumes and volumes[name] != volume:
            raise DeployManagerError("Cloud Run secret file conflicts with an existing volume")
        volumes[name] = volume
        mounts.append(VolumeMount(name=name, mount_path=directory))
    return list(volumes.values()), mounts


def _secret_file_volume_name(directory: str) -> str:
    return "kestrel-secret-file-" + hashlib.sha256(directory.encode()).hexdigest()[:16]


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
        """Setup GCP authentication from environment or project credentials.

        Delegates to :func:`kestrel_sovereign.features.deploy._gcp_auth.setup_gcp_auth`
        so this provider and ``kestrel deploy secrets sync`` share one
        credential discovery chain. The temp file path (if any) is
        captured for cleanup via :meth:`cleanup` / ``__del__``.
        """
        from .._gcp_auth import setup_gcp_auth

        self._temp_cred_file = setup_gcp_auth()

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

    @staticmethod
    def _status_from_conditions(conditions) -> str:
        """Derive a service status from its Ready condition.

        ``condition.state`` is a proto ``Condition.State`` enum, so it must be
        compared against the enum member — never the string
        ``"CONDITION_SUCCEEDED"`` (which would never match, leaving every
        service reporting ``deploying``).
        """
        from google.cloud.run_v2.types import Condition

        for condition in conditions or []:
            if (
                condition.type == "Ready"
                and condition.state != Condition.State.CONDITION_SUCCEEDED
            ):
                return "deploying"
        return "active"

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
            # Validate the effective values, including direct-call overrides.
            # Otherwise a caller could validate a durable profile and then
            # replace its PostgreSQL/persistence env with local SQLite here.
            all_env_vars = {**profile.env_vars, **(env_vars or {})}
            all_secrets = {**profile.secrets, **(secrets or {})}
            effective_profile = replace(
                profile,
                env_vars=all_env_vars,
                secrets=all_secrets,
            )
            validate_cloudrun_persistence(effective_profile)

            from google.api_core.exceptions import NotFound
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
            except NotFound:
                is_update = False
                logger.info(f"Creating new Cloud Run service: {service_name}")

            # Build environment variables
            env_list = []
            for key, value in all_env_vars.items():
                env_list.append(EnvVar(name=key, value=value))

            # Build secret references (mounted as environment variables from Secret Manager)
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

            if is_update:
                # Read-modify-write: mutate the FETCHED service in place so the
                # settings Kestrel does NOT manage — service_account, vpc_access,
                # labels, traffic split, annotations, etc. — survive the update.
                # Building a fresh Service() and replacing the whole spec (no
                # update mask) silently reset all of them out-of-band (F174).
                service = existing_service
                service.name = service_path
                _apply_managed_template(service, container, effective_profile)
                operation = await asyncio.to_thread(
                    client.update_service, service=service
                )
            else:
                # Create new service from a fresh spec.
                service = Service()
                service.name = service_name
                _apply_managed_template(service, container, effective_profile)
                operation = await asyncio.to_thread(
                    client.create_service, parent=parent, service=service, service_id=service_name
                )

            # Wait for operation to complete
            logger.info("Waiting for Cloud Run operation to complete...")
            result = await asyncio.to_thread(operation.result, timeout=600)

            # Allow unauthenticated invocations — matches the bash
            # ``gcloud run deploy --allow-unauthenticated`` flag, which
            # binds ``allUsers`` to ``roles/run.invoker``. Without this
            # the freshly-created service returns 401 to public traffic
            # and the app's OAuth flow / /health endpoint can't be
            # reached. Codex review on PR #1064 caught the regression.
            iam_warning = await self._allow_unauthenticated(client, service_path)

            # Get service URL
            service_url = result.uri

            logger.info(f"Deployment complete: {service_url}")

            deployment = {
                "service_url": service_url,
                "revision": result.latest_ready_revision,
                "status": "active",
                "operation": "update" if is_update else "create",
                "warnings": [iam_warning] if iam_warning else [],
            }
            if _is_seeded_demo(effective_profile):
                revision = result.latest_created_revision.rsplit("/", 1)[-1]
                active = [target for target in result.traffic_statuses if target.percent > 0]
                deployment["revision"] = revision
                deployment["serves_created_revision"] = (
                    bool(revision)
                    and len(active) == 1
                    and active[0].percent == 100
                    and active[0].revision.rsplit("/", 1)[-1] == revision
                )
            return deployment

        except ImportError as e:
            raise DeployManagerError(f"Missing dependency: {e}") from e
        except Exception as e:
            logger.error(f"Deployment failed: {e}", exc_info=True)
            raise DeployManagerError(f"Deployment failed: {e}") from e

    async def _allow_unauthenticated(self, client, service_path: str) -> Optional[str]:
        """Bind ``allUsers`` to ``roles/run.invoker`` on the service.

        Equivalent to ``gcloud run deploy --allow-unauthenticated``,
        which the bash deploy_*.sh scripts always passed. Without this
        binding Cloud Run rejects public traffic with 401, so the
        application's OAuth flow and /health endpoint are unreachable
        for new services or services recreated after teardown.

        We fetch the current IAM policy, add the binding only if it's
        not already present (idempotent), and set it back. A failure is
        logged and returned as a warning rather than raised so the deploy
        itself doesn't regress on transient IAM API errors — the operator
        can grant the role manually as a backstop. That is safe only
        because the manager's readiness gate probes the service
        unauthenticated afterwards: a missing binding surfaces there as an
        unready deploy, not as success.

        Returns:
            None when the binding is in place, else a warning message.
        """
        try:
            from google.iam.v1 import iam_policy_pb2

            get_req = iam_policy_pb2.GetIamPolicyRequest(resource=service_path)
            policy = await asyncio.to_thread(client.get_iam_policy, request=get_req)

            already_set = any(
                b.role == "roles/run.invoker" and "allUsers" in b.members
                for b in policy.bindings
            )
            if already_set:
                logger.debug("allUsers already has run.invoker; no IAM change needed")
                return None

            policy.bindings.add(
                role="roles/run.invoker",
                members=["allUsers"],
            )
            set_req = iam_policy_pb2.SetIamPolicyRequest(
                resource=service_path, policy=policy
            )
            await asyncio.to_thread(client.set_iam_policy, request=set_req)
            logger.info(
                f"Granted allUsers run.invoker on {service_path} "
                f"(equivalent to --allow-unauthenticated)"
            )
            return None
        except ImportError as e:  # protobuf module missing
            logger.warning(
                "Could not import google.iam.v1 — skipping "
                f"allUsers/run.invoker binding: {e}. The service may "
                "require manual IAM granting before public traffic works."
            )
            return (
                "allUsers/run.invoker was not granted (google.iam.v1 is not "
                "installed); the service may reject public traffic"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to grant allUsers/run.invoker (the service is "
                "deployed but may not accept public traffic until you "
                f"run `gcloud run services add-iam-policy-binding`): {e}"
            )
            return (
                f"allUsers/run.invoker could not be granted ({type(e).__name__}); "
                "the service may reject public traffic until you run "
                "`gcloud run services add-iam-policy-binding`"
            )

    async def get_status(self, service_name: str) -> Dict[str, Any]:
        """Get Cloud Run service status."""
        try:
            client = self._get_services_client()
            # List services to find the exact location
            parent = f"projects/{self.project_id}/locations/-"
            services = await asyncio.to_thread(client.list_services, parent=parent)

            for service in services:
                if service.name.endswith(f"/services/{service_name}"):
                    # Parse status
                    status = self._status_from_conditions(service.conditions)

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
                "google-cloud-logging not installed. Run: pip install google-cloud-logging"
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
                    status = self._status_from_conditions(service.conditions)

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
