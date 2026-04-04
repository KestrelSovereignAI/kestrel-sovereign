"""
GCP Compute Engine Core SDK Operations.

Contains the core manager class with SDK operations, authentication,
profile loading, and session management.
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.config import load_config
from kestrel_sdk.config.constants import (
    HTTP_TIMEOUT_SHORT,
    GCP_OPERATION_POLL_INTERVAL,
)

from .models import (
    GCPComputeManagerError,
    GCPComputeSession,
    GPUProfile,
    InstanceStatus,
)

logger = logging.getLogger(__name__)


def _sanitize_env_vars(env_vars: Dict[str, Any]) -> Dict[str, str]:
    """Drop unset environment values before generating startup scripts."""
    return {
        key: str(value)
        for key, value in env_vars.items()
        if value is not None
    }


class GCPComputeEngineManagerCore:
    """
    Core GCP Compute Engine operations.

    Handles SDK initialization, authentication, profile loading,
    and basic instance lifecycle management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("gcp_compute_config.toml")
        self.manager_config = self.config.get("manager", {})

        self.project_id = os.getenv("GCP_PROJECT_ID") or self.manager_config.get(
            "project_id"
        )
        if not self.project_id:
            logger.warning(
                "GCP_PROJECT_ID not set - GCP Compute features will be unavailable"
            )

        self.default_zone = self.manager_config.get("default_zone", "us-central1-a")
        self.default_region = self.manager_config.get("default_region", "us-central1")
        self.prefer_spot = self.manager_config.get("prefer_spot", True)

        self.default_ttl_seconds = int(
            os.getenv(
                "GCP_DEFAULT_TTL_SECONDS",
                self.manager_config.get("default_ttl_seconds", 3600),
            )
        )
        self.max_ttl_seconds = int(self.manager_config.get("max_ttl_seconds", 7200))
        self.poll_interval = int(self.manager_config.get("poll_interval_seconds", 15))
        self.readiness_timeout = int(
            self.manager_config.get("readiness_timeout_seconds", 600)
        )

        self.ssh_key_file = os.path.expanduser(
            self.manager_config.get("ssh_key_file", "~/.ssh/kestrel_gcp")
        )
        self.ssh_user = self.manager_config.get("ssh_user", "kestrel")

        self.profiles = self._load_profiles(self.config.get("profiles", {}))
        self.disk_config = self.config.get("persistent_disk", {})

        self._instances_client = None
        self._disks_client = None
        self._session: Optional[GCPComputeSession] = None
        self._lock = asyncio.Lock()
        self._credentials_file: Optional[str] = None

        # Setup authentication
        self._setup_auth()

    def _setup_auth(self):
        """Setup GCP authentication from environment or project credentials."""
        # Priority 1: Explicit GOOGLE_APPLICATION_CREDENTIALS env var
        creds_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_file and os.path.exists(creds_file):
            logger.info(f"Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {creds_file}")
            self._credentials_file = creds_file
            return

        # Priority 2: Inline JSON key from GCP_SERVICE_ACCOUNT_KEY
        key_json = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
        if key_json:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                f.write(key_json)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
                self._credentials_file = f.name
                logger.info("Using service account key from GCP_SERVICE_ACCOUNT_KEY")
            return

        # Priority 3: Project-local service account file
        project_creds = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "credentials",
            "kestrel-agent-admin.json",
        )
        if os.path.exists(project_creds):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = project_creds
            self._credentials_file = project_creds
            logger.info(f"Using project service account: {project_creds}")
            return

        # Fallback: Application Default Credentials (NOT recommended)
        logger.warning(
            "No explicit credentials found. Using Application Default Credentials. "
            "Set GOOGLE_APPLICATION_CREDENTIALS or place credentials/kestrel-agent-admin.json"
        )
        self._credentials_file = None

    def _load_profiles(self, raw_profiles: Dict[str, Any]) -> Dict[str, GPUProfile]:
        """Load GPU profiles from config."""
        profiles: Dict[str, GPUProfile] = {}
        for key, data in raw_profiles.items():
            try:
                profiles[key] = GPUProfile(
                    id=data.get("id", key),
                    name=data["name"],
                    task_type=data.get("task_type", key),
                    machine_type=data["machine_type"],
                    gpu_type=data["gpu_type"],
                    gpu_count=int(data.get("gpu_count", 1)),
                    boot_disk_size_gb=int(data.get("boot_disk_size_gb", 100)),
                    boot_disk_type=data.get("boot_disk_type", "pd-ssd"),
                    image_name=data["image_name"],
                    persistent_disk=data.get("persistent_disk"),
                    inference_port=int(data.get("inference_port", 8000)),
                    inference_protocol=data.get("inference_protocol", "http"),
                    inference_base_path=data.get("inference_base_path", ""),
                    default_model=data.get("default_model"),
                    cost_per_hr_on_demand=data.get("cost_per_hr_on_demand"),
                    cost_per_hr_spot=data.get("cost_per_hr_spot"),
                    readiness_timeout_seconds=int(
                        data.get("readiness_timeout_seconds", 600)
                    ),
                    env=data.get("env", {}),
                )
            except KeyError as exc:
                raise GCPComputeManagerError(
                    f"Incomplete profile '{key}': missing {exc}"
                ) from exc
        return profiles

    def _get_instances_client(self):
        """Lazy-load the GCP Compute Engine instances client."""
        if self._instances_client is None:
            if not self.project_id:
                raise GCPComputeManagerError("GCP_PROJECT_ID is required")
            try:
                from google.cloud import compute_v1

                self._instances_client = compute_v1.InstancesClient()
            except ImportError:
                raise GCPComputeManagerError(
                    "google-cloud-compute not installed. Run: pip install google-cloud-compute"
                )
        return self._instances_client

    def _get_disks_client(self):
        """Lazy-load the GCP Compute Engine disks client."""
        if self._disks_client is None:
            if not self.project_id:
                raise GCPComputeManagerError("GCP_PROJECT_ID is required")
            try:
                from google.cloud import compute_v1

                self._disks_client = compute_v1.DisksClient()
            except ImportError:
                raise GCPComputeManagerError(
                    "google-cloud-compute not installed. Run: pip install google-cloud-compute"
                )
        return self._disks_client

    def _select_profile(self, profile_name: str) -> GPUProfile:
        """Select a profile by name."""
        if profile_name not in self.profiles:
            available = ", ".join(self.profiles.keys())
            raise GCPComputeManagerError(
                f"Unknown profile '{profile_name}'. Available: {available}"
            )
        return self.profiles[profile_name]

    def _validate_ttl(self, ttl: Optional[int]) -> int:
        """Validate and constrain TTL."""
        if ttl is None:
            return self.default_ttl_seconds
        return min(max(60, ttl), self.max_ttl_seconds)

    def _generate_instance_name(self, profile: GPUProfile) -> str:
        """Generate a unique instance name."""
        import uuid

        suffix = uuid.uuid4().hex[:8]
        return f"kestrel-{profile.task_type}-{suffix}"

    def _build_startup_script(
        self, profile: GPUProfile, env_vars: Dict[str, str]
    ) -> str:
        """Build startup script for the instance."""
        mount_path = self.disk_config.get("mount_path", "/workspace")
        disk_name = profile.persistent_disk or self.disk_config.get("name")

        script_parts = [
            "#!/bin/bash",
            "set -e",
            "",
            "# Log startup",
            'echo "Kestrel GPU instance starting..." | tee /var/log/kestrel-startup.log',
            "",
        ]

        # Mount persistent disk if specified
        if disk_name:
            script_parts.extend(
                [
                    "# Mount persistent disk",
                    f'DEVICE=$(readlink -f /dev/disk/by-id/google-{disk_name} 2>/dev/null || echo "")',
                    'if [ -n "$DEVICE" ]; then',
                    f"    mkdir -p {mount_path}",
                    f"    if ! mount | grep -q {mount_path}; then",
                    f"        mount -o discard,defaults $DEVICE {mount_path}",
                    f'        echo "Mounted persistent disk to {mount_path}"',
                    "    fi",
                    "else",
                    f'    echo "WARNING: Persistent disk {disk_name} not found"',
                    "fi",
                    "",
                ]
            )

        env_vars = _sanitize_env_vars(env_vars)

        # Set environment variables
        script_parts.append("# Set environment variables")
        for key, value in env_vars.items():
            # Expand env var references
            if value.startswith("${") and value.endswith("}"):
                env_name = value[2:-1]
                value = os.getenv(env_name, "")
            script_parts.append(f'export {key}="{value}"')
        script_parts.append("")

        # Install NVIDIA container toolkit if needed
        script_parts.extend(
            [
                "# Ensure NVIDIA container toolkit is available",
                "if ! command -v nvidia-container-cli &> /dev/null; then",
                "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
                '    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list',
                "    apt-get update && apt-get install -y nvidia-container-toolkit",
                "    nvidia-ctk runtime configure --runtime=docker",
                "    systemctl restart docker",
                "fi",
                "",
            ]
        )

        # Pull and run the container
        script_parts.extend(
            [
                "# Configure GCR authentication",
                "gcloud auth configure-docker gcr.io --quiet",
                "",
                f"# Pull container image",
                f"docker pull {profile.image_name}",
                "",
                "# Run container with GPU support",
                f"docker run -d --gpus all \\",
                f"    --name kestrel-workload \\",
                f"    -v {mount_path}:{mount_path} \\",
                f"    -p {profile.inference_port}:{profile.inference_port} \\",
            ]
        )

        # Add environment variables to docker run
        for key, value in env_vars.items():
            if value.startswith("${") and value.endswith("}"):
                env_name = value[2:-1]
                value = os.getenv(env_name, "")
            script_parts.append(f'    -e {key}="{value}" \\')

        script_parts.append(f"    {profile.image_name}")
        script_parts.extend(
            [
                "",
                'echo "Container started successfully"',
                'echo "KESTREL_READY" >> /var/log/kestrel-startup.log',
            ]
        )

        return "\n".join(script_parts)

    async def ensure_persistent_disk(
        self, disk_name: Optional[str] = None, zone: Optional[str] = None
    ) -> bool:
        """
        Ensure persistent disk exists, create if not.

        Args:
            disk_name: Disk name (defaults to config)
            zone: Zone for the disk (defaults to config)

        Returns:
            True if disk exists or was created
        """
        disk_name = disk_name or self.disk_config.get("name")
        zone = zone or self.disk_config.get("zone", self.default_zone)

        if not disk_name:
            logger.info("No persistent disk configured")
            return False

        client = self._get_disks_client()

        # Check if disk exists
        try:
            disk = await asyncio.to_thread(
                client.get, project=self.project_id, zone=zone, disk=disk_name
            )
            logger.info(f"Persistent disk {disk_name} exists ({disk.size_gb}GB)")
            return True
        except ImportError as e:
            raise GCPComputeManagerError(f"google-cloud-compute not installed: {e}") from e
        except Exception as e:
            if "404" not in str(e) and "not found" not in str(e).lower():
                raise GCPComputeManagerError(f"Failed to check disk: {e}") from e

        # Create the disk
        logger.info(f"Creating persistent disk {disk_name}...")

        try:
            from google.cloud import compute_v1

            disk_config = compute_v1.Disk()
            disk_config.name = disk_name
            disk_config.size_gb = self.disk_config.get("size_gb", 200)
            disk_config.type_ = (
                f"zones/{zone}/diskTypes/{self.disk_config.get('type', 'pd-ssd')}"
            )

            operation = await asyncio.to_thread(
                client.insert, project=self.project_id, zone=zone, disk_resource=disk_config
            )

            # Wait for operation to complete
            await self._wait_for_operation(operation.name, zone)

            logger.info(f"Created persistent disk {disk_name}")
            return True

        except ImportError as e:
            raise GCPComputeManagerError(f"google-cloud-compute not installed: {e}") from e
        except Exception as e:
            raise GCPComputeManagerError(f"Failed to create disk: {e}") from e

    async def _wait_for_operation(
        self, operation_name: str, zone: str, timeout: int = 300
    ) -> None:
        """Wait for a GCP operation to complete."""
        try:
            from google.cloud import compute_v1

            operations_client = compute_v1.ZoneOperationsClient()

            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < timeout:
                operation = await asyncio.to_thread(
                    operations_client.get,
                    project=self.project_id,
                    zone=zone,
                    operation=operation_name,
                )

                if operation.status == compute_v1.Operation.Status.DONE:
                    if operation.error:
                        errors = [e.message for e in operation.error.errors]
                        raise GCPComputeManagerError(
                            f"Operation failed: {', '.join(errors)}"
                        )
                    return

                await asyncio.sleep(GCP_OPERATION_POLL_INTERVAL)

            raise GCPComputeManagerError(f"Operation {operation_name} timed out")

        except ImportError:
            raise GCPComputeManagerError("google-cloud-compute not installed")

    async def _find_existing_instance_with_disk(
        self, disk_name: str, zone: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find an existing running instance that has the specified disk attached.

        Args:
            disk_name: Name of the persistent disk to look for
            zone: Zone to search in

        Returns:
            Instance info dict if found, None otherwise
        """
        try:
            client = self._get_instances_client()
            from google.cloud import compute_v1

            # List instances in the zone that match our naming pattern
            request = compute_v1.ListInstancesRequest(
                project=self.project_id,
                zone=zone,
                filter='name:kestrel-*',
            )

            for instance in client.list(request=request):
                # Check if this instance has our disk attached
                if instance.status in ("RUNNING", "STAGING", "PROVISIONING"):
                    for disk in instance.disks:
                        if disk.source and disk_name in disk.source:
                            # Found a running instance with our disk
                            external_ip = None
                            internal_ip = None
                            if instance.network_interfaces:
                                ni = instance.network_interfaces[0]
                                internal_ip = ni.network_i_p
                                if ni.access_configs:
                                    external_ip = ni.access_configs[0].nat_i_p

                            return {
                                "name": instance.name,
                                "zone": zone,
                                "status": instance.status,
                                "machine_type": instance.machine_type.split("/")[-1],
                                "external_ip": external_ip,
                                "internal_ip": internal_ip,
                                "is_spot": (
                                    instance.scheduling
                                    and instance.scheduling.provisioning_model == "SPOT"
                                ),
                            }

            return None

        except ImportError as e:
            logger.error(f"google-cloud-compute not installed: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.warning(f"Error finding existing instance: {e}", exc_info=True)
            return None

    async def _adopt_existing_instance(
        self,
        instance_info: Dict[str, Any],
        profile: GPUProfile,
        task_profile: str,
        chosen_model: Optional[str],
        ttl: int,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Adopt an existing running instance as our session.

        Args:
            instance_info: Info about the existing instance
            profile: GPU profile to use
            task_profile: Task profile name
            chosen_model: Model name
            ttl: TTL for the session
            metadata: Additional metadata

        Returns:
            Session status dict
        """
        started_at = datetime.now(timezone.utc)
        is_spot = instance_info.get("is_spot", False)
        cost_per_hr = (
            profile.cost_per_hr_spot if is_spot else profile.cost_per_hr_on_demand
        )
        external_ip = instance_info.get("external_ip")

        self._session = GCPComputeSession(
            instance_name=instance_info["name"],
            zone=instance_info["zone"],
            profile=profile,
            task_profile=task_profile,
            model_name=chosen_model,
            status=InstanceStatus.from_gcp_status(instance_info["status"]),
            ttl_seconds=ttl,
            started_at=started_at,
            expires_at=started_at + timedelta(seconds=ttl),
            is_spot=is_spot,
            external_ip=external_ip,
            internal_ip=instance_info.get("internal_ip"),
            ssh_host=external_ip,
            actual_cost_per_hr=cost_per_hr,
            persistent_disk_attached=True,
            runtime=metadata,
        )

        if external_ip:
            self._session.backend_base_url = (
                f"{profile.inference_protocol}://{external_ip}:{profile.inference_port}"
            )
            self._session.inference_url = (
                f"{self._session.backend_base_url}{profile.inference_base_path}"
            )

        logger.info(
            f"Adopted existing GCP instance {instance_info['name']} at {external_ip}"
        )

        # Wait for it to be ready (in case it's still starting up)
        if self._session.status != InstanceStatus.RUNNING:
            await self._wait_for_ready(self._session)

        return self._session.to_dict()

    async def start_session(
        self,
        task_profile: str,
        model_name: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        use_spot: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Start a new GCP Compute GPU session.

        Args:
            task_profile: Profile name from gcp_compute_config.toml
            model_name: Model to use (defaults to profile default)
            ttl_seconds: Session TTL (for tracking)
            use_spot: Use spot instance (defaults to config prefer_spot)
            metadata: Additional metadata for the session

        Returns:
            Session status dict
        """
        profile = self._select_profile(task_profile)
        ttl = self._validate_ttl(ttl_seconds)
        chosen_model = model_name or profile.default_model
        use_spot = use_spot if use_spot is not None else self.prefer_spot
        metadata = metadata or {}

        async with self._lock:
            if self._session and self._session.is_active:
                raise GCPComputeManagerError("A GCP session is already active")

            zone = metadata.get("zone", self.default_zone)

            # Check for existing running instance with the disk attached
            if profile.persistent_disk:
                existing = await self._find_existing_instance_with_disk(
                    profile.persistent_disk, zone
                )
                if existing:
                    logger.info(
                        f"Found existing instance {existing['name']} with disk "
                        f"{profile.persistent_disk} attached - reusing"
                    )
                    return await self._adopt_existing_instance(
                        existing, profile, task_profile, chosen_model, ttl, metadata
                    )

            # Ensure persistent disk exists if needed
            if profile.persistent_disk:
                await self.ensure_persistent_disk(profile.persistent_disk)

            client = self._get_instances_client()
            instance_name = self._generate_instance_name(profile)

            # Build environment variables
            env_vars = _sanitize_env_vars(
                {**profile.env, **metadata.get("env_overrides", {})}
            )

            # Build instance config
            try:
                from google.cloud import compute_v1

                instance = compute_v1.Instance()
                instance.name = instance_name
                instance.machine_type = f"zones/{zone}/machineTypes/{profile.machine_type}"

                # Configure boot disk
                boot_disk = compute_v1.AttachedDisk()
                boot_disk.boot = True
                boot_disk.auto_delete = True
                boot_disk.initialize_params = compute_v1.AttachedDiskInitializeParams()
                boot_disk.initialize_params.disk_size_gb = profile.boot_disk_size_gb
                boot_disk.initialize_params.disk_type = (
                    f"zones/{zone}/diskTypes/{profile.boot_disk_type}"
                )
                # Use Google's Deep Learning VM image (PyTorch 2.7 + CUDA 12.8)
                boot_disk.initialize_params.source_image = (
                    "projects/deeplearning-platform-release/global/images/family/pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
                )
                instance.disks = [boot_disk]

                # Attach persistent disk if specified
                if profile.persistent_disk:
                    data_disk = compute_v1.AttachedDisk()
                    data_disk.boot = False
                    data_disk.auto_delete = False
                    data_disk.source = f"projects/{self.project_id}/zones/{zone}/disks/{profile.persistent_disk}"
                    data_disk.device_name = profile.persistent_disk
                    instance.disks.append(data_disk)

                # Configure GPU
                gpu = compute_v1.AcceleratorConfig()
                gpu.accelerator_type = (
                    f"zones/{zone}/acceleratorTypes/{profile.gpu_type}"
                )
                gpu.accelerator_count = profile.gpu_count
                instance.guest_accelerators = [gpu]

                # Required for GPU instances
                instance.scheduling = compute_v1.Scheduling()
                instance.scheduling.on_host_maintenance = "TERMINATE"

                # Use spot/preemptible if requested
                if use_spot:
                    instance.scheduling.provisioning_model = "SPOT"
                    instance.scheduling.instance_termination_action = "STOP"

                # Network config with external IP
                network_interface = compute_v1.NetworkInterface()
                network_interface.network = "global/networks/default"
                access_config = compute_v1.AccessConfig()
                access_config.name = "External NAT"
                access_config.type_ = "ONE_TO_ONE_NAT"
                network_interface.access_configs = [access_config]
                instance.network_interfaces = [network_interface]

                # Service account for GCR access
                service_account = compute_v1.ServiceAccount()
                service_account.email = "default"
                service_account.scopes = [
                    "https://www.googleapis.com/auth/cloud-platform",
                    "https://www.googleapis.com/auth/devstorage.read_only",
                ]
                instance.service_accounts = [service_account]

                # Startup script
                startup_script = self._build_startup_script(profile, env_vars)
                instance.metadata = compute_v1.Metadata()
                instance.metadata.items = [
                    compute_v1.Items(key="startup-script", value=startup_script)
                ]

                # Add SSH key if available
                if os.path.exists(self.ssh_key_file + ".pub"):
                    with open(self.ssh_key_file + ".pub", encoding="utf-8") as f:
                        ssh_key = f.read().strip()
                    instance.metadata.items.append(
                        compute_v1.Items(
                            key="ssh-keys", value=f"{self.ssh_user}:{ssh_key}"
                        )
                    )

                # Create the instance
                logger.info(
                    f"Creating GCP instance {instance_name} ({profile.machine_type} + {profile.gpu_type})"
                )
                operation = await asyncio.to_thread(
                    client.insert,
                    project=self.project_id,
                    zone=zone,
                    instance_resource=instance,
                )

                # Wait for instance creation
                await self._wait_for_operation(operation.name, zone)

                # Get instance details
                instance_info = await asyncio.to_thread(
                    client.get, project=self.project_id, zone=zone, instance=instance_name
                )

                external_ip = None
                internal_ip = None
                if instance_info.network_interfaces:
                    ni = instance_info.network_interfaces[0]
                    internal_ip = ni.network_i_p
                    if ni.access_configs:
                        external_ip = ni.access_configs[0].nat_i_p

            except ImportError as e:
                raise GCPComputeManagerError(f"google-cloud-compute not installed: {e}") from e
            except OSError as e:
                raise GCPComputeManagerError(f"Failed to read SSH key: {e}") from e
            except Exception as e:
                raise GCPComputeManagerError(f"Failed to create instance: {e}") from e

            started_at = datetime.now(timezone.utc)
            cost_per_hr = (
                profile.cost_per_hr_spot if use_spot else profile.cost_per_hr_on_demand
            )

            self._session = GCPComputeSession(
                instance_name=instance_name,
                zone=zone,
                profile=profile,
                task_profile=task_profile,
                model_name=chosen_model,
                status=InstanceStatus.PROVISIONING,
                ttl_seconds=ttl,
                started_at=started_at,
                expires_at=started_at + timedelta(seconds=ttl),
                is_spot=use_spot,
                external_ip=external_ip,
                internal_ip=internal_ip,
                ssh_host=external_ip,
                actual_cost_per_hr=cost_per_hr,
                persistent_disk_attached=bool(profile.persistent_disk),
                runtime=metadata,
            )

            if external_ip:
                self._session.backend_base_url = (
                    f"{profile.inference_protocol}://{external_ip}:{profile.inference_port}"
                )
                self._session.inference_url = (
                    f"{self._session.backend_base_url}{profile.inference_base_path}"
                )

            logger.info(f"Created GCP instance {instance_name} at {external_ip}")

            # Wait for instance to be ready
            await self._wait_for_ready(self._session)

            return self._session.to_dict()

    async def _wait_for_ready(
        self, session: GCPComputeSession, timeout: Optional[int] = None
    ) -> None:
        """Wait for instance to be fully ready."""
        import httpx

        timeout = timeout or session.profile.readiness_timeout_seconds
        start_time = datetime.now()

        logger.info(f"Waiting for instance {session.instance_name} to be ready...")

        while (datetime.now() - start_time).total_seconds() < timeout:
            # Update status from GCP
            try:
                client = self._get_instances_client()
                instance = await asyncio.to_thread(
                    client.get,
                    project=self.project_id,
                    zone=session.zone,
                    instance=session.instance_name,
                )
                session.status = InstanceStatus.from_gcp_status(instance.status)

                if session.status == InstanceStatus.RUNNING:
                    # Check if container is ready via HTTP
                    if session.backend_base_url:
                        try:
                            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SHORT) as http:
                                resp = await http.get(
                                    f"{session.backend_base_url}/health"
                                )
                                if resp.status_code < 500:
                                    logger.info(
                                        f"Instance {session.instance_name} is ready"
                                    )
                                    return
                        except (OSError, ConnectionError) as e:
                            logger.debug(f"Container not ready yet: {e}")
                        except Exception as e:
                            logger.debug(f"Unexpected error checking container readiness: {e}", exc_info=True)

            except ImportError as e:
                logger.error(f"google-cloud-compute not installed: {e}", exc_info=True)
            except Exception as e:
                logger.warning(f"Error checking instance status: {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)

        logger.warning(
            f"Instance {session.instance_name} did not become ready within {timeout}s"
        )

    async def get_status(self, refresh: bool = True) -> Dict[str, Any]:
        """
        Get current session status.

        Args:
            refresh: Whether to refresh status from GCP API

        Returns:
            Session status dict or offline status
        """
        if not self._session:
            return {"status": "offline", "message": "No active session"}

        if refresh:
            try:
                client = self._get_instances_client()
                instance = await asyncio.to_thread(
                    client.get,
                    project=self.project_id,
                    zone=self._session.zone,
                    instance=self._session.instance_name,
                )
                self._session.status = InstanceStatus.from_gcp_status(instance.status)

                # Update IP if needed
                if instance.network_interfaces:
                    ni = instance.network_interfaces[0]
                    self._session.internal_ip = ni.network_i_p
                    if ni.access_configs:
                        self._session.external_ip = ni.access_configs[0].nat_i_p
                        self._session.ssh_host = self._session.external_ip

            except ImportError as e:
                logger.error(f"google-cloud-compute not installed: {e}", exc_info=True)
                self._session.status = InstanceStatus.ERROR
            except Exception as e:
                logger.error(f"Failed to refresh status: {e}", exc_info=True)
                self._session.status = InstanceStatus.ERROR

        return self._session.to_dict()

    async def stop_session(self) -> Dict[str, Any]:
        """
        Stop and terminate the current session.

        Returns:
            Final session status
        """
        if not self._session:
            return {"status": "offline", "message": "No active session"}

        async with self._lock:
            session = self._session

            try:
                client = self._get_instances_client()

                logger.info(f"Deleting GCP instance {session.instance_name}")
                operation = await asyncio.to_thread(
                    client.delete,
                    project=self.project_id,
                    zone=session.zone,
                    instance=session.instance_name,
                )

                await self._wait_for_operation(operation.name, session.zone)

                session.status = InstanceStatus.TERMINATED
                logger.info(f"Deleted instance {session.instance_name}")

            except ImportError as e:
                logger.error(f"google-cloud-compute not installed: {e}", exc_info=True)
                session.status = InstanceStatus.ERROR
            except Exception as e:
                logger.error(f"Failed to delete instance: {e}", exc_info=True)
                session.status = InstanceStatus.ERROR

            self._session = None
            return session.to_dict()

    async def terminate_session(self, session: GCPComputeSession) -> None:
        """
        Terminate a specific session.

        Args:
            session: Session to terminate
        """
        try:
            client = self._get_instances_client()

            logger.info(f"Deleting GCP instance {session.instance_name}")
            operation = await asyncio.to_thread(
                client.delete,
                project=self.project_id,
                zone=session.zone,
                instance=session.instance_name,
            )

            await self._wait_for_operation(operation.name, session.zone)

            session.status = InstanceStatus.TERMINATED
            logger.info(f"Deleted instance {session.instance_name}")

            if self._session and self._session.instance_name == session.instance_name:
                self._session = None

        except ImportError as e:
            logger.error(f"google-cloud-compute not installed: {e}", exc_info=True)
            session.status = InstanceStatus.ERROR
        except Exception as e:
            logger.error(f"Failed to terminate session: {e}", exc_info=True)
            session.status = InstanceStatus.ERROR

    async def list_instances(self) -> List[Dict[str, Any]]:
        """List all Kestrel instances in the project."""
        try:
            client = self._get_instances_client()
            from google.cloud import compute_v1

            request = compute_v1.AggregatedListInstancesRequest(
                project=self.project_id,
                filter='name:kestrel-*',
            )

            instances = []
            for zone, response in client.aggregated_list(request=request):
                if response.instances:
                    for inst in response.instances:
                        instances.append({
                            "name": inst.name,
                            "zone": zone.split("/")[-1],
                            "status": inst.status,
                            "machine_type": inst.machine_type.split("/")[-1],
                            "created": inst.creation_timestamp,
                        })

            return instances

        except ImportError as e:
            logger.error(f"google-cloud-compute not installed: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Failed to list instances: {e}", exc_info=True)
            return []

    async def list_disks(self) -> List[Dict[str, Any]]:
        """List all Kestrel persistent disks."""
        try:
            client = self._get_disks_client()
            from google.cloud import compute_v1

            request = compute_v1.AggregatedListDisksRequest(
                project=self.project_id,
                filter='name:kestrel-*',
            )

            disks = []
            for zone, response in client.aggregated_list(request=request):
                if response.disks:
                    for disk in response.disks:
                        disks.append({
                            "name": disk.name,
                            "zone": zone.split("/")[-1],
                            "size_gb": disk.size_gb,
                            "status": disk.status,
                            "type": disk.type_.split("/")[-1],
                        })

            return disks

        except ImportError as e:
            logger.error(f"google-cloud-compute not installed: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Failed to list disks: {e}", exc_info=True)
            return []
