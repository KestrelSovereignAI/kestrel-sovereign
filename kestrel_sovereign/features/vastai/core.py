"""
Vast.ai GPU Instance Manager - Core SDK Operations.

Coordinates Vast.ai GPU instance lifecycles including:
- Instance search and creation
- Session management
- Profile configuration
- Status monitoring

Key differences from RunPod:
- Marketplace model (peer-to-peer) vs fixed pricing
- Local volumes only (tied to physical machine)
- No pod resume feature - must create new instances
- Generally cheaper but variable reliability
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from kestrel_sovereign.config import load_config

from .models import (
    GPUProfile,
    InstanceStatus,
    VastAIManagerError,
    VastAISession,
)

logger = logging.getLogger(__name__)


def _sanitize_env_vars(env_vars: Dict[str, Any]) -> Dict[str, str]:
    """Drop unset environment values before building Vast.ai env strings."""
    return {
        key: str(value)
        for key, value in env_vars.items()
        if value is not None
    }


class VastAIManagerCore:
    """Core Vast.ai SDK operations and session management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("vastai_config.toml")
        self.manager_config = self.config.get("manager", {})

        self.api_key = os.getenv("VASTAI_API_KEY") or self.manager_config.get("api_key")
        if not self.api_key:
            logger.warning("VASTAI_API_KEY not set - Vast.ai features will be unavailable")

        self.default_ttl_seconds = int(
            os.getenv("VASTAI_DEFAULT_TTL_SECONDS", self.manager_config.get("default_ttl_seconds", 3600))
        )
        self.max_ttl_seconds = int(self.manager_config.get("max_ttl_seconds", 7200))
        self.poll_interval = int(self.manager_config.get("poll_interval_seconds", 15))
        self.readiness_timeout = int(self.manager_config.get("readiness_timeout_seconds", 600))

        self.profiles = self._load_profiles(self.config.get("profiles", {}))
        self._sdk: Optional[Any] = None
        self._session: Optional[VastAISession] = None
        self._lock = asyncio.Lock()

    def _load_profiles(self, raw_profiles: Dict[str, Any]) -> Dict[str, GPUProfile]:
        profiles: Dict[str, GPUProfile] = {}
        for key, data in raw_profiles.items():
            try:
                profiles[key] = GPUProfile(
                    id=data.get("id", key),
                    name=data["name"],
                    task_type=data.get("task_type", key),
                    image_name=data["image_name"],
                    disk_gb=int(data.get("disk_gb", 50)),
                    gpu_ram_min=int(data.get("gpu_ram_min", 16)),
                    num_gpus=int(data.get("num_gpus", 1)),
                    reliability_min=float(data.get("reliability_min", 0.9)),
                    compute_cap_min=int(data.get("compute_cap_min", 0)),
                    cuda_vers_min=float(data.get("cuda_vers_min", 11.0)),
                    ports=data.get("ports", ["8888/http"]),
                    inference_port=int(data.get("inference_port", 8888)),
                    inference_protocol=data.get("inference_protocol", "http"),
                    inference_base_path=data.get("inference_base_path", "/v1"),
                    onstart_cmd=data.get("onstart_cmd"),
                    default_model=data.get("default_model"),
                    cost_per_hr_max=data.get("cost_per_hr_max"),
                    env=data.get("env", {}),
                    docker_login=data.get("docker_login"),
                )
            except KeyError as exc:
                raise VastAIManagerError(f"Incomplete profile '{key}': missing {exc}") from exc
        return profiles

    def _get_sdk(self) -> Any:
        """Lazy-load the Vast.ai SDK."""
        if self._sdk is None:
            if not self.api_key:
                raise VastAIManagerError("VASTAI_API_KEY is required")
            try:
                from vastai_sdk import VastAI
                self._sdk = VastAI(api_key=self.api_key)
            except ImportError:
                raise VastAIManagerError(
                    "vastai_sdk not installed. Run: pip install vastai-sdk"
                )
        return self._sdk

    def _build_search_query(self, profile: GPUProfile) -> str:
        """Build a Vast.ai search query from profile requirements."""
        conditions = []

        if profile.gpu_ram_min:
            conditions.append(f"gpu_ram >= {profile.gpu_ram_min}")
        if profile.num_gpus > 1:
            conditions.append(f"num_gpus >= {profile.num_gpus}")
        if profile.reliability_min:
            conditions.append(f"reliability > {profile.reliability_min}")
        if profile.compute_cap_min:
            conditions.append(f"compute_cap > {profile.compute_cap_min}")
        if profile.cuda_vers_min:
            conditions.append(f"cuda_vers >= {profile.cuda_vers_min}")
        if profile.cost_per_hr_max:
            conditions.append(f"dph <= {profile.cost_per_hr_max}")

        # Always filter for rentable instances
        conditions.append("rentable = true")

        return " ".join(conditions)

    async def search_offers(
        self,
        profile: Optional[GPUProfile] = None,
        query: Optional[str] = None,
        limit: int = 10,
        sort_by: str = "dph+",  # Sort by price ascending
    ) -> List[Dict[str, Any]]:
        """
        Search for available GPU instances.

        Args:
            profile: GPU profile to use for search criteria
            query: Custom search query (overrides profile)
            limit: Maximum results to return
            sort_by: Sort order (dph+, dph-, num_gpus-, reliability-)

        Returns:
            List of available offers
        """
        sdk = self._get_sdk()

        if query is None and profile:
            query = self._build_search_query(profile)
        elif query is None:
            query = "rentable = true"

        logger.info(f"Searching Vast.ai offers: {query}")

        try:
            # The SDK returns offers directly
            offers = await asyncio.to_thread(sdk.search_offers, query=query)

            if isinstance(offers, dict) and "offers" in offers:
                offers = offers["offers"]

            if not offers:
                return []

            # Sort results
            if sort_by == "dph+":
                offers.sort(key=lambda x: x.get("dph_total", float("inf")))
            elif sort_by == "dph-":
                offers.sort(key=lambda x: x.get("dph_total", 0), reverse=True)
            elif sort_by == "num_gpus-":
                offers.sort(key=lambda x: x.get("num_gpus", 0), reverse=True)
            elif sort_by == "reliability-":
                offers.sort(key=lambda x: x.get("reliability", 0), reverse=True)

            return offers[:limit]

        except Exception as e:
            logger.error(f"Failed to search Vast.ai offers: {e}")
            raise VastAIManagerError(f"Search failed: {e}") from e

    async def start_session(
        self,
        task_profile: str,
        model_name: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        offer_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Start a new Vast.ai GPU session.

        Args:
            task_profile: Profile name from vastai_config.toml
            model_name: Model to use (defaults to profile default)
            ttl_seconds: Session TTL (for tracking, not enforced by Vast.ai)
            offer_id: Specific offer ID to use (skips search)
            metadata: Additional metadata for the session

        Returns:
            Session status dict
        """
        profile = self._select_profile(task_profile)
        ttl = self._validate_ttl(ttl_seconds)
        chosen_model = model_name or profile.default_model
        metadata = metadata or {}

        async with self._lock:
            if self._session and self._session.is_active:
                raise VastAIManagerError("A Vast.ai session is already active")

            sdk = self._get_sdk()

            # Find an offer if not specified
            if offer_id is None:
                offers = await self.search_offers(profile=profile, limit=1)
                if not offers:
                    raise VastAIManagerError(
                        f"No available instances matching profile '{task_profile}'"
                    )
                offer_id = offers[0]["id"]
                logger.info(f"Selected offer {offer_id}: {offers[0].get('gpu_name', 'Unknown GPU')}")

            # Build environment variables as "-e KEY=VAL -e KEY2=VAL2" string
            env_vars = _sanitize_env_vars(
                {**profile.env, **metadata.get("env_overrides", {})}
            )
            env_string = None
            if env_vars:
                # SDK expects env as string: "-e KEY=VAL -e KEY2=VAL2"
                env_parts = [f"-e {k}={v}" for k, v in env_vars.items()]
                env_string = " ".join(env_parts)

            # Build docker login string for private registry auth
            docker_login = self._build_docker_login(profile)

            # Create the instance
            try:
                logger.info(f"Creating Vast.ai instance from offer {offer_id}")
                create_kwargs = {
                    "id": offer_id,  # SDK uses lowercase 'id'
                    "image": profile.image_name,
                    "disk": profile.disk_gb,
                    "onstart": profile.onstart_cmd,
                    "label": metadata.get("label", f"kestrel-{profile.id}"),
                }
                if env_string:
                    create_kwargs["env"] = env_string
                if docker_login:
                    create_kwargs["login"] = docker_login
                    logger.info(f"Using private registry auth for {profile.image_name}")

                result = await asyncio.to_thread(
                    sdk.create_instance,
                    **create_kwargs,
                )

                # Handle various SDK return types
                instance_id = self._parse_instance_id(result, sdk)

                if not instance_id:
                    raise VastAIManagerError(f"Failed to create instance: {result}")

            except Exception as e:
                raise VastAIManagerError(f"Failed to create instance: {e}") from e

            started_at = datetime.now(timezone.utc)
            self._session = VastAISession(
                instance_id=instance_id,
                profile=profile,
                task_profile=task_profile,
                model_name=chosen_model,
                status=InstanceStatus.CREATING,
                ttl_seconds=ttl,
                started_at=started_at,
                expires_at=started_at + timedelta(seconds=ttl),
            )

        # Wait for instance to be ready
        await self._wait_until_ready()
        return await self.get_status()

    def _parse_instance_id(self, result: Any, sdk: Any) -> Optional[int]:
        """Parse instance ID from SDK create_instance response."""
        instance_id = None
        if isinstance(result, dict):
            instance_id = result.get("new_contract") or result.get("instance_id")
        elif isinstance(result, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    instance_id = parsed.get("new_contract") or parsed.get("instance_id")
                elif isinstance(parsed, (int, str)):
                    instance_id = int(parsed)
            except (json.JSONDecodeError, ValueError):
                # Not JSON - might be the instance ID directly or an error message
                if result.isdigit():
                    instance_id = int(result)
                else:
                    # Check if it contains "new_contract" in the string (SDK output)
                    logger.debug(f"SDK returned string: {result}")
                    # Try to extract instance ID from output like "{'new_contract': 12345}"
                    match = re.search(r"['\"]?new_contract['\"]?\s*:\s*(\d+)", result)
                    if match:
                        instance_id = int(match.group(1))
                    else:
                        # Check sdk.last_output for actual response
                        if hasattr(sdk, 'last_output') and sdk.last_output:
                            logger.debug(f"SDK last_output: {sdk.last_output}")
                            match = re.search(r"['\"]?new_contract['\"]?\s*:\s*(\d+)", sdk.last_output)
                            if match:
                                instance_id = int(match.group(1))
        elif isinstance(result, int):
            instance_id = result
        return instance_id

    async def get_status(self, refresh: bool = True) -> Dict[str, Any]:
        """Get current session status."""
        async with self._lock:
            session = self._session

        if not session:
            return {"active": False, "status": InstanceStatus.OFFLINE.value}

        if refresh:
            await self._refresh_session_status(session)

        payload = session.to_dict()
        payload["active"] = session.is_active
        return payload

    async def _refresh_session_status(self, session: VastAISession) -> None:
        """Refresh session status from Vast.ai API."""
        sdk = self._get_sdk()

        try:
            instance_info = await asyncio.to_thread(
                sdk.show_instance, id=session.instance_id
            )

            if not instance_info:
                session.status = InstanceStatus.OFFLINE
                return

            self._update_session_from_runtime(session, instance_info)

        except Exception as e:
            logger.error(f"Failed to get instance status: {e}")
            session.status = InstanceStatus.ERROR

    def _update_session_from_runtime(self, session: VastAISession, info: Dict[str, Any]) -> None:
        """Update session fields from instance info."""
        session.runtime = info

        # Map status
        raw_status = info.get("actual_status", info.get("cur_state", ""))
        session.status = self._map_status(raw_status)

        # Update connection info
        session.ssh_host = info.get("ssh_host")
        session.ssh_port = info.get("ssh_port")
        session.gpu_name = info.get("gpu_name")
        session.actual_cost_per_hr = info.get("dph_total")

        # Build backend URL from public IP and port
        public_ip = info.get("public_ipaddr")
        ports = info.get("ports", {})

        if public_ip and ports:
            # Find the inference port mapping
            port_key = f"{session.profile.inference_port}/tcp"
            if port_key in ports:
                mapped_ports = ports[port_key]
                if mapped_ports:
                    external_port = mapped_ports[0].get("HostPort")
                    if external_port:
                        base_url = f"{session.profile.inference_protocol}://{public_ip}:{external_port}"
                        session.backend_base_url = base_url
                        session.inference_url = f"{base_url}{session.profile.inference_base_path}".rstrip("/")

        # Check TTL expiration
        if session.remaining_ttl_seconds == 0:
            session.status = InstanceStatus.STOPPING

    @staticmethod
    def _map_status(raw_status: Optional[str]) -> InstanceStatus:
        """Map Vast.ai status string to InstanceStatus enum."""
        normalized = (raw_status or "").lower()

        if normalized in {"running", "ready"}:
            return InstanceStatus.RUNNING
        if normalized in {"creating", "starting", "provisioning"}:
            return InstanceStatus.CREATING
        if normalized in {"loading", "pulling"}:
            return InstanceStatus.LOADING
        if normalized in {"stopping", "exited", "stopped"}:
            return InstanceStatus.EXITED
        if normalized in {"error", "failed"}:
            return InstanceStatus.ERROR

        return InstanceStatus.OFFLINE

    async def stop_session(self) -> Dict[str, Any]:
        """Stop and destroy the current session."""
        async with self._lock:
            session = self._session
            if not session:
                return {"active": False, "status": InstanceStatus.OFFLINE.value}
            self._session = None

        sdk = self._get_sdk()

        try:
            await asyncio.to_thread(sdk.destroy_instance, id=session.instance_id)
            logger.info(f"Destroyed Vast.ai instance {session.instance_id}")
        except Exception as e:
            logger.error(f"Failed to destroy instance {session.instance_id}: {e}")

        session.status = InstanceStatus.EXITED
        payload = session.to_dict()
        payload["active"] = False
        return payload

    async def _wait_until_ready(self) -> None:
        """Wait for instance to be ready."""
        async with self._lock:
            session = self._session

        if not session:
            return

        deadline = datetime.now(timezone.utc) + timedelta(seconds=self.readiness_timeout)

        while datetime.now(timezone.utc) < deadline:
            status = await self.get_status(refresh=True)

            if status.get("status") == InstanceStatus.RUNNING.value:
                logger.info(f"Vast.ai instance {session.instance_id} is ready")
                return

            if status.get("status") == InstanceStatus.ERROR.value:
                raise VastAIManagerError("Instance entered error state")

            await asyncio.sleep(self.poll_interval)

        raise VastAIManagerError("Instance did not become ready before timeout")

    def _select_profile(self, task_profile: str) -> GPUProfile:
        """Select a GPU profile by name."""
        profile = self.profiles.get(task_profile)
        if not profile:
            available = list(self.profiles.keys()) if self.profiles else ["none configured"]
            raise VastAIManagerError(
                f"Unknown task_profile '{task_profile}'. Available: {available}"
            )
        return profile

    def _validate_ttl(self, ttl_seconds: Optional[int]) -> int:
        """Validate and return TTL value."""
        ttl = ttl_seconds or self.default_ttl_seconds
        if ttl > self.max_ttl_seconds:
            raise VastAIManagerError(f"TTL {ttl}s exceeds max allowed {self.max_ttl_seconds}s")
        return ttl

    def _build_docker_login(self, profile: GPUProfile) -> Optional[str]:
        """
        Build Docker login string for private registry authentication.

        Supports:
        1. Explicit docker_login in profile config
        2. GCR auth via GCR_SERVICE_ACCOUNT_KEY or GCR_SERVICE_ACCOUNT_KEY_FILE env var

        For GCR, uses _json_key as username and service account JSON as password.
        Format: "-u <user> -p <pass> <registry>"
        """
        # Check for explicit docker_login in profile
        if profile.docker_login:
            return profile.docker_login

        # Check if image is from GCR and we have credentials
        if "gcr.io" in profile.image_name:
            gcr_key = self._load_gcr_key()
            if gcr_key:
                # Escape any special characters in the key
                # The key is JSON, may contain quotes and special chars
                escaped_key = gcr_key.replace("'", "'\"'\"'")
                return f"-u _json_key -p '{escaped_key}' gcr.io"
            else:
                logger.warning(
                    f"Image {profile.image_name} is from GCR but GCR credentials not found. "
                    "Set GCR_SERVICE_ACCOUNT_KEY or GCR_SERVICE_ACCOUNT_KEY_FILE env var."
                )

        return None

    def _load_gcr_key(self) -> Optional[str]:
        """
        Load GCR service account key from env var or file.

        Checks in order:
        1. GCR_SERVICE_ACCOUNT_KEY - raw JSON string
        2. GCR_SERVICE_ACCOUNT_KEY_FILE - path to JSON file
        """
        # Direct JSON key
        gcr_key = os.getenv("GCR_SERVICE_ACCOUNT_KEY")
        if gcr_key:
            return gcr_key

        # Path to JSON file
        key_file = os.getenv("GCR_SERVICE_ACCOUNT_KEY_FILE")
        if key_file:
            try:
                # Handle relative paths from project root
                if not os.path.isabs(key_file):
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    key_file = os.path.join(project_root, key_file)

                with open(key_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Compact JSON to single line
                    parsed = json.loads(content)
                    return json.dumps(parsed, separators=(",", ":"))
            except FileNotFoundError:
                logger.error(f"GCR key file not found: {key_file}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in GCR key file: {e}")
            except Exception as e:
                logger.error(f"Failed to load GCR key file: {e}")

        return None

    async def show_instances(self) -> List[Dict[str, Any]]:
        """List all instances for this account."""
        sdk = self._get_sdk()
        try:
            result = await asyncio.to_thread(sdk.show_instances)
            if isinstance(result, dict) and "instances" in result:
                return result["instances"]
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Failed to list instances: {e}")
            return []

    async def execute_command(self, instance_id: int, command: str) -> str:
        """
        Execute a command on an instance.

        Note: Vast.ai execute only supports ls, rm, du commands on inactive instances.
        For active instances, use SSH.
        """
        sdk = self._get_sdk()
        try:
            result = await asyncio.to_thread(sdk.execute, ID=instance_id, COMMAND=command)
            return str(result)
        except Exception as e:
            logger.error(f"Failed to execute command on {instance_id}: {e}")
            raise VastAIManagerError(f"Command execution failed: {e}") from e

    async def get_ssh_url(self, instance_id: Optional[int] = None) -> Optional[str]:
        """Get SSH URL for an instance."""
        sdk = self._get_sdk()

        if instance_id is None:
            async with self._lock:
                if self._session:
                    instance_id = self._session.instance_id

        if instance_id is None:
            return None

        try:
            result = await asyncio.to_thread(sdk.scp_url, id=instance_id)
            # scp_url returns something like: root@ssh.vast.ai -p 12345
            # We can construct SSH URL from this
            return result
        except Exception as e:
            logger.error(f"Failed to get SSH URL: {e}")
            return None

    async def terminate_session(self, session: VastAISession) -> None:
        """Terminate a specific session."""
        sdk = self._get_sdk()
        try:
            await asyncio.to_thread(sdk.destroy_instance, id=session.instance_id)
            logger.info(f"Terminated instance {session.instance_id}")
        except Exception as e:
            logger.error(f"Failed to terminate {session.instance_id}: {e}")
