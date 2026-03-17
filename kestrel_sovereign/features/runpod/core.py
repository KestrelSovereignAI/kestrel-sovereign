"""
RunPod Core Manager Operations.

Contains the core RunPodManagerCore class with SDK operations,
profile loading, and session management.
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from kestrel_sovereign.config import load_config
from kestrel_sovereign.kestrel_config.constants import RUNPOD_URL_POLL_INTERVAL

from .models import GPUProfile, PodStatus, RunPodManagerError, RunPodSession
from .providers import DirectRunPodProvider, ManagedRunPodProvider, GPUProvider

logger = logging.getLogger(__name__)


class RunPodManagerCore:
    """
    Core RunPod operations.

    Handles SDK initialization, profile loading, session management,
    and basic pod lifecycle.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, mode: Optional[str] = None):
        self.config = config or load_config("runpod_config.toml")
        self.manager_config = self.config.get("manager", {})
        self.mode = mode or os.getenv("RUNPOD_MODE") or self.manager_config.get("mode", "direct")
        self.default_ttl_seconds = int(
            os.getenv(
                "GPU_DEFAULT_TTL_SECONDS",
                self.manager_config.get("default_ttl_seconds", 1800),
            )
        )
        self.max_ttl_seconds = int(self.manager_config.get("max_ttl_seconds", self.default_ttl_seconds))
        self.poll_interval = int(self.manager_config.get("poll_interval_seconds", 10))
        self.readiness_timeout = int(self.manager_config.get("readiness_timeout_seconds", 600))
        self.profiles = self._load_profiles(self.config.get("profiles", {}))
        if not self.profiles:
            raise RunPodManagerError("No GPU profiles configured. Create runpod_config.toml.")
        self.provider = self._build_provider()
        self._session: Optional[RunPodSession] = None
        self._lock = asyncio.Lock()

        # Metering callback for usage billing (Vending Machine)
        # Set via set_metering_callback() after initialization
        self._metering_callback = None

    def set_metering_callback(self, callback) -> None:
        """Set the metering callback for GPU usage billing (Vending Machine).

        The callback will be called when a session ends with:
            await callback(
                companion_id=str,
                user_id=str,
                provider=str,  # 'runpod'
                resource_type=str,  # GPU type
                duration_seconds=float,
                operation_id=str,  # pod_id
            )

        Args:
            callback: Async function to call when session ends
        """
        self._metering_callback = callback
        logger.info("RunPod metering enabled")

    def _load_profiles(self, raw_profiles: Dict[str, Any]) -> Dict[str, GPUProfile]:
        profiles: Dict[str, GPUProfile] = {}
        for key, data in raw_profiles.items():
            try:
                # Expand environment variables in env dict (e.g., "${HF_TOKEN}" -> actual value)
                raw_env = data.get("env", {})
                expanded_env = self._expand_env_vars(raw_env)

                profiles[key] = GPUProfile(
                    id=data.get("id", key),
                    name=data["name"],
                    task_type=data.get("task_type", key),
                    gpu_type_id=data["gpu_type_id"],
                    image_name=data["image_name"],
                    container_disk_gb=int(data.get("container_disk_gb", 50)),
                    volume_gb=int(data.get("volume_gb", 0)),
                    ports=data.get("ports", ["8888/http"]),
                    inference_port=int(data.get("inference_port", 8888)),
                    inference_protocol=data.get("inference_protocol", "http"),
                    inference_base_path=data.get("inference_base_path", "/v1"),
                    image_invoke_path=data.get("image_invoke_path"),
                    default_model=data.get("default_model"),
                    pod_type=data.get("pod_type"),
                    vram_gb=data.get("vram_gb"),
                    cost_per_hr=data.get("cost_per_hr"),
                    max_context_window=data.get("max_context_window"),
                    readiness_timeout_seconds=data.get("readiness_timeout_seconds"),
                    template_id=data.get("template_id"),  # For private registry auth
                    network_volume_id=data.get("network_volume_id"),  # Persistent network storage
                    volume_mount_path=data.get("volume_mount_path"),  # Mount path (e.g., /workspace)
                    persistent_pod_id=data.get("persistent_pod_id"),  # Raw value - expanded at runtime
                    env=expanded_env,
                )
            except KeyError as exc:
                raise RunPodManagerError(f"Incomplete profile '{key}': missing {exc}") from exc
        return profiles

    @staticmethod
    def _expand_single_env_var(value: Optional[str]) -> Optional[str]:
        """Expand ${VAR} syntax in a single string value."""
        if not value or not isinstance(value, str) or "${" not in value:
            return value

        def replace_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, "")  # Empty string if not set

        expanded = re.sub(r'\$\{([^}]+)\}', replace_var, value)
        return expanded if expanded else None  # Return None if result is empty

    @staticmethod
    def _expand_env_vars(env_dict: Dict[str, str]) -> Dict[str, str]:
        """Expand ${VAR} syntax in environment variable values."""
        expanded = {}
        for key, value in env_dict.items():
            if isinstance(value, str) and "${" in value:
                # Expand ${VAR_NAME} patterns
                def replace_var(match):
                    var_name = match.group(1)
                    return os.environ.get(var_name, f"${{{var_name}}}")
                expanded[key] = re.sub(r'\$\{([^}]+)\}', replace_var, value)
            else:
                expanded[key] = value
        return expanded

    def _build_provider(self) -> GPUProvider:
        if self.mode == "managed":
            api_base = os.getenv("KESTREL_API_BASE") or self.manager_config.get("managed_api_base")
            api_key = os.getenv("KESTREL_API_KEY") or self.manager_config.get("managed_api_key")
            return ManagedRunPodProvider(api_base=api_base, api_key=api_key)
        api_key = os.getenv("RUNPOD_API_KEY")
        cloud_type = os.getenv("RUNPOD_CLOUD_TYPE", self.manager_config.get("cloud_type", "COMMUNITY"))
        return DirectRunPodProvider(api_key=api_key, cloud_type=cloud_type)

    async def start_session(
        self,
        task_profile: str,
        model_name: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        pod_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        profile = self._select_profile(task_profile)
        ttl = self._validate_ttl(ttl_seconds)
        chosen_model = model_name or profile.default_model
        if not chosen_model:
            raise RunPodManagerError("Model name is required when profile has no default_model configured")
        metadata = metadata or {}
        async with self._lock:
            if self._session and self._session.is_active:
                raise RunPodManagerError("A RunPod session is already active")
            try:
                response = await asyncio.to_thread(self.provider.start_pod, profile, metadata)
            except Exception as e:
                # Wrap SDK errors in RunPodManagerError for consistent handling
                error_msg = str(e)
                if "no longer any instances available" in error_msg.lower():
                    raise RunPodManagerError(f"No {profile.gpu_type_id} GPUs available: {error_msg}") from e
                raise RunPodManagerError(f"Failed to create pod: {error_msg}") from e
            pod_id = response.get("id") or response.get("podId")
            if not pod_id:
                raise RunPodManagerError("RunPod did not return a pod id")
            started_at = datetime.now(timezone.utc)
            self._session = RunPodSession(
                pod_id=pod_id,
                profile=profile,
                task_profile=task_profile,
                model_name=chosen_model,
                pod_type=pod_type or profile.pod_type,
                status=PodStatus.PROVISIONING,
                ttl_seconds=ttl,
                started_at=started_at,
                expires_at=started_at + timedelta(seconds=ttl),
            )
        await self._wait_until_ready()
        return await self.get_status()

    async def get_status(self, refresh: bool = True) -> Dict[str, Any]:
        async with self._lock:
            session = self._session
        if not session:
            return {"active": False, "status": PodStatus.OFFLINE.value}
        if refresh:
            pod_info = await asyncio.to_thread(self.provider.get_status, session.pod_id)
            self._update_session_from_runtime(session, pod_info)
        payload = session.to_dict()
        payload["active"] = session.is_active
        return payload

    async def stop_session(self) -> Dict[str, Any]:
        async with self._lock:
            session = self._session
            if not session:
                return {"active": False, "status": PodStatus.OFFLINE.value}
            self._session = None
        await asyncio.to_thread(self.provider.stop_pod, session.pod_id)
        session.status = PodStatus.TERMINATING

        # Record GPU usage for billing if metering is enabled
        if self._metering_callback and session.companion_id and session.user_id:
            try:
                duration_seconds = (datetime.now(timezone.utc) - session.started_at).total_seconds()
                await self._metering_callback(
                    companion_id=session.companion_id,
                    user_id=session.user_id,
                    provider="runpod",
                    resource_type=session.profile.gpu_type_id,
                    duration_seconds=duration_seconds,
                    operation_id=session.pod_id,
                )
                logger.info(
                    f"Recorded GPU usage: {duration_seconds:.1f}s on {session.profile.gpu_type_id} "
                    f"for companion {session.companion_id}"
                )
            except Exception as e:
                logger.error(f"Failed to record GPU metering: {e}")

        payload = session.to_dict()
        payload["active"] = False
        return payload

    async def find_stopped_pod(self, purpose: str, profile_name: str) -> Optional[Dict[str, Any]]:
        """
        Find a stopped pod that can be resumed instead of creating a new one.

        Resuming is ~10-30s vs 2-5min for creating new. No cost while stopped.

        Args:
            purpose: e.g. "lora_training" or "lora_inference"
            profile_name: e.g. "training" or "image"

        Returns:
            Pod dict if found, None otherwise
        """
        if not isinstance(self.provider, DirectRunPodProvider):
            return None  # Only works with direct RunPod API

        try:
            all_pods = await asyncio.to_thread(self.provider.list_pods)

            for pod in all_pods:
                # Check if pod is stopped (EXITED status)
                if pod.get('desiredStatus') != 'EXITED':
                    continue

                # Check pod name matches our naming convention
                pod_name = pod.get('name', '')
                if purpose == "lora_training" and 'kestrel-lora' in pod_name:
                    logger.info(f"Found stopped training pod {pod['id']} - can resume")
                    return pod
                elif purpose == "lora_inference" and 'kestrel-selfie' in pod_name:
                    logger.info(f"Found stopped inference pod {pod['id']} - can resume")
                    return pod
                elif purpose == "ollama_server" and 'kestrel-ollama' in pod_name:
                    logger.info(f"Found stopped Ollama pod {pod['id']} - can resume")
                    return pod

            return None
        except Exception as e:
            logger.warning(f"Failed to list pods for reuse check: {e}")
            return None

    async def resume_stopped_pod(self, pod: Dict[str, Any], profile: GPUProfile, ttl_seconds: int) -> RunPodSession:
        """
        Resume a stopped pod instead of creating new.

        ~10-30s resume time vs 2-5min for new pod creation.
        """
        pod_id = pod['id']
        gpu_count = pod.get('gpuCount', 1)

        logger.info(f"Resuming stopped pod {pod_id} (faster than creating new)")
        await asyncio.to_thread(self.provider.resume_pod, pod_id, gpu_count)

        started_at = datetime.now(timezone.utc)
        if not profile.default_model:
            raise RunPodManagerError(
                f"Profile '{profile.id}' has no default_model configured; cannot resume pod"
            )
        session = RunPodSession(
            pod_id=pod_id,
            profile=profile,
            task_profile=profile.task_type,
            model_name=profile.default_model,
            pod_type=profile.pod_type,
            status=PodStatus.PROVISIONING,
            ttl_seconds=ttl_seconds,
            started_at=started_at,
            expires_at=started_at + timedelta(seconds=ttl_seconds),
        )

        async with self._lock:
            self._session = session

        await self._wait_until_ready()
        return session

    async def get_logs(self, tail: int = 100) -> str:
        """Retrieves the last N lines of logs from the active pod."""
        async with self._lock:
            session = self._session

        if not session or not session.is_active:
            raise RunPodManagerError("No active session to get logs from")

        if isinstance(self.provider, DirectRunPodProvider):
            # Use SSH to tail the logs of the main container
            command = f"docker logs --tail {tail} $(docker ps -q | head -n 1)"
            return await asyncio.to_thread(self.provider.exec_command, session.pod_id, command)

        elif isinstance(self.provider, ManagedRunPodProvider):
            raise NotImplementedError("Log retrieval not yet implemented for managed provider")

        return ""

    async def _wait_until_ready(self) -> None:
        async with self._lock:
            session = self._session
        if not session:
            return
        timeout = session.profile.readiness_timeout_seconds or self.readiness_timeout
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)

        # Phase 1: Wait for pod status to be READY
        while datetime.now(timezone.utc) < deadline:
            status = await self.get_status(refresh=True)
            if status.get("status") == PodStatus.READY.value:
                logger.info("RunPod session %s status is READY", session.pod_id)
                break
            await asyncio.sleep(self.poll_interval)
        else:
            raise RunPodManagerError("RunPod pod did not become ready before timeout")

        # Phase 2: Wait for backend URL to be populated (ports may lag behind status)
        # This is critical - RunPod sometimes reports RUNNING before ports are assigned
        # Increased from 60s to 120s as cold-start pods can take longer for ports
        url_deadline = datetime.now(timezone.utc) + timedelta(seconds=120)  # 120s extra for URL
        while datetime.now(timezone.utc) < url_deadline:
            async with self._lock:
                if session.backend_base_url:
                    logger.info("RunPod session %s backend URL ready: %s", session.pod_id, session.backend_base_url)
                    return
            # Refresh to get updated port info
            await self.get_status(refresh=True)
            await asyncio.sleep(RUNPOD_URL_POLL_INTERVAL)  # Shorter interval for URL polling

        logger.warning("RunPod session %s ready but no backend URL after 120s", session.pod_id)
        # Don't raise error - let caller handle missing URL if needed

    async def wait_for_ready(
        self,
        session: Optional[RunPodSession] = None,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Wait for a RunPod session to be ready.

        This is the public API for waiting after start_session() returns.
        Note: start_session() already calls _wait_until_ready() internally,
        so this is primarily useful when resuming a stopped pod or checking
        readiness after an external event.

        Args:
            session: Session to wait for. Defaults to current session.
            timeout: Timeout in seconds. Defaults to profile's readiness_timeout_seconds.

        Returns:
            True if ready within timeout, False otherwise.
        """
        target_session = session
        if target_session is None:
            async with self._lock:
                target_session = self._session

        if not target_session:
            logger.warning("wait_for_ready called with no session")
            return False

        effective_timeout = timeout or target_session.profile.readiness_timeout_seconds or self.readiness_timeout
        deadline = datetime.now(timezone.utc) + timedelta(seconds=effective_timeout)

        while datetime.now(timezone.utc) < deadline:
            status = await self.get_status(refresh=True)
            if status.get("status") == PodStatus.READY.value:
                logger.info("RunPod session %s is ready", target_session.pod_id)
                return True
            if status.get("status") in {PodStatus.ERROR.value, PodStatus.TERMINATING.value}:
                logger.error("RunPod session %s entered error/terminating state", target_session.pod_id)
                return False
            await asyncio.sleep(self.poll_interval)

        logger.warning("RunPod session %s did not become ready before timeout (%ds)", target_session.pod_id, effective_timeout)
        return False

    def _select_profile(self, task_profile: str) -> GPUProfile:
        profile = self.profiles.get(task_profile)
        if not profile:
            raise RunPodManagerError(f"Unknown task_profile '{task_profile}'. Available: {list(self.profiles.keys())}")
        return profile

    def _validate_ttl(self, ttl_seconds: Optional[int]) -> int:
        ttl = ttl_seconds or self.default_ttl_seconds
        if ttl > self.max_ttl_seconds:
            raise RunPodManagerError(f"TTL {ttl}s exceeds max allowed {self.max_ttl_seconds}s")
        return ttl

    def _update_session_from_runtime(self, session: RunPodSession, pod_info: Dict[str, Any]) -> None:
        if not pod_info:
            logger.warning("pod_info is None for session %s", session.pod_id)
            return
        raw_status = pod_info.get("status") or pod_info.get("desiredStatus")
        session.status = self._map_status(raw_status)
        session.runtime = pod_info

        # Extract port information from runtime
        runtime = pod_info.get("runtime") or {}
        ports = runtime.get("ports", [])

        # Log port info for debugging
        if not ports and raw_status in ("RUNNING", "running"):
            logger.debug("Pod %s is RUNNING but runtime.ports is empty. Full runtime: %s", session.pod_id, runtime)

        for port in ports:
            private_port = port.get("privatePort")
            if private_port and int(private_port) == session.profile.inference_port:
                ip = port.get("ip")
                public_port = port.get("publicPort")
                is_public = port.get("isIpPublic", False)

                if ip and public_port:
                    # Use RunPod proxy URL for private IPs (most common case)
                    # Format: https://{pod_id}-{internal_port}.proxy.runpod.net
                    if not is_public:
                        base_url = f"https://{session.pod_id}-{private_port}.proxy.runpod.net"
                        logger.debug("Using RunPod proxy URL: %s (private IP %s)", base_url, ip)
                    else:
                        # Direct connection for public IPs (rare)
                        base_url = f"{session.profile.inference_protocol}://{ip}:{public_port}"
                        logger.debug("Using direct URL: %s (public IP)", base_url)

                    session.backend_base_url = base_url
                    session.inference_url = f"{base_url}{session.profile.inference_base_path}".rstrip("/")
                    if session.profile.image_invoke_path:
                        session.image_endpoint = f"{base_url}{session.profile.image_invoke_path}"
                    logger.debug("Backend URL set: %s", base_url)

        if session.remaining_ttl_seconds == 0:
            session.status = PodStatus.TERMINATING

    @staticmethod
    def _map_status(raw_status: Optional[str]) -> PodStatus:
        normalized = (raw_status or "").lower()
        if normalized in {"running", "ready"}:
            return PodStatus.READY
        if normalized in {"starting", "provisioning"}:
            return PodStatus.PROVISIONING
        if normalized in {"loading"}:
            return PodStatus.LOADING
        if normalized in {"stopping", "terminating", "stopped"}:
            return PodStatus.TERMINATING
        if normalized in {"failed", "error"}:
            return PodStatus.ERROR
        return PodStatus.OFFLINE

    async def terminate_session(self, session: RunPodSession) -> None:
        """Terminate a specific session's pod."""
        if session and session.pod_id:
            try:
                await asyncio.to_thread(self.provider.stop_pod, session.pod_id)
                logger.info(f"Terminated pod {session.pod_id}")
            except Exception as e:
                logger.error(f"Failed to terminate pod {session.pod_id}: {e}")

    async def terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod by ID."""
        try:
            await asyncio.to_thread(self.provider.stop_pod, pod_id)
            logger.info(f"Terminated pod {pod_id}")
        except Exception as e:
            logger.error(f"Failed to terminate pod {pod_id}: {e}")
