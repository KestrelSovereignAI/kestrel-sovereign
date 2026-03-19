"""
RunPod GPU Providers.

Contains provider abstractions for direct RunPod API and managed proxy.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import paramiko
import requests
import runpod
from runpod.cli.utils.rp_info import get_pod_ssh_ip_port
from runpod.cli.utils.rp_userspace import find_ssh_key_file

from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_DEFAULT,
    HTTP_TIMEOUT_QUICK,
)
from .models import RunPodManagerError

logger = logging.getLogger(__name__)


def _sanitize_env_vars(env_vars: Dict[str, Any]) -> Dict[str, str]:
    """Drop unset environment values before sending pod env to RunPod."""
    return {
        key: str(value)
        for key, value in env_vars.items()
        if value is not None
    }


class GPUProvider(ABC):
    """Abstract provider that knows how to manage pods."""

    @abstractmethod
    def start_pod(self, profile, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new GPU pod with the given profile and metadata."""
        ...

    @abstractmethod
    def get_status(self, pod_id: str) -> Dict[str, Any]:
        """Get the current status of a pod."""
        ...

    @abstractmethod
    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        """Stop a running pod."""
        ...


class DirectRunPodProvider(GPUProvider):
    """Provider that talks to RunPod directly using the runpod SDK."""

    def __init__(self, api_key: str, cloud_type: str = "COMMUNITY"):
        if not api_key:
            raise RunPodManagerError("RUNPOD_API_KEY is required for direct mode")
        self.api_key = api_key
        self.cloud_type = cloud_type
        runpod.api_key = api_key

    def start_pod(self, profile, metadata: Dict[str, Any]) -> Dict[str, Any]:
        pod_config = {
            "name": metadata.get("name", f"kestrel-{profile.id}"),
            "image_name": profile.image_name,
            "gpu_type_id": profile.gpu_type_id,
            "cloud_type": metadata.get("cloud_type", self.cloud_type),
            "container_disk_in_gb": profile.container_disk_gb,
            "ports": ",".join(profile.ports),
            "env": _sanitize_env_vars({**profile.env, **metadata.get("env_overrides", {})}),
        }

        # Use network volume if specified (persistent storage, survives pod restart)
        if profile.network_volume_id:
            pod_config["network_volume_id"] = profile.network_volume_id
            # volume_mount_path defaults to /runpod-volume if not specified
            if profile.volume_mount_path:
                pod_config["volume_mount_path"] = profile.volume_mount_path
            logger.info("Using network volume %s mounted at %s",
                        profile.network_volume_id,
                        profile.volume_mount_path or "/runpod-volume")
        else:
            # Fall back to ephemeral volume (lost on pod termination)
            pod_config["volume_in_gb"] = profile.volume_gb

        docker_args = metadata.get("docker_args")
        if docker_args:
            pod_config["docker_args"] = docker_args
        # Use template_id if available (for private registry auth)
        if profile.template_id:
            pod_config["template_id"] = profile.template_id
            logger.info("Using RunPod template %s for registry auth", profile.template_id)
        logger.info("Creating RunPod pod with config: %s", pod_config["name"])
        return runpod.create_pod(**pod_config)

    def get_status(self, pod_id: str) -> Dict[str, Any]:
        result = runpod.get_pod(pod_id)
        # runpod SDK can return None right after pod creation
        if result is None:
            return {"id": pod_id, "status": "PROVISIONING", "desiredStatus": "RUNNING"}
        return result

    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        return runpod.stop_pod(pod_id)

    def resume_pod(self, pod_id: str, gpu_count: int = 1) -> Dict[str, Any]:
        """Resume a stopped pod. Much faster than creating new (~10-30s vs 2-5min)."""
        return runpod.resume_pod(pod_id, gpu_count)

    def terminate_pod(self, pod_id: str) -> Dict[str, Any]:
        """Permanently destroy a pod. Use stop_pod to pause instead."""
        return runpod.terminate_pod(pod_id)

    def list_pods(self) -> List[Dict[str, Any]]:
        """List all pods for this account."""
        return runpod.get_pods()

    def exec_command(self, pod_id: str, command: str) -> str:
        """Executes a command on the pod via SSH and returns the output."""
        try:
            pod = runpod.get_pod(pod_id)
            if not pod:
                raise RunPodManagerError(f"Pod {pod_id} not found")

            ip, port = get_pod_ssh_ip_port(pod)
            if not ip or not port:
                raise RunPodManagerError(f"Could not determine SSH IP/port for pod {pod_id}")

            key_file = find_ssh_key_file()
            if not key_file:
                raise RunPodManagerError("No SSH key file found")

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, port=port, username="root", key_filename=key_file)

            stdin, stdout, stderr = ssh.exec_command(command)
            output = stdout.read().decode("utf-8")
            error = stderr.read().decode("utf-8")
            ssh.close()

            if error:
                logger.warning("SSH command stderr: %s", error)
            return output
        except Exception as e:
            logger.error("Failed to execute command on pod %s: %s", pod_id, e)
            raise RunPodManagerError(f"SSH command failed: {e}") from e


class ManagedRunPodProvider(GPUProvider):
    """Provider proxying through a managed platform API."""

    def __init__(self, api_base: str, api_key: str):
        if not api_base or not api_key:
            raise RunPodManagerError("Managed provider requires KESTREL_API_BASE and KESTREL_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def start_pod(self, profile, metadata: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "profile": profile.id,
            "metadata": metadata,
        }
        response = self.session.post(f"{self.api_base}/runpod/pods", json=payload, timeout=HTTP_TIMEOUT_DEFAULT)
        response.raise_for_status()
        return response.json()

    def get_status(self, pod_id: str) -> Dict[str, Any]:
        response = self.session.get(f"{self.api_base}/runpod/pods/{pod_id}", timeout=HTTP_TIMEOUT_QUICK)
        response.raise_for_status()
        return response.json()

    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        response = self.session.post(f"{self.api_base}/runpod/pods/{pod_id}/stop", timeout=HTTP_TIMEOUT_QUICK)
        response.raise_for_status()
        return response.json()
