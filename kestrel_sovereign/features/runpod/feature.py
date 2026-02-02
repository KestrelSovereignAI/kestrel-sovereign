import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.runpod.manager import RunPodManager
from kestrel_sovereign.features.runpod.models import RunPodManagerError
from kestrel_sovereign.llm.service import BackendType
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class RunPodFeature(Feature):
    """Feature layer that exposes GPU orchestration via the tool system."""

    @property
    def tool_description(self) -> str:
        return (
            "Manage RunPod GPU sessions - start and stop on-demand GPU pods for "
            "heavy inference, check status, view logs, and route LLM traffic to GPU"
        )

    async def initialize(self):
        try:
            self.manager = RunPodManager()
        except RunPodManagerError as e:
            logger.warning(f"RunPodFeature disabled: {e}")
            self.manager = None
            self.disabled = True
            self.disabled_reason = str(e)
            return
        self.disabled = False
        self.llm_service = getattr(self.agent, "llm_service", None)
        if not self.llm_service:
            logger.warning("LLMService not available; GPU routing disabled")

    @tool(
        name="manage_gpu",
        description="Start, stop, or inspect RunPod GPU sessions (usage: !gpu <action> [...]).",
        category=ToolCategory.SYSTEM,
        command_prefix="!gpu",
    )
    async def manage_gpu(
        self,
        action: str = "status",
        model_name: str = "",
        task_profile: str = "llm",
        ttl_seconds: str = "",
        pod_type: str = "",
        lines: str = "100",
    ) -> Dict[str, Any]:
        if getattr(self, 'disabled', False):
            return {
                "action": action,
                "error": "RunPod feature is disabled",
                "reason": getattr(self, 'disabled_reason', 'RUNPOD_API_KEY not set'),
            }
        action_normalized = (action or "status").lower()

        if action_normalized in {"status"}:
            return await self._status()
        if action_normalized in {"logs", "log"}:
            num_lines = self._coerce_optional_int(lines) or 100
            return await self._logs(lines=num_lines)
        if action_normalized in {"off", "stop"}:
            return await self._stop()
        if action_normalized in {"on", "start"}:
            return await self._start(
                model_name=model_name,
                task_profile=task_profile,
                ttl_seconds=ttl_seconds,
                pod_type=pod_type,
            )
        raise ValueError("Unsupported GPU action. Use on, off, or status.")

    # NOTE: !dream command removed
    # generate_image_on_runpod() exists below but NOTHING CALLS IT
    # visual_identity feature exists but is also NOT USED
    

    async def generate_image_on_runpod(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        ttl_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Internal method for generating images on RunPod GPU.

        STATUS: NOT USED - Nothing calls this method.
        

        This method exists for future RunPod integration but is dead code.
        Automatically starts pod, generates image, and stops pod.

        Args:
            prompt: Image generation prompt
            model_name: Optional model to use
            ttl_seconds: Optional TTL (default: min of default_ttl or 900)

        Returns:
            Dict with image result and session info
        """
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt is required for image generation")

        ttl = ttl_seconds or min(self.manager.default_ttl_seconds, 900)

        image_status = await self.manager.start_session(
            task_profile="image",
            model_name=model_name,
            ttl_seconds=ttl,
        )

        endpoint = image_status.get("image_endpoint") or image_status.get("inference_url")
        if not endpoint:
            await self.manager.stop_session()
            raise RunPodManagerError("Image endpoint not provided by pod")

        payload = {"prompt": prompt, "model": model_name or image_status.get("model_name")}
        image_result = await asyncio.to_thread(self._post_json, endpoint, payload)

        # CRITICAL: Always stop the pod after use to avoid cost runaway
        teardown = await self.manager.stop_session()
        self._detach_gpu_backend("image generation completed")

        logger.info(f"✅ RunPod image generation complete, pod stopped")

        return {
            "action": "generate_image",
            "prompt": prompt,
            "result": image_result,
            "session": image_status,
            "teardown": teardown,
        }

    async def _start(
        self,
        *,
        model_name: str,
        task_profile: str,
        ttl_seconds: str,
        pod_type: str,
    ) -> Dict[str, Any]:
        profile_key = (task_profile or "llm").lower()
        if profile_key not in self.manager.profiles:
            raise RunPodManagerError(
                f"Unknown task_profile '{task_profile}'. Available: {', '.join(self.manager.profiles.keys())}"
            )

        ttl = self._coerce_optional_int(ttl_seconds)
        target_model = model_name or None
        metadata = {
            "name": f"kestrel-{profile_key}-{datetime.now(timezone.utc).strftime('%H%M%S')}",
            "env_overrides": {
                "TARGET_MODEL": target_model,
                "KESTREL_PROFILE": profile_key,
            },
            "pod_type": pod_type or None,
        }

        status = await self.manager.start_session(
            task_profile=profile_key,
            model_name=target_model,
            ttl_seconds=ttl,
            pod_type=pod_type or None,
            metadata=metadata,
        )

        if profile_key == "llm":
            self._attach_gpu_backend(status)

        return {
            "action": "start",
            "session": status,
            "router": self._router_status(),
        }

    async def _stop(self) -> Dict[str, Any]:
        status = await self.manager.stop_session()
        self._detach_gpu_backend("Requested via !gpu off")
        return {
            "action": "stop",
            "session": status,
            "router": self._router_status(),
        }

    async def _status(self) -> Dict[str, Any]:
        status = await self.manager.get_status()
        return {
            "action": "status",
            "session": status,
            "router": self._router_status(),
        }

    async def _logs(self, lines: int) -> Dict[str, Any]:
        logs = await self.manager.get_logs(lines=lines)
        return {
            "action": "logs",
            "lines": lines,
            "logs": logs,
            "router": self._router_status(),
        }

    def _attach_gpu_backend(self, session_status: Dict[str, Any]) -> None:
        if not self.llm_service:
            logger.warning("Cannot attach GPU backend without LLMService")
            return

        base_url = session_status.get("inference_url")
        if not base_url:
            logger.warning("RunPod session missing inference URL; skipping activation")
            return

        remaining = session_status.get("remaining_ttl_seconds")
        profile_id = session_status.get("task_profile")
        profile = self.manager.profiles.get(profile_id)
        context_window = profile.max_context_window if profile else None

        config = {
            "base_url": base_url,
            "model": session_status.get("model_name"),
            "ttl_seconds": remaining,
            "context_window": context_window,
            "metadata": {
                "pod_id": session_status.get("pod_id"),
                "profile": profile_id,
            },
        }
        self.llm_service.switch_backend(BackendType.REMOTE_GPU, config=config)

    def _detach_gpu_backend(self, reason: str) -> None:
        if self.llm_service:
            self.llm_service._deactivate_remote_backend(reason=reason)

    def _router_status(self) -> Optional[Dict[str, Any]]:
        if not self.llm_service:
            return None
        return self.llm_service.get_backend_status()

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if not text:
            return None
        if not text.isdigit():
            raise ValueError("TTL must be an integer number of seconds")
        return int(text)

    @staticmethod
    def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
