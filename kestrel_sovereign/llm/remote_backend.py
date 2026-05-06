"""Remote GPU backend management for LLM Service.

Extracted from service.py to reduce file size. These methods handle:
- Backend switching (cloud/local/remote_gpu)
- Remote GPU activation and deactivation
- Backend status reporting
- Remote session expiry checks
"""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

import openai

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_MEDIUM

# BackendType lives in kestrel-sovereign-sdk so feature packages
# (gcp_compute, vastai, runpod, etc.) can import it without depending
# on the full framework. Re-exported here so existing callers like
# `from kestrel_sovereign.llm.service import BackendType` keep working.
from kestrel_sdk.llm.types import BackendType  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class RemoteGPUConfig:
    """Configuration for remote GPU backend (RunPod, etc.)."""
    base_url: str
    model: str
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    context_window: Optional[int] = None
    ttl_seconds: Optional[int] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = HTTP_TIMEOUT_MEDIUM


class RemoteBackendMixin:
    """Mixin class providing remote GPU backend methods for LLMService.

    Expects the following attributes on the host class:
    - _backend: BackendType
    - _default_backend: BackendType
    - _remote_config: Optional[RemoteGPUConfig]
    - _remote_client: Optional[AsyncOpenAI]
    - _remote_adapter: OpenAIAdapter
    - _last_remote_error: Optional[str]
    """

    def switch_backend(self, backend: BackendType, config: Optional[Dict[str, Any]] = None) -> None:
        """Switch the active backend (cloud/local/remote_gpu)."""
        from .service import LLMServiceError

        if backend == BackendType.REMOTE_GPU:
            if not config:
                raise LLMServiceError("Remote GPU backend requires configuration")
            self._activate_remote_backend(config)
            return

        # Switching to cloud/local clears any remote session
        self._deactivate_remote_backend()
        logger.info(f"LLMService switched to {backend.value} backend")
        self._backend = backend

    def _activate_remote_backend(self, config: Dict[str, Any]) -> None:
        """Activate a remote GPU backend."""
        from .service import LLMServiceError

        base_url = config.get("base_url") or config.get("inference_url")
        if not base_url:
            raise LLMServiceError("Remote backend requires base_url")
        model = config.get("model") or config.get("model_name")
        if not model:
            raise LLMServiceError("Remote backend requires a model name")

        expires_at = config.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        ttl_seconds = config.get("ttl_seconds")
        if expires_at is None and ttl_seconds:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(ttl_seconds))

        self._remote_config = RemoteGPUConfig(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=config.get("api_key"),
            headers=config.get("headers") or {},
            context_window=config.get("context_window"),
            ttl_seconds=ttl_seconds,
            expires_at=expires_at,
            metadata=config,
            timeout_seconds=int(config.get("timeout_seconds", HTTP_TIMEOUT_MEDIUM)),
        )
        self._remote_client = openai.AsyncOpenAI(
            base_url=self._remote_config.base_url,
            api_key=self._remote_config.api_key or os.environ.get("RUNPOD_API_KEY", "sk-kestrel-gpu"),
            default_headers=self._remote_config.headers or None,
            timeout=self._remote_config.timeout_seconds,
        )
        self._backend = BackendType.REMOTE_GPU
        logger.info(f"Remote GPU backend activated at {base_url}")

    def _deactivate_remote_backend(self, reason: Optional[str] = None) -> None:
        """Deactivate remote GPU backend."""
        if self._remote_client is None and self._backend != BackendType.REMOTE_GPU:
            return
        if reason:
            logger.info(f"Deactivating remote backend: {reason}")
        self._remote_client = None
        self._remote_config = None
        self._backend = self._default_backend

    def get_backend_status(self) -> Dict[str, Any]:
        """Return current backend status for telemetry/UIs."""
        return {
            "current_backend": self._backend.value,
            "default_backend": self._default_backend.value,
            "remote_active": self._remote_config is not None,
            "remote_metadata": self._remote_config.metadata if self._remote_config else None,
            "last_remote_error": self._last_remote_error,
        }

    def _ensure_remote_active(self) -> None:
        """Verify remote GPU backend is active and not expired."""
        from .service import LLMServiceError

        if not self._remote_config or not self._remote_client:
            raise LLMServiceError("Remote backend is not active")
        if self._remote_config.expires_at and datetime.now(timezone.utc) >= self._remote_config.expires_at:
            raise LLMServiceError("Remote backend session expired")
