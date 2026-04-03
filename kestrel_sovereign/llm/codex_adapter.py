"""
Codex CLI Adapter (Stub)

Adapter for routing to a local Codex CLI session.  Codex runs as a
local process and exposes no HTTP API, so this adapter is a **stub**
that makes the provider visible in the registry and preference system
while raising clear errors if someone tries to use it before the
backend is wired up.

When the Codex integration is complete, this adapter will delegate to
the local CLI/session via subprocess or Unix socket.

Requirements:
- codex CLI installed and on PATH
- Active session (``codex login``)
"""
import logging
from typing import Any, Dict, List, Optional

from .adapter import LLMAdapter, LLMResponse
from .model_metadata import ModelInfo, ModelCategory

logger = logging.getLogger(__name__)


class CodexAdapter(LLMAdapter):
    """Stub adapter for the Codex local CLI backend.

    This adapter is registered so that agents can persist ``provider=codex``
    as their preference.  Actual inference is not yet implemented — all
    call-time methods raise ``NotImplementedError`` with a clear message.
    """

    def create_messages(
        self,
        user_prompt: str,
        system_prompt: str = "",
        images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def get_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        format: Optional[str] = None,
        response_format: Any = None,
        **kwargs,
    ) -> LLMResponse:
        raise NotImplementedError(
            "Codex backend is not yet implemented. "
            "Remove provider=codex from your agent preference, "
            "or wait for the Codex integration to land."
        )

    async def list_models(self) -> List[ModelInfo]:
        """Return the models that will be available once Codex is wired up."""
        return [
            ModelInfo(
                id="codex",
                display_name="Codex (local CLI)",
                provider="codex",
                category=ModelCategory.CHAT,
                supports_tools=False,
                supports_vision=False,
                supports_streaming=False,
                is_featured=False,
            ),
        ]
