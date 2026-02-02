"""
LLM Protocol definitions.

Defines interfaces for LLM requests and responses without importing concrete implementations,
breaking circular dependencies.
"""

from typing import Protocol, Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LLMRequest:
    """Request to an LLM provider."""

    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    finish_reason: str
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    provider_name: str

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM."""
        ...

    async def stream_generate(self, request: LLMRequest):
        """Stream responses from the LLM."""
        ...

    async def list_models(self) -> List[str]:
        """List available models."""
        ...

    async def validate_model(self, model: str) -> bool:
        """Check if a model is available."""
        ...


class LLMService(Protocol):
    """Protocol for the main LLM service orchestrator."""

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a chat completion."""
        ...

    async def get_provider(self, provider_name: str) -> Optional[LLMProvider]:
        """Get a specific LLM provider by name."""
        ...

    async def discover_models(self) -> List[str]:
        """Discover all available models across providers."""
        ...
