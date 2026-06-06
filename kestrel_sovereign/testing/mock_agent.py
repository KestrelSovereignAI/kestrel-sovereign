"""
MockAgent — lightweight KestrelAgent stub for feature testing.

Provides stubbed versions of all services that features typically access:
- llm_service: returns configurable responses
- storage: in-memory SQLite (:memory:)
- hooks_manager: records hook registrations
- features: empty dict (populated by test)
- privacy_agent: stubbed with NORMAL mode
"""

import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.privacy import PrivacyMode

logger = logging.getLogger(__name__)


class MockLLMService:
    """Stubbed LLM service that returns configurable responses."""

    def __init__(self, default_response: str = "Mock LLM response"):
        self.default_response = default_response
        self._responses: List[str] = []
        self._call_history: List[Dict[str, Any]] = []
        # Mirrors LLMService's per-agent claim contract introduced in
        # the PayerPolicy work. Real LLMService raises on duplicate
        # attach; the mock just records (or no-ops) so tests that wrap
        # this fake in a KestrelAgent work without modification.
        self._owner_agent_did: Any = None

    def attach_to_agent(self, agent_did: str) -> None:
        """No-op claim. Tests that need to assert on this can read
        ``_owner_agent_did`` directly."""
        if not agent_did:
            raise ValueError("agent_did is required for attach_to_agent")
        self._owner_agent_did = agent_did

    def queue_response(self, response: str) -> None:
        """Queue a response to be returned by the next get_response call."""
        self._responses.append(response)

    def queue_responses(self, responses: List[str]) -> None:
        """Queue multiple responses."""
        self._responses.extend(responses)

    async def get_response(self, messages: List[Dict], **kwargs) -> Any:
        """Return queued response or default."""
        self._call_history.append({"messages": messages, **kwargs})
        text = self._responses.pop(0) if self._responses else self.default_response

        # Return a minimal LLMResponse-like object
        return MagicMock(
            text=text,
            content=text,
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

    async def get_streaming_response(self, messages: List[Dict], **kwargs):
        """Return a mock streaming response."""
        self._call_history.append({"messages": messages, "streaming": True, **kwargs})
        text = self._responses.pop(0) if self._responses else self.default_response

        async def _stream():
            yield MagicMock(text=text, content=text, delta=text)

        return _stream()

    def get_model_preference(self) -> str:
        return "mock-model"

    def set_preference_persistence_callback(self, callback) -> None:
        pass

    async def close(self) -> None:
        pass


class MockHooksManager:
    """Stubbed hooks manager that records registrations."""

    def __init__(self):
        self.registered_hooks: List[Any] = []
        self.executed_hooks: List[Dict[str, Any]] = []

    def register(self, hook) -> None:
        """Record hook registration."""
        self.registered_hooks.append(hook)

    def unregister(self, hook) -> None:
        """Record hook unregistration."""
        if hook in self.registered_hooks:
            self.registered_hooks.remove(hook)

    async def execute_hooks(self, event, input_data=None, **kwargs):
        """Record hook execution and return ALLOW."""
        self.executed_hooks.append({"event": event, "input": input_data, **kwargs})
        # Return a mock ALLOW result
        return MagicMock(decision="ALLOW", blocked=False)


class MockPrivacyAgent:
    """Stubbed privacy agent with configurable mode."""

    def __init__(self, mode: PrivacyMode = PrivacyMode.NORMAL):
        self._mode = mode

    @property
    def privacy_mode(self) -> PrivacyMode:
        return self._mode

    @privacy_mode.setter
    def privacy_mode(self, value: PrivacyMode) -> None:
        self._mode = value

    def get_privacy_mode(self) -> PrivacyMode:
        return self._mode

    async def set_privacy_mode(self, mode) -> None:
        self._mode = mode


class MockAgent:
    """
    Lightweight KestrelAgent substitute for feature testing.

    Provides stubbed versions of all services that features access via
    self.agent, without requiring real databases, LLM providers, or
    full agent initialization.

    Usage:
        agent = MockAgent()
        feature = MyFeature(agent)
        await feature.initialize()
        result = await feature.my_tool(param="test")

    With custom LLM responses:
        agent = MockAgent(llm_responses=["Hello!", "Goodbye!"])
        feature = MyFeature(agent)
        # First LLM call returns "Hello!", second returns "Goodbye!"

    With in-memory storage:
        agent = await MockAgent.create()  # initializes async storage
        feature = MyFeature(agent)
        await feature.initialize()
        # feature can use agent.storage for real SQLite operations
    """

    def __init__(
        self,
        did: str = "did:test:mock-agent",
        llm_responses: Optional[List[str]] = None,
        default_llm_response: str = "Mock LLM response",
        privacy_mode: PrivacyMode = PrivacyMode.NORMAL,
        storage: Optional[Any] = None,
    ):
        self.did = did
        self.agent_id = did

        # Stubbed services
        self.llm_service = MockLLMService(default_response=default_llm_response)
        if llm_responses:
            self.llm_service.queue_responses(llm_responses)

        self.hooks_manager = MockHooksManager()
        self.privacy_agent = MockPrivacyAgent(mode=privacy_mode)
        self.features: Dict[str, Any] = {}
        self.storage = storage

        # Attributes that features may access
        self.wallet = None
        self.task_manager = None
        self._privacy_mode = privacy_mode
        self._safe_mode = False
        self._explored_features: dict = {}
        self._direct_tools: dict = {}
        self._direct_tool_defs: list = []
        self._tool_to_feature: dict = {}
        self._event_listeners: list = []
        self._pending_task_notifications: list = []
        self._current_request_id: Optional[str] = None
        self._active_request_ids: set = set()
        self._active_request_started_at: dict = {}
        self._cancelled_requests: set = set()
        self._session_briefed = False
        self._cached_features_prompt = ""

    @classmethod
    async def create(
        cls,
        did: str = "did:test:mock-agent",
        llm_responses: Optional[List[str]] = None,
        default_llm_response: str = "Mock LLM response",
        privacy_mode: PrivacyMode = PrivacyMode.NORMAL,
    ) -> "MockAgent":
        """
        Create a MockAgent with initialized in-memory storage.

        Use this when your feature needs real storage operations (SQLite :memory:).
        """
        from kestrel_sovereign.storage.async_storage import AsyncStorage
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

        backend = SQLiteBackend(":memory:")
        storage = AsyncStorage(backend=backend)
        await storage.initialize()

        return cls(
            did=did,
            llm_responses=llm_responses,
            default_llm_response=default_llm_response,
            privacy_mode=privacy_mode,
            storage=storage,
        )

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.storage and hasattr(self.storage, "close"):
            await self.storage.close()
        await self.llm_service.close()

    # Convenience methods that features may call on the agent

    def get_model_preference(self) -> str:
        return self.llm_service.get_model_preference()

    async def emit_event(self, event_type: str, data: Any = None) -> None:
        """No-op event emission for testing."""
        pass
