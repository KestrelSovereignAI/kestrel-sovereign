"""
Agent Card Types for A2A Protocol.

Re-exports from kestrel_sdk.a2a.agent_card for backward compatibility.
Feature packages should import from kestrel_sdk.a2a.agent_card directly.
"""

# Re-export everything from kestrel_sdk
from kestrel_sdk.a2a.agent_card import (  # noqa: F401
    AgentProvider,
    AgentCapabilities,
    AgentAuthentication,
    AgentSkill,
    AgentCard,
)
