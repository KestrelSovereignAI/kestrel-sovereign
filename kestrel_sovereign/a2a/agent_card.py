"""
Agent Card Types for A2A Protocol.

The Agent Card is a standardized way to describe an agent's capabilities,
allowing discovery and interoperability between agents.

Based on https://a2a-protocol.org/latest/specification/#agent-card
"""

from pydantic import BaseModel


class AgentProvider(BaseModel):
    """Information about the agent's provider/creator."""
    organization: str
    url: str | None = None


class AgentCapabilities(BaseModel):
    """Capabilities supported by the agent."""
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentAuthentication(BaseModel):
    """Authentication requirements for the agent."""
    schemes: list[str]
    credentials: str | None = None


class AgentSkill(BaseModel):
    """A skill/capability offered by the agent."""
    id: str
    name: str
    description: str | None = None
    tags: list[str] | None = None
    examples: list[str] | None = None
    inputModes: list[str] | None = None
    outputModes: list[str] | None = None


class AgentCard(BaseModel):
    """
    Agent Card - describes an agent's identity and capabilities.

    Served at /.well-known/agent-card.json for agent discovery.

    Example:
        AgentCard(
            name="Kestrel Agent",
            description="AI companion platform",
            url="https://YOUR_DOMAIN.com",
            version="1.0.0",
            capabilities=AgentCapabilities(
                streaming=True,
                pushNotifications=False,
                stateTransitionHistory=True
            ),
            skills=[
                AgentSkill(
                    id="chat",
                    name="Companion Chat",
                    description="Conversational AI companion",
                    inputModes=["text"],
                    outputModes=["text"]
                )
            ]
        )
    """
    name: str
    description: str | None = None
    url: str  # Base URL for the agent's A2A service
    provider: AgentProvider | None = None
    version: str
    documentationUrl: str | None = None
    capabilities: AgentCapabilities
    authentication: AgentAuthentication | None = None
    defaultInputModes: list[str] = ['text']
    defaultOutputModes: list[str] = ['text']
    skills: list[AgentSkill]
