"""
KestrelClaw Bridge Feature.

Exposes the sovereign brain's capabilities via a clean API for external
gateway integration (e.g., KestrelClaw browser extension, Discord bots,
Slack integrations).

Components:
- BridgeFeature: Feature subagent with !bridge commands
- protocol: Pydantic models for bridge request/response protocol
- router: FastAPI router for bridge HTTP endpoints
"""

from .feature import BridgeFeature

__all__ = ["BridgeFeature"]
