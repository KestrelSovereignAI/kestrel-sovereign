"""
OpenRouter API Key Provisioning Service.

Creates per-agent API keys so each Kestrel agent has isolated billing,
usage tracking, and the ability to self-fund via crypto payments.

Environment Variables:
    OPENROUTER_MANAGEMENT_API_KEY: Required for key creation/management
    OPENROUTER_API_KEY: Default key for agents without their own key

Usage:
    service = OpenRouterProvisioningService()

    # Create key for new agent
    key_info = await service.create_agent_key("emma-001", limit_usd=10.0)

    # Check agent's usage
    usage = await service.get_key_usage(key_info.key_hash)

    # Delete key when agent retires
    await service.delete_key(key_info.key_hash)
"""

import asyncio
import os
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Literal

import httpx

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DEFAULT
from kestrel_sovereign.kestrel_config.defaults import get_openrouter_api_base

logger = logging.getLogger(__name__)


@dataclass
class AgentKeyInfo:
    """Information about an agent's OpenRouter API key."""

    key: str  # The actual API key (only available at creation time)
    key_hash: str  # Hash for subsequent API calls
    name: str  # Agent name/identifier
    limit_cents: int  # Spending limit in cents
    limit_reset: Optional[str]  # daily, weekly, monthly, or None

    def to_dict(self) -> dict:
        """Serialize for storage in agent metadata."""
        return {
            "key_hash": self.key_hash,
            "name": self.name,
            "limit_cents": self.limit_cents,
            "limit_reset": self.limit_reset,
        }

    @classmethod
    def from_dict(cls, data: dict, key: str = "") -> "AgentKeyInfo":
        """Deserialize from agent metadata."""
        return cls(
            key=key,
            key_hash=data["key_hash"],
            name=data["name"],
            limit_cents=data["limit_cents"],
            limit_reset=data.get("limit_reset"),
        )


@dataclass
class KeyUsage:
    """Usage information for an agent's API key."""

    key_hash: str
    limit_cents: int
    limit_remaining_cents: int
    usage_cents: int
    usage_monthly_cents: int
    is_free_tier: bool
    rate_limit_requests: Optional[int]
    rate_limit_interval: Optional[str]

    @property
    def limit_usd(self) -> Decimal:
        return Decimal(self.limit_cents) / 100

    @property
    def remaining_usd(self) -> Decimal:
        return Decimal(self.limit_remaining_cents) / 100

    @property
    def usage_usd(self) -> Decimal:
        return Decimal(self.usage_cents) / 100

    @property
    def usage_monthly_usd(self) -> Decimal:
        return Decimal(self.usage_monthly_cents) / 100


class OpenRouterProvisioningError(Exception):
    """Error during OpenRouter key provisioning."""
    pass


class OpenRouterProvisioningService:
    """
    Service for provisioning per-agent OpenRouter API keys.

    Each Kestrel agent gets its own API key with:
    - Isolated billing and usage tracking
    - Configurable spending limits with auto-reset
    - Clean audit trail for compliance
    - Ability to self-fund via crypto (USDC on Base/Polygon/ETH)
    """

    def __init__(self, management_key: Optional[str] = None):
        """
        Initialize the provisioning service.

        Args:
            management_key: OpenRouter management API key. If not provided,
                           reads from OPENROUTER_MANAGEMENT_API_KEY env var.
        """
        self.management_key = management_key or os.getenv("OPENROUTER_MANAGEMENT_API_KEY")
        if not self.management_key:
            raise OpenRouterProvisioningError(
                "OPENROUTER_MANAGEMENT_API_KEY not set. "
                "Get one at https://openrouter.ai/settings/keys"
            )

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=get_openrouter_api_base(),
                headers={
                    "Authorization": f"Bearer {self.management_key}",
                    "Content-Type": "application/json",
                },
                timeout=HTTP_TIMEOUT_DEFAULT,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def create_agent_key(
        self,
        agent_name: str,
        limit_usd: float = 0.10,
        limit_reset: Optional[Literal["daily", "weekly", "monthly"]] = "monthly",
    ) -> AgentKeyInfo:
        """
        Create a new API key for an agent.

        Args:
            agent_name: Unique identifier for the agent (e.g., "emma-001")
            limit_usd: Spending limit in USD (default $1.00)
            limit_reset: When to reset the limit (daily/weekly/monthly/None)

        Returns:
            AgentKeyInfo with the new key details

        Raises:
            OpenRouterProvisioningError: If key creation fails
        """
        client = await self._get_client()

        limit_cents = int(limit_usd * 100)

        payload = {
            "name": f"kestrel-agent-{agent_name}",
            "limit": limit_cents,
        }
        if limit_reset:
            payload["limit_reset"] = limit_reset

        # Retry with exponential backoff for rate limits
        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries + 1):
            try:
                response = await client.post("/keys", json=payload)
                response.raise_for_status()
                data = response.json()

                key_info = AgentKeyInfo(
                    key=data["key"],
                    key_hash=data["data"]["hash"],
                    name=agent_name,
                    limit_cents=limit_cents,
                    limit_reset=limit_reset,
                )

                logger.info(
                    f"Created OpenRouter key for agent '{agent_name}' "
                    f"(limit: ${limit_usd:.2f}, reset: {limit_reset})"
                )

                return key_info

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limited creating key for '{agent_name}', "
                        f"retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                error_msg = f"Failed to create key for agent '{agent_name}': {e.response.text}"
                logger.error(error_msg)
                raise OpenRouterProvisioningError(error_msg) from e
            except Exception as e:
                error_msg = f"Unexpected error creating key for agent '{agent_name}': {e}"
                logger.error(error_msg)
                raise OpenRouterProvisioningError(error_msg) from e

        # Should not reach here, but just in case
        raise OpenRouterProvisioningError(f"Max retries exceeded for agent '{agent_name}'")

    async def get_key_usage(self, key_hash: str) -> KeyUsage:
        """
        Get usage information for an agent's API key.

        Args:
            key_hash: The key hash from AgentKeyInfo

        Returns:
            KeyUsage with current usage and limits
        """
        client = await self._get_client()

        try:
            response = await client.get(f"/keys/{key_hash}")
            response.raise_for_status()
            data = response.json()["data"]

            return KeyUsage(
                key_hash=key_hash,
                limit_cents=data.get("limit", 0),
                limit_remaining_cents=data.get("limit_remaining", 0),
                usage_cents=data.get("usage", 0),
                usage_monthly_cents=data.get("usage_monthly", 0),
                is_free_tier=data.get("is_free_tier", False),
                rate_limit_requests=data.get("rate_limit", {}).get("requests"),
                rate_limit_interval=data.get("rate_limit", {}).get("interval"),
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"Failed to get usage for key '{key_hash}': {e.response.text}"
            logger.error(error_msg)
            raise OpenRouterProvisioningError(error_msg) from e

    async def update_key_limit(
        self,
        key_hash: str,
        limit_usd: float,
        limit_reset: Optional[Literal["daily", "weekly", "monthly"]] = None,
    ) -> KeyUsage:
        """
        Update the spending limit for an agent's key.

        Args:
            key_hash: The key hash from AgentKeyInfo
            limit_usd: New spending limit in USD
            limit_reset: Optional new reset interval

        Returns:
            Updated KeyUsage
        """
        client = await self._get_client()

        payload = {"limit": int(limit_usd * 100)}
        if limit_reset is not None:
            payload["limit_reset"] = limit_reset

        try:
            response = await client.patch(f"/keys/{key_hash}", json=payload)
            response.raise_for_status()

            logger.info(f"Updated key '{key_hash}' limit to ${limit_usd:.2f}")

            return await self.get_key_usage(key_hash)

        except httpx.HTTPStatusError as e:
            error_msg = f"Failed to update key '{key_hash}': {e.response.text}"
            logger.error(error_msg)
            raise OpenRouterProvisioningError(error_msg) from e

    async def delete_key(self, key_hash: str) -> bool:
        """
        Delete an agent's API key (e.g., on retirement).

        Args:
            key_hash: The key hash from AgentKeyInfo

        Returns:
            True if deleted successfully
        """
        client = await self._get_client()

        try:
            response = await client.delete(f"/keys/{key_hash}")
            response.raise_for_status()

            logger.info(f"Deleted OpenRouter key '{key_hash}'")
            return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Key '{key_hash}' not found (already deleted?)")
                return True
            error_msg = f"Failed to delete key '{key_hash}': {e.response.text}"
            logger.error(error_msg)
            raise OpenRouterProvisioningError(error_msg) from e

    async def list_keys(self) -> list[dict]:
        """
        List all API keys under this management key.

        Returns:
            List of key information dictionaries
        """
        client = await self._get_client()

        try:
            response = await client.get("/keys")
            response.raise_for_status()
            return response.json().get("data", [])

        except httpx.HTTPStatusError as e:
            error_msg = f"Failed to list keys: {e.response.text}"
            logger.error(error_msg)
            raise OpenRouterProvisioningError(error_msg) from e


async def provision_agent_key(
    agent_name: str,
    limit_usd: float = 0.10,
    limit_reset: str = "monthly",
) -> AgentKeyInfo:
    """
    Convenience function to create an agent key.

    This is the main entry point for inception_service.py to use.

    Args:
        agent_name: Unique agent identifier
        limit_usd: Initial spending limit
        limit_reset: Reset interval (daily/weekly/monthly)

    Returns:
        AgentKeyInfo with the new key
    """
    service = OpenRouterProvisioningService()
    try:
        return await service.create_agent_key(
            agent_name=agent_name,
            limit_usd=limit_usd,
            limit_reset=limit_reset,
        )
    finally:
        await service.close()


async def get_agent_usage(key_hash: str) -> KeyUsage:
    """
    Convenience function to get agent usage.

    Args:
        key_hash: The agent's key hash

    Returns:
        KeyUsage with current stats
    """
    service = OpenRouterProvisioningService()
    try:
        return await service.get_key_usage(key_hash)
    finally:
        await service.close()


async def delete_agent_key(key_hash: str) -> bool:
    """
    Convenience function to delete an agent's key.

    Called by retirement_service.py when an agent is retired.

    Args:
        key_hash: The agent's key hash

    Returns:
        True if deleted
    """
    service = OpenRouterProvisioningService()
    try:
        return await service.delete_key(key_hash)
    finally:
        await service.close()
