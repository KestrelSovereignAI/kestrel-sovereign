"""
Key Resolution Service for Kestrel.

Resolves API keys for external services with fail-fast pattern.
Uses agent-scoped key storage - each agent has isolated keys.
"""

import logging
import os
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

logger = logging.getLogger(__name__)


class KeyNotConfiguredError(Exception):
    """Raised when required key is not available."""

    def __init__(self, provider: str, message: Optional[str] = None):
        self.provider = provider
        super().__init__(message or f"No API key configured for {provider}")


class KeyResolutionService:
    """
    Resolves API keys with fail-fast pattern.

    Resolution order:
    1. Agent's key storage (encrypted, agent-scoped)
    2. Environment variable fallback (for standalone mode)

    NEVER falls back to platform keys for user operations.
    All keys are agent-scoped - each agent has isolated key storage.
    """

    def __init__(
        self,
        storage: Optional["ServiceKeyStorage"] = None,
        agent_did: Optional[str] = None,
    ):
        """
        Initialize key resolution service.

        Args:
            storage: ServiceKeyStorage instance (already bound to an agent)
            agent_did: Agent DID (used only if creating storage on-demand)
        """
        self._storage = storage
        self._agent_did = agent_did

    @classmethod
    def from_agent(cls, agent: Any) -> "KeyResolutionService":
        """
        Create resolver from a KestrelAgent instance.

        Args:
            agent: KestrelAgent with storage and identity

        Returns:
            Configured KeyResolutionService
        """
        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
        from kestrel_sovereign.security.agent_encryption import MasterKeyNotConfiguredError

        # Get agent DID - REQUIRED for key storage
        agent_did = getattr(agent, "did", None) or getattr(agent, "agent_id", None)

        if not agent_did:
            logger.warning("KeyResolutionService: No agent DID available")
            return cls(storage=None, agent_did=None)

        # Get database from agent storage
        db = None
        if hasattr(agent, "storage") and agent.storage:
            if hasattr(agent.storage, "db"):
                db = agent.storage.db
            elif hasattr(agent.storage, "database"):
                db = agent.storage.database
            elif hasattr(agent.storage, "_db"):
                db = agent.storage._db

        # Initialize storage
        storage = None
        if db:
            try:
                storage = ServiceKeyStorage(db, agent_did)
            except MasterKeyNotConfiguredError as e:
                logger.warning(f"Key storage not available: {e}")
            except Exception as e:
                logger.warning(f"Could not initialize ServiceKeyStorage: {e}")

        return cls(storage=storage, agent_did=agent_did)

    async def resolve_key(
        self,
        provider: str,
        require: bool = True,
    ) -> Optional[str]:
        """
        Resolve API key for a provider.

        Resolution order:
        1. Agent's encrypted key storage
        2. Environment variable fallback

        Args:
            provider: Service provider ID (openrouter, lighthouse, openai, anthropic, github, runpod, vastai)
            require: If True, raise KeyNotConfiguredError when key not found

        Returns:
            Decrypted API key or None if not found and require=False

        Raises:
            KeyNotConfiguredError: If require=True and no key available
        """
        provider = provider.lower()
        key = None

        # Try storage first (if available)
        if self._storage:
            try:
                key = await self._storage.get_key(provider_id=provider)
                if key:
                    logger.debug(f"Resolved key for {provider} from agent storage")
                    return key
            except Exception as e:
                logger.debug(f"Storage key lookup for {provider}: {e}")

        # Fall back to environment variables
        env_var_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "lighthouse": "LIGHTHOUSE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "github": ["GITHUB_PAT", "GITHUB_TOKEN"],
            "runpod": "RUNPOD_API_KEY",
            "vastai": "VASTAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY",
            "fireworks": "FIREWORKS_API_KEY",
            "xai": "XAI_API_KEY",
        }

        env_vars = env_var_map.get(provider)
        if env_vars:
            if isinstance(env_vars, list):
                for var in env_vars:
                    key = os.environ.get(var)
                    if key:
                        logger.debug(f"Resolved key for {provider} from ${var}")
                        return key
            else:
                key = os.environ.get(env_vars)
                if key:
                    logger.debug(f"Resolved key for {provider} from ${env_vars}")
                    return key

        # No key found
        if require:
            raise KeyNotConfiguredError(
                provider,
                f"No API key configured for {provider}. "
                f"Add via !add-key or set environment variable.",
            )

        return None

    async def has_key(self, provider: str) -> bool:
        """
        Check if a key is configured for a provider.

        Args:
            provider: Service provider

        Returns:
            True if key exists (in storage or environment)
        """
        provider = provider.lower()

        # Check storage
        if self._storage:
            try:
                if await self._storage.has_key(provider_id=provider):
                    return True
            except Exception:
                pass

        # Check environment
        key = await self.resolve_key(provider, require=False)
        return key is not None

    async def check_quota(
        self,
        provider: str,
        units: int = 1,
        max_retries: int = 3,
    ) -> bool:
        """
        Check if operation is allowed within quota.

        Uses retry-then-deny pattern: retries on transient errors, denies after max retries.
        This is a fail-closed security pattern to prevent abuse during outages.

        Args:
            provider: Service provider
            units: Units to consume
            max_retries: Maximum retry attempts before denying

        Returns:
            True if allowed, False if quota exceeded or check failed after retries
        """
        import asyncio

        if not self._storage:
            return True  # No storage = no quota enforcement

        provider = provider.lower()

        for attempt in range(max_retries):
            try:
                return await self._storage.check_quota(
                    provider_id=provider,
                    units=units,
                )
            except Exception as e:
                logger.warning(f"Quota check attempt {attempt + 1}/{max_retries} failed for {provider}: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.5s, 1s, 2s...
                    await asyncio.sleep(0.5 * (2 ** attempt))

        # Fail-closed: deny after all retries exhausted
        logger.error(f"Quota check failed after {max_retries} attempts for {provider}, denying request")
        return False

    async def record_usage(
        self,
        provider: str,
        operation: str,
        units: int = 1,
        cost_estimate: Optional[float] = None,
    ) -> None:
        """
        Record usage for billing/quota tracking.

        Args:
            provider: Service provider
            operation: Operation type (e.g., 'upload', 'inference', 'search')
            units: Units consumed
            cost_estimate: Estimated cost in USD
        """
        if not self._storage:
            return

        provider = provider.lower()

        try:
            await self._storage.record_usage(
                provider_id=provider,
                operation=operation,
                units=units,
                cost_estimate=cost_estimate,
            )
        except Exception as e:
            logger.warning(f"Failed to record usage for {provider}: {e}")

    async def get_key_info(self, provider: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a configured key (no secret exposed).

        Args:
            provider: Service provider

        Returns:
            Key metadata or None if not configured
        """
        provider = provider.lower()

        if self._storage:
            try:
                keys = await self._storage.list_keys()
                for key in keys:
                    if key.provider_id == provider:
                        from kestrel_sovereign.security.service_key_storage import KNOWN_PROVIDERS

                        provider_info = KNOWN_PROVIDERS.get(provider, {"name": provider})
                        return {
                            "provider": provider,
                            "provider_name": provider_info["name"],
                            "quota_limit": key.quota_limit,
                            "quota_used": key.quota_used,
                            "is_active": key.is_active,
                            "source": "storage",
                        }
            except Exception as e:
                logger.warning(f"Failed to get key info for {provider}: {e}")

        # Check environment variable
        key = await self.resolve_key(provider, require=False)
        if key:
            from kestrel_sovereign.security.service_key_storage import KNOWN_PROVIDERS

            provider_info = KNOWN_PROVIDERS.get(provider, {"name": provider})
            return {
                "provider": provider,
                "provider_name": provider_info["name"],
                "quota_limit": None,
                "quota_used": None,
                "is_active": True,
                "source": "environment",
            }

        return None


# Convenience function for quick key resolution
async def resolve_key(
    provider: str,
    agent_did: Optional[str] = None,
    storage: Optional["ServiceKeyStorage"] = None,
    require: bool = True,
) -> Optional[str]:
    """
    Quick key resolution without creating a service instance.

    Args:
        provider: Service provider
        agent_did: Agent DID (unused if storage provided)
        storage: ServiceKeyStorage instance (already bound to agent)
        require: If True, raise on missing key

    Returns:
        API key or None
    """
    resolver = KeyResolutionService(
        storage=storage,
        agent_did=agent_did,
    )
    return await resolver.resolve_key(provider, require=require)
