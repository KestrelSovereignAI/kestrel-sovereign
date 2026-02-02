"""
Layered Key Resolution Service for Kestrel.

Resolves API keys with billing attribution using three-tier priority:
1. Agent's own keys (highest priority - autonomous agents)
2. User's BYOK keys (user pays provider directly)
3. Platform keys (vending machine - companion wallet + margin)

Each resolution returns the key AND billing metadata for proper attribution.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from asyncpg import Pool
    from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
    from kestrel_sovereign.security.user_key_storage import UserKeyStorage
    from kestrel_sovereign.security.platform_key_storage import PlatformKeyStorage

logger = logging.getLogger(__name__)


class KeyNotConfiguredError(Exception):
    """Raised when no key is available from any source."""

    def __init__(self, provider: str, message: Optional[str] = None):
        self.provider = provider
        super().__init__(
            message or f"No API key available for {provider}. "
            f"Add your own key (BYOK) or ensure platform has access configured."
        )


@dataclass
class KeyResolutionResult:
    """
    Result of key resolution with billing attribution.

    Attributes:
        api_key: The decrypted API key
        source: Key source ('agent', 'user', 'platform')
        source_id: Reference ID (agent DID, user ID, or platform key ID)
        billing_mode: How to bill ('agent_wallet', 'no_charge', 'wallet_debit')
        margin_pct: Margin percentage (0 for agent/user, platform margin for vending)
    """
    api_key: str
    source: str  # 'agent', 'user', 'platform'
    source_id: Optional[str]
    billing_mode: str  # 'agent_wallet', 'no_charge', 'wallet_debit'
    margin_pct: Decimal


class LayeredKeyResolver:
    """
    Three-tier key resolution with billing attribution.

    Resolution Priority:
    1. Agent Key - Agent's own provisioned key (billing: agent_wallet)
    2. User BYOK - User's passphrase-encrypted key (billing: no_charge)
    3. Platform Key - Shared vending machine pool (billing: wallet_debit + margin)

    Usage:
        resolver = LayeredKeyResolver(pool)
        result = await resolver.resolve(
            companion_id="...",
            user_id="...",
            agent_did="did:...",
            provider="openrouter",
            user_passphrase="optional-if-byok",
        )
        # result.api_key, result.billing_mode, etc.
    """

    def __init__(
        self,
        pool: "Pool",
        agent_storage: Optional["ServiceKeyStorage"] = None,
    ):
        """
        Initialize the resolver.

        Args:
            pool: asyncpg connection pool for user/platform key storage
            agent_storage: Optional pre-configured agent key storage
                          (if not provided, will be created per-request with agent_did)
        """
        self._pool = pool
        self._agent_storage = agent_storage

    async def resolve(
        self,
        provider: str,
        companion_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_did: Optional[str] = None,
        user_passphrase: Optional[str] = None,
        require: bool = True,
    ) -> Optional[KeyResolutionResult]:
        """
        Resolve API key with billing attribution.

        Resolution order:
        1. Agent's own key (if agent_did provided)
        2. User's BYOK key (if user_id and passphrase provided)
        3. Platform key (vending machine)

        Args:
            provider: Service provider ID (openrouter, openai, etc.)
            companion_id: Companion ID (for billing context)
            user_id: User ID (for BYOK lookup)
            agent_did: Agent DID (for agent key lookup)
            user_passphrase: User's passphrase for BYOK decryption
            require: If True, raise KeyNotConfiguredError when not found

        Returns:
            KeyResolutionResult with key and billing info, or None if require=False

        Raises:
            KeyNotConfiguredError: If require=True and no key found
        """
        provider = provider.lower()

        # Tier 1: Agent's own key (highest priority)
        if agent_did:
            result = await self._try_agent_key(agent_did, provider)
            if result:
                logger.debug(f"Resolved {provider} key from agent storage")
                return result

        # Tier 2: User's BYOK key
        if user_id:
            result = await self._try_user_key(user_id, provider, user_passphrase)
            if result:
                logger.debug(f"Resolved {provider} key from user BYOK")
                return result

        # Tier 3: Platform key (vending machine)
        result = await self._try_platform_key(provider)
        if result:
            logger.debug(f"Resolved {provider} key from platform pool")
            return result

        # No key found
        if require:
            raise KeyNotConfiguredError(provider)

        return None

    async def _try_agent_key(
        self,
        agent_did: str,
        provider: str,
    ) -> Optional[KeyResolutionResult]:
        """
        Try to get key from agent's own storage.

        Returns:
            KeyResolutionResult or None
        """
        try:
            # Use pre-configured storage or create on-demand
            storage = self._agent_storage
            if not storage:
                # Import here to avoid circular imports
                from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
                from kestrel_sovereign.storage.async_database import AsyncDatabase

                # Agent storage needs the agent's local database
                # This is typically passed in, but we can try to locate it
                # For now, if not pre-configured, we skip agent storage
                logger.debug(f"No agent storage configured, skipping agent key lookup")
                return None

            key = await storage.get_key(provider_id=provider)
            if key:
                return KeyResolutionResult(
                    api_key=key,
                    source="agent",
                    source_id=agent_did,
                    billing_mode="agent_wallet",  # Debit agent/companion wallet
                    margin_pct=Decimal("0"),  # No margin for own key
                )
        except Exception as e:
            logger.debug(f"Agent key lookup for {provider}: {e}")

        return None

    async def _try_user_key(
        self,
        user_id: str,
        provider: str,
        passphrase: Optional[str],
    ) -> Optional[KeyResolutionResult]:
        """
        Try to get key from user's BYOK storage.

        Note: Requires passphrase for decryption.

        Returns:
            KeyResolutionResult or None
        """
        # Import here to avoid circular imports
        from kestrel_sovereign.security.user_key_storage import (
            UserKeyStorage,
            KeyNotFoundError,
            DecryptionError,
            PassphraseRequiredError,
        )

        try:
            user_storage = UserKeyStorage(self._pool, user_id)

            # First check if user has a key (doesn't need passphrase)
            if not await user_storage.has_key(provider):
                return None

            # If no passphrase, we know key exists but can't use it
            if not passphrase:
                logger.debug(f"User has {provider} BYOK but no passphrase provided")
                return None

            # Try to decrypt with passphrase
            key = await user_storage.get_key(provider, passphrase)
            return KeyResolutionResult(
                api_key=key,
                source="user",
                source_id=user_id,
                billing_mode="no_charge",  # User pays provider directly
                margin_pct=Decimal("0"),  # No platform margin
            )

        except KeyNotFoundError:
            return None
        except DecryptionError as e:
            logger.warning(f"User BYOK decryption failed for {provider}: {e}")
            return None
        except Exception as e:
            logger.debug(f"User key lookup for {provider}: {e}")
            return None

    async def _try_platform_key(
        self,
        provider: str,
    ) -> Optional[KeyResolutionResult]:
        """
        Try to get key from platform vending machine pool.

        Returns:
            KeyResolutionResult or None
        """
        # Import here to avoid circular imports
        from kestrel_sovereign.security.platform_key_storage import (
            PlatformKeyStorage,
            KeyNotFoundError,
            MasterKeyNotConfiguredError,
        )

        try:
            platform_storage = PlatformKeyStorage(self._pool)

            # Check if platform has this provider
            if not await platform_storage.has_key(provider):
                return None

            # Get key and margin
            key = await platform_storage.get_key(provider)
            margin = await platform_storage.get_margin(provider)

            return KeyResolutionResult(
                api_key=key,
                source="platform",
                source_id=None,  # Platform keys don't have individual IDs in result
                billing_mode="wallet_debit",  # Debit companion wallet + margin
                margin_pct=margin,
            )

        except KeyNotFoundError:
            return None
        except MasterKeyNotConfiguredError as e:
            logger.warning(f"Platform key not available: {e}")
            return None
        except Exception as e:
            logger.debug(f"Platform key lookup for {provider}: {e}")
            return None

    async def has_any_key(
        self,
        provider: str,
        user_id: Optional[str] = None,
        agent_did: Optional[str] = None,
    ) -> bool:
        """
        Check if any key source has a key for this provider.

        Does NOT require passphrase - just checks existence.

        Args:
            provider: Service provider
            user_id: User ID for BYOK check
            agent_did: Agent DID for agent key check

        Returns:
            True if any source has a key
        """
        provider = provider.lower()

        # Check agent storage
        if agent_did and self._agent_storage:
            try:
                if await self._agent_storage.has_key(provider_id=provider):
                    return True
            except Exception as e:
                logger.debug(f"Agent key check failed for {provider}: {e}")

        # Check user storage
        if user_id:
            try:
                from kestrel_sovereign.security.user_key_storage import UserKeyStorage
                user_storage = UserKeyStorage(self._pool, user_id)
                if await user_storage.has_key(provider):
                    return True
            except Exception as e:
                logger.debug(f"User key check failed for {provider}: {e}")

        # Check platform storage
        try:
            from kestrel_sovereign.security.platform_key_storage import PlatformKeyStorage
            platform_storage = PlatformKeyStorage(self._pool)
            if await platform_storage.has_key(provider):
                return True
        except Exception as e:
            logger.debug(f"Platform key check failed for {provider}: {e}")

        return False

    async def get_available_sources(
        self,
        provider: str,
        user_id: Optional[str] = None,
        agent_did: Optional[str] = None,
    ) -> dict:
        """
        Get which key sources are available for a provider.

        Useful for UI to show what options the user has.

        Args:
            provider: Service provider
            user_id: User ID for BYOK check
            agent_did: Agent DID for agent key check

        Returns:
            Dict with availability status:
            {
                "agent": True/False,
                "user": True/False,
                "platform": True/False,
                "platform_margin": Decimal or None
            }
        """
        provider = provider.lower()
        result = {
            "agent": False,
            "user": False,
            "platform": False,
            "platform_margin": None,
        }

        # Check agent storage
        if agent_did and self._agent_storage:
            try:
                result["agent"] = await self._agent_storage.has_key(provider_id=provider)
            except Exception as e:
                logger.debug(f"Agent source check failed for {provider}: {e}")

        # Check user storage
        if user_id:
            try:
                from kestrel_sovereign.security.user_key_storage import UserKeyStorage
                user_storage = UserKeyStorage(self._pool, user_id)
                result["user"] = await user_storage.has_key(provider)
            except Exception as e:
                logger.debug(f"User source check failed for {provider}: {e}")

        # Check platform storage
        try:
            from kestrel_sovereign.security.platform_key_storage import PlatformKeyStorage
            platform_storage = PlatformKeyStorage(self._pool)
            if await platform_storage.has_key(provider):
                result["platform"] = True
                result["platform_margin"] = await platform_storage.get_margin(provider)
        except Exception as e:
            logger.debug(f"Platform source check failed for {provider}: {e}")

        return result


# Convenience function for quick resolution
async def resolve_key(
    pool: "Pool",
    provider: str,
    companion_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_did: Optional[str] = None,
    user_passphrase: Optional[str] = None,
    agent_storage: Optional["ServiceKeyStorage"] = None,
    require: bool = True,
) -> Optional[KeyResolutionResult]:
    """
    Quick key resolution without creating resolver instance.

    Args:
        pool: asyncpg connection pool
        provider: Service provider
        companion_id: Companion ID (for billing)
        user_id: User ID (for BYOK)
        agent_did: Agent DID (for agent keys)
        user_passphrase: Passphrase for BYOK decryption
        agent_storage: Pre-configured agent storage
        require: If True, raise on not found

    Returns:
        KeyResolutionResult or None
    """
    resolver = LayeredKeyResolver(pool, agent_storage=agent_storage)
    return await resolver.resolve(
        provider=provider,
        companion_id=companion_id,
        user_id=user_id,
        agent_did=agent_did,
        user_passphrase=user_passphrase,
        require=require,
    )
