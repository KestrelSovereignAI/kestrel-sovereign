"""
Key Management Feature for Kestrel.

Provides agent tools for managing external service API keys:
- Add/remove service keys
- List configured keys (no secrets exposed)
- View usage statistics
- Set quotas
- Rotate keys (with constitutional approval)

Security:
- All keys encrypted with agent-derived AES-256-GCM keys
- Each agent has isolated key storage
- No secrets logged or exposed in responses
"""

import logging
from typing import Any, Dict, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.tools.base import ToolCategory
from kestrel_sovereign.kestrel_config.constants import APPROVAL_TIMEOUT_DEFAULT
from kestrel_sovereign.security.service_key_storage import (
    ServiceKeyStorage,
    KeyNotConfiguredError,
    KeyStorageError,
    KNOWN_PROVIDERS,
)
from kestrel_sovereign.security.agent_encryption import MasterKeyNotConfiguredError

logger = logging.getLogger(__name__)


class KeyManagementFeature(Feature):
    """
    Manage API keys for external services.

    This feature provides secure storage and management of API keys
    for services like OpenRouter, OpenAI, Lighthouse, GitHub, RunPod, and Vast.ai.

    All keys are encrypted with agent-derived keys - each agent's keys
    are cryptographically isolated from other agents.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage API keys for external services (OpenRouter, OpenAI, Lighthouse, GitHub, RunPod, Vast.ai). "
            "Add keys, view usage, set quotas, and rotate keys."
        )

    async def initialize(self):
        """Initialize the key management feature."""
        self._storage: Optional[ServiceKeyStorage] = None
        self._agent_did: Optional[str] = None

        db = resolve_feature_database(self.agent)

        # Get agent DID - REQUIRED for key storage
        self._agent_did = self.agent.did

        if not self._agent_did:
            logger.error("KeyManagementFeature: No agent DID available - key storage disabled")
            return

        # Initialize storage (requires KESTREL_DATA_KEY)
        if db:
            try:
                self._storage = ServiceKeyStorage(db, self._agent_did)
                logger.info(f"KeyManagementFeature initialized for agent {self._agent_did[:30]}...")
            except MasterKeyNotConfiguredError as e:
                logger.warning(f"KeyManagementFeature: Secure storage not available: {e}")
            except ValueError as e:
                logger.error(f"KeyManagementFeature: Invalid configuration: {e}")
        else:
            logger.warning("KeyManagementFeature: Database not available")

    # =========================================================================
    # Key Management Tools
    # =========================================================================

    @tool(
        name="add_service_key",
        description="Add an API key for an external service",
        category=ToolCategory.SYSTEM,
        command_prefix="!add-key"
    )
    async def add_service_key(
        self,
        provider: str,
        api_key: str,
        quota_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Add an API key for an external service.

        This stores the key encrypted with agent-derived keys.
        The key is never logged or exposed.

        Args:
            provider: Service provider (openrouter, openai, anthropic, lighthouse, github, runpod, vastai)
            api_key: The API key to store (will be encrypted)
            quota_limit: Optional usage limit

        Returns:
            Result with key_id if successful
        """
        if not self._storage:
            return {
                "success": False,
                "error": "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            }

        # Validate provider
        provider = provider.lower()
        if provider not in KNOWN_PROVIDERS:
            valid_providers = ", ".join(KNOWN_PROVIDERS.keys())
            return {
                "success": False,
                "error": f"Unknown provider '{provider}'. Valid providers: {valid_providers}",
            }

        try:
            key_id = await self._storage.store_key(
                provider_id=provider,
                api_key=api_key,
                quota_limit=quota_limit,
            )

            provider_info = KNOWN_PROVIDERS[provider]
            return {
                "success": True,
                "key_id": key_id,
                "provider": provider,
                "provider_name": provider_info["name"],
                "quota_limit": quota_limit,
                "message": f"API key for {provider_info['name']} stored securely.",
            }

        except Exception as e:
            logger.error(f"Failed to add service key: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="list_service_keys",
        description="List configured service keys (no secrets exposed)",
        category=ToolCategory.SYSTEM,
        command_prefix="!list-keys"
    )
    async def list_service_keys(self) -> Dict[str, Any]:
        """
        List all configured service keys.

        Returns key metadata without exposing secrets:
        - Provider name
        - Quota usage
        - Active status

        Returns:
            List of configured keys
        """
        if not self._storage:
            return {
                "success": False,
                "error": "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            }

        try:
            keys = await self._storage.list_keys()

            # Format for display
            key_list = []
            for key in keys:
                provider_info = KNOWN_PROVIDERS.get(key.provider_id, {"name": key.provider_id})
                key_list.append({
                    "provider": key.provider_id,
                    "provider_name": provider_info["name"],
                    "quota_limit": key.quota_limit,
                    "quota_used": key.quota_used,
                    "is_active": key.is_active,
                    "created_at": key.created_at.isoformat() if key.created_at else None,
                })

            return {
                "success": True,
                "keys": key_list,
                "total": len(key_list),
            }

        except Exception as e:
            logger.error(f"Failed to list service keys: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="get_key_usage",
        description="Get usage statistics for a service key",
        category=ToolCategory.SYSTEM,
        command_prefix="!key-usage"
    )
    async def get_key_usage(
        self,
        provider: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get usage statistics for a service key.

        Shows operations, units consumed, and estimated costs.

        Args:
            provider: Service provider
            days: Number of days to look back (default: 30)

        Returns:
            Usage statistics
        """
        if not self._storage:
            return {
                "success": False,
                "error": "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            }

        provider = provider.lower()

        try:
            usage_records = await self._storage.get_usage(
                provider_id=provider,
                days=days,
            )

            # Aggregate by operation
            by_operation = {}
            total_units = 0
            total_cost = 0.0

            for record in usage_records:
                op = record.operation
                if op not in by_operation:
                    by_operation[op] = {"count": 0, "units": 0, "cost": 0.0}
                by_operation[op]["count"] += 1
                by_operation[op]["units"] += record.units_consumed
                if record.cost_estimate_usd:
                    by_operation[op]["cost"] += record.cost_estimate_usd

                total_units += record.units_consumed
                if record.cost_estimate_usd:
                    total_cost += record.cost_estimate_usd

            provider_info = KNOWN_PROVIDERS.get(provider, {"name": provider})
            return {
                "success": True,
                "provider": provider,
                "provider_name": provider_info["name"],
                "period_days": days,
                "total_operations": len(usage_records),
                "total_units": total_units,
                "estimated_cost_usd": round(total_cost, 4),
                "by_operation": by_operation,
            }

        except Exception as e:
            logger.error(f"Failed to get key usage: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="remove_service_key",
        description="Remove/deactivate a service key",
        category=ToolCategory.SYSTEM,
        command_prefix="!remove-key"
    )
    async def remove_service_key(
        self,
        provider: str,
    ) -> Dict[str, Any]:
        """
        Remove (deactivate) a service key.

        The key is marked inactive, not deleted, for audit purposes.

        Args:
            provider: Service provider to remove key for

        Returns:
            Result of deactivation
        """
        if not self._storage:
            return {
                "success": False,
                "error": "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            }

        provider = provider.lower()

        try:
            await self._storage.deactivate_key(provider_id=provider)

            provider_info = KNOWN_PROVIDERS.get(provider, {"name": provider})
            return {
                "success": True,
                "provider": provider,
                "provider_name": provider_info["name"],
                "message": f"API key for {provider_info['name']} has been deactivated.",
            }

        except Exception as e:
            logger.error(f"Failed to remove service key: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="delete_service_key",
        description="Permanently delete a service key",
        category=ToolCategory.SYSTEM,
        command_prefix="!delete-key"
    )
    async def delete_service_key(
        self,
        provider: str,
    ) -> Dict[str, Any]:
        """
        Permanently delete a service key.

        This hard-deletes the key from kestrel_sovereign.storage.

        Args:
            provider: Service provider to delete key for

        Returns:
            Result of deletion
        """
        if not self._storage:
            return {
                "success": False,
                "error": "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            }

        provider = provider.lower()

        try:
            await self._storage.delete_key(provider_id=provider)

            provider_info = KNOWN_PROVIDERS.get(provider, {"name": provider})
            return {
                "success": True,
                "provider": provider,
                "provider_name": provider_info["name"],
                "message": f"API key for {provider_info['name']} has been permanently deleted.",
            }

        except Exception as e:
            logger.error(f"Failed to delete service key: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="rotate_service_key",
        description="Rotate an API key (requires constitutional approval)",
        category=ToolCategory.SYSTEM,
        command_prefix="!rotate-key"
    )
    async def rotate_service_key(
        self,
        provider: str,
        new_api_key: str,
    ) -> Dict[str, Any]:
        """
        Rotate an API key for a service.

        This requires constitutional approval for security.
        The old key is replaced with the new key.

        Args:
            provider: Service provider
            new_api_key: The new API key

        Returns:
            Result of key rotation
        """
        if not self._storage:
            return {
                "success": False,
                "error": "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            }

        provider = provider.lower()

        # Get security feature for approval
        security = None
        if hasattr(self.agent, 'get_feature'):
            security = self.agent.get_feature("security")
        elif hasattr(self.agent, 'features'):
            security = self.agent.features.get("security")

        if security and hasattr(security, 'approval_queue'):
            try:
                approved, approval_type = await security.approval_queue.request_approval(
                    feature_name="keys",
                    tool_name="rotate_service_key",
                    tool_args={"provider": provider},
                    timeout=APPROVAL_TIMEOUT_DEFAULT,
                )

                if not approved:
                    return {
                        "success": False,
                        "error": "Key rotation not approved by constitutional review",
                    }
            except Exception as e:
                logger.warning(f"Approval request failed: {e}, proceeding without approval")

        try:
            # Store new key (replaces old one due to UNIQUE constraint on agent_did + provider_id)
            key_id = await self._storage.store_key(
                provider_id=provider,
                api_key=new_api_key,
            )

            provider_info = KNOWN_PROVIDERS.get(provider, {"name": provider})
            return {
                "success": True,
                "key_id": key_id,
                "provider": provider,
                "provider_name": provider_info["name"],
                "message": f"API key for {provider_info['name']} has been rotated.",
            }

        except Exception as e:
            logger.error(f"Failed to rotate service key: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="list_providers",
        description="List available service providers",
        category=ToolCategory.SYSTEM,
        command_prefix="!providers"
    )
    async def list_providers(self) -> Dict[str, Any]:
        """
        List all supported service providers.

        Shows provider info including sub-account support.

        Returns:
            List of supported providers
        """
        providers = []
        for provider_id, info in KNOWN_PROVIDERS.items():
            providers.append({
                "id": provider_id,
                "name": info["name"],
                "supports_sub_accounts": info.get("supports_sub_accounts", False),
            })

        return {
            "success": True,
            "providers": providers,
            "total": len(providers),
        }

    # =========================================================================
    # Internal API (for other features to use)
    # =========================================================================

    async def get_key(self, provider: str) -> Optional[str]:
        """
        Get decrypted API key for a provider.

        This is for internal use by other features (e.g., LighthouseProvider).

        Args:
            provider: Service provider

        Returns:
            Decrypted API key or None if not configured
        """
        if not self._storage:
            return None

        try:
            return await self._storage.get_key(provider_id=provider.lower())
        except KeyNotConfiguredError:
            return None
        except Exception as e:
            logger.error(f"Failed to get key for {provider}: {e}")
            return None

    async def has_key(self, provider: str) -> bool:
        """
        Check if agent has a key configured for a provider.

        Args:
            provider: Service provider

        Returns:
            True if key exists and is active
        """
        if not self._storage:
            return False

        try:
            return await self._storage.has_key(provider_id=provider.lower())
        except Exception as e:
            logger.error(f"Failed to check key for {provider}: {e}")
            return False

    async def check_and_record_usage(
        self,
        provider: str,
        operation: str,
        units: int = 1,
        cost_estimate: Optional[float] = None,
    ) -> bool:
        """
        Check quota and record usage.

        Args:
            provider: Service provider
            operation: Operation type
            units: Units consumed
            cost_estimate: Estimated cost

        Returns:
            True if operation allowed, False if quota exceeded
        """
        if not self._storage:
            return True  # No storage = no quota enforcement

        provider = provider.lower()

        # Check quota
        allowed = await self._storage.check_quota(
            provider_id=provider,
            units=units,
        )

        if not allowed:
            logger.warning(f"Quota exceeded for provider={provider}")
            return False

        # Record usage
        await self._storage.record_usage(
            provider_id=provider,
            operation=operation,
            units=units,
            cost_estimate=cost_estimate,
        )

        return True
