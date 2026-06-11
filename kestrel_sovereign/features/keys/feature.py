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

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import (
    hides_persisted_user_content,
    resolve_feature_database,
)
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

        # Get agent DID - REQUIRED for key storage
        self._agent_did = self.agent.did

        if not self._agent_did:
            logger.error("KeyManagementFeature: No agent DID available - key storage disabled")
            return

        if hides_persisted_user_content(self.agent):
            logger.info(
                "KeyManagementFeature: persistent key storage unavailable "
                "in current privacy mode"
            )
            return

        self._ensure_storage()

    def _ensure_storage(self) -> bool:
        if self._storage is not None:
            return True
        if not self._agent_did:
            return False
        db = resolve_feature_database(self.agent)
        # Initialize storage (requires KESTREL_DATA_KEY)
        if db:
            try:
                self._storage = ServiceKeyStorage(db, self._agent_did)
                logger.info(f"KeyManagementFeature initialized for agent {self._agent_did[:30]}...")
                return True
            except MasterKeyNotConfiguredError as e:
                logger.warning(f"KeyManagementFeature: Secure storage not available: {e}")
            except ValueError as e:
                logger.error(f"KeyManagementFeature: Invalid configuration: {e}")
        else:
            logger.warning("KeyManagementFeature: Database not available")
        return False

    def _persistent_key_storage_hidden(self) -> bool:
        return hides_persisted_user_content(self.agent)

    def _privacy_unavailable_result(self) -> ToolResult:
        return ToolResult.failed(
            "Service key storage is unavailable in the current privacy mode",
            data={"privacy_mode_blocks_persistent_storage": True},
        )

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
    ) -> ToolResult:
        """
        Add an API key for an external service.

        This stores the key encrypted with agent-derived keys.
        The key is never logged or exposed.

        Args:
            provider: Service provider (openrouter, openai, anthropic, lighthouse, github, runpod, vastai)
            api_key: The API key to store (will be encrypted)
            quota_limit: Optional usage limit
        """
        if self._persistent_key_storage_hidden():
            return self._privacy_unavailable_result()

        if not self._ensure_storage():
            return ToolResult.failed(
                "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            )

        provider = provider.lower()
        if provider not in KNOWN_PROVIDERS:
            valid_providers = ", ".join(KNOWN_PROVIDERS.keys())
            return ToolResult.failed(
                f"Unknown provider '{provider}'. Valid providers: {valid_providers}",
                data={"provider": provider},
            )

        try:
            key_id = await self._storage.store_key(
                provider_id=provider,
                api_key=api_key,
                quota_limit=quota_limit,
            )
        except Exception as e:
            logger.error(f"Failed to add service key: {e}")
            return ToolResult.failed(str(e))

        provider_info = KNOWN_PROVIDERS[provider]
        return ToolResult.ok(
            confirmation=f"API key for {provider_info['name']} stored securely",
            data={
                "success": True,
                "key_id": key_id,
                "provider": provider,
                "provider_name": provider_info["name"],
                "quota_limit": quota_limit,
                "message": f"API key for {provider_info['name']} stored securely.",
            },
        )

    @tool(
        name="list_service_keys",
        description="List configured service keys (no secrets exposed)",
        category=ToolCategory.SYSTEM,
        command_prefix="!list-keys"
    )
    async def list_service_keys(self) -> ToolResult:
        """
        List all configured service keys.

        Returns key metadata without exposing secrets:
        - Provider name
        - Quota usage
        - Active status
        """
        if self._persistent_key_storage_hidden():
            return self._privacy_unavailable_result()

        if not self._ensure_storage():
            return ToolResult.failed(
                "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            )

        try:
            keys = await self._storage.list_keys()
        except Exception as e:
            logger.error(f"Failed to list service keys: {e}")
            return ToolResult.failed(str(e))

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

        return ToolResult.ok(
            confirmation=f"Listed {len(key_list)} service key(s)",
            data={
                "success": True,
                "keys": key_list,
                "total": len(key_list),
            },
        )

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
    ) -> ToolResult:
        """
        Get usage statistics for a service key.

        Shows operations, units consumed, and estimated costs.

        Args:
            provider: Service provider
            days: Number of days to look back (default: 30)
        """
        if self._persistent_key_storage_hidden():
            return self._privacy_unavailable_result()

        if not self._ensure_storage():
            return ToolResult.failed(
                "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            )

        provider = provider.lower()

        try:
            usage_records = await self._storage.get_usage(
                provider_id=provider,
                days=days,
            )
        except Exception as e:
            logger.error(f"Failed to get key usage: {e}")
            return ToolResult.failed(str(e))

        by_operation: Dict[str, Dict[str, Any]] = {}
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
        return ToolResult.ok(
            confirmation=(
                f"Usage for {provider_info['name']} over last {days} day(s): "
                f"{len(usage_records)} ops, {total_units} units, "
                f"${round(total_cost, 4)}"
            ),
            data={
                "success": True,
                "provider": provider,
                "provider_name": provider_info["name"],
                "period_days": days,
                "total_operations": len(usage_records),
                "total_units": total_units,
                "estimated_cost_usd": round(total_cost, 4),
                "by_operation": by_operation,
            },
        )

    @tool(
        name="remove_service_key",
        description="Remove/deactivate a service key",
        category=ToolCategory.SYSTEM,
        command_prefix="!remove-key"
    )
    async def remove_service_key(
        self,
        provider: str,
    ) -> ToolResult:
        """
        Remove (deactivate) a service key.

        The key is marked inactive, not deleted, for audit purposes.

        Args:
            provider: Service provider to remove key for
        """
        if self._persistent_key_storage_hidden():
            return self._privacy_unavailable_result()

        if not self._ensure_storage():
            return ToolResult.failed(
                "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            )

        provider = provider.lower()

        try:
            await self._storage.deactivate_key(provider_id=provider)
        except Exception as e:
            logger.error(f"Failed to remove service key: {e}")
            return ToolResult.failed(str(e))

        provider_info = KNOWN_PROVIDERS.get(provider, {"name": provider})
        return ToolResult.ok(
            confirmation=f"API key for {provider_info['name']} has been deactivated",
            data={
                "success": True,
                "provider": provider,
                "provider_name": provider_info["name"],
                "message": f"API key for {provider_info['name']} has been deactivated.",
            },
        )

    @tool(
        name="delete_service_key",
        description="Permanently delete a service key",
        category=ToolCategory.SYSTEM,
        command_prefix="!delete-key"
    )
    async def delete_service_key(
        self,
        provider: str,
    ) -> ToolResult:
        """
        Permanently delete a service key.

        This hard-deletes the key from kestrel_sovereign.storage.

        Args:
            provider: Service provider to delete key for
        """
        if self._persistent_key_storage_hidden():
            return self._privacy_unavailable_result()

        if not self._ensure_storage():
            return ToolResult.failed(
                "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            )

        provider = provider.lower()

        try:
            await self._storage.delete_key(provider_id=provider)
        except Exception as e:
            logger.error(f"Failed to delete service key: {e}")
            return ToolResult.failed(str(e))

        provider_info = KNOWN_PROVIDERS.get(provider, {"name": provider})
        return ToolResult.ok(
            confirmation=f"API key for {provider_info['name']} has been permanently deleted",
            data={
                "success": True,
                "provider": provider,
                "provider_name": provider_info["name"],
                "message": f"API key for {provider_info['name']} has been permanently deleted.",
            },
        )

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
    ) -> ToolResult:
        """
        Rotate an API key for a service.

        This requires constitutional approval for security.
        The old key is replaced with the new key.

        Args:
            provider: Service provider
            new_api_key: The new API key
        """
        if self._persistent_key_storage_hidden():
            return self._privacy_unavailable_result()

        if not self._ensure_storage():
            return ToolResult.failed(
                "Key storage not available - check KESTREL_DATA_KEY and agent DID",
            )

        provider = provider.lower()

        # Get security feature for approval
        security = None
        if hasattr(self.agent, 'get_feature'):
            security = self.agent.get_feature("security")
        elif hasattr(self.agent, 'features'):
            security = self.agent.features.get("security")

        # Track whether the approval flow ran cleanly. If the queue is
        # available but the approval request itself raised, the legacy
        # path silently proceeded "without approval" and reported success.
        # That hides a security-relevant bypass; surface it as PARTIAL so
        # the agent must speak that the rotation took effect WITHOUT
        # constitutional review.
        approval_bypassed_reason: Optional[str] = None
        if security and hasattr(security, 'approval_queue'):
            try:
                approved, approval_type = await security.approval_queue.request_approval(
                    feature_name="keys",
                    tool_name="rotate_service_key",
                    tool_args={"provider": provider},
                    timeout=APPROVAL_TIMEOUT_DEFAULT,
                )

                if not approved:
                    return ToolResult.failed(
                        "Key rotation not approved by constitutional review",
                        data={"provider": provider},
                    )
            except Exception as e:
                logger.warning(f"Approval request failed: {e}, proceeding without approval")
                approval_bypassed_reason = str(e)

        try:
            # Store new key (replaces old one due to UNIQUE constraint on agent_did + provider_id)
            key_id = await self._storage.store_key(
                provider_id=provider,
                api_key=new_api_key,
            )
        except Exception as e:
            logger.error(f"Failed to rotate service key: {e}")
            return ToolResult.failed(str(e))

        provider_info = KNOWN_PROVIDERS.get(provider, {"name": provider})
        data = {
            "success": True,
            "key_id": key_id,
            "provider": provider,
            "provider_name": provider_info["name"],
            "message": f"API key for {provider_info['name']} has been rotated.",
        }

        if approval_bypassed_reason is not None:
            data["approval_bypassed_reason"] = approval_bypassed_reason
            return ToolResult.partial(
                confirmation=f"API key for {provider_info['name']} rotated",
                error=(
                    "constitutional approval queue raised "
                    f"({approval_bypassed_reason!r}); rotation proceeded "
                    "WITHOUT review. Investigate the approval pipeline "
                    "before relying on rotation as a security control."
                ),
                data=data,
            )

        return ToolResult.ok(
            confirmation=f"API key for {provider_info['name']} has been rotated",
            data=data,
        )

    @tool(
        name="list_providers",
        description="List available service providers",
        category=ToolCategory.SYSTEM,
        command_prefix="!providers"
    )
    async def list_providers(self) -> ToolResult:
        """
        List all supported service providers.

        Shows provider info including sub-account support.
        """
        providers = []
        for provider_id, info in KNOWN_PROVIDERS.items():
            providers.append({
                "id": provider_id,
                "name": info["name"],
                "supports_sub_accounts": info.get("supports_sub_accounts", False),
            })

        return ToolResult.ok(
            confirmation=f"Listed {len(providers)} supported provider(s)",
            data={
                "success": True,
                "providers": providers,
                "total": len(providers),
            },
        )

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
        if self._persistent_key_storage_hidden():
            return None

        if not self._ensure_storage():
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
        if self._persistent_key_storage_hidden():
            return False

        if not self._ensure_storage():
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
        if self._persistent_key_storage_hidden():
            return True

        if not self._ensure_storage():
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
