"""
Self-Model Manager for Reflection Feature.

Manages the agent's self-model stored in decentralized storage (Lighthouse/IPFS/Filecoin).
The self-model captures the agent's learned personality traits, communication style,
preferences, and behavior patterns.

All self-model updates require constitutional approval.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kestrel_sovereign.storage.providers.base import StorageProvider
    from kestrel_sovereign.features.security.feature import SecurityFeature
    from .models import Insight, SelfModel

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


class SelfModelManager:
    """Manages agent self-model in decentralized storage.

    The self-model is a structured representation of the agent's:
    - Personality traits (e.g., helpfulness: 0.9, humor: 0.3)
    - Communication style (e.g., formal vs casual, verbose vs concise)
    - Learned preferences (e.g., "user prefers code examples")
    - Behavior patterns (e.g., "always explain reasoning")

    The model is stored encrypted in Lighthouse/IPFS/Filecoin for
    sovereignty and persistence.
    """

    # Database table for tracking model CIDs
    MODEL_POINTER_KEY = "self_model_cid"

    def __init__(
        self,
        storage_provider: "StorageProvider",
        agent_did: str,
        db=None,
    ):
        """Initialize the self-model manager.

        Args:
            storage_provider: Decentralized storage provider (Lighthouse)
            agent_did: The agent's DID (decentralized identifier)
            db: Database connection for storing model pointers

        Raises:
            ConfigurationError: If no decentralized storage provider is available
        """
        # FAIL FAST - no fallbacks
        if not os.environ.get("LIGHTHOUSE_API_KEY"):
            raise ConfigurationError(
                "A decentralized storage provider is required for self-model storage. "
                "Set LIGHTHOUSE_API_KEY."
            )
        if not storage_provider.is_available():
            raise ConfigurationError(
                f"Storage provider '{storage_provider.provider_name}' is not available — "
                "check your configuration."
            )

        self.storage = storage_provider
        self.agent_did = agent_did
        self._db = db
        self._key_hash: Optional[str] = None
        self._current_model: Optional["SelfModel"] = None

    async def load_self_model(self) -> "SelfModel":
        """Load the current self-model from decentralized storage.

        If no model exists yet, returns a default model.

        Returns:
            The agent's current SelfModel
        """
        from .models import SelfModel

        # Get the latest model CID from local database
        cid = await self._get_latest_model_cid()

        if not cid:
            logger.info("No existing self-model found, creating default")
            return SelfModel.default(self.agent_did)

        try:
            # Retrieve from Lighthouse
            data = await self.storage.retrieve(cid, encryption_key_hash=self._key_hash)
            model = SelfModel.from_bytes(data)
            self._current_model = model
            logger.info(f"Loaded self-model v{model.version} from {cid[:16]}...")
            return model

        except Exception as e:
            logger.error(f"Failed to load self-model from {cid}: {e}")
            # Return default on error - don't fail silently
            raise

    async def update_self_model(
        self,
        insights: List["Insight"],
        security_feature: "SecurityFeature",
        timeout: float = 300.0,
    ) -> bool:
        """Update the self-model based on new insights.

        This method:
        1. Loads the current model
        2. Applies insights to generate a proposed model
        3. Requests constitutional approval for the changes
        4. If approved, stores the new model in Filecoin
        5. Updates the local pointer to the new CID

        Args:
            insights: List of insights to apply to the model
            security_feature: SecurityFeature for constitutional approval
            timeout: Approval timeout in seconds

        Returns:
            True if model was updated, False if not approved
        """
        from .models import SelfModel

        # Load current model
        current = await self.load_self_model()

        # Apply insights to generate proposed model
        proposed = self._apply_insights_to_model(current, insights)

        # Compute diff for approval request
        diff = self._compute_diff(current, proposed)

        if not diff:
            logger.info("No changes to self-model from insights")
            return True

        # Request constitutional approval
        logger.info(f"Requesting approval for self-model update: {len(diff)} changes")

        approval_request = {
            "feature_name": "reflection",
            "tool_name": "self_model_update",
            "tool_args": {
                "changes": diff[:500],  # Truncate for display
                "version": proposed.version,
                "insight_count": len(insights),
            },
        }

        try:
            approved, approval_type = await security_feature.approval_queue.request_approval(
                **approval_request,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Approval request failed: {e}")
            return False

        if not approved:
            logger.info("Self-model update not approved")
            return False

        # Store in Filecoin
        try:
            result = await self.storage.store(
                content=proposed.to_bytes(),
                metadata={
                    "agent_did": self.agent_did,
                    "type": "self-model",
                    "version": proposed.version,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                encrypt=True,
            )

            # Update local pointer
            await self._update_model_pointer(result.cid, result.encryption_key_hash)
            self._current_model = proposed
            self._key_hash = result.encryption_key_hash

            logger.info(f"Self-model v{proposed.version} stored at {result.cid[:16]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to store self-model: {e}")
            raise

    async def get_self_model(self) -> Optional["SelfModel"]:
        """Get the cached self-model or load from kestrel_sovereign.storage.

        Returns:
            The current SelfModel or None if not available
        """
        if self._current_model:
            return self._current_model
        return await self.load_self_model()

    def _apply_insights_to_model(
        self,
        current: "SelfModel",
        insights: List["Insight"],
    ) -> "SelfModel":
        """Apply insights to generate a new model version.

        Args:
            current: Current self-model
            insights: Insights to apply

        Returns:
            New proposed SelfModel
        """
        from .models import SelfModel, InsightType

        # Create a copy with incremented version
        new_model = SelfModel(
            agent_did=current.agent_did,
            version=current.version + 1,
            personality_traits=dict(current.personality_traits),
            communication_style=dict(current.communication_style),
            learned_preferences=list(current.learned_preferences),
            behavior_patterns=list(current.behavior_patterns),
            created_at=current.created_at,
            updated_at=datetime.now(timezone.utc),
        )

        for insight in insights:
            if not insight.actionable:
                continue

            # Extract trait adjustments from patterns
            if insight.type == InsightType.PATTERN:
                self._apply_pattern_insight(new_model, insight)
            elif insight.type == InsightType.SUCCESS:
                self._apply_success_insight(new_model, insight)
            elif insight.type == InsightType.FAILURE:
                self._apply_failure_insight(new_model, insight)
            elif insight.type == InsightType.IMPROVEMENT:
                self._apply_improvement_insight(new_model, insight)

        return new_model

    def _apply_pattern_insight(self, model: "SelfModel", insight: "Insight") -> None:
        """Apply a pattern insight to the model."""
        # Add to learned preferences if it's about user preferences
        if "prefer" in insight.description.lower():
            preference = f"[Pattern] {insight.title}"
            if preference not in model.learned_preferences:
                model.learned_preferences.append(preference)

    def _apply_success_insight(self, model: "SelfModel", insight: "Insight") -> None:
        """Apply a success insight to the model."""
        # Reinforce successful behavior patterns
        pattern = f"[Success] {insight.title}"
        if pattern not in model.behavior_patterns:
            model.behavior_patterns.append(pattern)

    def _apply_failure_insight(self, model: "SelfModel", insight: "Insight") -> None:
        """Apply a failure insight to the model."""
        # Add to behavior patterns as something to avoid
        pattern = f"[Avoid] {insight.title}"
        if pattern not in model.behavior_patterns:
            model.behavior_patterns.append(pattern)

    def _apply_improvement_insight(self, model: "SelfModel", insight: "Insight") -> None:
        """Apply an improvement insight to the model."""
        if insight.suggested_action:
            preference = f"[Improvement] {insight.suggested_action}"
            if preference not in model.learned_preferences:
                model.learned_preferences.append(preference)

    def _compute_diff(self, old: "SelfModel", new: "SelfModel") -> str:
        """Compute a human-readable diff between models.

        Args:
            old: Previous model
            new: Proposed model

        Returns:
            Human-readable diff string
        """
        changes = []

        # Check trait changes
        for trait, new_val in new.personality_traits.items():
            old_val = old.personality_traits.get(trait)
            if old_val != new_val:
                changes.append(f"Trait '{trait}': {old_val} -> {new_val}")

        # Check new traits
        for trait in set(new.personality_traits) - set(old.personality_traits):
            changes.append(f"New trait '{trait}': {new.personality_traits[trait]}")

        # Check preference changes
        new_prefs = set(new.learned_preferences) - set(old.learned_preferences)
        for pref in new_prefs:
            changes.append(f"New preference: {pref}")

        # Check behavior changes
        new_behaviors = set(new.behavior_patterns) - set(old.behavior_patterns)
        for behavior in new_behaviors:
            changes.append(f"New behavior: {behavior}")

        return "\n".join(changes)

    async def _get_latest_model_cid(self) -> Optional[str]:
        """Get the CID of the latest stored model.

        Returns:
            CID string or None if no model stored
        """
        if not self._db:
            return None

        try:
            result = await self._db.fetchall(
                """
                SELECT value FROM agent_metadata
                WHERE agent_id = ? AND key = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (self.agent_did, self.MODEL_POINTER_KEY),
            )
            if result:
                data = json.loads(result[0][0])
                self._key_hash = data.get("key_hash")
                return data.get("cid")
            return None
        except Exception as e:
            logger.warning(f"Failed to get model CID: {e}")
            return None

    async def _update_model_pointer(self, cid: str, key_hash: Optional[str]) -> None:
        """Update the local pointer to the current model CID.

        Args:
            cid: New model CID
            key_hash: Encryption key hash
        """
        if not self._db:
            return

        try:
            value = json.dumps({"cid": cid, "key_hash": key_hash})
            await self._db.execute(
                """
                INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.agent_did, self.MODEL_POINTER_KEY, value, datetime.now(timezone.utc)),
            )
        except Exception as e:
            logger.error(f"Failed to update model pointer: {e}")
            raise


async def create_self_model_manager(
    storage_provider: "LighthouseProvider",
    agent_did: str,
    db=None,
) -> SelfModelManager:
    """Factory function to create a SelfModelManager.

    Args:
        storage_provider: Lighthouse provider
        agent_did: Agent's DID
        db: Optional database connection

    Returns:
        Initialized SelfModelManager

    Raises:
        ConfigurationError: If required configuration is missing
    """
    return SelfModelManager(storage_provider, agent_did, db)
