"""
Self-model management handler for reflection feature.

Handles retrieval and updates of the agent's self-model.
"""

import logging
from typing import Dict, Any, Optional

from kestrel_sovereign.kestrel_config.constants import APPROVAL_TIMEOUT_DEFAULT

from .db_helpers import ReflectionDatabaseHelper

logger = logging.getLogger(__name__)


class SelfModelHandler:
    """Handles self-model retrieval and updates."""

    def __init__(
        self,
        self_model_manager,
        economic_gate,
        db_helper: ReflectionDatabaseHelper,
        agent,
    ):
        """
        Initialize the self-model handler.

        Args:
            self_model_manager: SelfModelManager instance
            economic_gate: EconomicGate for access control
            db_helper: Database helper for insight retrieval
            agent: Agent instance for feature access
        """
        self.self_model_manager = self_model_manager
        self.economic_gate = economic_gate
        self.db_helper = db_helper
        self.agent = agent

    async def get_self_model(self) -> Dict[str, Any]:
        """
        Get the agent's current self-model.

        The self-model captures:
        - Personality traits (e.g., helpfulness, formality)
        - Communication style preferences
        - Learned user preferences
        - Behavior patterns (successes and failures)

        Returns:
            Current self-model data
        """
        if not self.self_model_manager:
            return {
                "success": False,
                "error": "Self-model manager not available - check LIGHTHOUSE_API_KEY configuration",
            }

        try:
            model = await self.self_model_manager.get_self_model()
            if model:
                return {
                    "success": True,
                    "self_model": model.to_dict(),
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to load self-model",
                }

        except Exception as e:
            logger.error(f"Failed to get self-model: {e}")
            return {"success": False, "error": str(e)}

    async def update_self_model(self, from_session_id: str = None) -> Dict[str, Any]:
        """
        Update the self-model based on recent insights.

        This requires:
        - Economic eligibility (paid tier)
        - Constitutional approval for self-modification
        - LIGHTHOUSE_API_KEY environment variable configured

        Args:
            from_session_id: Optional session ID to get insights from (default: most recent)

        Returns:
            Update result including new model version
        """
        if not self.self_model_manager:
            return {
                "success": False,
                "error": "Self-model manager not available - check LIGHTHOUSE_API_KEY configuration",
            }

        # Check economic eligibility
        if self.economic_gate and not self.economic_gate.can_update_self_model():
            return {
                "success": False,
                "error": "Self-model updates require paid tier",
            }

        # Get security feature for approval
        security = self._get_security_feature()
        if not security:
            return {
                "success": False,
                "error": "Security feature not available for constitutional approval",
            }

        # Get recent actionable insights
        try:
            if not self.db_helper:
                return {"success": False, "error": "Database not available"}

            insights = await self.db_helper.get_actionable_insights(
                from_session_id=from_session_id,
                limit=20,
            )

            if not insights:
                return {
                    "success": False,
                    "error": "No actionable insights found to update self-model",
                }

            # Update the self-model
            updated = await self.self_model_manager.update_self_model(
                insights=insights,
                security_feature=security,
                timeout=APPROVAL_TIMEOUT_DEFAULT,
            )

            if updated:
                model = await self.self_model_manager.get_self_model()
                return {
                    "success": True,
                    "updated": True,
                    "insights_applied": len(insights),
                    "new_version": model.version if model else None,
                }
            else:
                return {
                    "success": False,
                    "error": "Self-model update not approved or failed",
                }

        except Exception as e:
            logger.error(f"Failed to update self-model: {e}")
            return {"success": False, "error": str(e)}

    def _get_security_feature(self):
        """Get the security feature from the agent."""
        if hasattr(self.agent, 'get_feature'):
            return self.agent.get_feature("security")
        elif hasattr(self.agent, 'features'):
            return self.agent.features.get("security")
        return None