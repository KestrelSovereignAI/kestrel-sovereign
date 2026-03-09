"""
Sleep integration hooks for the Reflection feature.

Provides hook points that integrate reflection into the agent's
sleep cycle for automatic nightly reflection.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ReflectionSleepHook:
    """
    Integrates reflection into the sleep cycle.

    This hook is called by the SleepMixin during the sleep process to:
    1. Run pre-sleep reflection (before consolidation)
    2. Run post-consolidation reflection (after episodes are created)

    The reflection results are included in the SleepReport.
    """

    def __init__(self, reflection_feature):
        """
        Initialize the sleep hook.

        Args:
            reflection_feature: ReflectionFeature instance
        """
        self.reflection = reflection_feature

    async def on_pre_sleep(self, agent) -> Dict[str, Any]:
        """
        Called before memory consolidation.

        Analyzes the current session before memories get reorganized.
        This is a quick, shallow analysis to capture immediate patterns.

        Args:
            agent: The KestrelAgent instance

        Returns:
            Dict with reflection results
        """
        logger.info("Running pre-sleep reflection")

        try:
            # Use the reflection feature's method
            result = await self.reflection.on_pre_sleep()

            if result.get("success"):
                logger.info(
                    f"Pre-sleep reflection complete: "
                    f"{result.get('insights_generated', 0)} insights"
                )
            else:
                logger.warning(
                    f"Pre-sleep reflection failed: {result.get('error', 'unknown')}"
                )

            return result

        except Exception as e:
            logger.error(f"Pre-sleep reflection error: {e}")
            return {
                "success": False,
                "error": str(e),
                "phase": "pre_sleep",
            }

    async def on_post_consolidation(
        self,
        agent,
        consolidation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Called after memory consolidation, before sovereignty export.

        Uses the newly created memory episodes for deeper reflection.
        Only runs if consolidation produced new episodes.

        Args:
            agent: The KestrelAgent instance
            consolidation_result: Results from memory consolidation

        Returns:
            Dict with reflection results
        """
        episodes_created = consolidation_result.get("episodes_created", 0)

        if episodes_created == 0:
            logger.debug("Skipping post-consolidation reflection (no new episodes)")
            return {
                "success": True,
                "skipped": True,
                "reason": "No new episodes from consolidation",
            }

        logger.info(
            f"Running post-consolidation reflection "
            f"({episodes_created} new episodes)"
        )

        try:
            result = await self.reflection.on_post_consolidation(consolidation_result)

            if result.get("success"):
                logger.info(
                    f"Post-consolidation reflection complete: "
                    f"{result.get('insights_generated', 0)} insights"
                )
            else:
                if result.get("skipped"):
                    logger.debug(f"Reflection skipped: {result.get('reason')}")
                else:
                    logger.warning(
                        f"Post-consolidation reflection failed: "
                        f"{result.get('error', 'unknown')}"
                    )

            return result

        except Exception as e:
            logger.error(f"Post-consolidation reflection error: {e}")
            return {
                "success": False,
                "error": str(e),
                "phase": "post_consolidation",
            }


def create_reflection_hook(agent) -> Optional[ReflectionSleepHook]:
    """
    Create a reflection hook for an agent if the reflection feature is available.

    Args:
        agent: KestrelAgent instance

    Returns:
        ReflectionSleepHook if reflection feature exists, None otherwise
    """
    # Try to get the reflection feature
    reflection_feature = None

    if hasattr(agent, 'get_feature'):
        reflection_feature = agent.get_feature("reflection") or agent.get_feature("ReflectionFeature")
    elif hasattr(agent, 'features'):
        reflection_feature = agent.features.get("ReflectionFeature") or agent.features.get("reflection")

    if reflection_feature is None:
        logger.debug("Reflection feature not found, sleep hook not created")
        return None

    return ReflectionSleepHook(reflection_feature)
