"""
Deploy Feature for Kestrel agents.

Exposes agent self-deployment orchestration via the tool system,
providing commands for deploying to cloud platforms like Cloud Run.

The orchestration logic itself lives on
:class:`kestrel_sovereign.features.deploy.core.DeployManagerCore` —
this feature is a thin agent-tool wrapper that adds the disabled-state
guard, parses ``!deploy <action>`` into a method call, and turns
"profile required" / "unknown action" into structured error dicts.
The operator-side ``kestrel deploy`` CLI delegates to the same manager
methods, so the two surfaces never drift out of sync.
"""

import logging
from typing import Any, Dict

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.deploy.manager import DeployManager
from kestrel_sovereign.features.deploy.models import DeployManagerError
from kestrel_sdk.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class DeployFeature(Feature):
    """Feature layer exposing agent self-deployment via the tool system."""

    @property
    def tool_description(self) -> str:
        return (
            "Deploy agent to cloud platforms - deploy to Cloud Run or Azure, "
            "check deployment status, view logs, and manage running deployments. "
            "Supports scale-to-zero and auto-scaling configurations."
        )

    async def initialize(self):
        """Initialize the deploy manager."""
        try:
            self.manager = DeployManager()

            # Check if we have any profiles configured
            if not self.manager.profiles:
                logger.warning("No deployment profiles configured. DeployFeature disabled.")
                self.disabled = True
                self.disabled_reason = "No deployment profiles in deploy_config.toml"
                return

            self.disabled = False
            logger.info(
                f"DeployFeature initialized with profiles: {', '.join(self.manager.profiles.keys())}"
            )

        except Exception as e:
            logger.warning(f"DeployFeature disabled: {e}")
            self.disabled = True
            self.disabled_reason = str(e)

    @tool(
        name="deploy_agent",
        description="Deploy or manage Kestrel agent on cloud platforms (usage: !deploy <action> [...]).",
        category=ToolCategory.SYSTEM,
        command_prefix="!deploy",
    )
    async def deploy_agent(
        self,
        action: str = "status",
        profile: str = "",
        tag: str = "latest",
    ) -> Dict[str, Any]:
        """
        Main entry point for agent deployment management.

        Actions:
            - status: Show current deployment status
            - deploy: Deploy agent to cloud platform
            - teardown: Delete deployed service
            - logs: View deployment logs
            - list: List all deployments
            - health: Check health of deployed service

        Args:
            action: Action to perform (status, deploy, teardown, logs, list, health)
            profile: Deployment profile name (required for deploy action)
            tag: Image tag to deploy (default: latest)

        Examples:
            !deploy status
            !deploy deploy profile=dev
            !deploy deploy profile=prod tag=v1.2.3
            !deploy teardown profile=dev
            !deploy logs profile=dev
            !deploy list
            !deploy health profile=dev
        """
        if getattr(self, "disabled", False):
            return {
                "action": action,
                "error": "Deploy feature is disabled",
                "reason": getattr(self, "disabled_reason", "No profiles configured"),
            }

        action_normalized = (action or "status").lower()

        if action_normalized in {"status"}:
            return await self._status()

        if action_normalized in {"deploy", "start"}:
            return await self._deploy(profile_name=profile, image_tag=tag)

        if action_normalized in {"teardown", "stop", "delete"}:
            return await self._teardown(profile_name=profile)

        if action_normalized in {"logs", "log"}:
            return await self._logs(profile_name=profile)

        if action_normalized in {"list", "ls"}:
            return await self._list_deployments()

        if action_normalized in {"health", "check"}:
            return await self._health_check(profile_name=profile)

        return {
            "success": False,
            "error": f"Unknown action: {action}",
            "available_actions": ["status", "deploy", "teardown", "logs", "list", "health"],
        }

    async def _status(self) -> Dict[str, Any]:
        """Get status of all active deployment sessions."""
        try:
            sessions = await self.manager.list_sessions()

            if not sessions:
                return {
                    "success": True,
                    "active_deployments": 0,
                    "sessions": [],
                    "message": "No active deployment sessions",
                }

            session_data = [session.to_dict() for session in sessions.values()]

            return {
                "success": True,
                "active_deployments": len(sessions),
                "sessions": session_data,
            }

        except DeployManagerError as e:
            return {"success": False, "error": str(e)}

    async def _deploy(self, profile_name: str, image_tag: str = "latest") -> Dict[str, Any]:
        """Deploy agent — delegates to ``DeployManagerCore.deploy_profile``.

        The agent-tool surface is the only one that knows the
        ``!deploy deploy profile=dev`` calling convention, so the
        "profile required" hint with the agent-flavored ``usage`` string
        stays here. The actual provider work lives on the manager.
        """
        if not profile_name:
            profiles = list(self.manager.profiles.keys())
            return {
                "success": False,
                "error": "Profile required",
                "available_profiles": profiles,
                "usage": "!deploy deploy profile=dev",
            }

        return await self.manager.deploy_profile(profile_name, image_tag)

    async def _teardown(self, profile_name: str) -> Dict[str, Any]:
        """Teardown deployed service — delegates to the manager."""
        if not profile_name:
            return {
                "success": False,
                "error": "Profile required",
                "usage": "!deploy teardown profile=dev",
            }

        return await self.manager.teardown_profile(profile_name)

    async def _logs(self, profile_name: str, lines: int = 100) -> Dict[str, Any]:
        """Get deployment logs — delegates to the manager."""
        if not profile_name:
            return {
                "success": False,
                "error": "Profile required",
                "usage": "!deploy logs profile=dev",
            }

        return await self.manager.get_profile_logs(profile_name, lines=lines)

    async def _list_deployments(self) -> Dict[str, Any]:
        """List all deployments — delegates to the manager."""
        return await self.manager.list_all_deployments()

    async def _health_check(self, profile_name: str) -> Dict[str, Any]:
        """Check health — delegates to the manager."""
        if not profile_name:
            return {
                "success": False,
                "error": "Profile required",
                "usage": "!deploy health profile=dev",
            }

        return await self.manager.health_check_profile(profile_name)
