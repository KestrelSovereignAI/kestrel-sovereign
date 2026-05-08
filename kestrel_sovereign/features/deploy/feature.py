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

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.deploy.manager import DeployManager
from kestrel_sovereign.features.deploy.models import DeployManagerError

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
    ) -> ToolResult:
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
            return ToolResult.failed(
                "Deploy feature is disabled",
                data={
                    "action": action,
                    "error": "Deploy feature is disabled",
                    "reason": getattr(self, "disabled_reason", "No profiles configured"),
                },
            )

        action_normalized = (action or "status").lower()

        # Internal helpers (_status, _deploy, etc.) still return
        # legacy dicts with {"success": True/False, ...}. Wrap the
        # outcome at the @tool boundary based on the success flag.
        if action_normalized in {"status"}:
            result_dict = await self._status()
        elif action_normalized in {"deploy", "start"}:
            result_dict = await self._deploy(profile_name=profile, image_tag=tag)
        elif action_normalized in {"teardown", "stop", "delete"}:
            result_dict = await self._teardown(profile_name=profile)
        elif action_normalized in {"logs", "log"}:
            result_dict = await self._logs(profile_name=profile)
        elif action_normalized in {"list", "ls"}:
            result_dict = await self._list_deployments()
        elif action_normalized in {"health", "check"}:
            result_dict = await self._health_check(profile_name=profile)
        else:
            return ToolResult.failed(
                f"Unknown action: {action}",
                data={
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": [
                        "status", "deploy", "teardown", "logs", "list", "health",
                    ],
                },
            )

        if isinstance(result_dict, dict) and result_dict.get("success") is False:
            return ToolResult.failed(
                result_dict.get("error") or f"deploy {action_normalized} failed",
                data=result_dict,
            )

        # Build a meaningful confirmation. The command renderer drops
        # scalar-only ``data`` blocks, so the confirmation is the only
        # surface the user sees for short outcomes (deploy status with
        # no sessions, health check with "Service not deployed", etc).
        # Pull the most informative legacy field per action — falling
        # back to a generic string only when nothing better exists.
        confirmation = self._format_deploy_confirmation(action_normalized, result_dict)
        return ToolResult.ok(
            confirmation=confirmation,
            data=result_dict if isinstance(result_dict, dict) else {"raw": result_dict},
        )

    @staticmethod
    def _format_deploy_confirmation(action: str, payload: Any) -> str:
        """Build a deploy-action confirmation string from the legacy
        manager dict. Codex round 2 (#1117): generic "deploy X ok"
        was hiding the actual outcome (e.g. "No active deployment
        sessions") because the command renderer suppresses scalar
        data; this puts the meaningful text in confirmation.
        """
        if not isinstance(payload, dict):
            return f"deploy {action} ok"
        # Prefer the manager's own message when present.
        msg = payload.get("message")
        if isinstance(msg, str) and msg:
            return msg
        # Health check often returns {"healthy": bool, ...} with no message.
        if action in {"health", "check"}:
            healthy = payload.get("healthy")
            if healthy is True:
                return "Service is healthy"
            if healthy is False:
                return f"Service is unhealthy: {payload.get('reason') or 'no detail'}"
        # Status with sessions but no message.
        if action == "status" and "active_deployments" in payload:
            n = payload.get("active_deployments", 0)
            return f"{n} active deployment(s)"
        # List of deployments. The manager returns
        # {"count": N, "deployments": [...]} on this path; older
        # surfaces also used "sessions". Read both.
        if action in {"list", "ls"}:
            if "count" in payload:
                return f"Listed {payload['count']} deployment(s)"
            if "deployments" in payload:
                return f"Listed {len(payload.get('deployments') or [])} deployment(s)"
            if "sessions" in payload:
                return f"Listed {len(payload.get('sessions') or [])} deployment(s)"
        return f"deploy {action} ok"

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
