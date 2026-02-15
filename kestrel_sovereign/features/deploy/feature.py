"""
Deploy Feature for Kestrel agents.

Exposes agent self-deployment orchestration via the tool system,
providing commands for deploying to cloud platforms like Cloud Run.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.deploy.manager import DeployManager
from kestrel_sovereign.features.deploy.models import (
    DeployManagerError,
    DeployStatus,
)
from kestrel_sovereign.tools.base import ToolCategory

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

            session_data = []
            for service_name, session in sessions.items():
                session_data.append(session.to_dict())

            return {
                "success": True,
                "active_deployments": len(sessions),
                "sessions": session_data,
            }

        except DeployManagerError as e:
            return {"success": False, "error": str(e)}

    async def _deploy(self, profile_name: str, image_tag: str = "latest") -> Dict[str, Any]:
        """Deploy agent to cloud platform."""
        if not profile_name:
            profiles = list(self.manager.profiles.keys())
            return {
                "success": False,
                "error": "Profile required",
                "available_profiles": profiles,
                "usage": "!deploy deploy profile=dev",
            }

        try:
            # Get profile
            profile = self.manager.get_profile(profile_name)

            # Build image reference
            image = self._build_image_reference(profile_name, image_tag)

            # Check if session already exists
            existing_session = await self.manager.get_session(profile.service_name)
            if existing_session:
                return {
                    "success": False,
                    "error": f"Service {profile.service_name} already deployed",
                    "session": existing_session.to_dict(),
                    "hint": f"Use !deploy teardown profile={profile_name} first",
                }

            # Create deployment session
            session = await self.manager.get_session(profile.service_name)
            if not session:
                from kestrel_sovereign.features.deploy.models import DeploymentSession

                session = DeploymentSession(
                    service_name=profile.service_name,
                    provider=profile.provider,
                    profile=profile,
                    status=DeployStatus.DEPLOYING,
                    started_at=datetime.now(timezone.utc),
                )
                await self.manager.add_session(session)

            # Get provider and deploy
            provider = self.manager._get_provider(profile.provider)

            logger.info(f"Deploying {image} to {profile.service_name}...")
            session.status = DeployStatus.DEPLOYING

            deploy_result = await provider.deploy(
                image=image,
                service_name=profile.service_name,
                profile=profile,
            )

            # Update session
            session.status = DeployStatus.ACTIVE
            session.service_url = deploy_result.get("service_url")
            session.revision = deploy_result.get("revision")
            session.last_updated = datetime.now(timezone.utc)

            # Verify health
            if session.service_url:
                logger.info(f"Verifying health of {session.service_url}...")
                healthy = await self.manager._verify_health(session.service_url)
                session.health_status = "healthy" if healthy else "unknown"

            logger.info(f"Deployment complete: {session.service_url}")

            return {
                "success": True,
                "action": "deploy",
                "session": session.to_dict(),
            }

        except DeployManagerError as e:
            # Clean up session on failure
            if profile_name:
                try:
                    profile = self.manager.get_profile(profile_name)
                    await self.manager.remove_session(profile.service_name)
                except Exception:
                    pass

            return {"success": False, "error": str(e)}

    async def _teardown(self, profile_name: str) -> Dict[str, Any]:
        """Teardown deployed service."""
        if not profile_name:
            return {
                "success": False,
                "error": "Profile required",
                "usage": "!deploy teardown profile=dev",
            }

        try:
            # Get profile
            profile = self.manager.get_profile(profile_name)

            # Get provider
            provider = self.manager._get_provider(profile.provider)

            # Teardown service
            logger.info(f"Tearing down service {profile.service_name}...")
            result = await provider.teardown(profile.service_name)

            # Remove session
            await self.manager.remove_session(profile.service_name)

            return {
                "success": True,
                "action": "teardown",
                "service": profile.service_name,
                "result": result,
            }

        except DeployManagerError as e:
            return {"success": False, "error": str(e)}

    async def _logs(self, profile_name: str, lines: int = 100) -> Dict[str, Any]:
        """Get deployment logs."""
        if not profile_name:
            return {
                "success": False,
                "error": "Profile required",
                "usage": "!deploy logs profile=dev",
            }

        try:
            # Get profile
            profile = self.manager.get_profile(profile_name)

            # Get provider
            provider = self.manager._get_provider(profile.provider)

            # Get logs
            logs = await provider.get_logs(profile.service_name, lines=lines)

            return {
                "success": True,
                "action": "logs",
                "service": profile.service_name,
                "lines": lines,
                "logs": logs,
            }

        except DeployManagerError as e:
            return {"success": False, "error": str(e)}

    async def _list_deployments(self) -> Dict[str, Any]:
        """List all deployments across all providers."""
        try:
            all_deployments = []

            # Query each provider type that has profiles
            provider_types = set(p.provider for p in self.manager.profiles.values())

            for provider_type in provider_types:
                try:
                    provider = self.manager._get_provider(provider_type)
                    deployments = await provider.list_deployments()
                    all_deployments.extend(deployments)
                except Exception as e:
                    logger.warning(f"Failed to list {provider_type.value} deployments: {e}")

            return {
                "success": True,
                "action": "list",
                "count": len(all_deployments),
                "deployments": all_deployments,
            }

        except DeployManagerError as e:
            return {"success": False, "error": str(e)}

    async def _health_check(self, profile_name: str) -> Dict[str, Any]:
        """Check health of deployed service."""
        if not profile_name:
            return {
                "success": False,
                "error": "Profile required",
                "usage": "!deploy health profile=dev",
            }

        try:
            # Get profile
            profile = self.manager.get_profile(profile_name)

            # Get session
            session = await self.manager.get_session(profile.service_name)
            if not session or not session.service_url:
                # No session, try to get status from provider
                provider = self.manager._get_provider(profile.provider)
                status = await provider.get_status(profile.service_name)

                if status.get("status") == "offline":
                    return {
                        "success": True,
                        "action": "health",
                        "service": profile.service_name,
                        "status": "offline",
                        "message": "Service not deployed",
                    }

                service_url = status.get("service_url")
            else:
                service_url = session.service_url

            if not service_url:
                return {
                    "success": False,
                    "error": "Service URL not available",
                }

            # Get provider and check health
            provider = self.manager._get_provider(profile.provider)
            health_result = await provider.health_check(service_url)

            return {
                "success": True,
                "action": "health",
                "service": profile.service_name,
                "url": service_url,
                "health": health_result,
            }

        except DeployManagerError as e:
            return {"success": False, "error": str(e)}

    def _build_image_reference(self, profile_name: str, tag: str = "latest") -> str:
        """
        Build container image reference.

        Args:
            profile_name: Name of the profile
            tag: Image tag

        Returns:
            Full image reference (e.g., gcr.io/project/kestrel:latest)
        """
        # For Cloud Run, use GCR
        if self.manager.gcp_project_id:
            image_name = self.manager.image_name
            return f"gcr.io/{self.manager.gcp_project_id}/{image_name}:{tag}"

        # Fallback to generic reference
        return f"{self.manager.image_name}:{tag}"
