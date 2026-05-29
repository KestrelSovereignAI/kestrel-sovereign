"""
Deploy Core Manager Operations.

Contains the core DeployManagerCore class with config loading,
profile management, provider registry, health verification, and
the orchestration logic shared between the agent-tool surface
(``DeployFeature``) and the operator CLI surface (``kestrel deploy``).

The ``deploy_profile``/``teardown_profile``/``get_profile_logs``/
``list_all_deployments``/``health_check_profile`` methods own the
"talk to a provider, manage sessions, handle errors" workflow so
both surfaces (`!deploy <action>` and `kestrel deploy <profile>`)
delegate to the same code path.
"""

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kestrel_sovereign.config import load_config

from .models import (
    DeployManagerError,
    DeploymentProfile,
    DeploymentSession,
    DeployProviderType,
    DeployStatus,
)
from .providers.azure_container import AzureContainerProvider
from .providers.base import DeployProvider
from .providers.cloudrun import CloudRunProvider

logger = logging.getLogger(__name__)


class DeployManagerCore:
    """
    Core deployment operations.

    Handles config loading, profile management, provider instantiation,
    and health verification for agent self-deployment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize deploy manager with config.

        Args:
            config: Optional config dict (defaults to loading deploy_config.toml)
        """
        self.config = config or load_config("deploy_config.toml")
        self.manager_config = self.config.get("manager", {})

        # Manager settings
        self.default_provider = self.manager_config.get("default_provider", "cloudrun")
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID") or self.manager_config.get("gcp_project_id")
        self.image_name = self.manager_config.get("image_name", "kestrel")
        self.build_strategy = self.manager_config.get("build_strategy", "prebuilt")
        self.health_check_timeout = int(
            self.manager_config.get("health_check_timeout_seconds", 120)
        )
        self.health_check_path = self.manager_config.get("health_check_path", "/health")

        # Load profiles from config
        self.profiles = self._load_profiles(self.config.get("profiles", {}))

        # Provider registry (lazy-loaded)
        # Cache key is ``(provider_type, gcp_project_id)`` so profiles
        # with different ``gcp_project_id`` values each get their own
        # CloudRunProvider — not a shared instance bound to whatever
        # deploy ran first. Azure providers ignore the project_id
        # component (always None for them).
        self._providers: Dict[
            tuple, DeployProvider
        ] = {}

        # Multi-session tracking (agents may have dev+prod simultaneously)
        self._sessions: Dict[str, DeploymentSession] = {}

        # Lock for session management
        self._lock = asyncio.Lock()

    def _load_profiles(self, raw_profiles: Dict[str, Any]) -> Dict[str, DeploymentProfile]:
        """
        Load and parse deployment profiles from config.

        Args:
            raw_profiles: Raw profile dict from TOML

        Returns:
            Dict mapping profile name to DeploymentProfile
        """
        profiles: Dict[str, DeploymentProfile] = {}

        for key, data in raw_profiles.items():
            try:
                # Parse provider type
                provider_str = data.get("provider", "cloudrun").lower()
                if provider_str == "cloudrun" or provider_str == "cloud_run":
                    provider = DeployProviderType.CLOUD_RUN
                elif provider_str == "azure" or provider_str == "azure_container_apps":
                    provider = DeployProviderType.AZURE_CONTAINER_APPS
                else:
                    logger.warning(f"Unknown provider '{provider_str}' in profile '{key}', skipping")
                    continue

                # Expand environment variables in env_vars and secrets
                raw_env = data.get("env_vars", {})
                expanded_env = self._expand_env_vars(raw_env)

                raw_secrets = data.get("secrets", {})
                expanded_secrets = self._expand_env_vars(raw_secrets)

                # Determine default dockerfile based on deployment mode
                deployment_mode = data.get("deployment_mode", "agent")
                default_dockerfile = (
                    "docker/Dockerfile.multi_agent"
                    if deployment_mode == "multi_agent"
                    else "docker/Dockerfile.cloudrun"
                )

                profiles[key] = DeploymentProfile(
                    provider=provider,
                    service_name=data["service_name"],
                    region=data["region"],
                    min_instances=int(data.get("min_instances", 0)),
                    max_instances=int(data.get("max_instances", 10)),
                    memory=data.get("memory", "2Gi"),
                    cpu=int(data.get("cpu", 2)),
                    port=int(data.get("port", 8080)),
                    timeout=int(data.get("timeout", 300)),
                    concurrency=int(data.get("concurrency", 80)),
                    deployment_mode=deployment_mode,
                    dockerfile=data.get("dockerfile", default_dockerfile),
                    env_vars=expanded_env,
                    secrets=expanded_secrets,
                    gcp_project_id=data.get("gcp_project_id") or self.gcp_project_id,
                    azure_resource_group=data.get("azure_resource_group"),
                )

                logger.debug(f"Loaded profile '{key}': {provider.value} -> {data['service_name']}")

            except KeyError as exc:
                logger.warning(f"Incomplete profile '{key}': missing {exc}")
                continue

        return profiles

    def _validate_no_unresolved_placeholders(
        self, profile_name: str, profile: DeploymentProfile
    ) -> None:
        """Raise DeployManagerError if any expanded value still
        contains ``${VAR}`` (env var unset) OR if the original config
        used ``${VAR}`` syntax and the expanded value is now empty
        (env var set but blank). The bash predecessors used
        ``${VAR:?...}`` which errored on EITHER condition; we mirror
        that so missing/empty secrets like ``KESTREL_ALLOWED_EMAILS``
        don't silently produce an OAuth-enabled service that locks
        everyone out. Codex review on the final epic→main PR caught
        the empty-string gap.
        """
        # Load the raw (pre-expansion) profile config so we can
        # distinguish "this value was a ${VAR} substitution" from
        # "this value was literally empty in the TOML".
        raw_profile = (self.config.get("profiles", {}) or {}).get(profile_name, {}) or {}
        raw_env = raw_profile.get("env_vars", {}) or {}
        raw_secrets = raw_profile.get("secrets", {}) or {}

        unresolved = []
        empty_after_expansion = []

        def _check(section: str, raw_dict, expanded_dict):
            for key, expanded in (expanded_dict or {}).items():
                if not isinstance(expanded, str):
                    continue
                if "${" in expanded:
                    unresolved.append((section, key, expanded))
                    continue
                # Empty after expansion + raw used ${...} → bash's
                # ``${VAR:?...}`` would have errored.
                raw = raw_dict.get(key)
                if (
                    isinstance(raw, str)
                    and "${" in raw
                    and expanded == ""
                ):
                    empty_after_expansion.append((section, key, raw))

        _check("env_vars", raw_env, profile.env_vars)
        _check("secrets", raw_secrets, profile.secrets)

        if unresolved or empty_after_expansion:
            details_unresolved = "; ".join(
                f"{section}.{key}={value!r}" for section, key, value in unresolved
            )
            details_empty = "; ".join(
                f"{section}.{key}={raw!r} (env var set but empty)"
                for section, key, raw in empty_after_expansion
            )
            details = "; ".join(d for d in (details_unresolved, details_empty) if d)
            missing_vars = sorted({
                m.group(1)
                for _, _, value in (unresolved + empty_after_expansion)
                for m in re.finditer(r'\$\{([^}]+)\}', value)
            })
            raise DeployManagerError(
                f"profile '{profile_name}' has unresolved or empty ${{...}} "
                f"placeholders (env vars: {', '.join(missing_vars)}). "
                f"Export them with non-empty values before running "
                f"`kestrel deploy {profile_name}`. Affected: {details}"
            )

    # Tags that point at moving aliases — deploying these silently
    # no-ops on Cloud Run because the template comparison is by string,
    # not by resolved digest. See #1441.
    _MOVING_ALIAS_TAGS = frozenset({"", "latest"})

    def _reject_moving_alias_tag(self, profile_name: str, tag: str) -> None:
        """Raise DeployManagerError if ``tag`` is a moving alias."""
        if tag in self._MOVING_ALIAS_TAGS:
            image_ref_no_tag = self.build_image_reference(profile_name, "").rstrip(":")
            raise DeployManagerError(
                f"Refusing to deploy '{profile_name}': image tag is "
                f"{tag!r}. Cloud Run treats moving aliases as stable "
                f"strings and won't roll a new revision when the "
                f"underlying digest changes (#1441). Pass a concrete "
                f"tag such as `--tag v0.15.1` or `--tag dev-abc1234`. "
                f"In CI, pass the build's resolved tag via "
                f"`--tag ${{ needs.build.outputs.tag }}`. "
                f"List recent tags with: gcloud container images "
                f"list-tags {image_ref_no_tag} --limit 5"
            )

    @staticmethod
    def _expand_env_vars(env_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Expand ${VAR} syntax in environment variable values.

        Args:
            env_dict: Dict with potentially unexpanded values

        Returns:
            Dict with expanded values
        """
        expanded = {}
        for key, value in env_dict.items():
            if isinstance(value, str) and "${" in value:
                # Expand ${VAR_NAME} patterns
                def replace_var(match):
                    var_name = match.group(1)
                    return os.environ.get(var_name, f"${{{var_name}}}")
                expanded[key] = re.sub(r'\$\{([^}]+)\}', replace_var, value)
            else:
                expanded[key] = value
        return expanded

    def _get_provider(
        self,
        provider_type: DeployProviderType,
        *,
        gcp_project_id: Optional[str] = None,
    ) -> DeployProvider:
        """
        Get or create a deployment provider instance.

        Args:
            provider_type: Type of provider to get.
            gcp_project_id: For ``CLOUD_RUN``, the GCP project the
                provider should target. Falls back to
                ``self.gcp_project_id`` when not given (legacy callers,
                Azure providers — which ignore it). The cache is keyed
                by ``(provider_type, gcp_project_id)`` so two profiles
                that target different GCP projects each get their own
                CloudRunProvider, not a single one bound to whichever
                deploy ran first. Codex review on the final epic→main
                PR caught the cache-shared-state bug.

        Returns:
            DeployProvider instance
        """
        cache_key = (provider_type, gcp_project_id)

        # Check cache first
        if cache_key in self._providers:
            return self._providers[cache_key]

        # Create new provider
        if provider_type == DeployProviderType.CLOUD_RUN:
            project = gcp_project_id or self.gcp_project_id
            provider = CloudRunProvider(project_id=project)
        elif provider_type == DeployProviderType.AZURE_CONTAINER_APPS:
            provider = AzureContainerProvider()
        else:
            raise DeployManagerError(f"Unknown provider type: {provider_type}")

        # Cache for reuse
        self._providers[cache_key] = provider
        logger.debug(f"Created {provider_type.value} provider")

        return provider

    async def _verify_health(
        self,
        service_url: str,
        timeout: Optional[int] = None,
        poll_interval: int = 5,
    ) -> bool:
        """
        Verify health of deployed service with exponential backoff.

        Polls the health endpoint until it responds or timeout is reached.
        Pattern from gcp_compute/core.py:743-793.

        Args:
            service_url: Base URL of the service
            timeout: Max time to wait in seconds (defaults to health_check_timeout)
            poll_interval: Initial poll interval in seconds

        Returns:
            True if healthy, False if timeout
        """
        import httpx

        timeout = timeout or self.health_check_timeout
        health_url = f"{service_url.rstrip('/')}{self.health_check_path}"

        start_time = time.time()
        current_interval = poll_interval

        logger.info(f"Verifying health at {health_url} (timeout: {timeout}s)")

        while (time.time() - start_time) < timeout:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(health_url)

                    if 200 <= response.status_code < 400:
                        logger.info(f"Service is healthy (status: {response.status_code})")
                        return True

                    logger.debug(f"Service returned {response.status_code}, retrying...")

            except (OSError, ConnectionError) as e:
                logger.debug(f"Service not ready yet: {e}")
            except Exception as e:
                logger.debug(f"Unexpected error checking health: {e}")

            # Wait with exponential backoff (max 30s)
            await asyncio.sleep(min(current_interval, 30))
            current_interval *= 1.5

        logger.warning(f"Health check timed out after {timeout}s")
        return False

    def get_profile(self, profile_name: str) -> DeploymentProfile:
        """
        Get a deployment profile by name.

        Args:
            profile_name: Name of the profile

        Returns:
            DeploymentProfile

        Raises:
            DeployManagerError: If profile not found
        """
        profile = self.profiles.get(profile_name)
        if not profile:
            available = ", ".join(self.profiles.keys())
            raise DeployManagerError(
                f"Unknown profile '{profile_name}'. Available: {available}"
            )
        return profile

    async def get_session(self, service_name: str) -> Optional[DeploymentSession]:
        """
        Get deployment session by service name.

        Args:
            service_name: Name of the service

        Returns:
            DeploymentSession if exists, None otherwise
        """
        async with self._lock:
            return self._sessions.get(service_name)

    async def add_session(self, session: DeploymentSession) -> None:
        """
        Add a deployment session.

        Args:
            session: DeploymentSession to add
        """
        async with self._lock:
            self._sessions[session.service_name] = session
            logger.debug(f"Added session for {session.service_name}")

    async def remove_session(self, service_name: str) -> None:
        """
        Remove a deployment session.

        Args:
            service_name: Name of the service
        """
        async with self._lock:
            if service_name in self._sessions:
                del self._sessions[service_name]
                logger.debug(f"Removed session for {service_name}")

    async def list_sessions(self) -> Dict[str, DeploymentSession]:
        """
        List all active deployment sessions.

        Returns:
            Dict mapping service name to DeploymentSession
        """
        async with self._lock:
            return dict(self._sessions)

    # ------------------------------------------------------------------
    # Orchestration: shared by DeployFeature (`!deploy`) and the
    # `kestrel deploy` CLI. These methods own provider dispatch, session
    # bookkeeping, and DeployManagerError handling. Both surfaces wrap
    # them with surface-specific guards (CLI prints to stderr, the
    # feature returns structured ``error`` dicts).
    # ------------------------------------------------------------------

    def build_image_reference(self, profile_name: str, tag: str = "latest") -> str:
        """
        Build container image reference for a profile.

        The image name is derived from the profile's ``deployment_mode``:

        * ``deployment_mode = "agent"`` (default) → ``<image_name>``
          (e.g. ``kestrel``) — built from ``docker/Dockerfile.cloudrun``.
        * ``deployment_mode = "multi_agent"`` → ``<image_name>-multi_agent``
          (e.g. ``kestrel-multi_agent``) — built from
          ``docker/Dockerfile.multi_agent``.

        These names match :data:`kestrel_sovereign.features.deploy.build.DEFAULT_TARGETS`
        and ``.github/workflows/deploy.yml`` so ``kestrel deploy build`` and
        ``kestrel deploy <profile>`` always reference the same registry refs.
        The legacy bash ``deploy_dev.sh``/``deploy_multi_agent_dev.sh`` used
        a hyphenated ``kestrel-multi-agent`` orphan that no build path ever
        produced; epic #1050 sub-PR 1.4 reconciled on the underscore form.

        Args:
            profile_name: Profile to look up. Unknown profiles fall back
                to the manager-level ``image_name`` so callers exercising
                the manager outside a profile context (legacy tests) still
                get a sensible ref.
            tag: Image tag (default ``latest``).

        Returns:
            Full image reference (e.g. ``gcr.io/project/kestrel-multi_agent:v1.2.3``).
        """
        # Resolve the image name from the profile's deployment_mode. We
        # don't raise on an unknown profile because some legacy callers
        # build a manager from a stripped-down config; they'll get the
        # global single-agent name, which is the historical default.
        image_name = self.image_name
        profile = self.profiles.get(profile_name)
        if profile is not None and profile.is_multi_agent:
            image_name = f"{self.image_name}-multi_agent"

        # Profile-scoped ``gcp_project_id`` wins over the manager-level
        # value — DeployManagerCore._load_profiles populates
        # ``profile.gcp_project_id`` with ``data.get("gcp_project_id")
        # or self.gcp_project_id``, so it's already the resolved
        # effective project. Without this, a config that legitimately
        # sets ``[profiles.prod].gcp_project_id`` would build
        # ``gcr.io/<manager-project>/...`` instead of the profile's.
        # Codex review on the final epic→main PR.
        gcp_project = (
            (profile.gcp_project_id if profile is not None else None)
            or self.gcp_project_id
        )

        # For Cloud Run, use GCR
        if gcp_project:
            return f"gcr.io/{gcp_project}/{image_name}:{tag}"

        # Fallback to generic reference
        return f"{image_name}:{tag}"

    async def deploy_profile(
        self, profile_name: str, tag: str = "latest"
    ) -> Dict[str, Any]:
        """
        Deploy an agent to the cloud platform configured by ``profile_name``.

        Returns the same shape DeployFeature historically returned:
            ``{"success": True, "action": "deploy", "session": {...}}``
        on success, ``{"success": False, "error": "..."}`` on failure.
        """
        try:
            profile = self.get_profile(profile_name)

            # Validate that every ``${VAR}`` placeholder in the profile's
            # env_vars / secrets actually resolved against runtime env.
            # ``_expand_env_vars`` returns the literal ``${VAR}`` when a
            # variable is unset — pushing that to Cloud Run silently
            # produces broken config (e.g. an OAuth allowlist with the
            # literal string ``${KESTREL_ALLOWED_EMAILS}``). The bash
            # scripts errored on missing env via ``${VAR:?...}``;
            # mirror that here. Codex review on PR #1064.
            self._validate_no_unresolved_placeholders(profile_name, profile)

            # Refuse to deploy a moving-alias tag on Cloud Run. Admin v2
            # ``update_service`` compares the new template against the
            # existing one as strings; if both reference ``:latest``,
            # the underlying digest can change in the registry and the
            # service silently keeps serving the prior revision. Every
            # ``kestrel deploy`` invocation since the legacy bash scripts
            # were retired hit this (#1441) — the workflow looked green
            # but no new revision rolled out. Force callers to pass a
            # concrete tag (``v0.15.1``, ``dev-abc1234``) so each deploy
            # produces a unique image string. Other providers (Azure
            # Container Apps) are not known to share this bug and are
            # left at the prior default-``latest`` behavior; if they
            # turn out to no-op similarly, widen this guard then.
            if profile.provider == DeployProviderType.CLOUD_RUN:
                self._reject_moving_alias_tag(profile_name, tag)

            image = self.build_image_reference(profile_name, tag)

            # Check if session already exists
            existing_session = await self.get_session(profile.service_name)
            if existing_session:
                return {
                    "success": False,
                    "error": f"Service {profile.service_name} already deployed",
                    "session": existing_session.to_dict(),
                    "hint": f"Tear it down first (e.g. `kestrel deploy teardown {profile_name}`)",
                }

            # Create deployment session
            session = DeploymentSession(
                service_name=profile.service_name,
                provider=profile.provider,
                profile=profile,
                status=DeployStatus.DEPLOYING,
                started_at=datetime.now(timezone.utc),
            )
            await self.add_session(session)

            provider = self._get_provider(profile.provider, gcp_project_id=profile.gcp_project_id)

            logger.info(f"Deploying {image} to {profile.service_name}...")
            session.status = DeployStatus.DEPLOYING

            deploy_result = await provider.deploy(
                image=image,
                service_name=profile.service_name,
                profile=profile,
            )

            session.status = DeployStatus.ACTIVE
            session.service_url = deploy_result.get("service_url")
            session.revision = deploy_result.get("revision")
            session.last_updated = datetime.now(timezone.utc)

            if session.service_url:
                logger.info(f"Verifying health of {session.service_url}...")
                healthy = await self._verify_health(session.service_url)
                session.health_status = "healthy" if healthy else "unknown"

            logger.info(f"Deployment complete: {session.service_url}")

            return {
                "success": True,
                "action": "deploy",
                "session": session.to_dict(),
            }

        except DeployManagerError as e:
            # Best-effort session cleanup on failure so a retry can
            # re-create one. We swallow the inner exception because the
            # outer DeployManagerError is what the operator needs to see.
            try:
                profile = self.get_profile(profile_name)
                await self.remove_session(profile.service_name)
            except Exception:
                pass

            return {"success": False, "error": str(e)}

    async def teardown_profile(self, profile_name: str) -> Dict[str, Any]:
        """Delete a deployed service for ``profile_name``."""
        try:
            profile = self.get_profile(profile_name)
            provider = self._get_provider(profile.provider, gcp_project_id=profile.gcp_project_id)

            logger.info(f"Tearing down service {profile.service_name}...")
            result = await provider.teardown(profile.service_name)

            await self.remove_session(profile.service_name)

            return {
                "success": True,
                "action": "teardown",
                "service": profile.service_name,
                "result": result,
            }

        except DeployManagerError as e:
            return {"success": False, "error": str(e)}

    async def get_profile_logs(
        self, profile_name: str, lines: int = 100
    ) -> Dict[str, Any]:
        """Fetch the last ``lines`` log lines from the deployed service."""
        try:
            profile = self.get_profile(profile_name)
            provider = self._get_provider(profile.provider, gcp_project_id=profile.gcp_project_id)

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

    async def list_all_deployments(self) -> Dict[str, Any]:
        """List every deployment across every provider configured in profiles.

        Iterates the ``(provider, gcp_project_id)`` pairs from
        ``self.profiles`` (deduped) rather than just ``provider_type``,
        so configs with profile-scoped ``gcp_project_id`` overrides
        list deployments from each project. Without this, the manager
        would only ever talk to its own ``self.gcp_project_id``,
        omitting deployments that live in profile-specific GCP projects.
        Codex review on the final epic→main PR.
        """
        try:
            all_deployments = []

            # Collect unique (provider, project) pairs. Azure profiles
            # don't have a project_id; encode that as None.
            provider_targets = set()
            for prof in self.profiles.values():
                if prof.provider == DeployProviderType.CLOUD_RUN:
                    provider_targets.add((prof.provider, prof.gcp_project_id))
                else:
                    provider_targets.add((prof.provider, None))

            # Add the manager-level CloudRun fallback if not already
            # covered (e.g. configs with no profile-level overrides
            # still want to list against ``self.gcp_project_id``).
            if any(p == DeployProviderType.CLOUD_RUN for p, _ in provider_targets):
                provider_targets.add(
                    (DeployProviderType.CLOUD_RUN, self.gcp_project_id)
                )

            for provider_type, project in provider_targets:
                try:
                    provider = self._get_provider(
                        provider_type, gcp_project_id=project
                    )
                    deployments = await provider.list_deployments()
                    all_deployments.extend(deployments)
                except Exception as e:
                    logger.warning(
                        f"Failed to list {provider_type.value} deployments "
                        f"(project={project}): {e}"
                    )

            return {
                "success": True,
                "action": "list",
                "count": len(all_deployments),
                "deployments": all_deployments,
            }

        except DeployManagerError as e:
            return {"success": False, "error": str(e)}

    async def health_check_profile(self, profile_name: str) -> Dict[str, Any]:
        """Health-check the deployed service for ``profile_name``."""
        try:
            profile = self.get_profile(profile_name)

            session = await self.get_session(profile.service_name)
            if not session or not session.service_url:
                # No tracked session — query the provider directly so
                # `kestrel deploy health` works against services that
                # were deployed in a previous process.
                provider = self._get_provider(profile.provider, gcp_project_id=profile.gcp_project_id)
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

            provider = self._get_provider(profile.provider, gcp_project_id=profile.gcp_project_id)
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
