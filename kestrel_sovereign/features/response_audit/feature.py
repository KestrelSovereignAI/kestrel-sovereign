"""Optional per-response LLM integrity audit feature."""
import logging
import os
from typing import List
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.hooks.base import Hook
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)


class ResponseAuditFeature(Feature):
    """Per-response LLM audit. Disabled by default (mode=skip)."""

    def __init__(self, agent):
        super().__init__(agent)
        self._hook = None
        self._mode = os.environ.get("KESTREL_RESPONSE_AUDIT_MODE", "skip")
        self._strategy = os.environ.get("KESTREL_RESPONSE_AUDIT_STRATEGY", "post")
        self._risk_threshold = int(os.environ.get("KESTREL_RESPONSE_AUDIT_RISK_THRESHOLD", "3"))

    @property
    def tool_description(self) -> str:
        return "Per-response LLM integrity audit with configurable modes"

    async def initialize(self):
        if self._mode == "skip":
            logger.info("ResponseAuditFeature: mode=skip, audit hook NOT registered")
            return

        self._create_hook(self._mode)

    def _create_hook(self, mode: str):
        """Create the audit hook instance (registration handled by lifecycle)."""
        from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook
        self._hook = ResponseAuditHook(
            agent=self.agent,
            mode=mode,
            strategy=self._strategy,
            risk_threshold=self._risk_threshold,
        )
        self._mode = mode
        logger.info(f"ResponseAuditHook created: mode={mode}, strategy={self._strategy}")

    def get_hooks(self) -> List[Hook]:
        """Return the audit hook for auto-registration (if active)."""
        if self._hook:
            return [self._hook]
        return []

    async def _register_hook(self, mode: str):
        """Dynamic hook registration for runtime enable (after initial lifecycle)."""
        self._create_hook(mode)
        if hasattr(self.agent, "hooks_manager") and self.agent.hooks_manager:
            self.agent.hooks_manager.register(self._hook)
            logger.info(f"ResponseAuditHook dynamically registered: mode={mode}")

    @tool("audit_enable", "Enable per-response audit", category=ToolCategory.SYSTEM, command_prefix="!audit-on")
    async def enable_audit(self, mode: str = "warn") -> ToolResult:
        """Enable response auditing.

        Args:
            mode: Audit mode - 'warn' (annotate risky responses) or 'strict' (block risky responses)

        Returns:
            ToolResult.ok on a successful enable or mode-update;
            ToolResult.failed when ``mode`` is not one of the
            supported values.
        """
        if mode not in ("warn", "strict"):
            return ToolResult.failed(
                error=f"Mode must be 'warn' or 'strict' (got {mode!r})"
            )

        if self._hook and self._hook.enabled:
            self._hook.mode = mode
            self._mode = mode
            return ToolResult.ok(
                f"Response audit mode updated to '{mode}'.",
                data={"status": "updated", "mode": mode},
            )

        await self._register_hook(mode)
        return ToolResult.ok(
            f"Response audit enabled (mode={mode}, strategy={self._strategy}, risk_threshold={self._risk_threshold}).",
            data={
                "status": "enabled",
                "mode": mode,
                "strategy": self._strategy,
                "risk_threshold": self._risk_threshold,
            },
        )

    @tool("audit_disable", "Disable per-response audit", category=ToolCategory.SYSTEM, command_prefix="!audit-off")
    async def disable_audit(self) -> ToolResult:
        """Disable response auditing."""
        was_enabled = bool(self._hook and self._hook.enabled)
        if self._hook:
            self._hook.enabled = False
        self._mode = "skip"
        confirmation = (
            "Response audit disabled."
            if was_enabled
            else "Response audit was already disabled."
        )
        return ToolResult.ok(confirmation, data={"status": "disabled"})

    @tool("audit_status", "Show audit configuration and status", category=ToolCategory.SYSTEM, command_prefix="!audit")
    async def audit_status(self) -> ToolResult:
        """Show current audit mode, strategy, and recent results.

        Returns OK with the full status dict in ``data``. PARTIAL when
        the configured mode is not 'skip' but the hook is not actually
        registered/enabled — that's a misconfiguration the LLM should
        speak so the sovereign doesn't believe audits are running when
        they are not.
        """
        hook_registered = self._hook is not None and self._hook.enabled
        data = {
            "mode": self._mode,
            "strategy": self._strategy,
            "risk_threshold": self._risk_threshold,
            "hook_registered": hook_registered,
            "audit_count": getattr(self._hook, "audit_count", 0) if self._hook else 0,
            "last_risk_level": getattr(self._hook, "last_risk_level", None) if self._hook else None,
        }
        confirmation = (
            f"Response audit: mode={self._mode}, strategy={self._strategy}, "
            f"hook_registered={hook_registered}, audit_count={data['audit_count']}."
        )
        if self._mode != "skip" and not hook_registered:
            return ToolResult.partial(
                confirmation,
                (
                    f"audit mode is '{self._mode}' but no hook is registered/enabled "
                    "— audits are NOT actually running; re-enable with !audit-on or "
                    "check why the hook failed to register."
                ),
                data=data,
            )
        return ToolResult.ok(confirmation, data=data)
