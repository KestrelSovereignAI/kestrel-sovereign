"""Response audit hook - evaluates LLM responses for integrity."""
import logging
from kestrel_sovereign.hooks.base import Hook, HookEvent, HookInput, HookOutput

logger = logging.getLogger(__name__)


class ResponseAuditHook(Hook):
    """Hook that audits LLM responses using a separate LLM call."""

    def __init__(self, agent, mode: str = "warn", strategy: str = "post", risk_threshold: int = 3):
        super().__init__(
            name="response_audit",
            events=[HookEvent.POST_RESPONSE],
            priority=50,
            timeout=30.0,
        )
        self.agent = agent
        self.mode = mode
        self.strategy = strategy
        self.risk_threshold = risk_threshold
        self.audit_count = 0
        self.last_risk_level = None

    async def execute(self, input: HookInput) -> HookOutput:
        response_text = input.response_text
        if not response_text or len(response_text.strip()) < 20:
            return HookOutput.allow("Response too short to audit")

        try:
            audit_result = await self.agent.llm_service.get_audit_response(response_text)
        except Exception as e:
            logger.warning(f"Response audit failed: {e}")
            return HookOutput.allow(f"Audit skipped due to error: {e}")

        risk_level = audit_result.get("risk_level", 1)
        reasoning = audit_result.get("reasoning", "")
        self.audit_count += 1
        self.last_risk_level = risk_level

        logger.info(f"Response audit: risk_level={risk_level}, reasoning={reasoning}")

        # Notify audit anchor feature for tamper-proof logging
        await self._notify_audit_anchor(risk_level, reasoning)

        if risk_level >= self.risk_threshold:
            if self.mode == "strict":
                return HookOutput.deny(f"Audit risk level {risk_level}: {reasoning}")
            elif self.mode == "warn":
                warning = f"\n\n---\n[Audit warning (risk {risk_level}): {reasoning}]"
                return HookOutput.modify(
                    updated_input={"response_text": response_text + warning},
                    reason=f"Audit warning: {reasoning}",
                )

        return HookOutput.allow(f"Audit passed (risk {risk_level})")

    async def _notify_audit_anchor(self, risk_level: int, reasoning: str):
        """Notify AuditAnchorFeature for tamper-proof logging on elevated risk."""
        if risk_level < 2:
            return
        try:
            features = getattr(self.agent, "features", {})
            if isinstance(features, dict):
                anchor = features.get("AuditAnchorFeature")
            elif isinstance(features, list):
                anchor = next((f for f in features if type(f).__name__ == "AuditAnchorFeature"), None)
            else:
                anchor = None
            if anchor and hasattr(anchor, "on_audit_complete"):
                await anchor.on_audit_complete({
                    "is_valid": risk_level < self.risk_threshold,
                    "message": reasoning,
                    "source": "response_audit",
                })
        except Exception as e:
            logger.debug(f"Audit anchor notification failed: {e}")
