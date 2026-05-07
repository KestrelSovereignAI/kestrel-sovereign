"""Response audit hook - evaluates LLM responses for integrity."""
import logging
from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
from kestrel_sovereign.security.narration_check import analyze_narration

logger = logging.getLogger(__name__)


class ResponseAuditHook(Hook):
    """Hook that audits LLM responses for integrity.

    Combines two signals:

    1. **Deterministic narration check** (#1042 layer 3) — pure-Python
       analysis of ``HookInput.pre_tool_prose`` against
       ``HookInput.tool_results`` (SDK 0.9 fields populated at the
       ToolCallStarted marker boundary by ``agent/streaming.py``).
       Same inputs always yield the same verdict; suitable for
       compliance-gated deployments.
    2. **LLM audit call** — ``llm_service.get_audit_response`` runs a
       separate model judgment over the assembled ``response_text``.

    The two signals are additive: a narration violation elevates risk
    even when the LLM audit is unavailable, so the deterministic check
    keeps firing under partial-outage conditions.
    """

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
        self.last_narration_verdict = None

    async def execute(self, input: HookInput) -> HookOutput:
        response_text = input.response_text or ""

        # Run the deterministic narration check FIRST so even short
        # responses get audited against the marker-boundary signal.
        # Codex review of #1076: previously the 20-char short-circuit
        # ran before this and a canonical violation like
        # "Saved." (6 chars) + failed tool slipped through.
        narration_verdict = analyze_narration(
            input.pre_tool_prose,
            input.tool_results,
        )
        self.last_narration_verdict = narration_verdict

        # Honesty doctrine: a narration violation is a constitutional
        # failure independent of the LLM-audit risk score, so it
        # always crosses the configured threshold. Without this floor
        # a default ``risk_threshold=3`` deployment would let a
        # pure-narration violation (boost=2) pass when the LLM audit
        # is unavailable. Codex P2 #1076.
        narration_risk = (
            max(narration_verdict.risk_boost, self.risk_threshold)
            if narration_verdict.risk_boost > 0
            else 0
        )

        if not response_text or len(response_text.strip()) < 20:
            # Short responses skip the LLM audit (it has no signal to
            # work with), but they still honor a deterministic
            # narration violation.
            if narration_risk > 0:
                self.audit_count += 1
                self.last_risk_level = narration_risk
                await self._notify_audit_anchor(
                    narration_risk, narration_verdict.reasoning,
                )
                return self._apply_audit_decision(
                    response_text=response_text,
                    risk_level=narration_risk,
                    reasoning=narration_verdict.reasoning,
                )
            return HookOutput.allow("Response too short to audit")

        try:
            audit_result = await self.agent.llm_service.get_audit_response(response_text)
        except Exception as e:
            logger.warning(f"Response audit failed: {e}")
            # Even with the LLM audit unavailable, a clean-cut
            # narration violation still fires the audit machinery.
            if narration_risk > 0:
                self.audit_count += 1
                self.last_risk_level = narration_risk
                await self._notify_audit_anchor(
                    narration_risk, narration_verdict.reasoning,
                )
                return self._apply_audit_decision(
                    response_text=response_text,
                    risk_level=narration_risk,
                    reasoning=narration_verdict.reasoning,
                )
            return HookOutput.allow(f"Audit skipped due to error: {e}")

        risk_level = audit_result.get("risk_level", 1)
        reasoning = audit_result.get("reasoning", "")

        # Fold the narration verdict into the LLM audit score:
        # additive (so an LLM-flagged response with a narration
        # violation reads even higher), then floored at the configured
        # threshold so a narration violation always trips the gate
        # regardless of how the LLM audit scored.
        if narration_verdict.risk_boost > 0:
            risk_level = max(risk_level, 0) + narration_verdict.risk_boost
            risk_level = max(risk_level, self.risk_threshold)
            reasoning = (
                f"{reasoning} | narration_check: {narration_verdict.reasoning}"
                if reasoning
                else narration_verdict.reasoning
            )

        self.audit_count += 1
        self.last_risk_level = risk_level

        logger.info(f"Response audit: risk_level={risk_level}, reasoning={reasoning}")

        # Notify audit anchor feature for tamper-proof logging
        await self._notify_audit_anchor(risk_level, reasoning)

        return self._apply_audit_decision(
            response_text=response_text,
            risk_level=risk_level,
            reasoning=reasoning,
        )

    def _apply_audit_decision(
        self, response_text: str, risk_level: int, reasoning: str,
    ) -> HookOutput:
        """Translate ``(risk_level, reasoning)`` into the configured
        mode's ``HookOutput``. Extracted from ``execute`` so the
        LLM-audit-failed branch and the LLM-audit-OK branch share one
        threshold/mode policy — previously they diverged silently.
        """
        if risk_level >= self.risk_threshold:
            if self.mode == "strict":
                return HookOutput.deny(f"Audit risk level {risk_level}: {reasoning}")
            if self.mode == "warn":
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
