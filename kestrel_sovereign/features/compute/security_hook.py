"""
Kestrel Compute Feature - Security Hook.

Pre-check hook that runs before SecurityHook to analyze scripts
and gate critical security risks.
"""

import logging
from typing import Optional

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput

from .models import ComputeScript, DenialResponse, ScriptState, calculate_risk_score
from .script_store import ScriptStore
from .script_signer import ScriptSigner
from .script_analyzer import ScriptAnalyzer

logger = logging.getLogger(__name__)


class ComputeSecurityHook(Hook):
    """
    Pre-check hook for compute script execution.
    
    This hook runs BEFORE the standard SecurityHook and:
    1. Verifies script signature (reject if tampered)
    2. Analyzes script for dangerous patterns
    3. Updates script with security findings
    4. Auto-DENY critical risks (fork bombs, etc.)
    
    For non-critical scripts, this hook returns ALLOW and lets
    the standard SecurityHook handle the actual approval flow
    (once/session/always) via the ApprovalQueue.
    
    Hook Chain:
    1. ComputeSecurityHook (priority=5) - This hook
       - Verify signature
       - Analyze patterns
       - Auto-DENY critical only
       - ALLOW → continues to next hook
    
    2. SecurityHook (priority=10) - Standard security hook
       - Check PermissionLevel for ComputeFeature.run_script
       - Default: ASK → queue for approval
       - User chooses: once / session / always
       - Persist based on scope
    """
    
    def __init__(
        self,
        script_store: ScriptStore,
        signer: Optional[ScriptSigner] = None,
        priority: int = 5,  # Run BEFORE SecurityHook (priority=10)
    ):
        """
        Initialize the compute security hook.
        
        Args:
            script_store: ScriptStore for accessing scripts
            signer: Optional ScriptSigner for signature verification
            priority: Hook priority (lower = earlier)
        """
        super().__init__(
            name="compute_security",
            events=[HookEvent.PRE_TOOL_USE],
            matcher=r"^run_script$",  # Match run_script tool only
            priority=priority,
        )
        self.script_store = script_store
        self.signer = signer
        self.analyzer = ScriptAnalyzer()
    
    async def execute(self, input: HookInput) -> HookOutput:
        """
        Analyze script and gate critical risks.
        
        Returns:
            DENY - Critical risk, auto-blocked
            ALLOW - Script analyzed, let SecurityHook handle approval
        """
        tool_input = input.tool_input or {}
        script_id = tool_input.get("script_id")
        
        if not script_id:
            return HookOutput.deny("No script_id provided")
        
        # Get the script
        script = await self.script_store.find_by_id_prefix(script_id)
        if not script:
            return HookOutput.deny(f"Script not found: {script_id}")
        
        # Verify signature if signer available
        if self.signer and script.signature:
            is_valid = await self.signer.verify(script)
            if not is_valid:
                script.state = ScriptState.REJECTED
                script.review_notes = "Invalid signature - possible tampering"
                await self.script_store.update(script)
                return HookOutput.deny("Invalid script signature - possible tampering")
        
        # Analyze for security concerns
        result = self.analyzer.analyze(script)
        script.security_findings = result.findings
        script.risk_score = result.risk_score
        
        # Check for critical findings (auto-DENY)
        if result.has_critical:
            script.state = ScriptState.REJECTED
            critical_finding = next(
                (f for f in result.findings if f.severity == "critical"),
                None
            )
            if critical_finding is None:
                # Defensive: has_critical was true but no critical finding found
                script.review_notes = "Auto-rejected: Critical security issue detected"
                await self.script_store.update(script)
                return HookOutput.deny("Script blocked: Critical security issue detected")
            script.review_notes = f"Auto-rejected: {critical_finding.description}"
            await self.script_store.update(script)
            
            # Build denial response with suggestions
            denial = DenialResponse(
                decision="auto_deny",
                reason="critical_pattern",
                findings=result.findings,
                suggested_fixes=self.analyzer.get_suggested_fixes(script),
                alternative_approaches=self._get_alternative_approaches(critical_finding.category),
            )
            
            logger.warning(
                f"Script {script.id[:8]}... auto-denied: {critical_finding.description}"
            )
            
            return HookOutput.deny(
                f"Script blocked: {critical_finding.description}\n\n"
                f"Suggested alternatives:\n" +
                "\n".join(f"- {a}" for a in denial.alternative_approaches)
            )
        
        # Non-critical: update script state and let SecurityHook handle approval
        script.state = ScriptState.PENDING_REVIEW
        script.review_notes = f"Risk score: {script.risk_score}/100, {len(result.findings)} findings"
        
        if result.has_rewritable:
            script.review_notes += " (destructive operations will be rewritten)"
        
        await self.script_store.update(script)
        
        logger.info(
            f"Script {script.id[:8]}... analyzed: risk={script.risk_score}, "
            f"findings={len(result.findings)}, rewritable={result.has_rewritable}"
        )
        
        # Return ALLOW so SecurityHook can handle the actual approval
        return HookOutput.allow(
            f"Script analyzed (risk: {script.risk_score}/100, {len(result.findings)} findings)"
        )
    
    def _get_alternative_approaches(self, category: str) -> list:
        """Get alternative approaches for a blocked pattern category."""
        alternatives = {
            "rce": [
                "Use a known package manager (apt, pip, brew) instead of curl|sh",
                "Download the file first, review it, then execute separately",
                "Write the logic directly instead of downloading scripts",
            ],
            "fork_bomb": [
                "This pattern has no legitimate use - remove it",
                "If you need to spawn processes, use controlled subprocess calls",
            ],
            "disk_format": [
                "File formatting commands are too dangerous for automated execution",
                "Use cloud storage or partition management tools with proper safeguards",
            ],
            "disk_overwrite": [
                "Direct disk writes are blocked - use file system APIs instead",
                "If you need low-level disk access, use specialized tools with confirmation",
            ],
            "credential_access": [
                "Access credentials through proper secrets management",
                "Use environment variables or secret stores instead of reading files directly",
            ],
            "sandbox_escape": [
                "This pattern is attempting to break security boundaries",
                "If you need specific functionality, request it explicitly",
            ],
            "master_key": [
                "The master encryption key should never be accessed by scripts",
                "Use the provided encryption APIs instead of direct key access",
            ],
        }
        return alternatives.get(category, [
            "Review the pattern and consider safer alternatives",
            "Contact the Kestrel team if you believe this is a false positive",
        ])


class ComputeDebugHook(Hook):
    """
    Debug hook for compute feature development.
    
    Logs detailed information about compute operations for debugging.
    Only enabled when KESTREL_COMPUTE_DEBUG=true.
    """
    
    def __init__(self):
        import os
        enabled = os.environ.get("KESTREL_COMPUTE_DEBUG") == "true"
        
        super().__init__(
            name="compute_debug",
            events=[HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE],
            matcher=r"^(write_script|run_script|list_scripts|show_script)$",
            priority=1,  # Run very early
        )
        self.enabled = enabled
    
    async def execute(self, input: HookInput) -> HookOutput:
        """Log compute operation details."""
        if not self.enabled:
            return HookOutput.allow()
        
        import json
        
        event = input.hook_event_name
        tool = input.tool_name
        args = input.tool_input or {}
        
        logger.debug(
            f"[COMPUTE_DEBUG] {event} {tool}\n"
            f"  Args: {json.dumps(args, indent=2)}"
        )
        
        if event == "PostToolUse" and input.tool_response:
            logger.debug(
                f"[COMPUTE_DEBUG] Response:\n"
                f"  {json.dumps(input.tool_response, indent=2)[:500]}"
            )
        
        return HookOutput.allow()
