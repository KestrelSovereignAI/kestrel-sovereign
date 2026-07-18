"""Deterministic genesis auditor for tests that exercise later boundaries."""


async def deterministic_safe_genesis_auditor(_prompt: str) -> dict:
    """Return a stable pass through the production genesis lifecycle."""
    return {
        "risk_level": 1,
        "reasoning": "Deterministic shared integration-test genesis audit.",
    }


async def complete_deterministic_genesis_audit(
    agent,
    *,
    provenance: str,
) -> None:
    """Complete genesis for a programmatically constructed integration agent."""
    original_auditor = agent.get_audit_response
    agent.get_audit_response = deterministic_safe_genesis_auditor
    try:
        await agent.perform_genesis_audit(provenance=provenance)
    finally:
        agent.get_audit_response = original_auditor
