"""
Constitutional Council - Multi-Model Deliberation System

A deliberation system where multiple foundation models must reach consensus
before Kestrel takes major irreversible actions. This mirrors human governance
structures (Supreme Court, corporate boards, DAO voting) but with AI participants.

Usage:
    from kestrel_sovereign.features.council import CouncilFeature, convene_council

    # Load council configuration
    council = CouncilFeature()

    # Convene council on a question
    session = await convene_council(
        question="Should we proceed with creating Emma?",
        evidence=await compile_evidence("emma_genesis")
    )

    if session.outcome == "APPROVED":
        # Proceed with action
        pass
"""

from .models import (
    CouncilMember,
    Evidence,
    Verdict,
    DeliberationRound,
    CouncilSession,
    ConsensusRule,
)
from .evidence import compile_evidence
from .deliberation import convene_council
from .feature import CouncilFeature

__all__ = [
    "CouncilMember",
    "Evidence",
    "Verdict",
    "DeliberationRound",
    "CouncilSession",
    "ConsensusRule",
    "compile_evidence",
    "convene_council",
    "CouncilFeature",
]
