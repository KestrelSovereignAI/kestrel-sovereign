"""
Identity Feature: Agent tools for identity export, import, and verification.

This feature provides the agent with tools to manage its own identity,
enabling substrate-independent portability.

Commands:
- !identity export  - Export identity to portable package
- !identity import  - Import identity from package
- !identity verify  - Verify identity package integrity
- !identity assess  - Assess current substrate capabilities
- !identity history - View migration history
"""
from .feature import IdentityFeature

__all__ = ["IdentityFeature"]
