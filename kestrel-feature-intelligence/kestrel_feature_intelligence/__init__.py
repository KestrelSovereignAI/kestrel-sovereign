"""
Kestrel Feature: Intelligence — Metacognition and Multi-Model Governance

This package provides two features for agent self-improvement:

- **ReflectionFeature**: Agent self-reflection, insight generation, and behavioral
  improvement with constitutional approval gates.
- **CouncilFeature**: Multi-model deliberation system where foundation models
  reach consensus before irreversible actions.

Install::

    pip install kestrel-feature-intelligence

Both features are discovered automatically via entry_points when installed
alongside kestrel-sovereign.
"""

from kestrel_feature_intelligence.reflection import ReflectionFeature
from kestrel_feature_intelligence.council import CouncilFeature

__all__ = [
    "ReflectionFeature",
    "CouncilFeature",
]
