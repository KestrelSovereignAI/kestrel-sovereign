"""FeatureForgeFeature — governed agent self-extension (issue #2434).

The feature that creates features: scaffold a feature package from a spec, gate
it with the Iron Rule (narrow-only permissions), and queue it for Sovereign
approval. Forged features are inert until approved.
"""

from .feature import FeatureForgeFeature

__all__ = ["FeatureForgeFeature"]
