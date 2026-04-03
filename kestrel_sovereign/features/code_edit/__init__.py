"""
Code Edit Feature - Self-modification with constitutional approval.

This in-tree module re-exports from the extracted kestrel-feature-code package
when available, falling back to the local implementation for backward compatibility.
"""

try:
    from kestrel_feature_code import CodeEditFeature
except ImportError:
    from .feature import CodeEditFeature

__all__ = ["CodeEditFeature"]
