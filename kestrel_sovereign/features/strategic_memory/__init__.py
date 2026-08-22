"""Strategic Memory Feature package.

Re-exports StrategicMemoryFeature for backward compatibility with:
    from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature
"""

from .feature import StrategicMemoryFeature
from .issue_selection import pick_top_issue

__all__ = ["StrategicMemoryFeature", "pick_top_issue"]
