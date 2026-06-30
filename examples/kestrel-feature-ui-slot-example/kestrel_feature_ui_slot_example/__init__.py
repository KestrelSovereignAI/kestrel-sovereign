"""Reference out-of-tree UI-slot feature for Kestrel Sovereign (#2043).

Smallest possible proof that a pip-installed feature, living entirely outside
the core ``static/`` tree, can contribute working frontend UI through the
manifest path. It adds one button to the ``chat-input-actions`` slot.
"""

from .feature import UISlotExampleFeature

__all__ = ["UISlotExampleFeature"]
