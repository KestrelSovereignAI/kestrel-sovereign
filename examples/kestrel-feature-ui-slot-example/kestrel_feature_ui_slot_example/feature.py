"""The reference feature class.

``get_ui_contributions()`` points the host at this package's OWN ``static/``
directory (resolved relative to this file — never a path in core ``static/``).
The host mounts it at ``/features/ui-slot-example/static/`` and the manifest
advertises ``ui.js``; the frontend boot loader imports it, and it registers one
button into the ``chat-input-actions`` slot.
"""

from pathlib import Path
from typing import Optional

from kestrel_sovereign.features.base import Feature, UIContributions

# This package's assets live HERE, beside this module — not in core static/.
_STATIC_DIR = Path(__file__).resolve().parent / "static"


class UISlotExampleFeature(Feature):
    """Minimal feature whose only job is to contribute one UI button."""

    @property
    def tool_description(self) -> str:
        return (
            "Reference feature demonstrating manifest-driven, out-of-tree UI "
            "asset loading (epic #2038, ticket #2043). Contributes one button "
            "to the chat input row; exposes no agent tools."
        )

    async def initialize(self):
        # Nothing to initialize — this feature is UI-only.
        return None

    def get_ui_contributions(self) -> Optional[UIContributions]:
        return UIContributions(
            modules=["ui.js"],
            static_dir=str(_STATIC_DIR),
            capability="ui_slot_example",
        )
