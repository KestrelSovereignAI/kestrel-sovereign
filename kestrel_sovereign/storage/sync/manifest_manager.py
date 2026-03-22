"""
Manifest Manager Mixin

Shared manifest management pattern used by sync targets that track
local state (content hashes, CIDs, blob names) to enable deduplication
and cold-start restore.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ManifestManagerMixin:
    """Mixin providing local manifest load/save for sync targets.

    Subclasses must set:
        _state_dir: Optional[Path]  — directory for manifest files
        _manifest_filename: str     — e.g. ".gcs_manifest_{agent_id}.json"

    Or override ``_manifest_path`` directly.
    """

    _state_dir: Optional[Path]
    _manifest_filename: str

    @property
    def _manifest_path(self) -> Optional[Path]:
        """Path to the local manifest file."""
        if self._state_dir:
            return self._state_dir / self._manifest_filename
        return None

    def _load_local_manifest(self) -> Optional[Dict[str, Any]]:
        """Load the local manifest if it exists."""
        path = self._manifest_path
        if path and path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load local manifest: {e}")
        return None

    def _save_local_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save manifest locally for quick lookup."""
        path = self._manifest_path
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save local manifest: {e}")
