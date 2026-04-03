"""
Deploy Manager.

Composition layer over DeployManagerCore with entry_point discovery
for external cloud deployment providers.

External packages can register deploy providers via entry_points::

    [project.entry-points."kestrel_sovereign.cloud_providers"]
    RunPodProvider = "kestrel_cloud_runpod:RunPodProvider"
"""

import logging
from typing import Dict, Type

from kestrel_sovereign.entrypoints import discover_entry_point_classes
from .core import DeployManagerCore
from .providers.base import DeployProvider

logger = logging.getLogger(__name__)

CLOUD_PROVIDER_ENTRY_POINT_GROUP = "kestrel_sovereign.cloud_providers"


class DeployManager(DeployManagerCore):
    """
    Deploy Manager for agent self-deployment.

    Extends DeployManagerCore with entry_point discovery for external
    cloud providers (RunPod, Vast.ai, etc.).
    """

    def __init__(self, *args, **kwargs):
        """Initialize deploy manager and discover external providers."""
        super().__init__(*args, **kwargs)
        self._external_provider_classes: Dict[str, Type[DeployProvider]] = {}
        self._discover_entrypoint_providers()

    def _discover_entrypoint_providers(self) -> None:
        """Discover external deploy provider classes via entry_points."""
        classes = discover_entry_point_classes(
            CLOUD_PROVIDER_ENTRY_POINT_GROUP, DeployProvider,
        )
        for ep_name, cls in classes.items():
            self._external_provider_classes[ep_name] = cls
            logger.info("Discovered entry_point cloud provider: %s", ep_name)

    def get_external_provider_class(self, name: str) -> type | None:
        """Get an external provider class by entry point name.

        Args:
            name: Entry point name (e.g. "runpod", "vastai").

        Returns:
            The DeployProvider subclass, or None if not found.
        """
        return self._external_provider_classes.get(name)

    def list_external_providers(self) -> list[str]:
        """List names of deploy providers discovered via entry_points."""
        return list(self._external_provider_classes.keys())
