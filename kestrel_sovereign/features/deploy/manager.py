"""
Deploy Manager.

Thin composition layer over DeployManagerCore for future extensibility.
"""

from .core import DeployManagerCore


class DeployManager(DeployManagerCore):
    """
    Deploy Manager for agent self-deployment.

    Currently a thin wrapper over DeployManagerCore.
    Can be extended with mixins for additional functionality:
    - Image building mixin
    - CI/CD integration mixin
    - Multi-cloud orchestration mixin
    """

    def __init__(self, *args, **kwargs):
        """Initialize deploy manager."""
        super().__init__(*args, **kwargs)
