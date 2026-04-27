"""Computer-use feature: bounded host access for sovereign agents.

Three independent gates wrap every tool call (privacy → constitution →
approval). Disabled by default — see ``KESTREL_CONSTITUTION.md`` Amendment
IX and ``[features.computer_use]`` in ``kestrel.toml``.
"""

from .feature import ComputerUseFeature

__all__ = ["ComputerUseFeature"]
