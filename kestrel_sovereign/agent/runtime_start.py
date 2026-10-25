"""Host-owned maintenance boot policy, separate from durable SafeMode/Hold.

This policy is not authorization or an egress sandbox. It prevents Core and
participating feature startup paths from acquiring business work while a host
reconstructs a real feature registry. Hosts must still supply provider-free
policy, exact configured inventory and original native execution custody.
"""

from __future__ import annotations

from typing import Any


class MaintenanceBootError(RuntimeError):
    """A maintenance generation cannot enter serving/runtime work."""

    code = "agent_maintenance_boot"


def maintenance_boot_active(agent: Any) -> bool:
    """Only the explicit boolean public policy constitutes a startup hold."""
    return getattr(agent, "maintenance_boot", False) is True


def require_runtime_start_allowed(agent: Any) -> None:
    if maintenance_boot_active(agent):
        raise MaintenanceBootError(
            "maintenance boot holds business work until this generation is retired"
        )
