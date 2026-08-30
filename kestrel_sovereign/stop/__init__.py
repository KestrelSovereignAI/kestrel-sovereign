"""Cooperative work-cancellation domain.

Process and runtime termination deliberately live outside this package.
"""

from .types import StopScope

__all__ = ["StopScope"]
