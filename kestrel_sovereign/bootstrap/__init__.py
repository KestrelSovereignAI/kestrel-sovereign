"""
Bootstrap module for Kestrel agent wake-up and personality discovery.

This module handles the first-time experience when a new agent comes online,
guiding them through a discovery conversation to establish their personality.
"""

from .service import BootstrapService, BootstrapState

__all__ = ["BootstrapService", "BootstrapState"]
