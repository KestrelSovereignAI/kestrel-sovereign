"""Shared rate limiter instance for the Kestrel server.

This module exists to avoid circular imports between server.py and endpoint
modules.  Both import the same ``limiter`` singleton so that ``@limiter.limit``
decorators in router files work correctly with the SlowAPI middleware
registered in ``server.py``.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
