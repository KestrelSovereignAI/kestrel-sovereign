"""Atomic native router inclusion for feature-contributed routers.

``FastAPI.include_router`` appends the included routes before it copies the
router's startup/shutdown handlers and merges its lifespan, so an inclusion
that raises part-way leaves a partial contribution registered on the app. Agent
feature routers and host feature routers both mount through this one rollback.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from fastapi import FastAPI


@contextmanager
def atomic_router_inclusion(app: FastAPI) -> Iterator[int]:
    """Undo every native registration made inside the block if it raises.

    Yields the index in ``app.routes`` at which the block's routes begin. On
    any exception, the routes added since then are removed and the app's
    startup/shutdown handlers and lifespan context are restored to what they
    were on entry; the exception then propagates unchanged.
    """
    routes_before = len(app.routes)
    startup = list(app.router.on_startup)
    shutdown = list(app.router.on_shutdown)
    lifespan = app.router.lifespan_context
    try:
        yield routes_before
    except BaseException:
        del app.routes[routes_before:]
        app.router.on_startup[:] = startup
        app.router.on_shutdown[:] = shutdown
        app.router.lifespan_context = lifespan
        raise


__all__ = ["atomic_router_inclusion"]
