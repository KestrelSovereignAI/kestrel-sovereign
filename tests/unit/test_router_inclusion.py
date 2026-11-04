"""The one rollback shared by agent and host feature router mounts (#3926)."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import iter_route_contexts

from kestrel_sovereign.router_inclusion import atomic_router_inclusion


async def _handler():
    pass


@asynccontextmanager
async def _lifespan(_app):
    yield


def _router() -> APIRouter:
    router = APIRouter(on_startup=[_handler], on_shutdown=[_handler], lifespan=_lifespan)

    @router.get("/included")
    async def included():
        return {"ok": True}

    return router


def _paths(app: FastAPI) -> set:
    return {context.path for context in iter_route_contexts(app.routes)}


def test_a_failed_block_restores_routes_handlers_and_lifespan():
    app = FastAPI()
    routes = list(app.routes)
    on_start = list(app.router.on_startup)
    on_stop = list(app.router.on_shutdown)
    lifespan = app.router.lifespan_context

    with pytest.raises(RuntimeError, match="after inclusion"):
        with atomic_router_inclusion(app):
            app.include_router(_router())
            # The premise: the inclusion registered all three before failing.
            assert "/included" in _paths(app)
            assert app.router.on_startup != on_start
            assert app.router.lifespan_context is not lifespan
            raise RuntimeError("after inclusion")

    assert app.routes == routes
    assert app.router.on_startup == on_start
    assert app.router.on_shutdown == on_stop
    assert app.router.lifespan_context is lifespan


def test_a_completed_block_keeps_everything_it_registered():
    app = FastAPI()
    before = len(app.routes)

    with atomic_router_inclusion(app) as start:
        app.include_router(_router())

    assert start == before
    assert "/included" in _paths(app)
    assert _handler in app.router.on_startup
    assert _handler in app.router.on_shutdown


def test_routes_registered_before_the_block_survive_its_failure():
    app = FastAPI()

    @app.get("/earlier")
    async def earlier():
        return {}

    with pytest.raises(KeyboardInterrupt):
        with atomic_router_inclusion(app):
            app.include_router(_router())
            raise KeyboardInterrupt

    assert "/earlier" in _paths(app)
    assert "/included" not in _paths(app)
