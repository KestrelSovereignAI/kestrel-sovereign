"""Host publications retain exact route ownership across replacement and failure."""

from dataclasses import dataclass

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Match, Route

from kestrel_sovereign.native_http import NativeHTTPRegistry


async def endpoint(_request):
    return PlainTextResponse("native")


@dataclass
class Binding:
    routing_key: str
    routes: tuple[Route, ...]

    def admits(self, scope):
        return any(route.matches(scope)[0] is Match.FULL for route in self.routes)


def binding(key, path):
    return Binding(key, (Route(path, endpoint, methods=["POST"]),))


def scope(app, path, method="POST"):
    return {"type": "http", "path": path, "root_path": "", "method": method, "app": app}


def test_replacement_and_stale_withdrawal_keep_current_native_generation():
    app = Starlette()
    registry = NativeHTTPRegistry(app, protocol="test")
    original, replacement = binding("owned", "/old"), binding("owned", "/new")
    registry.publish(original)
    assert registry.admits(scope(app, "/old"))
    registry.publish(replacement)
    assert not registry.is_current(original)
    assert not registry.withdraw(original)
    assert not registry.admits(scope(app, "/old"))
    assert registry.admits(scope(app, "/new"))
    assert not registry.admits(scope(app, "/new", "GET"))
    assert registry.withdraw(replacement)
    assert not registry.admits(scope(app, "/new"))
    assert not app.routes


def test_collision_is_refused_before_either_registration_changes():
    app = Starlette()
    registry = NativeHTTPRegistry(app, protocol="test")
    owner, other = binding("owned", "/owned"), binding("other", "/other")
    registry.publish(owner)
    registry.publish(other)
    before = tuple(app.routes)
    with pytest.raises(RuntimeError, match="already owned"):
        registry.publish(binding("owned", "/other"))
    assert tuple(app.routes) == before
    assert registry.is_current(owner) and registry.is_current(other)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "changed-binding"])
def test_corrupt_route_ownership_cannot_admit_withdraw_or_replace(fault):
    app = Starlette()
    registry = NativeHTTPRegistry(app, protocol="test")
    original = binding("owned", "/owned")
    registry.publish(original)
    if fault == "missing":
        app.routes.clear()
    elif fault == "duplicate":
        app.routes.append(original.routes[0])
    else:
        original.routes = ()
    before = tuple(app.routes)
    for operation in (lambda: registry.admits(scope(app, "/owned")),
                      lambda: registry.withdraw(original),
                      lambda: registry.publish(binding("owned", "/replacement"))):
        with pytest.raises(RuntimeError, match="lost its published route"):
            operation()
        assert tuple(app.routes) == before


def test_admission_cannot_borrow_another_application():
    registry = NativeHTTPRegistry(Starlette(), protocol="test")
    with pytest.raises(ValueError, match="different application"):
        registry.admits(scope(Starlette(), "/owned"))


@pytest.mark.parametrize("fault", ["empty", "duplicate-object", "duplicate-path"])
def test_invalid_registration_is_refused_before_publication(fault):
    app = Starlette()
    registry = NativeHTTPRegistry(app, protocol="test")
    original = binding("owned", "/owned")
    registry.publish(original)
    route = Route("/replacement", endpoint)
    routes = () if fault == "empty" else (route, route if fault == "duplicate-object" else Route("/replacement", endpoint))
    with pytest.raises(ValueError, match="distinct published routes"):
        registry.publish(Binding("owned", routes))
    assert registry.is_current(original)
    assert tuple(app.routes) == original.routes
