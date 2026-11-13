"""Publish exact native HTTP route objects and classify their installed surface."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from starlette.applications import Starlette
from starlette.routing import BaseRoute
from starlette.types import Scope


class NativeHTTPBinding(Protocol):
    """The host registration, SDK routes and request classifier are one object."""

    @property
    def routing_key(self) -> str: ...

    @property
    def routes(self) -> Sequence[BaseRoute]: ...

    def admits(self, scope: Scope) -> bool: ...


_Binding = TypeVar("_Binding", bound=NativeHTTPBinding)


@dataclass(frozen=True)
class _Registration(Generic[_Binding]):
    binding: _Binding
    routes: tuple[BaseRoute, ...]


class NativeHTTPRegistry(Generic[_Binding]):
    """Publish and withdraw one exact generation under the host lifecycle writer.

    Classification is not authentication. Each admitted request still enters
    its protocol's original authorization and current-agent authority guard.
    """

    def __init__(self, app: Starlette, *, protocol: str):
        self._app, self._protocol = app, protocol
        self._registrations: dict[str, _Registration[_Binding]] = {}

    def publish(self, binding: _Binding) -> None:
        routes = tuple(binding.routes)
        paths = [route.path for route in routes]
        if not routes or len(set(paths)) != len(paths) or len({id(r) for r in routes}) != len(routes):
            raise ValueError(f"Native {self._protocol} requires distinct published routes")
        previous = self._registrations.get(binding.routing_key)
        previous_ids = {id(route) for route in previous.routes} if previous else set()
        if previous is not None:
            self._require_routes(previous)
        if any(id(route) not in previous_ids and getattr(route, "path", None) in paths
               for route in self._app.routes):
            raise RuntimeError(f"Native {self._protocol} mount is already owned by another registration")
        # Check every previous route and collision before changing dispatch.
        self._app.routes[:] = [route for route in self._app.routes if id(route) not in previous_ids]
        self._app.routes[0:0] = routes
        self._registrations[binding.routing_key] = _Registration(binding, routes)

    def _require_routes(self, registration: _Registration[_Binding]) -> None:
        current = tuple(registration.binding.routes)
        if (len(current) != len(registration.routes)
                or any(actual is not original for actual, original in zip(current, registration.routes))
                or any(sum(route is owned for route in self._app.routes) != 1
                       for owned in registration.routes)):
            raise RuntimeError(f"Native {self._protocol} registry lost its published route")

    def is_current(self, binding: _Binding) -> bool:
        registration = self._registrations.get(binding.routing_key)
        if registration is None or registration.binding is not binding:
            return False
        self._require_routes(registration)
        return True

    def withdraw(self, binding: _Binding) -> bool:
        """Only the current generation may remove the routes it published."""
        if not self.is_current(binding):
            return False
        registration = self._registrations[binding.routing_key]
        owned_ids = {id(route) for route in registration.routes}
        self._app.routes[:] = [route for route in self._app.routes if id(route) not in owned_ids]
        del self._registrations[binding.routing_key]
        return True

    def _current_bindings(self) -> Iterator[_Binding]:
        for registration in self._registrations.values():
            self._require_routes(registration)
            yield registration.binding

    def admits(self, scope: Scope) -> bool:
        self.require_application(scope.get("app"))
        return any(binding.admits(scope) for binding in self._current_bindings())

    def require_application(self, app: Starlette) -> None:
        if app is not self._app:
            raise ValueError(f"Native {self._protocol} registry belongs to a different application")
