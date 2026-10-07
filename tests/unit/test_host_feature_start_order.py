"""Host features start provider-before-consumer (#3501).

Talon, Eye and Flight start polling workers from their own start hooks. Started
in discovery order -- before Workflows -- each worker's first claim found the
Workflows run service registered but not yet bound to a store, and logged an
ERROR on every boot. Core now orders host-feature start by the
``ServiceRequirement`` values each feature holds.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from kestrel_sdk.features.host_base import HostFeature
from kestrel_sdk.operator import (
    ServiceDescriptor,
    ServiceRegistration,
    ServiceRequirement,
    ServiceScope,
)

from kestrel_sovereign.host_features import (
    SovereignHostContext,
    start_host_features,
    start_order,
    stop_host_features,
)
from kestrel_sovereign.host_features.start_order import scan_service_requirements

_LOGGER = "kestrel_sovereign.host_features.start_order"


def _requirement(name: str, minimum_version: str = "1.0.0") -> ServiceRequirement:
    return ServiceRequirement(name, minimum_version, ServiceScope.HOST)


class _RunService:
    """Registered at activation, ready only once its owner's start hook ran."""

    def __init__(self) -> None:
        self.is_ready = False


class _Operator:
    """Holds the requirement where the real consumers do: on an inner object."""

    def __init__(self, requirements: tuple[ServiceRequirement, ...]) -> None:
        self._requirements = requirements


class _OrderedHostFeature(HostFeature):
    """A provider of ``provides`` and a consumer of ``requires``.

    Like the Talon/Eye/Flight operators, a consumer's start hook launches a
    worker whose first claim runs after an await.
    """

    def __init__(
        self,
        name: str,
        events: list[tuple[str, str]],
        *,
        provides: tuple[tuple[str, str], ...] = (),
        requires: tuple[ServiceRequirement, ...] = (),
    ) -> None:
        self.name = name
        self.events = events
        self.service = _RunService()
        self.operator = _Operator(requires)
        self.observed_ready: dict[str, bool] = {}
        self.worker: asyncio.Task[None] | None = None
        self._registrations = tuple(
            ServiceRegistration(
                descriptor=ServiceDescriptor(service, version, ServiceScope.HOST),
                service=self.service,
                owner=self.contribution_owner,
            )
            for service, version in provides
        )

    @property
    def contribution_owner(self) -> str:
        return f"tests-start-order:{self.name}"

    def get_service_registrations(self) -> tuple[ServiceRegistration, ...]:
        return self._registrations

    async def on_host_start(self, ctx) -> None:
        self.events.append(("start-begin", self.name))
        if self.operator._requirements:
            self.worker = asyncio.create_task(self._claim(ctx.operator_registry))
        await asyncio.sleep(0)
        self.service.is_ready = True
        self.events.append(("start-end", self.name))

    async def _claim(self, registry) -> None:
        await asyncio.sleep(0)
        for requirement in self.operator._requirements:
            service = registry.resolve_compatible_service(requirement)
            if service is not None:
                self.observed_ready[requirement.name] = service.is_ready

    async def on_host_stop(self, ctx) -> None:
        self.events.append(("stop", self.name))


async def _settle(features) -> None:
    await asyncio.gather(*(f.worker for f in features if f.worker is not None))


def _names(features) -> list[str]:
    return [feature.name for feature in features]


def _start_order(ctx) -> list[str]:
    return _names(ctx.started_host_features)


def _records(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == _LOGGER]


@pytest.mark.asyncio
async def test_a_provider_finishes_starting_before_a_consumer_listed_first():
    events: list[tuple[str, str]] = []
    consumer = _OrderedHostFeature(
        "consumer", events, requires=(_requirement("workflows.runs"),)
    )
    provider = _OrderedHostFeature(
        "provider", events, provides=(("workflows.runs", "1.1.0"),)
    )
    ctx = SovereignHostContext()

    started = await start_host_features([consumer, provider], ctx)
    await _settle(started)

    assert ctx.started_host_features == (provider, consumer)
    assert events.index(("start-end", "provider")) < events.index(
        ("start-begin", "consumer")
    )
    # The race itself: the consumer's worker resolved a service that was
    # already ready, not one registered and still waiting on its start hook.
    assert consumer.observed_ready == {"workflows.runs": True}
    await stop_host_features(started, ctx)


@pytest.mark.asyncio
async def test_consumers_stop_before_their_providers():
    events: list[tuple[str, str]] = []
    consumer = _OrderedHostFeature(
        "consumer", events, requires=(_requirement("workflows.runs"),)
    )
    provider = _OrderedHostFeature(
        "provider", events, provides=(("workflows.runs", "1.0.0"),)
    )
    ctx = SovereignHostContext()
    started = await start_host_features([consumer, provider], ctx)
    await _settle(started)
    assert started == [consumer, provider]

    # Handed the list in discovery order, as the server's shutdown does.
    await stop_host_features(started, ctx)

    stops = [name for kind, name in events if kind == "stop"]
    assert stops == ["consumer", "provider"]


@pytest.mark.asyncio
async def test_the_started_set_is_returned_in_discovery_order_for_mounting():
    """Routers and console panels mount from the returned list, in its order.

    Only the lifecycle moves: the console's panel order must not change
    because a provider had to start first.
    """

    events: list[tuple[str, str]] = []
    consumer = _OrderedHostFeature(
        "consumer", events, requires=(_requirement("workflows.runs"),)
    )
    middle = _OrderedHostFeature("middle", events)
    provider = _OrderedHostFeature(
        "provider", events, provides=(("workflows.runs", "1.0.0"),)
    )
    ctx = SovereignHostContext()

    started = await start_host_features([consumer, middle, provider], ctx)
    await _settle(started)

    assert started == [consumer, middle, provider]
    assert _start_order(ctx) == ["provider", "consumer", "middle"]
    await stop_host_features(started, ctx)


@pytest.mark.asyncio
async def test_independent_features_keep_discovery_order(caplog):
    events: list[tuple[str, str]] = []
    features = [_OrderedHostFeature(name, events) for name in ("c", "a", "b")]
    ctx = SovereignHostContext()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        started = await start_host_features(features, ctx)

    assert _start_order(ctx) == ["c", "a", "b"]
    assert [name for kind, name in events if kind == "start-begin"] == ["c", "a", "b"]
    assert _records(caplog) == []
    await stop_host_features(started, ctx)


@pytest.mark.asyncio
async def test_only_a_provider_moves_and_only_to_just_before_its_first_consumer():
    events: list[tuple[str, str]] = []
    features = [
        _OrderedHostFeature("first", events),
        _OrderedHostFeature(
            "consumer", events, requires=(_requirement("workflows.runs"),)
        ),
        _OrderedHostFeature("middle", events),
        _OrderedHostFeature(
            "provider", events, provides=(("workflows.runs", "1.0.0"),)
        ),
        _OrderedHostFeature("last", events),
    ]
    ctx = SovereignHostContext()

    started = await start_host_features(features, ctx)
    await _settle(started)

    assert _start_order(ctx) == ["first", "provider", "consumer", "middle", "last"]
    await stop_host_features(started, ctx)


@pytest.mark.asyncio
async def test_a_transitive_provider_starts_before_the_provider_that_needs_it():
    events: list[tuple[str, str]] = []
    flight = _OrderedHostFeature(
        "flight", events, requires=(_requirement("eye.reviews"),)
    )
    eye = _OrderedHostFeature(
        "eye",
        events,
        provides=(("eye.reviews", "1.0.0"),),
        requires=(_requirement("workflows.runs"),),
    )
    workflows = _OrderedHostFeature(
        "workflows", events, provides=(("workflows.runs", "1.1.0"),)
    )
    ctx = SovereignHostContext()

    started = await start_host_features([flight, eye, workflows], ctx)
    await _settle(started)

    assert _start_order(ctx) == ["workflows", "eye", "flight"]
    assert eye.observed_ready == {"workflows.runs": True}
    assert flight.observed_ready == {"eye.reviews": True}
    await stop_host_features(started, ctx)


@pytest.mark.asyncio
async def test_a_dependency_cycle_is_logged_by_name_and_every_feature_starts(caplog):
    events: list[tuple[str, str]] = []
    cycle_a = _OrderedHostFeature(
        "cycle-a",
        events,
        provides=(("alpha.service", "1.0.0"),),
        requires=(_requirement("beta.service"), _requirement("shared.base")),
    )
    cycle_b = _OrderedHostFeature(
        "cycle-b",
        events,
        provides=(("beta.service", "1.0.0"),),
        requires=(_requirement("alpha.service"),),
    )
    base = _OrderedHostFeature(
        "base", events, provides=(("shared.base", "1.0.0"),)
    )
    ctx = SovereignHostContext()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        started = await start_host_features([cycle_a, cycle_b, base], ctx)
    await _settle(started)

    # Nothing dropped, nothing hung. The cycle's members fall back to discovery
    # order between themselves, still after the provider outside the cycle.
    assert _start_order(ctx) == ["base", "cycle-a", "cycle-b"]
    [record] = _records(caplog)
    assert record.levelno == logging.ERROR
    message = record.getMessage()
    assert "cycle" in message
    assert "cycle-a requires beta.service from cycle-b" in message
    assert "cycle-b requires alpha.service from cycle-a" in message
    assert "base" not in message.split("(", 1)[0]
    await stop_host_features(started, ctx)


@pytest.mark.asyncio
async def test_a_requirement_no_host_feature_provides_is_logged_and_still_starts(
    caplog,
):
    events: list[tuple[str, str]] = []
    orphan = _OrderedHostFeature(
        "orphan", events, requires=(_requirement("absent.service", "1.2.0"),)
    )
    peer = _OrderedHostFeature("peer", events)
    ctx = SovereignHostContext()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        started = await start_host_features([orphan, peer], ctx)
    await _settle(started)

    assert _start_order(ctx) == ["orphan", "peer"]
    [record] = _records(caplog)
    assert record.levelno == logging.WARNING
    assert "orphan" in record.getMessage()
    assert "absent.service >= 1.2.0" in record.getMessage()
    await stop_host_features(started, ctx)


@pytest.mark.asyncio
async def test_an_incompatible_provider_version_neither_orders_nor_satisfies(caplog):
    events: list[tuple[str, str]] = []
    consumer = _OrderedHostFeature(
        "consumer", events, requires=(_requirement("workflows.runs", "2.0.0"),)
    )
    provider = _OrderedHostFeature(
        "provider", events, provides=(("workflows.runs", "1.9.0"),)
    )
    ctx = SovereignHostContext()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        started = await start_host_features([consumer, provider], ctx)
    await _settle(started)

    # The order follows the registry's own compatibility rule: a provider the
    # consumer could never resolve is not a provider of it.
    assert _start_order(ctx) == ["consumer", "provider"]
    assert consumer.observed_ready == {}
    [record] = _records(caplog)
    assert "workflows.runs >= 2.0.0" in record.getMessage()
    await stop_host_features(started, ctx)


@pytest.mark.asyncio
async def test_a_provider_already_running_satisfies_a_later_start(caplog):
    events: list[tuple[str, str]] = []
    provider = _OrderedHostFeature(
        "provider", events, provides=(("workflows.runs", "1.0.0"),)
    )
    consumer = _OrderedHostFeature(
        "consumer", events, requires=(_requirement("workflows.runs"),)
    )
    ctx = SovereignHostContext()
    await start_host_features([provider], ctx)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        started = await start_host_features([consumer], ctx)
    await _settle(started)

    assert _records(caplog) == []
    assert consumer.observed_ready == {"workflows.runs": True}
    await stop_host_features(started, ctx)
    await stop_host_features([provider], ctx)


@pytest.mark.asyncio
async def test_a_scan_that_hits_its_budget_is_logged_and_the_feature_starts(
    caplog, monkeypatch
):
    class _Link:
        def __init__(self, nxt) -> None:
            self.nxt = nxt

    events: list[tuple[str, str]] = []
    deep = _OrderedHostFeature("deep", events)
    chain = None
    for _ in range(50):
        chain = _Link(chain)
    deep.chain = chain
    monkeypatch.setattr(start_order, "REQUIREMENT_SCAN_BUDGET", 10)
    ctx = SovereignHostContext()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        started = await start_host_features([deep], ctx)

    assert started == [deep]
    [record] = _records(caplog)
    assert "deep" in record.getMessage()
    assert "scan stopped after 10 objects" in record.getMessage()
    await stop_host_features(started, ctx)


# ---------------------------------------------------------------------------
# The requirement scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SlottedPolicy:
    requirement: ServiceRequirement


class _Unbound:
    """Talon's pre-start application placeholder: every attribute read raises."""

    def __getattr__(self, name):
        raise RuntimeError("application is not initialized")


class _Lazy:
    @property
    def requirement(self):
        raise AssertionError("the scan must not run a property")


class _ScannedFeature(HostFeature):
    name = "scanned"


def test_the_scan_finds_requirements_in_containers_and_slots():
    in_tuple = _requirement("in.tuple")
    in_dict = _requirement("in.dict")
    in_set = _requirement("in.set")
    in_slot = _requirement("in.slot")
    feature = _ScannedFeature()
    feature.operator = _Operator((in_tuple,))
    feature.by_name = {"policy": [in_dict, {in_set}]}
    feature.policy = _SlottedPolicy(in_slot)
    feature.again = _Operator((in_tuple,))

    scan = scan_service_requirements(feature)

    assert scan.complete
    assert set(scan.requirements) == {in_tuple, in_dict, in_set, in_slot}
    assert len(scan.requirements) == 4


def test_the_scan_runs_no_code_of_the_objects_it_reads():
    held = _requirement("held.elsewhere")
    feature = _ScannedFeature()
    feature.application = _Unbound()
    feature.lazy = _Lazy()
    feature.operator = _Operator((held,))

    assert scan_service_requirements(feature).requirements == (held,)


def test_the_scan_does_not_enter_objects_from_other_packages():
    """It walks the feature's own state, never a router, lock or client."""

    from types import SimpleNamespace

    feature = _ScannedFeature()
    feature.foreign = SimpleNamespace(requirement=_requirement("foreign.service"))

    assert scan_service_requirements(feature).requirements == ()


def test_the_scan_trusts_types_not_a_double_claiming_to_be_one():
    feature = _ScannedFeature()
    feature.requirement_double = MagicMock(spec=ServiceRequirement)
    feature.list_double = MagicMock(spec=list)
    feature.dict_double = MagicMock(spec=dict)

    scan = scan_service_requirements(feature)

    assert scan.complete
    assert scan.requirements == ()


@pytest.mark.asyncio
async def test_a_feature_that_consumes_its_own_service_needs_no_other_provider(
    caplog,
):
    events: list[tuple[str, str]] = []
    first = _OrderedHostFeature("first", events)
    loopback = _OrderedHostFeature(
        "loopback",
        events,
        provides=(("own.service", "1.0.0"),),
        requires=(_requirement("own.service"),),
    )
    ctx = SovereignHostContext()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        started = await start_host_features([first, loopback], ctx)
    await _settle(started)

    assert _start_order(ctx) == ["first", "loopback"]
    assert _records(caplog) == []
    await stop_host_features(started, ctx)


@pytest.mark.asyncio
async def test_two_providers_of_one_consumer_keep_their_discovery_order():
    """Talon needs Workflows and the Claws catalog, both discovered after it."""

    events: list[tuple[str, str]] = []
    talon = _OrderedHostFeature(
        "talon",
        events,
        requires=(_requirement("claws.catalog"), _requirement("workflows.runs")),
    )
    workflows = _OrderedHostFeature(
        "workflows", events, provides=(("workflows.runs", "1.1.0"),)
    )
    claws = _OrderedHostFeature(
        "claws-catalog", events, provides=(("claws.catalog", "1.3.0"),)
    )
    ctx = SovereignHostContext()

    started = await start_host_features([talon, workflows, claws], ctx)
    await _settle(started)

    assert _start_order(ctx) == ["workflows", "claws-catalog", "talon"]
    assert talon.observed_ready == {"claws.catalog": True, "workflows.runs": True}
    await stop_host_features(started, ctx)


def test_a_duck_typed_double_feature_scans_to_nothing():
    """Embedders still start ``MagicMock`` doubles as host features.

    The scan begins at the double itself, so its spec's claims about being a
    container, or about the children it holds, must not be believed.
    """

    root = MagicMock(spec=list)
    root.child = MagicMock(spec=ServiceRequirement)

    scan = scan_service_requirements(root)

    assert scan.complete
    assert scan.requirements == ()


@pytest.mark.asyncio
async def test_features_with_no_recorded_start_stop_in_reverse_request_order():
    """A context that never recorded a start keeps the old stop order."""

    from types import SimpleNamespace

    events: list[tuple[str, str]] = []
    features = [_OrderedHostFeature(name, events) for name in ("x", "y", "z")]

    await stop_host_features(features, SimpleNamespace())

    assert [name for kind, name in events if kind == "stop"] == ["z", "y", "x"]
