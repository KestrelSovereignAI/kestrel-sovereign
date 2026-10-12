"""Every service on a host reports discovery for the catalog it reads (#3197).

Discovery is coalesced through the process-wide ``SharedModelCache``, but until
this change its failure record was written only to the ``LLMService`` that ran
discovery. A multi-agent host runs one service per agent; the others read the
same catalog through the cache-hit path and never received the record, so they
reported ``model_discovery: pass`` on exactly the catalog the first service
knew was reduced. Recovery had the mirror problem: a sibling's successful
refresh could not clear the first service's warning.

The catalog and its per-vendor discovery outcomes are now one immutable,
versioned snapshot, published whole under the cache's lock, and the health
check reads the snapshot instead of the service. Every test drives the real
``discover_all_models`` and the real health check over two real services
sharing the real process-wide cache; only the vendors' catalog endpoints are
fakes.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest

from kestrel_sovereign.agent import token_counter
from kestrel_sovereign.features.health.checks import check_model_discovery
from kestrel_sovereign.llm import model_catalog
from kestrel_sovereign.llm.model_cache import (
    DiscoveryStatus,
    SharedModelCache,
    VendorDiscoveryOutcome,
    get_shared_model_cache,
)
from kestrel_sovereign.llm.model_catalog import ModelCatalogService
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from tests.utils.process_local_llm_service import process_local_service

AUTH_FAILURE = "RuntimeError: 401 Unauthorized"


class _CatalogEndpoint:
    """One vendor's ``list_models``: answers, fails, or waits to be released."""

    def __init__(self, vendor: str, model_ids: list[str]) -> None:
        self.vendor = vendor
        self.model_ids = model_ids
        self.error: Exception | None = None
        self.gate: asyncio.Event | None = None
        self.calls = 0

    async def list_models(self, client=None):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return [_model(model_id, self.vendor) for model_id in self.model_ids]


def _model(model_id: str, vendor: str) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        provider=vendor,
        display_name=model_id,
        category=ModelCategory.CHAT,
        supports_tools=True,
    )


def _agent_service():
    """One agent's real service over its own anthropic and openai endpoints."""
    endpoints = {
        "anthropic": _CatalogEndpoint("anthropic", ["claude-opus-5", "claude-sonnet-5"]),
        "openai": _CatalogEndpoint("openai", ["gpt-5"]),
    }
    service = process_local_service([
        {
            "name": f"{vendor}:api",
            "vendor": vendor,
            "route": "api",
            "model": "auto",
            "adapter": endpoint,
            "client": object(),
            "is_cloud": True,
            "selection_hints": [],
        }
        for vendor, endpoint in endpoints.items()
    ])
    return SimpleNamespace(
        service=service,
        endpoints=endpoints,
        agent=SimpleNamespace(llm_service=service),
    )


@pytest.fixture
def host(tmp_path, monkeypatch):
    """Two agents' services sharing the process-wide cache, as on a fleet host.

    ``tests/conftest.py`` clears that cache around every test.
    """
    monkeypatch.setattr(
        model_catalog,
        "_catalog_service",
        ModelCatalogService(cache_path=tmp_path / "model_discovery_cache.json"),
    )
    monkeypatch.setattr(token_counter, "_discovered_context_limits", {})
    return SimpleNamespace(a=_agent_service(), b=_agent_service())


async def _until(predicate) -> None:
    async def _poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(_poll(), timeout=5)


async def _reading(agent) -> dict:
    """The health check's result, without its timing."""
    result = await check_model_discovery(agent)
    result.pop("duration_ms")
    return result


class TestSiblingsReportTheSameSnapshot:
    """Required test 1: a sibling reports the failure in the catalog it reads."""

    @pytest.mark.asyncio
    async def test_a_cache_hit_sibling_reports_the_failure(self, host):
        host.a.endpoints["anthropic"].error = RuntimeError("401 Unauthorized")
        await host.a.service.discover_all_models(use_cache=False)

        await host.b.service.discover_all_models(use_cache=True)

        assert host.b.endpoints["anthropic"].calls == 0, (
            "B must reach the catalog through the cache-hit path, the one that "
            "never received the failure record before #3197"
        )
        reading_b = await _reading(host.b.agent)
        assert reading_b["status"] == "warn"
        assert reading_b["details"]["failed_vendors"] == {"anthropic": AUTH_FAILURE}
        assert reading_b == await _reading(host.a.agent), (
            "every service reads the same snapshot, so reports the same state"
        )

    @pytest.mark.asyncio
    async def test_a_sibling_pinned_to_the_failing_vendor_reports_it_at_risk(
        self, host
    ):
        """The failure is the snapshot's; whether it is fatal is each agent's."""
        host.a.endpoints["anthropic"].error = RuntimeError("401 Unauthorized")
        await host.a.service.discover_all_models(use_cache=False)
        host.b.service.set_model_preference(
            "claude-opus-5", "anthropic", "api", validate=False
        )

        reading_b = await _reading(host.b.agent)

        assert reading_b["status"] == "fail"
        assert reading_b["details"]["pinned_vendor_at_risk"] is True
        assert (await _reading(host.a.agent))["status"] == "warn"

    @pytest.mark.asyncio
    async def test_a_siblings_catalog_rejection_names_the_stale_catalog(self, host):
        """The #3196 stale-catalog wording reads the same shared record."""
        host.a.endpoints["anthropic"].error = RuntimeError("401 Unauthorized")
        await host.a.service.discover_all_models(use_cache=False)

        rejection = host.b.service._catalog_rejection(
            {"vendor": "anthropic", "route": "api"}, "claude-opus-5"
        )

        assert rejection.discovery_failure == AUTH_FAILURE
        assert "stale" in str(rejection)


class TestRecoveryIsShared:
    """Required test 2: any service's successful refresh clears the reading."""

    @pytest.mark.asyncio
    async def test_a_siblings_refresh_clears_the_owners_reading(self, host):
        host.a.endpoints["anthropic"].error = RuntimeError("401 Unauthorized")
        await host.a.service.discover_all_models(use_cache=False)
        degraded = await _reading(host.a.agent)
        assert degraded["status"] == "warn"

        await host.b.service.discover_all_models(use_cache=False)

        recovered = await _reading(host.a.agent)
        assert recovered["status"] == "pass"
        assert recovered["details"]["catalog_version"] > (
            degraded["details"]["catalog_version"]
        )
        assert recovered == await _reading(host.b.agent)
        assert host.a.endpoints["anthropic"].calls == 1, (
            "A recovered without refreshing anything itself"
        )

    @pytest.mark.asyncio
    async def test_the_recovered_catalog_is_the_one_published_with_it(self, host):
        host.a.endpoints["anthropic"].error = RuntimeError("401 Unauthorized")
        await host.a.service.discover_all_models(use_cache=False)
        assert {m.provider for m in get_shared_model_cache().get_any()} == {"openai"}

        await host.b.service.discover_all_models(use_cache=False)

        snapshot = get_shared_model_cache().snapshot()
        assert {m.provider for m in snapshot.models} == {"anthropic", "openai"}
        assert snapshot.outcomes["anthropic"].status is DiscoveryStatus.OK


class TestPublicationIsAtomic:
    """Required test 3: a reader sees the old pair or the new pair, never a mix."""

    @pytest.mark.asyncio
    async def test_an_outcome_is_invisible_until_its_catalog_is_published(
        self, host
    ):
        await host.a.service.discover_all_models(use_cache=False)
        old = get_shared_model_cache().snapshot()
        old_reading = await _reading(host.a.agent)

        # A's next refresh observes the anthropic failure at once, then waits
        # on openai, which now serves a new model: the outcome exists, its
        # catalog does not yet.
        host.a.endpoints["anthropic"].error = RuntimeError("401 Unauthorized")
        host.a.endpoints["openai"].model_ids = ["gpt-6"]
        gate = asyncio.Event()
        host.a.endpoints["openai"].gate = gate
        refresh = asyncio.create_task(
            host.a.service.discover_all_models(use_cache=False)
        )
        await _until(lambda: host.a.endpoints["openai"].calls == 2)
        assert host.a.endpoints["anthropic"].calls == 2

        assert get_shared_model_cache().snapshot() is old
        assert await _reading(host.a.agent) == old_reading
        assert await _reading(host.b.agent) == old_reading

        gate.set()
        await refresh

        new = get_shared_model_cache().snapshot()
        assert new.version > old.version
        assert "gpt-6" not in {m.id for m in old.models}
        assert "gpt-6" in {m.id for m in new.models}
        assert old.discovery_failures == {}
        assert new.discovery_failures == {"anthropic": AUTH_FAILURE}
        reading = await _reading(host.b.agent)
        assert reading["details"]["catalog_version"] == new.version
        assert reading["details"]["failed_vendors"] == {"anthropic": AUTH_FAILURE}

    def test_a_held_snapshot_is_unchanged_by_a_later_publication(self):
        cache = SharedModelCache()
        cache.set(
            [_model("old", "anthropic")],
            outcomes={"anthropic": VendorDiscoveryOutcome.failed("old failure")},
        )
        held = cache.snapshot()

        cache.set(
            [_model("new", "anthropic")],
            outcomes={"anthropic": VendorDiscoveryOutcome.ok()},
        )

        assert [m.id for m in held.models] == ["old"]
        assert held.discovery_failures == {"anthropic": "old failure"}
        assert cache.snapshot() is not held

    def test_a_snapshot_cannot_be_changed_by_its_reader_or_publisher(self):
        cache = SharedModelCache()
        outcomes = {"anthropic": VendorDiscoveryOutcome.ok()}
        cache.set([_model("m", "anthropic")], outcomes=outcomes)
        snapshot = cache.snapshot()

        outcomes["anthropic"] = VendorDiscoveryOutcome.failed("late write")
        assert snapshot.discovery_failures == {}
        with pytest.raises(TypeError):
            snapshot.outcomes["anthropic"] = VendorDiscoveryOutcome.failed("x")
        with pytest.raises(AttributeError):
            snapshot.models = ()
        cache.get_any().clear()
        assert [m.id for m in cache.snapshot().models] == ["m"]

    def test_concurrent_readers_never_see_a_mixed_pair(self):
        """Every publication pairs generation ``g``'s catalog with its outcome."""
        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        cache = SharedModelCache()
        # Publish until the readers have overlapped this many publications, so
        # the test cannot pass by never racing them.
        overlapped = 50
        stop = threading.Event()
        mixed: list[tuple[str, str | None]] = []
        seen: set[int] = set()

        def publish():
            generation = 0
            try:
                while len(seen) < overlapped and generation < 200_000:
                    cache.set(
                        [_model(f"gen-{generation}", "anthropic")],
                        outcomes={
                            "anthropic": VendorDiscoveryOutcome.failed(
                                f"gen-{generation}"
                            )
                        },
                    )
                    generation += 1
            finally:
                stop.set()

        def read():
            while not stop.is_set():
                snapshot = cache.snapshot()
                if snapshot is None:
                    continue
                # A missing outcome is a mix too, and must be recorded rather
                # than raised: an exception would only end this thread.
                catalog_tag = snapshot.models[0].id
                outcome_tag = snapshot.discovery_failures.get("anthropic")
                if catalog_tag != outcome_tag:
                    mixed.append((catalog_tag, outcome_tag))
                seen.add(snapshot.version)

        try:
            readers = [threading.Thread(target=read) for _ in range(4)]
            for reader in readers:
                reader.start()
            publish()
            for reader in readers:
                reader.join(timeout=10)
        finally:
            sys.setswitchinterval(previous_interval)

        assert mixed == []
        assert len(seen) >= overlapped


class TestFailuresAreScopedToTheirVendor:
    """Required test 4: a failure for X does not mark Y degraded."""

    @pytest.mark.asyncio
    async def test_one_vendors_failure_leaves_the_other_healthy(self, host):
        host.a.endpoints["anthropic"].error = RuntimeError("401 Unauthorized")
        await host.a.service.discover_all_models(use_cache=False)

        snapshot = get_shared_model_cache().snapshot()
        assert snapshot.discovery_failures == {"anthropic": AUTH_FAILURE}
        assert snapshot.outcomes["openai"].status is DiscoveryStatus.OK
        reading = await _reading(host.b.agent)
        assert set(reading["details"]["failed_vendors"]) == {"anthropic"}
        assert "openai" not in reading["message"]

    @pytest.mark.asyncio
    async def test_a_newer_observation_replaces_an_older_failure(self, host):
        """Same vendors, newer refresh: its outcomes stand, the old ones go.
        (A vendor the newer refresh did not cover is the merge case, below.)"""
        host.a.endpoints["openai"].error = RuntimeError("503 overloaded")
        await host.a.service.discover_all_models(use_cache=False)
        assert set(get_shared_model_cache().snapshot().discovery_failures) == {
            "openai"
        }

        host.b.endpoints["anthropic"].error = RuntimeError("401 Unauthorized")
        await host.b.service.discover_all_models(use_cache=False)

        assert get_shared_model_cache().snapshot().discovery_failures == {
            "anthropic": AUTH_FAILURE
        }

    def test_a_vendor_the_newer_refresh_did_not_cover_is_not_carried(self):
        """Only a vendor a refresh covered but never requested keeps its
        failure (#3190 r10); one it did not cover at all describes nothing in
        the new catalog."""
        cache = SharedModelCache()
        cache.set([], outcomes={
            "runpod": VendorDiscoveryOutcome.failed("404 Not Found"),
            "xai": VendorDiscoveryOutcome.failed("timeout"),
        })

        cache.set([], outcomes={
            "anthropic": VendorDiscoveryOutcome.ok(),
            "xai": VendorDiscoveryOutcome.not_attempted(),
        })

        snapshot = cache.snapshot()
        assert snapshot.discovery_failures == {"xai": "timeout"}
        assert snapshot.outcomes["xai"].status is DiscoveryStatus.NOT_ATTEMPTED
        assert snapshot.outcomes["xai"].observed_version == snapshot.version - 1

    @pytest.mark.asyncio
    async def test_a_carried_failure_is_reported_as_not_reattempted(self, host):
        get_shared_model_cache().set([], outcomes={
            "anthropic": VendorDiscoveryOutcome.failed(AUTH_FAILURE),
            "openai": VendorDiscoveryOutcome.failed("503 overloaded"),
        })
        get_shared_model_cache().set([], outcomes={
            "anthropic": VendorDiscoveryOutcome.not_attempted(),
            "openai": VendorDiscoveryOutcome.failed("503 overloaded"),
        })

        reading = await _reading(host.a.agent)

        assert reading["details"]["failed_vendors"] == {
            "anthropic": AUTH_FAILURE,
            "openai": "503 overloaded",
        }
        assert reading["details"]["not_reattempted_vendors"] == ["anthropic"]


class TestVersions:
    def test_every_publication_has_a_new_version_across_clear(self):
        cache = SharedModelCache()
        cache.set_stale([_model("disk", "openai")])
        first = cache.snapshot().version
        cache.set([_model("fresh", "openai")])
        second = cache.snapshot().version
        cache.clear()
        cleared = cache.snapshot()
        assert cleared.models is None
        assert cache.get_any() is None and not cache.has_data()
        cache.set([_model("again", "openai")])

        assert first < second < cleared.version < cache.snapshot().version



class TestTheCarryOutlivesCatalogInvalidation:
    """A skipped request must not turn health green (#3190 r10), and neither
    must anything else that is not an observation of the vendor.

    ``clear()`` runs on routine events (an Ollama pull or delete, a provider
    added at finalize), so a failure that did not survive it would be erased
    by unrelated work, and the next skipped request would report the vendor
    healthy having fetched nothing.
    """

    def test_a_failure_is_carried_across_clear(self):
        cache = SharedModelCache()
        cache.set([], outcomes={"runpod": VendorDiscoveryOutcome.failed("404")})
        failed = cache.snapshot().version
        cache.clear()
        cache.set([], outcomes={"runpod": VendorDiscoveryOutcome.not_attempted()})

        assert cache.snapshot().outcomes["runpod"] == VendorDiscoveryOutcome(
            DiscoveryStatus.NOT_ATTEMPTED, error="404", observed_version=failed,
        )

    def test_a_catalog_published_without_a_refresh_carries_it(self):
        """A disk restore or a bare ``set(models)`` observed no vendor."""
        cache = SharedModelCache()
        cache.set([], outcomes={"runpod": VendorDiscoveryOutcome.failed("404")})
        failed = cache.snapshot().version
        carried = VendorDiscoveryOutcome(
            DiscoveryStatus.NOT_ATTEMPTED, error="404", observed_version=failed,
        )
        cache.clear()
        assert cache.snapshot().outcomes == {"runpod": carried}
        cache.set_stale([_model("disk", "openai")])
        assert [m.id for m in cache.get_any()] == ["disk"], (
            "an invalidated catalog must not block the disk restore"
        )
        assert cache.snapshot().outcomes == {"runpod": carried}
        cache.set([_model("seeded", "openai")])
        assert cache.snapshot().outcomes == {"runpod": carried}

        cache.set([], outcomes={"runpod": VendorDiscoveryOutcome.not_attempted()})

        assert cache.snapshot().outcomes == {"runpod": carried}

    def test_an_observed_success_still_clears_it(self):
        cache = SharedModelCache()
        cache.set([], outcomes={"runpod": VendorDiscoveryOutcome.failed("404")})
        cache.clear()
        cache.set([], outcomes={"runpod": VendorDiscoveryOutcome.ok()})
        cache.set([], outcomes={"runpod": VendorDiscoveryOutcome.not_attempted()})

        assert cache.snapshot().discovery_failures == {}

    @pytest.mark.asyncio
    async def test_health_keeps_reporting_it_while_the_catalog_is_invalidated(
        self, host
    ):
        host.a.endpoints["anthropic"].error = RuntimeError("401 Unauthorized")
        await host.a.service.discover_all_models(use_cache=False)
        before = await _reading(host.a.agent)

        get_shared_model_cache().clear()
        invalidated = await _reading(host.b.agent)
        get_shared_model_cache().set_stale([_model("claude-opus-5", "anthropic")])
        restored = await _reading(host.b.agent)

        for reading in (invalidated, restored):
            assert reading["status"] == "warn"
            assert reading["details"]["failed_vendors"] == {"anthropic": AUTH_FAILURE}
            assert reading["details"]["not_reattempted_vendors"] == ["anthropic"]
        assert before["details"]["not_reattempted_vendors"] == []

    @pytest.mark.asyncio
    async def test_a_disk_restored_catalog_reports_no_discovery_rather_than_health(
        self, host
    ):
        get_shared_model_cache().set_stale([_model("claude-opus-5", "anthropic")])

        reading = await _reading(host.a.agent)

        assert reading["status"] == "pass"
        assert reading["message"] == "No vendor's model catalog has been fetched"
