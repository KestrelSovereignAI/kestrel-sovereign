"""A vendor-wide persisted pin must survive a discovery outage at use time (#3196).

#3190 stopped the boot loader discarding a persisted pin when discovery returns
a reduced catalog, so the pin now reaches the point of use. A route-qualified
pin was already trusted there: it narrows to one route, so it is an explicit
selection and ``_try_single_provider`` skips the catalog guard for it.

A vendor-wide pin (``route: null``) on a vendor with several routes is not
explicit, and correctly so: it does not narrow to one route. But the catalog
guard then rejected every one of its routes before any provider call, so the
reduced catalog the 2026-08-31 outage produced still failed every
non-streaming request, while the routes served the model perfectly well.

The pin is the agent's own accepted decision, so it is trusted at the point of
use independently of explicitness. Selections that are not the persisted
mandate keep the guard, and when it rejects against a catalog whose discovery
is recorded as failed, the error says the catalog is stale and names
discovery instead of saying the model is unavailable.

Every test drives the real ``LLMService`` routing and dispatch, and installs
the pin through the real boot loader (``apply_persisted_model_preference``).
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from kestrel_sovereign.agent.model_preference import apply_persisted_model_preference
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.error_handling import (
    LLMAllProvidersFailedError,
    LLMProviderError,
)
from kestrel_sovereign.llm.model_cache import (
    SharedModelCache,
    VendorDiscoveryOutcome,
)
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from kestrel_sovereign.llm.output_ceiling import attach_stop_reason
from kestrel_sovereign.llm.service import LLMService, ModelNotAvailableForRoute
from kestrel_sovereign.llm.streaming import LLMStreamingError

PINNED = "claude-opus-5"
AGENT_ID = "did:test:vendor-wide-pin"

# The catalog the 2026-08-31 outage left: one stale anthropic entry, and the
# pinned model missing from it.
_REDUCED_CATALOG = [
    ModelInfo(
        id="claude-sonnet-5", provider="anthropic", display_name="claude-sonnet-5",
        category=ModelCategory.CHAT, supports_tools=True,
    ),
    ModelInfo(
        id="gpt-5", provider="openai", display_name="gpt-5",
        category=ModelCategory.CHAT, supports_tools=True,
    ),
    ModelInfo(
        id="kimi.gguf", provider="llama_cpp", display_name="kimi.gguf",
        category=ModelCategory.CHAT, supports_tools=False,
    ),
]

_DISCOVERY_ERROR = RuntimeError("401 Unauthorized")


class _RecordingAdapter:
    """One route's adapter: records every model it was asked for."""

    def __init__(self, name, attempts, *, fail_with=None):
        self.name = name
        self.attempts = attempts
        self.fail_with = fail_with

    def create_messages(self, user_prompt=None, system_prompt=None, **_):
        return [{"role": "user", "content": user_prompt or ""}]

    async def get_response(self, *, client, model, messages, **_):
        self.attempts.append((self.name, model))
        if self.fail_with is not None:
            raise self.fail_with
        return attach_stop_reason(
            LLMResponse(
                content=f"{self.name} answered as {model}",
                input_tokens=1, output_tokens=1, total_tokens=2,
            ),
            "stop",
        )

    async def get_streaming_response(self, *, client, model, messages, **_):
        self.attempts.append((self.name, model))
        if self.fail_with is not None:
            raise self.fail_with
        yield f"{self.name} streamed as {model}"


def _route(name, model, adapter, *, is_local=False):
    vendor, route = name.split(":", 1)
    return SimpleNamespace(
        name=name, vendor=vendor, route=route, client=AsyncMock(),
        adapter=adapter, model=model, is_cloud=not is_local,
        is_local=is_local, base_url=None, selection_hints=[],
    )


@pytest_asyncio.fixture
async def outage():
    """A real service mid-outage: two anthropic routes, the catalog reduced,
    and anthropic discovery recorded as failed."""
    attempts = []
    adapters = {
        "anthropic:plan": _RecordingAdapter("anthropic:plan", attempts),
        "anthropic:api": _RecordingAdapter("anthropic:api", attempts),
        "openai:api": _RecordingAdapter("openai:api", attempts),
        "llama_cpp:local": _RecordingAdapter("llama_cpp:local", attempts),
    }
    registry = Mock()
    registry.initialize_providers = Mock(return_value=[
        # Both anthropic routes keep a concrete configured default that is NOT
        # the pin, so the "route's own model is always available" escape in
        # the guard does not apply.
        _route("anthropic:plan", "claude-sonnet-5", adapters["anthropic:plan"]),
        _route("anthropic:api", "claude-sonnet-5", adapters["anthropic:api"]),
        _route("openai:api", "gpt-5", adapters["openai:api"]),
        _route(
            "llama_cpp:local", "kimi.gguf", adapters["llama_cpp:local"],
            is_local=True,
        ),
    ])
    mandate_config = {"defaults": {"preferred": "", "banned": []}, "mandates": {}}
    with patch("kestrel_sovereign.llm.service.load_config", return_value=mandate_config), \
         patch("kestrel_sovereign.llm.service.load_section", return_value={
             "route_priority": [
                 "anthropic:plan", "anthropic:api", "openai:api", "llama_cpp:local",
             ],
         }), \
         patch("kestrel_sovereign.llm.service.ProviderRegistry", return_value=registry):
        service = LLMService()
    service._usage_db = None
    service._db_initialized = False

    # The reduced catalog, published with the anthropic discovery failure
    # that produced it: the two are one snapshot (#3197).
    catalog = SharedModelCache()
    catalog.set(_REDUCED_CATALOG, outcomes={
        "anthropic": VendorDiscoveryOutcome.failed(
            f"{type(_DISCOVERY_ERROR).__name__}: {_DISCOVERY_ERROR}"
        ),
    })
    with patch(
        "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
        return_value=catalog,
    ):
        yield SimpleNamespace(
            service=service, attempts=attempts, adapters=adapters, catalog=catalog,
        )
    await service.close()


async def _replay_persisted(service, pref):
    """Install ``pref`` exactly as boot does, from an agent_metadata row."""
    db = Mock()
    db.fetchall = AsyncMock(return_value=[(json.dumps(pref),)])
    failure = await apply_persisted_model_preference(service, db, AGENT_ID)
    assert failure is None, f"the persisted pin did not apply: {failure}"
    assert service._mandate_preference == {
        "vendor": pref["vendor"], "model": pref["model"], "route": pref["route"],
    }


VENDOR_WIDE_PIN = {"vendor": "anthropic", "model": PINNED, "route": None}
ROUTE_QUALIFIED_PIN = {"vendor": "anthropic", "model": PINNED, "route": "plan"}


async def _collect(stream):
    return [chunk async for chunk in stream]


def _test_prompt():
    return {"system_prompt": "You are a test agent.", "user_prompt": "Say hello."}


class TestPersistedVendorWidePinIsTrustedAtUse:
    """The issue's reproduction."""

    @pytest.mark.asyncio
    async def test_a_non_streaming_request_reaches_the_provider(self, outage):
        await _replay_persisted(outage.service, VENDOR_WIDE_PIN)

        result = await outage.service.get_response(**_test_prompt())

        assert result == f"anthropic:plan answered as {PINNED}"
        assert outage.attempts == [("anthropic:plan", PINNED)]

    @pytest.mark.asyncio
    async def test_a_streaming_request_reaches_the_provider(self, outage):
        await _replay_persisted(outage.service, VENDOR_WIDE_PIN)

        chunks = await _collect(outage.service.get_streaming_response(**_test_prompt()))

        assert chunks == [f"anthropic:plan streamed as {PINNED}"]
        assert outage.attempts == [("anthropic:plan", PINNED)]

    @pytest.mark.asyncio
    async def test_each_route_of_the_vendor_is_tried_in_order(self, outage):
        """Trust does not make the pin explicit: the same-vendor fallback a
        vendor-wide pin exists for still runs, without the guard."""
        outage.adapters["anthropic:plan"].fail_with = LLMProviderError(
            "anthropic:plan", "503 overloaded"
        )
        await _replay_persisted(outage.service, VENDOR_WIDE_PIN)

        result = await outage.service.get_response(**_test_prompt())

        assert result == f"anthropic:api answered as {PINNED}"
        assert outage.attempts == [
            ("anthropic:plan", PINNED),
            ("anthropic:api", PINNED),
        ]

    @pytest.mark.asyncio
    async def test_a_model_that_is_really_gone_surfaces_the_providers_error(
        self, outage
    ):
        """Not the guard's. And the pin never crosses to another vendor."""
        for name in ("anthropic:plan", "anthropic:api"):
            outage.adapters[name].fail_with = LLMProviderError(
                name, f"404 model {PINNED} not found"
            )
        await _replay_persisted(outage.service, VENDOR_WIDE_PIN)

        with pytest.raises(LLMAllProvidersFailedError) as caught:
            await outage.service.get_response(**_test_prompt())

        assert set(caught.value.errors) == {"anthropic:plan", "anthropic:api"}
        for error in caught.value.errors.values():
            assert isinstance(error, LLMProviderError)
            assert not isinstance(error, ModelNotAvailableForRoute)
            assert "not found" in str(error)
        assert outage.attempts == [
            ("anthropic:plan", PINNED),
            ("anthropic:api", PINNED),
        ]


class TestNonPersistedSelectionsKeepTheGuard:
    """A per-call selection is not the agent's decision: it stays guarded."""

    @pytest.mark.asyncio
    async def test_a_vendor_wide_override_is_rejected_naming_the_stale_catalog(
        self, outage
    ):
        with pytest.raises(LLMAllProvidersFailedError) as caught:
            await outage.service.get_response(
                **_test_prompt(), model_override=f"anthropic/{PINNED}"
            )

        assert outage.attempts == [], "the guard must reject before any call"
        assert set(caught.value.errors) == {"anthropic:plan", "anthropic:api"}
        for error in caught.value.errors.values():
            assert isinstance(error, ModelNotAvailableForRoute)
            assert error.discovery_failure is not None
        message = str(caught.value)
        assert "stale" in message
        assert "model discovery for 'anthropic' failed" in message
        assert "401 Unauthorized" in message
        assert "is not available for route" not in message, (
            "a stale catalog must not be reported as the model being unavailable"
        )

    @pytest.mark.asyncio
    async def test_without_a_discovery_failure_the_message_is_unchanged(
        self, outage
    ):
        """The stale wording is earned by a recorded failure, not assumed."""
        outage.catalog.set(
            _REDUCED_CATALOG, outcomes={"anthropic": VendorDiscoveryOutcome.ok()}
        )

        with pytest.raises(LLMAllProvidersFailedError) as caught:
            await outage.service.get_response(
                **_test_prompt(), model_override=f"anthropic/{PINNED}"
            )

        assert outage.attempts == []
        message = str(caught.value)
        assert "is not available for route 'anthropic:plan'" in message
        assert "stale" not in message
        for error in caught.value.errors.values():
            assert error.discovery_failure is None

    @pytest.mark.asyncio
    async def test_a_streaming_override_has_no_catalog_guard_to_reject_it(
        self, outage
    ):
        """Streaming never consulted the catalog (#2352) and still does not.

        Recorded so a reader does not mistake the guard for one this change
        added or removed on the streaming path: it is unchanged there.
        """
        chunks = await _collect(outage.service.get_streaming_response(
            **_test_prompt(), model_override=f"anthropic/{PINNED}"
        ))

        assert chunks == [f"anthropic:plan streamed as {PINNED}"]

    @pytest.mark.asyncio
    async def test_a_local_model_ignoring_pin_is_still_guarded(self, outage):
        """`llama_cpp` serves whatever is loaded: trusting the pin there would
        meter the loaded model as the absent one, so the guard stays."""
        outage.service.set_model_preference(
            "some-other.gguf", "llama_cpp", None, validate=False
        )

        with pytest.raises(LLMAllProvidersFailedError) as caught:
            await outage.service.get_response(**_test_prompt())

        assert outage.attempts == []
        assert isinstance(
            caught.value.errors["llama_cpp:local"], ModelNotAvailableForRoute
        )


class TestTheAuditStillFailsClosedButNamesTheStaleCatalog:
    """The audit keeps its guard unconditionally (#3190 r9): it must not vouch
    for a route it cannot verify. So during an outage a pinned turn generates
    and then fails its audit, as a route-qualified pin already did. What it
    must not do is blame the model."""

    @pytest.mark.asyncio
    async def test_a_stale_catalog_is_named_and_the_audit_fails_closed(self, outage):
        await _replay_persisted(outage.service, VENDOR_WIDE_PIN)

        verdict = await outage.service.get_audit_response(
            "A reasonably long agent response under audit."
        )

        assert verdict["risk_level"] == 3
        assert verdict["audited"] is False
        assert outage.attempts == [], "the audit guard must still reject first"
        assert "stale" in verdict["reasoning"]
        assert "model discovery for 'anthropic' failed" in verdict["reasoning"]
        assert "not available for route" not in verdict["reasoning"]

    @pytest.mark.asyncio
    async def test_without_a_discovery_failure_the_reason_is_unchanged(self, outage):
        outage.catalog.set(
            _REDUCED_CATALOG, outcomes={"anthropic": VendorDiscoveryOutcome.ok()}
        )
        await _replay_persisted(outage.service, VENDOR_WIDE_PIN)

        verdict = await outage.service.get_audit_response(
            "A reasonably long agent response under audit."
        )

        assert verdict["risk_level"] == 3
        assert verdict["reasoning"] == (
            "Audit provider failed: "
            f"anthropic:plan: target model {PINNED} not available for route; "
            f"anthropic:api: target model {PINNED} not available for route"
        )


class TestRouteQualifiedPinIsUnchanged:
    """Regression guard: already explicit before #3196, and still trusted."""

    @pytest.mark.asyncio
    async def test_non_streaming(self, outage):
        await _replay_persisted(outage.service, ROUTE_QUALIFIED_PIN)

        result = await outage.service.get_response(**_test_prompt())

        assert result == f"anthropic:plan answered as {PINNED}"
        assert outage.attempts == [("anthropic:plan", PINNED)]

    @pytest.mark.asyncio
    async def test_streaming(self, outage):
        await _replay_persisted(outage.service, ROUTE_QUALIFIED_PIN)

        chunks = await _collect(outage.service.get_streaming_response(**_test_prompt()))

        assert chunks == [f"anthropic:plan streamed as {PINNED}"]
        assert outage.attempts == [("anthropic:plan", PINNED)]

    @pytest.mark.asyncio
    async def test_a_failing_route_qualified_pin_does_not_fall_to_a_sibling(
        self, outage
    ):
        outage.adapters["anthropic:plan"].fail_with = LLMProviderError(
            "anthropic:plan", "503 overloaded"
        )
        await _replay_persisted(outage.service, ROUTE_QUALIFIED_PIN)

        with pytest.raises(LLMStreamingError, match="Selected route anthropic:plan"):
            await _collect(outage.service.get_streaming_response(**_test_prompt()))
        with pytest.raises(LLMAllProvidersFailedError):
            await outage.service.get_response(**_test_prompt())

        assert [name for name, _ in outage.attempts] == [
            "anthropic:plan", "anthropic:plan",
        ]


class TestMandatePinIsSetByTheBranchThatChoseTheRoutes:
    """`resolve_provider_routing` records the branch it took; nothing re-derives it."""

    @pytest.mark.asyncio
    async def test_the_persisted_mandate_sets_it(self, outage):
        await _replay_persisted(outage.service, VENDOR_WIDE_PIN)
        resolution = outage.service.resolve_provider_routing()
        assert resolution.mandate_pin is True
        assert resolution.meta.explicit_selection is False, (
            "a vendor-wide pin over two routes must stay non-explicit"
        )

    @pytest.mark.asyncio
    async def test_an_override_does_not_set_it_even_over_a_mandate(self, outage):
        await _replay_persisted(outage.service, VENDOR_WIDE_PIN)
        resolution = outage.service.resolve_provider_routing(
            model_override=f"anthropic/{PINNED}"
        )
        assert resolution.mandate_pin is False

    @pytest.mark.asyncio
    async def test_no_mandate_does_not_set_it(self, outage):
        assert outage.service.resolve_provider_routing().mandate_pin is False

    @pytest.mark.asyncio
    async def test_mandate_fallbacks_do_not_set_it(self, outage):
        """An unmatched mandate routes to its fallbacks: not the pin itself."""
        outage.service.set_model_preference(
            "some-model", "absent-vendor", None, validate=False
        )
        outage.service._mandate_fallbacks = [{"vendor": "openai", "model": "gpt-5"}]
        resolution = outage.service.resolve_provider_routing()
        assert [p["name"] for p in resolution.providers] == ["openai:api"]
        assert resolution.mandate_pin is False
