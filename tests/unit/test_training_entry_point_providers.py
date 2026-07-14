"""
Tests for entry-point-based TrainingProvider discovery (#2445).

External packages register generation/training backends under the
``kestrel_sovereign.training_providers`` entry-point group. These tests verify
the factory discovers them, honors declared priority, and skips providers that
report themselves unavailable — without any changes to the built-in providers.
"""

import importlib.metadata

import pytest

from kestrel_sovereign.features.training.factory import (
    TRAINING_PROVIDER_ENTRY_POINT_GROUP,
    TrainingProviderFactory,
)
from kestrel_sovereign.features.training.protocol import TrainingProvider
from kestrel_sovereign.features.training.types import ProviderCapabilities, ProviderType


class _StubProvider:
    """Minimal available provider registered via entry point.

    Implements the full TrainingProvider surface for both training and
    generation (per codex round-2 on #2445: providers advertising a
    capability MUST implement the methods that back it — the factory
    rejects violators at discovery).
    """

    # Slots between local_mps (0) and runpod (10) in the built-in ordering.
    priority = 5
    capabilities = ProviderCapabilities(
        training=True,
        generation=True,
        uncensored=True,
        flux_version="2.x",
        supports_lora_download=True,
    )

    provider_name = "stub"
    provider_type = ProviderType.SERVERLESS

    def is_available(self) -> bool:
        return True

    async def start_training(self, *_args, **_kw): ...
    async def get_status(self, *_args, **_kw): ...
    async def download_weights(self, *_args, **_kw): ...
    async def cancel(self, *_args, **_kw): ...
    async def cleanup(self, *_args, **_kw): ...
    async def generate_image(self, *_args, **_kw): ...


class _UnavailableProvider:
    """Provider whose is_available() is False — must be silently skipped."""

    provider_name = "stub_unavailable"
    provider_type = ProviderType.SERVERLESS

    def is_available(self) -> bool:
        return False


class _GenerationOnlyProvider:
    """Available provider that generates but cannot train.

    Highest priority so it would win a naive priority loop — the factory must
    still refuse to route it into a training flow.
    """

    priority = 0
    capabilities = ProviderCapabilities(
        training=False,
        generation=True,
        uncensored=True,
        supports_lora_download=False,
    )

    provider_name = "gen_only"
    provider_type = ProviderType.SERVERLESS

    def is_available(self) -> bool:
        return True

    async def generate_image(self, *_args, **_kw): ...


class _TrainingOnlyUncensoredProvider:
    """Available uncensored provider that cannot generate images.

    Highest priority so it would win a naive uncensored loop — but
    get_uncensored_provider() promises a *generation* provider, so it must be
    skipped.
    """

    priority = 0
    capabilities = ProviderCapabilities(
        training=True,
        generation=False,
        uncensored=True,
        supports_lora_download=True,
    )

    provider_name = "train_only"
    provider_type = ProviderType.SERVERLESS

    def is_available(self) -> bool:
        return True

    async def start_training(self, *_args, **_kw): ...
    async def get_status(self, *_args, **_kw): ...
    async def download_weights(self, *_args, **_kw): ...
    async def cancel(self, *_args, **_kw): ...
    async def cleanup(self, *_args, **_kw): ...


class _MissingTrainingMethodsProvider:
    """Declares training=True but doesn't implement start_training et al.
    Must be REJECTED at discovery (codex round-2 P2 on #2445)."""

    priority = 5
    capabilities = ProviderCapabilities(training=True, generation=False)
    provider_name = "bad_train"
    provider_type = ProviderType.SERVERLESS

    def is_available(self) -> bool:
        return True


class _MissingGenerationMethodProvider:
    """Declares generation=True but doesn't implement generate_image.
    Must be REJECTED at discovery (codex round-2 P1 on #2445)."""

    priority = 5
    capabilities = ProviderCapabilities(training=False, generation=True)
    provider_name = "bad_gen"
    provider_type = ProviderType.SERVERLESS

    def is_available(self) -> bool:
        return True


class _RaisingAvailabilityProvider:
    """is_available() raises. Must NOT crash the factory — treated as
    unavailable and logged (codex round-2 P1 on #2445)."""

    priority = 0  # would win priority ordering if not for the raise
    capabilities = ProviderCapabilities(
        training=True, generation=True, uncensored=True,
    )
    provider_name = "raising"
    provider_type = ProviderType.SERVERLESS

    def is_available(self) -> bool:
        raise ConnectionError("probe failed")

    async def start_training(self, *_args, **_kw): ...
    async def get_status(self, *_args, **_kw): ...
    async def download_weights(self, *_args, **_kw): ...
    async def cancel(self, *_args, **_kw): ...
    async def cleanup(self, *_args, **_kw): ...
    async def generate_image(self, *_args, **_kw): ...


class _BehavioralOnlyProvider:
    """Implements every TrainingProvider method but declares no routing metadata.

    Mirrors the built-in adapters, which set neither ``priority`` nor
    ``capabilities`` — they must still satisfy the runtime-checkable protocol.
    """

    provider_name = "behavioral"
    provider_type = ProviderType.SERVERLESS

    def is_available(self) -> bool:
        return True

    async def start_training(self, companion_id, avatar_data, config=None):
        ...

    async def get_status(self, job_id):
        ...

    async def download_weights(self, job_id):
        ...

    async def cancel(self, job_id):
        ...

    async def cleanup(self, job_id):
        ...


class _FakeEntryPoint:
    def __init__(self, name, cls):
        self.name = name
        self.value = f"{cls.__module__}:{cls.__name__}"
        self._cls = cls

    def load(self):
        return self._cls


class _FakeEntryPoints:
    """Wraps the real EntryPoints, injecting our stub providers for our group.

    Every other group/caller (e.g. torch's own import-time entry-point lookup)
    is delegated to the real result, so the shim can't break unrelated code.
    """

    def __init__(self, real, eps):
        self._real = real
        self._eps = eps

    def select(self, group=None, **kwargs):
        if group == TRAINING_PROVIDER_ENTRY_POINT_GROUP:
            return list(self._eps)
        return self._real.select(group=group, **kwargs)

    def __iter__(self):
        return iter(self._real)

    def __getattr__(self, item):
        # Proxy anything else (e.g. .groups, .names) to the real object.
        return getattr(self._real, item)


@pytest.fixture
def register_entry_points(monkeypatch):
    """Register the given (name, class) providers as discoverable entry points."""

    real_entry_points = importlib.metadata.entry_points

    def _register(*pairs):
        eps = [_FakeEntryPoint(name, cls) for name, cls in pairs]

        def _fake(*args, **kwargs):
            real = real_entry_points(*args, **kwargs)
            group = kwargs.get("group")
            # Preserve legacy call shape: entry_points(group=...) returns the
            # selected sequence directly; our stubs only apply to our group.
            if group == TRAINING_PROVIDER_ENTRY_POINT_GROUP:
                return list(eps)
            if group is not None:
                return real
            return _FakeEntryPoints(real, eps)

        monkeypatch.setattr(importlib.metadata, "entry_points", _fake)
        # Reset lazy discovery so the new registration is picked up.
        TrainingProviderFactory.clear_cache()

    yield _register

    # Clean discovery state so a leaked registration never bleeds into other tests.
    TrainingProviderFactory.clear_cache()


def test_get_provider_returns_entry_point_provider(register_entry_points):
    register_entry_points(("stub", _StubProvider))

    provider = TrainingProviderFactory.get_provider("stub")
    assert isinstance(provider, _StubProvider)

    assert "stub" in TrainingProviderFactory.list_available_providers()


def test_entry_point_capabilities_are_exposed(register_entry_points):
    register_entry_points(("stub", _StubProvider))

    caps = TrainingProviderFactory.get_capabilities("stub")
    assert caps is not None
    assert caps.generation is True
    assert caps.uncensored is True


def test_priority_interleaves_builtins_and_entry_points(register_entry_points):
    register_entry_points(("stub", _StubProvider))

    order = TrainingProviderFactory._effective_priority()

    # Built-in ordering is preserved...
    assert order.index("local_mps") < order.index("runpod")
    assert order.index("runpod") < order.index("vertex_ai")
    # ...and the stub (priority=5) lands between local_mps (0) and runpod (10).
    assert order.index("local_mps") < order.index("stub") < order.index("runpod")


def test_undeclared_priority_sorts_after_builtins(register_entry_points):
    register_entry_points(("stub_unavailable", _UnavailableProvider))

    order = TrainingProviderFactory._effective_priority()

    # No declared priority => sorts after every built-in.
    for builtin in TrainingProviderFactory.PROVIDER_PRIORITY:
        assert order.index(builtin) < order.index("stub_unavailable")


def test_unavailable_entry_point_provider_is_skipped(register_entry_points):
    register_entry_points(("stub_unavailable", _UnavailableProvider))

    assert TrainingProviderFactory.get_provider("stub_unavailable") is None
    assert "stub_unavailable" not in TrainingProviderFactory.list_available_providers()


def test_entry_point_provider_does_not_shadow_builtin(register_entry_points):
    # An entry point that collides with a built-in name is ignored, not merged.
    register_entry_points(("local_mps", _StubProvider))

    TrainingProviderFactory._ensure_entry_points_loaded()
    assert "local_mps" not in TrainingProviderFactory._ep_provider_classes


def test_default_provider_skips_generation_only(register_entry_points, monkeypatch):
    # A generation-only backend must never be selected for a training flow,
    # even at the highest priority.
    monkeypatch.delenv("GENERATION_PROVIDER", raising=False)
    register_entry_points(("gen_only", _GenerationOnlyProvider))

    # It IS an available provider...
    assert "gen_only" in TrainingProviderFactory.list_available_providers()

    # ...but get_default_provider() (training) must not return it.
    provider = TrainingProviderFactory.get_default_provider()
    assert not isinstance(provider, _GenerationOnlyProvider)


def test_forced_generation_only_rejected_for_training(register_entry_points, monkeypatch):
    register_entry_points(("gen_only", _GenerationOnlyProvider))
    monkeypatch.setenv("GENERATION_PROVIDER", "gen_only")

    # Forcing a generation-only provider for the training default is refused.
    provider = TrainingProviderFactory.get_default_provider()
    assert not isinstance(provider, _GenerationOnlyProvider)


def test_uncensored_provider_requires_generation(register_entry_points):
    # A training-only uncensored provider must not be returned as an uncensored
    # *generation* provider.
    register_entry_points(("train_only", _TrainingOnlyUncensoredProvider))

    provider = TrainingProviderFactory.get_uncensored_provider()
    assert not isinstance(provider, _TrainingOnlyUncensoredProvider)


def test_behavioral_provider_satisfies_runtime_protocol():
    # Regression: routing metadata (priority/capabilities) is intentionally NOT
    # a member of the runtime-checkable protocol, so a provider that implements
    # only the behavioral methods still passes isinstance() — as the built-in
    # adapters (which declare no such attrs) must.
    assert isinstance(_BehavioralOnlyProvider(), TrainingProvider)


# ---------------------------------------------------------------------------
# Codex round-2 regressions on #2445
# ---------------------------------------------------------------------------

def test_missing_training_methods_provider_rejected_at_discovery(
    register_entry_points, caplog,
):
    # Codex round-2 P2 on #2445: a plugin declaring training=True but missing
    # start_training / get_status / etc. must be dropped at discovery. Left in,
    # it would win priority routing then crash with AttributeError at the
    # first training call.
    register_entry_points(("bad_train", _MissingTrainingMethodsProvider))
    with caplog.at_level("WARNING"):
        available = TrainingProviderFactory.list_available_providers()
    assert "bad_train" not in available
    assert any("bad_train" in r.message and "missing" in r.message.lower()
               for r in caplog.records)


def test_missing_generation_method_provider_rejected_at_discovery(
    register_entry_points, caplog,
):
    # Codex round-2 P1 on #2445: a plugin declaring generation=True but
    # missing generate_image would be selected by get_generation_provider and
    # crash with AttributeError at call time. Must be dropped at discovery.
    register_entry_points(("bad_gen", _MissingGenerationMethodProvider))
    with caplog.at_level("WARNING"):
        available = TrainingProviderFactory.list_available_providers()
    assert "bad_gen" not in available
    assert any("bad_gen" in r.message and "missing" in r.message.lower()
               for r in caplog.records)


def test_raising_is_available_does_not_crash_factory(
    register_entry_points, caplog,
):
    # Codex round-2 P1 on #2445: an entry-point provider whose is_available()
    # raises must NOT propagate up through get_provider / default routing /
    # list_available_providers — one broken plugin bricking the factory is
    # exactly the anti-pattern the entry-point contract is meant to avoid.
    register_entry_points(("raising", _RaisingAvailabilityProvider))
    with caplog.at_level("WARNING"):
        # Must not raise; must not select the broken provider.
        provider = TrainingProviderFactory.get_provider("raising")
    assert provider is None
    assert any("raising" in r.message and "is_available" in r.message
               for r in caplog.records)
    # Also should not shortcircuit the default-provider fallback chain.
    # (No other provider is registered, so we expect None or a built-in
    # depending on the env — the key assertion is "does not raise".)
    TrainingProviderFactory.get_default_provider()
