"""Pre-boot discovery and execution of SDK setup-step contributions."""

import pytest
from kestrel_sdk.features import (
    SetupFlow,
    SetupStepClassification,
    SetupStepRegistration,
)

from kestrel_sovereign.feature_registry import FeaturePackageInfo, PackageBoundary
from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.contributions import (
    SETUP_STEP_ENTRY_POINT_GROUP,
    SetupContributionDiscoveryError,
    discover_setup_steps,
    missing_setup_step_message,
)
from kestrel_sovereign.setup.prompts import NonInteractivePrompter
from kestrel_sovereign.setup.wizard import run_wizard


class _EntryPoint:
    def __init__(self, name, value, loaded):
        self.name = name
        self.value = value
        self._loaded = loaded
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        if isinstance(self._loaded, BaseException):
            raise self._loaded
        return self._loaded


def _registration(
    name,
    step=lambda ctx: None,
    *,
    classification=SetupStepClassification.OPTIONAL,
    order=1000,
    before=(),
    after=(),
):
    return SetupStepRegistration(
        owner=f"tests:{name}",
        name=name,
        step=step,
        classification=classification,
        order=order,
        before=before,
        after=after,
    )


def _context(tmp_path, flow=Flow.QUICKSTART):
    return SetupContext(
        project_dir=tmp_path,
        agent_data_root=tmp_path / "agent_data",
        flow=flow,
        prompter=NonInteractivePrompter(),
    )


def test_discovery_combines_core_and_contributed_steps_in_sdk_order():
    between = _registration(
        "between-keys-and-llm",
        classification=SetupStepClassification.DEFAULT,
        before=("llm",),
        after=("keys",),
    )
    late = _registration("late", order=2000, after=("verify",))
    discovered = discover_setup_steps(
        [_EntryPoint("fixture", "fixture.setup:steps", lambda: (late, between))]
    )

    names = [registration.name for registration in discovered.registry.ordered()]
    assert names.index("keys") < names.index("between-keys-and-llm")
    assert names.index("between-keys-and-llm") < names.index("llm")
    assert names.index("verify") < names.index("late")
    assert discovered.selected("fixture") == (between, late)


@pytest.mark.parametrize(
    "registrations, match",
    [
        ((_registration("unknown", after=("absent",)),), "unknown step"),
        (
            (
                _registration("cycle-a", after=("cycle-b",)),
                _registration("cycle-b", after=("cycle-a",)),
            ),
            "cycle",
        ),
        (
            (_registration("duplicate"), _registration("duplicate")),
            "duplicate setup step",
        ),
    ],
)
def test_invalid_ordering_unknown_references_cycles_and_duplicates_fail_closed(
    registrations, match
):
    with pytest.raises(SetupContributionDiscoveryError, match=match):
        discover_setup_steps(
            [_EntryPoint("fixture", "fixture.setup:steps", registrations)]
        )


def test_contributed_default_cannot_order_before_core_key_custody():
    before_keys = _registration(
        "before-keys",
        classification=SetupStepClassification.DEFAULT,
        order=-1,
        before=("keys",),
    )

    with pytest.raises(
        SetupContributionDiscoveryError,
        match="before the core 'keys' custody boundary",
    ):
        discover_setup_steps(
            [_EntryPoint("fixture", "fixture.setup:steps", (before_keys,))]
        )


def test_duplicate_provider_slugs_fail_closed():
    with pytest.raises(SetupContributionDiscoveryError, match="duplicate.*slug"):
        discover_setup_steps(
            [
                _EntryPoint("fixture", "a:steps", ()),
                _EntryPoint("fixture", "b:steps", ()),
            ]
        )


def test_sync_and_async_steps_execute_with_sdk_context(tmp_path):
    observed = []

    def sync_step(ctx):
        observed.append(("sync", ctx.flow, ctx.project_dir))

    async def async_step(ctx):
        observed.append(("async", ctx.flow, ctx.project_dir))

    discovered = discover_setup_steps(
        [
            _EntryPoint(
                "fixture",
                "fixture.setup:steps",
                (
                    _registration("sync-step", sync_step),
                    _registration("async-step", async_step),
                ),
            )
        ]
    )
    ctx = _context(tmp_path)

    assert run_wizard(ctx, only_step="fixture", setup_steps=discovered) == 0
    assert observed == [
        ("async", SetupFlow.SETUP, tmp_path),
        ("sync", SetupFlow.SETUP, tmp_path),
    ]


def test_default_flow_skips_optional_contributions_and_explicit_selection_runs_them(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.setup import steps as core_steps

    monkeypatch.setattr(core_steps, "ORDERED", ())
    monkeypatch.setattr(core_steps, "OPTIONAL", ())
    calls = []
    default = _registration(
        "default-step",
        lambda ctx: calls.append("default"),
        classification=SetupStepClassification.DEFAULT,
    )
    optional = _registration(
        "optional-step",
        lambda ctx: calls.append("optional"),
        classification=SetupStepClassification.OPTIONAL,
    )
    discovered = discover_setup_steps(
        [_EntryPoint("fixture", "fixture.setup:steps", (optional, default))]
    )

    assert run_wizard(_context(tmp_path), setup_steps=discovered) == 0
    assert calls == ["default"]
    assert (
        run_wizard(
            _context(tmp_path),
            only_step="optional-step",
            setup_steps=discovered,
        )
        == 0
    )
    assert calls == ["default", "optional"]


def test_check_flow_refuses_contributed_python_without_import_or_execution(tmp_path):
    observed = []

    def check(ctx):
        observed.append(ctx.flow)
        if not (ctx.project_dir / "expected.conf").exists():
            ctx.block("expected.conf is missing")

    discovered = discover_setup_steps(
        [_EntryPoint("fixture", "fixture.setup:check", (_registration("check", check),))]
    )

    ctx = _context(tmp_path, Flow.CHECK)
    rc = run_wizard(
        ctx,
        only_step="check",
        setup_steps=discovered,
    )

    assert rc == 1
    assert observed == []
    assert list(tmp_path.iterdir()) == []
    assert "not sandboxed" in ctx.blockers[0]


def test_core_recovery_selection_does_not_discover_broken_provider(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.setup import steps as core_steps

    calls = []
    monkeypatch.setattr(
        core_steps,
        "ORDERED",
        (("keys", lambda ctx: calls.append("keys")),),
    )
    monkeypatch.setattr(core_steps, "OPTIONAL", ())

    def broken_discovery():
        raise AssertionError("provider discovery must not run for core recovery")

    monkeypatch.setattr(
        "kestrel_sovereign.setup.wizard.discover_setup_steps",
        broken_discovery,
    )

    assert run_wizard(_context(tmp_path), only_step="keys") == 0
    assert calls == ["keys"]


def test_check_uses_explicit_core_only_recovery_path(tmp_path, monkeypatch):
    from kestrel_sovereign.setup import steps as core_steps

    calls = []
    monkeypatch.setattr(
        core_steps,
        "ORDERED",
        (("verify", lambda ctx: calls.append("verify")),),
    )
    monkeypatch.setattr(core_steps, "OPTIONAL", ())

    def broken_discovery():
        raise AssertionError("provider discovery must not run under --check")

    monkeypatch.setattr(
        "kestrel_sovereign.setup.wizard.discover_setup_steps",
        broken_discovery,
    )

    assert run_wizard(_context(tmp_path, Flow.CHECK)) == 0
    assert calls == ["verify"]


def test_step_exception_becomes_blocker_and_stops_later_steps(tmp_path):
    calls = []

    def fail(ctx):
        raise ValueError("credential format is invalid")

    def later(ctx):
        calls.append("later")

    discovered = discover_setup_steps(
        [
            _EntryPoint(
                "fixture",
                "fixture.setup:steps",
                (
                    _registration("fail", fail, order=1),
                    _registration("later", later, order=2),
                ),
            )
        ]
    )
    ctx = _context(tmp_path)

    assert run_wizard(ctx, only_step="fixture", setup_steps=discovered) == 1
    assert calls == []
    assert ctx.halted
    assert "credential format is invalid" in ctx.blockers[0]


def test_provider_load_failure_is_actionable():
    with pytest.raises(
        SetupContributionDiscoveryError,
        match="could not load contributed setup steps for 'fixture'",
    ):
        discover_setup_steps(
            [_EntryPoint("fixture", "fixture.setup:steps", ImportError("broken"))]
        )


def test_broken_provider_runs_no_core_step_in_normal_complete_transition(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.setup import steps as core_steps

    calls = []
    monkeypatch.setattr(
        core_steps,
        "ORDERED",
        (("keys", lambda ctx: calls.append("keys")),),
    )
    monkeypatch.setattr(core_steps, "OPTIONAL", ())

    def invalid_complete_transition():
        raise SetupContributionDiscoveryError(
            "could not load contributed setup steps for 'broken-provider'"
        )

    monkeypatch.setattr(
        "kestrel_sovereign.setup.wizard.discover_setup_steps",
        invalid_complete_transition,
    )
    ctx = _context(tmp_path)

    assert run_wizard(ctx) == 1
    assert calls == []
    assert "broken-provider" in ctx.blockers[0]


def test_missing_feature_slug_names_install_command_without_importing_checkout(
    monkeypatch
):
    discovered = discover_setup_steps([])
    info = FeaturePackageInfo(
        name="voice",
        package="kestrel-feature-voice",
        git="https://example.invalid/voice.git",
        features=["VoiceFeature"],
        description="voice",
        boundary=PackageBoundary.FEATURE_PACKAGE,
    )
    monkeypatch.setattr(
        "kestrel_sovereign.feature_registry.load_registry",
        lambda: {"voice": info},
    )

    def missing_distribution(name):
        raise __import__("importlib").metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        "kestrel_sovereign.setup.contributions.importlib.metadata.distribution",
        missing_distribution,
    )

    message = missing_setup_step_message("voice", discovered)

    assert "kestrel-feature-voice" in message
    assert "kestrel feature install voice" in message
    assert SETUP_STEP_ENTRY_POINT_GROUP not in message
