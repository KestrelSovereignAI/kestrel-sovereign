"""Wizard orchestrator.

Public entry: :func:`run_wizard`. The CLI builds a :class:`SetupContext`,
chooses the steps to run (all of them, or just one), and calls this.

Reset semantics: ``--reset`` does not delete files. It moves ``.env``
and ``kestrel.toml`` aside to ``<file>.backup-<timestamp>`` and lets the
wizard regenerate from scratch. Agents are *never* removed by the wizard
(touching ``agent_data/`` could destroy memory).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from kestrel_sdk.features import SetupStepClassification

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.contributions import (
    DiscoveredSetupSteps,
    SetupContributionDiscoveryError,
    discover_core_setup_steps,
    discover_setup_steps,
    missing_setup_step_message,
    run_setup_step,
)


def run_wizard(
    ctx: SetupContext,
    *,
    only_step: str | None = None,
    setup_steps: DiscoveredSetupSteps | None = None,
    core_only: bool = False,
) -> int:
    """Run the wizard.

    Returns a process exit code: ``0`` if every step finished without
    blockers, ``1`` otherwise. Callers (CLI) propagate that to ``sys.exit``.

    ``--check`` is read-only by contract: even if ``ctx.reset`` is True,
    we refuse to move files in CHECK mode. The CLI rejects this combo
    upfront with a clear error; this guard catches anyone calling
    ``run_wizard`` directly (e.g. tests, embedders).
    """
    core_steps = discover_core_setup_steps()
    core_selection = (
        core_steps.selected(only_step) if only_step is not None else None
    )
    if ctx.flow is Flow.CHECK and only_step is not None and not core_selection:
        ctx.block(
            "Contributed setup steps are not executed by --check: Python plugin "
            "code is not sandboxed, so Sovereign cannot enforce the read-only "
            "contract. Run the provider's checker outside `kestrel setup "
            "--check` or select a built-in recovery step."
        )
        return 1

    try:
        # Built-in recovery selections and CHECK never import provider code.
        # Normal all-step setup still validates the complete prospective set
        # atomically, as required by the SDK ordering contract.
        discovered = (
            core_steps
            if core_only or ctx.flow is Flow.CHECK or core_selection
            else (setup_steps or discover_setup_steps())
        )
        selected = (
            discovered.selected(only_step) if only_step is not None else None
        )
    except SetupContributionDiscoveryError as exc:
        ctx.block(str(exc))
        if ctx.flow is not Flow.CHECK:
            _print_summary(ctx)
        return 1

    if only_step is not None and not selected:
        ctx.prompter.info(missing_setup_step_message(only_step, discovered))
        return 1

    if ctx.reset and ctx.flow is Flow.CHECK:
        ctx.block(
            "refused to reset in --check mode (read-only by contract)"
        )
    elif ctx.reset:
        _reset_config_files(ctx)

    registrations = (
        selected
        if selected is not None
        else tuple(
            registration
            for registration in discovered.registry.ordered()
            if registration.classification is SetupStepClassification.DEFAULT
        )
    )
    for registration in registrations:
        if only_step is None:
            ctx.prompter.info(f"\n— {registration.name} —")
        try:
            run_setup_step(registration, ctx)
        except Exception as exc:  # noqa: BLE001 - third-party setup boundary
            reason = (
                f"Setup step {registration.name!r} failed "
                f"({type(exc).__name__}: {exc})"
            )
            ctx.block(reason)
            ctx.halt(reason)
        if ctx.halted:
            # A step (e.g. ``keys`` on a KESTREL_DATA_KEY custody conflict)
            # declared the whole workflow unsafe to continue. Stop before any
            # later, key-dependent step mutates state under it (#2468).
            ctx.prompter.info(
                f"\nHalting setup — {ctx.halt_reason}"
            )
            break

    if ctx.flow is not Flow.CHECK:
        _print_summary(ctx)

    return 0 if not ctx.blockers else 1


def _reset_config_files(ctx: SetupContext) -> None:
    """Move ``.env`` and ``kestrel.toml`` aside so the wizard rewrites them.

    Never deletes — uses ``shutil.move`` to a timestamped backup.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    for path in (ctx.env_path, ctx.kestrel_toml_path):
        if not path.exists():
            continue
        backup = path.with_name(f"{path.name}.backup-{ts}")
        shutil.move(str(path), str(backup))
        ctx.record(f"Reset: moved {path.name} → {backup.name}")


def _print_summary(ctx: SetupContext) -> None:
    p = ctx.prompter
    p.info("")
    p.info("=" * 50)
    if ctx.changes:
        p.info("Changes:")
        for line in ctx.changes:
            p.info(f"  • {line}")
    else:
        p.info("No changes — already configured.")

    if ctx.blockers:
        p.info("")
        p.info("Blockers:")
        for line in ctx.blockers:
            p.info(f"  ! {line}")
        p.info("")
        p.info("Re-run: kestrel setup")
    else:
        p.info("")
        p.info("Setup complete.")


def build_context(
    project_dir: Path,
    *,
    flow: Flow,
    reset: bool,
    is_test_instance: bool = False,
) -> SetupContext:
    """Construct a SetupContext with the right prompter for the env."""
    from kestrel_sovereign.setup.prompts import (
        NonInteractivePrompter,
        QuestionaryPrompter,
        is_tty,
    )

    project_dir = project_dir.resolve()
    prompter = QuestionaryPrompter() if (flow is Flow.INTERACTIVE and is_tty()) else NonInteractivePrompter()
    return SetupContext(
        project_dir=project_dir,
        agent_data_root=project_dir / "agent_data",
        flow=flow,
        prompter=prompter,
        reset=reset,
        is_test_instance=is_test_instance,
    )
