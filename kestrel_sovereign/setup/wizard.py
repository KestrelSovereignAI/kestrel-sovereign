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

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.steps import BY_NAME, ORDERED


def run_wizard(
    ctx: SetupContext,
    *,
    only_step: str | None = None,
) -> int:
    """Run the wizard.

    Returns a process exit code: ``0`` if every step finished without
    blockers, ``1`` otherwise. Callers (CLI) propagate that to ``sys.exit``.

    ``--check`` is read-only by contract: even if ``ctx.reset`` is True,
    we refuse to move files in CHECK mode. The CLI rejects this combo
    upfront with a clear error; this guard catches anyone calling
    ``run_wizard`` directly (e.g. tests, embedders).
    """
    if ctx.reset and ctx.flow is Flow.CHECK:
        ctx.block(
            "refused to reset in --check mode (read-only by contract)"
        )
    elif ctx.reset:
        _reset_config_files(ctx)

    if only_step is not None:
        if only_step not in BY_NAME:
            ctx.prompter.info(
                f"Unknown step: {only_step!r}. "
                f"Valid: {', '.join(name for name, _ in ORDERED)}"
            )
            return 1
        BY_NAME[only_step](ctx)
    else:
        for name, step_fn in ORDERED:
            ctx.prompter.info(f"\n— {name} —")
            step_fn(ctx)

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
    )
