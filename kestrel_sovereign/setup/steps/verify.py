"""End-of-run readiness check.

Wraps :func:`kestrel_sovereign.doctor.diagnose` so the wizard prints a
single combined summary: changes the wizard made + the doctor's report.
"""

from __future__ import annotations

from kestrel_sovereign.doctor import diagnose, format_report
from kestrel_sovereign.setup.context import Flow, SetupContext


def run(ctx: SetupContext) -> None:
    """Run doctor and surface findings as wizard blockers."""
    report = diagnose(ctx.project_dir)
    if ctx.flow is Flow.CHECK:
        # Doctor output is the only output in --check mode.
        ctx.prompter.info(format_report(report))
        for msg in report.fail:
            ctx.block(msg)
        return

    ctx.prompter.info(format_report(report))
    for msg in report.fail:
        ctx.block(msg)
