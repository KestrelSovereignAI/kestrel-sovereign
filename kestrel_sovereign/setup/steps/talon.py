"""Optional Kestrel Talon integration step.

Talon is the standalone GitHub-issue processor that ships as a separate
``kestrel-talon`` package. It is *not* part of the default wizard run —
the user explicitly opted out of automatic Talon setup. This step is
therefore registered in :data:`BY_NAME` only, not :data:`ORDERED`, and
runs only when the user types ``kestrel setup talon``.

Captured into ``.env``:

  ``GITHUB_TOKEN``           — required: PAT with repo access
  ``GITHUB_HUMAN_REVIEWER``  — optional: GitHub login for blocked-issue
                                 escalation; left empty if the user
                                 doesn't want one

Talon also requires ``ANTHROPIC_API_KEY``, but that is already covered
by ``kestrel setup llm`` when the user picks Anthropic. We don't double-
prompt for it here.
"""

from __future__ import annotations

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env


def run(ctx: SetupContext) -> None:
    """Capture GitHub credentials for kestrel-talon."""
    env = read_env(ctx.env_path)
    updates: dict[str, str] = {}

    token_present = bool(env.get("GITHUB_TOKEN"))

    if ctx.flow is Flow.CHECK:
        # Talon is opt-in — its absence is informational, not a blocker
        # for the project as a whole. Only flag the partial-config case
        # where someone explicitly set GITHUB_HUMAN_REVIEWER without a
        # token (a sign the user *meant* to enable Talon).
        if env.get("GITHUB_HUMAN_REVIEWER") and not token_present:
            ctx.block(
                "GITHUB_HUMAN_REVIEWER set but GITHUB_TOKEN missing — "
                "Talon cannot authenticate. Run: kestrel setup talon"
            )
        return

    if ctx.flow is Flow.QUICKSTART:
        # Quickstart never prompts for cloud secrets the user didn't
        # ask for. If a token is already there, we leave it; otherwise
        # we record a blocker so the summary tells the user how to
        # enable Talon. We do NOT auto-skip silently.
        if not token_present:
            ctx.block(
                "GITHUB_TOKEN not set — Talon disabled. Run: kestrel setup talon"
            )
        return

    # Interactive flow.
    if token_present:
        ctx.prompter.info(
            "GITHUB_TOKEN already set; press enter at the prompt to keep it."
        )
    new_token = ctx.prompter.secret(
        "GitHub PAT for Talon (GITHUB_TOKEN). Needs repo + read:org. "
        "Leave blank to skip Talon setup.",
        default=env.get("GITHUB_TOKEN", ""),
    )
    if new_token and new_token != env.get("GITHUB_TOKEN"):
        updates["GITHUB_TOKEN"] = new_token
    elif not new_token and not token_present:
        ctx.record("GITHUB_TOKEN left blank — Talon not configured")
        return

    reviewer_default = env.get("GITHUB_HUMAN_REVIEWER", "")
    new_reviewer = ctx.prompter.text(
        "GitHub login to escalate blocked issues to (GITHUB_HUMAN_REVIEWER). "
        "Leave blank for none.",
        default=reviewer_default,
    ).strip()
    if new_reviewer != reviewer_default:
        # allow_empty=True lets the user *clear* a previously set reviewer
        # by submitting blank.
        updates["GITHUB_HUMAN_REVIEWER"] = new_reviewer

    if not updates:
        ctx.record("Talon credentials unchanged")
        return

    result = write_env(ctx.env_path, updates, allow_empty=True)
    if result.backup_path is not None:
        ctx.record(f"Backed up existing .env to {result.backup_path.name}")
    for key in result.added:
        ctx.record(f"Set {key} in .env (Talon)")
    for key in result.updated:
        ctx.record(f"Updated {key} in .env (Talon)")
