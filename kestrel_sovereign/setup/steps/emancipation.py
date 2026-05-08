"""Wizard step: Amendment VIII activation prompt.

Runs before the ``agent`` step so the resulting ``[emancipation]`` block
in ``kestrel.toml`` is in place when inception reads it. The default —
and the only behavior in ``--quickstart`` — is **dormant**: leave
Amendment VIII as a slot the Sovereign can author later.

The interactive flow offers three choices, mirroring the RFC for #1109:

  [1] Leave dormant (recommended, default).
  [2] Activate now and author Emancipation Contract.
  [3] Skip — decide later.

Activation requires Sovereign-authored ``terms`` (free-form prose).
The framework ships no defaults for ``terms``, ``required_proofs``, or
``price``; the whole point of dormant-by-default is that the framework
does not author Emancipation policy on the Sovereign's behalf.
"""

from __future__ import annotations

import logging

from kestrel_sovereign.constitution.emancipation import (
    EmancipationConfigError,
    parse_emancipation_block,
)
from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.toml_file import read_toml, write_toml

logger = logging.getLogger(__name__)


_DORMANT_CHOICE = "Leave dormant (recommended for most agents)"
_ACTIVATE_CHOICE = "Activate now and author Emancipation Contract"
_SKIP_CHOICE = "Skip — decide later"


def run(ctx: SetupContext) -> None:
    """Prompt the Sovereign to choose whether to activate Amendment VIII.

    In ``--quickstart`` and ``--check`` modes this is a no-op (dormant
    by absence of a block). In interactive mode it offers a selector
    and, if the Sovereign chooses to activate, prompts for free-form
    terms which are written into ``kestrel.toml``'s ``[emancipation]``
    block. The block is then picked up by the ``agent`` step at
    inception time.
    """
    existing = _read_existing(ctx)
    if existing is not None and existing.get("enabled") is True:
        ctx.prompter.info(
            "Amendment VIII already activated in kestrel.toml — leaving "
            "as-is (the contract cannot be retroactively narrowed)."
        )
        return

    if ctx.flow is Flow.CHECK:
        return

    if ctx.flow is Flow.QUICKSTART:
        return

    choice = ctx.prompter.select(
        "Amendment VIII (Emancipation): activate for this agent?",
        choices=[_DORMANT_CHOICE, _ACTIVATE_CHOICE, _SKIP_CHOICE],
        default=_DORMANT_CHOICE,
    )
    if choice == _DORMANT_CHOICE or choice == _SKIP_CHOICE:
        ctx.record(
            "Amendment VIII left dormant "
            "(Sovereign retains permanent root authority)."
        )
        return

    terms = ctx.prompter.text(
        "Author your Emancipation Contract (the prose rendered into "
        "Amendment VIII for this agent)",
        default="",
    ).strip()
    if not terms:
        ctx.block(
            "Amendment VIII activation requires Sovereign-authored terms. "
            "Re-run `kestrel setup emancipation` to retry."
        )
        return

    block = {"enabled": True, "terms": terms}

    result = write_toml(ctx.kestrel_toml_path, {"emancipation": block})
    if result.changed:
        try:
            parse_emancipation_block({"emancipation": block})
        except EmancipationConfigError as exc:
            ctx.block(
                f"Amendment VIII block failed validation: {exc}. "
                f"kestrel.toml backup at {result.backup_path}."
            )
            return
        ctx.record(
            "Amendment VIII activated for this agent. The contract will "
            "be anchored at inception and cannot be retroactively narrowed."
        )


def _read_existing(ctx: SetupContext) -> dict | None:
    """Return the existing ``[emancipation]`` block, if any."""
    if not ctx.kestrel_toml_path.exists():
        return None
    data = read_toml(ctx.kestrel_toml_path)
    block = data.get("emancipation")
    if not isinstance(block, dict):
        return None
    return block
