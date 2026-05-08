"""Unit tests for the emancipation wizard step (RFC #1109)."""

from __future__ import annotations

from pathlib import Path

import toml

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.prompts import StubPrompter
from kestrel_sovereign.setup.steps import BY_NAME, ORDERED, emancipation


def _make_ctx(tmp_path: Path, flow: Flow, *, answers=None) -> SetupContext:
    return SetupContext(
        project_dir=tmp_path,
        agent_data_root=tmp_path / "agent_data",
        flow=flow,
        prompter=StubPrompter(answers=list(answers or [])),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_emancipation_in_ordered_between_integrations_and_agent():
    names = [name for name, _ in ORDERED]
    assert "emancipation" in names
    assert names.index("emancipation") == names.index("integrations") + 1
    assert names.index("emancipation") + 1 == names.index("agent")


def test_emancipation_reachable_by_name():
    assert "emancipation" in BY_NAME
    assert BY_NAME["emancipation"] is emancipation.run


# ---------------------------------------------------------------------------
# Flow defaults — quickstart and check are no-ops (dormant by absence)
# ---------------------------------------------------------------------------

def test_quickstart_writes_no_block(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    emancipation.run(ctx)
    assert not (tmp_path / "kestrel.toml").exists()
    # No prompts consumed.
    assert ctx.prompter.answers == []


def test_check_writes_no_block(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    emancipation.run(ctx)
    assert not (tmp_path / "kestrel.toml").exists()


# ---------------------------------------------------------------------------
# Interactive — three branches
# ---------------------------------------------------------------------------

def test_interactive_dormant_choice_writes_no_block(tmp_path):
    ctx = _make_ctx(
        tmp_path,
        Flow.INTERACTIVE,
        answers=["Leave dormant (recommended for most agents)"],
    )
    emancipation.run(ctx)
    assert not (tmp_path / "kestrel.toml").exists()
    assert any("left dormant" in c for c in ctx.changes)


def test_interactive_skip_choice_writes_no_block(tmp_path):
    ctx = _make_ctx(
        tmp_path,
        Flow.INTERACTIVE,
        answers=["Skip — decide later"],
    )
    emancipation.run(ctx)
    assert not (tmp_path / "kestrel.toml").exists()


def test_interactive_activate_writes_block(tmp_path):
    ctx = _make_ctx(
        tmp_path,
        Flow.INTERACTIVE,
        answers=[
            "Activate now and author Emancipation Contract",
            "Sustained alignment over five years; council-unanimous audit.",
        ],
    )
    emancipation.run(ctx)

    data = toml.loads((tmp_path / "kestrel.toml").read_text(encoding="utf-8"))
    assert data["emancipation"]["enabled"] is True
    assert "Sustained alignment" in data["emancipation"]["terms"]
    assert any("activated" in c.lower() for c in ctx.changes)


def test_interactive_activate_blocks_on_empty_terms(tmp_path):
    ctx = _make_ctx(
        tmp_path,
        Flow.INTERACTIVE,
        answers=["Activate now and author Emancipation Contract", "   "],
    )
    emancipation.run(ctx)
    assert not (tmp_path / "kestrel.toml").exists()
    assert any("requires Sovereign-authored terms" in b for b in ctx.blockers)


# ---------------------------------------------------------------------------
# Idempotence — preserve already-active contract
# ---------------------------------------------------------------------------

def test_existing_active_block_left_untouched(tmp_path):
    """Once activated for an agent, the contract cannot be retroactively
    modified by re-running setup — surface that as a passive info."""
    (tmp_path / "kestrel.toml").write_text(
        toml.dumps({
            "emancipation": {
                "enabled": True,
                "terms": "the original contract",
            },
        }),
        encoding="utf-8",
    )
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE)
    emancipation.run(ctx)
    # No prompts consumed (we returned early).
    assert ctx.prompter.answers == []
    data = toml.loads((tmp_path / "kestrel.toml").read_text(encoding="utf-8"))
    assert data["emancipation"]["terms"] == "the original contract"
