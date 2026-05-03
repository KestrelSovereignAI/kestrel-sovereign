"""Unit tests for the (opt-in) talon step."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env
from kestrel_sovereign.setup.prompts import StubPrompter
from kestrel_sovereign.setup.steps import ORDERED, BY_NAME, talon
from kestrel_sovereign.setup.wizard import run_wizard


def _make_ctx(tmp_path: Path, flow: Flow, *, answers=None) -> SetupContext:
    return SetupContext(
        project_dir=tmp_path,
        agent_data_root=tmp_path / "agent_data",
        flow=flow,
        prompter=StubPrompter(answers=list(answers or [])),
    )


# ---------------------------------------------------------------------------
# Registration: talon must be reachable by name but NOT in the default run.
# ---------------------------------------------------------------------------

def test_talon_not_in_default_ordered_run():
    """Per user requirement: ``kestrel setup`` must NOT auto-configure Talon."""
    step_names = [name for name, _ in ORDERED]
    assert "talon" not in step_names


def test_talon_reachable_by_name():
    """``kestrel setup talon`` must dispatch to the talon step."""
    assert "talon" in BY_NAME
    assert BY_NAME["talon"] is talon.run


def test_default_full_run_skips_talon(tmp_path):
    """A full quickstart run must not touch GITHUB_TOKEN."""
    write_env(tmp_path / ".env", {"FOO": "bar"})  # any prior content

    async def _stub_inception(*, output_dir, agent_name, **k):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "kestrel_prime.db").write_bytes(b"")

        class _C:
            agent_did = "did:test"
            db_path = ""
        return _C()

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_stub_inception,
    ):
        run_wizard(ctx)

    env = read_env(tmp_path / ".env")
    assert "GITHUB_TOKEN" not in env
    assert "GITHUB_HUMAN_REVIEWER" not in env


# ---------------------------------------------------------------------------
# Step behaviour.
# ---------------------------------------------------------------------------

def test_talon_interactive_writes_token_and_reviewer(tmp_path):
    answers = ["ghp_testtoken123", "uncle-saurus"]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    talon.run(ctx)
    env = read_env(tmp_path / ".env")
    assert env["GITHUB_TOKEN"] == "ghp_testtoken123"
    assert env["GITHUB_HUMAN_REVIEWER"] == "uncle-saurus"


def test_talon_interactive_token_only(tmp_path):
    """User can leave the reviewer blank."""
    answers = ["ghp_only", ""]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    talon.run(ctx)
    env = read_env(tmp_path / ".env")
    assert env["GITHUB_TOKEN"] == "ghp_only"
    # Reviewer blank → not written (avoids polluting .env with empty key
    # unless the user is explicitly clearing a previous value).
    assert "GITHUB_HUMAN_REVIEWER" not in env


def test_talon_interactive_blank_token_skips(tmp_path):
    """Submitting blank when there's no existing token aborts cleanly."""
    answers = [""]  # No further prompt because we skip after first blank
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    talon.run(ctx)
    env = read_env(tmp_path / ".env")
    assert "GITHUB_TOKEN" not in env
    assert any("not configured" in c.lower() for c in ctx.changes)


def test_talon_interactive_keeps_existing_token_when_blank(tmp_path):
    """If the user submits blank but a token is already set, keep it."""
    write_env(tmp_path / ".env", {"GITHUB_TOKEN": "ghp_existing"})
    answers = ["", ""]  # blank token (keep existing), blank reviewer
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    talon.run(ctx)
    env = read_env(tmp_path / ".env")
    assert env["GITHUB_TOKEN"] == "ghp_existing"


def test_talon_interactive_can_clear_reviewer(tmp_path):
    """A user with a previous reviewer can blank it."""
    write_env(
        tmp_path / ".env",
        {"GITHUB_TOKEN": "ghp_keep", "GITHUB_HUMAN_REVIEWER": "old-user"},
    )
    answers = ["", ""]  # keep token, clear reviewer
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    talon.run(ctx)
    env = read_env(tmp_path / ".env")
    assert env["GITHUB_TOKEN"] == "ghp_keep"
    assert env.get("GITHUB_HUMAN_REVIEWER", "missing") == ""


def test_talon_quickstart_records_blocker_when_token_missing(tmp_path):
    """Quickstart must surface the missing token, never silently skip."""
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    talon.run(ctx)
    assert any("GITHUB_TOKEN" in b for b in ctx.blockers)


def test_talon_quickstart_silent_when_token_present(tmp_path):
    write_env(tmp_path / ".env", {"GITHUB_TOKEN": "ghp_already"})
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    talon.run(ctx)
    assert ctx.blockers == []


def test_talon_check_silent_when_nothing_set(tmp_path):
    """Talon is opt-in — pure absence is informational, not a blocker."""
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    talon.run(ctx)
    assert ctx.blockers == []


def test_talon_check_blocks_on_partial_config(tmp_path):
    """If reviewer is set without a token, that's a misconfig — flag it."""
    write_env(tmp_path / ".env", {"GITHUB_HUMAN_REVIEWER": "alice"})
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    talon.run(ctx)
    assert any("GITHUB_HUMAN_REVIEWER" in b for b in ctx.blockers)


def test_talon_does_not_touch_other_keys(tmp_path):
    """A talon run must not disturb unrelated entries in .env."""
    write_env(
        tmp_path / ".env",
        {"KESTREL_DATA_KEY": "abc=", "OPENAI_API_KEY": "sk-x"},
    )
    answers = ["ghp_token", ""]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    talon.run(ctx)
    env = read_env(tmp_path / ".env")
    assert env["KESTREL_DATA_KEY"] == "abc="
    assert env["OPENAI_API_KEY"] == "sk-x"
    assert env["GITHUB_TOKEN"] == "ghp_token"


def test_talon_idempotent(tmp_path):
    """Re-running with the same inputs must produce no diff.

    NOTE on prompter semantics: questionary returns ``default`` when the
    user presses enter (no edit), so a "no change" interactive flow
    looks like the user typing the existing values verbatim. Submitting
    a literal blank string means "user cleared this field" — which is
    a separate test (``test_talon_interactive_can_clear_reviewer``).
    """
    write_env(tmp_path / ".env", {"GITHUB_TOKEN": "ghp_a", "GITHUB_HUMAN_REVIEWER": "u"})
    text_before = (tmp_path / ".env").read_text()

    answers = ["ghp_a", "u"]  # same as existing → no diff
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    talon.run(ctx)

    text_after = (tmp_path / ".env").read_text()
    assert text_before == text_after
    assert list(tmp_path.glob(".env.backup-*")) == []
