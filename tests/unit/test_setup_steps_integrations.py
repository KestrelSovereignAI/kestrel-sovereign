"""Unit tests for the cloud-integrations step.

Covers all five curated onboarders, the three flow modes, the
"managed but missing required key" blocker path, the partial-config
rule (blank required key never marks managed), and a couple of
file-preservation guarantees (existing [features.disabled] survives;
unknown user-authored [features.foo] survives).
"""

from __future__ import annotations

from pathlib import Path

import toml as _toml

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env
from kestrel_sovereign.setup.prompts import StubPrompter
from kestrel_sovereign.setup.steps import BY_NAME, ORDERED, integrations
from kestrel_sovereign.setup.toml_file import read_toml, write_toml


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

def test_integrations_in_ordered_after_llm_before_agent():
    names = [name for name, _ in ORDERED]
    assert "integrations" in names
    # integrations runs immediately after llm.
    assert names.index("integrations") == names.index("llm") + 1
    # integrations precedes agent. The emancipation step (#1109) sits
    # between them so the [emancipation] block lands in kestrel.toml
    # before inception.
    assert names.index("integrations") < names.index("agent")


def test_integrations_reachable_by_name():
    assert "integrations" in BY_NAME
    assert BY_NAME["integrations"] is integrations.run


def test_curated_onboarders_match_locked_in_list():
    """Adding a 6th integration is a deliberate code change — surface it
    so reviewers notice when this list grows."""
    ids = [i.id for i in integrations._INTEGRATIONS]
    assert ids == [
        "tavily", "elevenlabs", "deepgram", "huggingface", "runpod",
    ]


# ---------------------------------------------------------------------------
# Interactive: each onboarder
# ---------------------------------------------------------------------------

def test_decline_all_writes_nothing(tmp_path):
    """User says no to every integration → no .env, no kestrel.toml."""
    answers = [False, False, False, False, False]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / "kestrel.toml").exists()


def test_select_tavily_writes_key_and_marks_managed(tmp_path):
    answers = [
        True, "tvly-test-key",  # tavily yes + key
        False, False, False, False,  # decline the rest
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)
    env = read_env(tmp_path / ".env")
    config = read_toml(tmp_path / "kestrel.toml")
    assert env["TAVILY_API_KEY"] == "tvly-test-key"
    assert config["features"]["managed"] == {"tavily": True}


def test_select_elevenlabs(tmp_path):
    answers = [
        False,  # tavily no
        True, "el-test-key",
        False, False, False,
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)
    assert read_env(tmp_path / ".env")["ELEVENLABS_API_KEY"] == "el-test-key"
    assert read_toml(tmp_path / "kestrel.toml")["features"]["managed"]["elevenlabs"] is True


def test_select_deepgram(tmp_path):
    answers = [
        False, False,
        True, "dg-test-key",
        False, False,
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)
    assert read_env(tmp_path / ".env")["DEEPGRAM_API_KEY"] == "dg-test-key"


def test_select_huggingface_uses_HF_TOKEN_not_HUGGINGFACE(tmp_path):
    """The codebase reads HF_TOKEN — verify the wizard writes that exact
    name, not the also-common HUGGINGFACE_API_TOKEN variant."""
    answers = [
        False, False, False,
        True, "hf_real_token",
        False,
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)
    env = read_env(tmp_path / ".env")
    assert env["HF_TOKEN"] == "hf_real_token"
    assert "HUGGINGFACE_API_TOKEN" not in env


def test_select_runpod(tmp_path):
    answers = [
        False, False, False, False,
        True, "rp-key",
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)
    assert read_env(tmp_path / ".env")["RUNPOD_API_KEY"] == "rp-key"


# ---------------------------------------------------------------------------
# Partial-config rule: blank required key → blocker + NOT marked managed
# ---------------------------------------------------------------------------

def test_blank_required_key_blocks_and_does_not_mark_managed(tmp_path):
    """Selecting Tavily but submitting blank for the key must record a
    blocker and NOT write [features.managed].tavily = true. 'Managed'
    means 'opted in AND has enough config to verify later.'"""
    answers = [
        True, "",  # tavily yes, but key blank
        False, False, False, False,
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)

    # No env update for Tavily.
    env = read_env(tmp_path / ".env")
    assert "TAVILY_API_KEY" not in env

    # No managed entry written.
    config = read_toml(tmp_path / "kestrel.toml")
    assert "tavily" not in (config.get("features", {}).get("managed") or {})

    # And a blocker explains why.
    assert any(
        "TAVILY_API_KEY" in b and "blank" in b.lower()
        for b in ctx.blockers
    )


# ---------------------------------------------------------------------------
# Quickstart and check: validate-only, never discover
# ---------------------------------------------------------------------------

def test_quickstart_with_no_managed_entries_is_silent(tmp_path):
    """No `[features.managed]` table at all → quickstart is a no-op."""
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    integrations.run(ctx)
    assert ctx.blockers == []
    assert not (tmp_path / ".env").exists()


def test_quickstart_blocks_on_managed_but_missing_key(tmp_path):
    """Captures the exact CI scenario: previous run marked tavily managed,
    but .env got wiped. Quickstart must surface the gap, not auto-fix."""
    write_toml(
        tmp_path / "kestrel.toml",
        {"features": {"managed": {"tavily": True}}},
    )
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    integrations.run(ctx)
    assert any(
        "TAVILY_API_KEY" in b and "managed" in b.lower()
        for b in ctx.blockers
    )


def test_quickstart_satisfied_when_managed_and_key_present(tmp_path):
    write_toml(
        tmp_path / "kestrel.toml",
        {"features": {"managed": {"tavily": True}}},
    )
    write_env(tmp_path / ".env", {"TAVILY_API_KEY": "tvly-set"})
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    integrations.run(ctx)
    assert ctx.blockers == []


def test_check_does_not_write_on_validation_only_path(tmp_path):
    """--check is read-only by contract — even when there are blockers,
    nothing is written."""
    write_toml(
        tmp_path / "kestrel.toml",
        {"features": {"managed": {"tavily": True}}},
    )
    backups_before = list(tmp_path.glob("*.backup-*"))
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    integrations.run(ctx)
    backups_after = list(tmp_path.glob("*.backup-*"))
    assert backups_before == backups_after


def test_managed_unknown_id_is_silent(tmp_path):
    """If a future-wizard or hand-edit produces [features.managed].future_thing,
    the current wizard should not error. We just skip what we don't know."""
    write_toml(
        tmp_path / "kestrel.toml",
        {"features": {"managed": {"future_thing": True}}},
    )
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    integrations.run(ctx)
    assert ctx.blockers == []


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

def test_keep_existing_managed_with_existing_key_is_no_diff(tmp_path):
    """Re-runs that match prior intent must produce no .env or kestrel.toml
    backup. (Specifically requested in the locked-in plan.)"""
    write_env(tmp_path / ".env", {"TAVILY_API_KEY": "tvly-prior"})
    write_toml(
        tmp_path / "kestrel.toml",
        {"features": {"managed": {"tavily": True}}},
    )
    env_before = (tmp_path / ".env").read_text()
    toml_before = (tmp_path / "kestrel.toml").read_text()

    answers = [
        True, "tvly-prior",  # keep tavily, key unchanged
        False, False, False, False,
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)

    assert (tmp_path / ".env").read_text() == env_before
    assert (tmp_path / "kestrel.toml").read_text() == toml_before
    assert list(tmp_path.glob(".env.backup-*")) == []
    assert list(tmp_path.glob("kestrel.toml.backup-*")) == []


def test_decline_previously_managed_unmanages(tmp_path):
    """User runs the wizard again and declines a previously-managed
    integration. We flip it to false so intent is auditable, but we
    do NOT delete the env var (user might still want it around)."""
    write_env(tmp_path / ".env", {"TAVILY_API_KEY": "tvly-keep"})
    write_toml(
        tmp_path / "kestrel.toml",
        {"features": {"managed": {"tavily": True}}},
    )

    answers = [
        False,  # decline tavily this time
        False, False, False, False,
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)

    config = read_toml(tmp_path / "kestrel.toml")
    assert config["features"]["managed"]["tavily"] is False
    # Env var preserved — we don't delete user secrets.
    assert read_env(tmp_path / ".env")["TAVILY_API_KEY"] == "tvly-keep"


# ---------------------------------------------------------------------------
# Preserves unrelated config
# ---------------------------------------------------------------------------

def test_preserves_existing_features_disabled_list(tmp_path):
    """[features.disabled] is a runtime-toggle list — orthogonal to the
    wizard's [features.managed] table. Must survive verbatim."""
    write_toml(
        tmp_path / "kestrel.toml",
        {"features": {"disabled": ["WellnessFeature", "VisualIdentityFeature"]}},
    )
    answers = [
        True, "tvly-key",
        False, False, False, False,
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)

    config = read_toml(tmp_path / "kestrel.toml")
    assert config["features"]["disabled"] == ["WellnessFeature", "VisualIdentityFeature"]
    assert config["features"]["managed"]["tavily"] is True


def test_preserves_unknown_user_authored_feature_table(tmp_path):
    write_toml(
        tmp_path / "kestrel.toml",
        {"features": {"user_extra": {"some_setting": "value"}}},
    )
    answers = [
        True, "tvly-key",
        False, False, False, False,
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)

    config = read_toml(tmp_path / "kestrel.toml")
    assert config["features"]["user_extra"] == {"some_setting": "value"}


def test_preserves_unrelated_env_keys(tmp_path):
    write_env(
        tmp_path / ".env",
        {
            "KESTREL_DATA_KEY": "fernet-key-stub",
            "OPENAI_API_KEY": "sk-x",
            "SOME_USER_VAR": "preserve-me",
        },
    )
    answers = [
        True, "tvly-key",
        False, False, False, False,
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    integrations.run(ctx)

    env = read_env(tmp_path / ".env")
    assert env["KESTREL_DATA_KEY"] == "fernet-key-stub"
    assert env["OPENAI_API_KEY"] == "sk-x"
    assert env["SOME_USER_VAR"] == "preserve-me"
    assert env["TAVILY_API_KEY"] == "tvly-key"
