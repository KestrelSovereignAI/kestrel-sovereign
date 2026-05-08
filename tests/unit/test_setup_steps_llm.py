"""Unit tests for the llm step."""

from __future__ import annotations

from pathlib import Path

import toml

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env
from kestrel_sovereign.setup.prompts import StubPrompter
from kestrel_sovereign.setup.steps import llm
from kestrel_sovereign.setup.toml_file import read_toml, write_toml


def _make_ctx(tmp_path: Path, flow: Flow, *, answers=None) -> SetupContext:
    return SetupContext(
        project_dir=tmp_path,
        agent_data_root=tmp_path / "agent_data",
        flow=flow,
        prompter=StubPrompter(answers=list(answers or [])),
    )


def test_llm_quickstart_falls_back_to_ollama_when_nothing_detected(
    tmp_path, monkeypatch
):
    """When no cloud API keys are exported and Ollama isn't reachable,
    quickstart still produces a valid config (Ollama-only) so the
    operator has something to work with — they can install Ollama or
    rerun the interactive wizard to pick a cloud vendor.
    """
    # Strip any cloud keys the developer may have exported in their shell.
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Ollama not running.
    monkeypatch.setattr(llm, "_is_ollama_reachable", lambda *a, **kw: False)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    assert config["llm"]["route_priority"] == ["ollama:local"]
    assert config["llm"]["vendors"]["ollama"]["is_cloud"] is False
    assert config["llm"]["vendors"]["ollama"]["routes"]["local"]["adapter"] == "OllamaAdapter"


def test_llm_quickstart_picks_ollama_when_only_ollama_reachable(
    tmp_path, monkeypatch
):
    """Ollama running, no cloud keys → quickstart writes Ollama-only.
    Same on-disk shape as the no-detection fallback, but the path
    that got us there is "we found Ollama" (logged at INFO)."""
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "_is_ollama_reachable", lambda *a, **kw: True)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    assert config["llm"]["route_priority"] == ["ollama:local"]


def test_llm_quickstart_picks_openrouter_when_key_in_env(tmp_path, monkeypatch):
    """``OPENROUTER_API_KEY`` exported in shell → quickstart picks
    OpenRouter (highest cloud priority per the README)."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(llm, "_is_ollama_reachable", lambda *a, **kw: False)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    assert config["llm"]["route_priority"] == ["openrouter:api"]
    assert (
        config["llm"]["vendors"]["openrouter"]["routes"]["api"]["api_key_env"]
        == "OPENROUTER_API_KEY"
    )


def test_llm_quickstart_combines_cloud_and_ollama_when_both_available(
    tmp_path, monkeypatch
):
    """Cloud key + Ollama reachable → cloud first, Ollama as fallback.

    Mirrors the README's "OpenRouter recommended; Ollama for free
    local fallback" recipe.
    """
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(llm, "_is_ollama_reachable", lambda *a, **kw: True)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    assert config["llm"]["route_priority"] == ["anthropic:api", "ollama:local"]


def test_llm_quickstart_orders_multiple_cloud_keys_openrouter_first(
    tmp_path, monkeypatch
):
    """Three cloud keys exported → priority order is OpenRouter,
    Anthropic, OpenAI (matches the README's recommendation)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_is_ollama_reachable", lambda *a, **kw: False)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    assert config["llm"]["route_priority"] == [
        "openrouter:api", "anthropic:api", "openai:api",
    ]


def test_llm_quickstart_reads_keys_from_dotenv_too(tmp_path, monkeypatch):
    """Auto-detect must consult .env in addition to ``os.environ`` —
    the wizard's keys-step may have populated .env earlier in the same
    --quickstart run, and the parent shell may have nothing exported.
    """
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "_is_ollama_reachable", lambda *a, **kw: False)
    write_env(tmp_path / ".env", {"OPENAI_API_KEY": "sk-from-dotenv"})

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    assert config["llm"]["route_priority"] == ["openai:api"]


def test_llm_quickstart_with_no_keys_marks_blocker_for_cloud(tmp_path, monkeypatch):
    """If existing config picks OpenAI but no key is anywhere
    (neither .env nor the parent shell), quickstart blocks. With a key
    in os.environ the wizard now persists it to .env (see
    test_llm_quickstart_promotes_shell_env_keys_to_dotenv); this test
    covers the genuinely-missing case.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    write_toml(
        tmp_path / "kestrel.toml",
        {"llm": {"route_priority": ["openai:api"]}},
    )
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx)
    assert any("OPENAI_API_KEY not set" in b for b in ctx.blockers)


def test_llm_quickstart_promotes_shell_env_keys_to_dotenv(tmp_path, monkeypatch):
    """When a vendor's API key is exported in the parent shell but
    missing from .env, the wizard persists it to .env so the runtime
    (which only reads .env via dotenv) can use it. Without this,
    autodetected cloud vendors would block quickstart even though the
    operator's shell has the right key.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shell-exported")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_is_ollama_reachable", lambda *a, **kw: False)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx)

    config = read_toml(tmp_path / "kestrel.toml")
    env = read_env(tmp_path / ".env")
    # Autodetect picked OpenAI; it's in the priority list.
    assert config["llm"]["route_priority"] == ["openai:api"]
    # And the shell-exported key landed in .env so the runtime can read it.
    assert env["OPENAI_API_KEY"] == "sk-shell-exported"
    # No blocker — the key is satisfied.
    assert ctx.blockers == []


def test_llm_interactive_picks_openai_with_key(tmp_path):
    answers = [
        # Pick OpenAI as primary
        "OpenAI (cloud — needs OPENAI_API_KEY)",
        # Decline every other vendor as a fallback (4 of them)
        False, False, False, False,
        # Provide key
        "sk-test-key",
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    env = read_env(tmp_path / ".env")
    assert config["llm"]["route_priority"] == ["openai:api"]
    assert env["OPENAI_API_KEY"] == "sk-test-key"
    assert config["llm"]["vendors"]["openai"]["routes"]["api"]["api_key_env"] == "OPENAI_API_KEY"


def test_llm_interactive_with_existing_priority_offered_to_keep(tmp_path):
    write_toml(
        tmp_path / "kestrel.toml",
        {
            "llm": {
                "route_priority": ["ollama:local", "openai:api"],
                "vendors": {
                    "ollama": {"is_cloud": False, "routes": {"local": {"adapter": "OllamaAdapter"}}},
                    "openai": {"is_cloud": True, "routes": {"api": {"api_key_env": "OPENAI_API_KEY"}}},
                },
            }
        },
    )
    write_env(tmp_path / ".env", {"OPENAI_API_KEY": "sk-existing"})
    answers = [True]  # Keep existing
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    assert config["llm"]["route_priority"] == ["ollama:local", "openai:api"]


def test_llm_anthropic_path(tmp_path):
    answers = [
        "Anthropic Claude (cloud — needs ANTHROPIC_API_KEY)",
        False,  # No ollama fallback
        False,  # No openai fallback
        False,  # No google fallback
        False,  # No openrouter fallback
        "sk-ant-test",
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    env = read_env(tmp_path / ".env")
    assert config["llm"]["route_priority"] == ["anthropic:api"]
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test"


def test_llm_google_path(tmp_path):
    answers = [
        "Google Gemini (cloud — needs GOOGLE_API_KEY)",
        False, False, False, False,  # No fallbacks
        "AIzaTestKey",
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    env = read_env(tmp_path / ".env")
    assert config["llm"]["route_priority"] == ["google:api"]
    assert env["GOOGLE_API_KEY"] == "AIzaTestKey"
    assert config["llm"]["vendors"]["google"]["routes"]["api"]["adapter"] == "GoogleAdapter"


def test_llm_openrouter_path(tmp_path):
    answers = [
        "OpenRouter (multi-vendor proxy — needs OPENROUTER_API_KEY)",
        False, False, False, False,
        "sk-or-v1-test",
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    env = read_env(tmp_path / ".env")
    assert config["llm"]["route_priority"] == ["openrouter:api"]
    assert env["OPENROUTER_API_KEY"] == "sk-or-v1-test"
    assert config["llm"]["vendors"]["openrouter"]["routes"]["api"]["adapter"] == "OpenRouterAdapter"


def test_llm_multi_vendor_chain(tmp_path):
    """Picking Google primary with Anthropic + OpenRouter as fallbacks."""
    answers = [
        # Primary: Google
        "Google Gemini (cloud — needs GOOGLE_API_KEY)",
        # Per-vendor fallback prompts in declared order: ollama, openai, anthropic, openrouter
        False,  # ollama
        False,  # openai
        True,   # anthropic
        True,   # openrouter
        # API keys for each cloud route in declared order: openai/anthropic/google/openrouter
        # but only google + anthropic + openrouter are selected
        "AIzaPrim",  # google primary
        "sk-ant-fb",  # anthropic fallback
        "sk-or-fb",  # openrouter fallback
    ]
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=answers)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    env = read_env(tmp_path / ".env")

    priority = config["llm"]["route_priority"]
    assert priority[0] == "google:api"
    assert "anthropic:api" in priority
    assert "openrouter:api" in priority
    assert env["GOOGLE_API_KEY"] == "AIzaPrim"
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-fb"
    assert env["OPENROUTER_API_KEY"] == "sk-or-fb"


def test_llm_check_mode_does_not_write(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    llm.run(ctx)
    assert not (tmp_path / "kestrel.toml").exists()
    assert not (tmp_path / ".env").exists()
    # No prior config = blocked
    assert ctx.blockers


def test_llm_check_mode_reports_missing_key_for_existing_cloud_route(tmp_path):
    write_toml(
        tmp_path / "kestrel.toml",
        {"llm": {"route_priority": ["openai:api"], "vendors": {"openai": {"routes": {"api": {"api_key_env": "OPENAI_API_KEY"}}}}}},
    )
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    llm.run(ctx)
    assert any("OPENAI_API_KEY" in b for b in ctx.blockers)


def test_llm_existing_unmanaged_routes_preserved(tmp_path):
    """A user-authored vendor we don't manage in v1 must survive a wizard run."""
    write_toml(
        tmp_path / "kestrel.toml",
        {
            "llm": {
                "route_priority": ["xai:api", "ollama:local"],
                "vendors": {
                    "xai": {"is_cloud": True, "routes": {"api": {"adapter": "XAIAdapter"}}},
                    "ollama": {"is_cloud": False, "routes": {"local": {"adapter": "OllamaAdapter"}}},
                },
            }
        },
    )
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    # The ollama:local managed route should be at the front; xai:api preserved at the tail.
    assert "xai:api" in config["llm"]["route_priority"]
    assert "ollama:local" in config["llm"]["route_priority"]
    # The xai vendor block must survive deep-merge.
    assert config["llm"]["vendors"]["xai"]["routes"]["api"]["adapter"] == "XAIAdapter"


def test_llm_idempotent_quickstart(tmp_path):
    """Running llm.run twice in quickstart must produce the same file."""
    ctx1 = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx1)
    text1 = (tmp_path / "kestrel.toml").read_text()

    ctx2 = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx2)
    text2 = (tmp_path / "kestrel.toml").read_text()

    assert text1 == text2
    backups = list(tmp_path.glob("kestrel.toml.backup-*"))
    assert backups == []  # No backup because no second-write diff


def test_llm_preserves_existing_council_and_voice(tmp_path):
    """[council] and [voice] are user-authored and must not be touched."""
    write_toml(
        tmp_path / "kestrel.toml",
        {
            "llm": {"route_priority": ["openai:api"]},
            "council": {"min_members": 3, "consensus_rule": "unanimous"},
            "voice": {"tts_provider_priority": ["piper"]},
        },
    )
    write_env(tmp_path / ".env", {"OPENAI_API_KEY": "sk-x"})
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    llm.run(ctx)
    config = read_toml(tmp_path / "kestrel.toml")
    assert config["council"] == {"min_members": 3, "consensus_rule": "unanimous"}
    assert config["voice"] == {"tts_provider_priority": ["piper"]}
