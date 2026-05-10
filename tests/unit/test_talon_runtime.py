"""Contracts for Talon's backend-aware runtime control surface."""

from pathlib import Path

import pytest

from kestrel_sovereign.features.talon.runtime import (
    TalonExecution,
    TalonPolicy,
    TalonPreference,
    TalonRuntimeError,
    TalonRuntimeRequest,
    build_talon_invocation,
    load_talon_policy_preference,
    sanitize_env_for_backend,
    write_talon_preference,
)


def _execution(**overrides):
    data = {
        "repo": "org/repo",
        "issue": 42,
        "repo_dir": Path("/tmp/talon-workspace"),
        "worktree_base": Path("/tmp"),
        "worktree": True,
        "max_iterations": 3,
        "max_turns": 50,
        "skip_clarification": True,
        "self_review": True,
    }
    data.update(overrides)
    return TalonExecution(**data)


def _env(**overrides):
    data = {
        "GITHUB_TOKEN": "ghp_test",
        "ANTHROPIC_API_KEY": "sk-ant-should-strip",
        "ANTHROPIC_AUTH_TOKEN": "oauth-should-strip",
        "CLAUDE_AGENT_SDK_FOO": "remove-me",
        "CODEX_AUTH_TOKEN": "codex-token",
    }
    data.update(overrides)
    return data


def test_codex_invocation_uses_single_model_field_and_sanitized_env():
    invocation = build_talon_invocation(
        TalonRuntimeRequest(backend="codex", model="gpt-5.4-mini", auth_lane="oauth"),
        _execution(),
        base_env=_env(),
    )

    assert invocation.backend == "codex"
    assert invocation.model == "gpt-5.4-mini"
    assert "--backend" in invocation.argv
    assert invocation.argv[invocation.argv.index("--backend") + 1] == "codex"
    assert "--codex-model" in invocation.argv
    assert invocation.argv[invocation.argv.index("--codex-model") + 1] == "gpt-5.4-mini"
    assert "--model" not in invocation.argv
    assert "ANTHROPIC_API_KEY" not in invocation.env
    assert "ANTHROPIC_AUTH_TOKEN" not in invocation.env
    assert "CLAUDE_AGENT_SDK_FOO" not in invocation.env
    assert invocation.env["CODEX_AUTH_TOKEN"] == "codex-token"


def test_claude_oauth_strips_api_keys_and_uses_model_alias():
    invocation = build_talon_invocation(
        TalonRuntimeRequest(backend="claude", model="opus", auth_lane="oauth"),
        _execution(),
        base_env=_env(),
    )

    assert invocation.argv[invocation.argv.index("--backend") + 1] == "claude"
    assert invocation.argv[invocation.argv.index("--model") + 1] == "opus"
    assert "--use-api-key" not in invocation.argv
    assert "ANTHROPIC_API_KEY" not in invocation.env
    assert "ANTHROPIC_AUTH_TOKEN" not in invocation.env


def test_claude_api_key_requires_policy_and_preserves_api_key():
    invocation = build_talon_invocation(
        TalonRuntimeRequest(backend="claude", model="sonnet", auth_lane="api_key"),
        _execution(),
        policy=TalonPolicy(allow_api_billing=True),
        base_env=_env(),
    )

    assert "--use-api-key" in invocation.argv
    assert invocation.env["ANTHROPIC_API_KEY"] == "sk-ant-should-strip"
    assert "ANTHROPIC_AUTH_TOKEN" not in invocation.env


def test_api_key_lane_rejected_by_default_policy():
    with pytest.raises(TalonRuntimeError, match="API-key billing"):
        build_talon_invocation(
            TalonRuntimeRequest(backend="claude", model="sonnet", auth_lane="api_key"),
            _execution(),
            base_env=_env(),
        )


def test_opencode_invocation_maps_model_to_opencode_model_flag():
    invocation = build_talon_invocation(
        TalonRuntimeRequest(
            backend="opencode",
            model="kimi-local/kimi-k2.5",
            auth_lane="provider_config",
        ),
        _execution(),
        base_env=_env(),
    )

    assert invocation.argv[invocation.argv.index("--backend") + 1] == "opencode"
    assert "--opencode-model" in invocation.argv
    assert invocation.argv[invocation.argv.index("--opencode-model") + 1] == "kimi-local/kimi-k2.5"


def test_policy_rejects_disallowed_backend():
    with pytest.raises(TalonRuntimeError, match="not allowed"):
        build_talon_invocation(
            TalonRuntimeRequest(backend="codex", model="gpt-5.4-mini", auth_lane="oauth"),
            _execution(),
            policy=TalonPolicy(allowed_backends=("claude",)),
            base_env=_env(),
        )


def test_policy_rejects_worktree_disabled_when_required():
    with pytest.raises(TalonRuntimeError, match="worktree=true"):
        build_talon_invocation(
            TalonRuntimeRequest(backend="codex", model="gpt-5.4-mini", auth_lane="oauth"),
            _execution(worktree=False),
            base_env=_env(),
        )


def test_invalid_claude_model_rejected():
    with pytest.raises(TalonRuntimeError, match="Claude Talon model"):
        build_talon_invocation(
            TalonRuntimeRequest(backend="claude", model="gpt-5.4-mini", auth_lane="oauth"),
            _execution(),
            base_env=_env(),
        )


def test_sanitize_requires_github_token():
    env = _env()
    env.pop("GITHUB_TOKEN")
    with pytest.raises(TalonRuntimeError, match="GITHUB_TOKEN"):
        sanitize_env_for_backend("codex", "oauth", env)


def test_codex_without_model_omits_model_flag():
    invocation = build_talon_invocation(
        TalonRuntimeRequest(backend="codex", auth_lane="oauth"),
        _execution(),
        preference=TalonPreference(default_backend="codex", default_model=""),
        base_env=_env(),
    )

    assert invocation.backend == "codex"
    assert invocation.model is None
    assert "--codex-model" not in invocation.argv


def test_string_false_preference_values_parse_as_false(tmp_path):
    path = tmp_path / "kestrel.toml"

    result = write_talon_preference(
        {
            "default_backend": "codex",
            "default_model": "gpt-5.4-mini",
            "default_auth_lane": "oauth",
            "skip_clarification": "false",
            "self_review": "0",
        },
        kestrel_toml_path=path,
    )
    _policy, preference = load_talon_policy_preference(path)

    assert result["preference"]["skip_clarification"] is False
    assert result["preference"]["self_review"] is False
    assert preference.skip_clarification is False
    assert preference.self_review is False


def test_invalid_string_bool_preference_rejected(tmp_path):
    with pytest.raises(TalonRuntimeError, match="skip_clarification"):
        write_talon_preference(
            {"skip_clarification": "definitely"},
            kestrel_toml_path=tmp_path / "kestrel.toml",
        )
