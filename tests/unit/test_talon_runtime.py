"""Contracts for Talon's backend-aware runtime control surface."""

from pathlib import Path

import pytest

from kestrel_sovereign.features.talon.runtime import (
    TalonBatchExecution,
    TalonExecution,
    TalonPolicy,
    TalonPreference,
    TalonRuntimeError,
    TalonRuntimeRequest,
    build_talon_batch_invocation,
    build_talon_invocation,
    load_talon_policy_preference,
    sanitize_env_for_backend,
    sanitize_untrusted_env,
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


def test_codex_with_api_key_reports_structural_rule_not_billing():
    # codex+api_key is structurally invalid regardless of billing policy, so the
    # error must name the true reason (codex requires oauth), not the misleading
    # "API-key billing not allowed" that fired first before the reorder (#1925
    # dogfooding finding). Default policy has allow_api_billing=False.
    with pytest.raises(TalonRuntimeError, match="Codex Talon backend requires"):
        build_talon_invocation(
            TalonRuntimeRequest(backend="codex", model="gpt-5.4-mini", auth_lane="api_key"),
            _execution(),
            base_env=_env(),
        )


def test_opencode_with_api_key_reports_structural_rule_not_billing():
    with pytest.raises(TalonRuntimeError, match="OpenCode Talon backend requires"):
        build_talon_invocation(
            TalonRuntimeRequest(backend="opencode", model="some/model", auth_lane="api_key"),
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


def test_sanitize_untrusted_env_strips_all_provider_creds_and_gh_token():
    """F302: verify runs untrusted code — no provider creds, no GitHub token."""
    base = {
        "ANTHROPIC_API_KEY": "sk-ant",
        "ANTHROPIC_AUTH_TOKEN": "oauth",
        "OPENAI_API_KEY": "sk-openai",
        "GOOGLE_API_KEY": "google",
        "GROQ_API_KEY": "groq",
        "KESTREL_API_KEY": "kestrel",
        "KESTREL_DATA_KEY": "fernet",
        "GITHUB_TOKEN": "ghp_x",
        "GH_TOKEN": "ghp_y",
        "GITHUB_PAT": "ghp_z",
        "PATH": "/usr/bin",
        "HOME": "/home/agent",
    }
    env, stripped = sanitize_untrusted_env(base)
    for leaked in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "KESTREL_API_KEY",
        "KESTREL_DATA_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_PAT",
    ):
        assert leaked not in env
        assert leaked in stripped
    # Non-secret operational vars survive.
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/agent"


def test_build_talon_batch_invocation_passes_repo_dir_and_abs_prd(tmp_path):
    prd = tmp_path / "prd.json"
    prd.write_text("{}")
    workspace = tmp_path / "ws"
    invocation = build_talon_batch_invocation(
        TalonRuntimeRequest(),
        TalonBatchExecution(repo="org/repo", prd_path=prd, repo_dir=workspace),
        base_env=_env(),
    )
    assert invocation.argv[0] == "batch"
    assert invocation.argv[invocation.argv.index("--prd") + 1] == str(prd)
    assert invocation.argv[invocation.argv.index("--repo-dir") + 1] == str(workspace)
    # The launched backend must be pinned so batch can't fall back to the
    # parser default ($TALON_BACKEND or claude) and bypass policy (F304).
    assert invocation.argv[invocation.argv.index("--backend") + 1] == "claude"
    assert invocation.argv[invocation.argv.index("--model") + 1] == "opus"
    # Batch runs in a sandbox and legitimately keeps the GitHub token,
    # but provider creds unrelated to the backend are stripped.
    assert invocation.env["GITHUB_TOKEN"] == "ghp_test"


def test_build_talon_batch_invocation_pins_policy_resolved_backend(tmp_path):
    prd = tmp_path / "prd.json"
    prd.write_text("{}")
    execution = TalonBatchExecution(repo="org/repo", prd_path=prd, repo_dir=tmp_path)

    # Policy restricts to codex — the launched argv must say so, never claude.
    by_policy = build_talon_batch_invocation(
        TalonRuntimeRequest(),
        execution,
        policy=TalonPolicy(allowed_backends=("codex",)),
        preference=TalonPreference(default_backend="codex", default_model="gpt-5.5"),
        base_env=_env(),
    )
    assert by_policy.backend == "codex"
    assert by_policy.argv[by_policy.argv.index("--backend") + 1] == "codex"
    assert by_policy.argv[by_policy.argv.index("--codex-model") + 1] == "gpt-5.5"
    assert "--model" not in by_policy.argv

    # Preference alone (permissive policy) also pins codex.
    by_preference = build_talon_batch_invocation(
        TalonRuntimeRequest(),
        execution,
        preference=TalonPreference(default_backend="codex", default_model="gpt-5.5"),
        base_env=_env(),
    )
    assert by_preference.argv[by_preference.argv.index("--backend") + 1] == "codex"


def test_build_talon_batch_invocation_rejects_relative_prd(tmp_path):
    with pytest.raises(TalonRuntimeError, match="absolute"):
        build_talon_batch_invocation(
            TalonRuntimeRequest(),
            TalonBatchExecution(
                repo="org/repo", prd_path=Path("prd.json"), repo_dir=tmp_path
            ),
            base_env=_env(),
        )


def test_build_talon_batch_invocation_rejects_missing_prd(tmp_path):
    with pytest.raises(TalonRuntimeError, match="not found"):
        build_talon_batch_invocation(
            TalonRuntimeRequest(),
            TalonBatchExecution(
                repo="org/repo", prd_path=tmp_path / "nope.json", repo_dir=tmp_path
            ),
            base_env=_env(),
        )


def test_build_talon_batch_invocation_enforces_background_and_backend_policy(tmp_path):
    prd = tmp_path / "prd.json"
    prd.write_text("{}")
    execution = TalonBatchExecution(repo="org/repo", prd_path=prd, repo_dir=tmp_path)
    with pytest.raises(TalonRuntimeError, match="background jobs are disabled"):
        build_talon_batch_invocation(
            TalonRuntimeRequest(),
            execution,
            policy=TalonPolicy(allow_background_jobs=False),
            base_env=_env(),
        )
    with pytest.raises(TalonRuntimeError, match="not allowed by policy"):
        build_talon_batch_invocation(
            TalonRuntimeRequest(backend="codex", auth_lane="oauth"),
            execution,
            policy=TalonPolicy(allowed_backends=("claude",)),
            base_env=_env(),
        )
