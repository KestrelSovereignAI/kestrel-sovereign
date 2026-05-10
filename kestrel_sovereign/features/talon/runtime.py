"""Deterministic Talon runtime command, policy, and env handling.

Talon is an external coding-agent process. Its runtime backend is related to,
but intentionally separate from, Kestrel's chat LLM preference. This module is
the narrow seam: given a requested Talon runtime, execution options, and policy,
produce the CLI argv/env that may be launched, or reject before subprocess
dispatch.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from kestrel_sovereign.paths import project_dir
from kestrel_sovereign.setup.toml_file import read_toml, write_toml

Backend = Literal["claude", "codex", "opencode"]
AuthLane = Literal["oauth", "api_key", "provider_config"]

CLAUDE_MODEL_ALIASES = {"opus", "sonnet", "haiku"}
DEFAULT_BACKEND: Backend = "claude"
DEFAULT_MODELS: dict[Backend, str] = {
    "claude": "opus",
    "codex": "",
    "opencode": "",
}
DEFAULT_AUTH_LANES: dict[Backend, AuthLane] = {
    "claude": "oauth",
    "codex": "oauth",
    "opencode": "provider_config",
}

ANTHROPIC_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_API_KEY",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)
CLAUDE_AGENT_KEYS_PREFIXES = (
    "CLAUDECODE",
    "CLAUDE_CODE_",
    "CLAUDE_AGENT_SDK_",
)


class TalonRuntimeError(ValueError):
    """Raised when a Talon runtime request violates policy or schema."""


@dataclass(frozen=True)
class TalonPolicy:
    """Operator-controlled guardrails for Talon dispatch."""

    allowed_backends: tuple[Backend, ...] = ("claude", "codex", "opencode")
    allow_api_billing: bool = False
    require_worktree: bool = True
    require_sandboxed_workspace: bool = True
    allow_background_jobs: bool = True


@dataclass(frozen=True)
class TalonPreference:
    """User/agent-mutatable Talon defaults."""

    default_backend: Backend = DEFAULT_BACKEND
    default_model: str = "opus"
    default_auth_lane: AuthLane | None = None
    max_iterations: int = 3
    max_turns: int = 50
    skip_clarification: bool = True
    self_review: bool = True


@dataclass(frozen=True)
class TalonRuntimeRequest:
    """Desired Talon runtime, before defaults and policy are applied."""

    backend: Backend | None = None
    model: str | None = None
    auth_lane: AuthLane | None = None


@dataclass(frozen=True)
class TalonExecution:
    """Claim execution options that become Talon CLI flags."""

    repo: str
    issue: int
    repo_dir: Path
    worktree_base: Path
    worktree: bool = True
    max_iterations: int = 3
    max_turns: int = 50
    skip_clarification: bool = True
    self_review: bool = True
    quality_checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class TalonInvocation:
    """Launch-ready Talon invocation details."""

    argv: list[str]
    env: dict[str, str]
    backend: Backend
    model: str | None
    auth_lane: AuthLane
    redacted_argv: list[str]
    stripped_env_keys: tuple[str, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "auth_lane": self.auth_lane,
            "command_argv": self.redacted_argv,
            "stripped_env_keys": list(self.stripped_env_keys),
        }


def normalize_backend(value: str | None) -> Backend | None:
    if value is None or value == "":
        return None
    if value not in ("claude", "codex", "opencode"):
        raise TalonRuntimeError(
            "Talon backend must be one of: claude, codex, opencode"
        )
    return value  # type: ignore[return-value]


def normalize_auth_lane(value: str | None) -> AuthLane | None:
    if value is None or value == "":
        return None
    if value not in ("oauth", "api_key", "provider_config"):
        raise TalonRuntimeError(
            "Talon auth_lane must be one of: oauth, api_key, provider_config"
        )
    return value  # type: ignore[return-value]


def parse_talon_bool(value: Any, key: str) -> bool:
    """Parse bool-like tool/config values without truthiness surprises."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "y", "on"):
            return True
        if normalized in ("false", "0", "no", "n", "off"):
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise TalonRuntimeError(
        f"Talon {key} must be a boolean or one of true/false, yes/no, 1/0"
    )


def resolve_runtime(
    request: TalonRuntimeRequest,
    preference: TalonPreference,
    policy: TalonPolicy,
) -> tuple[Backend, str | None, AuthLane]:
    backend = request.backend or preference.default_backend
    if backend not in policy.allowed_backends:
        raise TalonRuntimeError(f"Talon backend '{backend}' is not allowed by policy")

    auth_lane = (
        request.auth_lane
        or preference.default_auth_lane
        or DEFAULT_AUTH_LANES[backend]
    )
    if auth_lane == "api_key" and not policy.allow_api_billing:
        raise TalonRuntimeError("Talon API-key billing is not allowed by policy")

    if backend == "claude" and auth_lane not in ("oauth", "api_key"):
        raise TalonRuntimeError("Claude Talon backend supports oauth or api_key auth_lane")
    if backend == "codex" and auth_lane != "oauth":
        raise TalonRuntimeError("Codex Talon backend requires auth_lane='oauth'")
    if backend == "opencode" and auth_lane != "provider_config":
        raise TalonRuntimeError("OpenCode Talon backend requires auth_lane='provider_config'")

    model = request.model
    if model is None:
        if preference.default_backend == backend and preference.default_model:
            model = preference.default_model
        else:
            model = DEFAULT_MODELS[backend] or None

    _validate_model(backend, model)
    return backend, model, auth_lane


def build_talon_invocation(
    request: TalonRuntimeRequest,
    execution: TalonExecution,
    policy: TalonPolicy | None = None,
    preference: TalonPreference | None = None,
    base_env: Mapping[str, str] | None = None,
) -> TalonInvocation:
    policy = policy or TalonPolicy()
    preference = preference or TalonPreference()

    if policy.require_worktree and not execution.worktree:
        raise TalonRuntimeError("Talon policy requires worktree=true")
    if not policy.allow_background_jobs:
        raise TalonRuntimeError("Talon background jobs are disabled by policy")
    if execution.max_iterations < 1:
        raise TalonRuntimeError("Talon max_iterations must be >= 1")
    if execution.max_turns < 1:
        raise TalonRuntimeError("Talon max_turns must be >= 1")

    backend, model, auth_lane = resolve_runtime(request, preference, policy)
    argv = _build_claim_argv(backend, model, auth_lane, execution)
    env, stripped = sanitize_env_for_backend(backend, auth_lane, base_env)
    return TalonInvocation(
        argv=argv,
        env=env,
        backend=backend,
        model=model,
        auth_lane=auth_lane,
        redacted_argv=list(argv),
        stripped_env_keys=stripped,
    )


def sanitize_env_for_backend(
    backend: Backend,
    auth_lane: AuthLane,
    base_env: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    env = dict(base_env or os.environ)
    stripped: list[str] = []

    if backend == "claude":
        if auth_lane == "oauth":
            stripped.extend(_pop_keys(env, ANTHROPIC_KEYS))
        else:
            if not env.get("ANTHROPIC_API_KEY"):
                raise TalonRuntimeError(
                    "Talon Claude API-key lane requires ANTHROPIC_API_KEY"
                )
            stripped.extend(_pop_keys(env, ("ANTHROPIC_AUTH_TOKEN",)))
    elif backend == "codex":
        stripped.extend(_pop_keys(env, ANTHROPIC_KEYS))
        stripped.extend(_pop_prefixed(env, CLAUDE_AGENT_KEYS_PREFIXES))
    elif backend == "opencode":
        stripped.extend(_pop_prefixed(env, CLAUDE_AGENT_KEYS_PREFIXES))

    gh_token = env.get("GH_TOKEN") or env.get("GITHUB_TOKEN") or env.get("GITHUB_PAT")
    if not gh_token:
        raise TalonRuntimeError(
            "kestrel-talon needs GITHUB_TOKEN, GH_TOKEN, or GITHUB_PAT "
            "in the kestrel-sovereign environment to access GitHub."
        )
    env.setdefault("GITHUB_TOKEN", gh_token)
    env.setdefault("GH_TOKEN", gh_token)
    return env, tuple(stripped)


def load_talon_policy_preference(
    kestrel_toml_path: Path | None = None,
) -> tuple[TalonPolicy, TalonPreference]:
    path = kestrel_toml_path or (project_dir() / "kestrel.toml")
    talon = read_toml(path).get("talon", {})
    policy_data = talon.get("policy", {}) if isinstance(talon, dict) else {}
    preference_data = talon.get("preference", {}) if isinstance(talon, dict) else {}
    return _policy_from_mapping(policy_data), _preference_from_mapping(preference_data)


def write_talon_preference(
    updates: Mapping[str, Any],
    kestrel_toml_path: Path | None = None,
) -> dict[str, Any]:
    """Persist mutable Talon preference keys under ``[talon.preference]``."""
    path = kestrel_toml_path or (project_dir() / "kestrel.toml")
    current_policy, current_preference = load_talon_policy_preference(path)
    merged = asdict(current_preference)
    for key, value in updates.items():
        if key not in merged:
            raise TalonRuntimeError(f"Unknown Talon preference key: {key}")
        if value is not None:
            merged[key] = value

    preference = _preference_from_mapping(merged)
    resolve_runtime(
        TalonRuntimeRequest(
            backend=preference.default_backend,
            model=preference.default_model or None,
            auth_lane=preference.default_auth_lane,
        ),
        preference,
        current_policy,
    )
    result = write_toml(path, {"talon": {"preference": asdict(preference)}})
    return {
        "changed": result.changed,
        "path": str(result.path),
        "backup_path": str(result.backup_path) if result.backup_path else None,
        "preference": asdict(preference),
    }


def _build_claim_argv(
    backend: Backend,
    model: str | None,
    auth_lane: AuthLane,
    execution: TalonExecution,
) -> list[str]:
    argv = [
        "claim",
        "--repo", execution.repo,
        "--issue", str(execution.issue),
        "--backend", backend,
        "--repo-dir", str(execution.repo_dir),
        "--max-iterations", str(execution.max_iterations),
        "--max-turns", str(execution.max_turns),
    ]
    if backend == "claude":
        if model:
            argv += ["--model", model]
        if auth_lane == "api_key":
            argv.append("--use-api-key")
    elif backend == "codex":
        if model:
            argv += ["--codex-model", model]
    elif backend == "opencode":
        if model:
            argv += ["--opencode-model", model]

    if execution.worktree:
        argv += ["--worktree", "--worktree-base", str(execution.worktree_base)]
    if execution.skip_clarification:
        argv.append("--skip-clarification")
    if execution.self_review:
        argv.append("--self-review")
    for check in execution.quality_checks:
        argv += ["--quality-check", check]
    return argv


def _validate_model(backend: Backend, model: str | None) -> None:
    if backend == "claude" and model not in CLAUDE_MODEL_ALIASES:
        raise TalonRuntimeError(
            "Claude Talon model must be one of: opus, sonnet, haiku"
        )
    if backend in ("codex", "opencode") and model is not None and not model.strip():
        raise TalonRuntimeError(f"{backend} Talon model cannot be blank")


def _policy_from_mapping(data: Mapping[str, Any]) -> TalonPolicy:
    allowed_raw = data.get("allowed_backends", ("claude", "codex", "opencode"))
    allowed = tuple(normalize_backend(str(v)) for v in allowed_raw)
    if not allowed or any(v is None for v in allowed):
        raise TalonRuntimeError("talon.policy.allowed_backends cannot be empty")
    return TalonPolicy(
        allowed_backends=allowed,  # type: ignore[arg-type]
        allow_api_billing=parse_talon_bool(
            data.get("allow_api_billing", False), "policy.allow_api_billing"
        ),
        require_worktree=parse_talon_bool(
            data.get("require_worktree", True), "policy.require_worktree"
        ),
        require_sandboxed_workspace=parse_talon_bool(
            data.get("require_sandboxed_workspace", True),
            "policy.require_sandboxed_workspace",
        ),
        allow_background_jobs=parse_talon_bool(
            data.get("allow_background_jobs", True), "policy.allow_background_jobs"
        ),
    )


def _preference_from_mapping(data: Mapping[str, Any]) -> TalonPreference:
    backend = normalize_backend(str(data.get("default_backend", DEFAULT_BACKEND)))
    auth_lane = normalize_auth_lane(data.get("default_auth_lane"))
    pref = TalonPreference(
        default_backend=backend or DEFAULT_BACKEND,
        default_model=str(
            data.get("default_model", DEFAULT_MODELS[backend or DEFAULT_BACKEND])
        ),
        default_auth_lane=auth_lane,
        max_iterations=int(data.get("max_iterations", 3)),
        max_turns=int(data.get("max_turns", 50)),
        skip_clarification=parse_talon_bool(
            data.get("skip_clarification", True), "preference.skip_clarification"
        ),
        self_review=parse_talon_bool(
            data.get("self_review", True), "preference.self_review"
        ),
    )
    if pref.max_iterations < 1:
        raise TalonRuntimeError("talon.preference.max_iterations must be >= 1")
    if pref.max_turns < 1:
        raise TalonRuntimeError("talon.preference.max_turns must be >= 1")
    return pref


def _pop_keys(env: dict[str, str], keys: tuple[str, ...]) -> list[str]:
    removed = []
    for key in keys:
        if key in env:
            env.pop(key, None)
            removed.append(key)
    return removed


def _pop_prefixed(env: dict[str, str], prefixes: tuple[str, ...]) -> list[str]:
    removed = []
    for key in list(env):
        if key.startswith(prefixes):
            env.pop(key, None)
            removed.append(key)
    return removed
