"""Optional cloud integration credentials step.

Captures API keys for integrations the user wants to enable, writes them
to ``.env``, and records the user's selection in
``[features.managed]`` of ``kestrel.toml`` so subsequent ``--check`` /
``--quickstart`` runs can validate that required keys are still present.

Scope (v1) — strictly credential capture + selection metadata:

  - Hand-curated list of five integrations: Tavily, ElevenLabs, Deepgram,
    Hugging Face, RunPod. Adding a sixth means a code change.
  - We do NOT pip install, do NOT mutate agent DBs, do NOT touch the
    runtime feature toggle (``[features.disabled]`` is left alone).
  - We do NOT validate keys against provider APIs (no key burning).

Idempotence:

  - Re-running interactively with the same answers produces no diff.
  - Declining a previously-managed integration sets its managed flag
    to ``false`` (rather than removing the key) so intent is auditable.
  - If a user opts into an integration but leaves a required key blank,
    we record a blocker AND do not mark the integration managed —
    "managed" means "user opted in and setup has enough config to
    check it later." Partial config never claims to be configured.

Env-var ground truth was verified against the codebase before
adding each onboarder; see commit message for the grep results.
"""

from __future__ import annotations

from dataclasses import dataclass

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env
from kestrel_sovereign.setup.toml_file import read_toml, write_toml


@dataclass(frozen=True)
class _EnvVar:
    """One environment variable owned by an integration."""

    name: str
    required: bool
    secret: bool
    label: str


@dataclass(frozen=True)
class _Integration:
    """Static metadata for one curated integration."""

    id: str  # internal stable identifier; matches [features.managed].<id>
    label: str  # one-line user-facing name
    description: str  # second-line user-facing summary
    registry_key: str | None  # key in feature_registry.toml when one exists
    env_vars: tuple[_EnvVar, ...]


# ---------------------------------------------------------------------------
# Onboarder table — env vars verified against the codebase, do not guess.
# ---------------------------------------------------------------------------

_INTEGRATIONS: tuple[_Integration, ...] = (
    _Integration(
        id="tavily",
        label="Tavily web search",
        description="Real-time web search results for the agent.",
        registry_key="web_search",
        env_vars=(
            _EnvVar("TAVILY_API_KEY", required=True, secret=True, label="Tavily API key"),
        ),
    ),
    _Integration(
        id="elevenlabs",
        label="ElevenLabs TTS",
        description="High-quality text-to-speech for the voice pipeline.",
        registry_key="voice_elevenlabs",
        env_vars=(
            _EnvVar("ELEVENLABS_API_KEY", required=True, secret=True, label="ElevenLabs API key"),
        ),
    ),
    _Integration(
        id="deepgram",
        label="Deepgram STT",
        description="Cloud speech-to-text for the voice pipeline.",
        registry_key="voice_deepgram",
        env_vars=(
            _EnvVar("DEEPGRAM_API_KEY", required=True, secret=True, label="Deepgram API key"),
        ),
    ),
    _Integration(
        id="huggingface",
        label="Hugging Face",
        description="Read access to gated/private HF model repos.",
        registry_key=None,  # Used by vertex_ai and others; no dedicated registry entry.
        env_vars=(
            _EnvVar("HF_TOKEN", required=True, secret=True, label="Hugging Face token"),
        ),
    ),
    _Integration(
        id="runpod",
        label="RunPod GPU compute",
        description="Remote GPU instances for inference and training.",
        registry_key="cloud",
        env_vars=(
            _EnvVar("RUNPOD_API_KEY", required=True, secret=True, label="RunPod API key"),
        ),
    ),
)

_BY_ID = {i.id: i for i in _INTEGRATIONS}


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run(ctx: SetupContext) -> None:
    """Capture integration credentials per the user's selection."""
    if ctx.flow in (Flow.CHECK, Flow.QUICKSTART):
        # Both flows validate already-managed integrations against .env
        # but never discover new ones or prompt. Quickstart additionally
        # never writes; check is read-only by contract.
        _validate_managed(ctx)
        return

    _interactive_run(ctx)


# ---------------------------------------------------------------------------
# Validation (used by --check and --quickstart)
# ---------------------------------------------------------------------------

def _validate_managed(ctx: SetupContext) -> None:
    """Block when a managed integration is missing a required env var."""
    config = read_toml(ctx.kestrel_toml_path)
    managed: dict = ((config.get("features") or {}).get("managed") or {})
    if not managed:
        return

    env = read_env(ctx.env_path)
    for integration_id, is_managed in managed.items():
        if not is_managed:
            continue
        integration = _BY_ID.get(integration_id)
        if integration is None:
            # User-managed entry the wizard doesn't know about. Don't
            # error — they may have hand-edited or there's a future
            # integration the current wizard hasn't been taught about.
            continue
        for var in integration.env_vars:
            if var.required and not env.get(var.name):
                ctx.block(
                    f"{var.name} not set but [features.managed].{integration_id} = true. "
                    f"Run: kestrel setup integrations"
                )


# ---------------------------------------------------------------------------
# Interactive flow
# ---------------------------------------------------------------------------

def _interactive_run(ctx: SetupContext) -> None:
    """Per-integration confirm + env-var prompts; write atomically at end."""
    config = read_toml(ctx.kestrel_toml_path)
    managed_existing: dict = (
        (config.get("features") or {}).get("managed") or {}
    )
    env = read_env(ctx.env_path)

    env_updates: dict[str, str] = {}
    managed_updates: dict[str, bool] = {}
    any_selected = False

    for integration in _INTEGRATIONS:
        was_managed = bool(managed_existing.get(integration.id))
        prompt_default = was_managed
        wants = ctx.prompter.confirm(
            f"Enable {integration.label}? — {integration.description}",
            default=prompt_default,
        )

        if not wants:
            if was_managed:
                # Explicit decline of a previously-managed integration.
                # Flip the flag to false so intent is recorded.
                managed_updates[integration.id] = False
            # No env-var changes; never delete a user's existing keys.
            continue

        any_selected = True
        captured = _prompt_integration(
            ctx, integration, env_existing=env
        )
        if captured is None:
            # Required key blank — blocker recorded by helper.
            # Do NOT mark managed; partial config never claims to be configured.
            continue

        for k, v in captured.items():
            env_updates[k] = v
        managed_updates[integration.id] = True

    if not env_updates and not managed_updates:
        if any_selected:
            ctx.record("Integrations: nothing changed")
        return

    if env_updates:
        env_result = write_env(ctx.env_path, env_updates)
        if env_result.backup_path is not None:
            ctx.record(f"Backed up existing .env to {env_result.backup_path.name}")
        for key in env_result.added:
            ctx.record(f"Set {key} in .env")
        for key in env_result.updated:
            ctx.record(f"Updated {key} in .env")

    if managed_updates:
        toml_result = write_toml(
            ctx.kestrel_toml_path,
            {"features": {"managed": managed_updates}},
        )
        if toml_result.backup_path is not None:
            ctx.record(
                f"Backed up existing kestrel.toml to {toml_result.backup_path.name}"
            )
        if toml_result.changed:
            enabled = [i for i, on in managed_updates.items() if on]
            disabled = [i for i, on in managed_updates.items() if not on]
            if enabled:
                ctx.record(f"Marked managed: {', '.join(enabled)}")
            if disabled:
                ctx.record(f"Unmanaged: {', '.join(disabled)}")


def _prompt_integration(
    ctx: SetupContext,
    integration: _Integration,
    *,
    env_existing: dict[str, str],
) -> dict[str, str] | None:
    """Collect env vars for one integration.

    Returns the dict of ``{env_var_name: value}`` to merge into ``.env``,
    or ``None`` if a required key was left blank (a blocker is recorded
    and the caller skips marking the integration managed).

    Blank input for an *optional* var is silently dropped — we never
    write empty optional keys to ``.env``.

    Blank input for a *required* var:

      - If a value already exists in ``.env``, this is the
        questionary-default-pressed-enter case: keep the existing
        value (no diff).
      - If no value exists, record a blocker and abort the integration.
    """
    captured: dict[str, str] = {}
    for var in integration.env_vars:
        existing = env_existing.get(var.name, "")
        prompt_fn = ctx.prompter.secret if var.secret else ctx.prompter.text
        new_value = prompt_fn(var.label, default=existing).strip()

        if not new_value:
            if var.required and not existing:
                ctx.block(
                    f"{var.name} required for {integration.label}; left blank"
                )
                return None
            # Optional + blank, or required + existing-kept: skip writing.
            continue

        if new_value != existing:
            captured[var.name] = new_value

    return captured
