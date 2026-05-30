# LLM Routing Lane Report

Source: subagent lane review, read-only, 2026-05-30.

## Scope Reviewed

LLM routing lane docs, current LLM code, package metadata, generated feature docs, and stale audit/strategy references.

## Canonical Doc Recommendation

Make `docs/architecture/LLM_SERVICE_ARCHITECTURE.md` the canonical LLM routing doc. It already correctly covers vendor/route/model, `kestrel.toml [llm]`, model preference schema, routing, streaming, and links to provider plugin and honesty-layer docs.

Keep these as canonical companion docs:

- `docs/architecture/LLM_PROVIDER_CAPABILITIES.md`
- `docs/architecture/llm/PROVIDER_PLUGINS.md`
- `docs/architecture/llm/HONESTY_LAYER.md`

## Stale Or Conflicting Claims

- `docs/audit/issues/codex-provider-06-nellie-proof.md` uses old route names `claude_plan` / `openai_plan` and tells operators to configure `llm_config.toml`. Current routes are `anthropic:plan` and `openai:plan` under `kestrel.toml [llm]`.
- `docs/audit/issues/256-auto-model-selection-update.md` says the changed file was `llm_config.toml` and proof says shipped `llm_config.toml` uses `auto`; should be historical or rewritten to `kestrel.toml [llm]`.
- `docs/audit/issues/256-model-catalog-boundary-update.md` says config intent lives in `llm_config.toml` and references `llm_config.toml.example`.
- `docs/strategy/MOLTBOOK_LAUNCH_PLAN.md` says "Does it support X provider? -> Check llm_config.toml".
- `docs/architecture/core/MULTI_MODEL_SUPPORT.md` correctly banners itself as historical, but `docs/architecture/README.md` still labels it Active.
- `KESTREL_FEATURES.md` and generated feature docs list in-tree providers but do not clearly explain the split between in-repo adapters and `kestrel-llms[all]` external provider packages.

## Code/Package Evidence

- `kestrel_sovereign/config.py`: `llm_config.toml` was removed from unified config mapping; LLM config should use `load_section("llm")`.
- `kestrel_sovereign/llm/service.py`: `LLMService` reads `kestrel.toml [llm]`; legacy `llm_config.toml` is not supported except migration warning.
- `kestrel_sovereign/llm/provider_registry.py`: current model is vendor/route/model; in-tree adapters are registered in `_ADAPTER_REGISTRY`; external LLM providers load from `kestrel_sovereign.llm_providers`.
- `pyproject.toml`: SDK floor is `kestrel-sovereign-sdk>=0.17.0,<1`; `kestrel-llms[all]==0.1.8` is installed.
- `kestrel_sovereign/endpoints/models.py`: `/api/models`, `/api/model/current`, and `/api/model/set` use `{vendor, route, model}`.
- `kestrel_sovereign/agent/model_preference.py`: persisted preference schema is `{"vendor", "model", "route"}`; legacy `{model, provider}` rows are ignored.
- `kestrel_sovereign/llm/streaming.py`: streaming-with-tools yields `str | ThinkingDelta | ToolCallStarted | LLMResponse`.
- Provider capability implementations in `kestrel_sovereign/llm/*_adapter.py` match the SDK `ProviderCapabilities` contract.

## Docs To Update

- `docs/architecture/README.md` - mark `core/MULTI_MODEL_SUPPORT.md` as historical, not active.
- `KESTREL_FEATURES.md` - distinguish in-tree LLM adapters from external `kestrel-llms[all]` packages.
- `README.md` - add `ANTHROPIC_AUTH_TOKEN` and `CODEX_AUTH_TOKEN` caveats near LLM env vars, since the config section already documents plan routes.
- `docs/architecture/LLM_SERVICE_ARCHITECTURE.md` - optionally add a short ownership note for `kestrel-llms[all]` versus in-tree adapters.

## Docs To Archive Or Mark Historical

- `docs/audit/issues/codex-provider-06-nellie-proof.md`
- `docs/audit/issues/256-auto-model-selection-update.md`
- `docs/audit/issues/256-model-catalog-boundary-update.md`
- `docs/strategy/MOLTBOOK_LAUNCH_PLAN.md`
- `docs/architecture/core/MULTI_MODEL_SUPPORT.md` should remain historical, but index labeling must match.

## Generated Docs To Regenerate

Regenerate after updating `KESTREL_FEATURES.md`:

- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_developer.md`
- `docs/generated/FEATURES_investor.md`

## Open Questions

- Should `kestrel-llms[all]` providers be described in public docs as bundled default installs, optional plugin examples, or supported-but-external packages?
- Should old audit issue docs be archived wholesale, or lightly patched with historical banners?
- Should README expose subscription-plan routes (`anthropic:plan`, `openai:plan`) in the main config example, or keep them in architecture docs only?

## Suggested First PR Slice

Fix the architecture index status, patch README env caveats, update `KESTREL_FEATURES.md` with in-tree vs `kestrel-llms` ownership, regenerate generated feature docs, and add historical banners to the three stale audit issue docs.

