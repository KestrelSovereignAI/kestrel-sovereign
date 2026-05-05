# Migration: Rookery → MultiAgent

The `Rookery` concept has been renamed to `MultiAgent` throughout the codebase. The new name describes the concept ("a host running multiple agents") rather than carrying the falconry-specific term that nobody outside the project knew. The old name was an open-source readability problem — see epic #986 for the UI theme system that decoupled user-facing labels from code identifiers, and the follow-up rename for the code itself.

This document is the migration guide for operators with deployments that touch any of the renamed surfaces.

## What changed

**Internal Python (hard rename, no compat):**

| Old | New |
|---|---|
| `kestrel_sovereign.rookery` package | `kestrel_sovereign.multi_agent` |
| `RookeryConfig` class | `MultiAgentConfig` |
| `RookeryAgentRoutingMiddleware` | `MultiAgentAgentRoutingMiddleware` |
| `is_rookery()` / `is_rookery_mode()` | `is_multi_agent()` / `is_multi_agent_mode()` |
| `_check_rookery()` / `load_rookery_config()` | `_check_multi_agent()` / `load_multi_agent_config()` |
| `RookeryConfig.load()` defaults to `multi_agent.toml` | (same loader; new default filename) |
| Test files `tests/unit/test_rookery_*.py` | `tests/unit/test_multi_agent_*.py` |

External code that imported from `kestrel_sovereign.rookery` must update its import paths. There is **no shim** for these — they're internal Python identifiers and updating an import is a one-line change.

**Operator-facing (with backward-compat shims):**

| Surface | Legacy | New | Compat? |
|---|---|---|---|
| Env var | `KESTREL_ROOKERY_CONFIG` | `KESTREL_MULTI_AGENT_CONFIG` | Yes — old still read, deprecation warning |
| Config filename | `rookery.toml` | `multi_agent.toml` | Yes — old still loaded, deprecation warning |
| `deployment_mode` value | `"rookery"` | `"multi_agent"` | Yes — old normalised to new, deprecation warning |
| Deploy profile names | `[profiles.rookery-dev]` / `[profiles.rookery-prod]` | `[profiles.multi-agent-dev]` / `[profiles.multi-agent-prod]` | Yes — `--profile rookery-*` resolves to the new name with a warning |
| Docker image tags | `*-rookery:latest`, `*-rookery:vX` | `*-multi_agent:latest`, `*-multi_agent:vX` | **No** — hard cutover; CI builds the new tag only |
| Dockerfile | `docker/Dockerfile.rookery` | `docker/Dockerfile.multi_agent` | **No** — file renamed |
| Entrypoint | `docker/rookery_entrypoint.sh` | `docker/multi_agent_entrypoint.sh` | **No** — file renamed |
| Build/deploy scripts | `scripts/cloudrun/build_rookery.sh`, `deploy_rookery_dev.sh` | `build_multi_agent.sh`, `deploy_multi_agent_dev.sh` | **No** — file renamed |

The compat shims live in [`kestrel_sovereign/multi_agent/compat.py`](../../kestrel_sovereign/multi_agent/compat.py).

## What you need to do

### If you have a deployed Kestrel instance

1. **Rename your config file** from `rookery.toml` → `multi_agent.toml`. Old name still works but emits a deprecation warning on every startup.
2. **Update your env var** if you set `KESTREL_ROOKERY_CONFIG` → set `KESTREL_MULTI_AGENT_CONFIG` instead. Old name still works with deprecation.
3. **Update `deployment_mode` in your `deploy_config.toml`** if it says `"rookery"` → change to `"multi_agent"`. Old value still accepted with deprecation.

These three steps are **soft** — your deployment keeps working without them. The deprecation warnings just push you to clean up over time.

### If your CI references the old Docker image tag

The CI workflow at `.github/workflows/deploy.yml` has already been updated to build/push `*-multi_agent:latest`. If you have **external** automation pointing at `gcr.io/PROJECT/IMAGE-rookery:latest`, update it to `IMAGE-multi_agent:latest`. There is no compat for image tags — Docker tags are not aliasable without re-pushing.

### If you have a build script that calls the old shell scripts

The scripts at `scripts/cloudrun/build_rookery.sh` and `deploy_rookery_dev.sh` were renamed to `build_multi_agent.sh` and `deploy_multi_agent_dev.sh`. Update any wrappers or runbooks that invoke them.

### If you have a deploy command using `--profile rookery-dev`

Use `--profile multi-agent-dev` instead. The old name still works — `DeployManager.get_profile()` resolves it to the new name with a deprecation warning — so existing automation keeps running. Clean up at your convenience.

## Verifying the migration

After updating your environment, search your logs for `DEPRECATED: KESTREL_ROOKERY_CONFIG`, `DEPRECATED: loading multi-agent config from legacy`, `DEPRECATED: deployment_mode='rookery'`, `DEPRECATED: profile 'rookery-`. If none of those appear, you're fully on the new names.

## When the compat shims will be removed

After at least one full release cycle on the new names. The compat layer is a single module (`kestrel_sovereign/multi_agent/compat.py`); when it's time to drop it, remove the module, delete the import statements, and the deprecation warnings stop firing because the old paths are no longer accepted at all. Operators who haven't migrated by then will get a clean error pointing at the new names.

## Why the rename

The `rookery` term was internal jargon — even the maintainer who liked it admitted it was opaque to outside readers. With the open-source split (epic #462) bringing the project to a wider audience, `multi_agent` reads as plain English and matches the dominant terminology in the AI-agent space. The user-facing UI keeps the falconry character via the theme system shipped in epic #986 (the `falconry` theme renames the multi-agent panel to "Mews"); the code goes plain.
