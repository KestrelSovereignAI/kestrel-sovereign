---
type: Audit Ledger
title: Documentation Audit - May 2026
description: 'Status: working audit ledger Created: 2026-05-30 Scope: Kestrel Sovereign
  public, operator, architecture, generated, and audit documentation after the recent
  package extraction,...'
resource: /docs/audit/DOCUMENTATION_AUDIT_5_2026.md
tags:
- docs
- audit
- audit-ledger
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: documentation
canonical: false
generated: false
privacy: public
---

# Documentation Audit - May 2026

Status: working audit ledger  
Created: 2026-05-30  
Scope: Kestrel Sovereign public, operator, architecture, generated, and audit documentation after the recent package extraction, context-management, memory, retrieval, storage, LLM, signal, and Talon work.

## Executive Summary

Kestrel's documentation is broad and valuable, but it now carries several overlapping eras of the project:

- The original monorepo/core-feature era.
- The feature package extraction era.
- The newer runtime package ecosystem: `kestrel-sovereign-sdk`, `kestrel-llms`, `kestrel-feature-*`, `kestrel-cloud-*`, `kestrel-storage-*`, `kestrel-voice-*`, and `kestrel-talon`.
- The recent context and memory architecture work: cache-stable history pruning, canonical conversation history vs rendered transport form, retrieval gates, vector search backends, encryption backfill, and saved-item search.
- The current signal-driven runtime model, where heartbeats, cron, A2A, webhooks, and task completions wake the agent through `SignalDispatcher`.

The top risk is not missing prose. The top risk is multiple docs claiming to be canonical while disagreeing about what is in core, what is an installed package, what is a provider package, and what behavior is currently shipped.

This audit should become the working ledger for cleanup PRs. The recommended order is:

1. Fix source-of-truth drift in feature/package docs.
2. Reconcile the main entry points: `README.md`, `docs/README.md`, `docs/architecture/README.md`, and `KESTREL_FEATURES.md`.
3. Re-audit the volatile systems: context, memory, storage, LLMs, signals, workflows, Talon, cloud, and voice.
4. Regenerate derived docs only after canonical inputs are correct.
5. Move stale aspirational or pre-extraction docs into archive, or add explicit status banners.

## Follow-Up Cleanup Status

2026-05-31 package-boundary cleanup:

- `README.md` no longer says the base `kestrel-sovereign` install includes voice or wallet/economics features. It now distinguishes bundled core, optional feature packages, and provider packages.
- `README.md` now distinguishes completed SQLAlchemy/vector storage groundwork from the still-ongoing provider embedding standardization work.
- `kestrel_sovereign/data/feature_registry.toml` now documents that the catalog contains bundled core entries, optional feature packages, and provider package entries used by the Feature Store UI.
- `docs/guides/BUILDING_FEATURES.md` now points external feature package authors at the SDK import surface.
- Generated audience docs remain intentionally untouched until canonical inventories and runtime docs are reconciled enough to regenerate them safely.

2026-05-31 memory/storage embedding cleanup:

- `docs/architecture/storage/STORAGE_ARCHITECTURE.md` was rewritten from a stale pre-async storage plan into a current implementation snapshot covering `AsyncStorage`, `AsyncDatabase`, SQLAlchemy session factories, vector backends, and startup migrations.
- `docs/architecture/MEMORY_SYSTEM.md` now reflects the actual six-factor `MemoryRetriever` weights and calls out that cognitive memory still uses keyword/concept overlap while `conversation_history.embedding_vec` prepares the vector path.
- `docs/architecture/LLM_SERVICE_ARCHITECTURE.md` now states the current embedding execution truth: embeddings still flow through the Ollama-backed `EmbeddingService`; provider-standard embedding functions are architecture direction, not shipped behavior yet.

## Evidence Snapshot

Repository observations from 2026-05-30:

- `docs/` contains roughly 285 documentation-like files.
- `README.md`, `docs/README.md`, `docs/architecture/README.md`, `KESTREL_FEATURES.md`, `docs/generated/*`, and `kestrel_sovereign/data/feature_registry.toml` all act like navigation or inventory surfaces.
- Recent history includes:
  - Feature extraction work: workflows and feature-management removal from core, voice extraction, reflection/council extraction, wallet extraction, GitHub feature extraction, observability extraction.
  - Runtime/package work: SDK floors, `kestrel-llms`, provider capabilities, feature-owned CLI adapters.
  - Context work: context assessment/redesign, durable-salvage design, lumpy prune for cache-stable history prefix.
  - Memory/storage work: canonical vs rendered conversation history split, retrieval gates, SQLAlchemy vector search, pgvector kNN, saved-item search through sovereign vector backends, external-ref asset restoration.
  - Signals and orchestration work: signal prompt templates, SignalDispatcher constitutional injection, workflows stage-to-signal mapping, A2A replacing mesh.
  - LLM routing work: standalone `llm_config.toml` removal, provider capability declarations, transport stall handling, Codex provider retry and error surfacing.
- The working tree has unrelated local changes and generated/runtime files. This audit file is the only intended documentation edit in this pass.

## Canonical Source Inventory

These files currently behave as source-of-truth documents and need explicit ownership rules:

| Surface | Current role | Risk | Recommended owner |
|---|---|---|---|
| `README.md` | Main public narrative, quick start, feature stability, architecture links | High drift risk because it mixes product narrative, install docs, feature status, and code map | Public entry point only; link out for canonical inventories |
| `docs/README.md` | Documentation index | Medium drift risk; links many directories and derived docs | Navigation only |
| `docs/architecture/README.md` | Architecture index with status labels | High drift risk; lists docs that may no longer exist or may predate extraction | Architecture map with explicit status taxonomy |
| `KESTREL_FEATURES.md` | Declared canonical maintained feature inventory | Critical drift risk; consumed by generated docs | Canonical feature and public surface inventory |
| `docs/generated/*.md` | Audience-specific generated docs | High risk if canonical input is stale | Derived artifacts only |
| `kestrel_sovereign/data/feature_registry.toml` | Runtime/static feature store catalog | Critical drift risk; user-facing install/enable surface | Runtime catalog; should be machine-checkable |
| `docs/audit/*` | Engineering quality and audit program | Medium risk; useful, but audit docs can become stale if not tied to code | Historical plus active audit ledgers |
| `docs/deployment/README.md` | Cloud Run operator runbook | Medium risk; seems focused and current but depends on deploy CLI behavior | Operator runbook |

## Priority Findings

### P0 - Feature Inventory Appears Stale After Extraction

`KESTREL_FEATURES.md` says the current audited snapshot has 35 discoverable modules and lists `feature_features`, `talon`, and `workflows` as core feature modules/classes. The May 2026 package-boundaries lane found 33 local discoverable feature classes in current code, with no local `voice` source and an empty `workflows` source directory. That count mismatch is itself part of the drift. Recent history includes `chore(features): remove in-core workflows + feature_features (extracted)`, and project instructions say autonomous GitHub issue processing is handled by standalone `kestrel-talon`.

Impact:

- Generated docs derived from `KESTREL_FEATURES.md` may publish incorrect core/add-on boundaries.
- User-facing docs may tell operators they already have functionality that now requires a package install.
- Agent/tool docs may route future work to the wrong repository.

Action:

- Recompute discoverable core features from `kestrel_sovereign/features/__init__.py`.
- Update `KESTREL_FEATURES.md` to distinguish:
  - core features in this repository,
  - external feature packages,
  - provider packages,
  - app surfaces such as the hosted GitHub bot,
  - standalone tools such as `kestrel-talon`.
- Regenerate `docs/generated/FEATURES_*.md` only after the canonical inventory is fixed.

### P0 - Runtime Feature Registry Has Ambiguous Core Flags

`kestrel_sovereign/data/feature_registry.toml` lists several entries with external package names while also setting `core = true`, including entries such as channels, web search, scheduler, webhooks, wellness, privacy, and talon. That may be intentional transitional metadata, but it conflicts with the extraction narrative.

Impact:

- The CLI, API, and Feature Store UI may present install state incorrectly.
- Documentation cannot safely say what `pip install kestrel-sovereign` includes.
- Support and onboarding docs become ambiguous: "core" may mean shipped, known, built-in, currently vendored, or default-enabled.

Action:

- Define exact meanings for `core`, `package`, and `features` in the registry header.
- Add a distinct field if needed, such as `bundled = true`, `known_package = true`, or `default_enabled = true`.
- Add a small registry validation test that catches impossible combinations.
- Update `docs/guides/BUILDING_FEATURES.md` and `README.md` after the field semantics are stable.

### P0 - Core vs Add-On Narrative Is Inconsistent

`README.md` says `pip install kestrel-sovereign` includes "voice (Piper TTS + FasterWhisper STT)" while `feature_registry.toml` lists `voice` as `kestrel-feature-voice` with `core = false`. The README also says cloud providers, MCP, GitHub App, wallet, and training adapters are add-ons, but other docs still describe some of those systems as in-core or active architecture.

Impact:

- Users cannot tell what they get from the base install.
- Developers cannot tell whether to patch this repo or an extracted package.
- Generated docs may accidentally reintroduce stale monolith claims.

Action:

- Add a single "Package Boundaries" section to `README.md`.
- Mirror that section in `docs/README.md` and `docs/architecture/README.md`.
- Make package categories explicit:
  - framework core,
  - SDK contracts,
  - LLM provider bundle,
  - feature packages,
  - provider packages,
  - standalone operational tools.

### P1 - Architecture Index References Potentially Missing or Stale Paths

`docs/architecture/README.md` links to several paths under `docs/architecture/core/`, `docs/architecture/storage/`, and `docs/architecture/security/`. Those directories now exist, but the index still reads like a PRD-era map and may overstate "Active" status for docs that predate recent extraction or runtime redesigns.

Impact:

- Readers may trust a design doc that is now only partly true.
- New contributors may implement against old abstractions.

Action:

- Add a required status banner to every architecture doc:
  - Active current behavior,
  - Active but package moved,
  - Design-of-record,
  - Historical,
  - Aspirational,
  - Needs re-audit.
- For each "Active" label, name the current owning code path or package.
- Make `docs/architecture/README.md` index only active and design-of-record docs. Move old planning material into an archive section.

### P1 - Context Management Docs Need A Fresh Canonical Story

Relevant docs include:

- `docs/architecture/CONTEXT_SYSTEM_DESIGN.md`
- `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md`
- `docs/architecture/LLM_SERVICE_ARCHITECTURE.md`
- `docs/research/LLAMA_SERVER_CACHE_FLAGS.md`
- generated feature docs that mention context utilization

Recent code/history mentions lumpy pruning for cache-stable history prefixes, canonical conversation history vs rendered transport form, route token caps, feature-prompt skipping, and durable-salvage design work.

Impact:

- Context behavior is a key user-facing reliability feature.
- Stale docs here lead to regressions in prompt assembly, memory retrieval, and provider routing.

Action:

- Pick one canonical context architecture doc.
- Update it to cover:
  - canonical conversation history vs transport rendering,
  - route-specific token budgets,
  - cache-stable prefix pruning,
  - retrieval insertion gates,
  - feature prompt budget handling,
  - provider-specific transport constraints,
  - diagnostics exposed through `/api/agent/context-status` or CLI surfaces.
- Demote or archive older context design docs after cross-linking them.

### P1 - Memory, Retrieval, And Storage Docs Need Reconciliation

Relevant docs include:

- `docs/architecture/MEMORY_SYSTEM.md`
- `docs/architecture/MEMORY_OWNERSHIP.md`
- `docs/architecture/storage/STORAGE_ARCHITECTURE.md`
- `docs/architecture/storage/HUMAN_MEMORY_SYSTEM.md`
- `docs/architecture/storage/DECENTRALIZED_STORAGE.md`
- `docs/SOVEREIGNTY.md`
- `docs/user-documentation/SOVEREIGNTY_USER_GUIDE.md`

Recent work includes per-turn retrieval gates, per-result similarity floors, vector-search backends, pgvector kNN, saved-item search through vector backends, encryption backfill, external-ref asset restoration, and the canonical/rendered history split.

Impact:

- Memory is one of the three pillars in the README.
- Users and operators need to know what is encrypted, searchable, exportable, local, cloud-backed, or provider-dependent.
- Developers need to know which layer owns memory agency, strategic memory, saved items, and retrieval.

Action:

- Create or update a single `docs/architecture/MEMORY_AND_RETRIEVAL.md` or make `MEMORY_SYSTEM.md` the canonical home.
- Reconcile memory docs with:
  - storage backends,
  - retrieval gates,
  - vector capabilities,
  - saved-item search,
  - encryption/backfill,
  - export/import and external refs,
  - privacy mode effects.
- Add a user-facing summary to `docs/user-documentation/KEY_CONCEPTS_EXPLAINED.md` or equivalent.

### P1 - LLM Provider Docs Need Package Boundary Updates

Relevant docs include:

- `docs/architecture/LLM_SERVICE_ARCHITECTURE.md`
- `docs/architecture/LLM_PROVIDER_CAPABILITIES.md`
- `docs/architecture/llm/PROVIDER_PLUGINS.md`
- `docs/architecture/llm/HONESTY_LAYER.md`
- README sections on model selection and `llm_config.toml` migration

Recent work includes provider capability declarations, `kestrel-llms[all]`, OpenAI 2.x, Codex transport handling, Anthropic prefix stripping, provider default resolution, and retirement of standalone `llm_config.toml`.

Impact:

- Model selection had a prior "what not to do" lesson: multiple APIs and fallback paths caused drift.
- Documentation must reinforce the single source of truth and current runtime routing path.

Action:

- Verify that all LLM docs point to `kestrel.toml [llm]` and do not resurrect `llm_config.toml`.
- Document what lives in `kestrel-sovereign` vs `kestrel-llms` vs third-party provider plugins.
- Ensure provider capability docs match SDK-owned contracts.
- Add a short "Do not add another model preference API" warning near contributor docs.

### P1 - Signals, Workflows, And Wakeups Need One Map

Relevant docs include:

- `docs/architecture/SIGNAL_DISPATCHER.md`
- `docs/architecture/SIGNAL_SOURCES_GUIDE.md`
- `docs/architecture/WORKFLOWS_FEATURE_DESIGN.md`
- `docs/architecture/WORKFLOWS_DEVELOPER_GUIDE.md`
- `docs/architecture/WORKFLOWS_STAGE_TO_SIGNAL_MAPPING.md`
- `docs/architecture/WORKFLOWS_REFLECTION_CYCLE_MIGRATION.md`
- `kestrel_sovereign/signals/sources/`

The project instructions now define signals as anything that wakes the bird, with hooks explicitly not being signals. Recent history also says workflows and feature-management were extracted from core.

Impact:

- Signals are now a central runtime concept.
- Workflow docs may still describe in-core behavior.
- Contributors need a clear extension path for new wake sources.

Action:

- Make `SIGNAL_DISPATCHER.md` the canonical architecture doc.
- Make `SIGNAL_SOURCES_GUIDE.md` the canonical extension guide.
- Re-label workflow docs by current package ownership and shipped status.
- Add a one-page signal source inventory generated or checked against `kestrel_sovereign/signals/sources/`.

### P1 - Talon Docs Need Separation Between Kestrel Feature, Runtime Preferences, And Standalone Package

Project instructions say autonomous GitHub issue processing is handled by standalone `kestrel-talon`, installed as a dependency. Runtime Talon preferences are controlled by `talon_get_config` / `talon_set_config`, while operator policy remains in `[talon.policy]`.

Impact:

- Contributors may patch in-core Talon code when they should patch `kestrel-talon`.
- Operators may confuse Kestrel chat LLM routing with Talon backend routing.

Action:

- Add a Talon boundary section to `README.md` and `docs/README.md`.
- Update generated feature docs after `KESTREL_FEATURES.md` is corrected.
- Ensure Talon docs distinguish:
  - standalone CLI/package,
  - in-agent control surface,
  - runtime preferences,
  - operator policy,
  - backend auth lanes.

### P2 - Cloud, GPU, And Training Docs Span Multiple Eras

Relevant docs include:

- `docs/deployment/README.md`
- `docs/architecture/TRAINING_PROVIDER_ARCHITECTURE.md`
- `docs/architecture/RUNPOD_LORA_TRAINING.md`
- `docs/architecture/VASTAI_TRAINING.md`
- `docs/architecture/PLAN_RUNPOD_INTEGRATION.md`
- `docs/research/LoRA/*`

The README says Cloud Run remains the core deployment path, while RunPod, Vast.ai, GCP Compute, and training adapters have been extracted or are provider packages. Some architecture docs still reference old module paths such as `features.runpod.runpod_manager`.

Impact:

- Operators may use outdated commands or expect in-core provider classes.
- Developers may modify old adapters instead of extracted provider packages.

Action:

- Split cloud docs into:
  - Cloud Run deployment, owned by this repo,
  - cloud provider package docs, owned by extracted packages,
  - research notes for LoRA/provider exploration.
- Add status banners to RunPod/Vast/LoRA docs.
- Remove or flag stale module paths.

### P2 - User Documentation Needs Product-Audience Refresh

Relevant directories:

- `docs/user-documentation/`
- `docs/use_cases/`
- `docs/demos/`
- `docs/concepts/`
- `docs/design/launch/`

The user docs are rich, but may not reflect package extraction, current install surfaces, or the newer memory/context behavior.

Impact:

- End users may see internal architecture or old feature availability.
- Launch/demo copy may promise functionality that moved into optional packages.

Action:

- Audit user docs for claims about:
  - included voice,
  - wallet availability,
  - GitHub automation,
  - cloud GPUs,
  - memory export/import,
  - privacy modes,
  - persistent context.
- Keep end-user docs benefit-oriented, but add "requires optional package" where needed.
- Confirm demo scripts match current UI and CLI.

### P2 - Diagrams Need Status And Date Stamps

Relevant directories:

- `docs/diagrams/`
- `docs/diagrams/data-architecture/`

The diagrams are useful, but architecture diagrams age quickly after package extraction and storage redesign.

Impact:

- Visual docs can mislead faster than prose.

Action:

- Add a short banner to each diagram doc:
  - last validated date,
  - active vs historical,
  - owning architecture doc.
- Prioritize data architecture diagrams after memory/storage reconciliation.

### P3 - Public Release Hygiene Still Needs A Pass

`docs/README.md` warns that `business/`, `outreach/`, and `legal/` may contain identifying information. The repository now also contains `docs/strategy`, `docs/vision`, `docs/planning`, `docs/plans`, and archive material.

Impact:

- Public release may expose internal planning, business, legal, or personal material.

Action:

- Add a pre-release doc hygiene checklist.
- Decide which directories are public, internal, archived, or gitignored.
- Keep public docs separate from internal strategy material where possible.

## High-Traffic Docs To Update First

1. `KESTREL_FEATURES.md`
   - Recompute current feature list.
   - Remove or reclassify extracted packages.
   - Make package ownership explicit.

2. `kestrel_sovereign/data/feature_registry.toml`
   - Clarify registry semantics.
   - Correct misleading `core` flags or document transitional meaning.

3. `README.md`
   - Reduce canonical burden.
   - Add package boundary section.
   - Fix feature stability anchor/version drift.
   - Check quick start against current CLI behavior.

4. `docs/README.md`
   - Update navigation.
   - Add "what is canonical" rules.
   - Link this audit.

5. `docs/architecture/README.md`
   - Revalidate status labels.
   - Add package ownership in index entries.
   - Flag stale docs explicitly.

6. `docs/guides/BUILDING_FEATURES.md`
   - Ensure it matches entry-point discovery, package scaffolding, feature-owned CLI adapters, provider packages, and current SDK contracts.

7. `docs/generated/FEATURES_*.md`
   - Regenerate only after canonical source fixes.

## System-Specific Audit Queue

### Package Extraction

Questions to answer:

- What does `pip install kestrel-sovereign` include today?
- What must be installed separately?
- Which packages are feature packages vs provider packages?
- Which packages depend on `kestrel-sovereign-sdk` only?
- Which packages still depend on this framework package?
- Which extracted packages are private, public, or planned?

Target docs:

- `README.md`
- `KESTREL_FEATURES.md`
- `docs/guides/BUILDING_FEATURES.md`
- `docs/architecture/core/MODULAR_RUNTIME.md`
- `docs/architecture/tools/*`

### Context

Questions to answer:

- What is the canonical prompt assembly pipeline?
- Which history form is persisted?
- Which history form is sent to providers?
- How are token budgets allocated by route?
- When are feature prompts skipped?
- What makes history cache-stable?
- How do retrieval results enter or fail to enter context?

Target docs:

- `docs/architecture/CONTEXT_SYSTEM_DESIGN.md`
- `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md`
- `docs/architecture/LLM_SERVICE_ARCHITECTURE.md`
- `docs/generated/FEATURES_developer.md`

### Memory, Retrieval, And Storage

Questions to answer:

- Which storage layer owns conversation history, memory graph, saved items, files, and external refs?
- Which backends support vector search?
- What is the fallback path when vector search is unavailable?
- How are privacy modes enforced at storage and retrieval time?
- What is encrypted at rest, and how is backfill handled?
- What export/import format preserves identity, memory, and external refs?

Target docs:

- `docs/architecture/MEMORY_SYSTEM.md`
- `docs/architecture/MEMORY_OWNERSHIP.md`
- `docs/architecture/storage/STORAGE_ARCHITECTURE.md`
- `docs/SOVEREIGNTY.md`
- `docs/user-documentation/SOVEREIGNTY_USER_GUIDE.md`

### LLM Routing

Questions to answer:

- What is the single model preference source of truth?
- Which package owns provider capability contracts?
- Which package owns concrete adapters?
- How are provider stalls, retries, and transport errors surfaced?
- How are structured output, vision, streaming, and reasoning markers represented?

Target docs:

- `docs/architecture/LLM_SERVICE_ARCHITECTURE.md`
- `docs/architecture/LLM_PROVIDER_CAPABILITIES.md`
- `docs/architecture/llm/PROVIDER_PLUGINS.md`
- `docs/architecture/llm/HONESTY_LAYER.md`

### Signals And Workflows

Questions to answer:

- What currently wakes the bird?
- Which wake sources are core vs feature-owned?
- Which workflow docs describe current shipped behavior?
- Which workflow docs are design history?
- How does SignalDispatcher avoid cycles?
- How does constitutional injection apply to signals?

Target docs:

- `docs/architecture/SIGNAL_DISPATCHER.md`
- `docs/architecture/SIGNAL_SOURCES_GUIDE.md`
- `docs/architecture/WORKFLOWS_*`

### Talon

Questions to answer:

- What lives in `kestrel-talon`?
- What lives in Kestrel's in-agent Talon control surface?
- How are Talon preferences read and updated?
- How does Talon backend/model selection differ from chat LLM routing?
- Which docs still describe old in-core behavior?

Target docs:

- `README.md`
- `docs/README.md`
- `KESTREL_FEATURES.md`
- `docs/generated/FEATURES_*`

### Cloud, Deployment, And Training

Questions to answer:

- Which deployment path is supported in this repo?
- Which cloud providers are external provider packages?
- Which training provider docs still reference old module paths?
- Which docs are operational runbooks vs research?

Target docs:

- `docs/deployment/README.md`
- `docs/architecture/TRAINING_PROVIDER_ARCHITECTURE.md`
- `docs/architecture/PLAN_RUNPOD_INTEGRATION.md`
- `docs/architecture/RUNPOD_LORA_TRAINING.md`
- `docs/architecture/VASTAI_TRAINING.md`
- `docs/research/LoRA/*`

## Proposed Documentation Taxonomy

Use this taxonomy consistently:

| Label | Meaning |
|---|---|
| Canonical | The file other docs must follow for this subject |
| Generated | Derived from a canonical source; never edited manually |
| Active | Matches current shipped behavior |
| Active but external | Current behavior, but owned by another package/repo |
| Design-of-record | Accepted architecture for in-progress work |
| Historical | Useful context, not current guidance |
| Research | Exploratory notes; not implementation guidance |
| Internal | Not intended for public docs or user-facing publication |
| Needs re-audit | Known drift risk; do not rely without checking code |

## Proposed Package Boundary Language

Use language like this in the README and docs index after verification:

> Kestrel core provides the sovereign agent runtime: identity, constitution, privacy controls, memory/storage foundations, agent orchestration, LLM routing, guarded compute, Cloud Run deployment, and the feature/package registry. Optional capabilities are installed as packages. Feature packages register under `kestrel_sovereign.features`; provider packages register under provider-specific entry-point groups such as cloud, voice, storage, or LLM providers. Standalone operational tools such as `kestrel-talon` may integrate with Kestrel but are not themselves core agent features.

This wording should be revised once the registry semantics and actual install contents are verified.

## Suggested Cleanup Plan

### Pass 1 - Truth Sources

- Update `KESTREL_FEATURES.md` from live discovery and current extraction state.
- Update `feature_registry.toml` semantics.
- Add validation for stale package/core combinations.
- Regenerate generated feature docs.

### Pass 2 - Main Entrypoints

- Update `README.md`.
- Update `docs/README.md`.
- Update `docs/architecture/README.md`.
- Add a link from `docs/audit/README.md` to this audit.

### Pass 3 - Volatile Architecture

- Reconcile context docs.
- Reconcile memory/retrieval/storage docs.
- Reconcile LLM provider docs.
- Reconcile signal/workflow docs.
- Reconcile Talon docs.

### Pass 4 - Operator And User Docs

- Revalidate deployment docs.
- Revalidate user guides and demo scripts.
- Add optional-package notes where needed.
- Check public release hygiene for business, outreach, legal, strategy, planning, and archive directories.

### Pass 5 - Diagrams And Long Tail

- Add status/date banners to diagram docs.
- Archive superseded plans.
- Add broken-link checks and doc freshness checks to CI if practical.

## Candidate Validation Commands

See `docs/audit/documentation-2026-05/VALIDATION_COMMANDS.md` for the working command list. Keep the commands there to avoid drift between the executive ledger and the audit workspace.

## Open Questions

- Is `feature_registry.toml` intended to be a current install manifest, an available package catalog, or both?
- Should `core = true` mean "ships inside `kestrel-sovereign`", "available by default", or "first-party maintained"?
- Are `workflows` and `feature_features` fully extracted, or is there still a compatibility wrapper in core?
- Is Talon still an agent feature in any sense, or should all public docs describe it as standalone plus control surface?
- Is local voice still bundled in the base install, or has all voice moved to `kestrel-feature-voice` plus provider packages?
- Which extracted packages are public and installable today vs planned/private?
- Should `KESTREL_FEATURES.md` remain hand-maintained, or should it be generated from discovery plus curated annotations?

## Definition Of Done For This Audit Program

This documentation audit is complete when:

- A new contributor can identify the correct repo/package to edit for every major feature.
- A user can tell what the base install includes without reading code.
- Generated docs contain no stale core/add-on claims.
- Architecture docs have status banners and current ownership.
- Context, memory, retrieval, LLM routing, signals, workflows, and Talon each have one canonical doc.
- Deployment docs distinguish Cloud Run core from external cloud/provider packages.
- Public-release-sensitive directories are classified.
- CI or a lightweight script catches broken links and the most important inventory drift.
