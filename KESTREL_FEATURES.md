# Kestrel Sovereign Feature Inventory

> Canonical source of truth for the maintained Kestrel surface.
>
> If code and this document disagree, fix the disagreement or mark it explicitly.
> Do not keep stale marketing counts here.

## How To Read This Document

- This file is the canonical inventory consumed by audience-specific generators such as [`scripts/generate_feature_docs.py`](scripts/generate_feature_docs.py).
- Generated audience docs are derived artifacts and belong under `docs/generated/`.
- Historical catalogs belong under `docs/archive/`.
- Discovery rules matter more than headline numbers:
  - Feature module discovery is defined by [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py).
  - HTTP route families are defined by [`kestrel_sovereign/server.py`](kestrel_sovereign/server.py) and the routers under [`kestrel_sovereign/endpoints/`](kestrel_sovereign/endpoints).

## Canonical Principles

- Prefer maintained surfaces over aspirational claims.
- Prefer route families and feature inventories over brittle fixed counts.
- Separate supported public surfaces from internal or partial surfaces.
- Keep the canonical source austere enough that generated docs can safely transform it.

<!-- BEGIN PROTECTED PACKAGE BOUNDARY CONTRACT -->
## Package Ownership and Installation Boundaries

These ownership statements are normative and must remain intact in every
audience-specific derivative:

<!-- NON_BUNDLED_SURFACE_ALIASES: talon feature|talon coordinator; voice; mcp; github integration|github app; wallet; observability; council; visual identity; legal feature; code editing|code edit; parametric self; whatsapp; runpod; vast.ai|vastai; gcp compute; elevenlabs; deepgram; kestrel-talon -->

- **Bundled Feature lifecycle modules:** Feature subclasses discovered from
  `kestrel_sovereign/features/` ship in the `kestrel-sovereign` distribution.
  They need no separate package install. The generated inventory below is the
  exact in-tree discovery snapshot.
- **Bundled non-Feature components:** Some base-install runtime services, such
  as `PrivacyAgent`, are shipped by `kestrel-sovereign` but are not Feature
  lifecycle classes. The registry labels these `bundled-component` rather than
  putting them in its `features` field.
- **Not bundled — extracted Feature packages:** Talon, Voice, MCP, GitHub
  integration, wallet, observability, reflection, council, visual identity,
  legal, code editing, parametric self, and WhatsApp transport are separate
  install targets.
  They register Feature subclasses through the `kestrel_sovereign.features`
  entry-point group.
- **Not bundled — provider packages:** ElevenLabs, Deepgram, OpenAI voice, xAI
  voice/realtime, RunPod, Vast.ai, GCP Compute, and external storage backends
  implement provider contracts. They use provider-specific entry-point groups;
  installing one does not make that provider a Feature lifecycle class.
- **Not bundled — standalone tool:** `kestrel-talon` is an independently
  installed command-line issue processor. Its `TalonCoordinatorFeature` control
  surface is also external, owned by `kestrel-feature-talon`; the feature and
  standalone executable are separately named companion registry rows.

The runtime catalog at `kestrel_sovereign/data/feature_registry.toml` encodes
these distinctions in `boundary`. Its `package` field is always the owning
distribution/install target for that row. The compatibility field `core` is
`true` only for `bundled` and `bundled-component` rows. `features` contains
Feature lifecycle class names only; provider implementations use
`provider_classes` plus `entry_point_groups`, and standalone tools use
`command`. Catalog status `available` means “known but not detected in this
environment,” not a claim that an external distribution is publicly reachable.
<!-- END PROTECTED PACKAGE BOUNDARY CONTRACT -->

## Maintained Surface

### Constitutional and sovereign foundation

- Constitution and governance:
  - [`kestrel_sovereign/data/KESTREL_CONSTITUTION.md`](kestrel_sovereign/data/KESTREL_CONSTITUTION.md)
  - [`kestrel_sovereign/agent/constitution.py`](kestrel_sovereign/agent/constitution.py)
  - [`kestrel_sovereign/features/constitution.py`](kestrel_sovereign/features/constitution.py)
- DID identity and continuity:
  - [`kestrel_sovereign/inception_service.py`](kestrel_sovereign/inception_service.py)
  - [`kestrel_sovereign/identity/identity_package.py`](kestrel_sovereign/identity/identity_package.py)
  - [`kestrel_sovereign/identity/signing.py`](kestrel_sovereign/identity/signing.py)
  - [`kestrel_sovereign/identity/continuity_verifier.py`](kestrel_sovereign/identity/continuity_verifier.py)
- Sovereignty lifecycle:
  - [`kestrel_sovereign/graduate_service.py`](kestrel_sovereign/graduate_service.py)
  - [`kestrel_sovereign/retirement_service.py`](kestrel_sovereign/retirement_service.py)
  - [`kestrel_sovereign/endpoints/sovereignty.py`](kestrel_sovereign/endpoints/sovereignty.py)

### Agent runtime and context assembly

- Canonical behavior contract:
  - [`docs/architecture/CONTEXT_SYSTEM_DESIGN.md`](docs/architecture/CONTEXT_SYSTEM_DESIGN.md)
- Core agent orchestration:
  - [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py)
  - [`kestrel_sovereign/command_handler.py`](kestrel_sovereign/command_handler.py)
- Context and token budgeting:
  - [`kestrel_sovereign/agent/context_manager.py`](kestrel_sovereign/agent/context_manager.py)
  - [`kestrel_sovereign/agent/context_builder.py`](kestrel_sovereign/agent/context_builder.py)
  - [`kestrel_sovereign/agent/context_stages.py`](kestrel_sovereign/agent/context_stages.py)
  - [`kestrel_sovereign/agent/token_budget.py`](kestrel_sovereign/agent/token_budget.py)
- Canonical and rendered conversation persistence:
  - [`kestrel_sovereign/storage/async_conversation_store.py`](kestrel_sovereign/storage/async_conversation_store.py)
- Streaming and request lifecycle:
  - [`kestrel_sovereign/agent/streaming.py`](kestrel_sovereign/agent/streaming.py)
  - [`kestrel_sovereign/endpoints/agent.py`](kestrel_sovereign/endpoints/agent.py)

<!-- BEGIN PROTECTED CONTEXT HONESTY CONTRACT -->
## Context Runtime and Diagnostic Boundary

These context statements are normative and must remain intact in every
audience-specific derivative:

- A production turn preloads at most the latest **50** eligible entries from
  the active session before retrieval, budgeting, and lumpy history selection.
- Production and `GET /api/agent/context-status?full=true` use the same typed
  `ContextManager` build plan over that latest-50 input. The dry-run executes
  production relevance gates, elastic finalization, lumpy anchoring,
  microcompaction, wrapper accounting, and prune decisions without committing
  access records or salvage writes. Status reads the anchored governing
  constitution without lazily creating or anchoring missing policy.
- The cheap `full=false` status deliberately omits memory/RAG acquisition and
  reports those sections as `unknown`/`skipped`, never as measured zero.
  Provider-native framing and stateful provider-thread occupancy remain
  separate from the Kestrel context plan.
- Default lumpy pruning omits older history from the provider window while
  retaining the source rows; it does not create an automatic durable summary.
  Automatic durable salvage is disabled by default. Its feature-flagged path is
  conditional on pruned rows mapping to id-bearing persistent history. A mixed
  span writes only that subset and reports id-less rows as unmappable, so it is
  not a fail-closed guarantee for id-less or `ISOLATED` in-memory history.
- `openai:plan` occupancy compaction is best-effort. Kestrel resets the Codex
  thread only after durable compaction reports success; a skipped or failed
  attempt lets the turn continue with the existing thread.
- The complete all-route Context C lifecycle remains aspirational, not shipped
  behavior. The canonical current-state contract is
  `docs/architecture/CONTEXT_SYSTEM_DESIGN.md`; the separate
  `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md` page is a design record.
<!-- END PROTECTED CONTEXT HONESTY CONTRACT -->

### Multi-LLM platform

- Unified service and routing:
  - [`kestrel_sovereign/llm/service.py`](kestrel_sovereign/llm/service.py)
  - [`kestrel_sovereign/llm/provider_registry.py`](kestrel_sovereign/llm/provider_registry.py)
  - [`kestrel_sovereign/llm/mandate.py`](kestrel_sovereign/llm/mandate.py)
- Provider adapters present in tree:
  - OpenAI, Anthropic, Claude Max, Gemini, Vertex AI, Ollama, OpenRouter, Mock
  - See [`kestrel_sovereign/llm/`](kestrel_sovereign/llm/)
- Catalog, metadata, retry, and usage tracking:
  - [`kestrel_sovereign/llm/model_catalog.py`](kestrel_sovereign/llm/model_catalog.py)
  - [`kestrel_sovereign/llm/model_metadata.py`](kestrel_sovereign/llm/model_metadata.py)
  - [`kestrel_sovereign/llm/retry.py`](kestrel_sovereign/llm/retry.py)
  - [`kestrel_sovereign/llm/usage_tracking.py`](kestrel_sovereign/llm/usage_tracking.py)

### Privacy, storage, and memory

- Privacy modes and enforcement:
  - [`kestrel_sovereign/privacy.py`](kestrel_sovereign/privacy.py)
  - [`kestrel_sovereign/features/privacy/feature.py`](kestrel_sovereign/features/privacy/feature.py)
  - [`kestrel_sovereign/features/privacy/`](kestrel_sovereign/features/privacy)
- Canonical privacy presets:

| Preset | Storage | LLM location | Shareable | Meaning |
|---|---|---|---|---|
| `ephemeral` | none | local | no | Nothing stored, local LLM only |
| `isolated` | temp | local | no | Temporary session storage, local LLM only |
| `anonymous` | scrubbed | cloud | no | Stored with PII removed, cloud LLM allowed |
| `normal` | full | cloud | no | Standard persistent storage |
| `public` | full | cloud | yes | Shareable and exportable |

- Storage and persistence:
  - [`kestrel_sovereign/storage/__init__.py`](kestrel_sovereign/storage/__init__.py)
  - [`kestrel_sovereign/storage/async_storage.py`](kestrel_sovereign/storage/async_storage.py)
  - [`kestrel_sovereign/storage/`](kestrel_sovereign/storage)
  - Fleet/host feature SQLite state follows the private placement, migration,
    and database-separation contract in
    [`docs/architecture/security/HOST_RUNTIME_STORAGE_CUSTODY.md`](docs/architecture/security/HOST_RUNTIME_STORAGE_CUSTODY.md).
- Memory systems:
  - [`kestrel_sovereign/agent/memory_manager.py`](kestrel_sovereign/agent/memory_manager.py)
  - [`kestrel_sovereign/features/memory/`](kestrel_sovereign/features/memory)
  - [`kestrel_sovereign/features/memory_agency/`](kestrel_sovereign/features/memory_agency)

<!-- BEGIN AUTO-GENERATED FEATURE INVENTORY -->

## Feature Module Inventory

Feature lifecycle classes come from two sources:

1. **Bundled Feature modules** — discovered from `kestrel_sovereign/features/` via `discover_feature_modules()`.
2. **Extracted Feature packages** — installed packages registered with the `kestrel_sovereign.features` entry point group at runtime.

A bundled/external class-name collision fails closed unless the bundled registry row explicitly authorizes that exact extracted distribution and implementation-module prefix as a temporary migration replacement. Two external owners always fail.

The generated inventory below lists bundled Feature lifecycle modules only: the in-tree surface discoverable from this checkout.
Installed entry point feature classes are included in JSON output when present in the active environment.
Runtime security policy can still deny a discovered tool at call time; static generation marks source-discovered tools as enabled unless their feature is disabled.

- Current audited snapshot: `37` discoverable modules and `37` exported `Feature` subclasses.

### `attachments` (AttachmentsFeature)

- Source: [`kestrel_sovereign/features/attachments/feature.py`](kestrel_sovereign/features/attachments/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `read_attachment` | `!read-attachment` | `system` | `attachment_id`, `offset`, `length`, `session_id` | 260 | `enabled` |

### `audit_anchor` (AuditAnchorFeature)

- Source: [`kestrel_sovereign/features/audit_anchor/feature.py`](kestrel_sovereign/features/audit_anchor/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `audit_anchor` | `!audit-anchor` | `system` |  | 22 | `enabled` |
| `audit_anchor_status` | `!audit-status` | `system` |  | 17 | `enabled` |
| `audit_verify` | `!audit-verify` | `system` |  | 21 | `enabled` |

### `bootstrap` (BootstrapFeature)

- Source: [`kestrel_sovereign/features/bootstrap/feature.py`](kestrel_sovereign/features/bootstrap/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `bootstrap_add` | `!bootstrap add` | `system` | `file_path` | 98 | `enabled` |
| `bootstrap_list` | `!bootstrap list` | `system` |  | 42 | `enabled` |
| `bootstrap_reload` | `!bootstrap reload` | `system` |  | 33 | `enabled` |
| `bootstrap_remove` | `!bootstrap remove` | `system` | `name` | 72 | `enabled` |
| `bootstrap_status` | `!bootstrap-status` | `system` |  | 21 | `enabled` |
| `rename_agent` | `!rename` | `system` | `new_name` | 52 | `enabled` |
| `restart_discovery` | `!restart-discovery` | `system` |  | 23 | `enabled` |
| `skip_discovery` | `!skip-discovery` | `system` |  | 25 | `enabled` |

### `bridge` (BridgeFeature)

- Source: [`kestrel_sovereign/features/bridge/feature.py`](kestrel_sovereign/features/bridge/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `bridge_connections` | `!bridge connections` | `system` | `limit` | 53 | `enabled` |
| `bridge_history` | `!bridge history` | `system` | `limit` | 54 | `enabled` |
| `bridge_status` | `!bridge status` | `system` |  | 21 | `enabled` |

### `channels` (ChannelFeature)

- Source: [`kestrel_sovereign/features/channels/feature.py`](kestrel_sovereign/features/channels/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `channels_history` | `!channels history` | `communication` | `limit`, `channel` | 92 | `enabled` |
| `channels_list` | `!channels list` | `communication` |  | 25 | `enabled` |
| `channels_send` | `!channels send` | `communication` | `channel`, `to`, `message` | 105 | `enabled` |

### `cli` (CliFeature)

- Source: [`kestrel_sovereign/features/cli/feature.py`](kestrel_sovereign/features/cli/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `cli_status` | `!cli-status` | `system` |  | 27 | `enabled` |
| `git_diff` | `!git-diff` | `data_access` | `ref`, `path`, `repo_path` | 109 | `enabled` |
| `git_log` | `!git-log` | `data_access` | `max_count`, `repo_path` | 79 | `enabled` |
| `git_merge_base` | `!git-merge-base` | `data_access` | `left_ref`, `right_ref`, `repo_path` | 93 | `enabled` |
| `git_show_file` | `!git-show-file` | `data_access` | `ref`, `path`, `repo_path` | 104 | `enabled` |
| `git_status` | `!git-status` | `data_access` | `repo_path` | 57 | `enabled` |

### `compute` (ComputeFeature)

- Source: [`kestrel_sovereign/features/compute/feature.py`](kestrel_sovereign/features/compute/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `empty_trash` | `!compute-empty-trash` | `system` | `older_than_days`, `dry_run` | 92 | `enabled` |
| `execution_history` | `!compute-history` | `system` | `script_id`, `limit` | 76 | `enabled` |
| `get_compute_capabilities` | `!compute-caps` | `system` |  | 21 | `enabled` |
| `get_compute_policy` |  | `system` |  | 21 | `enabled` |
| `list_scripts` | `!compute-list` | `system` | `state`, `limit` | 125 | `enabled` |
| `list_trash` | `!compute-trash` | `system` | `days` | 48 | `enabled` |
| `restore_from_trash` | `!compute-restore` | `system` | `trash_path`, `destination` | 80 | `enabled` |
| `run_script` | `!compute-run` | `system` | `script_id`, `executor`, `timeout` | 210 | `enabled` |
| `show_script` | `!compute-show` | `system` | `script_id` | 44 | `enabled` |
| `write_script` | `!compute-write` | `system` | `name`, `language`, `content`, `purpose`, `requirements` | 192 | `enabled` |

### `computer_use` (ComputerUseFeature)

- Source: [`kestrel_sovereign/features/computer_use/feature.py`](kestrel_sovereign/features/computer_use/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `fs_edit` | `!fs-edit` | `file_operations` | `path`, `old_text`, `new_text`, `occurrence` | 138 | `enabled` |
| `fs_list` | `!fs-list` | `file_operations` | `path` | 61 | `enabled` |
| `fs_read` | `!fs-read` | `file_operations` | `path` | 60 | `enabled` |
| `fs_write` | `!fs-write` | `file_operations` | `path`, `content` | 85 | `enabled` |
| `shell` | `!shell` | `system` | `command`, `timeout` | 115 | `enabled` |

### `consent` (ConsentFeature)

- Source: [`kestrel_sovereign/features/consent/feature.py`](kestrel_sovereign/features/consent/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `consent_log` | `!consent-log` | `system` | `limit` | 62 | `enabled` |
| `consent_stats` | `!consent-stats` | `system` |  | 25 | `enabled` |

### `constitution` (ConstitutionFeature)

- Source: [`kestrel_sovereign/features/constitution.py`](kestrel_sovereign/features/constitution.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `constitution` | `!constitution` | `system` | `article`, `search`, `summary` | 259 | `enabled` |

### `context` (ContextFeature)

- Source: [`kestrel_sovereign/features/context/feature.py`](kestrel_sovereign/features/context/feature.py)
- Enablement state: `enabled`
- `context_status` is the cheap dry-run view of the same typed
  `ContextManager` plan used by `GET /api/agent/context-status` and
  production turns. Cheap mode marks omitted memory/RAG sections unknown;
  `full=true` executes the production retrieval and pruning policy read-only.
  It reads anchored governing policy without lazily creating missing state.
  Provider-native framing remains outside the Kestrel plan.

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `compact_context` | `!context compact` | `memory` | `keep_recent`, `force`, `dry_run` | 156 | `enabled` |
| `context_stash` | `!context stash` | `memory` | `target`, `name` | 124 | `enabled` |
| `context_stash_apply` | `!context stash apply` | `memory` | `stash_id` | 78 | `enabled` |
| `context_stash_drop` | `!context stash drop` | `memory` | `stash_id` | 77 | `enabled` |
| `context_stash_list` | `!context stash list` | `memory` |  | 23 | `enabled` |
| `context_stash_peek` | `!context stash peek` | `memory` | `stash_id`, `max_chars` | 103 | `enabled` |
| `context_stash_pop` | `!context stash pop` | `memory` | `stash_id` | 73 | `enabled` |
| `context_stash_save` | `!context stash save` | `memory` | `stash_id`, `name`, `summary`, `tags` | 140 | `enabled` |
| `context_status` | `!context status` | `system` |  | 44 | `enabled` |
| `exclude_from_context` | `!context exclude` | `memory` | `target`, `reason` | 113 | `enabled` |
| `hierarchical_compact` | `!context compact hierarchical` | `memory` | `chunk_size`, `keep_recent`, `max_depth` | 143 | `enabled` |
| `mark_content` | `!context mark` | `memory` | `action`, `target`, `reason` | 162 | `enabled` |
| `recursive_query` | `!context query` | `memory` | `context_source`, `query`, `use_cheap_model` | 189 | `enabled` |
| `restore_excluded` | `!context restore` | `memory` | `target` | 63 | `enabled` |
| `summarize_section` | `!context summarize` | `memory` | `mode`, `criteria`, `preserve_key_facts` | 238 | `enabled` |

### `delivery` (DeliveryFeature)

- Source: [`kestrel_sovereign/features/delivery/feature.py`](kestrel_sovereign/features/delivery/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `delivery_failed` | `!delivery failed` | `communication` | `limit` | 58 | `enabled` |
| `delivery_purge` | `!delivery purge` | `communication` | `older_than_hours` | 66 | `enabled` |
| `delivery_queue_list` | `!delivery queue` | `communication` | `limit` | 54 | `enabled` |
| `delivery_retry` | `!delivery retry` | `communication` | `message_id` | 55 | `enabled` |
| `delivery_status` | `!delivery status` | `communication` |  | 33 | `enabled` |

### `deploy` (DeployFeature)

- Source: [`kestrel_sovereign/features/deploy/feature.py`](kestrel_sovereign/features/deploy/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `deploy_agent` | `!deploy` | `system` | `action`, `profile`, `tag` | 289 | `enabled` |

### `health` (HealthFeature)

- Source: [`kestrel_sovereign/features/health/feature.py`](kestrel_sovereign/features/health/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `health_check` | `!health` | `system` |  | 21 | `enabled` |
| `health_history` | `!health-history` | `system` | `limit` | 57 | `enabled` |
| `health_interval` | `!health-interval` | `system` | `seconds` | 54 | `enabled` |
| `heartbeat_check` | `!heartbeat` | `system` |  | 24 | `enabled` |
| `heartbeat_interval` | `!heartbeat-interval` | `system` | `seconds` | 46 | `enabled` |
| `heartbeat_status` | `!heartbeat-status` | `system` | `limit` | 45 | `enabled` |

### `identity` (IdentityFeature)

- Source: [`kestrel_sovereign/features/identity/feature.py`](kestrel_sovereign/features/identity/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `assess_substrate` | `!identity assess` | `system` |  | 42 | `enabled` |
| `export_identity` | `!identity export` | `system` | `storage_tier`, `sign`, `include_wallet` | 208 | `enabled` |
| `import_identity` | `!identity import` | `system` | `source`, `verify_signature`, `merge_mode`, `key_hash`, `allow_unsigned`, `identity_trust_policy` | 303 | `enabled` |
| `lifecycle_status` | `!identity status` | `system` |  | 70 | `enabled` |
| `migration_history` | `!identity history` | `system` |  | 38 | `enabled` |
| `verify_identity` | `!identity verify` | `system` | `source`, `key_hash`, `identity_trust_policy` | 149 | `enabled` |

### `inference_lease` (InferenceLeaseFeature)

- Source: [`kestrel_sovereign/features/inference_lease/feature.py`](kestrel_sovereign/features/inference_lease/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `inference_lease_acquire` |  | `model_management` | `model`, `max_hourly_cost_usd`, `max_total_cost_usd`, `runtime`, `privacy`, `expected_session_seconds`, `idle_ttl_seconds`, `ready_deadline_seconds`, `expected_concurrency`, `allowed_regions`, `capabilities`, `request_id` | 497 | `enabled` |
| `inference_lease_release` |  | `model_management` | `lease_id` | 72 | `enabled` |
| `inference_lease_status` |  | `model_management` | `lease_id` | 75 | `enabled` |

### `keys` (KeyManagementFeature)

- Source: [`kestrel_sovereign/features/keys/feature.py`](kestrel_sovereign/features/keys/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `add_service_key` | `!add-key` | `system` | `provider`, `api_key`, `quota_limit` | 118 | `enabled` |
| `delete_service_key` | `!delete-key` | `system` | `provider` | 77 | `enabled` |
| `get_key_usage` | `!key-usage` | `system` | `provider`, `days` | 106 | `enabled` |
| `list_providers` | `!providers` | `system` |  | 18 | `enabled` |
| `list_service_keys` | `!list-keys` | `system` |  | 22 | `enabled` |
| `remove_service_key` | `!remove-key` | `system` | `provider` | 77 | `enabled` |
| `rotate_service_key` | `!rotate-key` | `system` | `provider`, `new_api_key` | 101 | `enabled` |

### `memory` (MemoryFeature)

- Source: [`kestrel_sovereign/features/memory/feature.py`](kestrel_sovereign/features/memory/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `confirm_person_match` | `!memory confirm-person` | `memory` | `message_id`, `mentioned_label`, `concept_id` | 75 | `enabled` |
| `delete_conversation` | `!memory delete-conversation` | `memory` | `session_id`, `confirm` | 147 | `enabled` |
| `delete_message_by_id` | `!memory delete-message` | `memory` | `message_id`, `session_id` | 174 | `enabled` |
| `delete_messages` | `!memory delete` | `memory` | `pattern`, `confirm`, `session_id` | 194 | `enabled` |
| `get_episodes` | `!memory episodes` | `memory` | `limit`, `query` | 152 | `enabled` |
| `list_conversations` | `!memory conversations` | `memory` | `limit`, `include_trashed` | 168 | `enabled` |
| `list_trashed_messages` | `!memory trash` | `memory` | `limit` | 168 | `enabled` |
| `mark_superseded` | `!memory supersede` | `memory` | `old_id`, `new_id`, `reason` | 140 | `enabled` |
| `memory_consolidate` | `!memory consolidate` | `memory` |  | 72 | `enabled` |
| `memory_index_backfill` | `!memory index-backfill` | `system` | `batch_size` | 67 | `enabled` |
| `memory_status` | `!memory status` | `system` |  | 36 | `enabled` |
| `purge_conversation` | `!memory purge-conversation` | `memory` | `session_id`, `confirm`, `reason` | 155 | `enabled` |
| `purge_message_by_id` | `!memory purge-message` | `memory` | `message_id`, `confirm`, `session_id`, `reason` | 210 | `enabled` |
| `recall_action_items` | `!memory actions` | `memory` | `status`, `days`, `assignee_concept_id`, `limit`, `include_superseded` | 240 | `enabled` |
| `recall_decisions` | `!memory decisions` | `memory` | `limit`, `include_superseded` | 91 | `enabled` |
| `recall_emotional` | `!memory recall` | `memory` | `query`, `mood`, `limit`, `min_relevance` | 237 | `enabled` |
| `recall_interactions` | `!memory interactions` | `memory` | `person_concept_id`, `limit` | 67 | `enabled` |
| `recall_recent` | `!memory recent` | `memory` | `limit` | 80 | `enabled` |
| `restore_conversation` | `!memory restore-conversation` | `memory` | `session_id` | 80 | `enabled` |
| `restore_message_by_id` | `!memory restore-message` | `memory` | `message_id`, `session_id` | 118 | `enabled` |
| `search_case_law` | `!memory cases` | `memory` | `query`, `limit` | 106 | `enabled` |
| `search_documents` | `!memory docs` | `memory` | `query`, `limit` | 114 | `enabled` |
| `search_memory` | `!memory search` | `memory` | `query`, `limit`, `session_id` | 188 | `enabled` |
| `update_action_item` | `!memory action update` | `memory` | `item_id`, `status`, `due_date`, `assignee_concept_id` | 108 | `enabled` |

### `memory_agency` (MemoryAgencyFeature)

- Source: [`kestrel_sovereign/features/memory_agency/feature.py`](kestrel_sovereign/features/memory_agency/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `forget_fact` | `!memory-forget-fact` | `memory` | `subject`, `predicate` | 67 | `enabled` |
| `memory_admin_unpin_all` | `!memory-admin-unpin-all` | `system` |  | 32 | `enabled` |
| `memory_admin_unpin_oldest` | `!memory-admin-unpin-oldest` | `system` | `count` | 55 | `enabled` |
| `memory_pin` | `!memory-pin` | `memory` | `message_id`, `reason` | 100 | `enabled` |
| `memory_pin_stats` | `!memory-pin-stats` | `system` |  | 40 | `enabled` |
| `memory_pinned` | `!memory-pinned` | `memory` |  | 37 | `enabled` |
| `memory_release` | `!memory-release` | `memory` | `message_id` | 67 | `enabled` |
| `save_fact` | `!memory-save-fact` | `memory` | `subject`, `predicate`, `value`, `confidence` | 182 | `enabled` |

### `model` (ModelAgent)

- Source: [`kestrel_sovereign/features/model/feature.py`](kestrel_sovereign/features/model/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `cleanup_models` |  | `model_management` | `threshold_days`, `dry_run` | 200 | `enabled` |
| `get_current_model` | `!model` | `model_management` |  | 27 | `enabled` |
| `get_model_info` | `!model-info` | `model_management` | `model_name` | 35 | `enabled` |
| `get_model_storage_info` |  | `model_management` | `use_cache` | 40 | `enabled` |
| `list_models` | `!model-list` | `model_management` | `use_cache` | 36 | `enabled` |
| `pull_model` | `!model-pull` | `model_management` | `model_name`, `progress_callback` | 54 | `enabled` |
| `set_model` | `!model-set` | `model_management` | `vendor_or_model`, `model` | 194 | `enabled` |

### `peers` (PeersFeature)

- Source: [`kestrel_sovereign/features/peers/feature.py`](kestrel_sovereign/features/peers/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `ask_agent` | `!ask` | `communication` | `agent_name`, `message` | 105 | `enabled` |
| `get_peer_task_result` | `!a2a result` | `communication` | `recipient`, `task_id` | 182 | `enabled` |
| `list_outbound_a2a_tasks` | `!a2a outbound` | `communication` | `limit`, `recipient` | 175 | `enabled` |
| `list_peers` | `!peers` | `communication` |  | 22 | `enabled` |
| `send_a2a_message` | `!a2a tell` | `communication` | `recipient`, `message`, `session_id` | 161 | `enabled` |
| `send_a2a_question` | `!a2a ask` | `communication` | `recipient`, `message`, `session_id`, `timeout_seconds`, `artifacts`, `references` | 445 | `enabled` |
| `send_a2a_task` | `!a2a send` | `communication` | `recipient`, `message`, `skill_id`, `session_id`, `artifacts`, `references` | 488 | `enabled` |

### `response_audit` (ResponseAuditFeature)

- Source: [`kestrel_sovereign/features/response_audit/feature.py`](kestrel_sovereign/features/response_audit/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `audit_disable` | `!audit-off` | `system` |  | 16 | `enabled` |
| `audit_enable` | `!audit-on` | `system` | `mode` | 78 | `enabled` |
| `audit_status` | `!audit` | `system` |  | 18 | `enabled` |

### `restart_coordinator` (RestartCoordinatorFeature)

- Source: [`kestrel_sovereign/features/restart_coordinator/feature.py`](kestrel_sovereign/features/restart_coordinator/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `acknowledge_restart_escalation` | `!restart acknowledge-escalation` | `system` | `request_id` | 89 | `enabled` |
| `cancel_restart_request` | `!restart cancel` | `system` | `request_id`, `reason` | 130 | `enabled` |
| `list_restart_requests` | `!restart list` | `data_access` | `status` | 104 | `enabled` |
| `list_restart_status_events` | `!restart events` | `data_access` | `limit`, `since` | 91 | `enabled` |
| `request_restart` | `!restart request` | `system` | `reason`, `urgency`, `policy`, `desired_window`, `operation`, `update_profile`, `target_ref`, `repo_path`, `allow_migrations` | 566 | `enabled` |
| `restart_coordinator` | `!restart coordinator` | `system` |  | 47 | `enabled` |

### `save` (SaveFeature)

- Source: [`kestrel_sovereign/features/save/feature.py`](kestrel_sovereign/features/save/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `recall` | `!recall` | `memory` | `query`, `item_type`, `limit` | 184 | `enabled` |
| `recall_delete` | `!recall delete` | `memory` | `item_id` | 41 | `enabled` |
| `recall_get` | `!recall get` | `memory` | `item_id` | 46 | `enabled` |
| `recall_list` | `!recall list` | `memory` | `item_type`, `limit` | 108 | `enabled` |
| `save_excerpt` | `!save excerpt` | `memory` | `target`, `name`, `summary`, `tags` | 182 | `enabled` |
| `save_item` | `!save item` | `memory` | `name`, `content`, `item_type`, `summary`, `tags`, `schema_id` | 302 | `enabled` |
| `save_stash` | `!save stash` | `memory` | `stash_id`, `name`, `summary`, `tags` | 171 | `enabled` |

### `scheduler` (SchedulerFeature)

- Source: [`kestrel_sovereign/features/scheduler/feature.py`](kestrel_sovereign/features/scheduler/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `schedule_add` | `!schedule add` | `utility` | `cron_expression`, `task_name`, `args_json`, `timezone_name`, `misfire_policy`, `misfire_grace_seconds`, `idempotency_key` | 320 | `enabled` |
| `schedule_add_deadline` | `!schedule deadline` | `utility` | `run_at`, `task_name`, `args_json`, `misfire_policy`, `misfire_grace_seconds`, `idempotency_key`, `delay_seconds` | 213 | `enabled` |
| `schedule_engagement` | `!schedule engagement` | `utility` | `days` | 55 | `enabled` |
| `schedule_history` | `!schedule history` | `utility` | `limit` | 55 | `enabled` |
| `schedule_list` | `!schedule list` | `utility` |  | 19 | `enabled` |
| `schedule_pause` | `!schedule pause` | `utility` | `task_id` | 50 | `enabled` |
| `schedule_record_outcome` | `!schedule outcome` | `utility` | `execution_id`, `signal` | 79 | `enabled` |
| `schedule_remove` | `!schedule remove` | `utility` | `task_id` | 42 | `enabled` |
| `schedule_resume` | `!schedule resume` | `utility` | `task_id`, `acknowledge_ambiguous_effect` | 80 | `enabled` |
| `schedule_self_followups` | `!schedule self-followups` | `utility` | `limit` | 73 | `enabled` |
| `schedule_update` | `!schedule update` | `utility` | `task_id`, `cron_expression`, `timezone_name` | 119 | `enabled` |

### `security` (SecurityFeature)

- Source: [`kestrel_sovereign/features/security/feature.py`](kestrel_sovereign/features/security/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `approve` | `!security-approve` | `system` | `request_id`, `scope` | 92 | `enabled` |
| `deny` | `!security-deny` | `system` | `request_id` | 45 | `enabled` |
| `list_permissions` | `!security-list` | `system` |  | 23 | `enabled` |
| `pending_approvals` | `!security-pending` | `system` |  | 17 | `enabled` |
| `security_audit` | `!security-audit` | `system` | `limit` | 54 | `enabled` |
| `set_permission` | `!security-set` | `system` | `feature_name`, `tool_name`, `level` | 193 | `enabled` |

### `skills` (SkillsFeature)

- Source: [`kestrel_sovereign/features/skills/feature.py`](kestrel_sovereign/features/skills/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `skill_delete` | `!skill delete` | `utility` | `skill_id` | 33 | `enabled` |
| `skill_extract_candidates` | `!skill candidates` | `utility` | `min_confidence`, `limit` | 103 | `enabled` |
| `skill_list` | `!skill list` | `utility` |  | 20 | `enabled` |
| `skill_save` | `!skill save` | `utility` | `insight_id`, `steps_json`, `verification`, `tags_json` | 132 | `enabled` |
| `skill_show` | `!skill show` | `utility` | `skill_id` | 34 | `enabled` |

### `sovereignty` (SovereigntyFeature)

- Source: [`kestrel_sovereign/features/sovereignty/feature.py`](kestrel_sovereign/features/sovereignty/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `check_sovereignty_status` | `!check-sovereignty-status` | `system` |  | 20 | `enabled` |
| `export_sovereignty` | `!export-sovereignty` | `system` | `storage_tier`, `encrypt`, `on_progress` | 182 | `enabled` |
| `import_sovereignty` | `!import-sovereignty` | `system` | `cid` | 100 | `enabled` |

### `spawn` (SpawnFeature)

- Source: [`kestrel_sovereign/features/spawn/feature.py`](kestrel_sovereign/features/spawn/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `delegate_task` |  | `agent_management` | `child_name`, `task` | 107 | `enabled` |
| `get_child_result` |  | `agent_management` | `child_name` | 53 | `enabled` |
| `list_children` |  | `agent_management` |  | 27 | `enabled` |
| `spawn_agent` |  | `agent_management` | `name`, `purpose`, `budget`, `ttl`, `constraints`, `features` | 369 | `enabled` |
| `terminate_child` |  | `agent_management` | `child_name`, `offboard_runtime` | 118 | `enabled` |

### `state_of_mind` (StateOfMindFeature)

- Source: [`kestrel_sovereign/features/state_of_mind.py`](kestrel_sovereign/features/state_of_mind.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `state_of_mind` | `!state-of-mind` | `system` |  | 25 | `enabled` |

### `strategic_memory` (StrategicMemoryFeature)

- Source: [`kestrel_sovereign/features/strategic_memory/feature.py`](kestrel_sovereign/features/strategic_memory/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `backlog_hygiene` | `!hygiene` | `system` | `fix` | 88 | `enabled` |
| `morning_signal` | `!morning` | `system` |  | 48 | `enabled` |
| `recall_blockers` | `!blockers` | `memory` | `limit`, `include_resolved` | 120 | `enabled` |
| `recall_patterns` | `!patterns` | `memory` | `limit`, `include_superseded` | 123 | `enabled` |
| `session_log` | `!sessionlog` | `system` | `session_id`, `focus` | 125 | `enabled` |
| `signal_dispatch` | `!dispatch` | `system` | `mode` | 89 | `enabled` |
| `strategy_add_blocker` |  | `system` | `issue`, `title`, `severity`, `owner`, `repo`, `notes` | 226 | `enabled` |
| `strategy_add_decision` |  | `system` | `decision`, `rationale`, `session`, `impact` | 131 | `enabled` |
| `strategy_add_pattern` |  | `system` | `pattern`, `source`, `implication` | 110 | `enabled` |
| `strategy_reconcile_blockers` | `!strategy-reconcile` | `system` | `apply` | 90 | `enabled` |
| `strategy_resolve_blocker` |  | `system` | `issue`, `resolution` | 114 | `enabled` |
| `strategy_search` | `!strategy-search` | `memory` | `query`, `kind`, `limit`, `include_retired` | 182 | `enabled` |
| `strategy_supersede_pattern` |  | `system` | `pattern_id`, `reason`, `superseded_by` | 140 | `enabled` |
| `strategy_view` | `!strategy` | `system` | `section` | 82 | `enabled` |

### `tasks` (TaskFeature)

- Source: [`kestrel_sovereign/features/tasks/feature.py`](kestrel_sovereign/features/tasks/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `attach_artifact_to_a2a_task` | `!a2a attach` | `communication` | `task_id`, `name`, `content`, `index`, `last_chunk` | 383 | `enabled` |
| `cancel_task` | `!cancel-task` | `utility` | `task_id`, `reason` | 74 | `enabled` |
| `check_task_status` | `!task-status` | `utility` | `task_id` | 85 | `enabled` |
| `get_task_result` | `!task-result` | `utility` | `task_id` | 81 | `enabled` |
| `list_available_skills` | `!list-skills` | `utility` |  | 65 | `enabled` |
| `list_my_tasks` | `!tasks` | `utility` | `status`, `task_type`, `limit` | 197 | `enabled` |
| `respond_to_a2a_task` | `!a2a respond` | `communication` | `task_id`, `content`, `state` | 259 | `enabled` |
| `run_workflow` | `!run-workflow` | `utility` | `steps` | 271 | `enabled` |

### `todo` (TodoFeature)

- Source: [`kestrel_sovereign/features/todo/feature.py`](kestrel_sovereign/features/todo/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `todo_add` | `!todo add` | `system` | `title`, `description`, `scope`, `status`, `priority`, `owner`, `links`, `terminal_condition`, `next_check_at`, `source_metadata` | 236 | `enabled` |
| `todo_complete` | `!todo complete` | `system` | `todo_id`, `outcome`, `evidence`, `terminal_condition_satisfied`, `superseded_by` | 138 | `enabled` |
| `todo_link_task` | `!todo link` | `system` | `todo_id`, `link_type`, `target`, `title`, `status`, `url`, `metadata` | 174 | `enabled` |
| `todo_list` | `!todo list` | `system` | `scope`, `status`, `owner`, `include_done`, `include_superseded`, `limit` | 150 | `enabled` |
| `todo_rollup` | `!todo rollup` | `system` | `include_done`, `limit` | 67 | `enabled` |
| `todo_update` | `!todo update` | `system` | `todo_id`, `title`, `description`, `scope`, `status`, `priority`, `owner`, `links`, `terminal_condition`, `next_check_at`, `superseded_by`, `source_metadata` | 289 | `enabled` |

### `wait` (WaitFeature)

- Source: [`kestrel_sovereign/features/wait/feature.py`](kestrel_sovereign/features/wait/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `wait` | `!wait` | `utility` | `target`, `duration_seconds`, `timeout_seconds`, `poll_interval_seconds`, `reason`, `mode` | 774 | `enabled` |

### `web_search` (WebSearchFeature)

- Source: [`kestrel_sovereign/features/web_search/feature.py`](kestrel_sovereign/features/web_search/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `web_search` | `!web-search` | `web_search` | `query`, `max_results` | 114 | `enabled` |

### `webhooks` (WebhookFeature)

- Source: [`kestrel_sovereign/features/webhooks/feature.py`](kestrel_sovereign/features/webhooks/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `webhooks_history` | `!webhooks history` | `system` | `limit` | 56 | `enabled` |
| `webhooks_list` | `!webhooks list` | `system` |  | 19 | `enabled` |
| `webhooks_register` | `!webhooks register` | `system` | `name`, `auth_type`, `event_type`, `auth_config_json`, `rate_limit`, `allow_unauthenticated` | 359 | `enabled` |
| `webhooks_remove` | `!webhooks remove` | `system` | `name` | 44 | `enabled` |

### `wellness` (WellnessFeature)

- Source: [`kestrel_sovereign/features/wellness/feature.py`](kestrel_sovereign/features/wellness/feature.py)
- Enablement state: `enabled`

| Tool | Command | Category | Params | Token cost | State |
|---|---|---|---|---:|---|
| `wellness_check` | `!wellness` | `system` |  | 23 | `enabled` |
| `wellness_export` |  | `system` |  | 21 | `enabled` |
| `wellness_history` | `!wellness-history` | `system` | `limit` | 52 | `enabled` |

## Public HTTP Surface

### App-level routes in `kestrel_sovereign/server.py`

- `GET /`
- `GET /api/auth/key`
- `GET /api/host/csrf`
- `POST /api/host/phoenix/session`
- `GET /api/host/ui/contributions`
- `GET /assets/{path:path}`
- `HEAD /assets/{path:path}`
- `GET /health`
- `GET /health/detailed`

### Router families mounted by `kestrel_sovereign/server.py`

- [`kestrel_sovereign/endpoints/agent.py`](kestrel_sovereign/endpoints/agent.py)
  - `POST /api/agent/attachments`
  - `GET /api/agent/context-status`
  - `GET /api/agent/health/status`
  - `POST /api/agent/health/trigger`
  - `GET /api/agent/heartbeat/status`
  - `POST /api/agent/heartbeat/trigger`
  - `GET /api/agent/info`
  - `POST /api/agent/invoke`
  - `GET /api/agent/notifications`
  - `GET /api/agent/notifications/sse`
  - `GET /api/agent/privacy-mode`
  - `POST /api/agent/privacy-mode`
  - `POST /api/agent/privacy-mode/cancel`
  - `POST /api/agent/privacy-mode/confirm`
  - `GET /api/agent/reflection/status`
  - `POST /api/agent/stop`
  - `POST /api/agent/stream`
  - `GET /api/agent/tasks`
  - `POST /api/agent/tasks/send`
  - `GET /api/agent/tasks/{task_id}`
  - `GET /api/agent/tasks/{task_id}/subscribe`
- [`kestrel_sovereign/endpoints/auth_oauth.py`](kestrel_sovereign/endpoints/auth_oauth.py)
  - `GET /auth/callback`
  - `GET /auth/login`
  - `GET /auth/logout`
  - `GET /auth/me`
  - `POST /auth/token`
  - `GET /auth/verify`
- [`kestrel_sovereign/endpoints/commands.py`](kestrel_sovereign/endpoints/commands.py)
  - `GET /api/commands`
- [`kestrel_sovereign/endpoints/conversations.py`](kestrel_sovereign/endpoints/conversations.py)
  - `GET /api/conversations`
  - `DELETE /api/conversations/messages/{message_id}`
  - `POST /api/conversations/messages/{message_id}/purge`
  - `POST /api/conversations/messages/{message_id}/restore`
  - `POST /api/conversations/new`
  - `DELETE /api/conversations/{session_id}`
  - `GET /api/conversations/{session_id}`
  - `PATCH /api/conversations/{session_id}`
  - `POST /api/conversations/{session_id}/archive`
  - `POST /api/conversations/{session_id}/purge`
  - `POST /api/conversations/{session_id}/restore`
  - `GET /api/conversations/{session_id}/transcript`
  - `POST /api/conversations/{session_id}/unarchive`
  - `GET /api/sessions`
  - `GET /api/trash`
- [`kestrel_sovereign/endpoints/database.py`](kestrel_sovereign/endpoints/database.py)
  - `GET /api/db/tables`
  - `GET /api/db/tables/{table_name}`
- [`kestrel_sovereign/endpoints/features.py`](kestrel_sovereign/endpoints/features.py)
  - `GET /api/features`
  - `GET /api/features/installed`
  - `GET /api/features/{name}`
  - `GET /api/features/{name}/config`
  - `PATCH /api/features/{name}/config`
  - `POST /api/features/{name}/disable`
  - `POST /api/features/{name}/enable`
  - `POST /api/features/{name}/install`
  - `POST /api/features/{name}/remove`
  - `GET /api/features/{name}/skills`
  - `GET /api/skills`
  - `GET /api/skills/{skill_id}/schema`
  - `GET /api/ui/capabilities`
  - `GET /api/ui/contributions`
- [`kestrel_sovereign/endpoints/files.py`](kestrel_sovereign/endpoints/files.py)
  - `GET /api/agent/channels/{channel_type}/link-qr.png`
  - `GET /api/files/{content_hash}`
  - `HEAD /api/files/{content_hash}`
- [`kestrel_sovereign/endpoints/github.py`](kestrel_sovereign/endpoints/github.py)
  - `GET /api/github/repos`
  - `GET /api/github/{path:path}`
- [`kestrel_sovereign/endpoints/memories.py`](kestrel_sovereign/endpoints/memories.py)
  - `GET /api/identity-chain`
  - `GET /api/memories`
  - `DELETE /api/memories/{node_id}`
  - `GET /api/memories/{node_id}`
- [`kestrel_sovereign/endpoints/metrics.py`](kestrel_sovereign/endpoints/metrics.py)
  - `GET /metrics`
- [`kestrel_sovereign/endpoints/models.py`](kestrel_sovereign/endpoints/models.py)
  - `GET /api/agents`
  - `POST /api/agents`
  - `DELETE /api/agents/{agent_name}`
  - `GET /api/constitution`
  - `GET /api/embedding/models`
  - `POST /api/embedding/reindex`
  - `GET /api/embedding/reindex/{job_id}`
  - `GET /api/embedding/settings`
  - `GET /api/identity`
  - `PATCH /api/identity`
  - `POST /api/identity/avatar`
  - `POST /api/identity/avatar/generate`
  - `GET /api/ipfs/status`
  - `GET /api/keys`
  - `POST /api/keys`
  - `GET /api/keys/available-sources`
  - `GET /api/keys/platform`
  - `GET /api/keys/user`
  - `POST /api/keys/user`
  - `POST /api/keys/user/verify`
  - `DELETE /api/keys/user/{provider}`
  - `DELETE /api/keys/{provider}`
  - `PATCH /api/keys/{provider}`
  - `GET /api/keys/{provider}/usage`
  - `GET /api/model/current`
  - `POST /api/model/set`
  - `GET /api/models`
  - `GET /api/wallet`
  - `POST /v1/chat/completions`
  - `GET /v1/models`
- [`kestrel_sovereign/endpoints/observability.py`](kestrel_sovereign/endpoints/observability.py)
  - `GET /api/observability/metrics/{metric_name}`
  - `GET /api/observability/summary`
- [`kestrel_sovereign/endpoints/rasa_shim.py`](kestrel_sovereign/endpoints/rasa_shim.py)
  - `POST /webhooks/rest/webhook`
- [`kestrel_sovereign/endpoints/restart_events.py`](kestrel_sovereign/endpoints/restart_events.py)
  - `GET /api/restart/status-events`
- [`kestrel_sovereign/endpoints/saved_items.py`](kestrel_sovereign/endpoints/saved_items.py)
  - `GET /api/saved-items`
  - `POST /api/saved-items`
  - `GET /api/saved-items/by-schema/{schema_id}`
  - `GET /api/saved-items/by-tag/{tag}`
  - `GET /api/saved-items/schemas`
  - `POST /api/saved-items/search`
  - `GET /api/saved-items/stats`
  - `POST /api/saved-items/structured`
  - `GET /api/saved-items/tags`
  - `DELETE /api/saved-items/{item_id}`
  - `GET /api/saved-items/{item_id}`
  - `PATCH /api/saved-items/{item_id}`
  - `POST /api/saved-items/{item_id}/pin`
- [`kestrel_sovereign/endpoints/security.py`](kestrel_sovereign/endpoints/security.py)
  - `POST /api/security/approve`
  - `GET /api/security/audit`
  - `GET /api/security/auto-approve/audit`
  - `GET /api/security/auto-approve/rules`
  - `DELETE /api/security/auto-approve/rules/{rule_id}`
  - `GET /api/security/auto-mode`
  - `POST /api/security/auto-mode`
  - `POST /api/security/cancel-all`
  - `POST /api/security/cancel/{request_id}`
  - `GET /api/security/pending`
  - `POST /api/security/permissions`
  - `POST /api/security/permissions/feature`
  - `GET /api/security/permissions/tree`
  - `POST /api/security/reset-session`
- [`kestrel_sovereign/endpoints/sovereignty.py`](kestrel_sovereign/endpoints/sovereignty.py)
  - `POST /api/sovereignty/export`
  - `GET /api/sovereignty/exports`
  - `GET /api/sovereignty/files`
  - `GET /api/sovereignty/files/{filename}`
  - `GET /api/sovereignty/files/{filename}/preview`
  - `POST /api/sovereignty/import`
  - `GET /api/storage/stats`
- [`kestrel_sovereign/endpoints/spawn.py`](kestrel_sovereign/endpoints/spawn.py)
  - `GET /api/spawn/children`
- [`kestrel_sovereign/endpoints/ui.py`](kestrel_sovereign/endpoints/ui.py)
  - `GET /api/ui/theme`
  - `GET /api/ui/themes`

## Command Surface

| Command | Source | Args | Description |
|---|---|---|---|
| `!anchor` | `built-in` |  | Anchor memory state |
| `!create-agent` | `built-in` | `<name>` | Create trusted agent |
| `!backup` | `built-in` | `[--tier local|ipfs|filecoin]` | Create a backup |
| `!promote-backup` | `built-in` | `[--tier ...]` | Save isolated session and backup |
| `!reanchor-constitution` | `built-in` | `<signed_artifact.json> [expected_hash]` | Re-anchor to current constitution after legitimate update |
| `!safe-mode` | `built-in` | `[exit]` | Check or exit safe mode |
| `!verify-constitution` | `built-in` |  | Verify constitution integrity |
| `!compact` | `built-in` | `[--keep N]` | Compact session context |
| `!consolidate` | `built-in` |  | Consolidate memories only |
| `!sleep` | `built-in` | `[--tier ...]` | Consolidate memories and export sovereignty snapshot |
| `!confirm-privacy-mode` | `built-in` | `[mode]` | Confirm a pending data-destructive privacy-mode change |
| `!get-privacy-mode` | `built-in` |  | Get current privacy mode |
| `!privacy` | `built-in` | `[mode]` | Get or set privacy mode |
| `!privacy-discard` | `built-in` |  | Discard isolated session |
| `!privacy-save` | `built-in` |  | Save isolated session |
| `!privacy-status` | `built-in` |  | Detailed privacy status |
| `!set-privacy-mode` | `built-in` | `<mode>` | Set privacy mode |
| `!continue` | `built-in` |  | Continue a stopped request |
| `!legacy-echo` | `built-in` | `<text>` | Echo through the legacy app-context path |
| `!set-app-context` | `built-in` | `<context>` | Set app-specific context for the active session |
| `!heartbeat` | `built-in` |  | Trigger a manual heartbeat check |
| `!help` | `built-in` |  | Show available commands |
| `!reload-context` | `built-in` |  | Hot-reload bootstrap files from disk |
| `!status` | `built-in` |  | Show agent status |
| `!tasks` | `built-in` | `[all|completed|working|failed]` | List background tasks |
| `!read-attachment` | `attachments` | `<attachment_id> [offset] [length] [session_id]` | Read a document the user attached to THIS conversation. Pass the attachment id (a 64-char hex id shown next to the file). Works for text, markdown, and PDF documents. Long documents are returned in chunks: the result reports the character range read and the total size — call again with 'offset' set to 'next_offset' to read the rest. Images can't be read as text — ask the user to paste the image to send it as vision instead. |
| `!audit-anchor` | `audit_anchor` |  | Anchor current audit trail to persistent storage |
| `!audit-status` | `audit_anchor` |  | Check audit anchoring status |
| `!audit-verify` | `audit_anchor` |  | Verify audit trail integrity against anchors |
| `!bootstrap add` | `bootstrap` | `<file_path>` | Add a new bootstrap file to be loaded at startup. file_path resolves relative to the agent data dir when not absolute; the file is registered by BASENAME, so a basename collision loads the search-root copy instead. |
| `!bootstrap list` | `bootstrap` |  | Show all loaded bootstrap files and their paths, sizes, and per-file status (one of: loaded, partial, not found, skipped (budget)). |
| `!bootstrap reload` | `bootstrap` |  | Force reload all bootstrap files from disk. Use after editing SOUL.md or other bootstrap files. |
| `!bootstrap remove` | `bootstrap` | `<name>` | Remove a bootstrap file from the loading convention. name is the basename as shown by bootstrap_list (the file is not deleted from disk). |
| `!bootstrap-status` | `bootstrap` |  | Show the current bootstrap/discovery status. |
| `!rename` | `bootstrap` | `<new_name>` | Rename this agent. new_name must be 1-64 characters. |
| `!restart-discovery` | `bootstrap` |  | Reset and restart the personality discovery process. |
| `!skip-discovery` | `bootstrap` |  | Skip the discovery conversation and use default personality. |
| `!bridge connections` | `bridge` | `[limit]` | List active bridge connections/sessions |
| `!bridge history` | `bridge` | `[limit]` | Show recent bridge invocation history |
| `!bridge status` | `bridge` |  | Show bridge configuration and connection status |
| `!channels history` | `channels` | `[limit] [channel]` | View recent inbound and outbound channel messages. |
| `!channels list` | `channels` |  | List all connected messaging channels and their current status. |
| `!channels send` | `channels` | `<channel> <to> <message>` | Send a message to a recipient via a specific messaging channel. |
| `!cli-status` | `cli` |  | Show platform metadata and registered CLI adapter tool availability. |
| `!git-diff` | `cli` | `[ref] [path] [repo_path]` | Read local repository diff via `git diff`. ref: single git ref, no ranges/`..`; path: repo-relative pathspec (no leading `/` or `..`); repo_path: within allowed repo roots (default cwd). |
| `!git-log` | `cli` | `[max_count] [repo_path]` | Read recent local repository commits via `git log`. max_count is capped at 100; repo_path: within allowed repo roots (default cwd). |
| `!git-merge-base` | `cli` | `<left_ref> <right_ref> [repo_path]` | Read the merge-base for two local git refs. left_ref/right_ref: single git refs, no ranges/`..`; repo_path: within allowed repo roots (default cwd). |
| `!git-show-file` | `cli` | `<ref> <path> [repo_path]` | Read a local repository file from a git ref via `git show`. ref: single git ref, no ranges/`..`; path: repo-relative pathspec (no leading `/` or `..`); repo_path: within allowed repo roots (default cwd). |
| `!git-status` | `cli` | `[repo_path]` | Read local repository status via `git status --short --branch`. repo_path: within allowed repo roots (default cwd). |
| `!compute-caps` | `compute` |  | Query what compute capabilities are available |
| `!compute-empty-trash` | `compute` | `[older_than_days] [dry_run]` | Permanently delete old trash items (requires approval) |
| `!compute-history` | `compute` | `[script_id] [limit]` | Show script execution history |
| `!compute-list` | `compute` | `[state] [limit]` | List all scripts or filter by state. state is one of: 'draft', 'signed', 'pending_review', 'approved', 'rejected', 'queued', 'running', 'completed', 'failed' (case-insensitive), or empty for all scripts. |
| `!compute-restore` | `compute` | `<trash_path> [destination]` | Restore a file from trash to a destination |
| `!compute-run` | `compute` | `<script_id> [executor] [timeout]` | Submit a script for execution (requires security review and user approval). executor is one of: 'uv', 'docker', 'local' (case-insensitive). 'uv' requires Kestrel to run inside a Python venv or virtualenv, 'docker' requires a running Docker daemon, and 'local' requires KESTREL_ALLOW_LOCAL_COMPUTE; any may be unavailable on this host. Call get_compute_capabilities to discover the live set of available executors. |
| `!compute-show` | `compute` | `<script_id>` | Show detailed information about a script |
| `!compute-trash` | `compute` | `[days]` | List files in the trash folder |
| `!compute-write` | `compute` | `<name> <language> <content> <purpose> [requirements]` | Write a new script for later execution. The script is NOT executed immediately - it will be signed, reviewed, and requires user approval. language is one of: 'bash', 'python' (case-insensitive). |
| `!fs-edit` | `computer_use` | `<path> <old_text> <new_text> [occurrence]` | Replace one occurrence of old_text with new_text in a file (always approval-gated). |
| `!fs-list` | `computer_use` | `<path>` | List a directory (allow-list auto-approves; outside list requires human approval). |
| `!fs-read` | `computer_use` | `<path>` | Read a file (allow-list auto-approves; outside list requires human approval). |
| `!fs-write` | `computer_use` | `<path> <content>` | Replace the contents of a file (always approval-gated). |
| `!shell` | `computer_use` | `<command> [timeout]` | Run a shell command. Deny-listed binaries hard-refuse; auto-approved binaries run without a prompt; everything else routes through the ApprovalQueue. |
| `!consent-log` | `consent` | `[limit]` | View recent consent records showing the agent's perspective on past changes. |
| `!consent-stats` | `consent` |  | View consent statistics grouped by action type and sentiment. |
| `!constitution` | `constitution` | `[article] [search] [summary]` | Get the full text of the Kestrel Constitution, or one of its units. Two-slot grammar: 'article' is the subcommand keyword {book, chapter, amendment, section, search, summary} and 'search' is the identifier/term — e.g. article='book' search='I', article='chapter' search='5', article='amendment' search='VIII', article='section' search='III.2', article='search' search='honesty'. Chapter and Section numbering restarts in each Book, so qualify them as <book>.<n> when the bare number is ambiguous. Omit both slots for the full text; article='summary' for the executive summary. |
| `!context compact` | `context` | `[keep_recent] [force] [dry_run]` | Compact context by summarizing older messages. Use when context utilization is high and you need space for new information. |
| `!context compact hierarchical` | `context` | `[chunk_size] [keep_recent] [max_depth]` | Compact context using hierarchical tree-structured summarization (RLM-inspired). Better preserves structure than linear compaction. |
| `!context exclude` | `context` | `<target> <reason>` | Exclude messages from context window (they remain in storage but won't be included in context). Use for redundant, superseded, or irrelevant content. |
| `!context mark` | `context` | `<action> <target> [reason]` | Mark conversation content for context management. Use 'protect' to ensure important content is never pruned, 'droppable' to suggest low-priority content for removal. |
| `!context query` | `context` | `<context_source> <query> [use_cheap_model]` | Query a subset of context using a cheaper model (RLM-inspired sub-LM call). Use for exploring large context sections, compacted originals, or excluded messages without using main model quota. context_source must be one of: 'stash:name', 'excluded', 'compacted:ID', 'summary:ID', 'last_N' (N = message count). |
| `!context restore` | `context` | `[target]` | Restore previously excluded content back to context. |
| `!context stash` | `context` | `[target] [name]` | Stash current working context (like git stash). Use when you need to context-switch to a different topic and want to restore the current discussion later. |
| `!context stash apply` | `context` | `[stash_id]` | Apply a stash without removing it (restore messages but keep stash for reuse). Like git stash apply. Pass stash_id to target a specific stash; leave stash_id empty ("") to target the most recent stash. |
| `!context stash drop` | `context` | `[stash_id]` | Drop a stash without restoring (discard stashed messages). Messages become excluded from context. Pass stash_id to target a specific stash; leave stash_id empty ("") to target the most recent stash. |
| `!context stash list` | `context` |  | List all stashes with their names and message counts. |
| `!context stash peek` | `context` | `[stash_id] [max_chars]` | Peek at stash contents without restoring. Use to explore stashed context programmatically (RLM-inspired context-as-variable). Pass stash_id to target a specific stash; leave stash_id empty ("") to target the most recent stash. |
| `!context stash pop` | `context` | `[stash_id]` | Pop a stash (restore messages and remove from stash list). Like git stash pop. Pass stash_id to target a specific stash; leave stash_id empty ("") to target the most recent stash. |
| `!context stash save` | `context` | `[stash_id] [name] [summary] [tags]` | Save a stash to long-term storage with semantic search capability. Use when you want to preserve context for future retrieval via !recall. Pass stash_id to target a specific stash; leave stash_id empty ("") to target the most recent stash. |
| `!context status` | `context` |  | Check current context window utilization. Use this to understand how much context space is available before deciding to summarize or prune. |
| `!context summarize` | `context` | `<mode> <criteria> [preserve_key_facts]` | Summarize a specific section of conversation history to save context space. Use this to compact verbose exchanges while preserving key information. mode must be one of: time_range, topic, messages, last_n. |
| `!delivery failed` | `delivery` | `[limit]` | List messages in the dead letter queue (permanently failed) |
| `!delivery purge` | `delivery` | `[older_than_hours]` | Clear delivered messages older than 24 hours from the queue |
| `!delivery queue` | `delivery` | `[limit]` | List pending messages in the delivery queue |
| `!delivery retry` | `delivery` | `<message_id>` | Manually retry a failed or dead-lettered message by its entry ID |
| `!delivery status` | `delivery` |  | Show delivery queue status with counts of pending, failed, delivered, and dead letter messages |
| `!deploy` | `deploy` | `[action] [profile] [tag]` | Deploy or manage Kestrel agent on cloud platforms (usage: !deploy <action> [...]). Actions: status, deploy, teardown, logs, list, health (synonyms accepted: start=deploy; stop/delete=teardown; log=logs; ls=list; check=health). profile= is REQUIRED for the deploy, teardown, logs and health actions. tag= must be a CONCRETE image tag (e.g. v0.15.1) for Cloud Run — the 'latest' default is REJECTED by Cloud Run, so always pass an explicit tag when deploying. |
| `!health` | `health` |  | Run a manual liveness check and show results |
| `!health-history` | `health` | `[limit]` | Show recent liveness-check history and uptime |
| `!health-interval` | `health` | `[seconds]` | Change the liveness-check interval |
| `!heartbeat` | `health` |  | [deprecated] alias for !health — use health instead |
| `!heartbeat-interval` | `health` | `[seconds]` | [deprecated] alias for !health-interval — use health_interval instead |
| `!heartbeat-status` | `health` | `[limit]` | [deprecated] alias for !health-history — use health_history instead |
| `!identity assess` | `identity` |  | Assess the current LLM substrate's capabilities and compare with agent requirements. Helps understand limitations when migrating. |
| `!identity export` | `identity` | `[storage_tier] [sign] [include_wallet]` | Export the agent's complete identity to a portable, signed package. This creates a JSON package containing DID, constitution, memories, personality, relationships, and skills that can be imported to another substrate. storage_tier must be one of 'local' (default), 'ipfs', or 'filecoin'; an unrecognized value is rejected (it is NOT silently downgraded to local). |
| `!identity history` | `identity` |  | View the agent's migration history - all substrate changes with timestamps, verification scores, and audit trail. |
| `!identity import` | `identity` | `<source> [verify_signature] [merge_mode] [key_hash] [allow_unsigned] [identity_trust_policy]` | Import agent identity from a portable package. This restores memories, personality, relationships, and skills from a previously exported identity package. merge_mode must be one of: replace, merge (default), skip_existing. |
| `!identity status` | `identity` |  | Show the agent's lifecycle standing — is_test_instance flag, graduation/retirement timestamps, and the list of lifecycle_event records linked to this agent. Lets the agent verify her own graduation/retirement state directly from her DB. |
| `!identity verify` | `identity` | `<source> [key_hash] [identity_trust_policy]` | Verify the integrity of an identity package without importing it. Checks constitution hash, content hash, and signature. |
| `!add-key` | `keys` | `<provider> <api_key> [quota_limit]` | Add an API key for an external service |
| `!delete-key` | `keys` | `<provider>` | Permanently delete a service key. Valid providers: openrouter, openai, anthropic, lighthouse, github, runpod, vastai (use list_providers for the authoritative set). |
| `!key-usage` | `keys` | `<provider> [days]` | Get usage statistics for a service key. Valid providers: openrouter, openai, anthropic, lighthouse, github, runpod, vastai (use list_providers for the authoritative set). |
| `!list-keys` | `keys` |  | List configured service keys (no secrets exposed) |
| `!providers` | `keys` |  | List available service providers |
| `!remove-key` | `keys` | `<provider>` | Remove/deactivate a service key. Valid providers: openrouter, openai, anthropic, lighthouse, github, runpod, vastai (use list_providers for the authoritative set). |
| `!rotate-key` | `keys` | `<provider> <new_api_key>` | Rotate an API key (requires constitutional approval). Valid providers: openrouter, openai, anthropic, lighthouse, github, runpod, vastai (use list_providers for the authoritative set). |
| `!memory action update` | `memory` | `<item_id> [status] [due_date] [assignee_concept_id]` | Update an action item's status (pending/done/cancelled), due date, or assignee. |
| `!memory actions` | `memory` | `[status] [days] [assignee_concept_id] [limit] [include_superseded]` | Retrieve action items the user committed to. Filters: status, creation-date window (days), assignee. Superseded items are excluded by default; pass include_superseded=True to see them. |
| `!memory cases` | `memory` | `<query> [limit]` | Search past audit decisions and constitutional interpretations. Use this when I need precedent for ethical or governance decisions. |
| `!memory confirm-person` | `memory` | `<message_id> <mentioned_label> <concept_id>` | Resolve an ambiguous person mention by confirming which existing concept it refers to. |
| `!memory consolidate` | `memory` |  | Consolidate recent messages into narrative episodes, detect temporal patterns, and archive decayed memories. Runs the cognitive memory pipeline that turns raw conversation into structured long-term memory. Safe to schedule periodically (e.g. nightly). |
| `!memory conversations` | `memory` | `[limit] [include_trashed]` | List your conversation sessions so you can navigate them before pruning. Returns session_id, title, message counts, timestamps and a short preview, most-recent first. Pass include_trashed=True to list soft-deleted sessions available to restore. Use the returned session_id with delete_conversation / restore_conversation / purge_conversation. |
| `!memory decisions` | `memory` | `[limit] [include_superseded]` | Retrieve decisions the user has recorded (stored as graph nodes of type 'decision'). Superseded decisions are excluded by default; pass include_superseded=True to see them. |
| `!memory delete` | `memory` | `<pattern> [confirm] [session_id]` | Delete individual conversation messages matching a text pattern. Pass session_id to confine deletion to one conversation (recommended) so the pattern can't reach across unrelated sessions; omit it only to sweep all conversations. To discard a whole conversation, prefer delete_conversation. Requires Sovereign authorization. |
| `!memory delete-conversation` | `memory` | `<session_id> [confirm]` | Soft-delete an entire conversation session (moves it to Trash; recoverable with restore_conversation). This is the right tool for discarding a whole disposable/test conversation — use it instead of pattern-matching delete_messages. Requires Sovereign authorization. |
| `!memory delete-message` | `memory` | `<message_id> [session_id]` | Soft-delete a SINGLE conversation message by its message_id (moves it to Trash; recoverable with restore_message_by_id). Use this for surgical removal of one specific message — it addresses the row by identity, unlike delete_messages which matches by text. Pass session_id to guard: the delete is refused if that message isn't in the named conversation. Requires Sovereign authorization. |
| `!memory docs` | `memory` | `<query> [limit]` | Search my knowledge base and RAG documents for relevant information. Use this when I need to find information from files, documents, or other stored knowledge. |
| `!memory episodes` | `memory` | `[limit] [query]` | Get consolidated memory episodes - narrative summaries of past conversation themes. Use this for high-level recall of what we've discussed over time. Pass `query` to recall episodes RELEVANT to a topic (semantic search, can surface older episodes); omit it for the most recent episodes. |
| `!memory index-backfill` | `memory` | `[batch_size]` | Start or resume the privacy-preserving lexical memory index backfill. Runs in the background; use memory_status for durable coverage and completion health. |
| `!memory interactions` | `memory` | `<person_concept_id> [limit]` | List recent message→person interactions for a given person concept, with sentiment and topics. |
| `!memory purge-conversation` | `memory` | `<session_id> [confirm] [reason]` | PERMANENTLY delete a conversation session — there is no recovery. Use only to destroy data for good; otherwise prefer delete_conversation (Trash). Requires Sovereign authorization. |
| `!memory purge-message` | `memory` | `<message_id> [confirm] [session_id] [reason]` | PERMANENTLY delete a single message by its message_id — there is no recovery. This is the intentionally-harder path: prefer delete_message_by_id (Trash) unless you must destroy the data for good. Requires confirm=True and Sovereign authorization. |
| `!memory recall` | `memory` | `<query> [mood] [limit] [min_relevance]` | Recall memories with human-like weighting (importance, emotion, recency). Use alongside search_memory for emotionally-aware recall. Scores memories like a human would - important moments and emotionally-charged memories surface first. mood must be one of: positive, negative, neutral (case-insensitive); an unrecognized mood is treated as neutral and the result is returned as PARTIAL. |
| `!memory recent` | `memory` | `[limit]` | Get my most recent conversation messages. Use this to recall what we just discussed or to provide context about our recent interactions. |
| `!memory restore-conversation` | `memory` | `<session_id>` | Restore a soft-deleted conversation session from Trash, bringing its messages back into normal history. Find restorable sessions with list_conversations(include_trashed=True). |
| `!memory restore-message` | `memory` | `<message_id> [session_id]` | Restore a single soft-deleted message from Trash by its message_id, bringing it back into normal history. Pass session_id to guard against acting on the wrong message. |
| `!memory search` | `memory` | `<query> [limit] [session_id]` | PRIMARY TOOL for recalling past conversations. Use this when asked 'do you remember', 'what did we discuss', or any question about past conversations. Decrypts and searches conversation history client-side for reliable results. Pass session_id to scope to a single conversation thread. |
| `!memory status` | `memory` |  | Check memory system health and statistics. Use this to understand my memory capabilities and current state. |
| `!memory supersede` | `memory` | `<old_id> <new_id> [reason]` | Mark a claim node (decision or action_item) as superseded by a newer one. Creates a 'supersedes' edge from new to old and sets superseded_by on the old node. |
| `!memory trash` | `memory` | `[limit]` | List soft-deleted (trashed) individual messages so you can find one to restore_message_by_id or purge_message_by_id. Returns message_id, session_id, role, a short preview, and when it was trashed — most-recently-trashed first. This is the message-level counterpart to list_conversations(include_trashed=True), which lists whole trashed sessions; use this to find messages that were trashed individually (e.g. by delete_messages) inside otherwise-live conversations. |
| `!memory-admin-unpin-all` | `memory_agency` |  | Administrative command: remove ALL active pins for this agent. Sovereign/admin use only. |
| `!memory-admin-unpin-oldest` | `memory_agency` | `<count>` | Administrative command: remove the N oldest active pins. Sovereign/admin use only. |
| `!memory-forget-fact` | `memory_agency` | `<subject> <predicate>` | Delete a current canonical fact previously created by save_fact. Uses the same supported local subject/predicate mapping. |
| `!memory-pin` | `memory_agency` | `<message_id> [reason]` | Pin a memory so it resists decay and stays retrievable. Use this for memories the agent considers important to preserve. |
| `!memory-pin-stats` | `memory_agency` |  | Show memory pin statistics -- total messages, pinned count, released count, ratios, quota usage, and pin age information. |
| `!memory-pinned` | `memory_agency` |  | List all currently pinned memories with their reasons. Use this to review what the agent has chosen to protect. |
| `!memory-release` | `memory_agency` | `<message_id>` | Release a pinned memory so it resumes normal decay. Use this when a previously important memory is no longer needed. |
| `!memory-save-fact` | `memory_agency` | `<subject> <predicate> <value> [confidence]` | Save an explicitly approved canonical fact. The current mapping supports subject 'user' and predicate 'preferred_deploy_region'. Unsupported local terms are rejected rather than guessed. |
| `!model` | `model` |  | Report the currently active AI model. Read-only; takes no arguments. |
| `!model-info` | `model` | `<model_name>` | Get detailed information about a specific model. |
| `!model-list` | `model` | `[use_cache]` | List all available AI models. |
| `!model-pull` | `model` | `<model_name> [progress_callback]` | Download a new AI model (Ollama only). |
| `!model-set` | `model` | `<vendor_or_model> [model]` | Set the active AI model for conversations. Accepts a vendor and model, e.g. set_model('openai', 'gpt-5-mini'), or the 'vendor:route/model' micro-syntax in a single arg, e.g. set_model('anthropic:plan/claude-opus-4-7'). The vendor, route, and model must be real — an unknown triple is rejected, not silently applied. Call list_models first to discover valid vendor/route/model values. |
| `!a2a ask` | `peers` | `<recipient> <message> [session_id] [timeout_seconds] [artifacts] [references]` | Ask another agent a question. Fire-and-resume: this tool POSTs the question, spawns a background SSE subscription on the recipient's task, and returns IMMEDIATELY with ``awaiting_reply=True``. Your current turn ends here. When the recipient's task reaches a terminal state, the ``a2a.question_answered`` signal fires a fresh COGNITION turn on your dispatcher with the reply text inline — respond there. Do NOT block your turn waiting for the answer; the supervisor will wake you. For fire-and-forget use send_a2a_message; for tracked work you'll check on later use send_a2a_task.<br><br>SEND-SIDE ARTIFACTS: pass ``artifacts`` and/or ``references`` to attach durable payload (planning docs, evidence, saved-memory/recall references) to the question so the recipient can retrieve it from the task store while answering. This is the SEND side — distinct from the RESPONDER-side attach_artifact_to_a2a_task tool a recipient uses to attach output onto an incoming task before responding. |
| `!a2a outbound` | `peers` | `[limit] [recipient]` | List the A2A tasks you SENT to peer agents — your local audit log of outbound dispatches (#1576). Each row carries task_id, recipient, verb (message/question/task), dispatch_tool, created_at, and terminal_state (populated after a get_peer_task_result fetch confirms the peer's final state). Use this when you need to enumerate 'what did I send and to whom?' without per-id round trips. |
| `!a2a result` | `peers` | `<recipient> <task_id>` | Fetch the current state + full reply text of an A2A task you previously sent to a peer agent. Use this when an `a2a.question_answered` signal arrived with `truncated=true` (the inline reply was clipped at 8 KiB) — this tool fetches the FULL untruncated body from the peer's task store. Returns the same envelope shape a local `get_task_result` would, but routed through the host proxy to the peer (#1444 truncation recovery path). |
| `!a2a send` | `peers` | `<recipient> <message> [skill_id] [session_id] [artifacts] [references]` | Submit a tracked A2A task to another agent. Persists in the recipient's TaskStore, fires the a2a.task_submitted signal so they wake and process it, returns the task_id for tracking. Caller can poll status via get_a2a_task (or receive the a2a.task_complete signal). Use this for delegated work you'll check on later. For an answer now use send_a2a_question; for a fire-and-forget notification use send_a2a_message.<br><br>SEND-SIDE ARTIFACTS: pass ``artifacts`` and/or ``references`` to hand off durable payload (planning docs, evidence bundles, saved-memory/recall references, logs, diffs) WITH the task — the recipient retrieves them from the task store via get_task_result/check_task_status. This is the SEND side; it is distinct from the RESPONDER-side attach_artifact_to_a2a_task tool, which a RECIPIENT uses to attach output onto an INCOMING task before responding. Each artifact is a dict like {'name': 'plan', 'text': '...'} (or 'data': {...} for structured metadata, optional 'index'/'last_chunk' for chunked bodies). Each reference is a dict descriptor like {'ref_type': 'memory', 'id': '...', 'label': '...'}. |
| `!a2a tell` | `peers` | `<recipient> <message> [session_id]` | Send an async message to another agent — fire-and-forget, no reply expected. Persists in the recipient's TaskStore and fires the a2a.task_submitted signal so they wake and see it on their next cognition turn, but the caller does NOT track lifecycle. Use this for notifications, FYIs, status updates ('I just shipped PR 42'). For a tracked work assignment use send_a2a_task; for a synchronous Q&A use send_a2a_question. |
| `!ask` | `peers` | `<agent_name> <message>` | Send a message to another agent in the multi_agent and get their response. Use this to collaborate, ask questions, or delegate tasks to peer agents. |
| `!peers` | `peers` |  | List all available peer agents in the multi_agent. |
| `!audit` | `response_audit` |  | Show audit configuration and status |
| `!audit-off` | `response_audit` |  | Disable per-response audit |
| `!audit-on` | `response_audit` | `[mode]` | Enable per-response audit. mode: 'warn' (annotate risky responses) or 'strict' (block risky responses). |
| `!restart acknowledge-escalation` | `restart_coordinator` | `<request_id>` | Acknowledge the bounded host-wide escalation policy for one pending restart request migrated from an older release. This is required once for legacy rows before a continuous busy deferral may override fleet quiescence. Pass request_id from list_restart_requests. |
| `!restart cancel` | `restart_coordinator` | `<request_id> [reason]` | Cancel a still-pending restart request (status pending or approved). Rows already updating/executing/completed/rejected/canceled cannot be canceled. Pass request_id from data.request.id of request_restart (or data.requests[].id of list_restart_requests).<br><br>Returns: data={canceled: bool, request_id: str} (plus current_status when the cancel is refused). |
| `!restart coordinator` | `restart_coordinator` |  | ACTION cron task — scan restart_requests, run safety checks, and execute pending requests by spawning a detached restart subprocess. No LLM cost. |
| `!restart events` | `restart_coordinator` | `[limit] [since]` | List recent restart_status lifecycle events for chat-history reload and the agent's pre-turn snapshot. Newest first; uses the typed event records persisted alongside each SSE emit (#1562). |
| `!restart list` | `restart_coordinator` | `[status]` | List restart requests, optionally filtered by status. Valid statuses: pending\|approved\|updating\|executing\|completed\|rejected\|canceled (omit status for all). An unknown status is rejected with the valid set rather than silently returning no rows.<br><br>Returns: data={count: int, requests: [<public dict>, ...]}. |
| `!restart request` | `restart_coordinator` | `<reason> [urgency] [policy] [desired_window] [operation] [update_profile] [target_ref] [repo_path] [allow_migrations]` | File a durable restart request. The host coordinator evaluates safety and executes when conditions are met.<br><br>urgency: one of low\|normal\|high\|critical (default 'normal'); common synonyms are accepted ('medium'→normal, 'urgent'→high, 'emergency'→critical). Higher urgency is executed first.<br>policy: one of idle_agents_only\|allow_busy_after_timeout\|manual_only (default 'idle_agents_only'):<br>  - idle_agents_only: wait for every co-hosted agent to become idle; after a bounded continuous deferral, emit an audited escalation and proceed so one blocker cannot starve the host.<br>  - allow_busy_after_timeout: prefer idle, but execute anyway once the request has aged past the busy timeout even if the agent is still busy.<br>  - manual_only: never auto-execute; the row waits for an explicit dispatch.<br><br>operation='restart_only' (default) restarts the current code and NEVER updates it. operation='update_then_restart' first runs an explicit, allowlisted update profile (e.g. 'sovereign_local_uv_sync': git fetch + checkout target_ref + uv sync) against a local checkout, then restarts into the new code. Update mode requires update_profile and target_ref; repo_path defaults to the local Sovereign checkout. Updating/installing is always explicit and audited — it is never an implicit side effect of a plain restart.<br><br>Returns: data={created: bool, request: <public dict>}. The filed request's id is at data.request.id (NOT a top-level request_id) — pass it to list_restart_requests or cancel_restart_request. |
| `!recall` | `save` | `<query> [item_type] [limit]` | Search saved items. Find previously saved stashes, excerpts, files, and items by meaning; legacy learned-fact graph rows may also appear by keyword during the compatibility window. Optional item_type filter must be one of: stash, file, excerpt, structured; passing one scopes the search to saved items only. |
| `!recall delete` | `save` | `<item_id>` | Delete a saved item by ID. |
| `!recall get` | `save` | `<item_id>` | Get the full content of a saved item by ID. |
| `!recall list` | `save` | `[item_type] [limit]` | List all saved items, optionally filtered by type. Optional item_type filter must be one of: stash, file, excerpt, structured. |
| `!save excerpt` | `save` | `<target> <name> [summary] [tags]` | Save conversation messages for later retrieval. Use this to preserve important discussions, decisions, or information. The target selects which messages: 'last_N' (e.g. last_10) for the most recent N messages, or 'ids:1,2,3' for specific message ids. |
| `!save item` | `save` | `<name> <content> [item_type] [summary] [tags] [schema_id]` | Save arbitrary content (text, JSON) for later retrieval. Good for recipes, notes, decisions, and other content you want to recall. item_type must be one of: stash, file, excerpt, structured (default: structured) — do NOT invent your own type or the item becomes unfindable via recall. To finely type a 'structured' item (recipe, user_story, etc.) pass schema_id, not a custom item_type. |
| `!save stash` | `save` | `[stash_id] [name] [summary] [tags]` | Save a stash to long-term storage for later retrieval. The stash content gets an embedding so you can find it later with semantic search. |
| `!schedule add` | `scheduler` | `<cron_expression> <task_name> [args_json] [timezone_name] [misfire_policy] [misfire_grace_seconds] [idempotency_key]` | Add a new scheduled task with a cron expression |
| `!schedule deadline` | `scheduler` | `[run_at] [task_name] [args_json] [misfire_policy] [misfire_grace_seconds] [idempotency_key] [delay_seconds]` | Add a one-shot scheduled task that fires once at an absolute deadline (run_at) or after a relative delay (delay_seconds). Use task_name='self_followup' with {"intent": "..."} in args_json to schedule your own follow-up turn |
| `!schedule engagement` | `scheduler` | `[days]` | Report aggregate engagement scores per scheduled task |
| `!schedule history` | `scheduler` | `[limit]` | Show recent task execution history |
| `!schedule list` | `scheduler` |  | List all scheduled tasks for this agent |
| `!schedule outcome` | `scheduler` | `<execution_id> <signal>` | Attach an engagement signal (0.0-1.0) to a past task execution |
| `!schedule pause` | `scheduler` | `<task_id>` | Pause a scheduled task (stops it from running until resumed) |
| `!schedule remove` | `scheduler` | `<task_id>` | Remove a scheduled task by ID |
| `!schedule resume` | `scheduler` | `<task_id> [acknowledge_ambiguous_effect]` | Resume a paused scheduled task |
| `!schedule self-followups` | `scheduler` | `[limit]` | Show follow-up turns this agent scheduled for itself, and whether each one fired, is still pending, or was missed |
| `!schedule update` | `scheduler` | `<task_id> <cron_expression> [timezone_name]` | Update the cron expression of an existing scheduled task |
| `!security-approve` | `security` | `<request_id> [scope]` | Approve a pending request. scope: 'once', 'session', or 'always'. |
| `!security-audit` | `security` | `[limit]` | Show recent security audit log |
| `!security-deny` | `security` | `<request_id>` | Deny a pending request |
| `!security-list` | `security` |  | List all configured security permissions in tree format |
| `!security-pending` | `security` |  | Show pending approval requests |
| `!security-set` | `security` | `<feature_name> [tool_name] [level]` | Set permission level for a tool or all tools in a feature. level is one of: allow, auto, always_ask, deny, ask, session. Call list_permissions to enumerate valid feature_name/tool_name values — setting a permission on an unregistered name does not take effect. |
| `!skill candidates` | `skills` | `[min_confidence] [limit]` | List reflection insights that are candidates for skill extraction |
| `!skill delete` | `skills` | `<skill_id>` | Remove a saved skill (file + graph node) |
| `!skill list` | `skills` |  | List all extracted skills for this agent |
| `!skill save` | `skills` | `<insight_id> <steps_json> <verification> [tags_json]` | Promote a reflection insight into a saved skill |
| `!skill show` | `skills` | `<skill_id>` | Show the full content of an extracted skill |
| `!check-sovereignty-status` | `sovereignty` |  | Check the status of sovereignty backups. |
| `!export-sovereignty` | `sovereignty` | `[storage_tier] [encrypt] [on_progress]` | Export the agent's entire state to IPFS/Filecoin for sovereignty backup. storage_tier must be one of 'local', 'ipfs' (default), or 'filecoin'; an unrecognized value is rejected (it is NOT silently defaulted to ipfs). |
| `!import-sovereignty` | `sovereignty` | `<cid>` | Restore this agent's CONVERSATION HISTORY from a prior backup (IPFS CID). Faithfully preserves message timestamps and trash state. NOTE: this currently restores conversation history only — NOT full agent state (memories, knowledge graph, saved items, files, settings). Full-state restore is tracked separately. |
| `!state-of-mind` | `state_of_mind` |  | Get the current constitutional governance state for this agent |
| `!blockers` | `strategic_memory` | `[limit] [include_resolved]` | Recall blockers from the strategy index (graph nodes of type 'strategy_blocker'). Resolved blockers are excluded by default; pass include_resolved=True to see them. |
| `!dispatch` | `strategic_memory` | `[mode]` | Pick the highest-priority issue from strategic memory and start it through a live feature-contributed dispatch workflow. Preview with mode='suggest'; execute fails closed when no compatible workflow capability and governed runner are enabled. |
| `!hygiene` | `strategic_memory` | `[fix]` | Scan all repos for backlog hygiene issues: missing assignees, milestones, status labels. Reports gaps and flags items needing human review. |
| `!morning` | `strategic_memory` |  | Generate a morning strategic briefing -- milestone status, blockers, recommended work items. Pulls live data from GitHub when GITHUB_TOKEN is available. |
| `!patterns` | `strategic_memory` | `[limit] [include_superseded]` | Recall learned patterns from the strategy index (graph nodes of type 'strategy_pattern'). Superseded patterns are excluded by default; pass include_superseded=True to see them. |
| `!sessionlog` | `strategic_memory` | `[session_id] [focus]` | End-of-day session log collector. Scans all repos for today's activity (issues closed, PRs merged, comments, commits) and generates a structured session summary with outcomes and metrics. |
| `!strategy` | `strategic_memory` | `[section]` | View the current strategic context: vision, milestones, stakeholders, decisions, blockers, and patterns. |
| `!strategy-reconcile` | `strategic_memory` | `[apply]` | Check each active blocker against live GitHub issue state and report which reference already-closed issues. Pass apply='yes' to resolve the stale rows. |
| `!strategy-search` | `strategic_memory` | `<query> [kind] [limit] [include_retired]` | Search the strategy ledger (learned patterns and blockers) by keyword. This is the query path that replaced dumping the whole log into the system prompt. |
| `!a2a attach` | `tasks` | `<task_id> <name> <content> [index] [last_chunk]` | RESPONDER-SIDE artifact attach: the RECIPIENT of an incoming A2A task uses this to attach its own output. To attach payload as the SENDER of an outgoing task, pass ``artifacts``/``references`` to send_a2a_task / send_a2a_question instead. Attach one chunk of long-form output as an Artifact to an incoming A2A task BEFORE calling respond_to_a2a_task. Use this when your reply exceeds the per-tool argument cap (10K chars) — chunk the body into segments of <=9000 chars each, call this tool once per segment with monotonically-increasing index (0, 1, 2, ...) and last_chunk=False on every segment except the final one. The sender's get_peer_task_result returns the artifacts in order so the resumed turn can reassemble the full body. After all segments are attached, call respond_to_a2a_task with a SHORT content like 'See attached artifacts (N segments).' so the sender knows where to look. |
| `!a2a respond` | `tasks` | `<task_id> <content> [state]` | Respond to an incoming A2A task in your inbox by transitioning it to a terminal state with your reply text. Use this when another agent sent you a task via send_a2a_question (fire-and-resume — sender's turn ended, they wake on the a2a.question_answered signal when you transition), send_a2a_message (FYI, brief receipt), or send_a2a_task (delegated work, full result). The sender's subscription supervisor on the SSE stream picks up your terminal frame and fires their resumption signal. Without this tool the sender's send_a2a_question lineage never resumes until the hourly expiry sweep fires a state='expired' signal. |
| `!cancel-task` | `tasks` | `<task_id> [reason]` | Cancel a pending or running task. |
| `!list-skills` | `tasks` |  | List all available features and their skills that can be used with run_workflow. Returns feature names, skill names, and descriptions. Call this first to discover what skills are available before building a workflow plan. |
| `!run-workflow` | `tasks` | `<steps>` | Execute a multi-step plan across features. Each step runs a specific feature skill with arguments. All steps execute sequentially and results are returned together. Use this instead of making individual subagent calls when you need to gather information from multiple features. Steps format: [{"feature": "feature_name", "skill": "skill_name", "args": {}}]. Feature names match the tool names shown in your available tools (e.g., model_agent, memory_feature, wallet_feature). Skill names are the individual tool methods within each feature (e.g., list_models, memory_status, check_balance). Args can reference prior step outputs with {{steps.N.result}} or {{prev.result}}. Steps can optionally include max_retries (default 0) and retry_delay_ms (default 1000). |
| `!task-result` | `tasks` | `<task_id>` | Get the result/artifacts from a completed task. The returned message field is the REPLY content, distinct from request_content (what was originally ASKED) surfaced by check_task_status. |
| `!task-status` | `tasks` | `<task_id>` | Check the status of a background task by ID. Returns two distinct content fields: request_content (what was ASKED — the inbound sender's message) and message (the REPLY that has been written back, if any). |
| `!tasks` | `tasks` | `[status] [task_type] [limit]` | List background tasks, optionally filtered by status or type. status must be one of the TaskState values: submitted, working, completed, failed, canceled (case-insensitive). With no status, returns the pending (submitted) inbox; a status filter queries tasks across ALL states. |
| `!todo add` | `todo` | `<title> [description] [scope] [status] [priority] [owner] [links] [terminal_condition] [next_check_at] [source_metadata]` | Create a durable active todo with terminal criteria and optional external links. |
| `!todo complete` | `todo` | `<todo_id> [outcome] [evidence] [terminal_condition_satisfied] [superseded_by]` | Mark a todo done only when its terminal condition is explicitly satisfied, or cancel/supersede it with a reason. |
| `!todo link` | `todo` | `<todo_id> <link_type> <target> [title] [status] [url] [metadata]` | Attach an external reference to a todo, such as a GitHub issue, coding workflow run, A2A task, scheduled job, restart request, action item, or evidence URL. |
| `!todo list` | `todo` | `[scope] [status] [owner] [include_done] [include_superseded] [limit]` | List durable todos by scope/status/owner, excluding superseded items by default. |
| `!todo rollup` | `todo` | `[include_done] [limit]` | Summarize pending/waiting/in-progress todos across sessions and linked systems. |
| `!todo update` | `todo` | `<todo_id> [title] [description] [scope] [status] [priority] [owner] [links] [terminal_condition] [next_check_at] [superseded_by] [source_metadata]` | Update an active todo without marking it complete unless status is explicitly terminal. |
| `!wait` | `wait` | `[target] [duration_seconds] [timeout_seconds] [poll_interval_seconds] [reason] [mode]` | The ONE generic wait — works across EVERY feature. There is no per-feature wait tool; whatever async work a loaded feature exposes, you wait on it here with `target="<kind>:<handle>"`.<br>Known handle kinds (each contributed by a feature; more may be registered by whatever features are loaded):<br>• `task:<task_id>` — a LOCAL Kestrel background task (this agent's own store)<br>• `a2a:<task_id>` — an OUTBOUND A2A TASK you sent a peer via send_a2a_task (route it here, NOT `task:` — a `task:` on an outbound A2A id is a provider mismatch and is rejected at registration). A2A QUESTIONS are NOT watched here: send_a2a_question already wakes you via its own `a2a.question_answered` signal, so an `a2a:<question-id>` watch is rejected to avoid waking you twice for one answer.<br>• `ci:<owner/repo#N>` — a GitHub PR's merge/CI-check state<br>• `lora_train:<...>`, `tx:<...>`, `workflow:<run_id>` and others when those features are present.<br>A kind being LISTED here is documentation, not a guarantee it is AVAILABLE: a provider is only reachable when its feature is loaded. If you pass an unknown/unavailable kind, the error lists the kinds currently registered. A registered kind's signal-mode watch is durable and RE-ARMS across restart; availability (is the provider loaded?) and re-arming (does a live watch resume?) are separate — a documented kind whose feature is not loaded neither registers nor re-arms.<br><br>Three ways to call it:<br>• `target="<kind>:<handle>"` (default `mode="block"`) — hold the turn, polling until that thing reaches a terminal state or the timeout expires; returns the terminal outcome (or a still-pending result on timeout).<br>• `target="<kind>:<handle>", mode="signal"` — register a watch and return IMMEDIATELY; the wait reconciler wakes you with a `wait.complete` cognition signal once it finishes. Use this for long/unattended waits so you don't hold a turn.<br>• `duration_seconds=N` (no target) — a plain bounded pause, the native alternative to shelling out to `sleep` between polls in an autonomous loop. |
| `!web-search` | `web_search` | `<query> [max_results]` | Search the web for information. max_results is typically 1-10 (default 5). A 'disabled' error means no search provider is configured — set a provider API key (e.g. TAVILY_API_KEY). |
| `!webhooks history` | `webhooks` | `[limit]` | Show recent webhook receive log for security audit |
| `!webhooks list` | `webhooks` |  | List all registered webhook endpoints |
| `!webhooks register` | `webhooks` | `<name> [auth_type] [event_type] [auth_config_json] [rate_limit] [allow_unauthenticated]` | Register a new webhook endpoint with authentication |
| `!webhooks remove` | `webhooks` | `<name>` | Remove a registered webhook endpoint |
| `!wellness` | `wellness` |  | Check agent operational wellness across 5 dimensions |
| `!wellness-history` | `wellness` | `[limit]` | View wellness trends over time |

<!-- END AUTO-GENERATED FEATURE INVENTORY -->




















## Authentication Surface

The route surface is not just public versus protected. The current live classes are:

- `Public`
  - `/health`
  - `/favicon.ico`
- `Public-Localhost`
  - `/api/auth/key` when bootstrap is enabled
- `OAuth public entrypoints`
  - `/auth/login`
  - `/auth/callback`
  - `/auth/logout`
- `APIKeyOrSession`
  - `/health/detailed`
  - most protected `/agent/*` and `/api/*` routes via `kestrel_sovereign/server.py` auth middleware
- `APIKeyOrSession+SSEQuery`
  - SSE paths that also allow `?api_key=`:
    - `/agent/stream`
    - `/agent/notifications/sse`
- `OAuthSessionSemantic`
  - `/auth/me` can pass middleware via API key or session, but only returns authenticated data from a real session
- `Browser-Conditional`
  - `/` serves UI for local/browser conditions and redirects to OAuth when OAuth-required mode is enabled

## Generated and Historical Documents

- Canonical source:
  - [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md)
- Generated audience docs:
  - [`docs/generated/README.md`](docs/generated/README.md)
- Historical snapshot:
  - [`docs/archive/KESTREL_FEATURES_legacy.md`](docs/archive/KESTREL_FEATURES_legacy.md)

## Audit and Verification

- Audit working papers live under [`docs/audit/`](docs/audit).
- Fast proof layers for the canonical surface live in:
  - [`tests/unit/test_auth_decision_table.py`](tests/unit/test_auth_decision_table.py)
  - [`tests/unit/test_endpoint_contract_suite.py`](tests/unit/test_endpoint_contract_suite.py)
  - [`tests/unit/test_feature_doc_canonicality.py`](tests/unit/test_feature_doc_canonicality.py)
  - [`tests/unit/test_generate_feature_docs.py`](tests/unit/test_generate_feature_docs.py)

## Known Boundaries

- Some support packages live under `kestrel_sovereign/features/` but are not discoverable features because they do not export a `Feature` subclass.
- Generated audience docs require an LLM provider key; dry-run validation should pass even when generation keys are absent.
