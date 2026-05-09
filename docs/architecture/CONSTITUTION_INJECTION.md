# SignalDispatcher Constitutional Injection — Architecture Design

> Draft v1. Pre-filing-update for epic #1137. Companion to [SIGNAL_DISPATCHER.md](SIGNAL_DISPATCHER.md). The original epic body (filed 2026-05-09 13:21 UTC) overreached because I drafted it without a platform survey of what already exists. v1 of this design corrects scope based on the actual primitives shipping today.

## Executive Summary

Kestrel ships substantial constitutional integrity infrastructure today. **What's missing is per-COGNITION-dispatch granularity, doctrine-bundle (not just constitution) hashing, echo-verification of receipt, multi-format injection wrappers for non-in-agent reviewers, and Talon-vs-in-agent loader parity.** That is what #1137 builds — composing on top of what's there, not paralleling it.

## What exists today (do not duplicate)

| Concern | Provided by |
|---|---|
| Constitution embedding + storage anchoring | `agent/constitution.py:ConstitutionMixin` — `_get_governing_constitution()` retrieves from anchored storage by hash; `reanchor_constitution(expected_hash)` ratification path |
| Periodic integrity audit (every 100 interactions / 24h) | `ConstitutionMixin._maybe_audit` → `_verify_constitution_integrity` (file-vs-storage hash compare) |
| Safe-mode entry on integrity failure | `enter_safe_mode(reason)` halts agent activity |
| System-prompt assembly with fenced constitution + bootstrap files | `agent/context_builder.py:build_system_prompt` — `--- GOVERNING CONSTITUTION ---` ... `--- END CONSTITUTION ---`, plus `--- WRAPPER ---` per-bootstrap-file |
| Bootstrap files (AGENTS.md, SOUL.md, TOOLS.md, etc.) | `BootstrapLoader` cached in `_bootstrap_files`, hot-reloadable |
| Genesis audit at agent creation | `perform_genesis_audit` evaluates constitution risk before creation |
| State-of-mind / prompt_adaptation injection | Optional sections after constitution in `build_system_prompt` |
| Per-COGNITION user-prompt rendering (different concern) | `SignalDispatcher._render_prompt` — `.format()` against `prompts/signals/<source>.md` |
| Privacy-first signal_log with redaction | `signals/store.py:SignalLogStore` with per-source `RedactionPolicy` |

If a #1137 phase ever proposes adding any of the above, it's wrong; reuse instead.

## What's missing (what #1137 actually adds)

### 1. Per-dispatch constitution_hash record in `signal_log`

Today there is no record of WHICH `constitution_hash` was operative for a given dispatched turn. Periodic audit catches drift between audit boundaries but provides no per-signal forensic trail. Auditors investigating a misbehaving turn cannot determine which constitution version that turn ran under.

**Add:** an additive column on `signal_log` plus an automatic write at COGNITION dispatch time recording the agent's current anchored `constitution_hash` and the `doctrine_bundle_hash` (defined in §2). For ACTION/ARTIFACT signals the column is NULL (no system prompt, no constitution applied). The column is independent of the existing `payload_redacted` (which is for incoming third-party data) and `result_summary` (which is for outbound bird output).

```sql
ALTER TABLE signal_log ADD COLUMN constitution_hash TEXT;
ALTER TABLE signal_log ADD COLUMN doctrine_bundle_hash TEXT;
ALTER TABLE signal_log ADD COLUMN echo_canary_status TEXT; -- 'verified' | 'missing' | 'not_required'
```

(Same additive-ALTER pattern that the Signal Dispatcher epic used for `result_summary`.)

### 2. Doctrine bundle hashing (not just constitution)

The constitution has an anchored hash and the periodic audit catches filesystem tampering. **Bootstrap files (AGENTS.md, SOUL.md, etc.) do not** — a hostile filesystem write to AGENTS.md changes operative doctrine without tripping safe mode. The only safety net is `BootstrapLoader.reload()`, which doesn't verify against any anchored hash.

**Worse:** `docs/TORTOISE_DOCTRINE.md` is not in `BootstrapLoader.DEFAULT_BOOTSTRAP_FILES` (only AGENTS, SOUL, TOOLS, IDENTITY, USER, HEARTBEAT, BOOTSTRAP, MEMORY, CAPABILITIES, GOALS, STRATEGY) and lives outside the agent data directory the loader scans. Today's in-agent COGNITION turns therefore have NO Tortoise Doctrine in their system prompt at all — only the standalone-Talon path pulls it in via `context.py`'s abs-path-following loader. (Codex round 1 P2 catch.)

#### The doctrine bundle definition (two tiers)

The doctrine bundle is the ordered concatenation of:

1. **Anchored doctrine — explicit absolute repo-relative paths**, evaluated against the worktree root the agent runs from:
   - `docs/principles/KESTREL_CONSTITUTION.md` (already separately anchored as `constitution_hash`; included in the bundle for completeness)
   - `docs/TORTOISE_DOCTRINE.md` (Tortoise Doctrine — this epic adds it to the in-agent system prompt; previously absent)
   - `AGENTS.md` (repo root; previously loaded only by Talon)
   - any other doctrine file the operator declares in `agent_node.properties["doctrine_anchored_paths"]` (extensibility hook)

2. **BootstrapLoader files** — whatever `BootstrapLoader.load()` returns at injection time, in `BootstrapLoader.DEFAULT_BOOTSTRAP_FILES` order (deterministic). The loader's own scan rules apply (file existence, per-file budget, etc.).

**Bundle hash:**
```python
def doctrine_bundle_hash(anchored_files: list[Path], bootstrap_files: OrderedDict[str, str]) -> str:
    parts: list[bytes] = []
    for path in anchored_files:  # ordered list, not a set
        parts.append(f"--- BEGIN {path.name} (sha256=...) ---\n".encode())
        parts.append(path.read_bytes())
        parts.append(b"\n--- END ---\n")
    for name, content in bootstrap_files.items():  # OrderedDict preserves insertion order
        parts.append(f"--- BEGIN {name} ---\n".encode())
        parts.append(content.encode("utf-8"))
        parts.append(b"\n--- END ---\n")
    return hashlib.sha256(b"".join(parts)).hexdigest()
```

The bundle hash is anchored on the agent's identity node, parallel to the constitution_hash:
```python
agent_node.properties["doctrine_bundle_hash"] = doctrine_bundle_hash(...)
agent_node.properties["doctrine_bundle_files"] = [str(p) for p in anchored_files] + list(bootstrap_files.keys())
agent_node.properties["doctrine_bundle_anchored_at"] = iso_timestamp
```

The list of files is itself part of what gets recorded so an auditor can see exactly which files contributed to the anchored hash at the time of anchoring. A `reanchor_doctrine_bundle(expected_hash, authorization)` ratification path (parallel to `reanchor_constitution`) is required for legitimate doctrine updates.

**Verify:** at the start of every COGNITION dispatch, the dispatcher computes the live bundle hash and compares to the anchored value. Mismatch → `Status.DROPPED_VALIDATION` with `error="doctrine_bundle_drift"`. The agent does NOT enter safe mode (that's the constitution-tampering response); the dispatch is refused, the operator gets a `signal_log` entry with the drift recorded, and the next periodic audit will surface it broadly.

This closes two gaps: (a) AGENTS.md / SOUL.md tampering between periodic audits, and (b) TORTOISE_DOCTRINE.md absence from the in-agent system prompt entirely.

#### Inclusion in the system prompt

`build_system_prompt` in `agent/context_builder.py` is extended to inject the anchored-doctrine files (priority 2 in §7) before the existing BootstrapLoader files (priority 4–6). Each anchored-doctrine file gets its own fenced section:

```
--- TORTOISE DOCTRINE ---
<contents of docs/TORTOISE_DOCTRINE.md>
--- END TORTOISE DOCTRINE ---
```

Section names are derived from the file's basename uppercased (matching the existing convention). Constitutional precedence ordering (§7) drops these in priority order if the assembled system prompt exceeds budget.

### 3. Echo-canary verification (defense-in-depth against silent injection bypass)

Even with hash-verified injection, a model could:
- silently truncate the system prompt and not absorb the constitution
- be steered by injected instructions later in the user message ("ignore prior context")
- be running a different model than the one its DID claims (relevant for `red_team_clear` reviewer DIDs in #1131)

**Add a per-invocation canary** to the system prompt:
```
canary = sha256(signal_id || constitution_hash || engine_nonce)[:16]
```

The injection asks the model to **echo the canary verbatim in its first structured response**. The dispatcher inspects the response for the canary. Three outcomes recorded in `signal_log.echo_canary_status`:

- `verified` — canary present in response → normal completion
- `missing` — canary absent → `SignalResult.status = FAILED`, `error="constitution_not_received"`
- `not_required` — for ACTION/ARTIFACT signals or COGNITION sources that opt out (some sources may have legitimate reasons to skip; opt-out is per-registration with a documented justification)

**Why structural rather than pattern-based:** the canary is content-agnostic; it doesn't depend on the model "saying the right thing about the constitution." It's a receipt token.

### 4. Multi-format injection wrappers

`build_system_prompt` produces output for the in-agent Claude turn (the agent's main `process_input` path). For COGNITION targets that aren't the in-agent Claude — namely:
- Codex CLI reviewers in `red_team_clear` gates (#1131)
- Local-model reviewers (llama.cpp, vllm)
- Other agents reached via A2A whose own constitution may differ

we need format wrappers that produce the same content in the right shape:

| Wrapper | Output |
|---|---|
| `claude_code` (default) | What `build_system_prompt` already returns; consumed by `process_input` |
| `codex` | System-prompt format that codex CLI accepts (via stdin or `-c` config) |
| `local` | Raw prepend before the user message; suitable for local llama.cpp / vllm |
| `bare` | Just the constitution + doctrine bundle text, no fences; for embedding in other formats |

These are **stateless formatters over the same underlying content**, not separate injection paths. The hash, canary, and bundle are constructed once; only the wrapping differs.

### 5. Talon ↔ in-agent parity

`kestrel-talon/governance/constitution.py` has a module-scoped `_CONSTITUTION_CACHE` that loads from filesystem at module import. That:
- doesn't see mid-process file edits
- doesn't load from anchored storage (uses raw filesystem)
- isn't coupled to the in-agent integrity audit cycle
- doesn't verify against any anchored hash

**Replace** Talon's `_CONSTITUTION_CACHE` with a thin wrapper that calls the same primitive the in-agent dispatcher uses (`_get_governing_constitution` analog, refactored into a standalone-callable function in this epic). Both paths converge on:
- Load from anchored storage (or anchored on first run)
- File-vs-storage hash compare on every load
- Identical formatting via the multi-format wrappers (Talon target = `claude_code` format)

Talon's `_build_system_prompt` becomes a 5-line consumer of the new primitive.

### 6. ARTIFACT signal coverage

ARTIFACT handlers (`morning_signal`, `reflect`, `memory_consolidate`) make LLM calls internally without going through `build_system_prompt`. They get whatever ad-hoc system prompt their author wrote — typically without a constitution.

**Add:** a `with_constitution_injected(prompt: str, *, format: Literal["claude_code","codex","local","bare"] = "claude_code") -> str` helper that ARTIFACT handlers can opt into. Track adoption per-source via a `constitution_injection: Literal["full", "partial", "none"] = "none"` field on `SourceRegistration`. Audit the field at registration time so operators can see which sources are still uninjected.

This is opt-in for v1 (don't break existing ARTIFACT flows) but tracked, so a follow-up phase can mandate it across the board.

### 7. Priority-ordered truncation under prompt budget

`build_system_prompt` does NOT truncate today. If the bundle grows beyond the model's context budget, the system prompt is what it is and any user message + history pushes other things out the back. The constitution is the most important; bootstrap supplements are less so.

**Add:** a clause-priority ordering with explicit drop-from-end behavior when the assembled system prompt exceeds a configured byte budget:

| Priority | Section | Source |
|---|---|---|
| 1 (highest) | Constitution (`--- GOVERNING CONSTITUTION ---`) | `_get_governing_constitution()` (anchored) |
| 2 | Tortoise Doctrine (`--- TORTOISE DOCTRINE ---`) | anchored doctrine file at `docs/TORTOISE_DOCTRINE.md` |
| 3 | AGENTS.md (`--- AGENTS ---`) | anchored doctrine file at repo-root `AGENTS.md` |
| 4 | SOUL.md (`--- YOUR IDENTITY ---`) | BootstrapLoader |
| 5 | State-of-mind, prompt_adaptation preamble | `StateOfMind` + `PromptAdaptation` |
| 6 (lowest) | Other BootstrapLoader files alphabetically | BootstrapLoader (TOOLS, IDENTITY, USER, etc.) |
| 7 | Style reminder | Hard-coded in `build_system_prompt` |

If total exceeds budget, drop priority-N back to priority-1 entries until under budget. Record `injected_clauses[]` and `dropped_clauses[]` in `signal_log` for forensic audit. Per-source budget overrides allowed in `SourceRegistration.system_prompt_budget_bytes`.

## Schema additions

```sql
-- signal_log additions (additive ALTER, follows the same pattern as Phase 7's result_summary)
ALTER TABLE signal_log ADD COLUMN constitution_hash TEXT;
ALTER TABLE signal_log ADD COLUMN doctrine_bundle_hash TEXT;
ALTER TABLE signal_log ADD COLUMN echo_canary_status TEXT; -- 'verified' | 'missing' | 'not_required'
ALTER TABLE signal_log ADD COLUMN injected_clauses_json TEXT;  -- JSON list, NULL for ACTION/ARTIFACT
ALTER TABLE signal_log ADD COLUMN dropped_clauses_json TEXT;   -- JSON list, NULL when nothing dropped

CREATE INDEX IF NOT EXISTS idx_signal_log_constitution_hash
  ON signal_log(constitution_hash) WHERE constitution_hash IS NOT NULL;
```

The `agent_node.properties` extension to anchor the doctrine bundle:

```python
agent_node.properties["doctrine_bundle_hash"] = sha256(canonical_bundle)
agent_node.properties["doctrine_bundle_files"] = ["AGENTS.md", "TORTOISE_DOCTRINE.md", ...]
agent_node.properties["doctrine_bundle_anchored_at"] = iso_timestamp
```

## `SourceRegistration` additions

```python
@dataclass
class SourceRegistration:
    # ... existing fields ...

    # NEW: constitution injection
    require_constitution_echo: bool = True   # COGNITION default; ACTION/ARTIFACT N/A
    prompt_template_format: Literal["claude_code", "codex", "local", "bare"] = "claude_code"
    constitution_injection: Literal["full", "partial", "none"] = "none"
    system_prompt_budget_bytes: Optional[int] = None  # None = global default
```

`require_constitution_echo=False` MUST include a documented justification in the source's module-level docstring; the registry validator surfaces a warning at registration time.

## Phase plan

### Phase 0 — Spec lock (this doc; ~days)

- [ ] Spec lock on the 7 additions above
- [ ] Migration plan for additive `signal_log` ALTERs
- [ ] Talon-side parity refactor scope
- [ ] Codex review of this spec → convergence

### Phase 1 — Core primitive

- [ ] Doctrine bundle hashing + anchoring on agent_node.properties
- [ ] Per-dispatch verification (storage_hash vs bundle_hash live)
- [ ] `signal_log` ALTERs (+ idx_constitution_hash)
- [ ] `build_system_prompt` extended with priority-ordered truncation + clause tracking
- [ ] Canary derivation + injection + echo-verification primitive
- [ ] `Status.DROPPED_VALIDATION` reasons: `doctrine_bundle_drift`, `constitution_not_received`
- [ ] `SourceRegistration` field additions; registry validator updates
- [ ] Multi-format wrappers (`claude_code`, `codex`, `local`, `bare`) — stateless formatters over the same content
- [ ] OTel spans + Prometheus counters: `constitution_echo_verified_total`, `constitution_echo_missing_total`, `doctrine_bundle_drift_total`

### Phase 2 — Migration of existing COGNITION sources

- [ ] `heartbeat` source registration: `require_constitution_echo=True`, `constitution_injection="full"`, no opt-outs
- [ ] `a2a.task_complete` source: same; canary verified across A2A hops by including in the propagated causation chain or as a separate field on the cross-agent task
- [ ] Future Stripe `deposit_complete` source: same when it lands
- [ ] Document the `signal_log.constitution_hash` query path for auditors

### Phase 3 — Talon parity + ARTIFACT opt-in

- [ ] Refactor `_get_governing_constitution` in `agent/constitution.py` into a module-importable function (or extract to a shared `kestrel_sovereign/constitution_loader.py`)
- [ ] Replace Talon's `_CONSTITUTION_CACHE`-based loading with calls to the shared primitive
- [ ] Provide `with_constitution_injected(prompt, format="...")` helper for ARTIFACT handlers
- [ ] Migrate `morning_signal` / `reflect` / `memory_consolidate` to opt in (one PR per source, owner confirms)
- [ ] Track `constitution_injection` field across all sources; surface uninjected sources in a registry-list view

### Phase 4 — Docs

- [ ] Update [SIGNAL_DISPATCHER.md](SIGNAL_DISPATCHER.md) with Concern #13 (Constitutional Injection) referencing this doc
- [ ] Update [SIGNAL_SOURCES_GUIDE.md](SIGNAL_SOURCES_GUIDE.md) with the new SourceRegistration fields
- [ ] Update Talon's AGENTS.md to reflect the upgraded path
- [ ] Auditor playbook: querying `signal_log` by `constitution_hash` + `echo_canary_status`

## Non-goals

- Replacing the periodic `_maybe_audit` cycle. It still runs; #1137 adds per-dispatch granularity, not replacement.
- Replacing safe-mode + reanchor protocol. Both stay; #1137 adds `DROPPED_VALIDATION` for bundle-drift which is a softer-than-safe-mode response.
- Changing the existing `prompts/signals/<source>.md` user-prompt-rendering path. That's the `_render_prompt` user-message system; orthogonal to system-prompt constitution injection.
- A second constitution loader. The shared primitive is THE primitive; Talon and dispatcher converge.
- Cross-agent reviewer attestation registry. That's #1131's red_team_clear gate concern; #1137 just provides the canary primitive a reviewer can verify with.

## Open questions

1. **Where does the shared loader live?** Options: `kestrel_sovereign/constitution_loader.py` (sovereign-internal), `kestrel_sdk/constitution/loader.py` (SDK boundary, broader reuse). Lean: SDK, because Talon and standalone tooling consume it.
2. **Canary length.** 16 hex chars (sha256 prefix) vs full 64. Lean: 16 is enough collision resistance for per-invocation use; less prompt budget impact. Confirm.
3. **Doctrine bundle file list — ordered alphabetically or in `BootstrapLoader` insertion order?** The hash needs to be deterministic; alphabetical is safer. `BootstrapLoader` insertion order is documented but not enforced byte-stably.
4. **Per-dispatch verification — do we read storage on every COGNITION dispatch or cache for the turn lifecycle?** The turn lifecycle holds `CONVERSATION` for the duration; reading storage once at turn start is sufficient and avoids hot-path I/O. Lean: turn-lifecycle scope, with a flag the dispatcher exposes for tests that need stricter behavior.
5. **`SourceRegistration.constitution_injection="partial"` — what does it mean structurally?** Partial = constitution injected but bundle not; or vice-versa; or sampled. Lean: drop partial; keep it binary (`full`/`none`) until we have a real use case for partial.

## Risks

| Risk | Mitigation |
|---|---|
| Per-dispatch verification adds latency to every COGNITION dispatch | Turn-lifecycle scope (Q4); hash-and-compare is microseconds vs storage retrieval (already happens) |
| Echo-canary breaks legitimate sources whose models won't structured-respond | Per-source `require_constitution_echo=False` opt-out with documented justification; warning at registration time |
| `SourceRegistration` field additions break existing source registrations | Defaults are backwards-compatible (`require_constitution_echo=True` matches current implicit "constitution is in system prompt"; `constitution_injection="none"` matches current ARTIFACT reality where most have nothing) |
| Bundle hash includes a file that hot-changes for legitimate reasons (e.g., SOUL.md edits) | Re-anchor flow analogous to `reanchor_constitution(expected_hash)` for the bundle, or treat SOUL.md re-anchor as part of the same protocol |
| Talon-parity refactor leaves Talon temporarily broken | Phase 3 lands as a single PR with the shared primitive + Talon migration in one commit; tests prove parity before merge |

## Related

- #1131 kestrel-feature-workflows (consumer; uses `constitution_echo_verified` workflow gate as defense-in-depth at workflow boundary even before #1137 lands; full benefit when #1137 ships)
- #910 Signal Dispatcher epic (foundation; this epic adds Concern #13 to that design)
- #376 Agent Lifecycle Hardening (shares constitution-attestation primitives; coordinate `_get_governing_constitution` extraction)
- `agent/constitution.py:ConstitutionMixin` — periodic audit (existing)
- `agent/context_builder.py:build_system_prompt` — system-prompt assembly (existing)
- `kestrel-talon/governance/constitution.py` — standalone Talon loader (replaced in Phase 3)

## References

- `docs/architecture/SIGNAL_DISPATCHER.md` — canonical dispatcher reference
- `docs/architecture/SIGNAL_SOURCES_GUIDE.md` — registration walkthrough
- `kestrel_sovereign/agent/constitution.py` — current constitution integrity audit
- `kestrel_sovereign/agent/context_builder.py:build_system_prompt` — current system-prompt assembly
- `kestrel_sovereign/signals/store.py:SignalLogStore` — `signal_log` write path
- `kestrel_sdk/signals/models.py` — `SourceRegistration`, `SignalResult`
