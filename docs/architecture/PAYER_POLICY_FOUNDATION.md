# Payer Policy Foundation

**Status:** Draft for codex review
**Branch:** `feat/payer-policy`
**Last updated:** 2026-05-09

## Problem

Each Kestrel agent should be able to "pay its own way" for the metered
resources it consumes — primarily LLM inference and IPFS storage today,
the broader long tail of metered services tomorrow. Today the foundation
only supports a single global `OPENROUTER_API_KEY` and `LIGHTHOUSE_API_KEY`
per host. Per-agent provisioning machinery exists for OpenRouter
([`kestrel_sovereign/features/llm_keys/openrouter_provisioning.py`][orp])
but is not invoked at inception, and Lighthouse has no per-agent path at
all even though [`LighthouseProvider`][lhp] already accepts an unused
`key_resolver` parameter.

The bigger gap: there is no agent-level abstraction for *who pays for
what*. Standalone Kestrel users, multi-tenant apps like Frinz that pay on
behalf of users, sponsorship patterns (a family member funding a relative
agent), and truly self-sovereign agents (their own wallet pays vendors via
x402) all need different funding shapes today and have to bolt them on
ad hoc.

## Goals

1. Introduce a single foundation primitive — `PayerPolicy` — that names
   *who pays* for each metered resource class on a per-agent basis.
2. Wire the existing per-agent OpenRouter provisioning, the existing
   Lighthouse `key_resolver` plumbing, and the existing Stripe crypto
   on-ramp through this primitive instead of inventing parallel
   resolution logic.
3. Make `host_env` (today's behavior) a first-class policy choice that
   never disappears, so deployments can opt out of per-agent vendor
   accounts entirely.
4. Provide a setup-wizard step that captures the operator's intent
   without requiring them to read code.
5. Keep the foundation foundation: no fiat handling, no PCI scope, no
   vendor-specific code in the SDK.

## Non-goals

- Implementing x402 self-pay for LLM inference. (Watch-and-wait; see
  Risks.)
- Migrating Frinz's vending-machine code to the new primitive. (Frinz
  can adopt it later in its own repo. The foundation primitive must be
  designed to admit Frinz's patterns, not to replace them.)
- Building an in-house billing or invoicing system.
- Touching credit card data. Ever.

## Verified current state (origin/main @ 7d5d644e)

| Capability | Location | State |
|---|---|---|
| OpenRouter per-agent provisioning service | [`features/llm_keys/openrouter_provisioning.py`][orp] | Built, tested, NOT called from inception ([`inception_service.py:500-501`][inc-comment] commented out) |
| Retirement guard for OpenRouter key | [`retirement_service.py:197-202`][ret] | Already defensive (`if openrouter_key_hash:`) |
| Encrypted vendor key store, agent-DID-scoped | [`security/service_key_storage.py`][sks] | `lighthouse`, `openrouter`, `anthropic`, `openai`, `github`, `runpod`, `vastai` listed as supported providers |
| Encrypted private-key store for wallets | [`features/wallet/filecoin_keys.py`][skstor] | Used by wallet feature; encrypts with `KESTREL_DATA_KEY` |
| Multi-currency agent wallets | [`features/wallet/wallet_feature.py`][wf] | FIL, USDC, USDT on FEVM/Ethereum/Polygon |
| Stripe **on-ramp** (fiat → crypto, into agent wallet) | [`features/wallet/onramp/stripe_onramp.py`][onramp] | Built; card never enters Kestrel |
| Lighthouse provider with unused key resolver hook | [`storage/providers/lighthouse_provider.py:72-82`][lhp] | Constructor accepts `key_resolver`; `_get_api_key()` delegates to it; never wired |
| Setup wizard with step modules | [`kestrel_sovereign/setup/`][setup] | Steps: agent, emancipation, integrations, keys, llm, talon, verify |
| Emma-named scripts (audited individually) | `scripts/provision_emma_openrouter.py`, `scripts/rotate_emma_key.py`, `scripts/emma_scheduler.py` | First two have real functionality with bad Emma-defaulted names (provision-and-persist-key, AEAD passphrase rotation) — generalize. Third is one-off Emma scaffolding (born in initial commit, never modified, never invoked anywhere) — delete. Phase 0 handles all three. |
| `kestrel-sovereign-sdk` companion repo | `/Volumes/data2/projects/kestrel-sovereign-sdk` | Clean package layout; active SDK work on `feat/1094-sdk-database` |

## Prior art (treat as inspiration, NOT canonical)

Two prior local-only design notes informed this plan but are NOT on
`origin/main` — they live as untracked files in the operator's main
checkout:

- A Frinz vending-machine funding layer (multi-source funding from
  user / family / sponsor / institutional, wallet with `available_usd`,
  usage events, billing modes off/warn/soft/enforce). Implemented in
  the Frinz repo, not the foundation.
- A standalone crypto-native LLM proxy product spec (x402 + USDC
  prepaid balance, OpenRouter backend, per-agent keys via Management
  API). The implementation paths it references are under `frinz/` and
  several features it claims complete are not present in the current
  `kestrel-sovereign` checkout.

**These are dated December 2025 and are out of date with respect to
the current foundation.** Their funding-source taxonomy and per-resource
metering shape are useful inspiration. Their code claims must be
re-verified against current code before any of it is lifted into the
foundation, which this PR does not attempt.

## Design

### `PayerPolicy` primitive (in SDK)

```python
# kestrel_sdk/payer_policy.py

class PayerKind(StrEnum):
    NONE                     = "none"                       # explicit: do not use this resource
    HOST_ENV                 = "host_env"                   # legacy single-key env var
    HOST_MASTER_PROVISIONED  = "host_master_provisioned"    # platform pays; child key per agent
    SELF_WALLET              = "self_wallet"                # agent's own wallet pays vendors
    SPONSOR                  = "sponsor"                    # third party pays; specifies sponsor DID

@dataclass(frozen=True)
class PayerSpec:
    vendor: str                            # "lighthouse" | "openrouter" | "local-disk" | "local-llm" | ...
    kind: PayerKind
    sponsor_did: Optional[str] = None
    monthly_cap_usd: Optional[Decimal] = None  # advisory; resolver may enforce per-vendor

@dataclass(frozen=True)
class PayerPolicy:
    llm:     PayerSpec
    storage: PayerSpec
    compute: PayerSpec       # GPU rentals etc.
    tools:   PayerSpec       # Tavily, Exa, ElevenLabs, ...
    comms:   PayerSpec       # Twilio, Resend, ...

    @classmethod
    def host_env_default(cls) -> "PayerPolicy": ...  # preserves today's behavior
```

`PayerSpec.vendor` is the second dimension of the support matrix —
`(resource_class, vendor, kind)` is what the matrix is keyed by, and it
is also what the resolver needs to decide which vendor-specific
side-effect (if any) to run before returning a `KeyResolutionService`.
Multi-vendor-per-class within a single agent is out of scope for v1; if
an operator needs e.g. both Lighthouse and a local-disk fallback, that
is modeled as a runtime fallback inside the storage provider, not as
two parallel policy slots.

`PayerSpec` is the only thing external features (Frinz, Talon, etc.)
depend on. The resolver implementation lives in the main repo.

### `PayerResolver` (in main repo)

`PayerResolver` is the *intent* layer that sits ABOVE the existing
[`KeyResolutionService.resolve_key(provider, require=False)`][kr]
contract — it does NOT replace it. The existing
`KeyResolutionService` keeps its current shape: agent-scoped lookup in
`ServiceKeyStorage`, fallback to env var, returns a `str | None`.
`PayerResolver` decides *which* `KeyResolutionService` instance (and
which provisioning side-effects) get applied for an agent before that
service is handed to a provider.

```python
class ResolvedResource:
    enabled: bool                           # False iff policy is NONE for this slot
    key_resolver: Optional[KeyResolutionService]   # only when enabled

class PayerResolver(Protocol):
    async def resolve_for(
        self,
        agent_did: str,
        resource_class: ResourceClass,  # llm | storage | compute | tools | comms
    ) -> ResolvedResource
```

The concrete implementation, called at agent-init time:

1. Reads the agent's `PayerPolicy` (from agent metadata; defaults to
   `host_env_default` if missing for back-compat).
2. Looks up the appropriate `PayerSpec` for the resource class. The
   spec carries both `vendor` and `kind`, which together index the
   support matrix.
3. Returns a `ResolvedResource`:
   - `NONE` → `ResolvedResource(enabled=False, key_resolver=None)`. The
     **agent-init layer must treat this as "do not construct the
     provider for this resource at all."** This is critical: returning
     a sentinel `KeyResolutionService` is not enough, because providers
     like `LighthouseProvider` accept an explicit `api_key` constructor
     argument that today defaults to `os.environ["LIGHTHOUSE_API_KEY"]`
     and is consulted as a fallback when the resolver returns `None`.
     Honoring `NONE` therefore means *skipping provider construction*,
     not relying on the resolver to neutralize the provider.
   - `HOST_ENV` → wraps today's `KeyResolutionService` instance and
     returns it. This is the back-compat path.
   - `HOST_MASTER_PROVISIONED` → invokes the vendor's provisioning API
     once for this `(agent_did, vendor)` pair (idempotent — skip if a
     child credential is already minted for this agent), stores the
     result in agent's `ServiceKeyStorage` under `provider=vendor`,
     then returns a normal `KeyResolutionService` that will find it
     there on subsequent calls.
   - `SELF_WALLET` → mints (idempotent) a wallet-signed credential
     using the agent's existing wallet keypair, stores in agent's
     `ServiceKeyStorage` under `provider=vendor`, returns a normal
     `KeyResolutionService`. For Lighthouse, this is the documented
     sign-message → API-key flow. For LLM, this branch raises an
     explicit `NotImplementedError` until x402-native LLM matures
     (see Risks).
   - `SPONSOR` → identical mechanics to `HOST_MASTER_PROVISIONED`
     except the master credential comes from the sponsor's
     `HostKeyStorage`, not the operator's.

Crucially, providers like `LighthouseProvider` keep their existing
`key_resolver: Optional[KeyResolutionService]` parameter unchanged.
The wiring change is at one site (`kestrel_agent.py:initialize()`):

```python
storage = await payer_resolver.resolve_for(agent_did, "storage")
if storage.enabled:
    lighthouse = LighthouseProvider(
        api_key=None,                       # explicit: do not seed from env at construction
        key_resolver=storage.key_resolver,  # resolver is the only credential source
    )
else:
    lighthouse = None  # storage explicitly disabled by policy
```

`api_key=None` paired with a non-None `key_resolver` is the contract:
the provider must consult the resolver and not fall back to its own
env-var path. Phase 3 includes a small refactor to
`LighthouseProvider.__init__` so this contract is unambiguous (today's
`api_key or os.environ.get(...)` pattern is replaced with explicit
"if api_key is None and key_resolver is None: read env; else: use what
was given").

### Storage tiers

Reaffirmed from prior conversation:

| Tier | Contents | Store | Encryption |
|---|---|---|---|
| Wallet private keys | secp256k1, Ed25519 | [`SecureKeyStorage`][skstor] (existing) | `KESTREL_DATA_KEY` |
| Vendor API keys (spend-granting) | `sk-or-v1-…` | [`ServiceKeyStorage`][sks] (existing) | agent-DID-derived AES |
| Vendor account refs (opaque pointers) | `cus_xxx`, `pm_xxx`, OnRampSession IDs | `ServiceKeyStorage` with new `kind="account_ref"` discriminator | same |
| Host/treasury credentials | master OpenRouter key, host Lighthouse key, sponsor master keys | **new** `HostKeyStorage` (mirrors `ServiceKeyStorage` but operator-scoped, not agent-scoped) | host identity-derived |
| Raw card / bank / CVV / SSN | n/a | **NEVER STORED** | n/a |

`HostKeyStorage` is the only new storage primitive. Everything else
reuses what exists.

### Support matrix (single source of truth)

The wizard, the resolver, and the verify step all read this matrix.
Combinations not listed here are NOT offered to the operator:

| Resource | Vendor | `host_env` | `host_master_provisioned` | `self_wallet` | `sponsor` | `none` |
|---|---|---|---|---|---|---|
| llm | openrouter | ✅ | ✅ | ❌ (deferred — see Risks #4) | ✅ | ✅ |
| llm | local (ollama/llama.cpp) | ✅ | n/a | n/a | n/a | ✅ |
| storage | lighthouse | ✅ | ⏳ Phase 3.5 | ⏳ Phase 3.5 | ⏳ Phase 3.5 | ✅ |
| storage | local-disk | ✅ | n/a | n/a | n/a | ✅ |
| compute | (any) | ✅ | ❌ (out of scope this PR) | ❌ | ❌ | ✅ |
| tools | (any) | ✅ | ❌ (out of scope this PR) | ❌ | ❌ | ✅ |
| comms | (any) | ✅ | ❌ (out of scope this PR) | ❌ | ❌ | ✅ |

`✅` = implemented and verifiable in this PR.
`⏳` = scaffolded as `NotImplementedError` in Phase 3, filled in Phase
3.5 of this PR (small follow-up phase before wizard).
`❌ (deferred)` = `PayerKind` enum value exists, resolver raises
`NotImplementedError`, wizard refuses to offer it.
`❌ (out of scope this PR)` = same as deferred but tracked as future
work.
`n/a` = not meaningful (e.g. `host_master_provisioned` for a local
model has no master to provision under).

This table is encoded as a Python constant
`SUPPORTED_PAYER_COMBINATIONS` in the SDK so it can be imported by both
the wizard and the resolver. Tests assert that the matrix and the
resolver implementations agree (no resolver path that the matrix says
is `❌` may succeed; no `✅` path may raise `NotImplementedError`).

### Setup wizard step

A new `payments` step at
`kestrel_sovereign/setup/steps/payments.py`, ordered after `keys` and
before `verify`. The step:

1. Reads existing `PayerPolicy` from kestrel.toml if present.
2. For each resource class in {llm, storage, compute, tools, comms},
   asks the operator: *"how should agents pay for this?"* — the choices
   shown are filtered through `SUPPORTED_PAYER_COMBINATIONS` so an
   operator never picks a path that the resolver cannot honor. Always
   includes "ask me later (defaults to host_env)" as an escape hatch.
3. For `HOST_MASTER_PROVISIONED`, prompts for the master credential and
   stores it via `HostKeyStorage`. Card details NEVER prompted; users
   are linked to the vendor's own dashboard for card-on-file setup.
4. Writes the resolved `PayerPolicy` to kestrel.toml under a new
   `[payments]` table.
5. Runs an explicit `verify` substep: for every `✅` slot in the
   policy, resolve a credential and confirm the vendor returns a 2xx
   on a no-op call. Slots set to `none` are skipped, not failed.

`--reset` semantics inherit from the wizard's contract: move
kestrel.toml aside, regenerate from scratch.

### Per-vendor wiring

- **OpenRouter:** the existing `OpenRouterProvisioningService` becomes
  the `HOST_MASTER_PROVISIONED` impl. The commented-out call in
  `inception_service.py:500-501` is replaced by a lazy provisioning
  step inside `PayerResolver.resolve()` — never at inception, only on
  first need. Retirement service already guards on
  `openrouter_key_hash`; no change needed.
- **Lighthouse:** Two changes. (1) Honoring `NONE`: if
  `PayerResolver.resolve_for(agent_did, "storage")` returns
  `enabled=False`, the agent-init layer skips constructing
  `LighthouseProvider` and `LighthouseTarget` entirely. Storage features
  that consume them already handle their absence (today's missing
  `LIGHTHOUSE_API_KEY` exercises the same code path). (2) When the
  resolver IS in charge (any kind other than `NONE`), the provider is
  constructed with `api_key=None` plus the resolver from
  `ResolvedResource`; a tiny refactor to `LighthouseProvider.__init__`
  ensures that `api_key=None` paired with a non-None resolver does NOT
  silently fall back to `os.environ["LIGHTHOUSE_API_KEY"]` at
  construction time. The provider's `_get_api_key()` continues to call
  `resolve_key("lighthouse", require=False)` exactly as it does today.
  `LighthouseTarget` gets the same treatment. Phase 3 ships
  `HOST_ENV`; Phase 3.5 ships `HOST_MASTER_PROVISIONED` and
  `SELF_WALLET` for storage (the wallet-signed API key flow).
- **Stripe on-ramp:** unchanged. It already serves the
  fiat→crypto→agent-wallet flow that backs `SELF_WALLET`. The wizard
  step's "I want to fund this agent" branch points at the existing
  on-ramp endpoint.

## Phases

Each phase is a separately-reviewable chunk. Codex CLI review runs at
the end of each before moving to the next. Per round velocity feedback
in memory: cap at 4-5 codex rounds per phase before merging the phase
internally.

### Phase 0 — Cleanup (~30 min)

Each Emma-named script is audited individually because the verdict is
not uniform. Originally drafted as "delete the redundant wrappers,"
but inspection revealed that `provision_emma_openrouter.py` did MORE
than `manage_openrouter_keys.py` (it also persisted the key into
`ServiceKeyStorage` and updated `openrouter_key_hash`), so it deserved
a rename, not a deletion. Same shape as `rotate_emma_key.py`. Codex
caught this during Phase 0 review; the correction is reflected here.

- `scripts/provision_emma_openrouter.py` → **rename to
  `provision_agent_openrouter.py`** and parameterize. Distinct from
  `manage_openrouter_keys.py provision`: that command creates and
  prints a key; this one ALSO stores the key in the agent's
  `ServiceKeyStorage` (encrypted with agent-derived key) and updates
  the agent's `openrouter_key_hash` metadata. Becomes generic via
  `--db`, `--name`, `--limit-usd`, `--limit-reset`.
- `scripts/rotate_emma_key.py` → **rename to `rotate_agent_key.py`**.
  Functionally generic AEAD passphrase rotation (argparse with
  `--db`, `--old-pass`, `--new-pass`, `--dry-run`). Rename drops the
  Emma-defaulted `--db` — it's now required — and refreshes the
  docstring/help text. References updated in
  [`docs/architecture/security/CRYPTO_INVENTORY.md`][crypto-inv] and
  [`tests/integration/test_quantum_hardening_e2e.py`][qh-test].
- `scripts/emma_scheduler.py` → **delete**. 355 lines hardcoding
  `EmmaScheduler` class and `kestrel_prime.db` defaults; born in the
  initial commit, never modified since, never invoked anywhere in the
  repo (no cron, systemd, Docker, service config). One-off scaffolding
  that was never wired up.
- `docs/audit/REPO_MAP.md` mentions all three scripts but is
  auto-regenerated by the nightly chore commit; no hand-edit.
- No design changes; pure cleanup.

**Codex review focus:** any caller I missed; whether the renamed
provision script preserves the full
provision-then-persist-then-update-metadata contract that the original
exercised.

### Phase 1 — `PayerPolicy` schema in SDK (~half day)

- Land `kestrel_sdk/payer_policy.py` in `kestrel-sovereign-sdk` repo
  with the dataclasses above and a `validate()` method.
- Pure types. No IO. No vendor knowledge.
- Add round-trip TOML serialization helpers (the wizard will use them).
- Tests: schema validation, round-trip serialization, defaulting.

**Codex review focus:** schema completeness, serialization stability,
forward-compat for new resource classes and new payer kinds.

### Phase 2 — `HostKeyStorage` + `PayerResolver` (~1 day)

- Add `kestrel_sovereign/security/host_key_storage.py` mirroring
  `ServiceKeyStorage` but keyed to a host identity rather than agent
  DID. Reuse the same encrypt/decrypt code path.
- Add `kestrel_sovereign/services/payer_resolver.py` implementing the
  `PayerResolver` protocol from the SDK against `ServiceKeyStorage` +
  `HostKeyStorage` + `os.environ`.
- Implement `HOST_ENV` and `NONE` first; stub the others to raise
  `NotImplementedError` so the surface is in place but the dependent
  flows are explicit about what they need.
- Tests: per-`PayerKind` resolution, agent isolation (one agent's
  child key not visible to another), missing-credential paths.

**Codex review focus:** key-isolation guarantees; whether
`HostKeyStorage` should genuinely be a separate file or an additional
mode of `ServiceKeyStorage`.

### Phase 3 — Wire OpenRouter and Lighthouse through the resolver (`HOST_ENV` + `HOST_MASTER_PROVISIONED`) (~1 day)

Storage and LLM have different init shapes today and the wiring respects
that. Both go through `PayerResolver.resolve_for(...)` but apply the
result differently.

#### Storage path (per-agent provider construction)

`LighthouseProvider` and `LighthouseTarget` are constructed per agent
in `kestrel_agent.py:initialize()`. Wiring:

```python
storage = await payer_resolver.resolve_for(agent_did, "storage")
if storage.enabled:
    lighthouse = LighthouseProvider(
        api_key=None,                       # no env-var seeding
        key_resolver=storage.key_resolver,  # only credential source
    )
else:
    lighthouse = None  # NONE policy: provider not constructed at all
```

Small refactor to `LighthouseProvider.__init__`: when both `api_key`
and `key_resolver` are `None`, fall back to `os.environ` (today's
behavior, kept for non-policy callers); otherwise use whichever was
supplied verbatim. No silent env-var bleed-through when the resolver is
in charge.

#### LLM path (per-agent service, per-agent key swap)

**Required invariant: one `LLMService` instance per agent.** Production
code already follows this de facto today —
[`multi_agent/agent_manager.py:90-91`][agm] explicitly constructs a
fresh `LLMService` per agent ("Each agent gets its own LLMService
(mutable model state)"). However, this invariant is currently NOT
enforced: [`kestrel_agent.py:292`][ka292] accepts any injected
`llm_service`, and at least one test
([`tests/integration/test_constitution_adversarial.py:232-247`][adv-test])
shares a single instance across two `KestrelAgent` constructions. The
plan REQUIRES this invariant because `LLMService.use_agent_key` mutates
`self.providers` in-place; without enforcement, the last agent loaded
would silently steal the OpenRouter client of every other agent
sharing the instance.

Phase 2 promotes the invariant from convention to enforcement:

1. Add `LLMService.attach_to_agent(agent_did: str)` which records the
   owning agent on first call and raises `LLMServiceAlreadyAttachedError`
   on any subsequent attach with a different DID. The method is
   idempotent for the same DID.
2. Modify `KestrelAgent.__init__` to call `attach_to_agent(self.did)`
   immediately after assigning `self.llm_service`. Construction fails
   loudly if a second agent tries to claim the same instance.
3. Migrate `test_constitution_adversarial.py:232-247` to construct
   distinct `LLMService` instances per agent. The test was inadvertently
   testing a non-invariant; fixing it is part of this PR.

Phase 3 then adds a regression test asserting that two `KestrelAgent`
instances in the same process get distinct `LLMService` instances and
that any attempt to share raises at agent construction.

`LLMService.__init__` and `ProviderRegistry` are NOT modified. Per-agent
customization continues to happen after agent load via the existing
[`LLMService.use_agent_key(agent_did, db, provider)`][use_agent_key]
mechanism. The current direct call at
[`kestrel_agent.py:806-820`][use-agent-key-call] (which keys off the
deprecated `openrouter_key_hash` agent metadata field) is replaced with
a policy-aware version. Because the agent's `LLMService` instance is
already its own, no `agent_did` parameter is needed for the disable
path — the flag lives on the instance:

```python
llm = await payer_resolver.resolve_for(agent_did, "llm")
match (llm.enabled, policy.llm.kind, policy.llm.vendor):
    case (False, _, _):
        # NONE: this agent's LLMService refuses every generation call.
        self.llm_service.disabled = True
    case (True, "host_env", _):
        # Shared-host key path; nothing to swap. Policy chose this vendor
        # but is paying via the host's env-var-configured key.
        pass
    case (True, "host_master_provisioned", vendor):
        # Resolver already minted the child key in agent's ServiceKeyStorage.
        # Swap this agent's LLMService client for the agent-specific key on
        # every route belonging to this vendor.
        await self.llm_service.use_agent_key(agent_did, db, provider=vendor)
    case (True, "sponsor", vendor):
        # Same as host_master but the resolver used sponsor's HostKeyStorage.
        await self.llm_service.use_agent_key(agent_did, db, provider=vendor)
    case (True, "self_wallet", _):
        raise NotImplementedError("self_wallet for llm deferred")
```

`LLMService.disabled` is a new instance attribute (default `False`).
Phase 3 adds a single private guard `_check_policy()` that raises
`PolicyDeniedError` when `self.disabled` is `True`, and inserts a call
to it at the top of EVERY public method on `LLMService` that hits a
provider client. Today's exhaustive list (verified by grep against
[`kestrel_sovereign/llm/service.py`][llm-svc]):

- `get_audit_response`
- `get_response`
- `get_response_with_model`
- `generate`
- `generate_with_messages`
- any streaming variants present in the codebase at implementation
  time (Phase 3 must enumerate these by inspection, not from this
  list, since streaming has been actively refactored)

Tests in Phase 3 walk the `LLMService` class (and its mixins —
`StreamingMixin` and friends, where streaming entry points actually
live) via reflection at runtime. The collection predicate is:

```
(inspect.iscoroutinefunction(m) OR inspect.isasyncgenfunction(m))
AND name matches a generation pattern
```

Generation patterns: name starts with `generate`, `get_response`,
`get_audit_response`, `stream_`, or contains `_streaming_response`. The
name filter is critical — without it the sweep would also collect
non-generation methods (`discover_all_models`, `get_storage_info`,
`pull_model`, `use_agent_key`, `close`) that are management or
discovery, not policy-relevant. Async-generator coverage is critical
too: the streaming entry points (`generate_stream`,
`get_streaming_response`, `stream_with_messages`,
`stream_with_tool_detection`) are async generators, so a sweep that
only checks `iscoroutinefunction` would silently miss them and leave
`llm = none` streamable through provider clients.

The test then asserts each collected entry point calls
`_check_policy()` at least once on the disabled path. This catches
future entry points that forget the guard, in either the coroutine or
async-generator form. The generation-pattern list is itself part of
the plan and updates require a same-PR change to the test (so adding
a new entry point under a new naming convention is a deliberate, not
accidental, choice). Callers (chat endpoints, reflection loops) treat
the `PolicyDeniedError` the same way they treat "no key configured"
today. The flag is in-memory per instance; persistence is unnecessary
because the policy in kestrel.toml is the source of truth and is
re-read at each agent init.

The `OpenRouterProvisioningService.create_agent_key` side-effect runs
inside `PayerResolver.resolve_for(agent_did, "llm")` when the spec is
`HOST_MASTER_PROVISIONED` — idempotent, skipped if a child key already
exists in `ServiceKeyStorage`. The standalone
`manage_openrouter_keys.py` script still calls the provisioning
service directly, unchanged.

#### Common

- Inception service stays simple: it does NOT provision vendor keys.
  First-use lazy provisioning is the contract.
- The deprecated `openrouter_key_hash` agent metadata field is left in
  place for one release for back-compat; the policy is the new source
  of truth. Removal is tracked separately, not in this PR.
- Tests:
  - per-agent OpenRouter key swap against a mock REST + a mock
    LLMService.use_agent_key.
  - existing Lighthouse env-var path still works (back-compat).
  - cold-start restore regression test still passes.
  - `storage = none` actually disables Lighthouse on a host that has
    `LIGHTHOUSE_API_KEY` set (regression for the round-2 finding).
  - `llm = none` causes every LLM generation entry point on the
    agent's LLMService instance to raise `PolicyDeniedError`, not
    silently fall through to the shared key.
  - Two `KestrelAgent` instances in the same process get distinct
    `LLMService` instances (asserts the per-agent invariant).

**Codex review focus:** the storage and LLM wiring contracts are
clearly distinct and neither bleeds into the other; the lazy-
provisioning contract is observable and idempotent under retry;
cold-start restore from Lighthouse still works; no regression in
today's `LIGHTHOUSE_API_KEY` env-var behavior; `NONE` policy actually
disables on both surfaces.

### Phase 3.5 — Lighthouse `HOST_MASTER_PROVISIONED` and `SELF_WALLET` (~half day)

Separated from Phase 3 so wizard work in Phase 4 does not advertise
unimplemented Lighthouse paths. After this phase the Lighthouse row of
the support matrix is fully `✅` (host_env) and `✅` (host_master,
self_wallet) — wizard can offer all three honestly.

- Implement the Lighthouse wallet-signed API key flow in
  `PayerResolver` for `SELF_WALLET`: GET
  `/api/auth/get_message?publicKey=<...>`, sign with the agent's
  existing secp256k1 wallet keypair, POST `/api/auth/create_api_key`,
  store the result in `ServiceKeyStorage`. Idempotent.
- Implement `HOST_MASTER_PROVISIONED` for Lighthouse: same wallet-signed
  flow but using the host's master Lighthouse wallet from
  `HostKeyStorage`. Each agent gets its own scoped child key.
- Tests gated behind a burner-wallet env-var: real round-trip against
  Lighthouse's auth API to confirm the documented protocol matches
  reality before we ship.

**Codex review focus:** the live test against Lighthouse passed;
`SELF_WALLET` cannot accidentally use the host's master wallet (and
vice versa); resolver caching does not leak credentials across agents.

### Phase 4 — Setup wizard `payments` step (~1 day)

- `kestrel_sovereign/setup/steps/payments.py` implementing the wizard
  flow above.
- Wired into `kestrel_sovereign/setup/steps/__init__.py` after `keys`
  and before `verify`.
- A `kestrel setup verify payments` substep mirrors the
  `cli_verify_install.py` pattern: resolve each declared payer and
  hit a no-op vendor endpoint to confirm the credential is live.
- Tests:
  - Idempotent re-run on a complete policy.
  - `--dry-run` previews without writing.
  - All six funding patterns
    (standalone / platform-pays / user-pays / sponsor-pays / self-pays /
    none-configured) round-trip through the wizard correctly.
  - `--reset` cleanly replaces an existing `[payments]` table.
  - Hand-edited kestrel.toml with malformed `[payments]` is rejected
    with a remediation hint, not silently coerced.

**Codex review focus:** the test matrix above is the primary review
target. Wizard correctness is the most expensive bug class to ship.

### Phase 5 — Documentation (~half day)

- Update [`docs/architecture/PROVIDER_ECONOMICS.md`][pe] with:
  - The `PayerPolicy` model and the six funding patterns.
  - The storage-tier table.
  - An explicit "Kestrel never stores credit cards" section, with the
    Stripe-on-ramp pattern shown as the canonical fiat path.
  - Frinz / Talon / standalone deployment recipes (for each, the exact
    `PayerPolicy` they declare).
- Update [`docs/diagrams/data-architecture/DA-06-filecoin-lighthouse.md`][da06]
  to point at the new resolver path instead of the global env var.
- Two prior local-only design notes (Frinz vending machine and the
  standalone LLM proxy product, both dated December 2025) inspired
  parts of this design but live as untracked files in the operator's
  main checkout, not on `origin/main`. Phase 5 does NOT promise
  modifications to those docs from this branch — if the operator wants
  them tracked, they get added in a separate change. The non-goals
  section below records this explicitly so a future reader does not go
  hunting for files that were never committed.

**Codex review focus:** docs match code; nothing is over-promised
(especially around `SELF_WALLET` for LLM, which we explicitly defer).

### Phase 6 — Single PR

After all five implementation phases pass codex review locally, push
the branch and open one PR. CI runs once at PR open, not per phase.

## Test strategy summary

| Phase | Unit | Integration | Notes |
|---|---|---|---|
| 0 | n/a | smoke: `kestrel --help` still works | cleanup only |
| 1 | yes (SDK tests) | n/a | pure types; matrix consistency test |
| 2 | yes | mock vendor APIs | key isolation is the load-bearing assertion; matrix-vs-resolver consistency test |
| 3 | yes | mock OpenRouter `/keys`; existing Lighthouse env-var path | cold-start restore regression test; today's behavior preserved |
| 3.5 | yes | gated real round-trip against Lighthouse `/api/auth/*` with a burner wallet | confirms the documented sign-message → API-key protocol matches reality |
| 4 | yes | wizard table-driven across the support matrix | hardest test surface |
| 5 | n/a | docs lint | n/a |

Real-credential tests (Lighthouse signed-message flow against a burner
wallet, OpenRouter `/keys` against the management key) gated behind
env-var presence. CI does not need them; local pre-PR run does.

## Risks and open questions

1. **`HostKeyStorage` vs. extending `ServiceKeyStorage`.** The
   foundation may be cleaner if `ServiceKeyStorage` simply admits a
   non-agent-DID owner ID. Codex review of Phase 2 should weigh in.
2. **Lighthouse wallet-signed API key flow** is mature externally but
   unverified in our codebase. Phase 3.5 must include a one-shot live
   test against a burner wallet to confirm the protocol behaves as
   documented before the wizard offers `SELF_WALLET` for storage.
3. **OpenRouter `POST /api/v1/keys` rate limit** on mass inception. If
   we ever spawn N agents in a burst, we need a backoff queue. The
   provisioning service has 3-retry exponential backoff; whether it's
   sufficient depends on actual rate-limit numbers we don't have yet.
4. **x402 deferral.** We're explicitly NOT shipping `SELF_WALLET` for
   LLM in this scope. Coinbase x402 facilitator covers Base/Polygon/
   Arbitrum/World/Solana ERC-20s and Stripe added x402 in Feb 2026,
   but no mature general-LLM endpoint exposes x402 yet. We add the
   `PayerKind.SELF_WALLET` enum value but the resolver returns
   `NotImplementedError` for `(llm, self_wallet)` until that landscape
   matures. We DO ship `SELF_WALLET` for Lighthouse storage — the
   wallet-signed key flow is real today.
5. **Frinz migration is out of scope** for this PR. Once the SDK
   primitive lands, Frinz can adopt it in a follow-up. The risk is
   shipping a foundation primitive that doesn't admit Frinz's actual
   funding shapes; mitigation is that Phase 1's schema review
   explicitly checks against Frinz's user/family/sponsor/institutional
   taxonomy.
6. **Setup wizard reset semantics.** The wizard already has a careful
   `--reset` contract that moves files aside rather than deletes.
   Phase 4 must not violate it. Tests cover this.

## What this plan deliberately does not do

- Build a fiat payment processor. Stripe on-ramp already covers the
  card path; nothing else.
- Migrate Frinz vending-machine code into the foundation.
- Implement x402 self-pay for LLM.
- Add new vendors. Same surface as today (OpenRouter, Lighthouse, plus
  the on-ramp), better organized.
- Touch the multi-tenant `proxy.py` machinery
  (`kestrel_sovereign/multi_agent/proxy.py`). That's a different proxy
  (process-level, not LLM).

[orp]: ../../kestrel_sovereign/features/llm_keys/openrouter_provisioning.py
[lhp]: ../../kestrel_sovereign/storage/providers/lighthouse_provider.py
[inc-comment]: ../../kestrel_sovereign/inception_service.py
[ret]: ../../kestrel_sovereign/retirement_service.py
[sks]: ../../kestrel_sovereign/security/service_key_storage.py
[skstor]: ../../kestrel_sovereign/features/wallet/filecoin_keys.py
[wf]: ../../kestrel_sovereign/features/wallet/wallet_feature.py
[onramp]: ../../kestrel_sovereign/features/wallet/onramp/stripe_onramp.py
[setup]: ../../kestrel_sovereign/setup/
[crypto-inv]: security/CRYPTO_INVENTORY.md
[qh-test]: ../../tests/integration/test_quantum_hardening_e2e.py
[mgmt]: ../../scripts/manage_openrouter_keys.py
[kr]: ../../kestrel_sovereign/services/key_resolution.py
[use_agent_key]: ../../kestrel_sovereign/llm/service.py
[use-agent-key-call]: ../../kestrel_sovereign/kestrel_agent.py
[agm]: ../../kestrel_sovereign/multi_agent/agent_manager.py
[ka292]: ../../kestrel_sovereign/kestrel_agent.py
[adv-test]: ../../tests/integration/test_constitution_adversarial.py
[llm-svc]: ../../kestrel_sovereign/llm/service.py
[pe]: PROVIDER_ECONOMICS.md
[da06]: ../diagrams/data-architecture/DA-06-filecoin-lighthouse.md
