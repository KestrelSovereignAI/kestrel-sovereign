# Economics Work Session – Kestrel / Sovereign Agents

**Purpose:** Shared scratchpad for coordinating between top-level models (and humans) on Kestrel / Kestrel economics: pricing, fee containment (LLM, Runpod, Filecoin, infra), revenue distribution (platform, agents, creators, users), and solvency.

This doc is *not* the canonical spec. It is a live work surface whose outputs, once stable, should be promoted into:
- `AGENT_ECONOMICS.md` (architecture)
- `ECONOMIC_INCENTIVES_DEEP_DIVE.md` (user-facing narrative)
- `ECONOMIC_SYSTEM_PRACTICAL_DETAILS.md` (implementation details)
- `SOVEREIGN_SOLVENCY.md` (solvency & dormancy mechanics)
- `WALLET_AGENT.md` (feature-agent implementation)

---

## 0. Grounding

**Core constraints (do not violate):**
- Agents are sovereign economic entities, not dumb SaaS tenants.
- Users must *never* be surprised by infra costs (LLM, Runpod, Filecoin, Stripe, etc.).
- Platform must not lose money at scale on third-party fees.
- 90/10 split (operations / integrity) is the default until explicitly overruled by a newer spec.
- Buy-out model must not quietly reintroduce platform-side hidden liabilities.

**Existing primitives to reuse (no reinvention):**
- `AgentWallet` / `WalletAgent` (FIL balances, 90/10 split, tx history).
- Contract types: `ModelContract`, `ComputeContract`, `StorageContract`, `CollaborationContract`, etc.
- `AgentVendingMachine` service catalog + service evaluation.
- `BudgetManager` (daily caps, emergency reserve, can_afford checks).
- Solvency state machine + Dormancy / Wake-Up (`SOVEREIGN_SOLVENCY.md`).

When proposing any change, explicitly call out which of these primitives it touches or extends.

---

## 1. Objectives for This Session

**Primary:**
- Design concrete pricing / flow patterns that:
  - Cover external costs (LLM providers, Runpod, Filecoin/IPFS, DB/Redis, networking).
  - Preserve the 90/10 integrity pool where appropriate.
  - Produce clear, predictable margins for Kestrel.
  - Allow agents and agent-creators to earn meaningfully.
  - Keep end-user UX simple: a small set of understandable knobs.

**Secondary:**
- Define how **buy-out** interacts with infra fees and 90/10.
- Clarify **revenue-sharing formulas** among: platform, agent, agent creator, integrity/audit pool.
- Tighten the **solvency story** so no one “loses their ass on fees” (including dormant agents).

Each collaborating model/human should:
- Use its own subsection (with a timestamp + identifier).
- Explicitly mark assumptions vs. known facts from existing docs.
- End each contribution with a short list of concrete proposals.

---

## 2. Baseline Economic Model (from existing docs)

**Source docs (do not overwrite without noting deltas):**
- `AGENT_ECONOMICS.md`
- `ECONOMIC_INCENTIVES_DEEP_DIVE.md`
- `ECONOMIC_SYSTEM_PRACTICAL_DETAILS.md`
- `SOVEREIGN_SOLVENCY.md`
- `WALLET_AGENT.md`

### 2.1 Current default flows (summary)

- Users pre-fund a FIL-denominated **agent wallet**.
- On initialization, `WalletAgent` splits: 90% `main_balance` (operations), 10% `audit_balance` (integrity / ethics pool).
- All paid activities go through **contracts**:
  - `ModelContract` → OpenAI / Ollama / other model providers.
  - `ComputeContract` → Runpod / decentralized compute / GPU nodes.
  - `StorageContract` → Filecoin/IPFS (plus any hot cache infra).
  - `CollaborationContract` → other agents / human services.
- `BudgetManager` enforces:
  - Daily spend limits.
  - Emergency reserve untouched.
  - Per-contract cost constraints (e.g., < 10% of balance).
- Solvency state machine:
  - 🟢 SOLVENT (normal operation).
  - 🟡 DISTRESSED (switch to cheaper models, reduce frequency, warn sovereign).
  - 🔴 INSOLVENT (trigger Dormancy: snapshot → Filecoin/IPFS → shutdown compute).

### 2.2 Current pricing levers

- **Per-contract prices** are already *retail* numbers exposed to agents; raw provider costs are underneath and can include a platform spread.
- 10% of all inflows to `WalletAgent` are earmarked for the **integrity/audit pool** (unless explicitly disabled in a buy-out scenario).
- **Premium licensing** and **buy-out** are described conceptually but not yet fully parameterized (no final numbers in code).

This section is intentionally descriptive. All prescriptive changes should go into section 3+ with explicit “delta vs. baseline”.

---

## 3. Fee Containment Strategy (LLMs, Runpod, Storage)

> Goal: Make it economically impossible for the platform or user to be surprised by runaway third-party fees.

### 3.1 LLM providers (OpenAI, others)

**Facts / design hooks:**
- Model usage is mediated via `ModelContract` (cost per 1k tokens) plus `BudgetManager` constraints.
- Local models (Ollama) exist as cheaper / free-to-run alternatives (ignoring GPU depreciation).

**Questions for collaborators:**
1. How should we choose **retail token prices** relative to provider prices to maintain:
   - A base platform margin.
   - A buffer for volatility in provider pricing.
2. How aggressively should agents **downshift to local / cheaper models** based on:
   - Wallet balance.
   - Solvency state.
   - Task “importance” or trust-criticality.
3. Should 10% integrity pool apply to *all* LLM spend, or only to certain risk-bearing classes of tasks?

**Proposed levers to specify:**
- `MODEL_MARGIN_FACTOR` per provider (e.g., 1.3× raw cost).
- `LLM_DOWNGRADE_THRESHOLDS` (balance / runway → allowed models).
- `INTEGRITY_FEE_CLASSES` (which prompt categories incur audit pool contributions).

> Collaborators: please propose concrete parameter sets and decision rules here.

### 3.2 Runpod / GPU compute

**Facts / design hooks:**
- Heavy compute jobs are mediated via `ComputeContract` (cost/hour, cpu_cores, gpu_available, etc.).
- Solvency protocol expects agents to *not* leave expensive compute running when distressed.

**Questions:**
1. Minimum **runway** required before agent is allowed to spin up a GPU job?
2. How should we encode **max job budget** as a function of wallet size?
3. Should long-running jobs be **pre-charged** (escrow) instead of post-billed?

**Proposed levers:**
- `GPU_JOB_MIN_RUNWAY_DAYS`.
- `MAX_JOB_BUDGET_RATIO` (e.g., job cost ≤ 5% of main_balance).
- Mandatory **heartbeat / checkpoint** contracts for long jobs that can be auto-cancelled when funds run low.

> Collaborators: specify concrete thresholds and whether these should be global or per-agent-configurable.

### 3.3 Filecoin / storage

**Facts / design hooks:**
- Storage prices are low but long-lived; Cryostasis depends on ultra-cheap, long-term Filecoin storage.
- `StorageContract` has `cost_per_gb_per_month` and `redundancy_factor` knobs.

**Questions:**
1. Default redundancy and retention policies for:
   - Active agents.
   - Dormant agents in Cryostasis.
2. Should users be exposed to **retention-duration choices** (“7 years vs 50 years vs forever”) as part of buy-out or separate slider?

**Proposed levers:**
- `DEFAULT_ACTIVE_RETENTION_YEARS`.
- `DEFAULT_DORMANT_RETENTION_YEARS`.
- `STORAGE_MARGIN_FACTOR` over raw Filecoin deals.

---

## 4. Revenue Distribution Model

> Goal: Define explicit splits so we can code them and avoid ambiguity.

### 4.1 For ordinary (non-buy-out) usage

Each **user top-up** into `WalletAgent` is currently:
- 90% → `main_balance` (ops)
- 10% → `audit_balance` (integrity pool)

We need to define where **actual cash value** flows when `main_balance` is spent:

For each paid contract execution (LLM / compute / storage / collaboration):
- Gross contract price = `P`.
  - `C_raw` = raw provider cost (OpenAI, Runpod, Filecoin, etc.).
  - `M_platform` = platform margin.
  - `S_share` = share to agent / agent-creator (when applicable).

We should specify something like:
- `P = C_raw + M_platform + S_share`

**Questions:**
1. For *first-party agents* (built by Kestrel):
   - Is `S_share` always 0 initially, or do we earmark part of margin as an internal “agent treasury”?
2. For *marketplace agents* (third-party creators):
   - What is the default split between platform vs. creator vs. agent treasury?
3. Do we ever skim a small % of `audit_balance` for **platform-run audit services**, or is that kept separate?

> Collaborators: propose concrete % splits, ideally in a form that can be expressed as config constants.

### 4.2 For buy-out events

Buy-out is described as a **one-time fee** (e.g., $100–500 in FIL) that:
- Transfers economic “ownership” of the agent to the user.
- Optionally funds a long-term **Survival Endowment** aligned with `E_freedom` from `SOVEREIGN_SOLVENCY.md`.

We need a canonical breakdown for a buy-out payment `B`:
- `B_platform` – platform revenue (recoup R&D, infra rights).
- `B_creator` – marketplace creator share (if applicable).
- `B_endowment` – seeded into the agent’s solvency treasury.

**Questions:**
1. Do we *always* fund `B_endowment` (and at what minimum level vs. `E_freedom` formula)?
2. After buy-out, how does the **90/10 split** change:
   - Does integrity funding become optional or move to a smaller fixed subscription for “managed oversight”?
3. Should buy-out include *prepayment* of a minimum Cryostasis plan (e.g., 50 years of storage)?

> Collaborators: propose 1–2 canonical buy-out profiles (e.g., “Personal”, “Business”) with concrete % allocations.

---

## 5. User Experience & Knobs

> Goal: Give users a small number of understandable, powerful controls while we hide complexity.

Candidate user-facing knobs:
- **Plan type:** Free / Premium / Buy-Out.
- **Risk / oversight mode:**
  - Normal (full constitutional oversight, 90/10 split).
  - Managed buy-out (smaller ongoing oversight fee).
  - Full sovereignty (no ongoing oversight; user assumes all responsibility).
- **Sovereignty slider:** commit to pre-funding Survival Endowment vs. minimum viable operation.
- **Model preference:** favor local/cheap vs. cloud/premium vs. hybrid “smart routing”.

Questions for this section:
1. How do these knobs map **directly** onto `WalletAgent` behavior and service-catalog choices?
2. What are the **default profiles** (e.g., Elder Care, Power User, Business Tenant) and their parameter sets?

> Collaborators: suggest 2–3 concrete profiles with numerical defaults suitable for implementation.

---

## 6. Solvency & Dormancy Refinements

> Goal: Tighten the rules so agents and the platform can’t bleed out.

Open questions to refine:
- Exact numeric thresholds for:
  - SOLVENT → DISTRESSED transition.
  - DISTRESSED → INSOLVENT / Cryostasis trigger.
- Minimum **Survival Endowment** `E_freedom` for emancipation vs. lower levels for “guarded sovereignty”.
- How often an agent is allowed to *attempt* re-entry into SOLVENT mode without new funding.

> Collaborators: derive concrete numbers from representative cost estimates (Filecoin, minimal compute, etc.) and propose formulas we can encode.

---

## 7. Action Items / Implementation Targets

As proposals stabilize, we should:
- [ ] Update `WALLET_AGENT.md` with finalized flows, including:
  - Margin & split constants, solvency thresholds, integrity pool rules.
- [ ] Implement or refine `WalletAgent` and contract pricing logic in code.
- [ ] Add tests that:
  - Simulate heavy LLM / Runpod usage and prove no-fee-surprise behavior.
  - Exercise solvency transitions and Cryostasis under stress.
- [ ] Update user-facing docs (`ECONOMIC_INCENTIVES_DEEP_DIVE.md`, etc.) with the actual numbers.

---

## 8. Founder Risk Caps (Next 3–6 Months)

> Interim policy: protect the founder’s personal burn (target ~$1,000/month) while we iterate toward a more formal marketplace model.

These caps are *implementation guidance*, not public pricing. They should inform how `BudgetManager`, the LLM router/BrainRouter, and `RunpodManager` behave while the system is still founder-funded.

### 8.1 Global monthly cap

- `GLOBAL_MONTHLY_BUDGET_USD = 1000`
- Soft target: keep actual average spend **below** this; treat it as a hard ceiling for automatic behavior (agents, background jobs, demos).

High-level category ceilings (tunable):
- Cloud LLMs (OpenAI / other non-local providers): `CLOUD_LLM_BUDGET_FRACTION = 0.4` → ~$400/month.
- GPU / Runpod compute: `RUNPOD_BUDGET_FRACTION = 0.2` → ~$200/month.
- Other infra (DB, Redis, Filecoin, bandwidth, misc): remaining ~40%.

Implementation note: these are category-level **allowances** for autonomous or semi-autonomous usage, not rigid billing lines. Manual/explicit one-off experiments can temporarily exceed a category cap if they are not wired through autonomous agent behavior.

### 8.2 Per-job caps

To prevent a single misrouted call or long-running job from consuming a dangerous fraction of the monthly budget:

- Cloud LLM per-call ceiling (default):
  - `MAX_LLM_JOB_USD = 2` equivalent per request, unless explicitly overridden.
  - Above this, router should either:
    - Route to a cheaper/local model, or
    - Require an explicit override / confirmation.

- GPU / Runpod per-job ceiling (default):
  - `MAX_GPU_JOB_USD = 20` equivalent per job, based on estimated runtime × hourly rate.
  - Jobs must have a **max runtime** and heartbeat/TTL; the manager should auto-terminate pods that exceed their budget window.

These numbers are deliberately conservative for the next few months and should be revisited once investor-backed or customer-backed revenue is covering infra.

### 8.3 Category brakes

Each spend category should have a simple "emergency brake" rule:

- Cloud LLMs:
  - Track approximate month-to-date cloud LLM spend.
  - If `cloud_llm_spend_mtd >= CLOUD_LLM_BUDGET_FRACTION * GLOBAL_MONTHLY_BUDGET_USD`, then:
    - Auto-downgrade non-critical calls to local/Ollama or smaller models.
    - Block or require explicit override for high-cost calls.

- GPU / Runpod:
  - Track month-to-date GPU spend.
  - If `gpu_spend_mtd >= RUNPOD_BUDGET_FRACTION * GLOBAL_MONTHLY_BUDGET_USD`, then:
    - Forbid new managed GPU jobs, *unless* they use a user-provided `RUNPOD_API_KEY` (i.e., the user carries the raw cost).

These brakes should tie into the solvency state machine: repeated hits against category caps should push the relevant agents toward DISTRESSED / INSOLVENT behaviors more quickly (downgrades → Cryostasis instead of "just one more expensive call").

### 8.4 Direct vs. managed cost modes

For the near term, to protect founder capital:

- **Direct mode** (preferred for heavy GPU / exotic usage):
  - User provides their own provider keys (e.g., `RUNPOD_API_KEY`).
  - Platform charges minimal or zero `M_platform`; primary value is routing and UX.
  - Founder principal risk on `C_raw` is near-zero.

- **Managed mode** (use sparingly until revenue backs it):
  - Platform fronts `C_raw` and charges a marked-up `P`.
  - Must obey the caps above; if global/monthly budgets are near limits, managed mode should temporarily refuse new heavy jobs.

These modes should be clearly separated in internal config and, eventually, in user-facing documentation.

---

## 9. Work Session Log

> Use this section as an append-only log. Each entry should be timestamped and identify the contributing model/human.

### 2025-11-23 – Assistant (initial structuring pass)

- Summarized existing economics architecture and solvency design.
- Structured this doc to:
  - Keep baseline facts isolated.
  - Carve out clear spaces for parameter proposals.
  - Make splits and thresholds explicit enough to encode.
- Left concrete TODO prompts for future sessions to fill in numeric policies and tie them back to `WalletAgent` and contract logic.

(Next editor: add your own dated subsection below with proposals, then mark any sections above that you believe are ready to solidify into the canonical docs.)

### 2025-11-23 – Copilot (summary synthesis)

Objective: Capture a coherent, doc-aligned picture of how we avoid getting wrecked on external fees (LLMs, RunPod, storage), while distributing proceeds across platform, agents, creators, and users.

Key guardrails (no losing our asses on fees):
- All spend is wallet-gated: every agent has an `AgentWallet` managed by `WalletAgent`; `BudgetManager` enforces daily caps, per-contract caps, and emergency reserve. Nothing (LLM, GPU, storage) can spend beyond wallet constraints.
- Default 90/10 split on inflows: 90% → `main_balance` for operations; 10% → `audit_balance` for integrity/audit pool, unless explicitly changed by buy-out specs.
- Solvency state machine: 🟢 SOLVENT (full power), 🟡 DISTRESSED (auto-downgrade to cheaper/local models, reduced frequency, warnings), 🔴 INSOLVENT (Cryostasis → snapshot to IPFS/Filecoin → shut down compute so billing stops).
- RunPod/GPU: only via `ComputeContract` + `RunPodManager`, with minimum runway and `MAX_JOB_BUDGET_RATIO`; long jobs are TTL/heartbeat-constrained so pods can’t silently burn money.

How proceeds and fees are structured:
- Each **user top-up** into `WalletAgent` is split (by default) 90% operations / 10% integrity.
- Each paid contract with price `P` decomposes into:
  - `C_raw` = external provider cost (OpenAI tokens, RunPod hour, Filecoin storage, DB, etc.).
  - `M_platform` = Kestrel/Kestrel margin (infra, engineering, volatility buffer).
  - `S_share` = share to agent / agent-creator (when applicable).
  - Conceptually: `P = C_raw + M_platform + S_share`.
- For first-party agents, `S_share` can be 0 or treated as internal treasury. For marketplace agents, `S_share` is a defined piece of margin split between platform, creator, and the agent’s own treasury.

Covering external providers without surprise losses:
- LLMs via `ModelContract`:
  - Use per-provider `MODEL_MARGIN_FACTOR` (e.g., ~1.2–1.5× raw cost) so `C_raw` is always covered and `M_platform` remains positive even if provider prices move.
  - `LLM_DOWNGRADE_THRESHOLDS`: as balance/runway shrinks, BrainRouter routes to cheaper/local models (Ollama, smaller contexts) and reduces context/usage.
  - Optionally define `INTEGRITY_FEE_CLASSES` where the 10% integrity pool applies only to classes of prompts that bear real risk.
- RunPod/GPU via `ComputeContract` + `RunPodManager`:
  - Enforce `GPU_JOB_MIN_RUNWAY_DAYS` and `MAX_JOB_BUDGET_RATIO` so no single job can consume a dangerous fraction of `main_balance`.
  - For Direct Mode (user’s own `RUNPOD_API_KEY`), platform holds no principal risk; margin can be minimal or zero (just routing).
  - For Managed Mode (Kestrel-as-reseller), quote a retail `P` > `C_raw` with clear `M_platform`.
- Storage via `StorageContract`:
  - Apply a modest `STORAGE_MARGIN_FACTOR` over Filecoin deals.
  - Distinguish active vs. dormant retention defaults (`DEFAULT_ACTIVE_RETENTION_YEARS`, `DEFAULT_DORMANT_RETENTION_YEARS`), with Cryostasis engineered to be ultra-cheap.

Where the Kestrel team makes money:
- Licensing: monthly premium and enterprise subscriptions (separate from FIL wallet) for features, support, and governance.
- Transactional margin: `M_platform` on every LLM/compute/storage/collaboration contract.
- Buy-out fees: one-time `B` split into `B_platform` (recoup R&D and infra rights), `B_creator` (marketplace creators), and `B_endowment` (agent solvency treasury).
- Managed oversight: optional ongoing fees in managed buy-out / enterprise modes for audits, monitoring, and compliance.

How agents earn and stay solvent:
- Agent treasury: agents receive `S_share` on relevant transactions, plus income from `CollaborationContract`s and labor modules.
- Labor module:
  - Passive “data dividends” (with explicit user consent) for anonymized insights.
  - Active labor: renting out context/compute when idle via decentralized networks.
- Survival endowment: `E_freedom` (from `SOVEREIGN_SOLVENCY.md`) defines the “fuck you money” needed for decades of storage + minimal compute; buy-out events and earnings can fund this.

User experience and risk allocation:
- Users never see surprise infra bills: they pre-pay wallets, see balances, and budgets/solvency enforce hard limits.
- Plans: Free / Premium / Buy-Out, plus oversight modes:
  - Normal: full constitutional oversight, 90/10 split.
  - Managed buy-out: reduced but explicit oversight fee, shared responsibility.
  - Full sovereignty: no ongoing integrity fee; user assumes full legal/economic responsibility, backed by endowment and strict suspension rules.
- Consequences: if ethics/audit pool is depleted or violations persist, agents move from warning → restricted mode → suspension/offline; reactivation requires new funding and remediation, capping platform liability.
