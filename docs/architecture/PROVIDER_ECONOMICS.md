---
type: Architecture Spec
title: 'Provider Economics: Middleman Architecture & Revenue Strategy'
description: '**Version:** 1.0 **Drafted:** December 2025 **Last verified:** 2026-04-25
  (referral-program details still accurate; revenue projections are illustrative,
  not date-stamped financ...'
resource: /docs/architecture/PROVIDER_ECONOMICS.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Provider Economics: Middleman Architecture & Revenue Strategy

**Version:** 1.0
**Drafted:** December 2025
**Last verified:** 2026-04-25 (referral-program details still accurate; revenue projections are illustrative, not date-stamped financials)
**Status:** Strategy Document

## Overview

Kestrel operates as a **provider middleman**, enabling users to either bring their own cloud credentials (sovereignty) or use platform-managed infrastructure (convenience). This architecture creates two distinct revenue streams:

1. **Referral Revenue**: Commission from providers when users sign up via our links (Direct Mode)
2. **Margin Revenue**: Markup on provider costs when we manage infrastructure (Managed Mode)

## PayerPolicy Foundation (2026-05)

Below the two-mode framing sits a foundation primitive — `PayerPolicy` — that names *who pays for which metered resource* per agent. Direct Mode and Managed Mode are user-facing labels; PayerPolicy is the per-agent declarative knob the framework reasons about. See [PAYER_POLICY_FOUNDATION.md](PAYER_POLICY_FOUNDATION.md) for the full plan; the schema lives in the [kestrel-sovereign-sdk repo](https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk) at `kestrel_sdk/payer_policy.py`.

### Six funding patterns

| `PayerKind` | Mode framing | Who pays | Implementation |
|---|---|---|---|
| `HOST_ENV` | Direct (operator's env) | Operator's env vars | Today's standalone behavior — preserved as a first-class default |
| `HOST_MASTER_PROVISIONED` | Managed | Operator's master account; child key per agent | Master in `HostKeyStorage`; resolver mints child via OpenRouter Provisioning API |
| `USER_MASTER_PROVISIONED` | Direct (user-bound) | User's own master account | Same mechanism; `master_did` carries user DID |
| `SPONSOR` | Managed (third-party) | Third party (family member, employer, etc.) | Same mechanism; `master_did` carries sponsor DID |
| `SELF_WALLET` | Sovereign-pays | Agent's own crypto wallet (e.g. x402) | **Deferred (matrix `NOT_IMPLEMENTED`)** — Phase 3.5 of the plan ships the Lighthouse wallet-signed key flow; LLM `x402` deferred indefinitely until the standard matures. The wizard refuses to offer this kind for any resource until Phase 3.5 lands. |
| `NONE` | (none) | Resource is unavailable to this agent | `LLMService.disabled = True`; storage provider not constructed |

### Resource classes

A `PayerPolicy` carries one `PayerSpec(vendor, kind, master_did?, monthly_cap_usd?)` per resource class:

- **llm** — LLM inference (OpenRouter, local Ollama, …)
- **storage** — IPFS/file storage (Lighthouse, local-disk)
- **compute** — GPU rentals (host_env-only in v1; Phase 3.5+ for delegated kinds)
- **tools** — third-party API calls (Tavily, Exa, ElevenLabs, …)
- **comms** — email/SMS (Twilio, Resend, …)

The wizard step (`kestrel setup payments`) only offers `(resource, vendor, kind)` triples that the SDK `SUPPORT_MATRIX` marks `READY` — operators cannot select a path the resolver cannot honor at runtime.

### Source of truth + safety

- The SDK's `SUPPORT_MATRIX` is the single source of truth that the wizard, resolver, and verify-step all consult.
- The resolver also keeps a defense-in-depth gate that refuses delegated-master kinds for non-LLM resources, in case a future SDK release accidentally regresses the matrix.
- The mint flow is **atomic-with-rollback**: on `HOST_MASTER_PROVISIONED` first use, the resolver mints a remote child key, persists `openrouter_key_hash` to `graph_nodes.properties` (so retirement can revoke), and only then stores the key locally. If persist fails (graph_nodes vanished mid-mint), the remote key is revoked before the resolver returns.
- Per-agent `asyncio.Lock` (class-level on the resolver) prevents concurrent agent inits from minting two remote keys for the same DID.

## Architecture: Two Modes

```
┌─────────────────────────────────────────────────────────┐
│                    USER CHOICE                          │
├─────────────────────────┬───────────────────────────────┤
│   DIRECT (Sovereign)    │    MANAGED (Kestrel Platform)   │
├─────────────────────────┼───────────────────────────────┤
│ User's own API key      │ Kestrel-provided credentials  │
│ User pays provider      │ User pays Kestrel (markup)    │
│ Referral link used      │ Kestrel absorbs provider cost │
│ Kestrel earns 3-10%     │ Kestrel earns margin (20-40%) │
│ True sovereignty        │ Convenience, no cloud hassle  │
│ User controls billing   │ Single bill from Kestrel      │
└─────────────────────────┴───────────────────────────────┘
```

### Direct Mode (Sovereignty First)

- User brings their own API keys (RunPod, OpenAI, etc.)
- User pays provider directly
- Kestrel provides referral-tagged signup URLs
- Revenue: 3-10% referral commission from providers
- **Value proposition**: True data sovereignty, no platform lock-in

### Managed Mode (Convenience First)

- Kestrel provides all infrastructure
- User pays single subscription to Kestrel
- Kestrel pays providers at wholesale rates
- Revenue: 20-40% margin on provider costs
- **Value proposition**: No cloud accounts needed, simplified billing

## Provider Analysis

### RunPod (GPU Compute) - PRIMARY OPPORTUNITY

**Referral Program Details:**
| Tier | Commission | Duration | Requirements |
|------|------------|----------|--------------|
| Standard Referral | 3% Pod + 5% Serverless | 6 months | None |
| Affiliate Program | 10% all spend (cash) | Lifetime | 25+ referred users |
| Template Creator | 1% compute | Lifetime | Publish template |
| Hub Maintainer | Up to 7% compute | Monthly tiers | Maintain repos |

**Key Details:**
- Both referrer AND referred get $5-500 random bonus after first $10 load
- Pre-June 2025 referrals: Lifetime 3%/5% (grandfathered)
- Affiliate payouts via PartnerStack (PayPal or Stripe)
- Payments verified and paid monthly (e.g., February → March 13th)

**Configuration:**
```bash
# Add to .env
RUNPOD_REFERRAL_CODE=your_referral_code_here
```

**Integration Points:**
- `features/runpod/runpod_manager.py` - Direct vs Managed provider selection
- `features/runpod/feature.py` - User-facing GPU management tools

### Lighthouse (Filecoin/IPFS Storage) - ALREADY INTEGRATED

**Current Implementation:**
- File: `storage/providers/lighthouse_provider.py` (589 lines)
- Hot IPFS pricing: $0.05/GB/month (our charge, matches Lighthouse hot tier)
- Perpetual pricing: ~$2-5/GB one-time (Lighthouse endowment pool)
- Raw Filecoin deal: ~$0.00005/GB (not viable alone - requires manual renewal)
- Margin on hot storage: minimal (pass-through pricing)

**Partnership Model:**
- NFT.Storage uses referral link model (% of first storage purchase)
- No formal public affiliate program found
- Consider direct partnership outreach for volume terms

**Configuration:**
```bash
# Add to .env
LIGHTHOUSE_API_KEY=your_api_key_here
```

### Lambda Labs (Enterprise GPU) - FORMAL PARTNER PROGRAM

**Partner Program:**
- Official page: https://lambda.ai/partners
- For VARs (Value Added Resellers), MSPs, and technology partners
- NVIDIA Partner of the Year 4 years running
- Requires formal partnership application

**Strategic Value:**
- Enterprise GPU market ($1.5B+ raised)
- Multi-billion dollar Microsoft partnership (November 2025)
- Access to NVIDIA GB300 NVL72 systems

**Next Steps:**
- Apply via partner portal once enterprise customers identified
- Negotiate volume discounts and revenue share

### Replicate (ML Models) - PAY-PER-USE, PORTABLE WEIGHTS

**Current Status:**
- No affiliate/partner program publicly available
- Pay-per-prediction and pay-per-training pricing model
- FLUX.1 only (NOT FLUX.2) - see limitations below

**Training Pricing (FLUX.1-dev LoRA):**
| Model | Cost | Duration | Output |
|-------|------|----------|--------|
| `ostris/flux-dev-lora-trainer` | ~$2-5 per run | ~15-20 min | .safetensors weights |

**Generation Pricing (FLUX Models):**
| Model | Cost per Image | Speed | Quality | License |
|-------|---------------|-------|---------|---------|
| `flux-schnell` | ~$0.003 | Fast | Good | Apache 2.0 |
| `flux-dev` | ~$0.025 | Medium | High | Non-commercial |
| `flux-1.1-pro` | ~$0.04 | Fast | Excellent | Commercial |

**Key Limitations:**
- ⚠️ **CENSORED**: Replicate applies content safety filters to all generation
- ⚠️ **FLUX.1 Only**: Training uses FLUX.1-dev, not FLUX.2
- ✅ **Portable Weights**: LoRA .safetensors can be downloaded and used elsewhere

**Cross-Provider Strategy:**
```
Train on Replicate (~$3)  →  Download weights  →  Generate on RunPod (uncensored)
   (cheap, serverless)          (portable)           (FLUX.2-dev, no filters)
```

**Integration Points:**
- `features/training/adapters/replicate_adapter.py` - Training + generation
- `features/training/factory.py` - Provider capabilities and routing
- `kestrel/image_generation.py` - FLUX.1-schnell for quick generation

**Recommendation:**
- Use for **cost-effective training** ($2-5 vs $5-10 on Vertex)
- Use for **safe content generation** (filtered output acceptable)
- For **uncensored generation**: download weights, use RunPod
- Monitor for affiliate program launch

### Modal Labs (Serverless GPU) - NO PUBLIC PROGRAM

**Current Status:**
- No affiliate program found
- Pay-per-CPU-cycle billing model
- $80M raised (September 2025)

**Recommendation:**
- Lower priority (no revenue opportunity currently)
- Consider for compute routing optimization

### OpenAI / Anthropic / Google (LLM Providers)

**Status:**
- No referral programs for API usage
- Volume discounts available for enterprise
- Revenue only via Managed Mode markup

**Strategy:**
- Focus on margin revenue (20-40% markup)
- Negotiate enterprise volume discounts when applicable

## Revenue Calculations

### Per-User Monthly Revenue (Estimated)

| Provider | User Spend | Direct Mode (Referral) | Managed Mode (Margin) |
|----------|------------|------------------------|------------------------|
| RunPod H100 | $50/mo | $2.50-5.00 (5-10%) | $10-20 (20-40%) |
| Replicate Training | $10/mo | $0 (no program) | $2-4 (20-40%) |
| Replicate Generation | $5/mo | $0 (no program) | $1-2 (20-40%) |
| Lighthouse Storage | $5/mo | ~$0.50 (est.) | $4.50 (90% margin) |
| OpenAI API | $20/mo | $0 (no program) | $4-8 (20-40%) |
| Anthropic API | $15/mo | $0 (no program) | $3-6 (20-40%) |

### Aggregate Revenue Projections

#### Conservative (100 users)
| Source | Users | Monthly/User | Total |
|--------|-------|--------------|-------|
| RunPod Referral (Direct) | 30 | $3.50 | $105 |
| RunPod Managed Margin | 20 | $15.00 | $300 |
| Lighthouse Margin | 50 | $4.50 | $225 |
| LLM Managed Margin | 70 | $6.00 | $420 |
| **Monthly Total** | | | **$1,050** |

#### At Scale (1,000 users)
| Source | Monthly Total |
|--------|---------------|
| Provider Revenue | $10,500 |
| Subscription Revenue | $15,000 (avg $15/user) |
| **Combined** | **$25,500/month** |

#### Enterprise Scale (10,000 users)
| Source | Monthly Total |
|--------|---------------|
| Provider Revenue | $105,000 |
| Subscription Revenue | $150,000 |
| **Combined** | **$255,000/month** |

## Implementation Roadmap

### Phase 1: Documentation (Current)
- [x] Create PROVIDER_ECONOMICS.md (this document)
- [ ] Update BUSINESS_PLAN_V2.md with provider revenue stream
- [ ] Update AGENT_ECONOMICS.md with provider economics
- [ ] Update PLAN_RUNPOD_INTEGRATION.md with referral section

### Phase 2: Basic Integration (Future)
- [ ] Add `RUNPOD_REFERRAL_CODE` usage in feature.py
- [ ] Generate referral URLs for Direct Mode users
- [ ] Track referral signups for affiliate qualification

### Phase 3: Provider Base Class (Future)
- [ ] Create `ReferralCapable` mixin in providers/base.py
- [ ] Standardize referral URL generation across providers
- [ ] Add referral tracking to user database

### Phase 4: Usage & Billing (Future)
- [ ] Implement usage tracking per provider
- [ ] Create billing system for Managed Mode
- [ ] Build revenue dashboard for visibility

### Phase 5: Optimization (Future)
- [ ] Apply for RunPod affiliate program (10% cash at 25+ users)
- [ ] Negotiate Lighthouse partnership terms
- [ ] Evaluate Lambda Labs VAR partnership

## Configuration Reference

### Environment Variables

```bash
# Provider API Keys
RUNPOD_API_KEY=rp_...              # Direct Mode: user's key
RUNPOD_REFERRAL_CODE=...           # Referral tracking code
LIGHTHOUSE_API_KEY=...             # Storage provider key
OPENAI_API_KEY=sk-...              # LLM provider key
ANTHROPIC_API_KEY=sk-ant-...       # LLM provider key

# Managed Mode Keys (Platform-owned)
KESTREL_RUNPOD_KEY=rp_...          # Platform GPU key
KESTREL_OPENAI_KEY=sk-...          # Platform LLM key

# Pricing Configuration
GPU_MARKUP_PERCENT=30              # Managed mode margin
LLM_MARKUP_PERCENT=25              # Managed mode margin
STORAGE_COST_PER_GB=0.05           # Monthly storage rate
```

### Feature Pattern

```python
class ProviderFeature(Feature):
    """Base pattern for provider-backed features."""

    # Mode support
    supports_direct_mode: bool = True      # User brings own key
    supports_managed_mode: bool = True     # Use platform credentials

    # Referral configuration
    referral_code: Optional[str] = None    # From env var
    referral_base_url: Optional[str] = None

    def get_signup_url(self, user_id: str) -> str:
        """Generate referral-tagged signup URL for Direct Mode."""
        if not self.referral_code:
            return self.referral_base_url
        return f"{self.referral_base_url}?ref={self.referral_code}"

    def get_usage_cost(self, operation: str, mode: str) -> Decimal:
        """Return cost with markup for Managed Mode."""
        base_cost = self._get_base_cost(operation)
        if mode == "managed":
            return base_cost * Decimal("1.30")  # 30% markup
        return base_cost  # Direct mode: user pays provider
```

## Strategic Considerations

### Why Both Modes?

1. **Sovereignty Users**: Privacy-first users want full control. They'll use their own keys and we earn referral revenue (smaller margin, higher trust).

2. **Convenience Users**: Most users want simplicity. They'll pay a premium for managed infrastructure (higher margin, better experience).

3. **Revenue Diversification**: Two revenue streams reduce risk. If referral programs change, margin revenue continues.

### Competitive Positioning

- **vs. Pure SaaS** (Character.AI, Replika): We offer sovereignty option they can't match
- **vs. Self-Hosted** (Ollama, LocalAI): We offer convenience they can't match
- **vs. Cloud Providers** (RunPod, Lambda): We offer integrated AI experience they don't focus on

### Future Opportunities

1. **RunPod Affiliate**: Apply at 25+ referred users for 10% lifetime cash commission
2. **Lighthouse Partnership**: Negotiate direct terms for volume discounts
3. **Lambda Labs VAR**: Formal reseller agreement for enterprise GPU
4. **Template Revenue**: Publish RunPod templates (1% ongoing compute revenue)
5. **Hub Revenue**: Maintain RunPod Hub repos (up to 7% compute revenue)

---

*This document defines Kestrel's provider economics strategy. Implementation details are tracked in the respective provider integration documents.*
