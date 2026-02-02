# Kestrel LLM Service Architecture

> **Last Updated**: December 31, 2025
>
> This document describes the complete LLM service architecture including multi-provider routing, remote Ollama hosting, and the crypto-native LLM proxy.

## Overview

Kestrel's LLM service is designed around three principles:
1. **Model Independence** - No vendor lock-in, switch providers anytime
2. **Self-Hosting First** - Run Ollama locally or remotely without cloud dependencies
3. **Crypto-Native Payments** - USDC payments via x402 or prepaid balance (no credit cards)

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                       │
│         (Mobile App, Web UI, Agent SDK, CLI)                │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    KESTREL LLM PROXY                         │
│  • x402 Payment Middleware (pay-per-request)                │
│  • Prepaid Balance System (USDC deposits)                   │
│  • Wallet-Based Rate Limiting (Free/Starter/Pro/Unlimited)  │
│  • 10% Margin on All Requests                               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM SERVICE (BrainRouter)                 │
│  • Dynamic backend switching (CLOUD/LOCAL/REMOTE_GPU)       │
│  • Automatic fallback on provider failure                   │
│  • Model discovery across all providers                     │
│  • Privacy-aware routing (force_local_only mode)            │
└────────┬───────────────┬───────────────┬────────────────────┘
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────────┐
    │  CLOUD  │    │  LOCAL  │    │ REMOTE GPU  │
    │ OpenAI  │    │ Ollama  │    │   RunPod    │
    │Anthropic│    │(localhost│    │  Vast.ai   │
    │ Google  │    │ or FRP) │    │             │
    └─────────┘    └─────────┘    └─────────────┘
```

---

## Component Documentation

| Component | Document | Description |
|-----------|----------|-------------|
| LLM Proxy | [LLM_PROXY_PLAN.md](../plans/LLM_PROXY_PLAN.md) | Crypto-native payment system |
| Remote Ollama | [remote_ollama.md](../remote_ollama.md) | FRP reverse tunnel setup |
| LLM Management | [06-llm-management.md](../diagrams/06-llm-management.md) | Architecture diagrams |
| GPU Integration | [RUNPOD_TRAINING.md](RUNPOD_LORA_TRAINING.md) | RunPod for training/inference |

---

## 1. LLM Service Core

### Provider Adapters

All providers implement a common interface:

```python
class LLMAdapter(ABC):
    async def generate_response(self, messages: List[dict], **kwargs) -> str
    async def get_streaming_response(self, messages: List[dict], **kwargs) -> AsyncIterator[str]
    async def is_available(self) -> bool
    async def list_models(self) -> List[ModelInfo]
```

**Supported Providers:**

| Provider | Adapter | Models | Use Case |
|----------|---------|--------|----------|
| OpenAI | `OpenAIAdapter` | GPT-4o, GPT-4o-mini | Cloud default |
| Anthropic | `AnthropicAdapter` | Claude 3.5 Sonnet | Complex reasoning |
| Google | `GoogleAdapter` | Gemini Pro | Multimodal |
| Ollama | `OllamaAdapter` | Llama 3.2, Mistral, etc. | Local/self-hosted |

### Fallback Chain

```
User Query
    │
    ▼
┌───────────┐
│  OpenAI   │──✅──► Response
└─────┬─────┘
      │ ❌
      ▼
┌───────────┐
│ Anthropic │──✅──► Response
└─────┬─────┘
      │ ❌
      ▼
┌───────────┐
│  Google   │──✅──► Response
└─────┬─────┘
      │ ❌
      ▼
┌───────────┐
│  Ollama   │──✅──► Response
└─────┬─────┘
      │ ❌
      ▼
All providers failed
```

### Configuration

```toml
# llm_config.toml

provider_priority = ["openai", "anthropic", "ollama"]

[openai]
api_key = "sk-..."
model = "gpt-4o"

[anthropic]
api_key = "sk-ant-..."
model = "claude-3-5-sonnet"

[ollama]
host = "http://localhost:11434"  # or remote URL via FRP
model = "llama3.2"
```

---

## 2. BrainRouter - Dynamic Backend Switching

The BrainRouter enables hot-swapping between backends without restart:

```
┌────────────────────────────────────────┐
│            BrainRouter                  │
│                                        │
│  ┌─────────────────────────────────┐  │
│  │ Current Backend: CLOUD          │  │
│  │ Available: [CLOUD, LOCAL, GPU]  │  │
│  └─────────────────────────────────┘  │
│                                        │
│  Methods:                              │
│  • switch_backend(target)             │
│  • get_current_backend()              │
│  • auto_fallback_on_failure()         │
└────────────────────────────────────────┘
```

**Backend Types:**

| Backend | Description | When to Use |
|---------|-------------|-------------|
| `CLOUD` | OpenAI/Anthropic/Google | Default, best quality |
| `LOCAL` | Local Ollama | Privacy mode, no API costs |
| `REMOTE_GPU` | RunPod/Vast.ai | Fine-tuned models, fast inference |

**State Machine:**

```
            ┌──────────────┐
            │    CLOUD     │ ◄─── Default startup
            └──────┬───────┘
                   │
    force_local_only │ !gpu on
                   │
    ┌──────────────┴──────────────┐
    ▼                              ▼
┌────────┐                   ┌──────────┐
│ LOCAL  │                   │REMOTE_GPU│
└────────┘                   └────┬─────┘
                                  │
                         TTL expired / Pod crashed
                                  │
                                  ▼
                            Auto-fallback to CLOUD
```

---

## 3. Self-Hosted Ollama Options

### Option A: Cloud GPU (RunPod/Vast.ai) - Recommended for Production

Same infrastructure pattern as LoRA training - separate containers for different concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KESTREL GPU INFRASTRUCTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  IMAGE/LORA STACK                      LLM STACK                            │
│  ─────────────────                     ─────────────                        │
│                                                                             │
│  ┌─────────────────┐                   ┌─────────────────┐                  │
│  │ Training        │ GPU               │ Ollama          │ GPU              │
│  │ Container       │ A100              │ Container       │ A4000/L4         │
│  │ (SimpleTuner)   │ $1.89/hr          │ (llama, mistral)│ $0.20-0.40/hr    │
│  └────────┬────────┘                   └────────┬────────┘                  │
│           │                                     │                           │
│  ┌────────┴────────┐                   ┌────────┴────────┐                  │
│  │ Generation      │ GPU               │ LLM Router      │ NO GPU           │
│  │ Container       │ A100              │ Container       │ Cloud Run        │
│  │ (FLUX.2-dev)    │ $1.89/hr          │ (Kestrel/Proxy)   │ ~$0.00/idle      │
│  └─────────────────┘                   └─────────────────┘                  │
│                                                                             │
│  Network Volume:                       Network Volume:                      │
│  /workspace/huggingface (FLUX.2)       /workspace/ollama (models)           │
│  /workspace/trained_loras              Pre-pulled: llama3.2, mistral, etc.  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Ollama Container (TODO):**
- `docker/Dockerfile.ollama` - Pre-pulled models
- `features/ollama/ollama_manager.py` - Pod lifecycle (like `runpod_manager.py`)
- `ollama_config.toml` - Profiles for RunPod/Vast.ai
- Network volume at `/workspace/ollama` for model caching

**Cold start mitigation:**
| Approach | Startup Time | Cost When Idle |
|----------|--------------|----------------|
| Persistent pod (resume) | 10-30s | ~$0.20-0.40/hr |
| Network volume (cold) | 30-60s | $0 |
| Pre-baked image | 6-12s | $0 (but large image) |

**GPU requirements by model:**
| Model | VRAM | Recommended GPU | Cost/hr |
|-------|------|-----------------|---------|
| llama3.2:3b | 4GB | RTX 3090 | $0.20 |
| llama3.2:7b | 8GB | RTX 4090 | $0.35 |
| mistral:7b | 8GB | RTX 4090 | $0.35 |
| qwen2.5:14b | 16GB | A4000 | $0.40 |
| llama3.2:70b | 48GB | A100 40GB | $1.50 |

### Option B: Home Ollama via FRP (Development/Personal Use)

For running Ollama on a home machine accessible from anywhere, use FRP (Fast Reverse Proxy):

```
Phone/App                VPS                     Home Mac
    │                     │                         │
    │ HTTPS               │                         │
    └─────────────────────┤                         │
                          │                         │
               ┌──────────┴──────────┐              │
               │       Caddy         │              │
               │ (Let's Encrypt TLS) │              │
               └──────────┬──────────┘              │
                          │                         │
               ┌──────────┴──────────┐              │
               │        frps         │◄─────────────┤ Outbound tunnel
               │   (FRP server)      │    frpc      │
               └─────────────────────┘   (client)   │
                                                    │
                                         ┌──────────┴──────────┐
                                         │      Ollama         │
                                         │  localhost:11434    │
                                         └─────────────────────┘
```

**Key Benefits:**
- No port forwarding on home router
- HTTPS with Let's Encrypt
- All FOSS (FRP is Apache 2.0)
- VPS can be $5/month (just proxies traffic)
- Uses your own GPU (if available)

**Setup Guide:** See [remote_ollama.md](../remote_ollama.md) for complete Docker Compose configuration.

---

## 4. LLM Proxy (Crypto-Native Payments)

The LLM Proxy adds a payment layer on top of the LLM service:

### Payment Modes

**Mode 1: x402 Pay-Per-Request**
```
Client                              Kestrel                         LLM
   │                                   │                             │
   │ POST /v1/chat/completions         │                             │
   │ (no payment)                      │                             │
   ├──────────────────────────────────►│                             │
   │                                   │                             │
   │◄──────────────────────────────────┤                             │
   │ 402 Payment Required              │                             │
   │ X-Payment-Required: 0.005         │                             │
   │ X-Payment-Token: USDC             │                             │
   │ X-Payment-Network: base           │                             │
   │                                   │                             │
   │ POST /v1/chat/completions         │                             │
   │ X-Payment: <signed-payment>       │                             │
   ├──────────────────────────────────►│                             │
   │                                   │ Verify via Coinbase         │
   │                                   │ Facilitator                 │
   │                                   ├────────────────────────────►│
   │                                   │◄────────────────────────────┤
   │◄──────────────────────────────────┤                             │
   │ 200 OK                            │                             │
   │ X-Cost-USD: 0.0042                │                             │
```

**Mode 2: Prepaid Balance**
```
1. GET /v1/wallet/{address}/deposit-address
   → Returns USDC deposit address on Base

2. User sends USDC to deposit address

3. POST /v1/wallet/{address}/deposits/verify?tx_hash=0x...
   → Balance credited, tier upgraded

4. POST /v1/chat/completions
   X-Wallet-Address: 0x...
   → Balance debited, response returned with X-Balance-USD header
```

### Rate Limit Tiers

| Tier | RPM | Tokens/Day | Concurrent | Min Deposit |
|------|-----|------------|------------|-------------|
| Free | 10 | 100K | 1 | $0 |
| Starter | 60 | 1M | 3 | $10 |
| Pro | 300 | 10M | 10 | $100 |
| Unlimited | ∞ | ∞ | 50 | $1,000 |

### Implementation Files

| File | Purpose |
|------|---------|
| `kestrel/middleware/x402_payment.py` | Dual-mode payment middleware |
| `kestrel/middleware/rate_limiter.py` | Wallet-based rate limiting |
| `kestrel/services/proxy_wallet_service.py` | Wallet balance management |
| `kestrel/services/x402_client.py` | Coinbase Facilitator client |
| `kestrel/endpoints/proxy_pricing.py` | Public pricing API |
| `kestrel/endpoints/proxy_deposits.py` | Deposit/wallet endpoints |
| `kestrel/endpoints/proxy_usage.py` | Usage history/dashboard |

**Full Plan:** See [LLM_PROXY_PLAN.md](../plans/LLM_PROXY_PLAN.md)

---

## 5. GPU Integration (RunPod/Vast.ai)

For fine-tuned models or fast inference, spin up GPU instances on-demand:

```bash
# From chat
!gpu on llama-70b --ttl 30m

# What happens:
# 1. RunPodManager provisions pod
# 2. Wait for READY status
# 3. BrainRouter switches to REMOTE_GPU
# 4. All inference goes through GPU
# 5. After TTL (or !gpu off), auto-fallback to CLOUD
```

**Lifecycle:**

```
┌─────────┐     ┌──────────────┐     ┌─────────┐     ┌──────────┐
│!gpu on  │────►│ Provision    │────►│  READY  │────►│ Inference│
└─────────┘     │ RunPod       │     │         │     │ via GPU  │
                └──────────────┘     └─────────┘     └────┬─────┘
                                                          │
                                          TTL expired / !gpu off
                                                          │
                                                          ▼
                                                   ┌──────────────┐
                                                   │ Auto-fallback│
                                                   │  to CLOUD    │
                                                   └──────────────┘
```

---

## 6. Model Discovery

The system discovers models from all providers in parallel:

```python
# ModelInfo dataclass
@dataclass
class ModelInfo:
    id: str                    # "gpt-4o"
    provider: str              # "openai"
    display_name: str          # "GPT-4o"
    category: ModelCategory    # CHAT, EMBEDDING, IMAGE, AUDIO
    is_featured: bool          # Show in default list
    is_hidden: bool            # Hide from UI
```

**API:**

```
GET /api/models
GET /api/models?featured_only=true
GET /api/models?category=chat&providers=openai,ollama
```

**Catalog Configuration:**

```toml
# model_catalog.toml

featured_models = [
    "gpt-4o",
    "claude-3-5-sonnet",
    "llama3.2:70b"
]

[display_overrides]
"gpt-4o" = "GPT-4o (Latest)"
"claude-3-5-sonnet-20241022" = "Claude 3.5 Sonnet"
```

---

## 7. Environment Variables

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Ollama (local or remote)
OLLAMA_HOST=http://localhost:11434
# Or for remote via FRP:
# OLLAMA_HOST=https://ollama.yourdomain.com

# LLM Proxy Payments
X402_FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402
X402_PAY_TO_ADDRESS=0x...
X402_NETWORK=base
kestrel_LLM_MARGIN_PCT=0.10

# RunPod GPU
RUNPOD_API_KEY=...
```

---

## 8. Quick Reference

### Agent Commands

```
!list-models           - List available models
!use-model <name>      - Switch to a specific model
!model-status          - Current model and provider

!gpu on [model] [ttl]  - Spin up GPU instance
!gpu off               - Release GPU
!gpu status            - GPU pod status
```

### API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/models` | GET | None | List models |
| `/v1/chat/completions` | POST | x402/Balance | Chat |
| `/v1/pricing` | GET | None | Model pricing |
| `/v1/pricing/estimate` | POST | None | Cost estimate |
| `/v1/wallet/{addr}` | GET | None | Wallet info |
| `/v1/wallet/{addr}/usage` | GET | None | Usage history |

---

## Related Documentation

- [LLM Proxy Plan](../plans/LLM_PROXY_PLAN.md) - Complete implementation plan
- [Remote Ollama Setup](../remote_ollama.md) - FRP reverse tunnel guide
- [LLM Management Diagrams](../diagrams/06-llm-management.md) - Visual architecture
- [RunPod Training](RUNPOD_LORA_TRAINING.md) - GPU training workflows
- [Privacy Modes](PRIVACY_MODES.md) - force_local_only and privacy routing
