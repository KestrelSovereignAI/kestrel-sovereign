# 06 - LLM Management

Multi-model fallback, BrainRouter, and GPU integration.

---

## Slide 1: The Model Independence Principle

```mermaid
graph LR
    subgraph locked["❌ Locked to One Provider"]
        APP1[Your App] --> GPT[OpenAI Only]
        GPT --> RISK1[Price changes]
        GPT --> RISK2[Outages]
        GPT --> RISK3[Policy changes]
    end
    
    subgraph free["✅ Model Independent"]
        APP2[Kestrel] --> ROUTER[LLM Router]
        ROUTER --> ANY[Any Provider]
    end
    
    style locked fill:#641e16,stroke:#ec7063
    style free fill:#145a32,stroke:#58d68d
```

**No vendor lock-in.** Switch models anytime.

---

## Slide 2: Supported Providers

```mermaid
graph TB
    subgraph providers["🧠 LLM Providers"]
        OAI[OpenAI<br/>GPT-4, GPT-4o]
        ANT[Anthropic<br/>Claude 3.5]
        GOOGLE[Google<br/>Gemini]
        OLLAMA[Ollama<br/>Local models]
    end
    
    subgraph config["Configuration"]
        TOML["kestrel.toml [llm]"]
        PRIO[route_priority]
        KEYS[API keys]
    end
    
    config --> providers
    
    style providers fill:#1a5276,stroke:#85c1e9
    style OLLAMA fill:#145a32,stroke:#58d68d
```

**Ollama = Always available locally.** No API key needed.

---

## Slide 3: The Fallback Chain

```mermaid
graph TD
    Q[User Query] --> TRY1{OpenAI?}
    TRY1 -->|✅ Success| R[Response]
    TRY1 -->|❌ Fail| TRY2{Anthropic?}
    TRY2 -->|✅ Success| R
    TRY2 -->|❌ Fail| TRY3{Google?}
    TRY3 -->|✅ Success| R
    TRY3 -->|❌ Fail| TRY4{Ollama?}
    TRY4 -->|✅ Success| R
    TRY4 -->|❌ Fail| FAIL[All providers failed]
    
    style TRY1 fill:#145a32,stroke:#58d68d
    style TRY2 fill:#512e5f,stroke:#af7ac5
    style TRY3 fill:#1a5276,stroke:#85c1e9
    style TRY4 fill:#7d6608,stroke:#f4d03f
    style FAIL fill:#641e16,stroke:#ec7063
```

**Resilient by design.** Agent keeps working.

---

## Slide 4: LLM Service Architecture

```mermaid
graph TD
    subgraph service["LLMService"]
        LOAD[Load config]
        INIT[Initialize adapters]
        PRIO[Set priority order]
    end
    
    subgraph adapters["Adapters"]
        OAI_A[OpenAIAdapter]
        ANT_A[AnthropicAdapter]
        GOOGLE_A[GoogleAdapter]
        OLLAMA_A[OllamaAdapter]
    end
    
    subgraph interface["Common Interface"]
        GEN[generate_response]
        AVAIL[is_available]
        STREAM[get_streaming_response]
    end
    
    service --> adapters --> interface
    
    style service fill:#1a5276,stroke:#85c1e9
```

**Unified interface.** All providers look the same to the agent.

---

## Slide 5: BrainRouter - Dynamic Backend Switching

```mermaid
graph LR
    subgraph router["🧠 BrainRouter"]
        STATE[Current backend]
        SWITCH[switch_backend]
        FALLBACK[Auto-fallback]
    end
    
    subgraph backends["Backends"]
        CLOUD[☁️ CLOUD<br/>OpenAI/Anthropic]
        LOCAL[💻 LOCAL<br/>Ollama]
        GPU[⚡ REMOTE_GPU<br/>RunPod]
    end
    
    router --> backends
    
    style router fill:#1a5276,stroke:#85c1e9,stroke-width:2px
    style GPU fill:#7d3c00,stroke:#f5b041
```

**Hot-swap.** Change backend without restart.

---

## Slide 6: BrainRouter State Machine

```mermaid
stateDiagram-v2
    [*] --> CLOUD: Default startup
    
    CLOUD --> LOCAL: force_local_only
    CLOUD --> REMOTE_GPU: !gpu on
    
    LOCAL --> CLOUD: Resume cloud
    LOCAL --> REMOTE_GPU: !gpu on
    
    REMOTE_GPU --> CLOUD: TTL expired
    REMOTE_GPU --> CLOUD: Pod crashed
    REMOTE_GPU --> LOCAL: Explicit + local
    
    note right of REMOTE_GPU: Auto-fallback on failure
```

**Always recovers.** GPU dies → falls back automatically.

---

## Slide 7: RunPod GPU Integration

```mermaid
graph LR
    subgraph user["User Request"]
        CMD["!gpu on llama-70b"]
    end
    
    subgraph provision["RunPodManager"]
        PROV[Provision pod]
        WAIT[Wait for READY]
        URL[Get endpoint]
    end
    
    subgraph router["BrainRouter"]
        SW[switch_backend REMOTE_GPU]
        CFG[Configure client]
    end
    
    subgraph infer["Inference"]
        GEN[Generate via GPU]
        RES[Fast response]
    end
    
    CMD --> PROV --> WAIT --> URL --> SW --> CFG --> GEN --> RES
    
    style SW fill:#7d3c00,stroke:#f5b041,stroke-width:2px
```

**On-demand GPU.** Pay only when you need power.

---

## Slide 8: GPU Lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant K as Kestrel
    participant RP as RunPod
    participant BR as BrainRouter
    
    U->>K: !gpu on llama-70b
    K->>RP: Provision pod
    RP-->>K: Pod ID, status: PROVISIONING
    loop Until READY
        K->>RP: Check status
        RP-->>K: Status update
    end
    RP-->>K: Status: READY, URL
    K->>BR: switch_backend(REMOTE_GPU, url)
    BR-->>K: Backend switched
    K-->>U: ✅ GPU ready, TTL: 30min
    
    Note over U,BR: User chats with GPU backend
    
    U->>K: !gpu off
    K->>RP: Terminate pod
    K->>BR: switch_backend(CLOUD)
    K-->>U: ✅ GPU released
```

---

## Slide 9: Model Commands

```mermaid
graph LR
    subgraph model_cmds["Model Commands"]
        LIST["!list-models"]
        USE["!use-model <name>"]
        STATUS["!model-status"]
    end
    
    subgraph gpu_cmds["GPU Commands"]
        ON["!gpu on [model] [ttl]"]
        OFF["!gpu off"]
        GPU_STAT["!gpu status"]
    end
    
    style model_cmds fill:#1a5276,stroke:#85c1e9
    style gpu_cmds fill:#7d3c00,stroke:#f5b041
```

---

## Slide 10: Configuration Example

```toml
# kestrel.toml — [llm] section

[llm]
route_priority = ["openai:api", "anthropic:api", "ollama:local"]

[llm.vendors.openai]
is_cloud = true
[llm.vendors.openai.routes.api]
adapter         = "OpenAIAdapter"
api_key_env     = "OPENAI_API_KEY"   # in .env, never here
model           = "auto"
selection_hints = ["gpt-5", "mini"]

[llm.vendors.anthropic]
is_cloud = true
[llm.vendors.anthropic.routes.api]
adapter         = "AnthropicAdapter"
api_key_env     = "ANTHROPIC_API_KEY"
model           = "auto"
selection_hints = ["sonnet", "haiku"]

[llm.vendors.ollama]
is_cloud = false
[llm.vendors.ollama.routes.local]
adapter         = "OllamaAdapter"
host            = "http://localhost:11434"
model           = "auto"
selection_hints = ["llama3.2", "qwen"]
```

```mermaid
graph LR
    CONFIG["kestrel.toml [llm]"] --> SERVICE[LLMService]
    SERVICE --> ADAPTERS[Initialized Adapters]
    ADAPTERS --> READY[Ready to generate]
    
    style CONFIG fill:#7d6608,stroke:#f4d03f
```

**Vendor/route/model.** Vendors group routes; routes carry adapter + auth. `model = "auto"` resolves via discovery from `selection_hints` — never hardcode IDs. API keys live in `.env`, referenced by `api_key_env`.

---

## Slide 11: Model Discovery - Parallel Scanning

```mermaid
graph TD
    subgraph discovery["ModelDiscoveryMixin"]
        TRIGGER["discover_all_models()"]
        PARALLEL["Parallel adapter calls"]
    end

    subgraph adapters["Adapters (concurrent)"]
        OAI["OpenAI.list_models()"]
        ANT["Anthropic.list_models()"]
        OLLAMA["Ollama.list_models()"]
        GOOGLE["Google.list_models()"]
    end

    subgraph result["Unified Result"]
        MERGE["Merge + deduplicate"]
        FILTER["Apply visibility filters"]
        CATALOG["Model catalog"]
    end

    TRIGGER --> PARALLEL --> adapters
    adapters --> MERGE --> FILTER --> CATALOG

    style discovery fill:#1a5276,stroke:#85c1e9,stroke-width:2px
    style PARALLEL fill:#7d6608,stroke:#f4d03f
```

**Fast discovery.** Query all providers simultaneously.

---

## Slide 12: ModelInfo & Model Catalog

```mermaid
graph TD
    subgraph modelinfo["ModelInfo Dataclass"]
        ID["id: 'gpt-4o'"]
        PROVIDER["provider: 'openai'"]
        DISPLAY["display_name: 'GPT-4o'"]
        CATEGORY["category: CHAT | EMBEDDING | IMAGE"]
        FEATURED["is_featured: bool"]
        HIDDEN["is_hidden: bool"]
    end

    subgraph catalog["ModelCatalogService"]
        TOML["model_catalog.toml"]
        FEATURED_LIST["Featured models list"]
        OVERRIDES["Display name overrides"]
    end

    subgraph api["API Endpoint"]
        ENDPOINT["/api/models"]
        PARAMS["?featured_only=true<br/>?category=chat<br/>?providers=openai,ollama"]
    end

    modelinfo --> catalog --> api

    style modelinfo fill:#1a5276,stroke:#85c1e9
    style catalog fill:#7d6608,stroke:#f4d03f
    style ENDPOINT fill:#145a32,stroke:#58d68d
```

**Curated catalog.** Surface the best models, hide the noise.

---

## Slide 13: Model Selector UI Flow

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Sovereign Console
    participant API as /api/models
    participant LLM as LLMService

    U->>UI: Open model selector
    UI->>API: GET /api/models?featured_only=true
    API->>LLM: discover_all_models()
    LLM-->>API: List[ModelInfo]
    API-->>UI: Featured models

    U->>UI: Toggle "Show All"
    UI->>API: GET /api/models
    API-->>UI: All models

    U->>UI: Select "claude-3-5-sonnet"
    UI->>API: POST /api/model/select
    API->>LLM: set_model("claude-3-5-sonnet")
    LLM-->>API: OK
    API-->>UI: Model changed
```

---

*Next: [07-economics.md](07-economics.md) - Wallet and economic system*
