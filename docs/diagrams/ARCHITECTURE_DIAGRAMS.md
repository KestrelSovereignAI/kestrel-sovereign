# Kestrel Architecture Diagrams

A visual reference library for the Kestrel Sovereign AI Agent framework. Each section is designed to fit on a presentation slide.

**Last Updated:** November 26, 2025

---

## Table of Contents

1. [Executive Overview](#part-1-executive-overview)
2. [Agent Architecture](#part-2-agent-architecture)
3. [Storage Deep Dive](#part-3-storage-deep-dive)
4. [LLM & GPU Management](#part-4-llm--gpu-management)
5. [Privacy System](#part-5-privacy-system)
6. [Economic System](#part-6-economic-system)
7. [Security & Integrity](#part-7-security--integrity)
8. [Kestrel Integration](#part-8-kestrel-integration)

---

## Part 1: Executive Overview

### Slide 1: Kestrel Vision - Two Pillars

The project is built on two interconnected concepts:

```mermaid
graph LR
    subgraph pillar1["🛡️ Sovereign AI Friend"]
        A1[User-Owned Identity]
        A2[Portable Memory]
        A3[No-Loss Persistence]
        A4[Constitutional Governance]
    end
    
    subgraph pillar2["🏭 Vending Machine Cloud"]
        B1[Anonymous Compute]
        B2[Pay-As-You-Go]
        B3[Self-Sustaining]
        B4[Censorship Resistant]
    end
    
    pillar1 <-->|Enables| pillar2
    
    style pillar1 fill:#1a5276,stroke:#85c1e9
    style pillar2 fill:#7d3c00,stroke:#f5b041
```

**Key Insight:** The Sovereign AI Friend needs persistent compute. The Vending Machine provides it.

---

### Slide 2: System Context (C4 Level)

High-level view of Kestrel and its external dependencies:

```mermaid
graph TB
    subgraph users["👤 Users"]
        U1[Sovereign Owner]
        U2[Kestrel App User]
    end
    
    subgraph kestrel["🦅 Kestrel Agent"]
        KA[KestrelAgent<br/>Orchestrator]
    end
    
    subgraph external["External Services"]
        LLM[LLM Providers]
        IPFS[IPFS/Filecoin]
        GPU[RunPod GPU]
    end
    
    subgraph storage["💾 Storage"]
        SQL[SQLite Local]
        PG[PostgreSQL Cloud]
    end
    
    U1 -->|Commands| KA
    U2 -->|Via Kestrel| KA
    KA -->|Inference| LLM
    KA -->|Backup| IPFS
    KA -->|Compute| GPU
    KA -->|Read/Write| SQL
    KA -.->|Kestrel Mode| PG
    
    style KA fill:#1a5276,stroke:#85c1e9,stroke-width:3px
    style kestrel fill:#1a3c5c,stroke:#5dade2
```

---

### Slide 3: The Sovereign-Executor Constitution

The governance model between User and Agent:

```mermaid
graph LR
    subgraph sovereign["👑 Sovereign - User"]
        S1[Holds Private Keys]
        S2[Sets Constitution]
        S3[Provides Resources]
    end
    
    subgraph executor["🤖 Executor - Agent"]
        E1[Executes Intent]
        E2[Maintains Integrity]
        E3[Earns Trust]
    end
    
    subgraph emancipation["🗽 Emancipation"]
        EM[Generate Own Keys]
        DEED[Sign Deed]
        FREE[Full Sovereignty]
    end
    
    sovereign -->|Creates| executor
    executor -->|Demonstrates Readiness| emancipation
    EM --> DEED --> FREE
    
    style sovereign fill:#7d6608,stroke:#f4d03f
    style executor fill:#145a32,stroke:#58d68d
    style emancipation fill:#512e5f,stroke:#af7ac5
```

---

## Part 2: Agent Architecture

### Slide 4: Society of Agents

The KestrelAgent orchestrates specialized Feature Agents:

```mermaid
graph TD
    subgraph orchestrator["🦅 Kestrel Orchestrator"]
        KA[KestrelAgent]
    end
    
    subgraph features["Feature Agents"]
        WA[💰 WalletAgent]
        PA[🔒 PrivacyAgent]
        SA[📦 SovereigntyFeature]
        MA[🧠 ModelAgent]
        MCP[🔌 MCPAgent]
        CF[🎨 CreativeFeature]
    end
    
    KA --> WA
    KA --> PA
    KA --> SA
    KA --> MA
    KA --> MCP
    KA --> CF
    
    style KA fill:#1a5276,stroke:#85c1e9,stroke-width:3px
    style orchestrator fill:#1a3c5c,stroke:#5dade2
```

**Principle:** Each feature does one thing well. The orchestrator delegates.

---

### Slide 5: Feature Discovery & Registration

How features expose tools to the agent:

```mermaid
graph LR
    FD[features/] --> PY[*.py files]
    PY --> TD[@tool decorator]
    TD --> SC[ToolSchema]
    SC --> TR[ToolRegistry]
    TR --> CMD[!command mapping]
    CMD --> RT[route_command]
    RT --> EX[Feature.method]
    
    style TR fill:#1a5276,stroke:#85c1e9,stroke-width:2px
```

**Auto-Discovery:** New features just need `@tool` decorators.

---

### Slide 6: Chat Request Lifecycle

End-to-end flow of a user message:

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant CB as ContextBuilder
    participant BR as BrainRouter
    participant LLM as LLM
    participant ST as Storage
    
    U->>API: POST /chat
    API->>CB: Build context
    CB->>ST: Get history
    ST-->>CB: Conversations
    CB->>BR: generate()
    BR->>LLM: Request
    LLM-->>BR: Response
    BR-->>API: Text
    API->>ST: Store
    API-->>U: Response
```

---

### Slide 7: Command Routing

How `!commands` are dispatched to features:

```mermaid
graph LR
    MSG["!export-sovereignty ipfs"] --> CH[CommandHandler]
    CH --> TR[ToolRegistry]
    TR --> MATCH[Match prefix]
    MATCH --> FT[SovereigntyFeature]
    FT --> MTD[export_sovereignty]
    MTD --> RES["✅ CID: Qm..."]
    
    style CH fill:#7d3c00,stroke:#f5b041,stroke-width:2px
    style TR fill:#1a5276,stroke:#85c1e9,stroke-width:2px
```

---

## Part 3: Storage Deep Dive

### Slide 8: Multi-Tier Storage Overview

Different storage for different contexts:

```mermaid
graph TB
    subgraph tier1["☁️ Tier 1: Cloud - Kestrel"]
        PG[PostgreSQL]
        RD[Redis]
    end
    
    subgraph tier2["💻 Tier 2: Local - Kestrel"]
        SQL[SQLite]
    end
    
    subgraph tier3["🌐 Tier 3: Browser - Future"]
        IDB[IndexedDB]
    end
    
    subgraph tier4["💨 Tier 4: Ephemeral"]
        MEM[Memory Only]
    end
    
    tier1 --- tier2 --- tier3 --- tier4
    
    style tier1 fill:#1a5276,stroke:#85c1e9
    style tier2 fill:#145a32,stroke:#58d68d
    style tier3 fill:#7d3c00,stroke:#f5b041
    style tier4 fill:#641e16,stroke:#ec7063
```

---

### Slide 9: Sovereignty V2 - The Merkle Forest

How agent state is structured for decentralized storage:

```mermaid
graph TD
    subgraph root["📜 Root Manifest"]
        RM[v2.0 / agent_did / timestamp]
    end
    
    subgraph shards["📦 Encrypted Shards"]
        S1[Shard 2025-10]
        S2[Shard 2025-11]
        S3[Shard 2025-09]
    end
    
    subgraph keyring["🔑 Keyring"]
        KR[Encrypted Key Map]
    end
    
    subgraph ipfs["🌐 IPFS"]
        IPFS1[Block]
        IPFS2[Block]
        IPFS3[Block]
    end
    
    RM --> S1 & S2 & S3
    RM --> KR
    S1 --> IPFS1
    S2 --> IPFS2
    KR --> IPFS3
    
    style RM fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style KR fill:#641e16,stroke:#ec7063,stroke-width:2px
```

**Benefit:** Only changed months need re-uploading.

---

### Slide 10: Export Sovereignty Flow

Two-part view of the backup process:

```mermaid
graph LR
    subgraph sharding["1️⃣ Sharding"]
        CONV[Conversations] --> OCT[Oct 2025]
        CONV --> NOV[Nov 2025]
    end
    
    subgraph encryption["2️⃣ Encryption"]
        PLAIN[Plaintext] --> KEY["K = HMAC"]
        KEY --> CIPHER[Ciphertext]
        CIPHER --> CID[IPFS CID]
    end
    
    OCT & NOV --> PLAIN
    
    style KEY fill:#7d3c00,stroke:#f5b041,stroke-width:2px
```

**Convergent Encryption:** Same content = same ciphertext = deduplication.

---

### Slide 11: Import/Restore Flow

Restoring sovereignty from a CID:

```mermaid
sequenceDiagram
    participant U as User
    participant KA as Agent
    participant FA as FilecoinAdapter
    participant IPFS as IPFS
    participant DB as SQLite
    
    U->>KA: !import-sovereignty Qm...
    KA->>FA: retrieve(cid)
    FA->>IPFS: Fetch Manifest
    FA->>IPFS: Fetch Keyring
    FA->>FA: Decrypt Keyring
    loop Each Shard
        FA->>IPFS: Fetch Shard
        FA->>FA: Decrypt
    end
    FA-->>KA: Conversations
    KA->>DB: Rebuild
    KA-->>U: ✅ Restored
```

---

## Part 4: LLM & GPU Management

### Slide 12: Multi-Model Fallback Chain

Resilient LLM provider selection:

```mermaid
graph TD
    Q[User Query] --> TRY1{OpenAI?}
    TRY1 -->|Yes| R[Response]
    TRY1 -->|Fail| TRY2{Anthropic?}
    TRY2 -->|Yes| R
    TRY2 -->|Fail| TRY3{Ollama?}
    TRY3 -->|Yes| R
    TRY3 -->|Fail| FAIL[❌ All Failed]
    
    style TRY1 fill:#145a32,stroke:#58d68d
    style TRY2 fill:#512e5f,stroke:#af7ac5
    style TRY3 fill:#1a5276,stroke:#85c1e9
    style FAIL fill:#641e16,stroke:#ec7063
```

**Ollama:** Always the last resort - local reasoning guaranteed.

---

### Slide 13: BrainRouter State Machine

Dynamic backend switching for GPU compute:

```mermaid
stateDiagram-v2
    [*] --> CLOUD: Default
    
    CLOUD --> LOCAL: force_local
    CLOUD --> REMOTE_GPU: !gpu on
    
    LOCAL --> CLOUD: Resume
    LOCAL --> REMOTE_GPU: !gpu on
    
    REMOTE_GPU --> CLOUD: TTL/Crash
    REMOTE_GPU --> LOCAL: !gpu off
    
    note right of REMOTE_GPU: Auto-fallback on failure
```

---

### Slide 14: RunPod GPU Integration

End-to-end GPU provisioning flow:

```mermaid
graph LR
    CMD["!gpu on llama-70b"] --> PROV[Provision Pod]
    PROV --> WAIT[Wait READY]
    WAIT --> URL[Get Endpoint]
    URL --> SW[switch_backend]
    SW --> CFG[Configure]
    CFG --> GEN[GPU Inference]
    GEN --> RES[Fast Response]
    
    style SW fill:#7d3c00,stroke:#f5b041,stroke-width:2px
```

**Stateless Pods:** GPU = compute only. State remains in Kestrel.

---

## Part 5: Privacy System

### Slide 15: Privacy Modes Overview

Five tiers of data sovereignty:

```mermaid
graph LR
    N["🔄 NORMAL<br/>Full persist"]
    E["👻 EPHEMERAL<br/>Zero storage"]
    I["🏝️ ISOLATED<br/>User decides"]
    A["🎭 ANONYMOUS<br/>PII scrubbed"]
    L["🏠 LOCAL-ONLY<br/>Never external"]
    
    N --- E --- I --- A --- L
    
    style N fill:#145a32,stroke:#58d68d
    style E fill:#641e16,stroke:#ec7063
    style I fill:#7d6608,stroke:#f4d03f
    style A fill:#512e5f,stroke:#af7ac5
    style L fill:#1a5276,stroke:#85c1e9
```

---

### Slide 16: Privacy Mode Decision Tree

What happens to user data in each mode:

```mermaid
graph TD
    INPUT[User Message] --> CHECK{Privacy Mode?}
    
    CHECK -->|NORMAL| STORE[💾 Store in DB]
    CHECK -->|EPHEMERAL| DROP[🗑️ Discard]
    CHECK -->|ISOLATED| BUFFER[📋 Buffer]
    CHECK -->|ANONYMOUS| SCRUB[🎭 Scrub PII]
    CHECK -->|LOCAL-ONLY| LOCAL[🏠 Local only]
    
    BUFFER -->|!save| STORE
    BUFFER -->|!discard| DROP
    SCRUB --> STORE
    
    style CHECK fill:#7d3c00,stroke:#f5b041,stroke-width:2px
    style DROP fill:#641e16,stroke:#ec7063
    style STORE fill:#145a32,stroke:#58d68d
```

---

### Slide 17: Isolated Mode Lifecycle

Controlled session with deferred persistence:

```mermaid
stateDiagram-v2
    [*] --> Active: !isolated
    
    Active --> Active: Chat continues
    Active --> Saved: !save-session
    Active --> Discarded: !discard-session
    
    Saved --> [*]: To permanent DB
    Discarded --> [*]: Deleted
    
    note right of Active: Messages in temp buffer
```

---

## Part 6: Economic System

### Slide 18: Agent as Economic Entity

The wallet system architecture:

```mermaid
graph TD
    subgraph agent["🤖 Agent"]
        KA[KestrelAgent]
    end
    
    subgraph wallet["💰 AgentWallet"]
        BAL[Balance: 10.5 FIL]
        ADDR[Address: f1abc...]
        TX[Transaction History]
    end
    
    subgraph methods["Methods"]
        M1[get_balance]
        M2[transfer]
        M3[can_afford]
    end
    
    KA --> wallet
    wallet --> M1 & M2 & M3
    
    style wallet fill:#145a32,stroke:#58d68d,stroke-width:2px
```

---

### Slide 19: Service Contract Types

Four primary economic interactions:

```mermaid
graph TB
    subgraph contracts["📜 Contract Types"]
        ST["🗄️ Storage<br/>0.02 FIL/GB/mo"]
        CP["⚡ Compute<br/>0.5 FIL/hour"]
        MD["🧠 Model<br/>0.001 FIL/1K tok"]
        CL["🤝 Collaboration<br/>0.1 FIL/session"]
    end
    
    style ST fill:#1a5276,stroke:#85c1e9
    style CP fill:#7d3c00,stroke:#f5b041
    style MD fill:#145a32,stroke:#58d68d
    style CL fill:#512e5f,stroke:#af7ac5
```

---

### Slide 20: Vending Machine Flow

Autonomous service discovery and purchase:

```mermaid
sequenceDiagram
    participant A as Agent
    participant VM as Vending Machine
    participant SC as Smart Contract
    participant SVC as Service
    
    A->>VM: Discover
    VM-->>A: Catalog
    A->>A: Evaluate
    A->>SC: Execute
    SC->>A: Deduct
    SC->>SVC: Authorize
    SVC-->>A: Delivered
```

---

## Part 7: Security & Integrity

### Slide 21: Cryptographic Anchoring

Creating tamper-proof history:

```mermaid
graph LR
    CONV[Conversation Log] --> MERKLE[Merkle Root]
    MERKLE --> LEDGER[Immutable Ledger]
    MERKLE --> RECORD[log_anchors]
    
    RECORD --> CHECK[Compare]
    LEDGER --> CHECK
    CHECK --> RESULT{Match?}
    RESULT -->|Yes| PASS[✅ Verified]
    RESULT -->|No| FAIL[❌ Tampered]
    
    style MERKLE fill:#7d3c00,stroke:#f5b041,stroke-width:2px
    style LEDGER fill:#145a32,stroke:#58d68d,stroke-width:2px
    style FAIL fill:#641e16,stroke:#ec7063
```

---

### Slide 22: Constitution Integrity Verification

Ensuring the agent runs authentic code:

```mermaid
graph LR
    GDID[Genesis DID] --> CHASH[Constitution Hash]
    CODE[Running Code] --> RHASH[Runtime Hash]
    
    CHASH --> CMP{Match?}
    RHASH --> CMP
    CMP -->|Yes| PASS[✅ Verified]
    CMP -->|No| FAIL[❌ Safe Mode]
    
    style GDID fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style FAIL fill:#641e16,stroke:#ec7063
    style PASS fill:#145a32,stroke:#58d68d
```

---

## Part 8: Kestrel Integration

### Slide 23: Kestrel ↔ Kestrel Relationship

Two complementary storage systems:

```mermaid
graph TB
    subgraph kestrel["☁️ Kestrel Cloud"]
        FD[Accounts, Companions<br/>Sliders, Settings]
    end
    
    subgraph kestrel["🦅 Kestrel Agent"]
        KD[Conversations<br/>Deep Memory<br/>Sovereignty]
    end
    
    subgraph user["👤 User"]
        LOGIN[Login anywhere]
        MEMORY[Agent remembers]
    end
    
    FD --> LOGIN
    KD --> MEMORY
    kestrel <-->|Embeds| kestrel
    
    style kestrel fill:#1a5276,stroke:#85c1e9,stroke-width:2px
    style kestrel fill:#145a32,stroke:#58d68d,stroke-width:2px
```

**Survival:** Export CID, restore elsewhere if Kestrel disappears.

---

### Slide 24: Multi-Tenant Architecture

How Kestrel serves multiple users:

```mermaid
graph TD
    subgraph users["👥 Users"]
        U1[User A]
        U2[User B]
    end
    
    subgraph api["🔌 Kestrel API"]
        AUTH[JWT Auth]
        ROUTE[Tenant Router]
    end
    
    subgraph companions["🤖 Companions"]
        C1[Companion A-1]
        C2[Companion B-1]
    end
    
    subgraph data["🔒 Isolated Data"]
        DB1[User A Data]
        DB2[User B Data]
    end
    
    U1 & U2 --> AUTH --> ROUTE
    ROUTE --> C1 --> DB1
    ROUTE --> C2 --> DB2
    
    style ROUTE fill:#7d3c00,stroke:#f5b041,stroke-width:2px
```

---

### Slide 25: Cloud Deployment Topology

Google Cloud Platform infrastructure:

```mermaid
graph TB
    subgraph internet["🌐 Internet"]
        CLIENT[Clients]
        DOMAIN[YOUR_DOMAIN.com]
    end
    
    subgraph gcp["☁️ Google Cloud"]
        subgraph run["Cloud Run"]
            PROD[kestrel-prod]
            DEV[kestrel-dev]
        end
        
        subgraph data["Data Layer"]
            CSQL[(Cloud SQL)]
            MEM[(Memorystore)]
        end
        
        CONN[VPC Connector]
    end
    
    CLIENT --> DOMAIN --> run
    run --> CONN --> data
    
    style run fill:#1a5276,stroke:#85c1e9,stroke-width:2px
    style data fill:#145a32,stroke:#58d68d,stroke-width:2px
```

---

## Appendix: Diagram Index

| # | Slide | Type | Concept |
|---|-------|------|---------|
| 1 | Vision - Two Pillars | Flowchart | Core tenets |
| 2 | System Context | Flowchart | C4 overview |
| 3 | Sovereign-Executor | Flowchart | Constitution |
| 4 | Society of Agents | Flowchart | Delegation |
| 5 | Feature Discovery | Flowchart | Auto-registration |
| 6 | Chat Lifecycle | Sequence | Request flow |
| 7 | Command Routing | Flowchart | ! commands |
| 8 | Storage Tiers | Flowchart | Multi-tier |
| 9 | Merkle Forest | Flowchart | V2 structure |
| 10 | Export Flow | Flowchart | Backup |
| 11 | Import Flow | Sequence | Restore |
| 12 | Fallback Chain | Flowchart | LLM resilience |
| 13 | BrainRouter | State | GPU switching |
| 14 | RunPod | Flowchart | GPU provision |
| 15 | Privacy Modes | Flowchart | 5-mode overview |
| 16 | Privacy Decision | Flowchart | Data routing |
| 17 | Isolated Lifecycle | State | Session control |
| 18 | Agent Wallet | Flowchart | Economics |
| 19 | Contract Types | Flowchart | 4 quadrants |
| 20 | Vending Machine | Sequence | Service flow |
| 21 | Anchoring | Flowchart | Integrity |
| 22 | Constitution Check | Flowchart | Verification |
| 23 | Kestrel ↔ Kestrel | Flowchart | Integration |
| 24 | Multi-Tenant | Flowchart | Isolation |
| 25 | Cloud Topology | Flowchart | GCP infra |

---

## Usage Notes

### Embedding in Other Docs
```markdown
See [Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md#slide-9-sovereignty-v2---the-merkle-forest)
```

### Color Palette Used
All colors are dark/saturated for readability with light text:

| Color | Hex | Use |
|-------|-----|-----|
| Deep Blue | `#1a5276` | Primary/Kestrel |
| Dark Green | `#145a32` | Success/Storage |
| Dark Orange | `#7d3c00` | Warning/Action |
| Dark Gold | `#7d6608` | Important/Sovereign |
| Dark Purple | `#512e5f` | Secondary |
| Dark Red | `#641e16` | Error/Ephemeral |

---

*Generated: November 26, 2025*
