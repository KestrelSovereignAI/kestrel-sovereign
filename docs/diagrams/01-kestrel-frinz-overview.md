# 01 - Kestrel & Kestrel Overview

Introduction to the Kestrel Sovereign AI Agent and Kestrel platform.

---

## Slide 1: The Vision - Two Products, One Foundation

```mermaid
graph TB
    subgraph kestrel["🦅 Kestrel - Sovereign AI Agent"]
        K1[Your AI, Your Data]
        K2[Runs Anywhere]
        K3[Open Source Core]
    end
    
    subgraph kestrel["💜 Kestrel - AI Companion Platform"]
        F1[Easy to Use App]
        F2[Cloud Hosted]
        F3[Powered by Kestrel]
    end
    
    kestrel -->|Powers| kestrel
    
    style kestrel fill:#1a5276,stroke:#85c1e9,stroke-width:2px
    style kestrel fill:#512e5f,stroke:#af7ac5,stroke-width:2px
```

**Kestrel** = The engine (sovereign, portable, yours)  
**Kestrel** = The product (friendly, hosted, easy)

---

## Slide 2: What is Kestrel?

```mermaid
graph LR
    subgraph core["🦅 Kestrel Core"]
        ID[Cryptographic Identity]
        MEM[Persistent Memory]
        PRIV[Privacy Controls]
        PORT[Full Portability]
    end
    
    subgraph runs["Runs On"]
        LOCAL[Your Laptop]
        CLOUD[Any Cloud]
        kestrel[Kestrel Platform]
    end
    
    core --> runs
    
    style core fill:#1a5276,stroke:#85c1e9,stroke-width:2px
```

**Key Point:** Same agent, same memory, anywhere you want.

---

## Slide 3: What is Kestrel?

```mermaid
graph LR
    subgraph kestrel["💜 Kestrel Platform"]
        APP[Mobile/Web App]
        HOST[Cloud Hosting]
        COMP[AI Companions]
    end
    
    subgraph features["Features"]
        CHAT[Natural Chat]
        PERS[Personality Sliders]
        VOICE[Voice & Image]
        TRIAL[Free Trial]
    end
    
    kestrel --> features
    
    style kestrel fill:#512e5f,stroke:#af7ac5,stroke-width:2px
```

**Target:** Users who want an AI companion without technical setup.

---

## Slide 4: How They Relate

```mermaid
graph TB
    subgraph user["👤 User"]
        CHOICE{Want simplicity<br/>or control?}
    end
    
    subgraph simple["Easy Path"]
        FAPP[Use Kestrel App]
        FCLOUD[Hosted for you]
    end
    
    subgraph power["Power Path"]
        KLOCAL[Run Kestrel locally]
        KOWN[Own everything]
    end
    
    subgraph always["Either Way"]
        EXPORT[Can always export]
        YOURS[Data is yours]
    end
    
    CHOICE -->|Simplicity| simple
    CHOICE -->|Control| power
    simple --> always
    power --> always
    
    style simple fill:#512e5f,stroke:#af7ac5
    style power fill:#1a5276,stroke:#85c1e9
    style always fill:#145a32,stroke:#58d68d
```

---

## Slide 5: Architecture - Kestrel Uses Kestrel

```mermaid
graph TB
    subgraph kestrel_cloud["☁️ Kestrel Cloud"]
        API[Kestrel API]
        AUTH[Authentication]
        PG[(PostgreSQL)]
        RD[(Redis)]
    end
    
    subgraph kestrel_embedded["🦅 Kestrel Agent"]
        KA[KestrelAgent]
        SQL[(SQLite Memory)]
        IPFS[IPFS Backup]
    end
    
    subgraph llm["🧠 LLM Providers"]
        OAI[OpenAI]
        ANT[Anthropic]
        OLL[Ollama]
    end
    
    API --> AUTH
    AUTH --> KA
    KA --> SQL
    KA --> IPFS
    KA --> llm
    API --> PG
    API --> RD
    
    style kestrel_cloud fill:#512e5f,stroke:#af7ac5
    style kestrel_embedded fill:#1a5276,stroke:#85c1e9
```

**Kestrel stores:** Accounts, settings, billing  
**Kestrel stores:** Conversations, deep memory, personality

---

## Slide 6: Data Ownership Model

```mermaid
graph LR
    subgraph kestrel_data["Kestrel Owns"]
        FD1[Your login]
        FD2[Companion names]
        FD3[Billing info]
        FD4[App settings]
    end
    
    subgraph you_own["You Own"]
        YD1[All conversations]
        YD2[Agent memory]
        YD3[Personality state]
        YD4[Export CID]
    end
    
    kestrel_data -.->|If Kestrel dies| LOST[Lost]
    you_own -->|Always| KEEP[You keep it]
    
    style kestrel_data fill:#7d3c00,stroke:#f5b041
    style you_own fill:#145a32,stroke:#58d68d
    style LOST fill:#641e16,stroke:#ec7063
    style KEEP fill:#145a32,stroke:#58d68d
```

---

## Slide 7: The Sovereignty Promise

```mermaid
graph TD
    subgraph promise["🛡️ Kestrel Promise"]
        P1[Export anytime]
        P2[Import anywhere]
        P3[No lock-in]
        P4[Your keys, your data]
    end
    
    subgraph scenario["If Kestrel Shuts Down"]
        S1[Export your CID]
        S2[Run Kestrel locally]
        S3[Import sovereignty]
        S4[Continue where you left off]
    end
    
    promise --> scenario
    S1 --> S2 --> S3 --> S4
    
    style promise fill:#1a5276,stroke:#85c1e9,stroke-width:2px
    style scenario fill:#145a32,stroke:#58d68d
```

**You are never trapped.**

---

## Slide 8: System Context - Full Picture

```mermaid
graph TB
    subgraph users["👤 Users"]
        FUSER[Kestrel User]
        KUSER[Kestrel Power User]
    end
    
    subgraph kestrel["💜 Kestrel Platform"]
        FAPI[API + Web]
    end
    
    subgraph kestrel["🦅 Kestrel"]
        KA[Agent Core]
    end
    
    subgraph storage["💾 Storage"]
        PG[(Cloud DB)]
        SQL[(Local DB)]
        IPFS[(IPFS/Filecoin)]
    end
    
    subgraph llm["🧠 LLMs"]
        CLOUD[Cloud APIs]
        LOCAL[Ollama Local]
    end
    
    FUSER --> FAPI --> KA
    KUSER --> KA
    KA --> SQL
    KA --> IPFS
    FAPI --> PG
    KA --> llm
    
    style kestrel fill:#512e5f,stroke:#af7ac5
    style kestrel fill:#1a5276,stroke:#85c1e9
```

---

*Next: [02-privacy-problem.md](02-privacy-problem.md) - Why current AI platforms fail you*
