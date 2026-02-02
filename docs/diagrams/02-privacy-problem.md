# 02 - The Privacy Problem

Why current AI platforms like ChatGPT fail you, and why data sovereignty matters.

---

## Slide 1: How ChatGPT Works Today

```mermaid
graph TB
    subgraph you["👤 You"]
        MSG[Your message]
        HIST[Your history]
        MEM[Your memories]
    end
    
    subgraph openai["🏢 OpenAI Servers"]
        API[Their API]
        DB[(Their Database)]
        TRAIN[Their Training]
    end
    
    MSG --> API
    API --> DB
    DB --> TRAIN
    HIST --> DB
    MEM --> DB
    
    style you fill:#641e16,stroke:#ec7063
    style openai fill:#0e4d45,stroke:#45b39d
```

**Reality:** Every word you type lives on their servers, forever.

---

## Slide 2: What OpenAI Stores

```mermaid
graph LR
    subgraph stored["📦 Stored by OpenAI"]
        C1[Every conversation]
        C2[Every prompt]
        C3[Your memories]
        C4[Your preferences]
        C5[Usage patterns]
        C6[IP addresses]
    end
    
    subgraph control["🔒 Your Control"]
        X1[❌ Can't truly delete]
        X2[❌ Can't export all]
        X3[❌ Can't move to competitor]
        X4[❌ Can't run locally]
    end
    
    stored --> control
    
    style stored fill:#641e16,stroke:#ec7063
    style control fill:#7d3c00,stroke:#f5b041
```

---

## Slide 3: The Memory Feature Trap

```mermaid
graph TD
    subgraph chatgpt["ChatGPT Memory"]
        MEM[Memories enabled]
        SAVE[AI saves facts about you]
        PERS[Personalizes responses]
    end
    
    subgraph reality["The Reality"]
        R1[Stored on OpenAI servers]
        R2[Used for training]
        R3[You can't export]
        R4[Deletion is fuzzy]
    end
    
    subgraph risk["Risks"]
        LEAK[Data breaches]
        POLICY[Policy changes]
        SHUTDOWN[Service shutdown]
        CENSOR[Content moderation]
    end
    
    chatgpt --> reality --> risk
    
    style chatgpt fill:#0e4d45,stroke:#45b39d
    style reality fill:#7d3c00,stroke:#f5b041
    style risk fill:#641e16,stroke:#ec7063
```

**Your "personal AI" isn't personal at all.**

---

## Slide 4: Who Really Owns Your AI Relationship?

```mermaid
graph LR
    subgraph chatgpt_model["ChatGPT Model"]
        THEM[OpenAI owns:]
        T1[The model]
        T2[Your data]
        T3[The memories]
        T4[The relationship]
    end
    
    subgraph kestrel_model["Kestrel Model"]
        YOU[You own:]
        Y1[Your keys]
        Y2[Your data]
        Y3[Your memories]
        Y4[The relationship]
    end
    
    style chatgpt_model fill:#641e16,stroke:#ec7063
    style kestrel_model fill:#145a32,stroke:#58d68d
```

---

## Slide 5: The Kestrel Difference

```mermaid
graph TB
    subgraph traditional["❌ Traditional AI"]
        T_YOU[You] --> T_CLOUD[Their Cloud]
        T_CLOUD --> T_DB[(Their DB)]
        T_DB --> T_TRAIN[Their Training]
    end
    
    subgraph sovereign["✅ Kestrel Sovereign AI"]
        K_YOU[You] --> K_AGENT[Your Agent]
        K_AGENT --> K_LOCAL[(Your Storage)]
        K_LOCAL --> K_BACKUP[Your Backup]
        K_AGENT -.->|Inference only| K_LLM[LLM API]
    end
    
    style traditional fill:#641e16,stroke:#ec7063
    style sovereign fill:#145a32,stroke:#58d68d
```

**Key:** LLM sees your prompt, but doesn't keep your history.

---

## Slide 6: What Kestrel Guarantees

```mermaid
graph LR
    subgraph guarantees["🛡️ Kestrel Guarantees"]
        G1[Data stays with you]
        G2[Export anytime]
        G3[No training on your data]
        G4[Choose your LLM]
        G5[Run fully offline]
    end
    
    subgraph how["How"]
        H1[Local SQLite DB]
        H2[Encrypted IPFS backup]
        H3[Your encryption keys]
        H4[Multi-model support]
        H5[Ollama local option]
    end
    
    G1 --> H1
    G2 --> H2
    G3 --> H3
    G4 --> H4
    G5 --> H5
    
    style guarantees fill:#145a32,stroke:#58d68d,stroke-width:2px
```

---

## Slide 7: Comparison Summary

| Feature | ChatGPT | Kestrel |
|---------|---------|---------|
| Data location | OpenAI servers | Your device/cloud |
| Export | Limited | Full sovereignty |
| Training | Yes (default) | Never |
| Offline mode | No | Yes (Ollama) |
| Model choice | GPT only | Any LLM |
| Delete data | Uncertain | Cryptographic |
| Survives shutdown | No | Yes |

```mermaid
graph LR
    CHATGPT[ChatGPT] --> LOCKED[🔒 Locked In]
    KESTREL[Kestrel] --> FREE[🔓 Sovereign]
    
    style CHATGPT fill:#641e16,stroke:#ec7063
    style KESTREL fill:#145a32,stroke:#58d68d
    style LOCKED fill:#7d3c00,stroke:#f5b041
    style FREE fill:#145a32,stroke:#58d68d
```

---

*Next: [03-privacy-modes.md](03-privacy-modes.md) - Kestrel's 5-tier privacy system*
