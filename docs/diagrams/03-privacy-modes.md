# 03 - Privacy Modes

Kestrel's privacy system uses **independent flags** with **named presets** for complete control.

---

## Slide 1: Privacy as Independent Flags

```mermaid
graph TD
    subgraph flags["Three Independent Concerns"]
        S["📦 Storage<br/>none | temp | scrubbed | full"]
        L["🧠 LLM Location<br/>local | cloud"]
        H["📤 Shareable<br/>true | false"]
    end
    
    subgraph presets["Named Presets"]
        E["👻 ephemeral"]
        I["🏝️ isolated"]
        A["🎭 anonymous"]
        N["🔄 normal"]
        P["📖 public"]
    end
    
    flags --> presets
    
    style S fill:#1a5276,stroke:#85c1e9
    style L fill:#512e5f,stroke:#af7ac5
    style H fill:#7d6608,stroke:#f4d03f
```

**Presets are convenience.** Flags are the truth.

---

## Slide 2: The Five Presets

| Preset | Storage | LLM | Shareable | Purpose |
|--------|---------|-----|-----------|---------|
| 👻 ephemeral | none | local | no | Maximum privacy - nothing stored, local only |
| 🏝️ isolated | temp | local | no | Session sandbox - decide later |
| 🎭 anonymous | scrubbed | cloud | no | Learn from me, don't know me |
| 🔄 normal | full | cloud | no | Standard operation |
| 📖 public | full | cloud | yes | Transparent/auditable agent |

```mermaid
graph LR
    E["👻 ephemeral"] --> I["🏝️ isolated"] --> A["🎭 anonymous"] --> N["🔄 normal"] --> P["📖 public"]
    
    style E fill:#641e16,stroke:#ec7063
    style I fill:#7d6608,stroke:#f4d03f
    style A fill:#512e5f,stroke:#af7ac5
    style N fill:#145a32,stroke:#58d68d
    style P fill:#1a5276,stroke:#85c1e9
```

---

## Slide 3: Normal Mode - Full Features

```mermaid
graph TD
    subgraph normal["🔄 NORMAL Mode"]
        DESC[Standard operation]
        FLAGS["storage=full, llm=cloud, shareable=false"]
    end
    
    subgraph behavior["Behavior"]
        B1[✅ Full conversation history]
        B2[✅ Agent learns & remembers]
        B3[✅ Cloud LLM available]
        B4[✅ Sovereignty backup]
    end
    
    subgraph usecase["Use Cases"]
        U1[Regular work]
        U2[Building knowledge]
        U3[Long-term projects]
    end
    
    normal --> behavior --> usecase
    
    style normal fill:#145a32,stroke:#58d68d,stroke-width:2px
```

**Default mode.** Your agent grows with you.

---

## Slide 4: Ephemeral Mode - Zero Trace

```mermaid
graph TD
    subgraph ephemeral["👻 EPHEMERAL Mode"]
        DESC[Nothing stored anywhere]
        FLAGS["storage=none, llm=local, shareable=false"]
    end
    
    subgraph behavior["Behavior"]
        B1[❌ No storage at all]
        B2[❌ Agent forgets immediately]
        B3[❌ Local LLM only]
        B4[❌ No backup possible]
    end
    
    subgraph usecase["Use Cases"]
        U1[Sensitive topics]
        U2[Off-the-record chat]
        U3[Mental health]
        U4[Confidential business]
    end
    
    ephemeral --> behavior --> usecase
    
    style ephemeral fill:#641e16,stroke:#ec7063,stroke-width:2px
```

**Like it never happened.** Message exists only in memory.

---

## Slide 5: Isolated Mode - Sandbox with Local LLM

```mermaid
graph TD
    subgraph isolated["🏝️ ISOLATED Mode"]
        DESC[Temporary session buffer + local processing]
        FLAGS["storage=temp, llm=local, shareable=false"]
    end
    
    subgraph during["During Session"]
        D1[Messages in temp buffer]
        D2[Local LLM only]
        D3[Nothing permanent yet]
    end
    
    subgraph after["After Session"]
        SAVE["!save-session"]
        DISCARD["!discard-session"]
    end
    
    subgraph result["Result"]
        KEPT[💾 Saved to permanent]
        GONE[🗑️ Deleted forever]
    end
    
    isolated --> during --> after
    SAVE --> KEPT
    DISCARD --> GONE
    
    style isolated fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style KEPT fill:#145a32,stroke:#58d68d
    style GONE fill:#641e16,stroke:#ec7063
```

**Experiment safely.** Local processing, commit only what you want.

---

## Slide 6: Anonymous Mode - Learn Without Identity

```mermaid
graph TD
    subgraph anonymous["🎭 ANONYMOUS Mode"]
        DESC[PII scrubbed before storage]
        FLAGS["storage=scrubbed, llm=cloud, shareable=false"]
    end
    
    subgraph scrubbing["What Gets Scrubbed"]
        S1[Names → NAME_REDACTED]
        S2[Emails → EMAIL_REDACTED]
        S3[Phones → PHONE_REDACTED]
        S4[SSNs → SSN_REDACTED]
    end
    
    subgraph behavior["Behavior"]
        B1[✅ Patterns preserved]
        B2[✅ Learning continues]
        B3[✅ Cloud LLM allowed]
        B4[❌ Personal details removed]
    end
    
    anonymous --> scrubbing --> behavior
    
    style anonymous fill:#512e5f,stroke:#af7ac5,stroke-width:2px
```

**Agent learns how to help you, not who you are.**

---

## Slide 7: Public Mode - Full Transparency

```mermaid
graph TD
    subgraph public["📖 PUBLIC Mode"]
        DESC[Fully auditable and shareable]
        FLAGS["storage=full, llm=cloud, shareable=true"]
    end
    
    subgraph behavior["Behavior"]
        B1[✅ Full conversation history]
        B2[✅ Agent learns & remembers]
        B3[✅ Cloud LLM available]
        B4[✅ Can export & share]
    end
    
    subgraph usecase["Use Cases"]
        U1[Community bots]
        U2[Transparent AI assistants]
        U3[Auditable agents]
    end
    
    public --> behavior --> usecase
    
    style public fill:#1a5276,stroke:#85c1e9,stroke-width:2px
```

**Nothing to hide.** Full transparency for trust.

---

## Slide 8: Privacy Mode Decision Flow

```mermaid
graph TD
    MSG[User sends message] --> CHECK{Current Privacy Config?}
    
    CHECK -->|storage=full| STORE[💾 Store everything]
    CHECK -->|storage=none| DROP[🗑️ Store nothing]
    CHECK -->|storage=scrubbed| SCRUB[🎭 Scrub PII then store]
    CHECK -->|storage=temp| BUFFER[📋 Buffer for later]
    
    BUFFER -->|!save| STORE
    BUFFER -->|!discard| DROP
    SCRUB --> STORE
    
    LLM{llm_location?}
    CHECK --> LLM
    LLM -->|local| LOCAL[🏠 Ollama only]
    LLM -->|cloud| CLOUD[☁️ Any provider]
    
    style CHECK fill:#7d3c00,stroke:#f5b041,stroke-width:2px
    style STORE fill:#145a32,stroke:#58d68d
    style DROP fill:#641e16,stroke:#ec7063
    style LOCAL fill:#1a5276,stroke:#85c1e9
    style CLOUD fill:#512e5f,stroke:#af7ac5
```

---

## Slide 9: Switching Modes - Commands

```mermaid
graph LR
    subgraph commands["Privacy Commands"]
        C1["!normal"]
        C2["!ephemeral"]
        C3["!anonymous"]
        C4["!isolated"]
        C5["!public"]
        C6["!status"]
    end
    
    subgraph effect["Immediate Effect"]
        E1[Next message uses new mode]
        E2[No data loss on switch]
        E3[Can switch anytime]
    end
    
    commands --> effect
    
    style commands fill:#1a5276,stroke:#85c1e9
```

**Instant switching.** Your conversation, your rules.

---

## Slide 10: Custom Configurations

Beyond presets, you can set flags directly:

```python
# Want permanent storage but local-only processing?
agent.set_privacy(PrivacyConfig(
    storage="full",
    llm_location="local",
    shareable=False
))

# Or use a preset
agent.set_privacy_preset("ephemeral")
```

**Presets are named combinations. Flags give you full control.**

---

*Next: [04-feature-framework.md](04-feature-framework.md) - Dynamic feature system architecture*
