---
type: Diagram
title: 'DA-08: Privacy & Encryption'
description: Encryption at rest, PII scrubbing, and privacy mode enforcement.
resource: /docs/diagrams/data-architecture/DA-08-privacy-encryption.md
tags:
- docs
- diagrams
- diagram
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: documentation
canonical: false
generated: false
privacy: public
---

# DA-08: Privacy & Encryption

Encryption at rest, PII scrubbing, and privacy mode enforcement.

---

## Slide 1: Encryption at Rest

```mermaid
graph LR
    subgraph key["KESTREL_DATA_KEY"]
        FERNET["Fernet (AES-128-CBC)"]
        ENV["From environment variable"]
    end

    subgraph encrypted["Encrypted Data"]
        FILES["📁 File blobs"]
        CONV["💬 Conversations"]
        MEMORY["🧠 Memories"]
    end

    key --> encrypted

    style key fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style encrypted fill:#512e5f,stroke:#af7ac5
```

**Set KESTREL_DATA_KEY.** Everything encrypted transparently.

---

## Slide 2: What Gets Encrypted

| Component | Encrypted | How |
|-----------|-----------|-----|
| File blobs | ✅ Yes | Fernet before storage |
| Conversations | ✅ Yes | Fernet on content field |
| Memories | ✅ Yes | Fernet on content field |
| Knowledge graph | ⚠️ Partial | Sensitive nodes only |
| Wallet state | ✅ Yes | Fernet on balances |
| Embeddings | ❌ No | Vectors not sensitive |

```mermaid
graph TD
    SENSITIVE["Sensitive data"] --> ENCRYPT["Fernet encrypt"]
    VECTORS["Vector embeddings"] --> PLAIN["Store as-is"]

    style ENCRYPT fill:#512e5f,stroke:#af7ac5
    style PLAIN fill:#1a5276,stroke:#85c1e9
```

---

## Slide 3: PrivacyEnforcingStorage Wrapper

```mermaid
graph TD
    subgraph wrapper["PrivacyEnforcingStorage"]
        CHECK["Check privacy mode"]
        FILTER["Apply storage policy"]
        DELEGATE["Delegate to backend"]
    end

    subgraph policies["Policies"]
        EPH["EPHEMERAL: Block all storage"]
        ISO["ISOLATED: Temp buffer only"]
        ANON["ANONYMOUS: Scrub PII first"]
        NORM["NORMAL: Full storage"]
    end

    wrapper --> policies

    style wrapper fill:#7d6608,stroke:#f4d03f,stroke-width:2px
```

**Policy enforcement at storage boundary.**

---

## Slide 4: PII Scrubbing (ANONYMOUS Mode)

```mermaid
graph LR
    subgraph input["Input"]
        MSG["'Call John at 555-1234<br/>and email john@test.com'"]
    end

    subgraph scrub["spaCy NER"]
        DETECT["Detect entities"]
        REPLACE["Replace with placeholders"]
    end

    subgraph output["Output"]
        CLEAN["'Call NAME_REDACTED at<br/>PHONE_REDACTED and email<br/>EMAIL_REDACTED'"]
    end

    MSG --> DETECT --> REPLACE --> CLEAN

    style DETECT fill:#7d6608,stroke:#f4d03f
    style CLEAN fill:#145a32,stroke:#58d68d
```

**spaCy NER.** Detects names, orgs, emails, phones, SSNs.

---

## Slide 5: Privacy Mode → Storage Rules

```mermaid
graph TD
    subgraph modes["Privacy Mode"]
        EPH["👻 EPHEMERAL"]
        ISO["🏝️ ISOLATED"]
        ANON["🎭 ANONYMOUS"]
        NORM["🔄 NORMAL"]
        PUB["📖 PUBLIC"]
    end

    subgraph rules["Storage Rules"]
        NONE["Store nothing"]
        TEMP["Temp buffer (session)"]
        SCRUB["Scrub PII first"]
        FULL["Full storage"]
        SHARE["Full + shareable"]
    end

    EPH --> NONE
    ISO --> TEMP
    ANON --> SCRUB
    NORM --> FULL
    PUB --> SHARE

    style NONE fill:#641e16,stroke:#ec7063
    style TEMP fill:#7d6608,stroke:#f4d03f
    style SCRUB fill:#512e5f,stroke:#af7ac5
    style FULL fill:#145a32,stroke:#58d68d
    style SHARE fill:#1a5276,stroke:#85c1e9
```

---

## Slide 6: Security Guarantees

| Property | Guarantee | How |
|----------|-----------|-----|
| **Confidentiality** | ✅ Data unreadable without key | Fernet encryption |
| **Integrity** | ✅ Tampering detected | HMAC in Fernet |
| **Availability** | ✅ Data survives platform loss | IPFS/Filecoin backup |
| **Privacy** | ✅ PII removable on demand | Scrubbing pipeline |
| **Sovereignty** | ✅ User controls all keys | No key escrow |

```mermaid
graph LR
    CIA["CIA Triad"] --> CONF["Confidentiality ✅"]
    CIA --> INT["Integrity ✅"]
    CIA --> AVAIL["Availability ✅"]

    style CONF fill:#145a32,stroke:#58d68d
    style INT fill:#145a32,stroke:#58d68d
    style AVAIL fill:#145a32,stroke:#58d68d
```

---

*End of Data Architecture Deep Dive series*

*Return to [README.md](README.md) for series index*
