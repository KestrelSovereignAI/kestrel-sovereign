---
type: Diagram
title: 05 - Storage & Sovereignty
description: Kestrel's multi-tier storage and the Sovereignty V2 backup system.
resource: /docs/diagrams/05-storage-sovereignty.md
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

# 05 - Storage & Sovereignty

Kestrel's multi-tier storage and the Sovereignty V2 backup system.

---

## Slide 1: Multi-Tier Storage Architecture

```mermaid
graph TB
    subgraph tier1["☁️ Tier 1: Cloud - Kestrel"]
        PG[(PostgreSQL)]
        RD[(Redis)]
        T1_USE[Accounts, settings, billing]
    end
    
    subgraph tier2["💻 Tier 2: Local - Kestrel"]
        SQL[(SQLite)]
        T2_USE[Conversations, memory]
    end
    
    subgraph tier3["🌐 Tier 3: Decentralized"]
        IPFS[(IPFS/Filecoin)]
        T3_USE[Sovereignty backups]
    end
    
    subgraph tier4["💨 Tier 4: Ephemeral"]
        MEM[Memory only]
        T4_USE[Privacy mode temp]
    end
    
    style tier1 fill:#512e5f,stroke:#af7ac5
    style tier2 fill:#1a5276,stroke:#85c1e9
    style tier3 fill:#145a32,stroke:#58d68d
    style tier4 fill:#641e16,stroke:#ec7063
```

**Right storage for right purpose.**

---

## Slide 2: Local Storage - SQLite

```mermaid
graph LR
    subgraph sqlite["💻 SQLite Database"]
        CONV[conversation_history]
        FILES[file_store]
        GRAPH[graph_nodes]
        ANCHOR[log_anchors]
        WALLET[wallet_state]
    end
    
    subgraph benefits["Benefits"]
        B1[Fast local access]
        B2[Single file]
        B3[No server needed]
        B4[Portable]
    end
    
    sqlite --> benefits
    
    style sqlite fill:#1a5276,stroke:#85c1e9
```

**Your data, your file.** `agent_data/{agent_id}.db`

---

## Slide 3: The Sovereignty Problem

```mermaid
graph TD
    subgraph problem["Without Sovereignty"]
        P1[Data on one device]
        P2[Device dies = data lost]
        P3[Can't move to new machine]
        P4[No disaster recovery]
    end
    
    subgraph solution["With Sovereignty V2"]
        S1[Encrypted backup to IPFS]
        S2[CID = your recovery key]
        S3[Restore anywhere]
        S4[True portability]
    end
    
    problem -->|Solved by| solution
    
    style problem fill:#641e16,stroke:#ec7063
    style solution fill:#145a32,stroke:#58d68d
```

---

## Slide 4: Sovereignty V2 - The Merkle Forest

```mermaid
graph TD
    subgraph root["📜 Root Manifest"]
        RM[version / agent_did / timestamp]
    end
    
    subgraph shards["📦 Time-Based Shards"]
        S1[2025-09 conversations]
        S2[2025-10 conversations]
        S3[2025-11 conversations]
    end
    
    subgraph keyring["🔑 Encrypted Keyring"]
        KR[Shard ID → Decryption Key]
    end
    
    subgraph ipfs["🌐 IPFS Blocks"]
        B1[CID: Qm...]
        B2[CID: Qm...]
        B3[CID: Qm...]
    end
    
    RM --> S1 & S2 & S3
    RM --> KR
    S1 --> B1
    S2 --> B2
    KR --> B3
    
    style RM fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style KR fill:#641e16,stroke:#ec7063
```

**Incremental.** Only changed months re-upload.

---

## Slide 5: Convergent Encryption

```mermaid
graph LR
    subgraph encrypt["Encryption Process"]
        CONTENT[Plaintext content]
        HMAC["K = HMAC(secret, content)"]
        AES[AES-GCM encrypt]
        CIPHER[Ciphertext]
    end
    
    CONTENT --> HMAC --> AES --> CIPHER
    
    subgraph benefit["Why Convergent?"]
        B1[Same content = same key]
        B2[Same key = same ciphertext]
        B3[IPFS deduplicates globally]
    end
    
    CIPHER --> benefit
    
    style HMAC fill:#7d3c00,stroke:#f5b041,stroke-width:2px
```

**Efficiency + Privacy.** Deduplicated but encrypted.

---

## Slide 6: Export Flow - !export-sovereignty

```mermaid
sequenceDiagram
    participant U as User
    participant K as Kestrel
    participant E as Encryptor
    participant I as IPFS
    
    U->>K: !export-sovereignty
    K->>K: Shard by month
    loop Each Shard
        K->>E: Encrypt shard
        E->>I: Upload ciphertext
        I-->>E: CID
    end
    K->>E: Create keyring
    E->>I: Upload keyring
    K->>I: Upload manifest
    I-->>K: Root CID
    K-->>U: ✅ CID: QmRoot...
```

**One CID to rule them all.** Your entire agent state.

---

## Slide 7: Import Flow - !import-sovereignty

```mermaid
sequenceDiagram
    participant U as User
    participant K as Kestrel
    participant I as IPFS
    participant E as Encryptor
    participant DB as SQLite
    
    U->>K: !import-sovereignty QmRoot...
    K->>I: Fetch manifest
    I-->>K: Shard list + keyring CID
    K->>I: Fetch keyring
    K->>E: Decrypt keyring
    loop Each Shard
        K->>I: Fetch shard
        K->>E: Decrypt with key
        E-->>K: Conversations
    end
    K->>DB: Rebuild database
    K-->>U: ✅ Restored N messages
```

**Full recovery.** From any device, anytime.

---

## Slide 8: The Keyring Solution

```mermaid
graph TD
    subgraph problem["The Chicken-Egg Problem"]
        P1[Each shard has unique key]
        P2[Keys derived from content]
        P3[How to store keys?]
    end
    
    subgraph solution["Keyring Solution"]
        KR[Keyring = map of shard → key]
        ENC[Encrypted with known constant]
        KEY["K = HMAC(secret, 'KESTREL_KEYRING_V2')"]
    end
    
    problem --> solution
    
    style problem fill:#7d3c00,stroke:#f5b041
    style solution fill:#145a32,stroke:#58d68d
```

**Deterministic key for keyring.** Always recoverable with your secret.

---

## Slide 9: Storage Commands

```mermaid
graph LR
    subgraph commands["Sovereignty Commands"]
        EXP["!export-sovereignty [tier]"]
        IMP["!import-sovereignty <cid>"]
        STAT["!check-sovereignty-status"]
    end
    
    subgraph tiers["Storage Tiers"]
        LOCAL[local - filesystem]
        IPFS_T[ipfs - IPFS network]
        FIL[filecoin - long-term]
    end
    
    EXP --> tiers
    
    style commands fill:#1a5276,stroke:#85c1e9
```

---

## Slide 10: Failure Modes & Recovery

```mermaid
graph TD
    subgraph failures["Potential Failures"]
        F1[Shard upload fails]
        F2[Shard disappears from IPFS]
        F3[Lost user secret]
    end
    
    subgraph handled["Handled"]
        H1[Abort before manifest]
        H2[Partial recovery possible]
    end
    
    subgraph fatal["Fatal - No Recovery"]
        X1[All data lost forever]
    end
    
    F1 --> H1
    F2 --> H2
    F3 --> X1
    
    style handled fill:#145a32,stroke:#58d68d
    style fatal fill:#641e16,stroke:#ec7063
```

**Your secret is sacred.** Lose it, lose everything. No backdoor.

---

*Next: [06-llm-management.md](06-llm-management.md) - Multi-model fallback and GPU integration*
