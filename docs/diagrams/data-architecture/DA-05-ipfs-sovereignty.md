# DA-05: IPFS Sovereignty Export

Decentralized storage with convergent encryption and Merkle forest architecture.

---

## Slide 1: The Sovereignty Promise

```mermaid
graph TD
    subgraph promise["The Promise"]
        MAIN["Your AI companion can never<br/>be taken away from you"]
    end

    subgraph how["How It Works"]
        H1["Export agent state to IPFS"]
        H2["Get a CID (content address)"]
        H3["Restore anywhere, anytime"]
    end

    subgraph proof["Proof of Ownership"]
        P1["CID = cryptographic proof"]
        P2["Put CID in your will"]
        P3["Digital inheritance"]
    end

    promise --> how --> proof

    style MAIN fill:#145a32,stroke:#58d68d,stroke-width:3px
    style proof fill:#7d6608,stroke:#f4d03f
```

**Not legal ownership. Cryptographic ownership.**

---

## Slide 2: Time-Based Sharding

```mermaid
graph TD
    subgraph conv["All Conversations"]
        ALL["1000 messages<br/>over 6 months"]
    end

    subgraph shards["Monthly Shards"]
        S1["2025-07<br/>150 msgs"]
        S2["2025-08<br/>200 msgs"]
        S3["2025-09<br/>180 msgs"]
        S4["2025-10<br/>170 msgs"]
        S5["2025-11<br/>200 msgs"]
        S6["2025-12<br/>100 msgs"]
    end

    subgraph benefit["Benefits"]
        B1["Only new months uploaded"]
        B2["Incremental backups"]
        B3["Efficient deduplication"]
    end

    ALL --> shards --> benefit

    style shards fill:#1a5276,stroke:#85c1e9
```

**Incremental.** Only new/changed months are uploaded.

---

## Slide 3: Convergent Encryption

```mermaid
graph LR
    subgraph input["Input"]
        CONTENT["Message content"]
        SECRET["User's secret key"]
    end

    subgraph derive["Key Derivation"]
        HMAC["HMAC(Content, Secret)"]
        KEY["Encryption key"]
    end

    subgraph encrypt["Encryption"]
        CIPHER["AES-256-GCM encrypt"]
        OUTPUT["Encrypted blob"]
    end

    CONTENT --> HMAC
    SECRET --> HMAC
    HMAC --> KEY --> CIPHER --> OUTPUT

    style HMAC fill:#7d6608,stroke:#f4d03f,stroke-width:2px
```

**Key = HMAC(Content, Secret)**

Same content → Same key → Same ciphertext → IPFS deduplicates!

---

## Slide 4: Why Convergent Encryption?

```mermaid
graph TD
    subgraph without["Without Convergent Encryption"]
        W1["Same message"]
        W2["Random key each time"]
        W3["Different ciphertext"]
        W4["No deduplication"]
    end

    subgraph with["With Convergent Encryption"]
        C1["Same message"]
        C2["Deterministic key"]
        C3["Same ciphertext"]
        C4["IPFS deduplicates!"]
    end

    style without fill:#641e16,stroke:#ec7063
    style with fill:#145a32,stroke:#58d68d
```

**Privacy + Efficiency.** Best of both worlds.

---

## Slide 5: Merkle Forest Structure

```mermaid
graph TD
    subgraph root["Root Manifest"]
        MANIFEST["manifest.json<br/>CID: Qm...abc"]
    end

    subgraph shards["Shards"]
        SHARD1["2025-11.enc<br/>CID: Qm...111"]
        SHARD2["2025-12.enc<br/>CID: Qm...222"]
    end

    subgraph keyring["Keyring"]
        KEYS["keyring.enc<br/>CID: Qm...key"]
    end

    MANIFEST --> SHARD1
    MANIFEST --> SHARD2
    MANIFEST --> KEYS

    style MANIFEST fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style SHARD1 fill:#1a5276,stroke:#85c1e9
    style SHARD2 fill:#1a5276,stroke:#85c1e9
    style KEYS fill:#512e5f,stroke:#af7ac5
```

**DAG structure.** Root points to shards and keyring.

---

## Slide 6: Export Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant S as SovereignAdapter
    participant IPFS as IPFS Gateway

    U->>A: !export-sovereignty
    A->>S: export_snapshot()

    S->>S: Group conversations by month
    S->>S: Compute convergent keys
    S->>S: Encrypt each shard

    loop For each shard
        S->>IPFS: Upload encrypted shard
        IPFS-->>S: Shard CID
    end

    S->>S: Build manifest
    S->>IPFS: Upload manifest
    IPFS-->>S: Root CID

    S-->>A: SovereigntyReceipt
    A-->>U: ✅ Exported: Qm...abc
```

---

## Slide 7: Import/Restore Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant S as SovereignAdapter
    participant IPFS as IPFS Gateway

    U->>A: !import-sovereignty Qm...abc
    A->>S: import_snapshot(cid, secret)

    S->>IPFS: Fetch manifest
    IPFS-->>S: manifest.json

    S->>IPFS: Fetch keyring
    IPFS-->>S: keyring.enc
    S->>S: Decrypt keyring with secret

    loop For each shard in manifest
        S->>IPFS: Fetch shard
        IPFS-->>S: shard.enc
        S->>S: Decrypt with convergent key
    end

    S->>S: Restore to local SQLite
    S-->>A: Restore complete
    A-->>U: ✅ Restored 847 messages
```

---

## Slide 8: Local Fallback

```mermaid
graph TD
    subgraph normal["Normal Export"]
        N1["Upload to IPFS"]
        N2["Get CID"]
        N3["Store receipt"]
    end

    subgraph fallback["IPFS Unavailable"]
        F1["Save to local cache"]
        F2["Generate pseudo-CID"]
        F3["Retry later"]
    end

    subgraph guarantee["Guarantee"]
        G1["Export NEVER fails"]
        G2["Data always saved"]
    end

    normal --> guarantee
    fallback --> guarantee

    style fallback fill:#7d6608,stroke:#f4d03f
    style guarantee fill:#145a32,stroke:#58d68d
```

**Graceful degradation.** IPFS down? Cache locally, retry later.

---

## Slide 9: What Gets Preserved

```mermaid
graph TD
    subgraph preserved["Sovereignty Snapshot Contains"]
        CONV["💬 Conversations<br/>All messages with metadata"]
        GRAPH["🕸️ Knowledge Graph<br/>Nodes and edges"]
        IDENTITY["🔑 Agent Identity<br/>DID and signing key"]
        WALLET["💰 Wallet State<br/>Balances and transactions"]
        CONFIG["⚙️ Configuration<br/>Privacy mode, preferences"]
    end

    subgraph complete["Complete Agent State"]
        C1["Restore = identical agent"]
        C2["No data loss"]
        C3["Platform-independent"]
    end

    preserved --> complete

    style preserved fill:#1a5276,stroke:#85c1e9
    style complete fill:#145a32,stroke:#58d68d
```

---

## Slide 10: The CID - Your Proof

```mermaid
graph LR
    subgraph cid["The CID"]
        HASH["Qm...abc123"]
        MEANING["Content-addressed hash"]
    end

    subgraph powers["Powers"]
        P1["Restore on any device"]
        P2["Survive platform shutdown"]
        P3["Transfer to heir"]
        P4["Prove ownership"]
    end

    subgraph storage["Store It"]
        S1["Password manager"]
        S2["Paper backup"]
        S3["Your will"]
    end

    cid --> powers
    powers --> storage

    style HASH fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style powers fill:#145a32,stroke:#58d68d
```

**The CID is your sovereignty.** Guard it like a private key.

---

*Next: [DA-06-filecoin-lighthouse.md](DA-06-filecoin-lighthouse.md) - Permanent storage via Lighthouse*
