# The Kestrel Sovereignty Stack

This document explains the three architectural pillars that give Kestrel agents their sovereignty properties: **Identity**, **Constitution**, and **Memory**. Read this if you want to understand how those properties are implemented, not just that they exist.

---

## What "sovereign" means here

A sovereign agent is one whose existence, behavior, and accumulated knowledge cannot be taken away, modified, or surveilled without the consent of the agent's owner.

That's a strong claim. It requires three independent guarantees:

1. **The agent has a unique, cryptographically verifiable identity** that isn't issued or controlled by any platform.
2. **The agent's behavior is governed by principles** that can't be silently overridden by an upstream model update, a configuration change, or a bad prompt.
3. **The agent's memory belongs to its owner** — portable, encrypted, and deletable on demand.

These map directly to Kestrel's three pillars.

---

## Pillar 1 — Identity (Sovereignty)

### Key generation

At inception, Kestrel generates a `secp256k1` elliptic curve key pair — the same curve used by Ethereum and Bitcoin. This is not incidental: it means the agent's identity is compatible with the widest possible ecosystem of cryptographic tooling and verifiable credential infrastructure.

```
private_key  →  secp256k1 (ECDSA-P256K)
public_key   →  uncompressed point (04 prefix + X + Y coordinates)
address      →  Keccak-256 of public_key bytes [-20:]  →  EIP-55 checksum applied
```

### DID format

The agent's identity is expressed as a W3C Decentralized Identifier using the `did:pkh` method:

```
did:pkh:eip155:1:0x8955b8cBe7...
```

Components:
- `did:pkh` — the "public key hash" DID method (W3C spec)
- `eip155:1` — Ethereum mainnet chain namespace (EIP-155)
- `0x8955b8cBe7...` — EIP-55 checksummed Ethereum address derived from the public key

The DID document is a standard W3C JSON-LD document saved alongside the agent's data. It contains the public key, authentication methods, and assertion methods needed for external verification. It contains no private information.

### Key storage

The private key is encrypted at rest using `KESTREL_DATA_KEY` (Fernet symmetric encryption) via `SecureKeyStorage`. If `KESTREL_DATA_KEY` is not set, Kestrel warns and falls back to a plaintext PEM file — usable for development, not for production.

```
agent_data/
  myagent/
    kestrel_{address}.json     # DID document (public)
    kestrel_{address}.key.enc  # Encrypted private key (secret)
    kestrel_prime.db           # Encrypted memory store
```

### Portability

The agent is fully portable. Everything needed to resume it is in `agent_data/myagent/`:
- The DID document (public key, identity)
- The encrypted private key
- The encrypted memory database

Copy that directory to any machine running Kestrel with the same `KESTREL_DATA_KEY` and the agent resumes with full identity and memory intact. No platform account required.

### What this protects against

| Threat | How identity addresses it |
|--------|--------------------------|
| Platform lock-in | Identity is self-issued. No vendor holds your agent's key. |
| Identity theft | Private key never leaves encrypted storage. |
| Impersonation | DID is cryptographically bound to the key pair. Any signature can be verified against the public key. |

---

## Pillar 2 — Constitution (Governance)

### What the constitution is

The Kestrel Constitution is a text document that defines what the agent can and cannot do — its values, constraints, and operating principles. It is loaded at inception, anchored in memory with a cryptographic hash, and verified automatically at runtime.

The default constitution lives at `docs/principles/KESTREL_CONSTITUTION.md`. You can replace it with your own at inception time.

### Genesis anchoring

When an agent is created, the inception service:

1. Loads the constitution file
2. Computes a SHA-256 hash of its contents
3. Stores the hash in the agent's root identity node in the graph store

This hash is the **constitution anchor** — a tamper-evident seal. From this point on, the agent can verify at any time that its constitution has not changed.

```python
# Stored in the agent's identity node
agent_node.properties["constitution_hash"] = sha256(constitution_text)
```

### Integrity auditing

The agent audits its own constitution automatically at two trigger points:

- Every **100 interactions** (configurable via `KESTREL_AUDIT_INTERVAL`)
- Every **24 hours** (wall-clock elapsed)

The audit compares the current on-disk constitution hash against the anchored hash in storage. If they differ, the agent enters **safe mode** — it stops processing new inputs and reports the integrity failure.

```
Audit triggers → _verify_constitution_integrity()
                    ↓ PASS: log + reset counter
                    ↓ FAIL: enter_safe_mode("Constitution audit failed: ...")
```

Safe mode is intentionally conservative: a tampered constitution cannot silently take effect. The operator must investigate and resolve the integrity failure before the agent resumes.

### Writing a custom constitution

A constitution is a plain text file. To use a custom one, pass its path at `kestrel create` time (see the CLI reference). Guidelines:

- Define explicit **values** the agent should embody
- Define explicit **prohibitions** — things the agent must never do regardless of instruction
- Define **escalation rules** — what to do when a user request conflicts with a prohibition
- Keep it concise enough to include in every LLM context window

The default constitution is a good starting point. Read it at `docs/principles/KESTREL_CONSTITUTION.md`.

### What this protects against

| Threat | How constitution addresses it |
|--------|------------------------------|
| Silent jailbreak via prompt | Constitution is in every system prompt — constraints travel with every LLM call. |
| Model update that changes behavior | Constitution hash detects any tampering with the governance document. |
| Operator misconfiguration | Genesis audit fails loudly if the agent is started without a valid constitution. |

---

## Pillar 3 — Memory (Persistence)

### Storage stack

Kestrel's memory is built on three layers:

| Layer | Component | What it stores |
|-------|-----------|---------------|
| Graph store | `AsyncGraphStore` | `GraphNode` entities + `Edge` relationships |
| File store | `AsyncFileStore` | Binary files, documents, images (content-addressed) |
| Database | `AsyncDatabase` | Structured records, conversation history, session metadata |

All three write into a single SQLite file (`kestrel_prime.db`) in the agent's data directory. SQLite was chosen deliberately: it's a single portable file, requires no server, and is trivially backed up.

### Knowledge graph

Every piece of information the agent learns is stored as a node in the knowledge graph, with typed edges connecting related nodes. This enables:

- **Associative recall** — "what do I know about X that connects to Y?"
- **Relationship traversal** — follow edges between concepts across sessions
- **Full-text search** — find nodes by content across the entire memory
- **RAG retrieval** — retrieve relevant memory chunks to augment LLM context

The agent builds this graph organically through conversation. When you tell your agent something, it creates nodes. When it connects two things you've said, it creates edges.

### Privacy modes

Privacy mode controls what the memory layer stores — and what it doesn't. Privacy is enforced at the storage layer, not just by convention.

| Mode | Storage behavior | LLM routing |
|------|-----------------|-------------|
| `EPHEMERAL` | Nothing written to disk. In-memory buffer only, cleared on session end. | Local providers only (Ollama, llama_cpp) |
| `ISOLATED` | Temporary storage within the session. Deleted when the session ends. | Local providers only |
| `ANONYMOUS` | Stored, but PII stripped before write. Encrypted backups required. | Any provider |
| `NORMAL` | Full persistence, full history. Encrypted at rest. | Any provider |
| `PUBLIC` | Full persistence with sharing/export enabled. | Any provider |

Mode is set programmatically when building agents, or via `POST /agent/privacy-mode`. The frontend model selector automatically switches to a local provider when a local-only mode is activated — and restores the previous cloud model when switching back.

### Encryption at rest

When `KESTREL_DATA_KEY` is set, the memory database is encrypted using Fernet symmetric encryption. The key never leaves the host environment; it is never sent to any LLM provider.

### What this protects against

| Threat | How memory addresses it |
|--------|------------------------|
| Cloud surveillance | EPHEMERAL mode: nothing leaves the device, LLM calls go local only. |
| Data breach | Fernet encryption at rest. Attacker with DB file cannot read without the key. |
| Platform data withholding | Single SQLite file — copy and you have everything. No export API required. |

---

## How the three pillars interact

The pillars are independent but complementary. Here's what happens at each lifecycle event:

**Agent creation (`kestrel create`)**
1. Identity generates the secp256k1 keypair and DID
2. Constitution is loaded, hashed, and the hash is written into the identity node (linking identity to governance)
3. Memory is initialized: the agent node is written to the graph with the DID and constitution hash embedded

**Each conversation turn**
1. Memory retrieves relevant context (RAG) and injects it into the LLM system prompt
2. Constitution is included in every system prompt — the LLM sees both the governance rules and the relevant memory
3. Privacy mode determines whether the response and any new knowledge is persisted
4. Every `AUDIT_INTERVAL` turns, the constitution is re-verified against the anchored hash

**When the agent moves to a new host**
1. Copy `agent_data/myagent/` + the `KESTREL_DATA_KEY` value
2. `kestrel start ./agent_data/myagent` — the agent reconnects its identity, verifies its constitution, and resumes with full memory

There is no step 4. No re-registration, no platform handshake, no token refresh.

---

## Threat model summary

Kestrel's sovereignty stack is designed against these adversary classes:

| Adversary | Mechanism | Kestrel's defense |
|-----------|-----------|------------------|
| Cloud platform (lock-in) | Holds data, identity, model access | Self-issued DID + portable SQLite |
| Cloud LLM provider (surveillance) | Receives all prompts | EPHEMERAL routes to local-only |
| Malicious operator (behavior change) | Modifies constitution | Genesis anchor + periodic audit → safe mode |
| Attacker with DB access | Reads stored memories | Fernet encryption at rest |
| Prompt injection | Overrides constitution in-context | Constitution is in system prompt, not user prompt |

This is not a complete security audit. See [docs/architecture/CRYPTOGRAPHIC_ANCHORING.md](CRYPTOGRAPHIC_ANCHORING.md) for the full cryptographic specification.

---

## Related documents

| Document | What it covers |
|----------|---------------|
| [QUICKSTART.md](../QUICKSTART.md) | Get a sovereign agent running in 30 minutes |
| [docs/principles/KESTREL_CONSTITUTION.md](principles/KESTREL_CONSTITUTION.md) | The default constitutional framework |
| [docs/architecture/PRIVACY_MODES.md](architecture/PRIVACY_MODES.md) | Privacy mode internals and policy enforcement |
| [docs/architecture/CRYPTOGRAPHIC_ANCHORING.md](architecture/CRYPTOGRAPHIC_ANCHORING.md) | DID generation, key management, genesis audit |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | How to contribute |
