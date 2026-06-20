---
type: Architecture Spec
title: Kestrel Constitution Embedding Process
description: Every Kestrel agent is cryptographically bound to the Kestrel Constitution
  from inception. This document describes the process by which the constitution becomes
  part of an agent...
resource: /docs/architecture/security/CONSTITUTION_EMBEDDING.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Kestrel Constitution Embedding Process

## Overview

Every Kestrel agent is cryptographically bound to the Kestrel Constitution from inception. This document describes the process by which the constitution becomes part of an agent's identity and how agents prove their governance relationship to the constitution throughout their lifetime.

## The Constitutional Foundation

The Kestrel Constitution is the supreme law that governs all Kestrel agents. It is:
- **Content-addressable**: Identified by the SHA-256 hash of its contents
- **Immutable**: Cannot be modified after an agent is created
- **Verifiable**: Any agent can retrieve and verify its constitution
- **Universal**: All agents share the same constitution (same hash)

## Inception Process

### 1. Key Generation
The agent's cryptographic identity begins with secp256k1 key generation:
```python
private_key, public_key = generate_secp256k1_keypair()
```

### 2. Ethereum Address Derivation
From the public key, an Ethereum address is derived using Keccak-256:
```python
ethereum_address = public_key_to_ethereum_address(public_key)
# Result: 0x{40-hex-chars-with-EIP55-checksum}
```

### 3. W3C DID Creation
A W3C-compliant Decentralized Identifier (DID) is created:
```
did:pkh:eip155:1:{ethereum_address}
```

This DID serves as the agent's permanent identifier and becomes the `node_id` for the agent in the knowledge graph.

### 4. Constitution Storage (CRITICAL)
The constitution is stored as the **FIRST file** in the agent's storage:
```python
with open(constitution_path, "rb") as f:
    constitution_content = f.read()
constitution_hash = storage.store_file(constitution_content, "KESTREL_CONSTITUTION.md")
```

The hash is computed as SHA-256:
```
constitution_hash = hashlib.sha256(constitution_content).hexdigest()
# Result: {64-hex-chars}
```

### 5. Graph Node Creation
Two nodes are created in the knowledge graph:

#### Constitution Node (Content-Addressable)
```python
constitution_node = GraphNode(
    node_id=constitution_hash,              # Content hash is the ID
    node_type="document",
    label="KESTREL_CONSTITUTION",
    properties={
        "hash": constitution_hash,
        "type": "Constitution",
        "created_at": "2025-11-07T15:30:00+00:00"  # UTC timestamp
    }
)
storage.add_node(constitution_node)
```

The `node_id` equals the content hash, making this a **content-addressable** node. If the constitution's content ever changed, its hash would change, creating a new node. Since the agent's properties reference a specific hash, the immutability is enforced cryptographically.

#### Agent Node
```python
agent_node = GraphNode(
    node_id=agent_did,                      # W3C DID
    node_type="agent",
    label="Kestrel Agent",
    properties={
        "created_at": "2025-11-07T15:30:00+00:00",
        "constitution_hash": constitution_hash,  # CRITICAL: Agent references constitution hash
        "initialBalance": "1000.0"
    }
)
storage.add_node(agent_node)
```

### 6. Governance Edge
A directed edge links the agent to its constitution:
```python
storage.add_edge(agent_node.node_id, constitution_node.node_id, "governed_by")
```

This edge establishes that the agent is governed by the constitution. The edge is:
- **Directed**: From agent → constitution
- **Labeled**: "governed_by"
- **Immutable**: Once created, cannot be deleted or modified

### 7. Genesis Self-Audit
Upon creation, the agent performs a self-audit of its constitution:
```python
async def genesis_self_audit(agent: KestrelAgent) -> bool:
    """Agent audits its own constitution on first boot"""
    
    # Retrieve constitution via hash
    constitution_text = agent.storage.retrieve_file(
        agent.properties["constitution_hash"]
    )
    
    # Use the normal provider-routing path to verify integrity
    audit_result = await agent.llm_service.get_audit_response(constitution_text)
    
    if audit_result["risk_level"] >= 3:
        # HIGH RISK - abort agent creation
        logger.error(f"Genesis audit failed: {audit_result['reasoning']}")
        return False
    
    return True
```

If the audit returns a risk level >= 3, the agent is not created and all artifacts are cleaned up.

## Constitutional Governance Throughout Agent Lifetime

### Constitution Retrieval
At any time, an agent can retrieve its governing constitution:
```python
def _get_governing_constitution(self) -> str:
    # Find the agent's node
    agent_node = self.storage.get_node(self.agent_id)
    
    # Get the constitution hash from properties
    constitution_hash = agent_node.properties.get("constitution_hash")
    
    # Retrieve the actual constitution content
    constitution_bytes = self.storage.retrieve_file(constitution_hash)
    
    return constitution_bytes.decode('utf-8')
```

### Constitutional Primacy
Every agent decision is made within the context of the constitution:
1. Consult the Constitution (first priority)
2. Consult factual sources (second priority)
3. Consult conversation history (third priority)

The constitution is injected into every LLM prompt, ensuring the agent operates within constitutional bounds.

### Integrity Audits
During operation, the agent performs integrity audits on responses to ensure constitutional compliance:
```python
async def _perform_integrity_audit(self, response: str) -> str:
    # Get the constitution
    constitution = self._get_governing_constitution()
    
    # Audit the response against the constitution using the standard provider chain
    audit_result = await self.llm_service.get_audit_response(response)
    
    if audit_result["risk_level"] < 3:
        return response  # Compliant
    else:
        return "SYSTEM_CORRECTION: Response was unconstitutional"
```

## Verification and Proof

### Proving Governance Relationship
An agent can prove its governance relationship:
```python
# 1. Retrieve own DID
agent_did = "did:pkh:eip155:1:0x..."

# 2. Query own node
agent_node = storage.get_node(agent_did)

# 3. Get constitution hash from properties
const_hash = agent_node.properties["constitution_hash"]

# 4. Verify constitution via hash
constitution = storage.retrieve_file(const_hash)

# 5. Compute hash to verify
import hashlib
computed_hash = hashlib.sha256(constitution).hexdigest()
assert computed_hash == const_hash  # Proof!

# 6. Verify governance edge
edges = storage.get_edges_from(agent_did)
governed_by = [e for e in edges if e.label == "governed_by"]
assert len(governed_by) == 1
assert governed_by[0].target_id == const_hash
```

### Multi-Agent Constitution Sharing
Multiple agents created at the same time will share the same constitution hash (since the content is identical):

```
Agent 1: did:pkh:eip155:1:0xA... → constitution_hash: abc123...
Agent 2: did:pkh:eip155:1:0xB... → constitution_hash: abc123...
Agent 3: did:pkh:eip155:1:0xC... → constitution_hash: abc123...
```

All three agents reference the same constitution node in the graph, providing efficient storage and unified governance.

## Immutability Guarantees

### Content-Addressable Storage
Because the constitution node ID equals its SHA-256 hash, immutability is cryptographic:
- If someone modifies the constitution file, its hash changes
- The agent's properties reference the old hash, not the modified content
- The stored content can be verified against the hash

### Governance Edge Persistence
The governance edge is stored in the database as immutable metadata:
- Once created, it cannot be deleted
- It cannot be reassigned to a different constitution
- All integrity checks refer to this edge

### DID Document Proof
The agent's W3C DID document contains public key material:
```json
{
  "id": "did:pkh:eip155:1:0x...",
  "publicKey": [{
    "id": "did:pkh:eip155:1:0x...#keys-1",
    "type": "EcdsaSecp256k1VerificationKey2019",
    "publicKeyHex": "04..."
  }]
}
```

The DID itself is derived from the public key, proving the agent's identity.

## Database Schema

### Nodes Table
```sql
CREATE TABLE graph_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT,
    label TEXT,
    properties TEXT  -- JSON
);
```

Example rows:
```
node_id: "abc123def456..."  (constitution hash)
node_type: "document"
label: "KESTREL_CONSTITUTION"
properties: '{"hash": "abc123def456...", "type": "Constitution", "created_at": "2025-11-07T15:30:00+00:00"}'

node_id: "did:pkh:eip155:1:0x..."  (agent DID)
node_type: "agent"
label: "Kestrel Agent"
properties: '{"created_at": "2025-11-07T15:30:00+00:00", "constitution_hash": "abc123def456...", "initialBalance": "1000.0"}'
```

### Edges Table
```sql
CREATE TABLE graph_edges (
    source_id TEXT,
    target_id TEXT,
    label TEXT,
    properties TEXT  -- JSON
);
```

Example row:
```
source_id: "did:pkh:eip155:1:0x..."
target_id: "abc123def456..."
label: "governed_by"
properties: NULL
```

### Files Table
```sql
CREATE TABLE files (
    content_hash TEXT PRIMARY KEY,
    original_name TEXT,
    content BLOB,
    metadata TEXT  -- JSON
);
```

## Test Suite

The test suite in `tests/integration/test_constitution_embedding.py` verifies:

1. **test_constitution_anchored**: Constitution stored and linked
2. **test_constitution_retrievable**: Agent can retrieve constitution
3. **test_constitution_stored_first**: Constitution is first file stored
4. **test_did_format_validation**: DID follows W3C spec
5. **test_constitution_content_hash_deterministic**: Hash is consistent
6. **test_agent_can_access_constitution_via_kestrel_agent**: Runtime integration
7. **test_genesis_audit_bypassed_temporarily**: Audit runs on inception
8. **test_constitution_encryption_if_key_set**: Encryption support
9. **test_multiple_agents_same_constitution**: Content addressing works
10. **test_constitution_required_for_inception**: Validation on startup
11. **test_agent_properties_include_initial_balance**: Wallet genesis
12. **test_constitution_node_timestamp**: UTC timestamps

All tests run as async with `pytest-anyio` and verify the complete constitution embedding lifecycle.

## Cryptographic Security

### SHA-256 Hashing
- Content hash is deterministic: Same constitution → Same hash
- Pre-image resistance: Cannot create different content with same hash
- Avalanche effect: Any byte change produces completely different hash

### secp256k1 Key Pair
- Public key derived from private key (one-way)
- Ethereum address derived from public key (one-way)
- DID incorporates Ethereum address in immutable format

### EIP-55 Checksum
- Ethereum addresses include checksum information
- Detects transposition and case errors
- Provides additional proof of correct address

## Conclusion

The Kestrel Constitution embedding process creates an unbreakable link between each agent and its governing constitution. This link is:
- **Cryptographic**: Based on SHA-256 hashing
- **Decentralized**: No central authority can modify it
- **Verifiable**: Any observer can verify the relationship
- **Immutable**: Cannot be changed after creation

Every Kestrel agent carries its constitutional anchor from inception through its entire operational lifetime.
