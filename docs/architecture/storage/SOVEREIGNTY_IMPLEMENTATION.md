---
type: Architecture Spec
title: Sovereignty Implementation
description: '**Date:** November 21, 2025 **Status:** ✅ **COMPLETE** - V2 Architecture
  with Convergent Sharding **Tests:** 16/16 passing (100%) **Vision:** "Your AI companion
  can never be tak...'
resource: /docs/architecture/storage/SOVEREIGNTY_IMPLEMENTATION.md
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

# Sovereignty Implementation

**Date:** November 21, 2025
**Status:** ✅ **COMPLETE** - V2 Architecture with Convergent Sharding
**Tests:** 16/16 passing (100%)
**Vision:** "Your AI companion can never be taken away from you"

---

## What Was Built

A complete IPFS-based sovereignty system where users cryptographically own their AI companion.

### Core Classes (`storage/sovereign_adapter.py`)

```python
class SovereigntyReceipt:
    """The user's proof of ownership"""
    cid: str                      # IPFS Content ID - THE KEY TO EVERYTHING
    content_hash: str             # SHA256 for integrity
    created_at: str
    agent_did: str
    storage_tier: str             # ipfs or filecoin
    encrypted: bool
    size_bytes: int
    manifest: Dict                # What's included

    def to_qr_string() -> str     # For mobile QR codes
    def to_json() -> str          # For saving
```

```python
class AgentSnapshot:
    """Complete agent state for export"""
    agent_did: str
    constitution: str              # Governing principles
    conversations: List[Dict]      # Full history
    memories: List[Dict]           # Semantic memories
    files: Dict[str, bytes]        # All files
    graph_nodes: List[Dict]        # Knowledge graph
    graph_edges: List[Dict]        # Relationships
    metadata: Dict
```

```python
class SovereignStorageAdapter:
    """High-level sovereignty operations"""

    def export_agent_to_sovereignty(...) -> SovereigntyReceipt:
        """Package agent, upload to IPFS, return CID"""

    def import_agent_from_sovereignty(cid) -> AgentSnapshot:
        """Load agent from IPFS using CID"""

    def verify_sovereignty_receipt(...) -> bool:
        """Verify CID is valid and accessible"""
```

### User Commands (`kestrel_agent.py`)

```bash
!export-sovereignty                # Export to IPFS (free)
!export-sovereignty --filecoin     # Export to Filecoin (paid, permanent)
!export-sovereignty --no-encrypt   # Don't encrypt (not recommended)

!import-sovereignty <cid>          # Restore from CID
!sovereignty-status                # Check export history, view CIDs
```

**What Happens on Export:**
1. User types `!export-sovereignty`
2. Agent packages entire state (constitution, conversations, files, graph)
3. Encrypts with convergent encryption (content-derived keys)
4. Shards by time period (monthly buckets)
5. Uploads to IPFS (distributed network)
6. Returns CID to user (e.g., `QmYwAPJzv5CZsnA636s8K6v...`)
7. User saves CID (paper, password manager, QR code)

**CID = Sovereignty Proof** - With just the CID, users can restore anywhere.

---

## V2 Architecture: Convergent Sharding

The V2 system replaced monolithic exports with an efficient sharded approach:

```
┌─────────────────────────────────────────────────────────┐
│ V2 MERKLE FOREST                                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Root Manifest (CID: QmRoot...)                        │
│    ├── agent_did: "did:pkh:eip155:1:0x..."            │
│    ├── shards:                                         │
│    │     ├── 2025-10: QmShard1...                     │
│    │     ├── 2025-11: QmShard2...                     │
│    │     └── ...                                       │
│    └── keyring: QmKeyring...                          │
│                                                         │
│  Each Shard (encrypted, content-addressed):            │
│    ├── conversations for that month                    │
│    ├── files created that month                        │
│    └── graph nodes/edges from that month               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Key Benefits:**
- **Deduplication**: Identical content = same CID (convergent encryption)
- **Incremental**: Only upload new/changed months
- **Scalable**: Years of data without massive single files

---

## Use Cases

### 1. Elderly Care (Story Preservation)
- Grandma collects lifetime of stories over 5 years
- Family receives her CID (in will)
- Stories preserved forever on IPFS
- Future grandkids can access in 2075

### 2. Platform Shutdown Survival
- User has deep bond with AI companion
- Platform announces shutdown
- User loads CID on new platform
- Companion restored exactly as it was

### 3. Platform Migration (Zero Lock-in)
- Export from Kestrel (`!export-sovereignty`)
- Import to competitor platform
- No vendor lock-in

### 4. Privacy with True Encryption
- Data encrypted before upload
- Even with CID, need decryption key
- Wrong key = no access

---

## Comparison

### vs. ChatGPT/Claude

| Feature | ChatGPT | Kestrel |
|---------|---------|---------|
| **Data Ownership** | OpenAI owns | You own (CID) |
| **Platform Lock-in** | 100% locked | 0% locked |
| **Exportable** | No | Yes (IPFS CID) |
| **Survives Shutdown** | No | Yes (decentralized) |
| **Inheritable** | No | Yes (CID in will) |
| **Provable Ownership** | No | Yes (cryptographic) |

### vs. Traditional Backups

| Feature | Cloud Backup | Sovereignty |
|---------|--------------|-------------|
| **Provider Trust** | Required | Not required |
| **Can Be Deleted** | Yes | No (IPFS) |
| **Vendor Lock-in** | Yes | No |
| **Globally Accessible** | No | Yes (IPFS network) |
| **Cryptographic Proof** | No | Yes (CID) |

---

## Test Results

### 16/16 Tests Passing (100%)

**V2 Core Tests** (`test_sovereignty_v2.py`):
- ✅ Sharding & Manifest creation
- ✅ Agent Command integration

**E2E Tests** (`test_sovereignty_e2e.py`):
- ✅ V2 Export Flow (Local Fallback)
- ✅ Convergent Encryption Deduplication
- ✅ Agent Command Integration

**Guarantee Tests** (`test_sovereignty_guarantees.py`):
- ✅ Persistence (DB actually saves)
- ✅ Shutdown Survival
- ✅ Encryption Works (SSN not visible)
- ✅ Content Integrity (tampering detected)
- ✅ Sovereignty vs Download (6 differences)
- ✅ Inheritance Scenario
- ✅ Performance Metrics

**Import Tests** (`test_sovereignty_import.py`):
- ✅ Export → Import Roundtrip
- ✅ Wrong Key Fails
- ✅ Agent Command Import
- ✅ Partial Shard Recovery

---

## Performance Metrics

- **200 messages** = ~18.5 KB
- **Snapshot creation**: 0.005 seconds
- **IPFS pinning cost**: ~$0.00/month (essentially free)
- **Retrieval**: Milliseconds from local IPFS node

---

## Honest Limitations

1. **Must Save CID**: Lose CID = lose access
   - Solution: QR codes, password managers, paper backups

2. **Must Store Encryption Key**: Lose key = unreadable data
   - Solution: Same storage as CID

3. **IPFS Requires Pinning**: Someone must host the data
   - Solution: Pinning services, Filecoin deals, own node

4. **Platform Must Support Import**: Not universal (yet)
   - Solution: Kestrel reference implementation, open spec

5. **Large Files = Higher Costs**: GB-scale data costs more
   - Solution: Selective export, Filecoin for long-term

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `storage/sovereign_adapter.py` | ~500 | Core sovereignty engine |
| `kestrel_agent.py` | +160 | User commands |
| `tests/integration/test_sovereignty_v2.py` | ~200 | V2 tests |
| `tests/integration/test_sovereignty_e2e.py` | ~500 | E2E tests |
| `tests/integration/test_sovereignty_guarantees.py` | ~600 | Guarantee tests |
| `tests/integration/test_sovereignty_import.py` | ~300 | Import tests |

---

## Next Steps

1. **Browser Integration**: js-ipfs in service worker
2. **Kestrel UI**: "Export Sovereignty" button, QR codes
3. **Mobile Apps**: React Native IPFS
4. **Filecoin Integration**: Paid permanent storage deals

---

*Last Updated: November 21, 2025*
