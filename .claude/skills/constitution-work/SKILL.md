---
name: constitution-work
description: Use when working on Kestrel Constitution embedding, DID creation, inception service, or constitutional governance verification. Automatically delegates to constitution-verifier subagent for validation work.
---

# Constitution & Inception Work Skill

This skill automatically activates when detecting work on Kestrel's constitutional governance system.

## Trigger Keywords
- "constitution", "Constitution", "KESTREL_CONSTITUTION"
- "inception", "DID", "decentralized identifier"
- "genesis audit", "constitutional governance"
- "inception_service.py", "governed_by edge"
- "agent creation", "cryptographic identity"
- "secp256k1", "Ethereum address", "W3C DID"

## What This Skill Does

When activated, this skill:
1. Detects you're working on constitutional governance
2. Suggests using the **constitution-verifier** subagent
3. Provides verification checklist
4. References Kestrel Constitution principles

## Quick Verification Checklist

### Constitution Anchoring (inception_service.py)
- [ ] Constitution stored as FIRST file
- [ ] Constitution hash in agent properties
- [ ] "governed_by" edge created
- [ ] DID format valid: `did:pkh:eip155:1:0x{address}`
- [ ] Genesis self-audit implemented

### Graph Structure
```
Agent Node (DID)
  └─[governed_by]─→ Constitution Node (hash)
```

### Critical Files
- **Constitution**: `docs/principles/KESTREL_CONSTITUTION.md`
- **Inception**: `inception_service.py` (lines 198-216)
- **Storage**: `storage/__init__.py`
- **Agent**: `kestrel_agent.py`

## Genesis Self-Audit Pattern

Every new agent must audit its own constitution:
```python
async def genesis_self_audit(agent: KestrelAgent) -> bool:
    constitution = agent.storage.retrieve_file_by_hash(
        agent.properties["constitution_hash"]
    )
    audit = await agent.llm_service.get_audit_response(constitution)
    if audit["risk_level"] >= 3:
        # High risk - abort creation
        cleanup_artifacts([agent.db_path, agent.key_path])
        return False
    return True
```

## Verification Commands

Check constitution embedding:
```
/constitution-check
```

## Recommendation

Use the constitution-verifier subagent for systematic verification:
```
"Use the constitution-verifier subagent to validate the inception process"
```

Or use the parallel workflow:
```
/parallel-work
```

## Key Principles

1. **Immutability** - Constitution hash is content-addressable
2. **Verifiability** - Any agent can prove its governance
3. **Sovereignty** - Each agent cryptographically bound to constitution
4. **Auditability** - Genesis self-audit prevents corruption
