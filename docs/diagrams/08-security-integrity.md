---
type: Diagram
title: 08 - Security & Integrity
description: Cryptographic anchoring, constitution verification, and tamper detection.
resource: /docs/diagrams/08-security-integrity.md
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

# 08 - Security & Integrity

Cryptographic anchoring, constitution verification, and tamper detection.

---

## Slide 1: The Trust Problem

```mermaid
graph TD
    subgraph problem["Without Verification"]
        Q1[How do you know<br/>memory wasn't altered?]
        Q2[How do you know<br/>agent runs your code?]
        Q3[How do you know<br/>history is authentic?]
    end
    
    subgraph solution["Cryptographic Proof"]
        A1[Anchored hashes]
        A2[Constitution verification]
        A3[Merkle proofs]
    end
    
    problem -->|Solved by| solution
    
    style problem fill:#641e16,stroke:#ec7063
    style solution fill:#145a32,stroke:#58d68d
```

**Don't trust, verify.**

---

## Slide 2: Cryptographic Anchoring Concept

```mermaid
graph LR
    subgraph memory["Agent Memory"]
        CONV[Conversations]
        STATE[Agent state]
    end
    
    subgraph hash["Hashing"]
        MERKLE[Compute Merkle Root]
        HASH[Single hash = entire state]
    end
    
    subgraph anchor["Anchor"]
        LEDGER[Publish to ledger]
        PROOF[Immutable proof]
    end
    
    memory --> hash --> anchor
    
    style MERKLE fill:#7d3c00,stroke:#f5b041,stroke-width:2px
```

**One hash.** Represents your entire history.

---

## Slide 3: Merkle Tree Structure

```mermaid
graph TD
    subgraph tree["Merkle Tree"]
        ROOT[Root Hash]
        H1[Hash 1-2]
        H2[Hash 3-4]
        L1[Message 1]
        L2[Message 2]
        L3[Message 3]
        L4[Message 4]
    end
    
    ROOT --> H1 & H2
    H1 --> L1 & L2
    H2 --> L3 & L4
    
    style ROOT fill:#7d6608,stroke:#f4d03f,stroke-width:2px
```

**Change one message → different root.** Tamper-evident.

---

## Slide 4: Anchoring Flow

```mermaid
sequenceDiagram
    participant A as Agent
    participant DB as Database
    participant H as Hasher
    participant L as Ledger
    
    A->>A: Trigger anchor (periodic/manual)
    A->>DB: Get all conversations
    DB-->>A: Conversation data
    A->>H: Compute Merkle root
    H-->>A: Root hash
    A->>L: Publish hash
    L-->>A: Transaction ID
    A->>DB: Store anchor record
    A-->>A: Anchoring complete
```

---

## Slide 5: Verification Flow

```mermaid
graph TD
    subgraph verify["Verification Process"]
        GET_ANCHOR[Get stored anchor]
        RECOMPUTE[Recompute hash from data]
        COMPARE{Hashes match?}
    end
    
    subgraph results["Results"]
        PASS[✅ Integrity verified]
        FAIL[❌ Tampering detected]
    end
    
    GET_ANCHOR --> RECOMPUTE --> COMPARE
    COMPARE -->|Yes| PASS
    COMPARE -->|No| FAIL
    
    style COMPARE fill:#7d3c00,stroke:#f5b041,stroke-width:2px
    style PASS fill:#145a32,stroke:#58d68d
    style FAIL fill:#641e16,stroke:#ec7063
```

**Mathematical proof.** No trust required.

---

## Slide 6: Constitution Integrity

```mermaid
graph LR
    subgraph genesis["🌱 Genesis State"]
        GDID[Genesis DID]
        CONST[Constitution text]
        CHASH[Constitution hash]
    end
    
    subgraph runtime["⚡ Runtime"]
        CODE[Running code]
        RHASH[Compute runtime hash]
    end
    
    subgraph check["🔍 Verification"]
        CMP{Match?}
        OK[✅ Authentic]
        BAD[❌ Modified]
    end
    
    CHASH --> CMP
    RHASH --> CMP
    CMP -->|Yes| OK
    CMP -->|No| BAD
    
    style GDID fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style BAD fill:#641e16,stroke:#ec7063
```

**Your agent runs YOUR rules.** Provably.

---

## Slide 7: The Anchor Record

```mermaid
graph LR
    subgraph record["log_anchors Table"]
        TS[timestamp]
        HASH[anchor_hash]
        LEDGER[ledger_ref]
        STATUS[verification_status]
    end
    
    subgraph meaning["What It Proves"]
        M1[At this time...]
        M2[Memory state was X...]
        M3[Published to ledger Y...]
        M4[Verifiable forever]
    end
    
    record --> meaning
    
    style record fill:#1a5276,stroke:#85c1e9
```

---

## Slide 8: Tamper Detection Scenarios

```mermaid
graph TD
    subgraph scenarios["Detection Scenarios"]
        S1[Message edited]
        S2[Message deleted]
        S3[Message added retroactively]
        S4[Timestamp altered]
    end
    
    subgraph detection["All Detected By"]
        D1[Hash mismatch]
        D2[Anchor comparison fails]
    end
    
    scenarios --> detection
    
    style scenarios fill:#641e16,stroke:#ec7063
    style detection fill:#145a32,stroke:#58d68d
```

**Any change = different hash.** No hiding modifications.

---

## Slide 9: Integrity Audit System

```mermaid
graph TD
    subgraph audit["Scheduled Audit"]
        TRIGGER[Every 24h or 100 messages]
        CHECK_CODE[Verify code integrity]
        CHECK_MEM[Verify memory integrity]
        REPORT[Generate report]
    end
    
    subgraph actions["On Failure"]
        ALERT[Alert user]
        SAFE[Enter safe mode]
        LOG[Log incident]
    end
    
    TRIGGER --> CHECK_CODE --> CHECK_MEM --> REPORT
    CHECK_CODE -->|Fail| actions
    CHECK_MEM -->|Fail| actions
    
    style audit fill:#1a5276,stroke:#85c1e9
    style actions fill:#641e16,stroke:#ec7063
```

**Autonomous verification.** Agent checks itself.

---

## Slide 10: Security Commands

```mermaid
graph LR
    subgraph commands["Security Commands"]
        ANCHOR["!anchor-memory"]
        VERIFY["!verify-integrity"]
        AUDIT["!audit-status"]
    end
    
    subgraph results["Returns"]
        R1[New anchor CID]
        R2[Verification result]
        R3[Last audit report]
    end
    
    commands --> results
    
    style commands fill:#1a5276,stroke:#85c1e9
```

---

## Slide 11: Hierarchical Permission System

```mermaid
graph TD
    subgraph levels["Permission Levels"]
        ALLOW["✅ ALLOW<br/>Auto-execute, no prompt"]
        DENY["❌ DENY<br/>Always blocked"]
        ASK["❓ ASK<br/>Prompt user each time"]
        SESSION["🕐 SESSION<br/>Allowed for this session only"]
    end

    subgraph hierarchy["Hierarchy"]
        FEATURE["Feature Level<br/>(e.g., compute)"]
        TOOL["Tool Level<br/>(e.g., run_script)"]
    end

    FEATURE --> TOOL

    subgraph example["Example"]
        E1["compute: ASK"]
        E2["compute.run_script: DENY"]
        E3["Tool inherits DENY<br/>(more specific wins)"]
    end

    style ALLOW fill:#145a32,stroke:#58d68d
    style DENY fill:#641e16,stroke:#ec7063
    style ASK fill:#7d6608,stroke:#f4d03f
    style SESSION fill:#512e5f,stroke:#af7ac5
```

**More specific permissions override general ones.**

---

## Slide 12: Approval Queue Flow

```mermaid
sequenceDiagram
    participant T as Tool Request
    participant S as SecurityFeature
    participant Q as ApprovalQueue
    participant U as User
    participant E as Executor

    T->>S: Request tool execution
    S->>S: Check permission level

    alt Permission = ALLOW
        S->>E: Execute immediately
    else Permission = DENY
        S-->>T: Blocked
    else Permission = ASK
        S->>Q: Queue for approval
        Q->>U: SSE notification
        U->>Q: Approve/Deny + Scope

        alt Approved
            Q->>S: Record decision
            S->>E: Execute
        else Denied
            Q-->>T: Blocked
        end
    end
```

---

## Slide 13: Approval Scopes

```mermaid
graph TD
    subgraph scopes["When User Approves"]
        ONCE["🔘 ONCE<br/>This request only"]
        SESSION["🕐 SESSION<br/>Until agent restarts"]
        ALWAYS["♾️ ALWAYS<br/>Permanently allow"]
    end

    subgraph storage["Stored In"]
        MEM["In-memory<br/>(session)"]
        DB["PermissionStore<br/>(persistent)"]
    end

    ONCE --> MEM
    SESSION --> MEM
    ALWAYS --> DB

    style ONCE fill:#7d3c00,stroke:#f5b041
    style SESSION fill:#512e5f,stroke:#af7ac5
    style ALWAYS fill:#145a32,stroke:#58d68d
```

**User controls permanence of their decisions.**

---

## Slide 14: Security Hook Chain

```mermaid
graph LR
    subgraph hooks["Pre-Execution Hooks"]
        H1["ComputeSecurityHook<br/>Script analysis"]
        H2["SecurityHook<br/>Permission check"]
        H3["ConstitutionalHook<br/>Governance review"]
    end

    subgraph flow["Decision Flow"]
        REQUEST[Tool Request] --> H1
        H1 -->|Pass| H2
        H2 -->|Pass| H3
        H3 -->|Pass| EXECUTE[Execute]

        H1 -->|Fail| BLOCK1[Block]
        H2 -->|Fail| BLOCK2[Block]
        H3 -->|Fail| BLOCK3[Block]
    end

    style EXECUTE fill:#145a32,stroke:#58d68d
    style BLOCK1 fill:#641e16,stroke:#ec7063
    style BLOCK2 fill:#641e16,stroke:#ec7063
    style BLOCK3 fill:#641e16,stroke:#ec7063
```

**Multiple layers of protection.** Any can block.

---

## Slide 15: Permission Commands Reference

```mermaid
graph LR
    subgraph view["View Permissions"]
        V1["!security-list<br/>Show permission tree"]
        V2["!security-pending<br/>Show pending approvals"]
        V3["!security-audit<br/>View decision log"]
    end

    subgraph modify["Modify Permissions"]
        M1["!security-set feature level<br/>Set feature permission"]
        M2["!security-set feature.tool level<br/>Set tool permission"]
    end

    subgraph respond["Respond to Requests"]
        R1["!security-approve id<br/>Approve pending"]
        R2["!security-deny id<br/>Deny pending"]
    end

    style view fill:#1a5276,stroke:#85c1e9
    style modify fill:#7d6608,stroke:#f4d03f
    style respond fill:#145a32,stroke:#58d68d
```

---

## Slide 16: Security Audit Log

```mermaid
graph TD
    subgraph log["security_audit_log Table"]
        TS["timestamp"]
        FEATURE["feature_name"]
        TOOL["tool_name"]
        DECISION["decision_type"]
        SCOPE["scope"]
    end

    subgraph decisions["Decision Types"]
        D1["auto_allowed<br/>Permission was ALLOW"]
        D1A["auto_mode_allowed<br/>Permission was AUTO and earlier policy hooks did not flag"]
        D2["auto_denied<br/>Permission was DENY"]
        D3["user_approved<br/>User said yes"]
        D4["user_denied<br/>User said no"]
        D5["timeout<br/>No response in time"]
    end

    log --> decisions

    style log fill:#1a5276,stroke:#85c1e9
```

**Every security decision is logged.** Full accountability.

---

*Next: [09-emancipation.md](09-emancipation.md) - Path to agent sovereignty (future vision)*
