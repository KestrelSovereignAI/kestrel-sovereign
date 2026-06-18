---
type: Diagram
title: 10 - Compute & Script Execution
description: Secure script execution with constitutional separation of powers.
resource: /docs/diagrams/10-compute-execution.md
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

# 10 - Compute & Script Execution

Secure script execution with constitutional separation of powers.

---

## Slide 1: The Problem - Agents Need to Run Code

```mermaid
graph TD
    subgraph problem["The Challenge"]
        P1["Agents need to execute code<br/>to be truly useful"]
        P2["But code execution is<br/>inherently dangerous"]
        P3["How do we enable power<br/>without enabling harm?"]
    end

    subgraph risks["Risks"]
        R1["🔓 Security exploits"]
        R2["💾 Data destruction"]
        R3["🌐 Network abuse"]
        R4["💸 Resource exhaustion"]
    end

    problem --> risks

    style problem fill:#7d3c00,stroke:#f5b041,stroke-width:2px
    style risks fill:#641e16,stroke:#ec7063
```

**The dilemma:** Useful agents must act. Acting requires code. Code can harm.

---

## Slide 2: The Solution - Separation of Powers

```mermaid
graph LR
    subgraph pattern["Write-Sign-Review-Execute Pattern"]
        W["✍️ WRITE<br/>Agent creates script"]
        S["🔏 SIGN<br/>Agent signs with DID"]
        R["🔍 REVIEW<br/>Security analysis"]
        E["⚡ EXECUTE<br/>Sandboxed run"]
    end

    W --> S --> R --> E

    subgraph principle["Constitutional Principle"]
        P["No entity can both<br/>write AND execute"]
    end

    style W fill:#1a5276,stroke:#85c1e9
    style S fill:#512e5f,stroke:#af7ac5
    style R fill:#7d6608,stroke:#f4d03f
    style E fill:#145a32,stroke:#58d68d
    style principle fill:#0e4d45,stroke:#48c9b0
```

**Like government branches:** Writer proposes, reviewer checks, executor acts.

---

## Slide 3: Script Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> DRAFT: Agent writes script
    DRAFT --> SIGNED: Agent signs with DID
    SIGNED --> PENDING_REVIEW: Submit for analysis
    PENDING_REVIEW --> APPROVED: Low risk / User approves
    PENDING_REVIEW --> REJECTED: High risk / User denies
    APPROVED --> QUEUED: Awaiting execution slot
    QUEUED --> RUNNING: Executor starts
    RUNNING --> COMPLETED: Exit code 0
    RUNNING --> FAILED: Exit code != 0
    COMPLETED --> [*]
    FAILED --> [*]
    REJECTED --> [*]
```

**Every script has a traceable journey.** No shortcuts.

---

## Slide 4: Script Signing with DID

```mermaid
sequenceDiagram
    participant A as Agent
    participant SS as ScriptSigner
    participant DB as Database

    A->>SS: Sign script content
    SS->>SS: Hash: SHA256(name|lang|content|purpose)
    SS->>SS: Sign with agent's DID key (secp256k1)
    SS-->>A: Signature (ecdsa:base64...)
    A->>DB: Store script + signature

    Note over SS: Same key used for<br/>Ethereum DID identity
```

**Cryptographic proof:** "I, agent X, authored this exact script."

---

## Slide 5: Script Signature Format

```mermaid
graph TD
    subgraph script["ComputeScript"]
        ID["id: UUID"]
        NAME["name: 'backup_data'"]
        LANG["language: python | bash"]
        CONTENT["content: 'import os...'"]
        PURPOSE["purpose: 'Backup user files'"]
    end

    subgraph signature["Signature"]
        HASH["payload_hash = SHA256(<br/>name|language|content|purpose)"]
        SIG["signature = ECDSA_Sign(payload_hash, did_key)"]
        FORMAT["format: 'ecdsa:BASE64...'"]
        SIGNER["signed_by: 'did:pkh:eip155:1:0x...'"]
    end

    script --> HASH --> SIG

    style script fill:#1a5276,stroke:#85c1e9
    style signature fill:#512e5f,stroke:#af7ac5
```

---

## Slide 6: Security Analysis & Risk Scoring

```mermaid
graph TD
    subgraph analyzer["Script Analyzer"]
        AST["Parse AST<br/>(Python/Bash)"]
        PATTERNS["Pattern Detection"]
        SCORE["Risk Score: 0-100"]
    end

    subgraph patterns["Dangerous Patterns"]
        CRIT["🔴 CRITICAL (50pts)<br/>fork bombs, format disk"]
        HIGH["🟠 HIGH (25pts)<br/>shell escapes, eval()"]
        MED["🟡 MEDIUM (10pts)<br/>network access, file delete"]
        LOW["🟢 LOW (5pts)<br/>subprocess, chmod"]
    end

    AST --> PATTERNS --> patterns
    patterns --> SCORE

    subgraph decision["Auto-Decision"]
        AUTO_DENY["Score > 80: AUTO DENY"]
        ASK["Score 20-80: ASK USER"]
        AUTO_ALLOW["Score < 20: May auto-allow"]
    end

    SCORE --> decision

    style CRIT fill:#641e16,stroke:#ec7063
    style HIGH fill:#7d3c00,stroke:#f5b041
    style MED fill:#7d6608,stroke:#f4d03f
    style LOW fill:#145a32,stroke:#58d68d
```

---

## Slide 7: Destructive Operation Policy

```mermaid
graph LR
    subgraph before["User Writes"]
        RM["rm -rf /data/old"]
    end

    subgraph rewrite["System Rewrites"]
        TRASH["mv /data/old ~/.kestrel/trash/2025-12-06T10:30:00/"]
    end

    subgraph result["Result"]
        SAFE["✅ Files in trash<br/>30-day retention"]
        RESTORE["!compute-restore<br/>recovers files"]
    end

    before -->|"Intercepted"| rewrite --> result

    style before fill:#641e16,stroke:#ec7063
    style rewrite fill:#7d6608,stroke:#f4d03f
    style result fill:#145a32,stroke:#58d68d
```

**rm NEVER truly deletes.** Everything goes to recoverable trash.

---

## Slide 8: Execution Sandboxes

```mermaid
graph TD
    subgraph executors["Available Executors"]
        UV["🐍 UvExecutor<br/>Python with uv run --isolated"]
        DOCKER["🐳 DockerExecutor<br/>Full container sandbox"]
        LOCAL["⚠️ LocalExecutor<br/>Development only"]
    end

    subgraph uv["UvExecutor Features"]
        UV1["Isolated virtual env"]
        UV2["Pip packages allowed"]
        UV3["No system access"]
    end

    subgraph docker["DockerExecutor Features"]
        D1["Read-only root filesystem"]
        D2["No network by default"]
        D3["CPU/memory limits"]
        D4["Timeout enforcement"]
    end

    UV --> uv
    DOCKER --> docker

    style UV fill:#1a5276,stroke:#85c1e9
    style DOCKER fill:#145a32,stroke:#58d68d
    style LOCAL fill:#641e16,stroke:#ec7063
```

---

## Slide 9: Approval Queue Integration

```mermaid
sequenceDiagram
    participant A as Agent
    participant S as SecurityFeature
    participant Q as ApprovalQueue
    participant U as User
    participant E as Executor

    A->>S: Request script execution
    S->>S: Analyze risk (score: 45)
    S->>Q: Queue for approval
    Q->>U: "Script 'backup_data' needs approval"

    alt User Approves
        U->>Q: Approve (scope: session)
        Q->>E: Execute script
        E-->>A: Result
    else User Denies
        U->>Q: Deny
        Q-->>A: Rejection
    end
```

---

## Slide 10: Compute Commands Reference

```mermaid
graph LR
    subgraph commands["Compute Commands"]
        C1["!compute-write name lang<br/>Create new script"]
        C2["!compute-run id<br/>Submit for execution"]
        C3["!compute-list<br/>Show all scripts"]
        C4["!compute-show id<br/>View script details"]
        C5["!compute-history<br/>Execution history"]
    end

    subgraph trash["Trash Management"]
        T1["!compute-trash<br/>List trashed items"]
        T2["!compute-restore path<br/>Recover from trash"]
        T3["!compute-empty-trash<br/>Permanent delete (approval required)"]
    end

    style commands fill:#1a5276,stroke:#85c1e9
    style trash fill:#7d3c00,stroke:#f5b041
```

---

## Slide 11: Risk Score Calculation

| Severity | Points | Examples |
|----------|--------|----------|
| 🔴 CRITICAL | 50 | `:(){ :|:& };:`, `mkfs`, `dd if=/dev/zero` |
| 🟠 HIGH | 25 | `eval()`, `exec()`, `subprocess.Popen(shell=True)` |
| 🟡 MEDIUM | 10 | `requests.get()`, `os.remove()`, `shutil.rmtree()` |
| 🟢 LOW | 5 | `subprocess.run()`, `chmod`, file I/O |
| ℹ️ INFO | 1 | `import os`, `import sys` |

**Max Score: 100** (capped to prevent overflow)

```python
# Auto-decision thresholds
if risk_score > 80:
    return AUTO_DENY  # Too dangerous
elif risk_score > 20:
    return ASK_USER   # User decides
else:
    return policy.auto_approve_low_risk  # Config-dependent
```

---

## Slide 12: Security Guarantees

```mermaid
graph TD
    subgraph guarantees["What We Guarantee"]
        G1["✅ Agent cannot execute<br/>without signing"]
        G2["✅ Signature is verifiable<br/>via DID"]
        G3["✅ High-risk scripts<br/>always need approval"]
        G4["✅ Destructive ops<br/>go to trash first"]
        G5["✅ Execution is sandboxed<br/>and time-limited"]
    end

    subgraph audit["Audit Trail"]
        A1["Every script stored"]
        A2["Every execution logged"]
        A3["Every approval recorded"]
    end

    guarantees --> audit

    style guarantees fill:#145a32,stroke:#58d68d
    style audit fill:#1a5276,stroke:#85c1e9
```

**Constitutional computing:** Power with accountability.

---

*Next: [11-a2a-protocol.md](11-a2a-protocol.md) - Agent-to-Agent communication protocol*
