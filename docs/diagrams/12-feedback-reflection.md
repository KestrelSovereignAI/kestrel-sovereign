---
type: Diagram
title: 12 - Feedback & Self-Reflection
description: Agent self-observation, feedback collection, and autonomous improvement.
resource: /docs/diagrams/12-feedback-reflection.md
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

# 12 - Feedback & Self-Reflection

Agent self-observation, feedback collection, and autonomous improvement.

> **Status:** ✅ IMPLEMENTED (2025-12-20)
>
> See `docs/plans/REFLECTION_FEATURE_PLAN.md` for full implementation details.
> Key features: Layered reflection (Arms→Memory→Mind→Action), Training cycle (`!train`), Sleep hook.

---

## Slide 1: The Problem - Agents Don't Learn from Users

```mermaid
graph TD
    subgraph problem["Current AI Systems"]
        P1["User gives thumbs down"]
        P2["Goes to centralized training"]
        P3["User never sees impact"]
        P4["No individual improvement"]
    end

    subgraph solution["Kestrel Approach"]
        S1["Feedback stored locally"]
        S2["Agent reflects on patterns"]
        S3["Proposes own improvements"]
        S4["User approves changes"]
    end

    problem -->|"vs"| solution

    style problem fill:#641e16,stroke:#ec7063
    style solution fill:#145a32,stroke:#58d68d
```

**Your feedback improves YOUR agent, not the vendor's model.**

---

## Slide 2: Feedback System Overview

```mermaid
graph LR
    subgraph collection["Feedback Collection"]
        EXPLICIT["👍👎 Explicit<br/>User ratings"]
        IMPLICIT["🔄 Implicit<br/>Regeneration, edits"]
        VERBAL["💬 Verbal<br/>'That's wrong'"]
    end

    subgraph storage["Feedback Storage"]
        FS["FeedbackStore<br/>Backend-agnostic"]
    end

    subgraph analysis["Analysis"]
        PATTERNS["Pattern detection"]
        INSIGHTS["Insight generation"]
        TICKETS["Improvement tickets"]
    end

    collection --> storage --> analysis

    style EXPLICIT fill:#145a32,stroke:#58d68d
    style IMPLICIT fill:#7d6608,stroke:#f4d03f
    style VERBAL fill:#1a5276,stroke:#85c1e9
```

---

## Slide 3: Feedback Categories & Severity

| Category | Severity | Example | Action |
|----------|----------|---------|--------|
| 🐛 Bug | Critical | "Wrong calculation" | Immediate logging |
| ❌ Incorrect | High | "That's not true" | Store + flag for review |
| 🔄 Improvement | Medium | "Could be clearer" | Pattern analysis |
| 💡 Suggestion | Low | "Add dark mode" | Feature backlog |
| ℹ️ Neutral | Info | "Thanks!" | Positive signal |

```mermaid
graph LR
    BUG["🐛 Bug"] --> CRIT["Critical<br/>Immediate action"]
    WRONG["❌ Incorrect"] --> HIGH["High<br/>Flag for review"]
    IMPROVE["🔄 Improvement"] --> MED["Medium<br/>Pattern analysis"]
    SUGGEST["💡 Suggestion"] --> LOW["Low<br/>Feature backlog"]

    style BUG fill:#641e16,stroke:#ec7063
    style WRONG fill:#7d3c00,stroke:#f5b041
    style IMPROVE fill:#7d6608,stroke:#f4d03f
    style SUGGEST fill:#145a32,stroke:#58d68d
```

---

## Slide 4: Abstract Store Pattern

```mermaid
graph TD
    subgraph interface["FeedbackStore (ABC)"]
        M1["submit_feedback()"]
        M2["get_feedback()"]
        M3["analyze_patterns()"]
    end

    subgraph backends["Backend Implementations"]
        SQLITE["SQLiteFeedbackStore<br/>Kestrel standalone"]
        POSTGRES["PostgresFeedbackStore<br/>Kestrel multi-tenant"]
    end

    subgraph features["Features"]
        F1["Works offline (SQLite)"]
        F2["Scales to millions (PostgreSQL)"]
        F3["Same API, different backends"]
    end

    interface --> backends --> features

    style interface fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style SQLITE fill:#1a5276,stroke:#85c1e9
    style POSTGRES fill:#512e5f,stroke:#af7ac5
```

**Write once, run anywhere.**

---

## Slide 5: Layered Reflection Architecture (IMPLEMENTED)

```mermaid
graph TD
    subgraph trigger["Reflection Triggers"]
        T1["⏰ Sleep Hook<br/>Nightly consolidation"]
        T2["🏋️ !train<br/>Intensive training"]
        T3["👤 !reflect<br/>On-demand"]
    end

    subgraph layers["4-Layer Reflection"]
        L1["🦾 ARMS<br/>Do my components work?"]
        L2["🧠 MEMORY<br/>Can I access what I know?"]
        L3["💭 MIND<br/>Is my reasoning good?"]
        L4["⚡ ACTION<br/>What's the quickest fix?"]
    end

    subgraph output["Outputs"]
        HEALTH["Health Score<br/>pass/warn/critical"]
        ACTIONS["Action Items<br/>prioritized by severity"]
        TICKET["🎫 GitHub Issue<br/>(with approval)"]
    end

    trigger --> L1 --> L2 --> L3 --> L4 --> output

    style L1 fill:#1a5276,stroke:#85c1e9
    style L2 fill:#7d6608,stroke:#f4d03f
    style L3 fill:#512e5f,stroke:#af7ac5
    style L4 fill:#145a32,stroke:#58d68d
```

**Concrete before abstract.** Check if arms work before analyzing thoughts.

### Layer Details
| Layer | Question | Checks |
|-------|----------|--------|
| **1. Arms** | Do my components work? | LLM, Storage, Encryption, Features, Tools |
| **2. Memory** | Can I access what I know? | Constitution, Conversations, Graph, RAG |
| **3. Mind** | Is my reasoning good? | Response Coherence, Interaction Patterns |
| **4. Action** | What's the quickest fix? | Prioritized by severity (CRITICAL → LOW) |

**Stop-on-Critical:** If Layer 1 or 2 has CRITICAL failures, stops immediately.

---

## Slide 6: Constitutional Approval for Self-Modification

```mermaid
sequenceDiagram
    participant A as Agent
    participant R as ReflectionEngine
    participant C as ConstitutionalReviewer
    participant U as User

    A->>R: Trigger reflection
    R->>R: Analyze feedback patterns
    R->>R: Generate improvement proposal

    R->>C: Submit for constitutional review
    C->>C: Check against governance rules

    alt Approved
        C->>U: "Agent proposes: improve X"
        U->>A: Approve/Deny
        alt User Approves
            A->>A: Apply improvement
        end
    else Rejected
        C->>A: Log rejection reason
    end
```

**No self-modification without governance approval.**

---

## Slide 7: Improvement Ticket Creation

```mermaid
graph TD
    subgraph gates["Economic Gates"]
        FREE["Free Tier<br/>❌ No ticket creation"]
        PAID["Paid Tier<br/>✅ Create tickets"]
        SHARE["Revenue Share<br/>✅ Priority tickets"]
    end

    subgraph flow["Ticket Flow"]
        INSIGHT["Agent identifies issue"]
        DRAFT["Draft ticket content"]
        REVIEW["Constitutional review"]
        CREATE["Create GitHub Issue"]
    end

    subgraph result["Result"]
        ISSUE["GitHub Issue<br/>with agent context"]
        TRACK["Trackable improvement"]
    end

    gates --> flow --> result

    style FREE fill:#641e16,stroke:#ec7063
    style PAID fill:#145a32,stroke:#58d68d
    style SHARE fill:#7d6608,stroke:#f4d03f
```

**Economic gates prevent spam.** Paid users get improvement proposals.

---

## Slide 8: Self-Model Storage

```mermaid
graph LR
    subgraph self_model["Agent Self-Model"]
        STRENGTHS["💪 Strengths<br/>'Good at summarization'"]
        WEAKNESSES["⚠️ Weaknesses<br/>'Struggles with dates'"]
        PREFERENCES["⚙️ Preferences<br/>'User likes bullets'"]
    end

    subgraph storage["Permanent Storage"]
        IPFS["IPFS/Filecoin<br/>Sovereignty export"]
    end

    subgraph recovery["Recovery"]
        RESTORE["Restore self-model<br/>after migration"]
    end

    self_model --> storage --> recovery

    style self_model fill:#1a5276,stroke:#85c1e9
    style storage fill:#512e5f,stroke:#af7ac5
    style recovery fill:#145a32,stroke:#58d68d
```

**Self-knowledge persists.** Even across platform changes.

---

## Slide 9: Reflection Commands (IMPLEMENTED)

```mermaid
graph LR
    subgraph reflection["Reflection Commands"]
        C1["!reflect<br/>Run layered reflection"]
        C2["!reflect all deep<br/>Deep analysis"]
        C3["!train<br/>Intensive training cycle"]
        C4["!insights<br/>View past insights"]
    end

    subgraph feedback["Feedback Commands"]
        C5["!feedback message<br/>Submit feedback"]
        C6["!self-model<br/>View self-model"]
        C7["!create-ticket id<br/>Create GitHub issue"]
    end

    subgraph results["Returns"]
        R1["Layer status + actions"]
        R2["Health trend"]
        R3["Insight list"]
        R4["Issue URL"]
    end

    reflection --> results
    feedback --> results

    style reflection fill:#145a32,stroke:#58d68d
    style feedback fill:#1a5276,stroke:#85c1e9
```

---

## Slide 10: Reflection Cycle

```mermaid
graph TD
    subgraph cycle["Continuous Improvement Cycle"]
        INTERACT["💬 User Interaction"]
        FEEDBACK["📝 Feedback Collected"]
        STORE["💾 Store in FeedbackStore"]
        REFLECT["🔍 Nightly Reflection"]
        INSIGHT["💡 Generate Insights"]
        PROPOSE["📋 Propose Changes"]
        APPROVE["✅ Constitutional Approval"]
        APPLY["🔧 Apply Improvements"]
    end

    INTERACT --> FEEDBACK --> STORE --> REFLECT
    REFLECT --> INSIGHT --> PROPOSE --> APPROVE --> APPLY
    APPLY --> INTERACT

    style INTERACT fill:#1a5276,stroke:#85c1e9
    style FEEDBACK fill:#7d6608,stroke:#f4d03f
    style REFLECT fill:#512e5f,stroke:#af7ac5
    style APPLY fill:#145a32,stroke:#58d68d
```

**Your agent gets better every day.**

---

*Next: Data Architecture Deep Dive - See [data-architecture/](data-architecture/) for detailed storage documentation*
