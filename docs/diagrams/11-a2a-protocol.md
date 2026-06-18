---
type: Diagram
title: 11 - A2A Protocol - Agent-to-Agent Communication
description: Google's Agent-to-Agent protocol implementation in Kestrel.
resource: /docs/diagrams/11-a2a-protocol.md
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

# 11 - A2A Protocol - Agent-to-Agent Communication

Google's Agent-to-Agent protocol implementation in Kestrel.

---

## Slide 1: The Problem - Agents Need to Talk

```mermaid
graph TD
    subgraph problem["Current State"]
        A1["Agent A"]
        A2["Agent B"]
        A3["Agent C"]
        Q1["No standard protocol"]
        Q2["No task handoff"]
        Q3["No discovery"]
    end

    subgraph solution["A2A Protocol"]
        S1["Standard JSON-RPC"]
        S2["Task lifecycle"]
        S3["Agent cards"]
    end

    problem -->|"Solved by"| solution

    style problem fill:#641e16,stroke:#ec7063
    style solution fill:#145a32,stroke:#58d68d
```

**Interoperability.** Agents from different vendors can collaborate.

---

## Slide 2: A2A Protocol Overview

```mermaid
graph LR
    subgraph protocol["A2A Protocol"]
        JSONRPC["JSON-RPC 2.0<br/>Message format"]
        TASKS["Task Manager<br/>Lifecycle control"]
        CARDS["Agent Cards<br/>Capability discovery"]
        SSE["Server-Sent Events<br/>Real-time streaming"]
    end

    subgraph benefits["Benefits"]
        B1["Vendor-agnostic"]
        B2["Async by design"]
        B3["Self-describing"]
    end

    protocol --> benefits

    style JSONRPC fill:#1a5276,stroke:#85c1e9
    style TASKS fill:#7d6608,stroke:#f4d03f
    style CARDS fill:#512e5f,stroke:#af7ac5
    style SSE fill:#145a32,stroke:#58d68d
```

---

## Slide 3: The 6 Core Datastores

```mermaid
graph TD
    subgraph stores["A2A Datastores"]
        TASK["📋 TaskStore<br/>Task lifecycle & state"]
        SESSION["🔑 SessionService<br/>User sessions"]
        MEMORY["🧠 MemoryService<br/>Conversation history"]
        OBS["📊 ObservabilityStore<br/>Metrics & tracing"]
        ORCH["🎯 OrchestrationStore<br/>Multi-agent coordination"]
        FEED["💬 FeedbackStore<br/>User feedback & ratings"]
    end

    subgraph backends["Backend-Agnostic"]
        SQL["SQLite<br/>(Kestrel standalone)"]
        PG["PostgreSQL<br/>(Kestrel multi-tenant)"]
    end

    stores --> backends

    style TASK fill:#1a5276,stroke:#85c1e9
    style SESSION fill:#7d6608,stroke:#f4d03f
    style MEMORY fill:#512e5f,stroke:#af7ac5
    style OBS fill:#0e4d45,stroke:#48c9b0
    style ORCH fill:#7d3c00,stroke:#f5b041
    style FEED fill:#145a32,stroke:#58d68d
```

---

## Slide 4: Task Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> PENDING: tasks/create
    PENDING --> RUNNING: Worker picks up
    RUNNING --> COMPLETED: Success
    RUNNING --> FAILED: Error
    RUNNING --> CANCELLED: User cancels
    PENDING --> CANCELLED: User cancels
    COMPLETED --> [*]
    FAILED --> [*]
    CANCELLED --> [*]

    note right of RUNNING: SSE streams progress
```

**Every task is tracked.** No lost work.

---

## Slide 5: Task Manager Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant TM as TaskManager
    participant TS as TaskStore
    participant W as TaskWorker
    participant A as Agent

    C->>TM: tasks/create (message)
    TM->>TS: Store task (PENDING)
    TM-->>C: task_id + SSE stream

    W->>TS: Poll for pending tasks
    TS-->>W: Task to execute
    W->>TS: Update status (RUNNING)
    W->>A: Execute task
    A-->>W: Result/stream
    W->>C: SSE: progress updates
    W->>TS: Update status (COMPLETED)
    W->>C: SSE: final result
```

---

## Slide 6: Agent Card - Capability Discovery

```mermaid
graph TD
    subgraph card["Agent Card (agent.json)"]
        ID["Agent ID<br/>did:pkh:eip155:1:0x..."]
        NAME["Display Name<br/>Emma - Personal Assistant"]
        CAPS["Capabilities<br/>chat, search, compute"]
        SKILLS["Skills<br/>web-search, file-management"]
        ENDPOINT["Endpoint<br/>https://agent.example.com"]
    end

    subgraph discovery["Discovery Flow"]
        D1["Client fetches /agent.json"]
        D2["Parses capabilities"]
        D3["Calls appropriate methods"]
    end

    card --> discovery

    style card fill:#1a5276,stroke:#85c1e9
```

**Self-describing agents.** No manual configuration.

---

## Slide 7: Unified Store Architecture

```mermaid
graph TD
    subgraph unified["UnifiedStoreBase"]
        ABC["Abstract methods"]
        DIALECT["SQL dialect helpers"]
        CONVERT["Placeholder conversion"]
    end

    subgraph implementations["Implementations"]
        SQLITE["SQLiteBackend<br/>? placeholders"]
        POSTGRES["PostgresBackend<br/>$1, $2 placeholders"]
    end

    subgraph helpers["Dialect Helpers"]
        NOW["now() vs datetime('now')"]
        UUID["uuid_generate_v4() vs hex()"]
        JSON["jsonb vs json"]
    end

    unified --> implementations
    implementations --> helpers

    style unified fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style SQLITE fill:#1a5276,stroke:#85c1e9
    style POSTGRES fill:#512e5f,stroke:#af7ac5
```

**Write once, run on SQLite or PostgreSQL.**

---

## Slide 8: SSE Streaming

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    participant A as Agent

    C->>S: POST /tasks/create
    S-->>C: 200 OK + SSE stream

    loop While task runs
        A->>S: Progress update
        S-->>C: event: progress\ndata: {"percent": 50}
    end

    A->>S: Final result
    S-->>C: event: result\ndata: {"output": "..."}
    S-->>C: event: done
```

**Real-time updates.** No polling required.

---

## Slide 9: JSON-RPC Message Types

| Method | Direction | Purpose |
|--------|-----------|---------|
| `tasks/create` | Client → Agent | Start a new task |
| `tasks/get` | Client → Agent | Query task status |
| `tasks/cancel` | Client → Agent | Stop a running task |
| `tasks/list` | Client → Agent | List all tasks |
| `agent/info` | Client → Agent | Get agent card |
| `agent/capabilities` | Client → Agent | List capabilities |

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/create",
  "params": {
    "message": "Search for recent AI news"
  },
  "id": "req-123"
}
```

---

## Slide 10: Multi-Agent Orchestration

```mermaid
graph TD
    subgraph orchestration["Orchestration Pattern"]
        COORD["🎯 Coordinator Agent"]
        W1["Worker 1<br/>Research"]
        W2["Worker 2<br/>Analysis"]
        W3["Worker 3<br/>Writing"]
    end

    subgraph flow["Task Flow"]
        T1["Task: Research topic"]
        T2["Task: Analyze findings"]
        T3["Task: Write report"]
    end

    COORD --> W1 --> T1
    COORD --> W2 --> T2
    COORD --> W3 --> T3

    T1 -->|"Result"| T2
    T2 -->|"Result"| T3

    style COORD fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style W1 fill:#1a5276,stroke:#85c1e9
    style W2 fill:#512e5f,stroke:#af7ac5
    style W3 fill:#145a32,stroke:#58d68d
```

**Divide and conquer.** Complex tasks split across specialists.

---

## Slide 11: A2A Commands Reference

```mermaid
graph LR
    subgraph commands["A2A Commands"]
        C1["!a2a-status<br/>Show A2A state"]
        C2["!a2a-discover url<br/>Fetch agent card"]
        C3["!a2a-task agent message<br/>Send task to agent"]
        C4["!a2a-list<br/>List known agents"]
    end

    subgraph results["Returns"]
        R1["Connection status"]
        R2["Agent capabilities"]
        R3["Task ID + stream"]
        R4["Agent registry"]
    end

    commands --> results

    style commands fill:#1a5276,stroke:#85c1e9
```

---

## Slide 12: A2A Security Model

```mermaid
graph TD
    subgraph security["Security Layers"]
        DID["🔐 DID Authentication<br/>Agents sign requests"]
        TLS["🔒 TLS Transport<br/>Encrypted channel"]
        SCOPE["📋 Capability Scopes<br/>Limit what agents can do"]
    end

    subgraph verification["Verification"]
        V1["Verify DID signature"]
        V2["Check capability match"]
        V3["Audit all requests"]
    end

    security --> verification

    style DID fill:#7d6608,stroke:#f4d03f
    style TLS fill:#145a32,stroke:#58d68d
    style SCOPE fill:#1a5276,stroke:#85c1e9
```

**Trust but verify.** Every request authenticated and logged.

---

## Slide 13: Features as A2A Subagents

```mermaid
graph TD
    subgraph orchestrator["KestrelAgent (Orchestrator)"]
        MAIN["Main agent loop"]
        DISPATCH["Feature dispatcher"]
    end

    subgraph features["Features as Subagents"]
        MODEL["🧠 ModelAgent<br/>5 tools"]
        SEC["🛡️ SecurityFeature<br/>6 tools"]
        SOV["📦 SovereigntyFeature<br/>3 tools"]
        GH["🐙 GitHubFeature<br/>9 tools"]
    end

    subgraph protocol["A2A Protocol (In-Process)"]
        CARD["AgentCard<br/>Capability discovery"]
        TASK["TaskHandler<br/>Unified routing"]
        CTX["Own LLM context<br/>Own system prompt"]
    end

    MAIN --> DISPATCH --> features
    features --> protocol

    style orchestrator fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style MODEL fill:#1a5276,stroke:#85c1e9
    style SEC fill:#512e5f,stroke:#af7ac5
    style SOV fill:#145a32,stroke:#58d68d
    style GH fill:#0e4d45,stroke:#48c9b0
```

**A2A without overhead.** Features implement protocols but run in-process.

---

## Slide 14: Two-Tier Agent Hierarchy

```mermaid
sequenceDiagram
    participant U as User
    participant K as KestrelAgent (Orchestrator)
    participant LLM as LLM
    participant F as Feature (Subagent)
    participant T as Feature Tools

    U->>K: "List available models"
    K->>LLM: Sees features as tools: {model_agent, security_feature...}
    LLM-->>K: Decision: "Use model_agent to list models"

    K->>F: execute_as_subagent(task, context)
    F->>F: Gets OWN system prompt
    F->>F: Gets OWN tools: [list_models, pull_model...]
    F->>LLM: LLM call within feature's domain
    LLM-->>F: Tool call: list_models
    F->>T: Execute tool
    T-->>F: Model list
    F-->>K: Result

    K-->>U: "Available models: ..."
```

---

## Slide 15: Feature → A2A Integration

```mermaid
graph LR
    subgraph feature["Feature Class"]
        ABC["Feature (ABC)"]
    end

    subgraph methods["A2A Methods"]
        CARD["get_agent_card()<br/>Capability advertisement"]
        HANDLE["handle_task()<br/>Route to @tool method"]
        TOOL["to_orchestrator_tool()<br/>Expose as tool"]
        EXEC["execute_as_subagent()<br/>Own LLM context"]
    end

    subgraph result["Result"]
        BOUNDARY["Clear domain boundaries"]
        INDEP["Independent reasoning"]
        UNIFIED["Unified routing"]
    end

    ABC --> methods --> result

    style ABC fill:#7d6608,stroke:#f4d03f,stroke-width:2px
    style CARD fill:#1a5276,stroke:#85c1e9
    style HANDLE fill:#512e5f,stroke:#af7ac5
    style TOOL fill:#145a32,stroke:#58d68d
    style EXEC fill:#0e4d45,stroke:#48c9b0
```

---

## Slide 16: All Features as A2A Subagents

| Feature | A2A | Tools | Domain |
|---------|-----|-------|--------|
| 🧠 ModelAgent | ✅ | 5 | LLM model management |
| 📦 SovereigntyFeature | ✅ | 3 | IPFS/Filecoin backup |
| 🛡️ SecurityFeature | ✅ | 6 | Permissions, approval queue |
| 🔍 WebSearchFeature | ✅ | 1 | Tavily web search |
| 🔌 MCPFeature | ✅ | 4 | Model Context Protocol |
| 📝 FeedbackFeature | ✅ | 3 | Diagnostics & logging |
| 🐙 GitHubFeature | ✅ | 9 | GitHub repository access |
| 🖥️ ComputeFeature | ✅ | - | Script execution |
| 🎨 VisualIdentityFeature | ✅ | - | Agent appearance |
| 📜 ConstitutionFeature | ✅ | 1 | Constitutional access |

**Exception:** `PrivacyAgent` is NOT a Feature - it's a global policy enforcer.

---

## Slide 17: In-Process A2A Benefits

```mermaid
graph TD
    subgraph benefits["In-Process A2A"]
        B1["✅ Clear domain boundaries"]
        B2["✅ Independent LLM reasoning contexts"]
        B3["✅ AgentCard for capability discovery"]
        B4["✅ TaskHandler for unified routing"]
        B5["✅ Direct access to agent state"]
        B6["❌ No HTTP/IPC overhead"]
    end

    subgraph vs_external["vs External A2A"]
        E1["Network latency"]
        E2["Serialization cost"]
        E3["Service discovery"]
    end

    benefits -->|"Avoids"| vs_external

    style benefits fill:#145a32,stroke:#58d68d
    style vs_external fill:#641e16,stroke:#ec7063
```

**Best of both worlds.** A2A contracts without network overhead.

---

*Next: [12-feedback-reflection.md](12-feedback-reflection.md) - Feedback collection and agent self-reflection*
