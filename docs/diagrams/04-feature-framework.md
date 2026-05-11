# 04 - Feature Framework

Kestrel's dynamic feature system - how capabilities are built, discovered, and extended.

---

## Slide 1: The Problem with Monolithic Agents

```mermaid
graph TD
    subgraph monolith["❌ Monolithic Agent"]
        AGENT[Giant Agent Class]
        AGENT --> F1[Storage code]
        AGENT --> F2[Privacy code]
        AGENT --> F3[Wallet code]
        AGENT --> F4[LLM code]
        AGENT --> F5[Command code]
        AGENT --> F6[...everything else]
    end
    
    subgraph problems["Problems"]
        P1[Hard to maintain]
        P2[Hard to test]
        P3[Hard to extend]
        P4[Tight coupling]
    end
    
    monolith --> problems
    
    style monolith fill:#641e16,stroke:#ec7063
    style problems fill:#7d3c00,stroke:#f5b041
```

---

## Slide 2: The Feature Agent Solution

```mermaid
graph TD
    subgraph orchestrator["🦅 KestrelAgent - Orchestrator"]
        KA[Delegates to features]
    end
    
    subgraph features["Feature Agents"]
        F1[💰 WalletAgent]
        F2[🔒 PrivacyAgent]
        F3[📦 SovereigntyFeature]
        F4[🧠 ModelAgent]
        F5[🔌 MCPAgent]
        F6[🎨 CreativeFeature]
    end
    
    KA --> F1 & F2 & F3 & F4 & F5 & F6
    
    style KA fill:#1a5276,stroke:#85c1e9,stroke-width:3px
    style orchestrator fill:#1a3c5c,stroke:#5dade2
```

**Single Responsibility:** Each feature does one thing well.

---

## Slide 3: Feature Directory Structure

```mermaid
graph LR
    subgraph structure["📁 features/"]
        DIR[features/]
        DIR --> W[wallet/]
        DIR --> P[privacy/]
        DIR --> S[sovereignty/]
        DIR --> M[model/]
        DIR --> MCP[mcp/]
        DIR --> C[creative/]
        DIR --> BASE[base.py]
    end
    
    subgraph contents["Each Feature Has"]
        INIT[__init__.py]
        FEAT[feature.py]
        TOOL[tools defined]
    end
    
    W --> contents
    
    style DIR fill:#1a5276,stroke:#85c1e9
```

**Convention over configuration.** Drop a feature, it works.

---

## Slide 4: The @tool Decorator

```mermaid
graph TD
    subgraph code["Feature Code"]
        DEC["@tool(name, description, command)"]
        METHOD[async def export_sovereignty]
    end
    
    subgraph creates["Decorator Creates"]
        SCHEMA[ToolSchema]
        NAME[name: export_sovereignty]
        DESC[description: Export agent...]
        CMD[command: !export-sovereignty]
        PARAMS[parameters from signature]
    end
    
    DEC --> METHOD
    METHOD --> creates
    
    style DEC fill:#7d6608,stroke:#f4d03f,stroke-width:2px
```

**Metadata from code.** No separate config files.

---

## Slide 5: Auto-Discovery Flow

```mermaid
graph LR
    STARTUP[Agent Starts] --> SCAN[Scan features/]
    SCAN --> LOAD[Load Feature classes]
    LOAD --> FIND[Find @tool methods]
    FIND --> REG[Register in ToolRegistry]
    REG --> READY[Ready for commands]
    
    style SCAN fill:#1a5276,stroke:#85c1e9
    style REG fill:#145a32,stroke:#58d68d
```

**Zero boilerplate.** Add feature → auto-available.

---

## Slide 6: Tool Registry - The Router

```mermaid
graph TD
    subgraph registry["📋 ToolRegistry"]
        TOOLS[All registered tools]
        CMDS[Command → Tool map]
    end
    
    subgraph methods["Methods"]
        ROUTE[route_command]
        EXEC[execute_tool]
        HELP[get_help_text]
        LIST[list_tools]
    end
    
    registry --> methods
    
    style registry fill:#1a5276,stroke:#85c1e9,stroke-width:2px
```

**Central routing.** Commands find their features.

---

## Slide 7: Command Flow - User to Feature

```mermaid
graph LR
    USER["!export-sovereignty ipfs"] --> CH[CommandHandler]
    CH --> TR[ToolRegistry]
    TR --> MATCH[Match: !export-sovereignty]
    MATCH --> FEAT[SovereigntyFeature]
    FEAT --> METHOD[export_sovereignty]
    METHOD --> RESULT["✅ CID: Qm..."]
    
    style CH fill:#7d3c00,stroke:#f5b041
    style TR fill:#1a5276,stroke:#85c1e9
    style FEAT fill:#145a32,stroke:#58d68d
```

---

## Slide 8: Current Features Overview

```mermaid
graph TB
    subgraph core["Core Features"]
        SOV["📦 SovereigntyFeature<br/>Export/Import backups"]
        PRIV["🔒 PrivacyAgent<br/>Mode management"]
        MODEL["🧠 ModelAgent<br/>LLM switching"]
        SEC["🛡️ SecurityFeature<br/>Permissions & approval"]
    end

    subgraph extended["Extended Features"]
        WALLET["💰 WalletAgent<br/>Economics"]
        MCP["🔌 MCPAgent<br/>Tool protocols"]
        COMPUTE["🖥️ ComputeFeature<br/>Script execution"]
        FEEDBACK["📝 FeedbackFeature<br/>Self-observation"]
    end

    subgraph tools["Tool Features"]
        GITHUB["🐙 GitHubFeature<br/>Code introspection"]
        SEARCH["🔍 WebSearchFeature<br/>Tavily search"]
        VISUAL["🎨 VisualIdentity<br/>Avatar generation"]
        CLOUD["⚡ Cloud providers<br/>entry-point packages"]
    end

    style core fill:#1a5276,stroke:#85c1e9
    style extended fill:#512e5f,stroke:#af7ac5
    style tools fill:#0e4d45,stroke:#48c9b0
```

**12 Features.** Modular, discoverable, extensible.

---

## Slide 9: Adding a New Feature

```mermaid
graph TD
    subgraph steps["Steps to Add Feature"]
        S1["1. Create features/myfeature/"]
        S2["2. Create feature.py with class"]
        S3["3. Inherit from Feature base"]
        S4["4. Add @tool decorated methods"]
        S5["5. Done - auto-discovered!"]
    end
    
    S1 --> S2 --> S3 --> S4 --> S5
    
    subgraph result["Result"]
        R1[Commands available]
        R2[Tools for LLM]
        R3[Help text generated]
    end
    
    S5 --> result
    
    style S5 fill:#145a32,stroke:#58d68d
```

**Extensible by design.** Third-party features welcome.

---

## Slide 10: Feature Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Discovered: Agent startup
    Discovered --> Initialized: initialize() called
    Initialized --> Active: Ready for commands
    Active --> Active: Handle requests
    Active --> Shutdown: Agent stopping
    Shutdown --> [*]: shutdown() cleanup
```

**Clean lifecycle.** Features manage their own resources.

---

*Next: [05-storage-sovereignty.md](05-storage-sovereignty.md) - Storage V2 and sovereignty backup*
