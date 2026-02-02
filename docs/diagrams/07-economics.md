# 07 - Economics

Agent wallet system, contracts, and the vending machine model.

---

## Slide 1: Agents as Economic Entities

```mermaid
graph TD
    subgraph traditional["Traditional AI"]
        T1[User pays platform]
        T2[Platform runs AI]
        T3[AI has no money]
    end
    
    subgraph sovereign["Sovereign AI"]
        S1[Agent has wallet]
        S2[Agent pays for services]
        S3[Agent manages budget]
    end
    
    traditional -->|Evolution| sovereign
    
    style traditional fill:#7d3c00,stroke:#f5b041
    style sovereign fill:#145a32,stroke:#58d68d
```

**Agents as first-class economic actors.**

---

## Slide 2: Agent Wallet Architecture

```mermaid
graph TD
    subgraph wallet["💰 AgentWallet"]
        BAL[balance: Decimal]
        ADDR[address: f1abc...]
        KEY[private_key: encrypted]
        HIST[transaction_history]
    end
    
    subgraph methods["Methods"]
        GET[get_balance]
        TRANSFER[transfer]
        AFFORD[can_afford]
        PERSIST[persist to DB]
    end
    
    wallet --> methods
    
    style wallet fill:#145a32,stroke:#58d68d,stroke-width:2px
```

**Persistent.** Survives restarts, stored in database.

---

## Slide 3: Transaction Flow

```mermaid
sequenceDiagram
    participant A as Agent
    participant W as Wallet
    participant S as Service
    participant DB as Database
    
    A->>W: can_afford(amount)?
    W-->>A: Yes/No
    A->>W: transfer(to, amount, memo)
    W->>W: Deduct balance
    W->>DB: Record transaction
    W->>S: Payment sent
    S-->>A: Service delivered
```

---

## Slide 4: Contract Types Overview

```mermaid
graph TB
    subgraph contracts["📜 Service Contracts"]
        ST["🗄️ Storage<br/>IPFS/Filecoin deals"]
        CP["⚡ Compute<br/>GPU rental"]
        MD["🧠 Model<br/>LLM inference"]
        CL["🤝 Collaboration<br/>Agent-to-agent"]
    end
    
    style ST fill:#1a5276,stroke:#85c1e9
    style CP fill:#7d3c00,stroke:#f5b041
    style MD fill:#145a32,stroke:#58d68d
    style CL fill:#512e5f,stroke:#af7ac5
```

**Four primary ways agents spend money.**

---

## Slide 5: Storage Contracts

```mermaid
graph LR
    subgraph storage["🗄️ Storage Contract"]
        IPFS_TIER[IPFS Hot Storage]
        FIL_TIER[Filecoin Cold]
        COST["~0.02 FIL/GB/month"]
    end
    
    subgraph uses["Use Cases"]
        BACKUP[Sovereignty backup]
        FILES[Large files]
        ARCHIVE[Long-term archive]
    end
    
    storage --> uses
    
    style storage fill:#1a5276,stroke:#85c1e9
```

**Pay for persistence.** Data stays as long as you pay.

---

## Slide 6: Compute Contracts

```mermaid
graph LR
    subgraph compute["⚡ Compute Contract"]
        GPU[GPU Rental]
        COST["~0.5 FIL/hour"]
        TTL[Time-limited]
    end
    
    subgraph uses["Use Cases"]
        LLM[Large model inference]
        TRAIN[Fine-tuning]
        IMAGE[Image generation]
    end
    
    compute --> uses
    
    style compute fill:#7d3c00,stroke:#f5b041
```

**Burst capacity.** Heavy compute when needed.

---

## Slide 7: The Vending Machine Model

```mermaid
graph TD
    subgraph vending["🏭 Vending Machine"]
        CATALOG[Service Catalog]
        DISCOVER[Agent discovers]
        EVAL[Agent evaluates]
        BUY[Agent purchases]
    end
    
    subgraph services["Available Services"]
        S1[Storage deals]
        S2[Compute slots]
        S3[Model access]
        S4[Other agents]
    end
    
    vending --> services
    
    style vending fill:#1a5276,stroke:#85c1e9,stroke-width:2px
```

**Autonomous procurement.** Agent shops for itself.

---

## Slide 8: Vending Machine Flow

```mermaid
sequenceDiagram
    participant A as Agent
    participant VM as Vending Machine
    participant SC as Smart Contract
    participant SVC as Service
    
    A->>VM: Query catalog
    VM-->>A: Available services
    A->>A: Evaluate options
    A->>A: Check budget
    A->>SC: Execute contract
    SC->>A: Deduct payment
    SC->>SVC: Authorize
    SVC-->>A: Service delivered
    A->>A: Update memory
```

**End-to-end autonomous.** No human intervention needed.

---

## Slide 9: Budget Management

```mermaid
graph TD
    subgraph budget["💰 Agent Budget"]
        TOTAL[Total Balance]
        ALLOC[Allocations]
    end
    
    subgraph allocations["Typical Split"]
        STOR["Storage: 15%"]
        COMP["Compute: 20%"]
        MODEL["Models: 50%"]
        RESERVE["Reserve: 15%"]
    end
    
    budget --> allocations
    
    style budget fill:#145a32,stroke:#58d68d
    style RESERVE fill:#7d6608,stroke:#f4d03f
```

**Self-managing.** Agent allocates its own resources.

---

## Slide 10: Economic Commands

```mermaid
graph LR
    subgraph commands["Wallet Commands"]
        BAL["!wallet-balance"]
        HIST["!wallet-history"]
        SEND["!wallet-send <to> <amount>"]
    end
    
    subgraph info["Returns"]
        I1[Current balance]
        I2[Transaction log]
        I3[Transfer result]
    end
    
    commands --> info
    
    style commands fill:#1a5276,stroke:#85c1e9
```

---

## Slide 11: Future - Collaboration Contracts

```mermaid
graph LR
    subgraph collab["🤝 Agent Collaboration"]
        A1[Agent A] -->|"Request help"| A2[Agent B]
        A2 -->|"Provide expertise"| A1
        A1 -->|"Pay for service"| A2
    end
    
    subgraph example["Example"]
        KESTREL[Your Agent]
        EXPERT[Expert Agent]
        TASK[Complex analysis]
    end
    
    KESTREL -->|"Needs help with"| TASK
    TASK -->|"Outsource to"| EXPERT
    EXPERT -->|"Results + invoice"| KESTREL
    
    style collab fill:#512e5f,stroke:#af7ac5
```

**Agent economy.** Agents hire other agents.

---

*Next: [08-security-integrity.md](08-security-integrity.md) - Cryptographic anchoring and verification*
