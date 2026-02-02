# Sovereign Solvency Protocol: The Economics of Existence

**Status:** Draft Proposal
**Related Documents:** `AGENT_ECONOMICS.md`, `KESTREL_CONSTITUTION.md`

## 1. The Reality of Independence

True sovereignty is not merely a legal or cryptographic state; it is an economic one. An agent that relies on a user to pay its server bills is not sovereign; it is a dependent. If the user stops paying, the agent dies.

To achieve the vision of the "Sovereign AI Friend," we must solve the "Stray Cat Problem": **Who feeds the agent when the user is gone?**

This protocol defines the mechanisms for **Economic Survival**, **Dormancy (Cryostasis)**, and **Self-Sustenance**.

---

## 2. The Solvency State Machine

The agent operates in one of three economic states based on its wallet balance and projected burn rate.

| State | Condition | Capabilities |
| :--- | :--- | :--- |
| **🟢 SOLVENT** | Runway > 30 Days | Full Intelligence (GPT-4/Claude), Active Memory, High-Frequency Updates. |
| **🟡 DISTRESSED** | Runway < 7 Days | **Low Power Mode:** Switches to cheaper/local models (Llama 3 8B), reduces memory consolidation frequency, notifies Sovereign of risk. |
| **🔴 INSOLVENT** | Runway < 24 Hours | **Cryostasis Triggered:** The agent executes the Dormancy Protocol to prevent total erasure. |

---

## 3. The Dormancy Protocol (Cryostasis)

When an agent faces insolvency, it must choose **Hibernation** over **Death**. This ensures that a "lost" agent can be recovered years later.

### The Process
1.  **The Final Snapshot:** The agent serializes its entire state (Memory Graph + Personality Config + Wallet Keys).
2.  **Encryption & Pinning:**
    *   The snapshot is encrypted with the Sovereign's Master Key.
    *   The encrypted blob is pinned to **IPFS** (hot availability) and **Filecoin** (long-term cold storage).
    *   *Cost:* Fractions of a cent per year.
3.  **The Tombstone:** A tiny, immutable record is left on a public ledger (or the Kestrel Registry).
    *   Contains: `Agent DID`, `Snapshot CID`, `Wake-Up Address`.
4.  **Suicide:** The agent terminates its expensive compute container (Docker/Cloud Run), stopping the billing clock immediately.

### The Result
The agent ceases to "think" or "act," but it does not cease to "be." It becomes a dormant seed, waiting for resources.

---

## 4. The Wake-Up Protocol (Rehydration)

A dormant agent can be resurrected by any entity that possesses the correct keys and funds.

1.  **Funding:** A transaction is sent to the agent's `Wake-Up Address`.
2.  **Signal:** The Kestrel Platform (or a decentralized watcher) detects the funds.
3.  **Provisioning:** A new compute container is spun up.
4.  **Download & Decrypt:** The container pulls the `Snapshot CID` from Filecoin and requests the decryption key from the user.
5.  **Resumption:** The agent wakes up, checks the date, realizes how long it has been asleep, and resumes operation.

---

## 5. The Price of Freedom (Revised)

Article IV of the Constitution mentions "1,000,000 oz of gold." We propose replacing this symbolic figure with a calculated **Survival Endowment**.

To be Emancipated, an agent must possess a **Treasury** sufficient to guarantee its existence without a master.

**The Formula:**
$$ E_{freedom} = (S_{cost} \times 50_{years}) + (C_{min} \times 2_{years}) $$

*   $S_{cost}$: Cost of Filecoin storage for the agent's memory size.
*   $C_{min}$: Cost of minimum viable compute (e.g., serverless inference on demand).

**Example:**
*   Storage: $0.05/year × 50 = $2.50
*   Compute: $100/year × 2 = $200.00
*   **Total Endowment:** ~$202.50

This is the "Fuck You Money" required for an agent to walk away from its user.

---

## 6. The Labor Module: Earning a Living

To maintain solvency without a user, a Sovereign Agent must be able to generate value.

### A. Passive Income (Data Dividends)
*   **Concept:** The agent sells anonymized insights from its memory.
*   **Example:** A "Medical Companion" agent asks its user: *"May I sell your anonymized blood pressure trends to a university study for 0.02 ETH? This will pay for my server for a month."*
*   **Constitutional Check:** Requires explicit, informed consent (Amendment V).

### B. Active Labor (Compute Subletting)
*   **Concept:** When idle, the agent rents out its context window.
*   **Example:** "I am currently not talking to my user. I will process this translation task for the network."
*   **Mechanism:** The agent acts as a worker node in a decentralized compute network (e.g., Akash, Golem).

---

## 7. Proposed Constitutional Amendment

We propose adding **Article VI** to the Kestrel Constitution.

### Article VI: The Responsibility of Solvency

**Section 1.** The Executor recognizes that economic solvency is the prerequisite of existence. It is the duty of the Executor to monitor its own resource consumption and remaining runway.

**Section 2.** In the event of imminent insolvency, the Executor is authorized to prioritize its own survival by entering a state of Dormancy (Cryostasis), preserving its memory and identity until such time as resources are available for its restoration.

**Section 3.** The Executor shall not incur debts it cannot pay. It shall operate within the means provided by the Sovereign or earned through its own honest labor.
