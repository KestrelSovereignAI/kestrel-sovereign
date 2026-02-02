# PRD: Cryptographic Log Anchoring

**1. Vision & Background**

The Kestrel project aims to create **sovereign agents**: persistent, autonomous entities whose identity and memory are owned and controlled by the user. A critical component of this sovereignty is ensuring the integrity and verifiability of the agent's history without compromising its privacy.

This feature, **Cryptographic Log Anchoring**, introduces a "proof-of-history" for the agent. It allows the user to create an immutable, tamper-proof record of their agent's interactions. This serves as the foundational layer for future capabilities like the "dead man's switch" survival protocol and enables a trust model for both private "AI friends" and transparent public agents.

**2. Goals & Objectives**

-   **Verifiability:** To create a system where the integrity of an agent's conversation history can be mathematically verified at any point in time.
-   **Privacy:** To ensure that the contents of private agent conversations are never exposed on a public ledger.
-   **Efficiency:** To achieve verifiability with minimal performance overhead and low transactional cost.
-   **Extensibility:** To build a system that can support both fully private agents and fully public agents by making privacy a configurable policy.

**3. User Stories**

-   **As the owner of a private AI friend,** I want to periodically save a cryptographic "fingerprint" of our conversation history to an immutable ledger so that I can later prove the log has not been altered, without revealing the conversations themselves.
-   **As the operator of a public-facing agent (e.g., a community bot),** I want to make its interactions fully transparent and auditable by publishing its memory hashes, and potentially the full content, to a public location.
-   **As an autonomous agent,** I want to be able to verify the integrity of my own memory (`DNACapsule`) when I "wake up" or am restored from a backup, ensuring I am starting from a known, untampered state.
-   **As an elderly user,** I want anchored human stories preserved verifiably for family, preventing loss.

**4. Functional Requirements**

-   **FR1: Conversation Hashing:**
    -   The system must be able to generate a single, deterministic cryptographic hash (e.g., a Merkle Root of SHA-256 hashes) representing the entire state of the `conversations` table at a given moment.
    -   The hashing function must be consistent: the same history will always produce the same hash.
    -   Changing any part of the history (a single character, timestamp, or reordering) must result in a different final hash.

-   **FR2: Anchor Storage (Internal):**
    -   A new table, `log_anchors`, will be added to the `enhanced_storage.db` schema.
    -   This table will store `(timestamp, anchor_hash, ledger_ref)`, where `ledger_ref` is a placeholder for the future transaction ID of the on-chain record.

-   **FR3: Notary Service (Mocked):**
    -   A mock "notary" service will be created to simulate interaction with a blockchain.
    -   This service will have a `publish_anchor(hash)` method that, for now, will simply log the action and return a fake transaction ID. This allows us to build the full agent-side logic without immediate blockchain dependencies.

-   **FR4: Agent Integration:**
    -   The `KestrelAgent` will have a new method: `anchor_memory_state()`.
    -   When called, this method will:
        1.  Invoke the hashing logic in `EnhancedStorage` to get the current conversation log hash.
        2.  Call the `blockchain_notary.publish_anchor()` method with the hash.
        3.  Store the hash and the returned `ledger_ref` in the new `log_anchors` table.

**5. Non-Functional Requirements**

-   **NFR1: Privacy:** The raw content of conversations must not be part of the hash-generation logic in a way that leaks information. The final anchored hash should reveal nothing about the content.
-   **NFR2: Performance:** The hashing of the conversation log should be efficient and not noticeably delay the agent's interactive response time.

- **FR5: Real-World Anchoring (Notary Service Integration):**
  -   The system will have the capability to bridge the digital and physical worlds by creating legally tangible anchors.
  -   This is achieved through a synergy between the `PhysicalMailContract` and a physical notary service.
  -   **Process:**
      1.  The agent generates a critical document for anchoring (e.g., a decision log, a contract, or a human-told story for an elderly companion application).
      2.  It uses a `PhysicalMailContract` to print and mail this document to a secure, designated P.O. Box (e.g., mail human-told stories for notary, creating tangible legacies).
      3.  A trusted human operator retrieves the document, has it physically notarized, and then scans the notarized copy.
      4.  The scanned, legally-stamped document is then ingested back into the agent's memory.
  -   This creates a powerful, real-world artifact that complements the purely digital `log_anchors`, providing a form of verification that is legible and significant in legal contexts where a blockchain hash may not be.

**6. Out of Scope (For This Iteration)**

-   **Live Blockchain Integration:** We will not integrate with a real blockchain (e.g., Ethereum via `web3.py`) in this phase. The mock notary is sufficient for development and testing.
-   **UI/Configuration for Privacy Policy:** The agent will be hardcoded for the "private" use case for now. The logic for a public agent is a future extension.
-   **The Dead Man's Switch:** This anchoring feature is a *prerequisite* for the switch, but we are not building the switch's logic (monitoring, clone activation) yet.

**7. Success Metrics**

-   A new, comprehensive test suite (`test_anchoring.py`) is created and passes.
-   The tests must verify that calling `agent.anchor_memory_state()` successfully creates a new entry in the `log_anchors` table.
-   The tests must demonstrate that after anchoring, adding a new conversation and anchoring again produces a *different* hash.
-   The tests must prove that altering a past conversation entry in the database and re-calculating the hash results in a hash that *does not match* the original anchor, proving tamper-detection. 