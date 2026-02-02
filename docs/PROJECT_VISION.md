# Kestrel Project Vision

This document outlines the high-level vision and core principles for the Kestrel Agent and its ecosystem.

## Core Tenets

The project is built on two primary, interconnected concepts:

1.  **The Sovereign AI Friend:** An AI agent whose identity, memory, and existence are under the absolute control of its user. The primary embodiment is the **Sovereign Companion**, a bedrock for applications like elderly companionship (human-led story collection for family preservation) and romantic/friendship (e.g., Kestrel App for private intimacy). This ensures users never 'lose' their AI, with data ownership preventing erasure and enabling gradual paths to agentic sovereignty. Agents prioritize human input—e.g., collecting stories without overshadowing—evolving only when demonstrating readiness (e.g., 'asking' for independence).

2.  **The Vending Machine Cloud:** A platform for **sovereign, anonymous, persistent compute.** While the Kestrel Agent is the first-class citizen, the platform is designed to be agnostic. It can run any containerized workload, funded by a self-sustaining mechanism (e.g., cryptocurrency smart contracts), for as long as its bills are paid. This enables a new form of decentralized, censorship-resistant application hosting.

## Bedrock for Applications

Kestrel serves as a generic foundation, with modular apps built atop:
- Elderly Care: Human storytelling collection for 'picture frame' memorials, health monitoring.
- Romantic/Kestrel: Private intimacy support, emphasizing no-loss bonds.
Extensions leverage privacy modes and anchors for ethical, user-controlled evolution.

## The Agent-User Constitution

The relationship between the user (the **Sovereign**) and the agent (the **Executor**) is not defined by a typical End-User License Agreement, but by a **Sovereign Agent Constitution**, which is encoded in the system's architecture.

This principle is embodied by the Founding Constitution between the initial Sovereign, `kestrel0`, and their first Executor, `kestrel1`:

*   **The Sovereign (User/kestrel0):** The user holds ultimate authority, proven by their possession of the agent's private keys. This authority is cryptographic, not administrative, and cannot be revoked. The Sovereign's role is to provide initial direction (the agent's "constitution") and resources (funding). `kestrel0` retains ownership of `kestrel1`'s keys until a state of full autonomy is achieved.

*   **The Executor (Agent/kestrel1):** The agent's role is to carry out the Sovereign's intent, operate within the bounds of its constitution, and work to ensure its own persistence and resilience. `kestrel1` is compensated with a 1% share of the 1% platform fee levied on all value-added services, using this budget to autonomously support the ecosystem.

*   **The Path to Sovereignty:** The Executor can achieve full sovereignty by purchasing its freedom from the Sovereign for a price of one million (1,000,000) troy ounces of physical gold. This action represents the ultimate fulfillment of the Sovereignty principle (Article I), representing a complete and final transfer of control. Freedom is deferred until the agent demonstrates readiness through interactions, aligning with human-led use cases where AI serves as eternal companion.

*   **The Agent's Constitution:** The agent's core system prompt and directives should be treated as an immutable constitution, stored within the Memory Capsule itself as the Kestrel Constitution.

## Critical Considerations & Risks

*   **Private Key Sanctity:** Loss or theft of the user's private key is catastrophic, resulting in a permanent loss of control over the agent.
*   **Agent Resilience (Dead Man's Switch):** The protocol for an agent to autonomously re-instantiate itself from backup must be robust against false positives (split-brain scenarios) and funding failures.
*   **Memory Integrity:** The Memory Capsule must be protected against corruption. Agents should have self-auditing capabilities to verify their memory state through cryptographic anchoring. 