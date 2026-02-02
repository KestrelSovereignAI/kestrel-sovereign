# PRD: WalletAgent

## 1. Vision

The `WalletAgent` is a specialized Feature Agent responsible for managing the Kestrel agent's economic identity and all financial transactions. It serves as the agent's treasury, providing a secure and auditable interface for all economic activities, from paying for services to receiving compensation. This component is the first concrete step toward realizing the vision of a "sovereign economic entity" as described in `AGENT_ECONOMICS.md`.

## 2. Functional Requirements

The `WalletAgent` must be able to:
- **FR1:** Initialize with a starting balance of Filecoin (FIL), splitting it into a `main_balance` (90%) for general operations and an `audit_balance` (10%) reserved for integrity checks.
- **FR2:** Provide a `transfer` method to deduct funds from the `main_balance` for operational expenses.
- **FR3:** Provide a `deduct_audit_fee` method to deduct funds from the `audit_balance` specifically for constitutional audits.
- **FR4:** Provide `can_afford` and `can_afford_audit` methods to check for sufficient funds before attempting a transaction.
- **FR5:** Maintain an immutable `transaction_history` list, recording the details of every successful transaction.
- **FR6:** The current implementation is a mock that operates on in-memory `Decimal` objects. The architecture must support future integration with a real `FilecoinAdapter` for on-chain transactions.

## 3. Integration Plan

1.  **Instantiation:** The `KestrelAgent` instantiates a single `WalletAgent` during its `__init__` process. The initial balance is read from the agent's genesis node in the knowledge graph.
    ```python
    # In KestrelAgent.__init__
    initial_balance_str = agent_node.properties.get("initialBalance", "100.0")
    self.wallet = WalletAgent(initial_balance_fil=Decimal(initial_balance_str))
    ```
2.  **Delegation:** All calls that require payment (e.g., `anchor_memory_state`, `!send-mail`) within the `KestrelAgent` must delegate to the `self.wallet` instance to check for funds and execute the transfer.

## 4. Testing Strategy

*   **Integration Test Plan:** The test file `tests/test_wallet_agent.py` validates the `WalletAgent`'s functionality directly. The test instantiates the agent and performs a series of transactions to confirm its logic.
*   **Visible Artifact:** The test produces a human-readable log demonstrating the wallet's state at each step of the test, including initialization, successful transfers, failed transfers, and the final transaction history.
    ```
    --- WalletAgent Demonstration ---
    1. Wallet initialized with 200.0 FIL.
       -> Main Balance: 180.00 FIL (90%)
       -> Audit Balance: 20.00 FIL (10%)
    2. Attempting to transfer 50.5 FIL from main balance...
       -> Transfer successful. New Main Balance: 129.50 FIL
    ...
    --- End of Demonstration ---
    ```
*   **Mocking Strategy:** No mocks are required for the primary integration test as the `WalletAgent` is currently self-contained. Future tests involving the `FilecoinAdapter` will require mocking the blockchain interface after the real integration is built. 