# PRD: PrivacyAgent

## 1. Vision

The `PrivacyAgent` will serve as the Kestrel agent's dedicated guardian of information and confidentiality. It will encapsulate all logic related to the handling of sensitive conversation data, ensuring that the Sovereign's privacy settings are strictly and verifiably enforced at all times. This removes ambiguity from the main agent's logic and centralizes all privacy-critical operations into a single, auditable component.

## 2. Functional Requirements

The `PrivacyAgent` must be able to:
- **FR1:** Maintain the agent's current `PrivacyMode` state (`NORMAL`, `EPHEMERAL`, `ISOLATED`, `ANONYMOUS`, `LOCAL_ONLY`).
- **FR2:** Provide public methods to `set_mode` and `get_status`.
- **FR3:** Internally manage a temporary, in-memory `isolated_session` list for use during `ISOLATED` mode.
- **FR4:** Provide `save_isolated_session` and `discard_isolated_session` methods that transfer data from the temporary session to permanent storage (or delete it) when instructed.
- **FR5:** Implement the text anonymization logic for `ANONYMOUS` mode.
- **FR6:** Provide a single, authoritative `add_conversation` method. This method will contain the core logic that inspects the current `PrivacyMode` and determines whether to store, discard, anonymize, or temporarily hold the conversation entry.

## 3. Integration Plan

1.  **Instantiation:** The `KestrelAgent` will instantiate a single `PrivacyAgent` during its `__init__` process, passing it a reference to the main `storage` object.
    ```python
    # In KestrelAgent.__init__
    self.privacy_agent = PrivacyAgent(self.storage)
    ```
2.  **State Removal:** The `KestrelAgent` will be stripped of the `self.privacy_mode` and `self.isolated_session` attributes. All state will be managed within the `PrivacyAgent`.
3.  **Delegation of Commands:** The `_handle_command` method in `KestrelAgent` will delegate all privacy-related commands (`!privacy`, `!privacy-save`, `!privacy-discard`) directly to the corresponding methods on `self.privacy_agent`.
4.  **Delegation of Storage:** The `KestrelAgent`'s responsibility for saving conversation history will be entirely delegated. The call to `self.storage.add_conversation` will be replaced with a call to `self.privacy_agent.add_conversation`.

## 4. Testing Strategy

*   **Integration Test Plan:** A new test file, `tests/test_privacy_agent.py`, will be created. The test function will instantiate a `KestrelAgent` (which creates the real `PrivacyAgent`). The test will then systematically:
    1.  Switch to `ISOLATED` mode.
    2.  Send several messages and verify they are in the isolated session, not permanent storage.
    3.  Call the command to save the session and verify the messages are now in permanent storage.
    4.  Switch to `EPHEMERAL` mode.
    5.  Send a message and verify it is not stored anywhere.
    6.  Switch to `ANONYMOUS` mode.
    7.  Send a message containing PII and verify that an anonymized version is what gets saved to permanent storage.

*   **Visible Artifact:** The test output will be a human-readable log of the agent's state transitions and actions, demonstrating that the privacy rules are being correctly enforced at each step.
    ```
    --- PrivacyAgent Demonstration ---
    - Initial Mode: NORMAL
    - Setting mode to ISOLATED...
    - STATUS: Current privacy mode: isolated (session has 0 messages)
    - Sending message: 'My secret password is...'
    - STATUS: Current privacy mode: isolated (session has 1 messages)
    - Permanent storage empty.
    - Saving isolated session...
    - Permanent storage now contains 1 message.
    - Setting mode to ANONYMOUS...
    - Sending message: 'My name is John Doe.'
    - Verifying permanent storage contains: 'My name is [REDACTED].'
    --- End of Demonstration ---
    ```

*   **Mocking Strategy:** No mocks are required for the primary integration test. The `PrivacyAgent`'s dependency is the `Storage` object, which can be configured to use an in-memory SQLite database for fast, isolated testing. 