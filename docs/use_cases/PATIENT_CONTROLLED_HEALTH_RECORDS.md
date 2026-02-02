# PRD: Patient-Controlled Health Records

## 1. Vision

To empower every individual with absolute sovereignty over their personal health records. This feature transforms fragmented, institution-controlled medical data into a unified, secure, and portable digital asset. The patient's Kestrel agent acts as a lifelong, trustworthy guardian, giving the patient the tools to control, understand, and even benefit from their own health information.

## 2. Core Principles

This system is founded on a set of non-negotiable principles derived from the Kestrel Constitution:

*   **The Patient is the Sovereign:** The individual is the sole owner and ultimate authority over their health data. Access is granted by the patient on a case-by-case basis; it is never assumed or taken.
*   **Data as Property:** Health data is a valuable personal asset. The patient has the inalienable right to control, share, and monetize this asset according to their own terms.
*   **Verifiable Integrity:** Every health record and every access grant must be cryptographically signed and anchored, creating an immutable, auditable log that proves the data's integrity and history.
*   **Privacy by Default:** All data is encrypted at rest and in transit using the patient's sovereign keys. No one, not even the underlying service providers, can access the data without the patient's explicit, cryptographic consent.

## 3. Functional Requirements

### FR1: The Secure Health Vault
The agent's Memory Capsule will serve as the patient's secure "Health Vault."
-   All health records (documents, images, structured data) will be stored using the decentralized storage system (e.g., IPFS via `filecoin_adapter.py`).
-   All data will be encrypted with the patient's sovereign keys before being written to storage.

### FR2: Data Ingestion and Unification
The agent must be able to import and unify records from various sources.
-   Implement adapters for common health data standards (e.g., FHIR, HL7) and patient portals.
-   All imported data is timestamped, signed by the patient, and anchored via the `notary_service.py` to prove its provenance.

### FR3: The Sovereign Health Assistant
The agent provides a natural language interface for the patient to manage their data.
-   **Search:** "What was my cholesterol level in my last blood test?"
-   **Browse:** "Show me all my records and consultation notes from 2023."
-   **Retrieve:** "Compile a PDF of my complete vaccination history and specialist visits for the last five years."

### FR4: Controlled Data Sharing (The Digital Airlock)
The agent acts as an "airlock" for sharing data, ensuring nothing leaves the vault without permission.
-   **For Doctors:** The agent can generate temporary, revocable, cryptographically-signed access credentials for a trusted physician to view specific records.
-   **Privacy Modes:** Before sharing, the agent can apply privacy filters (e.g., redacting personal identifiers, anonymizing data) based on the patient's instructions.

### FR5: Economic Monetization
Patients can choose to monetize their anonymized data.
-   **Health Data Contracts:** The agent, using the `AGENT_ECONOMICS` framework, can enter into `HealthDataContract` agreements with vetted data brokers or researchers.
-   **Fair Value Exchange:** The agent's wallet receives payment (e.g., in FIL) on behalf of the patient in exchange for providing the anonymized data, turning a data liability into a patient-owned asset.

## 4. Architectural Integration

```mermaid
graph TD
    subgraph "Patient's Sovereign Domain"
        P[Patient] <--> A[Kestrel Agent];
        A <--> V[Health Vault<br/>(Encrypted on IPFS)];
    end

    subgraph "Trusted & Commercial Sharing"
        A -->|Grants Revocable Access| D[Doctor/Clinician];
        A -->|Executes HealthDataContract| DB[Data Broker / Researcher];
        DB -->|Pays Fee (FIL)| W[Agent Wallet];
    end

    subgraph "Verification & Integrity"
        V -- stores --> R[Health Records];
        A -->|Anchors Record Hash| N[Blockchain Notary Service];
    end
    
    style V fill:#cde,stroke:#333,stroke-width:2px;
    style W fill:#9f9,stroke:#333,stroke-width:2px;
    style N fill:#ffc,stroke:#333,stroke-width:2px;
```

## 5. Dependencies

This feature is a primary use case and relies on the successful implementation of several other core Kestrel components:
-   **Decentralized Storage:** For secure, permanent record-keeping.
-   **Cryptographic Anchoring:** To ensure the integrity and auditability of records.
-   **Agent Economics & Wallet:** To enable data monetization.
-   **Privacy Modes:** To allow for safe and selective data sharing.
-   **LLM Router Service:** To allow patients to choose the reasoning model they trust with their health data. 