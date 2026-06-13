# Privacy Modes: Complete Data Sovereignty

## 1. Vision

Kestrel provides unprecedented control over data privacy through **independent privacy flags** with **named presets**. Users can use presets for common configurations or set flags directly for custom setups.

**Core Principle**: The user, not the platform, controls data retention and processing location.

## 2. Privacy Architecture: Flags + Presets

Privacy is controlled by generic independent flags. Named modes are presets
over these flags; they are not separate enforcement systems.

| Flag | Options | Controls |
|------|---------|----------|
| `storage` | none, temp, pii_redacted, deidentified, full | How/whether data is persisted |
| `processing` | local, trusted, cloud | Where inference/processing may happen |
| `sharing` | private, research, public | Whether content can be shared/exported |
| `assurance` | none, pii_redacted, safe_harbor, expert_determination | Privacy assurance backing the preset |
| `audit` | optional, required | Whether evidence/audit artifacts are required |
| `computer_access` | true, false | Whether tools may touch the host computer; always explicit and orthogonal |

**Named presets** are convenient combinations of these flags:

```python
PRIVACY_PRESETS = {
    "ephemeral": PrivacyConfig(storage="none", processing="local", sharing="private"),
    "isolated": PrivacyConfig(storage="temp", processing="local", sharing="private"),
    "anonymous": PrivacyConfig(storage="pii_redacted", processing="local", sharing="private", assurance="pii_redacted"),
    "normal": PrivacyConfig(storage="full", processing="cloud", sharing="private"),
    "public": PrivacyConfig(storage="full", processing="cloud", sharing="public"),
    "deidentified": PrivacyConfig(storage="deidentified", processing="trusted", sharing="research", assurance="safe_harbor", audit="required"),
}
```

```mermaid
graph TD
    A[User Input] --> B{Privacy Config}
    
    B -->|storage=full| C[Persistent Storage]
    B -->|storage=none| D[No Storage]
    B -->|storage=temp| E[Temporary Session]
    B -->|storage=pii_redacted| F[PII-Redacted Storage]
    B -->|storage=deidentified| J[Deidentified Research Storage]
    
    B --> G{processing}
    G -->|local| H[Ollama Only]
    G -->|trusted| K[Trusted/BAA-Capable Route]
    G -->|cloud| I[Any Provider]
    
    style D fill:#ff9999,stroke:#333,stroke-width:2px
    style H fill:#99ccff,stroke:#333,stroke-width:2px
    style C fill:#99ff99,stroke:#333,stroke-width:2px
```

## 3. Preset Specifications

### Normal Mode (`!privacy normal`)

**Config**: `storage=full, processing=cloud, sharing=private, assurance=none, audit=optional`

**Use Case**: Standard operation for most interactions
**Storage**: Persistent SQLite database
**Processing**: Multi-model (Ollama + OpenAI fallback)
**Memory Anchoring**: Full cryptographic anchoring available
**Learning**: Agent learns and remembers everything

**Example Scenarios**:
- Regular conversation and collaboration
- Building long-term knowledge base
- Professional work that benefits from context
- For human-led elderly storytelling preservation.

### Ephemeral Mode (`!privacy ephemeral`)

**Config**: `storage=none, processing=local, sharing=private, assurance=none, audit=optional`

**Use Case**: Sensitive topics requiring zero digital footprint
**Storage**: Nothing stored anywhere
**Processing**: Local LLM only (Ollama) - data never leaves device
**Memory Anchoring**: Disabled (nothing to anchor)
**Learning**: Agent cannot learn or remember anything from this session

**Example Scenarios**:
- Personal matters and confidential discussions
- Sensitive business strategy sessions
- Mental health conversations
- Brainstorming with proprietary information
- For private intimate sharing without eternal records.

**Technical Implementation**:
```python
def add_conversation(self, role: str, content: str, metadata: Optional[Dict] = None):
    if self._privacy_config.is_ephemeral():
        # Don't store anything - use in-memory ephemeral session
        self.ephemeral_session.add_message(role, content, metadata)
        return  # Do NOT persist
```

### Isolated Mode (`!privacy isolated`)

**Config**: `storage=temp, processing=local, sharing=private, assurance=none, audit=optional`

**Use Case**: Complex analysis where you want to control what becomes permanent
**Storage**: Temporary session buffer only
**Processing**: Local LLM only (data stays on device during experimentation)
**Memory Anchoring**: Not available during session
**Learning**: Deferred until user decides

**Session Control Commands**:
- `!save-session` - Make session permanent
- `!discard-session` - Delete session without saving

**Example Scenarios**:
- Experimental analysis of sensitive data
- Trying different approaches before committing
- Working with third-party data that may not belong in permanent memory

### Anonymous Mode (`!privacy anonymous`)

**Config**: `storage=pii_redacted, processing=local, sharing=private, assurance=pii_redacted, audit=optional`

**Use Case**: Learning from interactions while protecting identity
**Storage**: PII-redacted persistent storage
**Processing**: Local LLM only
**Memory Anchoring**: Available (preserves integrity)
**Learning**: Agent learns patterns but not personal details

**PII Redaction**:
- Email addresses → `[EMAIL_REDACTED]`
- Phone numbers → `[PHONE_REDACTED]`
- SSNs → `[SSN_REDACTED]`
- Names → `[NAME_REDACTED]`

Anonymous is not HIPAA de-identification. It is PII redaction plus local
processing. Use Deidentified mode for research workflows that require
Safe Harbor or Expert Determination assurance.

**Example Scenarios**:
- Contributing to agent training without personal exposure
- Discussing sensitive topics while preserving learning value
- Interactions that have educational value but contain personal information

### Public Mode (`!privacy public`)

**Config**: `storage=full, processing=cloud, sharing=public, assurance=none, audit=optional`

**Use Case**: Fully transparent and auditable agents
**Storage**: Full persistent storage
**Processing**: Full multi-model access
**Memory Anchoring**: Full anchoring available
**Learning**: Full learning with export capability

**Example Scenarios**:
- Community bots that need to be auditable
- Transparent AI assistants
- Agents where trust requires visibility
- Public-facing services

### Deidentified Mode (`!privacy deidentified`)

**Config**: `storage=deidentified, processing=trusted, sharing=research, assurance=safe_harbor, audit=required`

**Use Case**: Clinical/research data that may be saved or exported only after
de-identification.
**Storage**: Deidentified persistent storage with evidence artifact required
**Processing**: Trusted route only; generic cloud routing is not sufficient
**Memory Anchoring**: Available for deidentified artifacts
**Learning**: Research use without direct identifiers

Deidentified mode is distinct from anonymous/PII-redacted mode. HIPAA Safe
Harbor requires removal of the Safe Harbor identifier set and no actual
knowledge that remaining information can identify the individual. Expert
Determination is a separate path requiring a qualified expert's documented
determination; Kestrel must not claim it without that evidence artifact.

## 4. Custom Configurations

Beyond presets, you can set flags directly for custom configurations:

```python
from privacy import PrivacyConfig

# Want permanent storage but local-only processing?
# (Not a preset, but perfectly valid)
agent.set_privacy(PrivacyConfig(
    storage="full",
    processing="local",
    sharing="private",
))
```

## 5. Privacy Mode Commands

### Chat Interface Commands

```bash
# Status and control
!status         # Show current privacy mode and session status
!privacy normal
!privacy ephemeral
!privacy isolated
!privacy anonymous
!privacy public
!privacy deidentified

# Session management (isolated mode only)
!privacy-save     # Save isolated session to permanent storage
!privacy-discard  # Discard isolated session without saving
```

### Programmatic API

```python
from privacy import PrivacyMode, PrivacyConfig, get_privacy_preset

# Using presets (backward compatible)
agent.privacy_agent.set_mode(PrivacyMode.EPHEMERAL)
agent.privacy_agent.set_mode("ephemeral")  # String also works

# Using custom config
agent.privacy_agent.set_mode(PrivacyConfig(
    storage="full",
    processing="local",
    sharing="private",
))

# Check current config
config = agent.privacy_agent.privacy_config
print(f"Storage: {config.storage}, processing: {config.processing}")

# Check if cloud is allowed
if config.allows_cloud_llm():
    # Use OpenAI, Anthropic, etc.
    pass

# Isolated mode session management
agent.privacy_agent.save_isolated_session()
agent.privacy_agent.discard_isolated_session()
```

## 6. Privacy Mode Transitions

### Safe Transitions

All privacy mode transitions are safe and immediate:

```mermaid
graph LR
    A[Any Mode] -->|Immediate| B[Normal]
    A -->|Immediate| C[Ephemeral]
    A -->|Immediate| D[Isolated]
    A -->|Immediate| E[Anonymous]
    A -->|Immediate| F[Public]
    
    D -->|save-session| B
    D -->|discard-session| G[Session Deleted]
```
Note: Transitions support app layers—e.g., switch to Isolated for reviewing human narratives before saving.

### Data Protection Guarantees

- **No Data Loss**: Switching modes never deletes existing data
- **Immediate Effect**: New privacy rules apply to the very next message
- **Backward Compatibility**: Existing stored data remains accessible in all modes
- **Isolation Integrity**: Isolated sessions cannot accidentally leak into permanent storage

## 7. Implementation Details

### Storage Layer Integration

Privacy config integrates directly with the enhanced storage system:

```python
def store_file_sovereign(self, file_path: str, privacy_config: PrivacyConfig):
    """Store file based on privacy config"""
    if privacy_config.is_ephemeral():
        # Calculate hash but don't store
        return self._calculate_hash_only(file_path)
    elif privacy_config.requires_anonymization():
        # Anonymize then store
        return self._store_anonymized(file_path)
    else:
        # Normal storage
        return super().store_file(file_path)
```

### LLM Service Integration

Privacy config controls which models can be used:

```python
def get_response(self, prompt: str, privacy_config: PrivacyConfig) -> str:
    """Get LLM response respecting privacy constraints"""
    if not privacy_config.allows_cloud_llm():
        # Filter to only local providers
        local_providers = [p for p in self.providers if p["name"] in ["ollama"]]
        if not local_providers:
            raise RuntimeError("Privacy config requires local providers only")
```

## 8. Security Considerations

### Threat Model

Privacy modes protect against:
- ✅ Accidental data exposure to external services
- ✅ Unintended persistence of sensitive information
- ✅ Identity correlation through PII
- ✅ Compliance violations in regulated environments

### Limitations

Privacy modes do NOT protect against:
- ❌ Compromised local system
- ❌ Network-level surveillance (use additional encryption)
- ❌ Physical access to storage devices
- ❌ Social engineering attacks

### Best Practices

1. **Start Restrictive**: Begin with ephemeral or isolated for new sensitive topics
2. **Regular Audits**: Use `!status` to verify current privacy mode
3. **Session Hygiene**: In isolated mode, regularly decide on session fate
4. **Mode Documentation**: Document which privacy mode was used for important decisions

## 9. Future Enhancements

### Planned Features

- **More Flag Options**: Additional flags for encryption, retention periods, etc.
- **Time-Based Auto-Expiry**: Automatic session cleanup
- **Geographic Compliance**: EU/US data residency modes
- **Collaborative Privacy**: Multi-user privacy negotiations

### Integration Roadmap

- **Filecoin Storage Tiers**: Map privacy configs to decentralized storage tiers
- **Zero-Knowledge Proofs**: Cryptographic privacy guarantees
- **Hardware Security Modules**: TPM-based local processing verification

---

**Privacy flags + presets represent a fundamental shift from platform-controlled to user-controlled data sovereignty. They ensure that AI agents work for users, not surveillance capitalism.**
