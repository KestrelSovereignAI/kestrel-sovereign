---
name: privacy-implementation
description: Use when implementing Kestrel privacy modes, storage policies, UI indicators, or testing EPHEMERAL/ISOLATED/ANONYMOUS/NORMAL/PUBLIC privacy levels. Automatically delegates to privacy-architect subagent.
---

# Privacy Implementation Skill

This skill automatically activates when detecting work on Kestrel's 5-level privacy system.

## Trigger Keywords
- "privacy mode", "privacy modes", "privacy settings"
- "EPHEMERAL", "ISOLATED", "ANONYMOUS", "NORMAL", "PUBLIC"
- "privacy indicator", "privacy UI", "privacy selector"
- "PII filtering", "local only", "off the record"
- "privacy policy", "data storage", "storage policy"
- "!set-privacy-mode", "!get-privacy-mode"

## What This Skill Does

When activated, this skill:
1. Detects you're working on privacy features
2. Suggests using the **privacy-architect** subagent
3. Provides 5-level privacy mode reference
4. Links to privacy testing commands

## The 5 Privacy Levels

### 1. 🔒 EPHEMERAL (Level 0)
- **Storage**: NOTHING stored, ever
- **LLM**: Local only (Ollama)
- **Use Case**: Truly off-the-record conversations

### 2. 🔐 ISOLATED (Level 1)
- **Storage**: Temporary, deleted on session end
- **LLM**: Local only
- **Use Case**: Experimenting, can promote to ANONYMOUS

### 3. 🎭 ANONYMOUS (Level 2)
- **Storage**: Full, with PII filtering
- **LLM**: Cloud allowed
- **Use Case**: Privacy-conscious with convenience

### 4. 📝 NORMAL (Level 3)
- **Storage**: Full persistence
- **LLM**: Cloud allowed
- **Use Case**: Standard user experience

### 5. 🌐 PUBLIC (Level 4)
- **Storage**: Full with sharing enabled
- **LLM**: Cloud allowed
- **Use Case**: Content meant to be shared

## Implementation Checklist

### Agent Commands
- [ ] `!set-privacy-mode <MODE>` - Change mode
- [ ] `!get-privacy-mode` - Show current mode
- [ ] `!privacy-status` - Detailed report

### Storage Layer
- [ ] PrivacyAwareStorage class
- [ ] EPHEMERAL session handler (in-memory only)
- [ ] PII filtering for ANONYMOUS mode
- [ ] Mode stored in agent properties

### UI Components
- [ ] Privacy indicator (color-coded by mode)
- [ ] Privacy mode selector dropdown
- [ ] Privacy status panel
- [ ] Mode transition warnings

### Testing
- [ ] Test each mode stores/doesn't store correctly
- [ ] Test mode transitions
- [ ] Test PII filtering
- [ ] Test LLM provider restrictions
- [ ] Test UI updates

## Critical Implementation Details

### EPHEMERAL Must Store NOTHING
```python
if privacy_mode == "EPHEMERAL":
    # Use in-memory session only
    self.ephemeral_session.add_message(role, content)
    return None  # DO NOT store in database
```

### PII Filtering (ANONYMOUS)
```python
def filter_pii(text: str) -> str:
    # Use NER or regex to detect:
    # - Names, addresses, phone numbers
    # - Email addresses, SSN, national IDs
    # Replace with [REDACTED] or [NAME]
    pass
```

### LLM Provider Restriction
```python
if privacy_mode == "EPHEMERAL":
    response = await llm_service.get_response(
        ...,
        force_local_only=True  # CRITICAL: Only Ollama
    )
```

## Testing Command

Run comprehensive privacy tests:
```
/privacy-test
```

## Recommendation

Use the privacy-architect subagent for systematic implementation:
```
"Use the privacy-architect subagent to implement the privacy system"
```

Or use the parallel workflow:
```
/parallel-work
```

## Key Principles

1. **User Sovereignty** - Users control what is stored
2. **Privacy by Default** - Start with strongest privacy
3. **Explicit Consent** - Mode changes require user action
4. **Transparent Storage** - Clear indication of storage
5. **True Ephemeral** - EPHEMERAL stores NOTHING
