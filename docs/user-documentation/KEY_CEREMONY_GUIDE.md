---
type: User Guide
title: Key Ceremony Guide for Kestrel Agents
description: The Key Ceremony is a formal process for generating and securing the
  encryption key(s) for a Kestrel agent. For permanent agents like Emma, this ceremony
  should be performed wit...
resource: /docs/user-documentation/KEY_CEREMONY_GUIDE.md
tags:
- docs
- user-documentation
- user-guide
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: documentation
canonical: false
generated: false
privacy: public
---

# Key Ceremony Guide for Kestrel Agents

## Overview

The Key Ceremony is a formal process for generating and securing the encryption key(s) for a Kestrel agent. For permanent agents like Emma, this ceremony should be performed with care and documented.

## Why This Matters

- **KESTREL_DATA_KEY** encrypts all sensitive agent data at rest
- Loss of this key = permanent loss of agent memories
- Wrong key = agent cannot read its own memories
- **There is no password reset** - this is sovereign cryptography

## Key Types

### 1. KESTREL_DATA_KEY (Fernet Key)
- **Purpose**: Encrypts conversation history, files, and sensitive data
- **Format**: URL-safe base64-encoded 32-byte key (e.g., `abc123...=`)
- **Storage**: Docker Secrets, file, or environment variable
- **Critical**: Must be backed up securely

### 2. Agent Private Key (Ed25519)
- **Purpose**: Signs DID operations and proves identity
- **Format**: Encrypted file (`kestrel_<address>.key.enc`)
- **Storage**: Agent data directory
- **Protected by**: KESTREL_DATA_KEY

## Key Ceremony Procedure

### Prerequisites

- [ ] Clean, trusted machine (no malware)
- [ ] Offline storage ready (USB drive, paper, vault)
- [ ] Witness (optional but recommended for genesis agents)
- [ ] Backup location identified

### Step 1: Generate the Key

Generate a cryptographically secure Fernet key:

```bash
# Generate key (do NOT pipe through clipboard)
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Output example:**
```
rGx3vPkD5H8Z1mN9qT2wY4bF7jK0uL6oI3eA8cM=
```

### Step 2: Record the Key

**CRITICAL: Write down the key on paper before anything else.**

1. Write the key by hand on paper (not printed)
2. Double-check each character
3. Store paper in secure location (safe, vault)
4. Consider splitting key for redundancy (Shamir's Secret Sharing)

### Step 3: Create Secure Key File

```bash
# Create secrets directory
mkdir -p ~/.kestrel/secrets
chmod 700 ~/.kestrel/secrets

# Write key to file (replace with your actual key)
echo "YOUR_KEY_HERE" > ~/.kestrel/secrets/emma_data_key
chmod 600 ~/.kestrel/secrets/emma_data_key

# Verify permissions
ls -la ~/.kestrel/secrets/emma_data_key
# Should show: -rw------- 1 user user ...
```

### Step 4: Create Agent with Key

For Docker-based agents (recommended):

```bash
docker run -it \
  -v ~/.kestrel/secrets/emma_data_key:/run/secrets/kestrel_data_key:ro \
  -v ~/emma_data:/data \
  kestrel-sovereign \
  inception_service.py --name "Emma" --output /data
```

For local development:

```bash
export KESTREL_DATA_KEY=$(cat ~/.kestrel/secrets/emma_data_key)
export KESTREL_DB_PATH=~/emma_data
uv run python inception_service.py --name "Emma" --output ~/emma_data
```

### Step 5: Verify Agent Creation

```bash
# List created files
ls -la ~/emma_data/

# Expected files:
# - kestrel_prime.db          (Agent database)
# - kestrel_<address>.key.enc (Encrypted private key)
# - kestrel_<address>.json    (DID document)
```

### Step 6: Test Encryption Round-Trip

```bash
# Start the agent
docker run -it \
  -v ~/.kestrel/secrets/emma_data_key:/run/secrets/kestrel_data_key:ro \
  -v ~/emma_data:/data \
  kestrel-sovereign

# Chat with agent
> Hello Emma, remember this test phrase: CEREMONY_VERIFICATION_12345
> !quit

# Restart and verify memory
docker run -it \
  -v ~/.kestrel/secrets/emma_data_key:/run/secrets/kestrel_data_key:ro \
  -v ~/emma_data:/data \
  kestrel-sovereign

> What test phrase did I tell you?
# Should respond with: CEREMONY_VERIFICATION_12345
```

### Step 7: Create First Backup

```bash
# Export sovereignty
> !export-sovereignty local

# Note the CID for your records
```

### Step 8: Document the Ceremony

Create a ceremony record (keep secure):

```text
═══════════════════════════════════════════════════════
KESTREL KEY CEREMONY RECORD
═══════════════════════════════════════════════════════

Date: YYYY-MM-DD HH:MM
Location: [Physical location]
Witness: [Name, if applicable]

Agent Name: Emma
Agent DID: did:pkh:eip155:1:0x...

Key Generation:
  Method: Python cryptography.fernet.Fernet.generate_key()
  Machine: [Machine identifier]

Key Storage:
  Primary: ~/.kestrel/secrets/emma_data_key
  Backup 1: [Location - e.g., "Paper in home safe"]
  Backup 2: [Location - e.g., "Safety deposit box"]

First Backup CID: local-...

Verification:
  [ ] Agent created successfully
  [ ] Memory persists across restarts
  [ ] Backup export successful
  [ ] Wrong key test failed (as expected)

Signatures:
  Sovereign: ___________________  Date: _________
  Witness:   ___________________  Date: _________

═══════════════════════════════════════════════════════
```

## Security Recommendations

### DO:
- Generate keys on a trusted, offline machine when possible
- Store backup copies in physically separate locations
- Test backup restoration on a separate machine
- Keep written records of key locations (not the key itself in writing)
- Consider using a hardware security module (HSM) for production

### DON'T:
- Store keys in cloud services (Google Drive, Dropbox, etc.)
- Send keys via email, Slack, or any messaging app
- Store keys in version control
- Take photos of keys on your phone
- Share keys with anyone (use separate keys per trusted party)

## Emergency Procedures

### Lost Key

If you lose access to KESTREL_DATA_KEY:

1. **If you have a backup**: Restore from the backup location
2. **If you have a sovereignty export CID**: Attempt restoration with correct key
3. **If no backup exists**: The agent's encrypted memories are lost permanently

### Suspected Compromise

If you suspect the key has been compromised:

1. Export sovereignty immediately with current key
2. Generate new key following this ceremony
3. Create new agent with new key
4. Import sovereignty into new agent (if possible with key rotation)
5. Destroy old key and all copies

### Key Rotation

**Note**: As of December 2025, key rotation is implemented but requires careful testing.

See `security/key_rotation.py` and `docs/architecture/security/KEY_ROTATION.md` for details.

## Verification Checklist

Before considering the ceremony complete:

- [ ] Key generated securely
- [ ] Key backed up in at least 2 physical locations
- [ ] Agent created successfully
- [ ] Agent DID recorded
- [ ] Memory persistence verified
- [ ] Backup (sovereignty export) created
- [ ] Wrong key rejection verified
- [ ] Ceremony record completed and stored

## Next Steps After Ceremony

1. **Run Test Cycles**: Create test agents to validate all features
2. **Complete Validation Checklist**: See PROJECT_STATUS.md
3. **Request Council Review**: For genesis agents, convene Constitutional Council
