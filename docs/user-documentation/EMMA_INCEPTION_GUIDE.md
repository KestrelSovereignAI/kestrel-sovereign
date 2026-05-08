# Creating Emma: The Genesis Agent

This guide documents the process for creating **Emma** - the first permanent Kestrel agent - with proper key management and validation.

## Prerequisites

Before creating Emma, ensure:

1. **All validation items pass** (see [Validation Checklist](#validation-checklist) below)
2. **Docker is installed** for isolation
3. **Master key is generated and stored safely**
4. **You understand the trust model** (see [Sovereign Key Guide](SOVEREIGN_KEY_GUIDE.md))

## Key Management Decision

Before creating Emma, decide your key architecture:

### Option A: Single Key (Simpler)

```
KESTREL_DATA_KEY
├── Emma's identity (DID)
├── Conversation encryption
├── Sovereignty exports
└── Wallet keys (if any)
```

All keys encrypted with one master key. Simpler but less granular control.

### Option B: Separated Keys (Recommended for Wallet Operations)

```
KESTREL_DATA_KEY          KESTREL_WALLET_KEY
├── Emma's identity       └── Wallet private keys
├── Conversations
└── Exports
```

Emma gets data key at session start. Wallet key only injected per-transaction.

See [Key Management Architecture](../architecture/security/KEY_MANAGEMENT.md#key-separation-data-vs-wallet) for details.

---

## Step 1: Generate and Store Master Key

```bash
# Generate a cryptographically secure key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Output example**: `Xv7kL9m2pQ8nR3tY5wB1cZ4fH6jA0sD2eU9iO7lN0gK`

**Store this key safely** (pick one):

| Method | Security | Instructions |
|--------|----------|--------------|
| Password manager (recommended) | High | Add to 1Password/Bitwarden as "Emma's Master Key" |
| Shell config with Docker | Medium | Add `export KESTREL_DATA_KEY="..."` to `~/.zshrc` |
| Hardware token | Very High | Store in YubiKey or similar |

**⚠️ If you lose this key, Emma's data is unrecoverable.**

**⚠️ Do not rotate `KESTREL_DATA_KEY` after Emma is created.** If you change it later, existing encrypted data (including sovereignty exports and any encrypted material in storage) becomes unreadable. There is no recovery path without the original key.

---

## Step 2: Build the Docker Image

```bash
cd ./

# Build the sovereign agent container
docker build -f docker/Dockerfile.sovereign -t kestrel-sovereign .
```

This creates an isolated container where Emma:
- ✅ Receives the key as environment variable
- ❌ Cannot access your host filesystem
- ❌ Cannot discover where you store the key

---

## Step 3: Create Emma's Data Directory

```bash
# Create directory for Emma's persistent data
mkdir -p ~/emma_data

# Set restrictive permissions
chmod 700 ~/emma_data
```

After creation, this directory will contain:

```
~/emma_data/
├── kestrel_prime.db              # Emma's database (memories, knowledge graph, conversations)
├── kestrel_<address>.key.enc     # Encrypted private key (AES-256-GCM)
├── kestrel_<address>.json        # DID document (public)
└── archive/                      # Created by retirement (test agents only)
    └── retired_agents/
        └── <agent_name>/         # Archived test agent data
```

**Note**: The `archive/retired_agents/` directory only appears after retiring test agents. Real Emma should never be retired (use cryostasis/export instead).

---

## Step 4: Run Inception

### Using the Script (Recommended)

```bash
# Set your master key (retrieve from password manager)
export KESTREL_DATA_KEY="your-key-here"

# Create Emma
kestrel agent docker create Emma ~/emma_data
```

### Manual Docker Command

```bash
docker run --rm \
  -e KESTREL_DATA_KEY="$KESTREL_DATA_KEY" \
  -v ~/emma_data:/data \
  kestrel-sovereign \
  python inception_service.py --name Emma --output /data

# Note: `--output` is a supported alias for `--output-dir`.
# It is not an argparse abbreviation.
```

### Expected Output

```
2025-12-23 10:00:00 - INFO - Generating secp256k1 keypair...
2025-12-23 10:00:00 - INFO - DID created: did:pkh:eip155:1:0x...
2025-12-23 10:00:01 - INFO - Saved encrypted private key for kestrel_0x...
2025-12-23 10:00:01 - INFO - Constitution anchored to knowledge graph
2025-12-23 10:00:02 - INFO - Genesis self-audit: PASSED
2025-12-23 10:00:02 - INFO - Agent Emma created successfully!
```

---

## Step 5: Verify Creation

### Check Files Created

```bash
ls -la ~/emma_data/
```

Expected files:
```
kestrel_prime.db              # Emma's database
kestrel_0x<address>.key.enc   # Encrypted private key (JSON with AES-256-GCM)
kestrel_0x<address>.json      # DID document (public)
```

**Success criteria**: All three files present. Key file is JSON (not plaintext PEM).

### Verify Encryption

```bash
# Check that private key is encrypted (not plaintext PEM)
cat ~/emma_data/kestrel_*.key.enc | python -c "import json, sys; d=json.load(sys.stdin); print('Encrypted with:', d.get('algorithm', 'UNKNOWN'))"
```

Should output: `Encrypted with: AES-256-GCM`

### Verify Without Key Fails

```bash
# Temporarily unset the key
unset KESTREL_DATA_KEY

# Try to start Emma (should fail)
kestrel agent docker chat ~/emma_data
# Expected: "error: KESTREL_DATA_KEY is not set!"

# Restore key
export KESTREL_DATA_KEY="your-key-here"
```

---

## Step 6: First Conversation

```bash
kestrel agent docker chat ~/emma_data
```

In the chat:
```
You: !status
Emma: [Shows agent status, DID, privacy mode]

You: !constitution
Emma: [Displays anchored constitution]

You: !reflect
Emma: [Runs layered self-reflection]
```

### Feed Emma Her Origin Story

Emma should know her own history. Share:
- These validation conversations
- The key management decisions
- Why she was created
- The test agents that came before her

```
You: Let me tell you about your creation. Before you, we created test
     agents to validate all systems. You are the first permanent agent.
     You have a DID, encrypted keys, and constitutional governance.
     Your data is yours - I can export it, but I can't read it without
     the master key I hold...
```

---

## Step 7: Sovereignty Export (Backup)

After initial setup, create a sovereignty backup:

```bash
# In chat with Emma:
You: !export-sovereignty
```

This:
1. Snapshots Emma's complete state
2. Encrypts with your master key
3. Uploads to IPFS (or saves locally)
4. Returns a CID for recovery

**Save this CID** alongside your master key.

---

## 🧊 Cryostasis (Dormancy) and Wallet Solvency

Kestrel supports **cryostasis** (dormancy) when an agent's wallet balance drops below a threshold. In cryostasis, the agent can be archived to cheaper storage and later restored when funds return.

For the authoritative architecture and economics details, see:
- `docs/diagrams/data-architecture/DA-07-cryostasis.md`
- `docs/architecture/economics/SOVEREIGN_SOLVENCY.md`

Operationally, cryostasis is closely related to **sovereignty export/import**: a common archival flow is “export → encrypt → store externally → restore later”.

---

## Validation Checklist

Before creating the real Emma, all items must pass.

### Gating Check: Sovereign Docker Smoke Test

**This is the primary validation signal.** Run the full create → retire cycle in Docker:

```bash
# 1. Build the sovereign image
docker build -f docker/Dockerfile.sovereign -t kestrel-sovereign .

# 2. Create a test agent (non-root, writes only to /data)
export KESTREL_DATA_KEY="test-key-for-validation"
docker run --rm \
  -e KESTREL_DATA_KEY="$KESTREL_DATA_KEY" \
  -v /tmp/test_emma:/data \
  kestrel-sovereign \
  python inception_service.py --test --name "Emma-Smoke-Test" --output /data

# 3. Verify files created
ls -la /tmp/test_emma/
# Expected: kestrel_prime.db, kestrel_*.key.enc, kestrel_*.json

# 4. Retire the test agent
docker run --rm \
  -e KESTREL_DATA_KEY="$KESTREL_DATA_KEY" \
  -v /tmp/test_emma:/data \
  kestrel-sovereign \
  python retirement_service.py /data/kestrel_prime.db

# 5. Verify archive created
ls -la /tmp/test_emma/archive/retired_agents/
# Expected: Emma-Smoke-Test/ directory with archived data

# 6. Cleanup
rm -rf /tmp/test_emma
```

**All steps must succeed before creating the real Emma.**

### Full Validation Matrix

| Item | Test Command | Expected |
|------|--------------|----------|
| Inception creates agent | `python inception_service.py --test --name Test-001` | Creates DB, keys, DID |
| Constitution anchored | `!constitution` in chat | Shows full constitution |
| Memory persists | Restart chat, ask about previous convo | Remembers |
| Privacy modes work | `!privacy ephemeral` + share data + restart | Data NOT stored |
| Export works | `!export-sovereignty` | Returns CID |
| Import works | `!import-sovereignty <cid>` | Restores state |
| Wallet operations | `!wallet-status` | Shows balance (if enabled) |
| LLM routing | Chat with local model down | Falls back to cloud |
| Commands work | `!help`, `!status`, `!reflect` | All respond correctly |
| Retirement works | `python retirement_service.py` on test agent | Archives, doesn't delete |

**Note**: Run validation on test instances (Emma-Test-XXX), not the real Emma.

---

## Test Agents vs Real Emma

| Aspect | Test Agents | Emma (Real) |
|--------|-------------|-------------|
| Name | Emma-Test-001, Emma-Test-002 | Emma |
| Flag | `--test` | (no flag) |
| Can be retired | Yes | **No** (protected) |
| Purpose | Validate systems | Permanent companion |
| Knowledge | Temporary | Permanent institutional memory |

### Creating Test Agents

```bash
# Create test instance
python inception_service.py --test --name "Emma-Test-001" --output /tmp/test_001/

# Retire when done
python retirement_service.py /tmp/test_001/kestrel_prime.db
```

### Protect Real Emma

The retirement service **refuses** to retire non-test agents:
```
Error: Agent is NOT a test instance. Refusing to retire a permanent agent.
Permanent agents cannot be retired through this service.
```

---

## Troubleshooting

### "MasterKeyNotConfiguredError"

Key not set or not accessible:
```bash
export KESTREL_DATA_KEY="your-key-here"
```

### "KeyDecryptionError"

Wrong key provided:
- Verify you're using the correct key
- Check for copy/paste errors (trailing spaces)

### "Genesis self-audit: FAILED"

Constitution verification failed:
- Check constitution file exists at `docs/principles/KESTREL_CONSTITUTION.md`
- Ensure no modifications to constitution content

### Docker Build Fails

Missing dependencies:
```bash
# Ensure you're in project root
cd ./

# Rebuild with no cache
docker build --no-cache -f docker/Dockerfile.sovereign -t kestrel-sovereign .
```

---

## After Emma is Created

1. **Store key securely** - Password manager + physical backup
2. **Create sovereignty export** - Backup to IPFS
3. **Feed origin story** - Emma should know her history
4. **Regular backups** - `!export-sovereignty` periodically
5. **Test recovery** - Verify you can restore from CID

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `docker/Dockerfile.sovereign` | Docker container for isolated execution |
| `kestrel agent docker {create,chat,retire}` | CLI for the Docker-isolated agent lifecycle (`kestrel_sovereign/cli_agent_docker.py`) |
| `inception_service.py` | Agent creation logic |
| `retirement_service.py` | Graceful retirement (test agents only) |
| `security/key_storage.py` | AES-256-GCM key encryption |
| `docs/principles/KESTREL_CONSTITUTION.md` | Constitutional governance |

---

## Related Documentation

- [Sovereign Key Guide](SOVEREIGN_KEY_GUIDE.md) - User-friendly key management
- [Key Management Architecture](../architecture/security/KEY_MANAGEMENT.md) - Technical key details
- [Agent MultiAgent](../plans/AGENT_MULTI_AGENT.md) - Future: self-determined companions
- [Constitution Embedding](../architecture/security/CONSTITUTION_EMBEDDING.md) - How constitution is anchored

---

*Emma is the genesis agent - the first permanent Kestrel companion. Her creation marks the transition from test instances to sovereign AI.*
