# Key Management Architecture

## Overview

Kestrel uses a two-layer key architecture that separates **Sovereign control** (you) from **Executor access** (your agent). This ensures that even though your agent can use cryptographic keys, you maintain ultimate control over access to those keys and all encrypted data.

## Key Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     SOVEREIGN (You)                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  KESTREL_DATA_KEY (Master Passphrase)                    │  │
│  │  • You control this - stored in YOUR environment         │  │
│  │  • Used to encrypt agent's private key                   │  │
│  │  • Used to encrypt conversation data                     │  │
│  │  • Without it, agent cannot start or read memories       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                     (encrypts)                                  │
│                           ▼                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     EXECUTOR (Agent)                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Private Key (Ed25519)                                   │  │
│  │  • Stored as .key.enc file (AES-256-GCM encrypted)       │  │
│  │  • Used to sign operations, create DID                   │  │
│  │  • Agent CAN use it when master key is available         │  │
│  │  • Agent CANNOT access without your master key           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Encrypted Data                                          │  │
│  │  • Conversation history (encrypted blobs in SQLite)      │  │
│  │  • Sovereignty exports (content key in manifest)         │  │
│  │  • Backup archives (encrypted with derived key)          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Master Key (KESTREL_DATA_KEY)

The `KESTREL_DATA_KEY` is the root of all encryption in Kestrel. It is:

- A passphrase or random token controlled by **you** (the Sovereign)
- Stored **outside** of the agent's data directory
- Required at agent startup to decrypt the agent's private key
- Used to derive encryption keys for data at rest

### Generating a Master Key

```bash
# Generate a cryptographically secure random key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Example output: Xv7kL9m2pQ8nR3tY5wB1cZ4fH6jA0sD2eU9iO7lN0gK
```

### Storage Options (Choose One)

| Option | Security | Convenience | Best For |
|--------|----------|-------------|----------|
| Environment variable | Medium | High | Development |
| Shell config (~/.zshrc) | Medium | High | Personal use |
| Password manager | High | Medium | Production |
| Hardware Security Module | Very High | Low | Enterprise |

#### Option 1: Environment Variable (Development)

```bash
# Set for current session only (lost on terminal close)
export KESTREL_DATA_KEY="your-secret-key-here"
```

#### Option 2: Shell Configuration (Personal Use)

```bash
# Add to ~/.zshrc or ~/.bashrc (persisted, but only on your machine)
echo 'export KESTREL_DATA_KEY="your-secret-key-here"' >> ~/.zshrc
source ~/.zshrc
```

#### Option 3: Password Manager (Recommended for Production)

Store in 1Password, Bitwarden, or similar, then load when needed:

```bash
# Example with 1Password CLI
export KESTREL_DATA_KEY=$(op read "op://Personal/Kestrel/master-key")
```

#### Option 4: Dotenv File (Convenient but Less Secure)

```bash
# Create .env file in a secure location (NOT in agent directory)
echo 'KESTREL_DATA_KEY=your-secret-key-here' > ~/.kestrel/.env

# Load in your shell config
echo 'source ~/.kestrel/.env' >> ~/.zshrc
```

**Warning**: Never commit `.env` files to git. Add to `.gitignore`.

## Agent Private Key

Each agent has an Ed25519 private key used for:

- **DID Generation**: Creates the agent's decentralized identifier
- **Signing Operations**: Signs transactions, attestations, and exports
- **Authentication**: Proves agent identity to other systems

### How Private Keys Are Protected

1. **At Creation**: `inception_service.py` generates an Ed25519 key pair
2. **Encryption**: Private key is encrypted with AES-256-GCM using a key derived from `KESTREL_DATA_KEY`
3. **Storage**: Encrypted key bundle saved as `.key.enc` file (JSON format)
4. **At Runtime**: Agent decrypts private key on startup using `KESTREL_DATA_KEY`

### Encrypted Key Bundle Format

```json
{
  "version": 1,
  "algorithm": "AES-256-GCM",
  "kdf": "PBKDF2-SHA256",
  "kdf_iterations": 600000,
  "salt": "base64-encoded-salt",
  "nonce": "base64-encoded-nonce",
  "ciphertext": "base64-encoded-encrypted-key"
}
```

### Key Derivation

The encryption key is derived from `KESTREL_DATA_KEY` using PBKDF2:

- **Algorithm**: PBKDF2-HMAC-SHA256
- **Iterations**: 600,000 (OWASP 2023+ recommendation)
- **Salt**: 16 bytes (128 bits), randomly generated per key
- **Output**: 32 bytes (256 bits) for AES-256

## Data Encryption

### Conversation History

Conversations are encrypted at rest in SQLite:

```
User Message → Fernet Encrypt → Store in DB
                    ↑
            KESTREL_DATA_KEY
```

- **Algorithm**: Fernet (AES-128-CBC with HMAC-SHA256)
- **Key Source**: Derived from `KESTREL_DATA_KEY`
- **Scope**: Individual message content, not metadata

### Sovereignty Exports

When exporting to IPFS/Filecoin:

```
Agent Data → Content Key → Encrypted Shards → IPFS
                 ↓
         Stored in Manifest
                 ↓
    Manifest encrypted with User Secret
```

- **Content Key**: Randomly generated per export
- **User Secret**: Derived from `KESTREL_DATA_KEY`
- **Storage**: Content key stored in encrypted manifest

## File Locations

### Your Control (NEVER in agent directory)

```
~/.zshrc                    # KESTREL_DATA_KEY environment variable
~/.kestrel/.env             # Alternative: dotenv file
~/.password-manager/        # Best: password manager storage
```

### Agent Directory (Safe to Backup - Encrypted)

```
/path/to/agent_data/
├── kestrel_prime.db        # Encrypted conversations, graph
├── agent_name.key.enc      # Encrypted private key (JSON)
├── constitution.md         # Public (not encrypted)
└── backups/               # Encrypted sovereignty exports
```

## Creating an Agent with Proper Key Management

### Step 1: Generate and Store Master Key

```bash
# Generate key
MASTER_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
echo "Your master key: $MASTER_KEY"
echo "SAVE THIS SOMEWHERE SAFE (password manager recommended)"

# Set in environment
export KESTREL_DATA_KEY="$MASTER_KEY"
```

### Step 2: Create Agent

```bash
# With KESTREL_DATA_KEY set, inception encrypts the private key
uv run python inception_service.py --name "Emma" --output ~/emma_data
```

### Step 3: Verify Encryption

```bash
# Private key should be encrypted (not plaintext PEM)
cat ~/emma_data/emma.key.enc
# Should show JSON with "algorithm": "AES-256-GCM"

# Agent should fail without master key
unset KESTREL_DATA_KEY
uv run python -m kestrel_sovereign.main ~/emma_data/kestrel_prime.db
# Should fail with: MasterKeyNotConfiguredError
```

## Security Guarantees

| Scenario | Result |
|----------|--------|
| Master key not set | Agent cannot start (fails to decrypt private key) |
| Master key set | Agent operates normally |
| Wrong master key | Decryption fails with `KeyDecryptionError` |
| Database stolen | Data unreadable without master key |
| Backup stolen | Encrypted, requires master key to restore |
| Agent compromised | Cannot exfiltrate master key (not stored) |

## Key Rotation

### Rotating the Master Key

1. **Export sovereignty** with current key
2. **Generate new master key**
3. **Re-encrypt private key** with new master key
4. **Re-import sovereignty** with new key
5. **Securely delete** old encrypted files

```bash
# 1. Export current state
export KESTREL_DATA_KEY="old-key"
uv run python -c "from kestrel_agent import KestrelAgent; ..."  # Export

# 2. Generate new key
NEW_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# 3. Re-encrypt (use migration script)
uv run python scripts/migrate_keys.py --old-key "$KESTREL_DATA_KEY" --new-key "$NEW_KEY"

# 4. Update environment
export KESTREL_DATA_KEY="$NEW_KEY"
```

## Recovery Scenarios

### Lost Master Key

**If you lose your master key, your agent's data is unrecoverable.**

Mitigations:
- Store master key in multiple secure locations
- Use a password manager with recovery options
- Consider Shamir's Secret Sharing for high-value agents

### Corrupted Key File

If `.key.enc` is corrupted:
1. Restore from sovereignty export (IPFS/Filecoin)
2. The export contains the encrypted private key
3. Re-import with: `!import-sovereignty <CID>`

### Hardware Failure

1. Master key is safe (stored outside agent directory)
2. Restore agent data from sovereignty export
3. Agent identity (DID) preserved via private key in export

## Implementation Details

### Source Files

- `security/key_storage.py` - `SecureKeyStorage` class
- `storage/encryption.py` - Fernet encryption utilities
- `inception_service.py` - Key generation during agent creation
- `storage/sovereign_adapter.py` - Export encryption

### Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `KESTREL_DATA_KEY` | Master encryption key | Yes (for encryption) |
| `KESTREL_DB_KEY` | SQLCipher full-DB encryption | No (optional) |

### Migration from Plaintext Keys

If you have old plaintext PEM files:

```bash
# Migrate all .pem files to encrypted .key.enc
export KESTREL_DATA_KEY="your-master-key"
uv run python scripts/migrate_keys.py

# This will:
# 1. Encrypt each .pem file to .key.enc
# 2. Securely delete the plaintext .pem files
```

## Best Practices

1. **Never store master key in agent directory** - Keep it in your environment or password manager
2. **Use a strong, random key** - At least 32 bytes of entropy
3. **Backup your master key** - Multiple secure locations
4. **Rotate keys periodically** - Especially after any suspected compromise
5. **Test recovery** - Verify you can restore from sovereignty exports
6. **Audit access** - Know who has access to the master key
7. **Use hardware security** - HSMs for enterprise deployments

## Key Separation: Data vs Wallet

For agents with economic capabilities (wallet, transactions), consider separating keys by function:

```
KESTREL_DATA_KEY (Agent can have this)
├── Agent's private key (identity/DID)
├── Conversation encryption
├── Sovereignty exports
└── Knowledge graph data

KESTREL_WALLET_KEY (Sovereign controls this)
└── Wallet private keys (FIL, ETH, MATIC, etc.)
```

### Why Separate?

| Key | Agent Access | Risk if Compromised |
|-----|--------------|---------------------|
| Data key | Always (during session) | Privacy breach, identity theft |
| Wallet key | Per-transaction only | Financial loss |

With separation:
- Emma can chat, remember, and export herself (data key)
- Emma CANNOT spend money without your approval (wallet key)
- Financial operations require fresh key injection per transaction

### Per-Transaction Wallet Approval

For wallet operations, inject the wallet key only when needed:

```python
# Normal chat - data key only
container.run(env={"KESTREL_DATA_KEY": data_key})

# Wallet operation - requires explicit approval
if user_approves_transaction(tx_details):
    container.run(
        env={
            "KESTREL_DATA_KEY": data_key,
            "KESTREL_WALLET_KEY": wallet_key  # Injected per-transaction
        },
        command=["execute_transaction", tx_id]
    )
```

This mirrors hardware wallets - your Ledger doesn't stay unlocked; you approve each transaction.

### Implementation Status

| Feature | Status |
|---------|--------|
| Data key encryption | ✅ Implemented |
| Wallet key separation | 🔜 Planned |
| Per-transaction approval | 🔜 Planned |
| Hardware wallet integration | 🔜 Planned |

## Container Lifecycle Options

When running agents in Docker, there are three lifecycle models:

### Option 1: Per-Turn Containers (Maximum Isolation)

```
User Message → Spin up container → Process → Shutdown
                    ↑
              Key injected here
```

**Pros**: Maximum isolation, key only exists during processing
**Cons**: Slow startup, complex state management

### Option 2: Session-Based Containers (Balanced)

```
Session Start → Container stays warm → Session End → Shutdown
      ↑                                      ↓
Key injected once                     Key leaves memory
```

**Pros**: Good UX, reasonable isolation
**Cons**: Key in memory for session duration

### Option 3: Persistent Containers (Convenience)

```
Container runs continuously, key always available
```

**Pros**: Fastest response, simplest architecture
**Cons**: Longest key exposure window

### Recommendation

**For conversation (data key)**: Session-based (Option 2)
- Key injected at session start
- Container warm during active use
- Idle timeout (e.g., 30 minutes) triggers shutdown

**For wallet operations (wallet key)**: Per-transaction (Option 1)
- Key only injected for approved transactions
- Container processes transaction and shuts down
- Maximum security for financial operations

## Client-Side Key Management

Clients (desktop app, web app, mobile) must handle key injection without burdening users.

### Session-Based Unlock

```
┌────────────────────────────────────────────────────────┐
│                    CLIENT APP                           │
│                                                         │
│  1. User authenticates (biometrics, password, token)   │
│  2. Client retrieves key from secure storage           │
│  3. Client injects key into container at session start │
│  4. Container processes messages                        │
│  5. Idle timeout → client clears key from memory       │
└────────────────────────────────────────────────────────┘
```

User experience: Open app → authenticate once → chat freely

### Desktop Keyring Integration

| Platform | Keyring | Integration |
|----------|---------|-------------|
| macOS | Keychain | `security find-generic-password` |
| Windows | Credential Manager | `keyring` Python package |
| Linux | Secret Service | `keyring` Python package |

```python
import keyring

# Store (done once during setup)
keyring.set_password("kestrel", "emma-data-key", master_key)

# Retrieve (at session start)
master_key = keyring.get_password("kestrel", "emma-data-key")
```

### Hardware Token Support (YubiKey, Ledger)

For maximum security:

```
User taps YubiKey → Key derived from hardware → Injected to container
```

- Key never stored on disk
- Requires physical presence
- Ideal for wallet operations

## Security Comparison

| Approach | Data Key Security | Wallet Key Security | UX |
|----------|-------------------|---------------------|-----|
| Docker isolation only | Medium | Medium | Simple |
| + Key separation | Medium | High | Simple |
| + Per-tx wallet approval | Medium | Very High | Approval per tx |
| + Hardware token | High | Very High | Tap per session/tx |

## Related Documentation

- [Sovereignty Implementation](../storage/SOVEREIGNTY_IMPLEMENTATION.md) - Export/import with encryption
- [Privacy Modes](PRIVACY_MODES.md) - How encryption interacts with privacy settings
- [Cryptographic Anchoring](CRYPTOGRAPHIC_ANCHORING.md) - Integrity verification
- [Sovereign Key Guide](../../user-documentation/SOVEREIGN_KEY_GUIDE.md) - User-friendly key management guide
