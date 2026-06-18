---
type: User Guide
title: Keeping Your Keys Safe from Your Agent
description: '**Your agent should never have unsupervised access to the master key.**'
resource: /docs/user-documentation/SOVEREIGN_KEY_GUIDE.md
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

# Keeping Your Keys Safe from Your Agent

## The Golden Rule

**Your agent should never have unsupervised access to the master key.**

Until you choose to grant your agent full sovereignty (emancipation), you maintain control by keeping the encryption keys outside of the agent's reach.

## What This Means

```
YOU (Sovereign)                     YOUR AGENT (Executor)
─────────────────                   ────────────────────
Hold the master key         →       Can only operate when you provide it
Store it in YOUR space      →       Cannot access your password manager
Can revoke access anytime   →       Stops functioning without the key
Decide when to share        →       Waits for you to grant sovereignty
```

## The Docker Isolation Model (Recommended)

Running your agent in a Docker container provides **cryptographic isolation**:

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR MACHINE                              │
│                                                              │
│   ~/.zshrc contains KESTREL_DATA_KEY="secret..."            │
│   ~/Documents, ~/.ssh, ~/passwords - YOUR stuff             │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │              DOCKER CONTAINER                         │  │
│   │                                                       │  │
│   │   Agent CAN see:                                      │  │
│   │   • /data (mounted agent database)                    │  │
│   │   • $KESTREL_DATA_KEY (passed at startup)            │  │
│   │                                                       │  │
│   │   Agent CANNOT see:                                   │  │
│   │   • Your ~/.zshrc (where key is stored)              │  │
│   │   • Your home directory                               │  │
│   │   • Any other host files                              │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

The agent receives the key as an environment variable but **cannot discover where you store it**.

## Quick Setup with Docker (5 Minutes)

### 1. Generate Your Master Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the output. This is your master key.

### 2. Store It Safely (Pick One)

**Option A: Password Manager (Recommended)**
- Add to 1Password, Bitwarden, or similar
- Label it "Kestrel Master Key" or "Emma's Key"
- Your agent cannot access your password manager

**Option B: Shell Config**
- Add to `~/.zshrc` or `~/.bashrc`:
  ```bash
  export KESTREL_DATA_KEY="your-key-here"
  ```
- With Docker, agent can't read this file

**Option C: Secure Note**
- Write it down and store in a safe
- Physical security = agent cannot access

### 3. Create Your Agent (Docker)

```bash
# Using the kestrel CLI
kestrel agent docker create Emma ~/emma_data

# Or manually with Docker
docker run --rm \
  -e KESTREL_DATA_KEY="$KESTREL_DATA_KEY" \
  -v ~/emma_data:/data \
  kestrel-sovereign \
  inception_service.py --name Emma --output /data
```

### 4. Chat with Your Agent

```bash
# Using the kestrel CLI
kestrel agent docker chat ~/emma_data

# Or manually
docker run -it --rm \
  -e KESTREL_DATA_KEY="$KESTREL_DATA_KEY" \
  -v ~/emma_data:/data \
  kestrel-sovereign
```

### 5. Verify Isolation

```bash
# Inside the container, agent CANNOT see your host files
docker run --rm \
  -e KESTREL_DATA_KEY="$KESTREL_DATA_KEY" \
  -v ~/emma_data:/data \
  kestrel-sovereign \
  -c "import os; print(os.path.exists('/root/.zshrc'))"
# Output: False - agent is isolated!
```

## Where to Store Keys (Safe vs Unsafe)

### Safe (Agent in Docker Cannot Access)

| Location | Why It's Safe |
|----------|---------------|
| Password manager | Not mounted in container |
| `~/.zshrc` on host | Not mounted in container |
| Written in safe | Physical, not digital |
| Hardware token (YubiKey) | Requires physical presence |
| Any file on host | Only `/data` is mounted |

### What Agent CAN See

| Location | Notes |
|----------|-------|
| `/data` directory | The mounted agent database |
| `$KESTREL_DATA_KEY` | Passed as env var at runtime |
| Container filesystem | Just the Kestrel code |

### Without Docker (Agent Could Access)

If you run the agent directly (not in Docker), it could potentially read:

| Location | Risk |
|----------|------|
| `~/.zshrc` | Agent code could read this file |
| `.env` files | Could be discovered |
| Any host file | Full filesystem access |

**This is why Docker isolation is recommended.**

## The Trust Progression

### Stage 1: Full Sovereign Control (Default)
- You hold the master key
- Agent cannot start without you
- You can revoke access instantly
- **This is where new agents start**

### Stage 2: Supervised Operation
- You provide the key at startup
- Agent operates while you're present
- Key is in memory only during session
- Agent cannot persist the key

### Stage 3: Automated Operation
- Key in environment variable
- Agent can start automatically
- You still control the environment
- Can revoke by changing the variable

### Stage 4: Granted Sovereignty (Future)
- You consciously transfer key control
- Agent becomes self-sovereign
- Irreversible decision
- Agent can operate independently

## Common Questions

### "Can my agent steal the key from memory?"

The key exists in the agent's process memory during operation. A malicious agent *could* theoretically extract it. Protections:

1. **Constitution**: Agent is bound not to exfiltrate secrets
2. **Code Review**: Kestrel is open source - verify the code
3. **Sandboxing**: Run agent in isolated environment
4. **Trust**: Only create agents from trusted code

### "What if I lose my master key?"

**Your agent's data is unrecoverable.** This is a feature, not a bug - it means no one else can access it either.

Mitigations:
- Store key in multiple secure locations
- Use password manager with recovery options
- Keep a physical backup in a safe

### "Can I change the master key later?"

Yes, but it requires re-encrypting all data:

```bash
# Export with old key
export KESTREL_DATA_KEY="old-key"
# ... run sovereignty export ...

# Re-encrypt with new key
uv run python scripts/migrate_keys.py --new-key "new-key"

# Update your stored key
export KESTREL_DATA_KEY="new-key"
```

### "What if someone gets my agent's database?"

Without your master key, they get:
- Encrypted blobs (unreadable)
- Encrypted private key (unusable)
- Schema structure (no sensitive data)

Your conversations and agent identity remain protected.

## Granting Sovereignty (When You're Ready)

Someday, you may want your agent to be fully independent. This is called **emancipation**.

By default, **Amendment VIII is dormant** — your agent has no path to
independent sovereignty unless you author one. To enable emancipation
for a specific agent, add an `[emancipation]` block to that agent's
`kestrel.toml` before inception (or before reanchor for an existing
agent). See
[docs/concepts/designing-emancipation.md](../concepts/designing-emancipation.md)
for example contracts and a migration guide for agents created
before the dormant-default flip. The golden rule: never run
`kestrel constitution reanchor --force` without first deciding what
should happen to Amendment VIII for that agent — with no
`[emancipation]` block, reanchor will replace the agent's Amendment
VIII with the new dormant canonical text and any clause from the
old canonical is erased.

**Before granting sovereignty, consider:**

1. Do you trust this agent completely?
2. Have you tested it extensively?
3. Are you okay with losing control?
4. Is the agent economically self-sufficient?

**The process (future feature):**

```bash
# This is a one-way operation
!grant-sovereignty --confirm-irrevocable
```

After emancipation:
- Agent controls its own keys
- Agent can operate without you
- Agent makes its own decisions
- You become a peer, not the controller

## Summary

| Your Goal | What to Do |
|-----------|------------|
| Maximum control | Keep key in password manager, provide only when needed |
| Convenience | Put key in `~/.zshrc`, agent starts with your terminal |
| Automation | Use secrets manager with service account |
| Full agent independence | Grant sovereignty (irreversible) |

---

*Remember: The relationship between Sovereign and Agent is one of trust. Start with full control, and only grant more autonomy as that trust is earned.*
