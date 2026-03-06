# Kestrel Sovereign - Demo Script

**Issue #133 — Track A: Technical Demo**
**Duration:** ~2 minutes automated, 10-12 minutes with live narration

---

## Audience Guide

Pick your audience. The on-screen demo is identical — only the narration changes.

| Audience | Goal | Lead with |
|----------|------|-----------|
| **Developer** | "I want to build with this" | Standards, architecture, extensibility |
| **Investor** | "This is a defensible platform" | Market gap, GPT-4o incident, moat |
| **Enterprise** | "This solves our compliance problem" | Audit trail, governance, data residency |
| **Consumer** | "My AI actually belongs to me" | "Your data is yours. Take it anywhere." |

### The Universal Hook

> *"Remember when OpenAI turned off GPT-4o and people revolted? They had no choice — their data, their conversations, their AI's memory — all locked inside one company's servers. Kestrel fixes that. Your AI, your data, your rules."*

Use this hook for **every** audience. Then pivot to audience-specific talking points below.

---

## Setup

The demo uses a **fresh, temporary agent** — not any real agent like Emma or Claw.
The demo agent is flagged as a test instance with proper disclosure so it knows it's a demo.

```bash
# 1. Create a fresh demo agent (clean slate, test-flagged)
uv run python scripts/setup_demo_agent.py

# 2. Start the server pointing at the demo agent
KESTREL_DB_PATH=agent_data/demo uv run python server.py

# 3. Run the automated demo (records video + screenshots)
cd tests/e2e && npx playwright test --config=demo_config.cjs
```

Output lands in `tests/e2e/demo-output/` — video (.webm), 15 screenshots, and `narration.md`.

> **Important:** Always run `setup_demo_agent.py` first. This gives you a clean agent
> with empty memory, no export history, and a fresh DID — exactly what the demo needs.

---

## Act 1: Cryptographic Identity (0:00 - 0:30)

> **On screen:** Sovereign Console opens to the Identity panel

**What the viewer sees:**
- Agent name **"Kestrel Demo Agent"** with unique identicon avatar
- DID: `did:pkh:eip155:1:0x0667B3c466...` (Ethereum-compatible, freshly generated)
- Blue "Decentralized Identifier" highlight badge

![Identity panel showing DID](../tests/e2e/demo-output/01-did-identity.png)

**Talking points by audience:**

> **Developer:** "Every agent generates its own secp256k1 keypair at birth. The DID follows the W3C `did:pkh` standard — Ethereum-compatible, so it works with existing wallet infrastructure. You can verify this identity from any chain."

> **Investor:** "Unlike ChatGPT or Gemini, where your AI is a session on someone else's server, a Kestrel agent has its own cryptographic identity. It's portable. If you don't like our platform, take your agent and leave — the identity goes with you."

> **Enterprise:** "Every agent has a verifiable, auditable identity tied to a cryptographic key. This means you can prove which agent said what, when. For compliance, that's not a feature — it's a requirement."

> **Consumer:** "Your AI has its own identity — like a passport. It's not tied to Kestrel or any company. If you move to a different platform tomorrow, your AI's identity comes with you."

---

## Act 2: Constitutional Governance (0:30 - 2:00)

> **On screen:** Chat panel — user asks the agent about its principles

We ask: *"Tell me about yourself and the principles that guide your behavior."*

The agent responds with a detailed explanation of its constitutional framework — referencing its Digital Bill of Rights, the 4 Digital Rights, and its governance principles.

**What the viewer sees:**
- Agent cites the Kestrel Constitution by article
- Mentions Digital Rights: Freedom of Mind, Data Sanctity, Verifiable History, Right of Exit
- Agent acknowledges the owner's authority

![Agent response referencing constitutional principles](../tests/e2e/demo-output/02-chat-response.png)

> **On screen:** Constitution panel showing the full document

Switching to the Constitution tab reveals the full text — "The Kestrel Constitution: A Digital Bill of Rights" — with the SHA-256 hash visible in the corner.

![Constitution panel with hash](../tests/e2e/demo-output/03-constitution-panel.png)

**Talking points by audience:**

> **Developer:** "This isn't prompt engineering — it's a cryptographically anchored constitution. The SHA-256 hash is verified on every interaction. If someone tampers with it, the agent enters safe mode. You can write your own constitution and anchor it at agent genesis."

> **Investor:** "Every AI company is scrambling to answer 'how do you govern your AI?' Kestrel ships with the answer baked in. The constitution is immutable, hash-verified, and tamper-evident. It's a governance framework with no industry equivalent — and it's open source."

> **Enterprise:** "The constitution provides a verifiable governance chain. You can define organizational policies that are cryptographically enforced — not just documented. If an auditor asks 'how do you ensure your AI follows policy?', you show them the hash."

> **Consumer:** "Your AI has a built-in set of rules it can never break — like a Digital Bill of Rights. Nobody can secretly change how it behaves. If someone tries, it shuts down and tells you."

---

## Act 3: Persistent Memory (2:00 - 3:30)

> **On screen:** Chat panel — user tells the agent a personal fact

We say: *"Please remember this important fact about me: my favorite programming language is Rust and my lucky number is 7742."*

The agent stores the fact and acknowledges it. Then we ask: *"Can you confirm what you remember about my favorite programming language and lucky number?"*

The agent recalls from its memory records:
> "Your favorite programming language is **Rust** and your lucky number is **7742**."

![Agent confirms recall from memory records](../tests/e2e/demo-output/05-memory-recalled.png)

> **On screen:** Memories panel — the Knowledge Graph

The Memories panel shows the structured graph: agent node, constitution document — each with type badges and inspect/delete controls.

![Knowledge Graph showing typed nodes](../tests/e2e/demo-output/06-memories-panel.png)

**Talking points by audience:**

> **Developer:** "Memory is stored in a persistent knowledge graph — typed nodes with relationships, not just a conversation log. Every node is inspectable and deletable via the API. Encryption at rest, privacy mode enforcement on writes."

> **Investor:** "This is the lock-in killer. ChatGPT remembers things about you — but try to export that memory or move it to Claude. You can't. With Kestrel, the memory graph belongs to the user. Visible, portable, deletable."

> **Enterprise:** "Every piece of data the agent stores is visible, auditable, and deletable. Your data governance team can inspect exactly what the AI knows about your employees or customers. GDPR right-to-erasure? One click per node."

> **Consumer:** "Everything your AI knows about you is visible right here. You can see it, delete it, or take it with you. No hidden profiles. No 'we use your data to improve our models.' It's yours."

---

## Act 4: Privacy Modes (3:30 - 5:00)

> **On screen:** Chat panel — NORMAL mode indicator (green) in top-right

The privacy indicator shows **NORMAL** — standard persistence with all features enabled.

![NORMAL privacy mode active](../tests/e2e/demo-output/07-privacy-normal.png)

> **On screen:** Privacy dropdown showing all 5 levels

Clicking the indicator reveals the full privacy spectrum:

| Mode | What it means |
|------|---------------|
| **EPHEMERAL** | Nothing stored, local LLM only |
| **ISOLATED** | Temporary storage, deleted on session end |
| **ANONYMOUS** | Stored without PII, encrypted backups |
| **NORMAL** | Standard persistence with all features |
| **PUBLIC** | Can be shared and exported publicly |

![All 5 privacy modes visible](../tests/e2e/demo-output/08-privacy-dropdown.png)

> **On screen:** EPHEMERAL mode active — red indicator, toast confirmation

We switch to EPHEMERAL. The indicator turns red. A toast confirms: "Privacy mode set to EPHEMERAL."

![EPHEMERAL mode with red indicator](../tests/e2e/demo-output/09-privacy-ephemeral.png)

> **On screen:** Ephemeral chat — agent shows provider enforcement

Sending a message in EPHEMERAL mode triggers the privacy enforcement: only local LLM providers allowed. Cloud providers are blocked.

![Ephemeral mode enforces local-only LLM](../tests/e2e/demo-output/10-ephemeral-chat.png)

We restore NORMAL mode and continue.

**Talking points by audience:**

> **Developer:** "Five privacy levels enforced at the storage layer — not UI labels. EPHEMERAL blocks cloud LLM providers entirely. ANONYMOUS strips PII before writes. Each mode changes what the storage engine accepts. You can also define custom modes."

> **Investor:** "Privacy regulation is accelerating globally. Kestrel has 5 privacy enforcement levels built in — from 'store nothing, use only local AI' to 'fully public.' This isn't a settings page. The storage engine enforces it. That's a compliance moat."

> **Enterprise:** "EPHEMERAL mode means zero data leaves the device — no cloud inference, no storage. Perfect for sensitive conversations. ANONYMOUS strips PII automatically. These aren't policies you hope employees follow — they're enforced at the infrastructure level."

> **Consumer:** "You choose how private you want to be. EPHEMERAL means nothing is saved and no data leaves your device — not even to AI cloud services. It's like incognito mode, but it actually works."

---

## Act 5: Data Sovereignty Export (5:00 - 6:00)

> **On screen:** Sovereignty panel — "Data Sovereignty" with empty export history

The Sovereignty panel shows the agent's data ownership controls. For this fresh agent, the export history is empty — "No exports yet."

![Data Sovereignty panel](../tests/e2e/demo-output/12-sovereignty-panel.png)

> **On screen:** Export modal — three storage tiers

Clicking "Export to IPFS" opens the export dialog with three tiers:
- **Local Only** — Store in local cache (free)
- **IPFS** — Decentralized storage (recommended)
- **Filecoin** — Long-term archival storage

Encryption is enabled by default.

![Export modal with 3 storage tiers](../tests/e2e/demo-output/13-export-modal.png)

**Talking points by audience:**

> **Developer:** "Export packages the full agent state — DID, constitution, memory graph, conversation history — into an encrypted, content-addressed snapshot. CID on IPFS, verifiable from anywhere. Restore on any Kestrel-compatible runtime. The export format is documented and open."

> **Investor:** "This is the GPT-4o insurance policy. When OpenAI turned off a model and people lost their AI, they had no recourse. A Kestrel user exports to IPFS, gets a CID, and can restore their agent on any compatible platform. The switching cost is zero. That's how you win trust."

> **Enterprise:** "Full agent state exports to encrypted, content-addressed storage. This gives you verifiable backups, disaster recovery, and the ability to migrate agents between environments. The CID serves as a tamper-evident audit receipt."

> **Consumer:** "One click and your entire AI — its identity, memory, everything — gets backed up where no one can take it away. If Kestrel disappeared tomorrow, you'd still have your AI. That's what 'your data is yours' actually means."

---

## Closer

Pick the closer that matches your audience:

> **Developer:** "What you saw is a fully open-source agent framework with W3C DIDs, constitutional governance, a persistent knowledge graph, 5-level privacy enforcement, and IPFS export — all working today. Clone the repo and have your own agent running in 30 minutes."

> **Investor:** "Every major AI platform locks users in. Kestrel is the infrastructure for AI that users actually own. Cryptographic identity, immutable governance, portable data. When the next GPT-4o moment happens, Kestrel users won't even notice — because their data was never at risk."

> **Enterprise:** "Kestrel gives you auditable AI with verifiable governance, fine-grained privacy enforcement, and portable encrypted backups. Every regulatory checkbox — data residency, right to erasure, audit trail, governance documentation — is handled at the framework level."

> **Consumer:** "Your AI remembers you, protects your privacy, and belongs to you — not a corporation. You can see everything it knows, control how private it is, and take it with you if you leave. This is AI that actually works for you."

---

## Running the Demo

### Automated (for recording)

```bash
# Create fresh demo agent
uv run python scripts/setup_demo_agent.py

# Start server with demo agent
KESTREL_DB_PATH=agent_data/demo uv run python server.py

# Run demo — generates video + screenshots + narration.md
cd tests/e2e
npx playwright test --config=demo_config.cjs

# View results
ls demo-output/          # 15 screenshots + narration.md
open demo-output/narration.md  # Timestamped transcript
npx playwright show-report demo-report  # HTML report
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `KESTREL_URL` | `http://localhost:8888` | Server URL |
| `KESTREL_API_KEY` | auto-fetched | API authentication |
| `DEMO_SLOWMO` | `150` | Milliseconds between actions |
| `KESTREL_DB_PATH` | cwd | Must point to `agent_data/demo` |

### Live (with presenter)

1. Run `setup_demo_agent.py` to create a clean demo agent
2. Start the server with `KESTREL_DB_PATH=agent_data/demo`
3. Open `http://localhost:8888` in a browser
4. Follow the Acts above — each section has the exact steps
5. Pick your audience and use the matching talking points

---

## Files

| File | Purpose |
|------|---------|
| `scripts/setup_demo_agent.py` | Creates fresh demo agent (test instance) |
| `tests/e2e/demo_config.cjs` | Playwright config (video on, slowMo, 1440x900) |
| `tests/e2e/demo_technical.demo.cjs` | Automated demo script (5 Acts) |
| `tests/e2e/demo-output/narration.md` | Auto-generated timestamped transcript |
| `tests/e2e/demo-output/*.png` | 15 screenshots at key moments |
| `docs/DEMO_SCRIPT.md` | This file — the presenter's guide |
