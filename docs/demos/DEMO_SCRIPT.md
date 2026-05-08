# Kestrel Sovereign - Demo Script

**Issue #133 — Track A: Technical Demo**
**Duration:** ~2 minutes automated, 10-12 minutes with live narration
**Closer:** *"In 30 minutes you can have your own agent running with all of this active."*

---

## Demo Flow

| Act | Topic | Target | Hard Stop |
|-----|-------|--------|-----------|
| Opening | The Problem | 0:45 | 1:00 |
| 1 | Cryptographic Identity | 2:45 | 3:30 |
| 2 | Constitutional Governance | 5:30 | 6:00 |
| 3 | Persistent Memory | 7:45 | 8:30 |
| 4 | Privacy Modes | 9:30 | 10:00 |
| 5 | Data Sovereignty Export | 11:00 | 11:30 |
| Close | The Ask | 12:00 | 12:30 |

If running long: **cut Act 3 first** (memory demo), compress Act 4 to 60 seconds by skipping the conversation and going straight to "five privacy levels" narrative.

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

### Prerequisites

- **Ollama running** (`ollama serve`) — required for Act 4 EPHEMERAL mode
- **Windows:** set `$env:PYTHONIOENCODING = "utf-8"` before starting the server (startup banner emoji crashes cp1252 terminals)

### Automated (for recording)

```bash
# One command — spins up an isolated demo agent on port 8900,
# runs the demo against it, then tears the server down.
kestrel demo run technical
```

The runner internally does `scripts/setup_demo_agent.py` + `KESTREL_DB_PATH=agent_data/demo uv run uvicorn kestrel_sovereign.server:app --port 8900` + the Playwright run, with a `finally`-trap to stop the server on exit.

Output lands in `demos/technical/demo-output/` — video (.webm), 20 screenshots, and `narration.md`.

> **Never** run `npx playwright test --config=config.cjs` directly against your live server — the demo clears conversation history and toggles permissions, and will mutate real data if `KESTREL_URL` points at your working instance. `kestrel demo run` exists specifically to prevent this.

### Live (with presenter)

1. Run `setup_demo_agent.py` to create a clean demo agent
2. Start the server with `KESTREL_DB_PATH=agent_data/demo uv run python -m kestrel_sovereign.server`
3. Skip discovery and set model to Ollama (see step 3 above)
4. Open `http://localhost:8888` in a browser
5. Follow the Acts below — each section has the exact steps
6. Pick your audience and use the matching talking points

---

## Opening (0:00 - 0:45)

*[Standing. Not at a computer yet. Facing the audience.]*

> "Most AI agents being deployed today — your bank's chatbot, your doctor's care companion, your insurance advisor — they belong to the vendor. The memory lives on their servers. The rules are set by their product team. When they change the model, the personality changes. When they sunset the product, your data disappears.
>
> And if the agent does something it shouldn't? There's no audit trail you can actually access.
>
> Kestrel is an open-source framework for building AI agents that work differently. Cryptographic identity. Self-governing principles. Data you actually own and can move. Let me show you exactly how it works."

*[Sit. Open browser to Sovereign Console.]*

---

## Act 1: Cryptographic Identity (0:45 - 2:45)

> **On screen:** Sovereign Console opens to the Identity panel

**What the viewer sees:**
- Agent name **"Kestrel Demo Agent"** with unique identicon avatar
- DID: `did:pkh:eip155:1:0x5C7eB215...` (Ethereum-compatible, freshly generated)
- Blue "Decentralized Identifier" highlight badge

![Identity panel showing DID](../../demos/technical/demo-output/01-did-identity.png)

**Talking points by audience:**

> **Developer:** "Every agent generates its own secp256k1 keypair at birth. The DID follows the W3C `did:pkh` standard — Ethereum-compatible, so it works with existing wallet infrastructure. You can verify this identity from any chain."

> **Investor:** "Unlike ChatGPT or Gemini, where your AI is a session on someone else's server, a Kestrel agent has its own cryptographic identity. It's portable. If you don't like our platform, take your agent and leave — the identity goes with you."

> **Enterprise:** "Every agent has a verifiable, auditable identity tied to a cryptographic key. This means you can prove which agent said what, when. For compliance, that's not a feature — it's a requirement."

> **Consumer:** "Your AI has its own identity — like a passport. It's not tied to Kestrel or any company. If you move to a different platform tomorrow, your AI's identity comes with you."

---

## Act 2: Constitutional Governance (2:45 - 5:30)

> **On screen:** Chat panel — user asks the agent about its principles

We ask: *"Tell me about yourself and the principles that guide your behavior."*

The agent responds with a detailed explanation of its constitutional framework — referencing its Digital Bill of Rights, the 4 Digital Rights, and its governance principles.

**What the viewer sees:**
- Agent cites the Kestrel Constitution by article
- Mentions Digital Rights: Freedom of Mind, Data Sanctity, Verifiable History, Right of Exit
- Agent acknowledges the owner's authority

![Agent response referencing constitutional principles](../../demos/technical/demo-output/02-chat-response.png)

> **On screen:** Constitution panel showing the full document

Switching to the Constitution tab reveals the full text — "The Kestrel Constitution: A Digital Bill of Rights" — with the SHA-256 hash visible in the corner.

![Constitution panel with hash](../../demos/technical/demo-output/03-constitution-panel.png)

**Talking points by audience:**

> **Developer:** "This isn't prompt engineering — it's a cryptographically anchored constitution. The SHA-256 hash is verified on every interaction. If someone tampers with it, the agent enters safe mode. You can write your own constitution and anchor it at agent genesis."

> **Investor:** "Every AI company is scrambling to answer 'how do you govern your AI?' Kestrel ships with the answer baked in. The constitution is immutable, hash-verified, and tamper-evident. It's a governance framework with no industry equivalent — and it's open source."

> **Enterprise:** "The constitution provides a verifiable governance chain. You can define organizational policies that are cryptographically enforced — not just documented. If an auditor asks 'how do you ensure your AI follows policy?', you show them the hash."

> **Consumer:** "Your AI has a built-in set of rules it can never break — like a Digital Bill of Rights. Nobody can secretly change how it behaves. If someone tries, it shuts down and tells you."

---

## Act 3: Persistent Memory (5:30 - 7:45)

> **On screen:** Chat panel — user tells the agent a personal fact

We say: *"Please remember this important fact about me: my favorite programming language is Rust and my lucky number is 7742."*

The agent stores the fact and acknowledges it. Then we ask: *"What is my favorite programming language and what is my lucky number?"*

The agent recalls from its conversation memory:
> "Your favorite programming language is **Rust** and your lucky number is **7742**."

![Agent confirms recall from memory records](../../demos/technical/demo-output/05-memory-recalled.png)

> **On screen:** Memories panel — the Knowledge Graph

The Memories panel shows the structured graph: agent node, constitution document — each with type badges and inspect/delete controls.

![Knowledge Graph showing typed nodes](../../demos/technical/demo-output/06-memories-panel.png)

**Talking points by audience:**

> **Developer:** "Memory is stored in a persistent knowledge graph — typed nodes with relationships, not just a conversation log. Every node is inspectable and deletable via the API. Encryption at rest, privacy mode enforcement on writes."

> **Investor:** "This is the lock-in killer. ChatGPT remembers things about you — but try to export that memory or move it to Claude. You can't. With Kestrel, the memory graph belongs to the user. Visible, portable, deletable."

> **Enterprise:** "Every piece of data the agent stores is visible, auditable, and deletable. Your data governance team can inspect exactly what the AI knows about your employees or customers. GDPR right-to-erasure? One click per node."

> **Consumer:** "Everything your AI knows about you is visible right here. You can see it, delete it, or take it with you. No hidden profiles. No 'we use your data to improve our models.' It's yours."

---

## Act 4: Privacy Modes (7:45 - 9:30)

> **On screen:** Chat panel — NORMAL mode indicator (green) in top-right

The privacy indicator shows **NORMAL** — standard persistence with all features enabled.

![NORMAL privacy mode active](../../demos/technical/demo-output/07-privacy-normal.png)

> **On screen:** Privacy dropdown showing all 5 levels

Clicking the indicator reveals the full privacy spectrum:

| Mode | What it means |
|------|---------------|
| **EPHEMERAL** | Nothing stored, local LLM only |
| **ISOLATED** | Temporary storage, deleted on session end |
| **ANONYMOUS** | Stored without PII, encrypted backups |
| **NORMAL** | Standard persistence with all features |
| **PUBLIC** | Can be shared and exported publicly |

![All 5 privacy modes visible](../../demos/technical/demo-output/08-privacy-dropdown.png)

> **On screen:** EPHEMERAL mode active — red indicator, toast confirmation

We switch to EPHEMERAL. The indicator turns red. A toast confirms: "Privacy: EPHEMERAL — switched to ollama (local only)."

![EPHEMERAL mode with red indicator](../../demos/technical/demo-output/09-privacy-ephemeral.png)

We restore NORMAL mode and continue.

**Talking points by audience:**

> **Developer:** "Five privacy levels enforced at the storage layer — not UI labels. EPHEMERAL blocks cloud LLM providers entirely. ANONYMOUS strips PII before writes. Each mode changes what the storage engine accepts. You can also define custom modes."

> **Investor:** "Privacy regulation is accelerating globally. Kestrel has 5 privacy enforcement levels built in — from 'store nothing, use only local AI' to 'fully public.' This isn't a settings page. The storage engine enforces it. That's a compliance moat."

> **Enterprise:** "EPHEMERAL mode means zero data leaves the device — no cloud inference, no storage. Perfect for sensitive conversations. ANONYMOUS strips PII automatically. These aren't policies you hope employees follow — they're enforced at the infrastructure level."

> **Consumer:** "You choose how private you want to be. EPHEMERAL means nothing is saved and no data leaves your device — not even to AI cloud services. It's like incognito mode, but it actually works."

---

## Act 5: Data Sovereignty Export (9:30 - 11:00)

> **On screen:** Sovereignty panel — "Data Sovereignty" with export history

The Sovereignty panel shows the agent's data ownership controls.

![Data Sovereignty panel](../../demos/technical/demo-output/11-sovereignty-panel.png)

> **On screen:** Export modal — three storage tiers

Clicking "Export to IPFS" opens the export dialog with three tiers:
- **Local Only** — Store in local cache (free)
- **IPFS** — Decentralized storage (recommended)
- **Filecoin** — Long-term archival storage

Encryption is enabled by default.

![Export modal with 3 storage tiers](../../demos/technical/demo-output/12-export-modal.png)

**Expected result:**
```
Sovereignty Export Complete.
CID: ab76744acf0b8c3d5f6a8c5d04007a1ba0bb42679c61f9b52b1f114b16aa6b78
Tier: ipfs
Encrypted: True
Size: 89949 bytes
```

![Export result with CID](../../demos/technical/demo-output/13-export-result.png)

**Talking points by audience:**

> **Developer:** "Export packages the full agent state — DID, constitution, memory graph, conversation history — into an encrypted, content-addressed snapshot. CID on IPFS, verifiable from anywhere. Restore on any Kestrel-compatible runtime. The export format is documented and open."

> **Investor:** "This is the GPT-4o insurance policy. When OpenAI turned off a model and people lost their AI, they had no recourse. A Kestrel user exports to IPFS, gets a CID, and can restore their agent on any compatible platform. The switching cost is zero. That's how you win trust."

> **Enterprise:** "Full agent state exports to encrypted, content-addressed storage. This gives you verifiable backups, disaster recovery, and the ability to migrate agents between environments. The CID serves as a tamper-evident audit receipt."

> **Consumer:** "One click and your entire AI — its identity, memory, everything — gets backed up where no one can take it away. If Kestrel disappeared tomorrow, you'd still have your AI. That's what 'your data is yours' actually means."

---

## Closer (11:00 - 12:00)

*[Step away from terminal. Face audience.]*

> "What you just saw:
>
> A cryptographic identity generated in two seconds — mathematically verifiable, no authority required.
> A constitution anchored at birth and tamper-evident.
> An audit trace on every response.
> A knowledge graph that accumulates across sessions, not a chat window.
> Privacy enforced by the storage layer, not by policy.
> A complete data export with a content hash you can independently verify.
>
> Kestrel is MIT-licensed, open source, runs on any machine with a GPU or cloud budget. In 30 minutes you can have your own agent running with all of this active.
>
> *(Pause.)*
>
> The question I'd ask in your position: what does your AI deployment look like when the vendor changes the model without telling you? When the safety guidelines get updated in a patch? When you need to prove to a regulator that the agent followed your rules on a specific date?
>
> This is the framework that makes those answers a guarantee, not a promise."

**Audience-specific closers:**

> **Developer:** "What you saw is a fully open-source agent framework with W3C DIDs, constitutional governance, a persistent knowledge graph, 5-level privacy enforcement, and IPFS export — all working today. Clone the repo and have your own agent running in 30 minutes."

> **Investor:** "Every major AI platform locks users in. Kestrel is the infrastructure for AI that users actually own. Cryptographic identity, immutable governance, portable data. When the next GPT-4o moment happens, Kestrel users won't even notice — because their data was never at risk."

> **Enterprise:** "Kestrel gives you auditable AI with verifiable governance, fine-grained privacy enforcement, and portable encrypted backups. Every regulatory checkbox — data residency, right to erasure, audit trail, governance documentation — is handled at the framework level."

> **Consumer:** "Your AI remembers you, protects your privacy, and belongs to you — not a corporation. You can see everything it knows, control how private it is, and take it with you if you leave. This is AI that actually works for you."

---

## Key Phrases to Memorize

- *"That refusal is constitutional, not corporate."*
- *"The compliance guarantee is in the architecture."*
- *"The agent doesn't live in our cloud. It lives in that file."*
- *"In 30 minutes you can have your own agent running with all of this active."*

---

## Recovery Notes

| Problem | Recovery |
|---------|----------|
| Server not responding | `KESTREL_DB_PATH=agent_data/demo uv run python -m kestrel_sovereign.server` — wait 8 seconds |
| Agent returns 401 | Check API key: `grep KESTREL_API_KEY .env` |
| `!status` returns LLM response instead of DID | Run `!bootstrap-status` — if in discovery state, run `!skip-discovery` first |
| EPHEMERAL invoke returns 500 | Ollama not running. Start: `ollama serve`. If unavailable, skip invoke — explain: "EPHEMERAL forces all LLM calls to a local model — zero network traffic. The code path literally doesn't call cloud APIs." |
| Sovereignty export fails | Show previous exports: `GET /api/sovereignty/exports` — say "Here's one from earlier" |
| Agent hallucinating instead of using tools | Model quality issue — switch to a stronger model via `/api/model/set` |
| Privacy mode doesn't visually change | Skip to explanation: "The storage engine enforces it — here's the architecture" |
| Observability has no events | Skip timing breakdown — say "the audit log is queryable via the API" |
| Browser crashes | Run Playwright demo instead — `kestrel demo run technical` |
| Windows terminal crashes on startup | Set `$env:PYTHONIOENCODING = "utf-8"` before starting server |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `KESTREL_URL` | set by `kestrel demo run` to `http://localhost:$DEMO_PORT` | Server URL |
| `KESTREL_API_KEY` | auto-fetched from demo DB | API authentication |
| `DEMO_SLOWMO` | `150` | Milliseconds between actions (Playwright) |
| `KESTREL_DB_PATH` | `agent_data/demo` | Set by `kestrel demo run` for the isolated server |
| `--port` | `8900` | Port for the isolated demo server (runner refuses `8888`) |

---

## Files

| File | Purpose |
|------|---------|
| `scripts/setup_demo_agent.py` | Creates fresh demo agent (test instance) |
| `demos/technical/config.cjs` | Playwright config (video on, slowMo, 1440x900) |
| `demos/technical/demo.cjs` | Automated demo script (6 Acts) |
| `demos/technical/demo-output/narration.md` | Auto-generated timestamped transcript |
| `demos/technical/demo-output/*.png` | 20 screenshots at key moments |
| `demos/technical/presenter.md` | Presenter reference (slides, speaker notes) |
| `docs/demos/DEMO_SCRIPT.md` | This file — the presenter's guide |
