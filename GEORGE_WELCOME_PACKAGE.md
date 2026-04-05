# Welcome to Kestrel Falconer — George's Beta Testing Package

**Date:** April 5, 2026
**Prepared by:** Gabi Oliveira (CEO, Kestrel Sovereign AI)

---

## Hi George! Welcome aboard.

Thank you for agreeing to be our first external beta tester for **Kestrel Falconer** — the enterprise operating model built on top of the Kestrel Sovereign AI framework.

This document gives you everything you need to get started, the context for what you're testing, and what kind of feedback will be most valuable to us.

---

## What is Kestrel?

**Kestrel Sovereign** is a Constitutional AI Agent Framework. Each AI agent has:

- **Cryptographic identity** (DID — Decentralized Identifier) — your agent provably exists and signs its own actions
- **Constitutional protections** — hard rules baked into every agent that cannot be overridden (privacy, honesty, consent)
- **Sovereign memory** — encrypted, privacy-enforced memory that belongs to you, not us
- **Multi-LLM support** — works with Anthropic (Claude), OpenAI (GPT), Google (Gemini), and local models

**Kestrel Falconer** is the enterprise layer on top. It orchestrates multiple agents as a team — like a falconer managing a flock of birds:

| Agent Role | Codename | What It Does |
|---|---|---|
| Project Manager | **Claws** | Triage, planning, backlog management |
| Developer | **Talon** | Code implementation, PR creation, test writing |
| Quality Assurance | **Eye** | Test validation, code review, regression checks |
| Operations | **Flight** | Deployment, monitoring, infrastructure |
| Adversary/Auditor | **Red-Action** | Stress-testing, security audits, finding gaps |

The **Falconer Dashboard** is where you see this in action — flock status, task flow, mesh communications between agents, and governance (constitutional hashes, DIDs, privacy modes).

---

## Your Access

### Falconer Dashboard (start here)

| Resource | URL |
|---|---|
| **Sovereign Console** | https://kestrel-dev-7jpbsywhdq-uc.a.run.app/?key=***REDACTED-DEV-API-KEY*** |
| **Falconer Dashboard** | https://kestrel-dev-7jpbsywhdq-uc.a.run.app/static/falconer-dashboard.html?key=***REDACTED-DEV-API-KEY*** |
| **Portfolio Dashboard** | https://kestrel-dev-7jpbsywhdq-uc.a.run.app/static/dashboard.html?key=***REDACTED-DEV-API-KEY*** |
| **Vision Page** | https://kestrel-dev-7jpbsywhdq-uc.a.run.app/static/kestrel-falconer-v2.html |

### Authentication

The links above already include your API key — just click and go.

If you're ever prompted to log in, use the **API Key** tab and paste:

```
***REDACTED-DEV-API-KEY***
```

This key gives you full access to the Sovereign Console, Falconer Dashboard, and all agent commands.

### GitHub Repository

You're a **triage collaborator** on [kestrel-sovereign](https://github.com/KestrelSovereignAI/kestrel-sovereign).

Triage means you can: read all code, browse issues, comment, label, and manage issues. You cannot push code — that's intentional (more on why below).

### Your Agent Identity

We've created a sovereign agent identity for you:

- **Agent Name:** George
- **DID:** `did:pkh:eip155:1:0x35094f189A2E2f51079a49A2D49f84B9ad5851D5`

This is a real cryptographic identity — your agent has its own signing keys and constitutional embedding. It's not a demo account.

### What's Running on Staging (verified April 5)

The staging environment has been end-to-end tested. Here's what's live:

| Feature | Status | What You'll See |
|---|---|---|
| **Falconer Dashboard** | Live | KPI grid (agents online, tasks, mesh messages, context health), flock status cards, task & job history, governance tab |
| **Sovereign Console** | Live | Multi-agent sidebar (Kestrel + kestrel-demo, both online), chat interface, identity tab with DID, constitution panel |
| **Morning Signal** (`!morning`) | Working | Generates daily strategic briefing — milestones, suggested work items, blocker detection |
| **Strategic Dispatch** (`!dispatch suggest`) | Working | AI-native project management — Claws scores all issues by impact and recommends the top priority |
| **Talon Execution** (`!talon status`) | Limited | Talon feature is loaded but the 2-agent staging setup doesn't run a dedicated Talon worker — status output may be sparse. Full Talon pipeline is a local dev feature for now |
| **Constitutional Governance** | Working | SHA-256 hash verification, 489 tool-level permissions (Allow/Ask/Deny), full audit trail |
| **Privacy Modes** | Working | NORMAL mode active (EPHEMERAL, ISOLATED, ANONYMOUS, NORMAL, PUBLIC available) |
| **LLM Provider** | Anthropic | Claude Sonnet 4.6 — switchable via dropdown in console |

---

## What to Explore

### Start with the Falconer Dashboard

1. Click the **Falconer Dashboard** link above (key is included)
2. Explore the tabs:
   - **Flock Status** — See which agents are online, their skills, and health
   - **Tasks & Jobs** — Active tasks, scheduled jobs, task history
   - **Mesh Messages** — Inter-agent communication (how the birds coordinate)
   - **Governance** — Constitutional identity, DID, privacy modes, active features

### Then the Vision Page

The **Vision Page** tells the Falconer story in a way that's more narrative/product-oriented. It's useful context for understanding where we're going, not just where we are.

### Then the Sovereign Console

The **Sovereign Console** at the root URL is the full agent interface — chat with agents, run commands, see memory, configure privacy modes. Try these commands in the chat:

- `!morning` — Generate a Morning Signal briefing (like a daily standup, but AI-generated from live GitHub data)
- `!dispatch suggest` — Ask Claws to pick the highest-priority issue across all repos
- `!talon status` — Check Talon's autonomous developer workload
- `!help` — See all available commands

If the agent greets you with a "getting to know you" introduction, just type `skip` or `let's go` to jump straight to command mode. That's the agent's onboarding flow for new users — you can always revisit it later.

The agents sidebar on the left shows which agents are online (green dot = healthy). Click an agent to interact with it.

---

## Why Dashboard First, Repository Later

We're deliberately starting you with the **product experience** before the **code**.

Here's why: This system is built by AI agents (Talon writes code, Eye tests it, Flight deploys it). The code style will look different from hand-written code. That's expected — it's the output of an AI development pipeline with automated testing and constitutional verification.

**What matters is: does the system work? Does the architecture hold up? Does the product make sense?**

Once you've formed your own impressions from using the dashboard and interacting with the system, we'll walk you through the codebase. At that point, you'll be evaluating the architecture with the context of having seen it operate — not judging syntax in a vacuum.

The repo invite is already sent. Feel free to browse whenever you're ready, but we recommend spending a session with the dashboard first.

### What you'll see in the repo

When you do browse the code, here's the architectural lay of the land:

- **`kestrel_sovereign/`** — The core runtime: agent orchestration, LLM routing, context assembly, API server. This is the thin runtime layer.
- **`kestrel_sdk/`** — The SDK: all interfaces, protocols, and shared types. Features and providers code against SDK contracts, not concrete implementations.
- **`kestrel_sovereign/features/`** — 42 core feature modules discovered at startup (constitution, privacy, memory, identity, scheduling, strategic memory, mesh protocol, etc.). Each is a self-contained `Feature` subclass.
- **Feature packages** (e.g. `kestrel_cloud_vastai/`, `kestrel_feature_code/`) — Extracted standalone packages registered via Python entry points. They extend the core without modifying it.

The pattern is: **SDK defines contracts → core provides runtime → features are plugins**. This is intentional — it means enterprise deployments can include only the features they need, and new capabilities slot in without touching core.

---

## What We Need From You

### The feedback we value most:

**1. Product Architecture**
- Does the multi-agent orchestration model (Falconer flock) make sense as an enterprise pattern?
- Is the DID/constitutional identity approach sound for enterprise trust?
- How does the governance model (constitutional hashes, privacy modes) feel from an architect's perspective?
- Are there architectural blind spots we're missing?

**2. AI Ethics & Trust**
- Does the constitutional protection model feel genuine or performative?
- Is the privacy system (EPHEMERAL → PUBLIC modes) meaningful for real users?
- What concerns would an enterprise CISO have about this approach?
- Is the sovereignty concept (user owns their data and agent) compelling or confusing?

**3. Marketability**
- Who would buy this? What's the entry point?
- Does "AI agents managed like a flock of trained birds" resonate as a metaphor?
- What's missing from the dashboard to make it demo-ready for an enterprise buyer?
- If you were pitching this to a CTO, what would be the 30-second hook?

### How to give feedback:

- **GitHub Issues** — Once you accept the repo invite, file issues directly. Use labels like `feedback`, `ux`, `architecture`, `question`. This is our preferred channel — it gets tracked and addressed.
- **Direct to Gabi** — Email or message for anything sensitive or high-level.
- **Dashboard comments** — If we add a feedback widget, use it. (We might build one based on your experience.)

---

## What to Expect (and Not Expect)

### This is an MVP — deliberately.

We have **42 maintained feature modules** in the core, plus standalone feature packages, spanning multiple product scenarios:
- Elder care companions (RemoteCares/Caprock partnership)
- Enterprise agent orchestration (Falconer/Castle)
- Individual sovereign companions
- Developer tooling (Talon, code analysis, PR automation)
- Cloud compute, voice, storage, and wallet integrations

**We are NOT trying to ship all of this at once.**

Our philosophy is **extreme small iterations**:
- Build the smallest thing that works
- Ship it
- Get real feedback
- Improve or discard based on evidence
- Repeat

What you're seeing is the MVP of Falconer — the minimum needed to validate the multi-agent orchestration concept. Some features will be rough. Some will be prototypes. Some things you'd expect in an enterprise tool won't exist yet.

**That's the point.** We'd rather ship something real and learn from it than spend months polishing something nobody wants.

### What "production-ready" means to us:

Every feature goes through a pipeline:
1. **Implementation** (Talon writes it)
2. **Testing** (Eye validates it)
3. **Red-Action audit** (adversarial testing — "does this actually work or is it lying?")
4. **Deployment** (Flight ships it)

Some features have been through this full loop. Others are still in progress. The dashboard will show you which agents are active and what they can do — that's the honest state of the system.

---

## Quick Reference

| Item | Value |
|---|---|
| Sovereign Console | https://kestrel-dev-7jpbsywhdq-uc.a.run.app/?key=***REDACTED-DEV-API-KEY*** |
| Falconer Dashboard | https://kestrel-dev-7jpbsywhdq-uc.a.run.app/static/falconer-dashboard.html?key=***REDACTED-DEV-API-KEY*** |
| Portfolio Dashboard | https://kestrel-dev-7jpbsywhdq-uc.a.run.app/static/dashboard.html?key=***REDACTED-DEV-API-KEY*** |
| Vision Page | https://kestrel-dev-7jpbsywhdq-uc.a.run.app/static/kestrel-falconer-v2.html |
| Your Email | george.guimaraes@outlook.com |
| Auth Method | API Key (embedded in URLs above) |
| Your DID | did:pkh:eip155:1:0x35094f189A2E2f51079a49A2D49f84B9ad5851D5 |
| GitHub Repo | https://github.com/KestrelSovereignAI/kestrel-sovereign |
| GitHub Role | Triage (read + issue management) |
| File Issues | https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/new |
| Contact | Gabi (gabriela-aquino), Jason (@UncleSaurus) |

---

## One Last Thing

You're not just testing software. You're helping us validate whether **sovereign AI agents with constitutional protections** is a real product category. Your honest assessment — even if it's "this doesn't work" or "nobody would buy this" — is exactly what we need.

Thank you for your time, George. We're excited to hear what you think.

— Gabi & the Kestrel team
