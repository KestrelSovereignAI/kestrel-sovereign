---
type: Generated Reference
title: What Can Kestrel Do For You?
description: Audience-specific user view generated from the canonical Kestrel feature inventory.
resource: /docs/generated/FEATURES_user.md
tags:
- features
- generated-docs
- user
timestamp: 2026-04-13T00:00:00Z
status: generated
generated: true
canonical: false
source: /KESTREL_FEATURES.md
audience: user
generator: scripts/generate_feature_docs.py
model: anthropic/claude-sonnet-4-6
regenerate: uv run python scripts/generate_feature_docs.py --audience user
---

<!-- BEGIN PROTECTED PACKAGE BOUNDARY CONTRACT -->
## Package Ownership and Installation Boundaries

These ownership statements are normative and must remain intact in every
audience-specific derivative:

<!-- NON_BUNDLED_SURFACE_ALIASES: voice; mcp; github integration|github app; wallet; observability; council; visual identity; legal feature; code editing|code edit; parametric self; whatsapp; runpod; vast.ai|vastai; gcp compute; elevenlabs; deepgram; kestrel-talon -->

- **Bundled Feature lifecycle modules:** Feature subclasses discovered from
  `kestrel_sovereign/features/` ship in the `kestrel-sovereign` distribution.
  They need no separate package install. The generated inventory below is the
  exact in-tree discovery snapshot.
- **Bundled non-Feature components:** Some base-install runtime services, such
  as `PrivacyAgent`, are shipped by `kestrel-sovereign` but are not Feature
  lifecycle classes. The registry labels these `bundled-component` rather than
  putting them in its `features` field.
- **Not bundled — extracted Feature packages:** Voice, MCP, GitHub integration,
  wallet, observability, reflection, council, visual identity, legal, code
  editing, parametric self, and WhatsApp transport are separate install targets.
  They register Feature subclasses through the `kestrel_sovereign.features`
  entry-point group.
- **Not bundled — provider packages:** ElevenLabs, Deepgram, OpenAI voice, xAI
  voice/realtime, RunPod, Vast.ai, GCP Compute, and external storage backends
  implement provider contracts. They use provider-specific entry-point groups;
  installing one does not make that provider a Feature lifecycle class.
- **Not bundled — standalone tool:** `kestrel-talon` is an independently
  installed command-line issue processor. The in-tree `TalonCoordinatorFeature`
  is only its bundled Kestrel control surface; the coordinator and the
  standalone executable are separately named registry rows.

The runtime catalog at `kestrel_sovereign/data/feature_registry.toml` encodes
these distinctions in `boundary`. Its `package` field is always the owning
distribution/install target for that row. The compatibility field `core` is
`true` only for `bundled` and `bundled-component` rows. `features` contains
Feature lifecycle class names only; provider implementations use
`provider_classes` plus `entry_point_groups`, and standalone tools use
`command`. Catalog status `available` means “known but not detected in this
environment,” not a claim that an external distribution is publicly reachable.
<!-- END PROTECTED PACKAGE BOUNDARY CONTRACT -->

<!-- BEGIN PROTECTED CONTEXT HONESTY CONTRACT -->
## Context Runtime and Diagnostic Boundary

These context statements are normative and must remain intact in every
audience-specific derivative:

- A production turn preloads at most the latest **50** eligible entries from
  the active session before retrieval, budgeting, and lumpy history selection.
- Production and `GET /api/agent/context-status?full=true` use the same typed
  `ContextManager` build plan over that latest-50 input. The dry-run executes
  production relevance gates, elastic finalization, lumpy anchoring,
  microcompaction, wrapper accounting, and prune decisions without committing
  access records or salvage writes.
- The cheap `full=false` status deliberately omits memory/RAG acquisition and
  reports those sections as `unknown`/`skipped`, never as measured zero.
  Provider-native framing and stateful provider-thread occupancy remain
  separate from the Kestrel context plan.
- Default lumpy pruning omits older history from the provider window while
  retaining the source rows; it does not create an automatic durable summary.
  Automatic durable salvage is disabled by default. Its feature-flagged path is
  conditional on a pruned span mapping to id-bearing persistent history, so it
  is not a fail-closed guarantee for id-less or `ISOLATED` in-memory history.
- `openai:plan` occupancy compaction is best-effort. Kestrel resets the Codex
  thread only after durable compaction reports success; a skipped or failed
  attempt lets the turn continue with the existing thread.
- The complete all-route Context C lifecycle remains aspirational, not shipped
  behavior. The canonical current-state contract is
  `docs/architecture/CONTEXT_SYSTEM_DESIGN.md`; the separate
  `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md` page is a design record.
<!-- END PROTECTED CONTEXT HONESTY CONTRACT -->

# Kestrel — What It Can Do For You

> A friendly, scannable overview of everything Kestrel offers — no technical jargon required.

---

## Table of Contents

- [Your Agent's Identity](#your-agents-identity)
- [Talking to Your Agent](#talking-to-your-agent)
- [Works With Your Favourite AI Models](#works-with-your-favourite-ai-models)
- [Privacy — You're in Control](#privacy--youre-in-control)
- [Memory and Conversations](#memory-and-conversations)
- [What Your Agent Can Do](#what-your-agent-can-do)
- [Your Agent's Voice](#your-agents-voice)
- [Security and Permissions](#security-and-permissions)
- [Saving and Organizing Information](#saving-and-organizing-information)
- [Wallet and Keys](#wallet-and-keys)
- [Connecting to Other Agents and Services](#connecting-to-other-agents-and-services)
- [Extending Your Agent with Features](#extending-your-agent-with-features)

---

## Your Agent's Identity

Your Kestrel agent isn't just a chatbot — it has its own persistent, secure identity that follows it across sessions and time.

- **Your agent has its own secure identity.** Each agent is born with a unique cryptographic identity, so it can prove who it is and can't be impersonated.
- **Continuity you can trust.** Your agent's identity is verified continuously, so you always know you're talking to the same agent — even after restarts or updates.
- **A constitution that governs behaviour.** Every agent runs under a set of core principles called its constitution. This shapes how it responds and what it will or won't do — giving you a principled, consistent companion.
- **Full lifecycle support.** Your agent can be started fresh, graduated to a more capable state, or gracefully retired — all while maintaining a trustworthy record of who it has been.

---

## Talking to Your Agent

Communicating with Kestrel is flexible and fast.

- **Ask questions and give instructions naturally.** Your agent understands plain language — just type what you need.
- **Responses can stream in real time.** Instead of waiting for a full response, you see answers as they're generated, just like a live conversation.
- **Stop a response mid-way.** Changed your mind? You can halt a response in progress at any time.
- **Your agent knows its own status.** You can check in on what your agent is working on, what tasks it has queued, and how its context is being used — all without needing to understand what's happening under the hood.
- **Live notifications.** Get real-time updates from your agent as things happen, including a live event stream you can keep open in the background.

---

## Works With Your Favourite AI Models

Kestrel isn't locked to a single AI provider. You choose the model, and Kestrel handles the rest.

- **Broad model support.** Kestrel works with ChatGPT (OpenAI), Claude (Anthropic, including Claude Max), Gemini (Google, including Vertex AI), Ollama for local models, OpenRouter, and more.
- **Switch models any time.** You can see which model is active and change it whenever you like.
- **Local models for maximum privacy.** If you want your conversations to stay entirely on your own device, choose a local model like Ollama — nothing leaves your machine.
- **Automatic retries.** If a model hiccups, Kestrel quietly retries so your conversation isn't interrupted.
- **Usage tracking.** Kestrel keeps an eye on how much you're using each provider, so you can stay informed about your consumption.

---

## Privacy — You're in Control

Privacy in Kestrel isn't an afterthought — it's built in from the ground up. You choose exactly how private each conversation is by picking a preset that matches what you need.

### Privacy Presets

| Preset | What gets stored | Where AI runs | Can be shared | What it means for you |
|---|---|---|---|---|
| `ephemeral` | Nothing at all | On your device only | No | Maximum privacy — nothing is ever saved, and your AI never touches the cloud |
| `isolated` | Temporarily, for this session only | On your device only | No | Good for sensitive work — storage clears when the session ends, AI stays local |
| `anonymous` | Saved, but with personal details removed | Cloud AI allowed | No | You get cloud AI power, but your personal information is scrubbed before anything is stored |
| `normal` | Fully saved | Cloud AI allowed | No | Standard everyday use — your history is kept so your agent can learn and remember |
| `public` | Fully saved | Cloud AI allowed | Yes | For work you're happy to share or export — full storage and sharable |

You can check or change your privacy mode at any time. Your choice is respected immediately.

---

## Memory and Conversations

Kestrel remembers — so you don't have to repeat yourself.

- **Conversation history.** Your past conversations are saved and browsable. You can pull up any previous session, read a transcript, or start something new.
- **Persistent memory.** Your agent builds up memories over time — facts about you, your preferences, ongoing projects — so future conversations feel continuous and informed.
- **Strategic memory.** Beyond simple recall, your agent can form longer-term, higher-level memories that help it understand your goals and context more deeply.
- **Memory agency.** Your agent can proactively manage its own memory — deciding what's worth remembering and what can be let go.
- **Browse and delete memories.** You're always in control. You can view your agent's memories, look up specific ones, and delete anything you don't want kept.
- **Identity chain.** You can inspect the record of your agent's identity history — a transparent log of continuity.

---

## What Your Agent Can Do

Kestrel combines a capable base install with optional Feature and provider
packages. Here is what your agent can do when the named capability is installed.

### Everyday Productivity

- **Web search.** Your agent can search the web and bring back answers — no copy-pasting required.
- **Task management.** Your agent tracks tasks, so you can ask it what's on its plate and follow up on work in progress.
- **Scheduling.** Your agent can manage scheduled jobs — reminders, recurring tasks, and time-based actions.
- **Save anything.** Clip and save items — notes, links, structured data — and retrieve them later by tag, type, or search.

### Code and Technical Work

- **Code editing (optional Feature package).** Your agent can help you write and edit code directly.
- **GitHub integration (optional Feature package).** Connect to GitHub to work with repositories, issues, and pull requests from your conversations.
- **Compute on demand.** Your agent can spin up cloud compute when it needs more power, with support for several cloud GPU providers.

### Reflection and Wellbeing

- **Reflection (optional Feature package).** Your agent can look back on recent activity and surface insights — like a thoughtful review of what's been happening.
- **State of mind.** Your agent has an awareness of its own condition and can report on how it's doing.
- **Wellness.** There are built-in mechanisms to keep your agent healthy and well-functioning over time.
- **Heartbeat.** Your agent regularly checks in on itself — you can see its heartbeat status and trigger a manual check at any time.

### Identity and Appearance

- **Visual identity (optional Feature package).** Your agent can have an avatar — you can upload one or have it generated automatically.
- **Constitution.** You can read your agent's governing constitution at any time to understand its principles.

### Peer-to-Peer and Mesh

- **Peer connections.** Your agent can connect with other Kestrel agents, forming a mesh network for more powerful, collaborative tasks.
- **Inbox.** Messages from peer agents land in your agent's mesh inbox, keeping cross-agent communication organised.
- **Spawn child agents.** Your agent can create sub-agents to delegate work, and you can see a list of any active children.

### Channels and Delivery

- **Channels.** Your agent can communicate through multiple channels, reaching you or others wherever makes sense.
- **Delivery.** Your agent can send outputs to their intended destinations — files, notifications, or other endpoints.
- **Webhooks.** Your agent can listen for and respond to external events through webhooks.
- **Bridge.** Connect Kestrel to external services and platforms through its bridge capability.

### Cloud and Deployment

- **GCP Compute, RunPod, Vast.ai (optional provider packages).** Your agent can work with these cloud compute backends after the corresponding provider is installed.
- **Talon coordination.** The in-tree coordinator can dispatch complex issue-processing work after the standalone `kestrel-talon` tool is installed.
- **IPFS status.** Your agent is aware of its decentralised storage status and can report on it.

---

## Your Agent's Voice

The optional extracted voice Feature package lets Kestrel speak and listen.

- **Text to speech.** Your agent can read its responses aloud, with support for streaming audio for a natural, real-time feel.
- **Speech to text.** Speak to your agent and it will transcribe your words into text.
- **Voice chat.** Have a live, two-way voice conversation with your agent over a persistent connection.
- **Voice configuration.** Choose from available voices and tweak your audio preferences.

---

## Security and Permissions

You decide what your agent is allowed to do — nothing happens behind your back.

- **Permission tree.** See a clear picture of all the permissions your agent has, organised in a way that's easy to understand.
- **Approval flow.** Sensitive actions can be queued for your review before they're carried out. You approve or decline — your agent waits.
- **Pending requests.** Check at any time what's waiting for your sign-off.
- **Audit log.** Every security-relevant event is logged, so you have a full record of what's been authorised.
- **Cancel actions.** Change your mind? Cancel individual pending requests or clear everything at once.
- **Session reset.** Start fresh with a clean security slate whenever you need to.
- **Consent.** Your agent is built with consent as a first-class concern — it won't overstep without asking.

---

## Saving and Organizing Information

Your agent is also a personal knowledge base.

- **Save items in any format.** Notes, links, structured data — save whatever you need and retrieve it later.
- **Tags and schemas.** Organise your saved items with tags or structured schemas, so finding things later is effortless.
- **Search.** Full search across your saved items — find what you need fast.
- **Pin items.** Keep important items at the top so they're always easy to find.
- **Stats at a glance.** See how much you've saved and what's in your collection.

---

## Wallet and Keys

Kestrel keeps your credentials and financial tools organised and secure.

- **Wallet (optional Feature package).** Add on-chain and payment tooling by installing the wallet Feature package.
- **API key management.** Add, update, or remove keys for each AI provider from one place. You can also check usage per provider.

---

## Connecting to Other Agents and Services

Kestrel doesn't live in isolation — it's designed to work within a wider ecosystem.

- **Sovereignty and data portability.** Your agent's data is yours. You can export it, import it, and browse your sovereignty files — including previews. Your data never gets locked in.
- **Storage stats.** See how much your agent is storing and where.
- **Council (optional Feature package).** Your agent can participate in council-style deliberation — bringing multiple perspectives together for better decisions.
- **Observability (optional Feature package).** See a live feed of events and a summary of what your agent has been doing — full transparency into its activity.

---

## Extending Your Agent with Features

Kestrel is modular. Feature lifecycle packages can be enabled, disabled, and
configured; provider packages and standalone tools keep their own distinct
runtime contracts.

- **Browse available features.** See everything that's available, whether it's already installed or ready to add.
- **Enable or disable on the fly.** Turn features on or off without restarting anything.
- **Configure each feature.** Every feature has its own settings you can adjust to suit your needs.
- **Install new features.** The feature system is open to expansion — new capabilities can be added without touching the core.
- **Skills.** Features can expose skills — discrete actions your agent can perform. You can browse all available skills and understand exactly what each one does.

---

*Kestrel is built around the idea that your agent should work for you — on your terms, with your privacy, under your control.*
