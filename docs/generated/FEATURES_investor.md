<!-- AUTO-GENERATED from KESTREL_FEATURES.md — do not edit manually -->
<!-- Audience: investor | Generated: 2026-04-13 | Model: anthropic/claude-sonnet-4-6 -->
<!-- Regenerate: uv run python scripts/generate_feature_docs.py --audience investor -->

# Kestrel Sovereign: Platform Feature Overview

> **Prepared for:** Investors & Business Stakeholders
> **Document class:** Derived audience overview — based on the canonical Kestrel Sovereign feature inventory

---

## Executive Summary

Kestrel Sovereign is a full-stack, sovereign AI agent platform that combines a governance framework with no industry equivalent, vendor-independent LLM routing, and portable decentralized identity — giving users and enterprises genuine ownership of their AI rather than dependency on any single provider. The platform ships with enterprise-grade privacy controls from day one, a deeply extensible plugin architecture, and a documented surface spanning authentication, agent orchestration, voice, memory, security, observability, and more. With 42 independently discoverable capability modules in its current audited snapshot and adapters for every major LLM provider, Kestrel Sovereign is positioned as the infrastructure layer for AI deployments where trust, portability, and control are non-negotiable requirements.

---

## Table of Contents

1. [Platform Architecture](#1-platform-architecture)
2. [AI Capabilities](#2-ai-capabilities)
3. [Data Sovereignty](#3-data-sovereignty)
4. [Security & Privacy](#4-security--privacy)
5. [Deployment Flexibility](#5-deployment-flexibility)
6. [Extensibility](#6-extensibility)
7. [Public API Surface](#7-public-api-surface)
8. [Authentication Model](#8-authentication-model)

---

## 1. Platform Architecture

### Sovereign-First Design

Kestrel Sovereign is architected around a principle that has no direct equivalent in the current market: every agent instance carries its own governance framework, cryptographic identity, and data sovereignty lifecycle from the moment it is created. This is not a compliance layer bolted on after the fact — it is the foundation on which every other capability is built.

The platform manages the full agent lifecycle, from initial provisioning ("inception") through active operation to graceful retirement, with cryptographic continuity preserved at each stage. This means an enterprise can demonstrate provenance and behavioral governance for any agent interaction across its entire operational lifespan.

### Agent Runtime

The agent runtime handles orchestration, context assembly, token budget management, and streaming response delivery as first-class concerns. Context management is designed to operate efficiently within the constraints of any connected LLM, ensuring that long-running conversations and complex multi-step tasks remain coherent regardless of the underlying model in use.

Real-time streaming is natively supported, enabling responsive user interfaces and downstream integrations without polling or latency penalties.

### Multi-Agent Topology

The platform supports agent mesh networking — agents can communicate peer-to-peer, maintain shared inboxes, and spawn child agents for delegated workloads. This enables enterprise deployments to compose sophisticated multi-agent workflows without relying on external orchestration middleware.

---

## 2. AI Capabilities

### Vendor-Independent, No Platform Lock-In

Kestrel Sovereign's multi-LLM layer is a core competitive differentiator. Rather than binding users or enterprises to a single model provider, the platform maintains adapters for a broad range of providers — including **OpenAI, Anthropic, Claude Max, Gemini, Vertex AI, Ollama, OpenRouter,** and a mock adapter for testing — all behind a unified routing and mandate layer.

Operators can set routing mandates that enforce which providers or models are permitted for a given agent or privacy context, enabling compliance with data residency requirements, cost policies, or capability preferences without application-level changes. A model catalog and metadata layer tracks available models, and the platform handles retries and usage tracking transparently across all providers.

Critically, this architecture means customers are never locked into a vendor relationship at the infrastructure level. As the LLM market evolves — new providers emerge, pricing shifts, models are deprecated — Kestrel Sovereign deployments adapt without re-architecture.

### Constitutional AI Governance Framework

Every Kestrel Sovereign agent operates under a governance framework with no industry equivalent: a persistent, auditable constitution that defines behavioral boundaries, consent requirements, and operational mandates. This is not a system prompt or a soft guardrail — it is a structured governance layer enforced at the agent runtime level and surfaced through the platform's API.

This capability directly addresses the enterprise risk question that most AI deployments leave unanswered: *how do you prove that an AI agent behaved within sanctioned boundaries?* Kestrel's governance framework provides the evidentiary basis for that proof.

### Memory and Reflection

The platform includes layered memory systems — session memory, persistent memory, strategic memory, and an autonomous memory agency — enabling agents to maintain meaningful continuity across interactions without requiring the client application to manage state. A reflection subsystem allows agents to reason about their own prior behavior, enabling self-correction and quality improvement over time.

### Voice

Native voice capabilities are integrated at the platform level, including text-to-speech, streaming text-to-speech, speech-to-text, and a full-duplex WebSocket voice chat channel. This is not a third-party integration shim — voice is a documented, maintained surface of the platform.

### Web Search and Tool Use

Agents have native access to web search, code execution, file operations, and a range of integration tools. These capabilities are governed by the same permission and consent framework that applies to all other agent actions, ensuring that tool use is auditable and controllable.

---

## 3. Data Sovereignty

### Portable Identity — Users Own Their AI

Kestrel Sovereign uses Decentralized Identifiers (DIDs) as the foundation of agent identity. Each agent receives a cryptographically verifiable identity package at inception that travels with it across its lifecycle. Identity is not stored in a proprietary silo — it is portable.

This has significant market implications:

- **Users can move between providers** without losing their agent's identity, history, or behavioral configuration.
- **Enterprises retain ownership** of their agents even if they migrate infrastructure or change deployment partners.
- **Continuity is cryptographically verifiable** — the identity chain can be inspected at any point to confirm that an agent's history has not been tampered with.

An identity chain API allows operators and auditors to traverse the full provenance of any agent instance.

### Data Export and Import

The sovereignty API provides full data export and import capabilities, including file access and preview. Customers are never in a position where their data is stranded inside the platform. This is a direct response to the vendor lock-in risk that enterprise buyers routinely cite when evaluating AI infrastructure.

### Storage Architecture

Storage is designed to support the full range of privacy presets (see [Section 4](#4-security--privacy)), from fully ephemeral sessions to persistent, shareable profiles. The storage layer is asynchronous and extensible, and its behavior is governed by the active privacy preset rather than hardcoded assumptions.

---

## 4. Security & Privacy

### Enterprise-Grade Privacy Controls From Day One

Privacy is not a feature that can be added to Kestrel Sovereign — it is structurally embedded in how every agent interaction is processed and stored. The platform defines five canonical privacy presets that govern storage behavior, LLM routing (local versus cloud), and shareability simultaneously:

| Preset | Storage | LLM Location | Shareable | Use Case |
|---|---|---|---|---|
| `ephemeral` | None | Local only | No | Maximum confidentiality; nothing persisted |
| `isolated` | Temporary session | Local only | No | Session-scoped work on sensitive material |
| `anonymous` | Stored with PII removed | Cloud allowed | No | Analytics-safe cloud usage |
| `normal` | Full persistent | Cloud allowed | No | Standard enterprise deployment |
| `public` | Full persistent | Cloud allowed | Yes | Shareable, exportable interactions |

These presets are enforced at the platform level and are surfaced through the agent API, allowing client applications and administrators to set and inspect the active privacy mode at runtime. This means enterprises can enforce data handling policies programmatically — not just through policy documents.

### Permission and Consent Framework

Every agent capability operates within a permission tree. The security API exposes endpoints for inspecting the permission tree, requesting capability grants, approving or cancelling pending permission requests, and auditing the full history of permission events. A consent module ensures that users can express and revoke consent in a structured, auditable way.

This framework is the operational complement to the constitutional governance layer — together they provide both the *policy* (what the agent is permitted to do) and the *audit record* (what the agent actually did).

### Audit Infrastructure

The platform includes an audit anchor capability and a response audit system that provide tamper-evident records of agent behavior. Observability endpoints expose structured event streams and summaries for integration with enterprise SIEM and monitoring systems. Security events are queryable through the API, and session state can be reset in a controlled manner.

### Authentication Architecture

The platform implements a layered authentication model appropriate for enterprise deployment:

- **Public** endpoints limited to health checks and infrastructure probes
- **OAuth flows** for human user authentication
- **API key or session** authentication for programmatic and application access
- **Server-sent events (SSE)** paths that support API key query parameters for streaming integrations
- **Bootstrap-conditional** local key provisioning for initial setup

This architecture supports both human-facing applications and machine-to-machine integrations within a single, coherent security model.

---

## 5. Deployment Flexibility

### Local and Cloud LLM Routing

The LLM mandate system allows operators to route inference to local models (via Ollama) or cloud providers depending on the active privacy preset, cost policy, or capability requirement. This means a single deployment can serve both air-gapped, high-sensitivity workloads (using local inference) and standard workloads (using cloud providers) without maintaining separate infrastructure stacks.

### Cloud Compute Integrations

The platform includes native integrations for GPU compute provisioning across multiple cloud environments, enabling deployments that require scalable inference infrastructure to manage that resource allocation through the same platform that manages agent behavior and governance.

### Standards-Compliant API Surface

The platform exposes a standards-aligned chat completions interface (compatible with the OpenAI API surface at `/v1/models` and `/v1/chat/completions`), enabling drop-in compatibility with tooling, applications, and workflows already built against that de facto standard.

### Webhook and Integration Infrastructure

Native webhook support, a Stripe crypto payment webhook endpoint, and a Rasa protocol shim are included in the maintained surface, enabling integration with payment systems, existing conversational AI infrastructure, and event-driven architectures.

---

## 6. Extensibility

### Extensible Platform Architecture

Kestrel Sovereign's capability system is built around a discoverable feature module architecture. The platform's current audited snapshot includes 42 core capability modules, spanning governance, identity, memory, voice, security, compute, scheduling, web search, wallet management, observability, and more.

Beyond the core inventory, the platform supports installable feature packages — third-party or proprietary capability modules that are registered at runtime and extend the platform without modifying its core. This is the extensibility model that enterprise platform buyers require: a stable, governed core with a documented, safe extension surface.

The features API allows administrators to discover, install, enable, disable, configure, and remove capability modules at runtime, without service interruption. Feature-level skill schemas are introspectable via the API, enabling client applications and integration tooling to adapt dynamically to the agent's current capability set.

### Current Core Capability Modules

The 42 modules in the current audited snapshot span the following capability domains:

| Domain | Capabilities |
|---|---|
| **Governance & Identity** | Constitutional governance, consent management, audit anchoring, identity, keys, visual identity |
| **Memory & Context** | Session memory, persistent memory, strategic memory, memory agency, context management, reflection |
| **Communication** | Voice, channels, delivery, webhooks, peers, bridge, mesh |
| **Compute & Deployment** | General compute, GCP compute, RunPod, Vast.ai, deployment management |
| **Productivity & Tools** | Tasks, scheduler, code editing, web search, file saving, GitHub, GitHub Apps |
| **Operations** | Observability, security, heartbeat, wellness, state of mind |
| **Platform** | Bootstrap, sovereignty, spawn, model/LLM management, wallet, response audit, Talon coordination |

Third-party and enterprise-proprietary modules can be added to this inventory without modifying the core platform.

---

## 7. Public API Surface

The platform exposes documented route families across the following areas. Route families are the maintained unit of surface — individual endpoints within each family are stable and versioned.

| Route Family | Scope |
|---|---|
| **Agent** | Invocation, streaming, stop, privacy mode, notifications (polling and SSE), context status, reflection, tasks, heartbeat, mesh |
| **Authentication & OAuth** | Login, callback, logout, identity (`/auth/me`), token exchange, verification |
| **Conversations** | Session listing, conversation retrieval, new conversation creation, message deletion, transcript export |
| **Memories** | Memory retrieval, individual memory access, identity chain traversal, memory deletion |
| **Sovereignty** | Storage statistics, data export and import, sovereign file access and preview |
| **Models & Identity** | Agent management, identity configuration, avatar management, constitution access, IPFS status, wallet, API key management, model selection, OpenAI-compatible chat completions |
| **Security** | Permission tree, permission requests, pending approvals, audit log, session management |
| **Features** | Feature discovery, installation, enable/disable, configuration, skill schemas |
| **Voice** | Voice configuration, text-to-speech (batch and streaming), speech-to-text, WebSocket voice chat |
| **Saved Items** | Structured item storage, tagging, schema management, search, pinning |
| **Observability** | Event streams, summaries, Prometheus metrics |
| **Infrastructure** | Health (simple and detailed), database introspection, file content retrieval, commands |
| **Integrations** | Stripe webhook, Rasa protocol shim, GitHub proxy |

---

## 8. Authentication Model

The platform's authentication surface is designed to support the full spectrum of enterprise deployment patterns without sacrificing security at any tier.

| Class | Routes | Notes |
|---|---|---|
| **Public** | `/health`, `/health/detailed`, webhook receivers | Infrastructure probes and inbound integrations |
| **Public-Localhost** | `/api/auth/key` (when bootstrap enabled) | Initial provisioning only |
| **OAuth Public Entrypoints** | `/auth/login`, `/auth/callback`, `/auth/logout` | Human user authentication flows |
| **API Key or Session** | Most `/agent/*` and `/api/*` routes | Standard programmatic and application access |
| **API Key or Session + SSE Query** | `/agent/stream`, `/agent/notifications/sse` | Streaming paths that accept key via query parameter |
| **OAuth Session Semantic** | `/auth/me` | Returns authenticated data only for real sessions |
| **Browser-Conditional** | `/` | Serves UI locally; redirects to OAuth when required |

This model ensures that public-facing health and integration endpoints are always reachable without credentials, OAuth flows are cleanly separated from programmatic access, and SSE streaming paths — which have inherent browser constraints — are handled correctly.

---

## Summary of Competitive Differentiation

| Dimension | Kestrel Sovereign Position |
|---|---|
| **Governance** | Constitutional governance framework with no industry equivalent |
| **Vendor Independence** | Multi-LLM routing with adapters for all major providers; no platform lock-in |
| **Identity Portability** | DID-based portable identity; users and enterprises own their AI across providers |
| **Privacy** | Five structured privacy presets enforced at the platform level from day one |
| **Extensibility** | 42-module core with runtime-installable third-party capability packages |
| **Standards Compliance** | OpenAI-compatible API surface for drop-in integration |
| **Audit & Trust** | Tamper-evident audit infrastructure aligned with enterprise governance requirements |
| **Deployment Range** | Local inference to multi-cloud GPU compute; single platform, full spectrum |

---

*This document is derived from the Kestrel Sovereign canonical feature inventory. Metrics and capability counts reflect the current audited snapshot of the maintained platform surface.*