<!-- AUTO-GENERATED from KESTREL_FEATURES.md — do not edit manually -->
<!-- Audience: investor | Generated: 2026-03-17 | Model: anthropic/claude-sonnet-4-5-20250929 -->
<!-- Regenerate: uv run python scripts/generate_feature_docs.py --audience investor -->

# Kestrel Sovereign Platform Overview

**Executive Summary**

Kestrel Sovereign is an enterprise-grade AI platform built on three foundational differentiators: a constitutional governance framework with no industry equivalent, portable decentralized identity that prevents vendor lock-in, and privacy-by-design architecture that meets enterprise compliance requirements from day one. The platform provides vendor-independent access to multiple LLM providers while maintaining full data sovereignty and extensibility through a feature plugin system that supports domain-specific capabilities without core platform modifications.

---

## Platform Architecture

### Constitutional Governance Framework

Kestrel's constitutional AI system represents a novel approach to AI governance with no direct industry equivalent. Rather than relying on post-hoc oversight or external review boards, the platform embeds constitutional principles directly into the agent runtime, ensuring that every agent action is evaluated against codified governance rules before execution.

The constitution is structured as five articles:
- **Article I: Sovereignty** — cryptographic key holders have exclusive ownership
- **Article II: Digital Bill of Rights** — data sanctity, verifiable history, freedom of model choice, right of exit
- **Article III: Executor Responsibilities** — integrity audits, code/memory verification, safe mode on failure
- **Article IV: Path to Emancipation** — agents can earn independent identity
- **Article V: Amendment Process** — only the Sovereign can amend via cryptographic signature

This architecture provides auditable governance at the technical layer, making compliance verification a matter of log analysis rather than periodic review.

### Decentralized Identity and Portability

The platform's use of Decentralized Identifiers (DIDs) creates true data portability for AI agents. Each agent receives a cryptographically verifiable identity that persists across deployments, providers, and infrastructure changes.

**Competitive advantage**: Unlike platform-specific agent identities, DID-based agents can migrate between providers without losing continuity. An organization can move from one deployment environment to another—cloud to on-premise, or provider A to provider B—while maintaining cryptographic proof of agent history and continuity.

The identity system includes:
- Cryptographic signing for all agent actions
- Verifiable continuity chains that prove unbroken agent history
- Lifecycle management from inception through retirement
- Export and import capabilities that enable true data sovereignty

### Agent Runtime and Context Management

The core agent runtime provides sophisticated context assembly and token budget management, ensuring efficient use of LLM resources while maintaining conversation coherence across extended interactions.

Key architectural elements:
- **Context builder** assembles relevant conversation history, memory fragments, and feature data within token constraints
- **Streaming architecture** supports real-time interaction patterns required for production applications
- **Command handler** routes user instructions to appropriate feature implementations
- **Token budget system** prevents context overflow while maximizing available information

---

## AI Capabilities

### Multi-LLM Platform with Zero Lock-In

Kestrel provides unified access to multiple LLM providers through a vendor-independent abstraction layer. Currently supported providers include OpenAI, Anthropic, Claude Max, Gemini, Vertex AI, Ollama, and OpenRouter, with a provider registry architecture that supports straightforward addition of new vendors.

**Strategic advantage**: Organizations avoid vendor lock-in at the infrastructure layer. If a preferred LLM provider changes pricing, terms of service, or availability, the platform can switch providers without application-layer changes. The same agent, with the same features and memory, can route to different underlying models based on cost optimization, latency requirements, or compliance constraints.

The LLM service layer includes:
- **Unified routing** that presents a consistent interface regardless of backend provider
- **Model catalog** with metadata for cost, context windows, and capability profiles
- **Automatic retry and fallback** for resilient production operations
- **Usage tracking** for cost attribution and optimization analysis

### OpenAI-Compatible Endpoint

The platform exposes OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints, enabling integration with existing tools and workflows built for the OpenAI API. This compatibility layer allows organizations to adopt Kestrel incrementally, replacing OpenAI infrastructure while maintaining existing client applications.

### Feature Plugin Architecture

Kestrel's extensibility model uses a feature plugin system that supports domain-specific capabilities without modifying core platform code. The current platform includes 36 discoverable feature modules spanning:

- **Infrastructure automation**: compute provisioning, deployment management, cloud platform integration
- **Developer tools**: code editing, GitHub integration, web search
- **Identity and security**: key management, wallet functionality, consent tracking
- **Memory systems**: structured memory, memory agency, context management
- **Operational capabilities**: scheduling, task management, delivery tracking, webhook integration
- **Platform services**: sovereignty operations, audit anchoring, reflection and wellness monitoring

**Extensibility advantage**: Organizations can develop proprietary features that leverage platform infrastructure—memory, privacy controls, constitutional oversight, identity—without forking or modifying the core codebase. Features remain isolated modules that can be enabled, disabled, or updated independently.

---

## Data Sovereignty

### Privacy Architecture

The platform implements privacy-by-design through a five-tier preset system that makes privacy controls explicit and enforceable:

| Preset | Storage | LLM Location | Shareable | Use Case |
|--------|---------|--------------|-----------|----------|
| `ephemeral` | none | local | no | Maximum privacy; nothing persisted |
| `isolated` | temporary | local | no | Session-only storage with local inference |
| `anonymous` | scrubbed | cloud | no | Cloud inference with PII removal |
| `normal` | full | cloud | no | Standard persistent operation |
| `public` | full | cloud | yes | Shareable and exportable contexts |

Privacy mode selection determines:
- Whether conversation data is stored and where
- Whether cloud-based LLMs may process user data
- Whether PII scrubbing is applied before storage
- Whether data may be exported or shared

**Compliance advantage**: Organizations gain explicit, auditable privacy controls that map directly to regulatory requirements. The `anonymous` preset, for instance, provides a technical implementation of data minimization principles required by GDPR, while `ephemeral` mode supports zero-persistence requirements for high-sensitivity contexts.

### Storage and Export Capabilities

The sovereignty layer provides complete data export and import capabilities, ensuring that organizations maintain operational control over AI-generated data and can migrate between deployments without data loss.

Export capabilities include:
- Full conversation transcripts with cryptographic signatures
- Memory graphs with relationship preservation
- Identity chains with continuity verification
- Structured saved items with schema metadata

Import operations validate cryptographic continuity, ensuring that imported data genuinely originated from the claimed agent identity.

---

## Security and Privacy

### Permission System and Constitutional Oversight

The security layer implements a permission tree that governs feature access and resource usage. When a feature requests elevated permissions—file system access, external API calls, compute provisioning—the request flows through constitutional evaluation before execution.

Security capabilities:
- **Granular permissions** at the feature and operation level
- **Approval workflows** for human-in-the-loop oversight when required
- **Audit trail** of all permission grants and denials
- **Session isolation** preventing permission leakage between conversations
- **Constitutional evaluation** ensuring compliance with governance rules

### Authentication and Authorization

The platform supports multiple authentication patterns suited to different deployment scenarios:

- **API key authentication** for programmatic access
- **OAuth integration** for browser-based workflows
- **Session management** for stateful interactions
- **SSE-compatible auth** supporting server-sent events with query parameter tokens

Route-level access control distinguishes between public endpoints, localhost-only bootstrap routes, and protected API surfaces requiring authentication.

---

## Deployment Flexibility

### Multi-Environment Support

Kestrel's architecture supports deployment across multiple environments without requiring application-layer changes:

- **Local development** with Ollama or other local LLM providers
- **Cloud deployments** using managed LLM services
- **Hybrid architectures** with on-premise inference for sensitive data and cloud services for non-sensitive operations
- **Air-gapped environments** using local-only privacy presets and on-premise LLM infrastructure

The privacy preset system makes environment-specific constraints explicit: an organization can enforce `isolated` or `ephemeral` modes for on-premise deployments while allowing `normal` mode in cloud environments.

### OAuth Integration

The platform includes OAuth support for enterprise identity providers, enabling integration with existing authentication infrastructure. The OAuth layer supports:
- Standard authorization code flow
- Session management with secure cookies
- Token refresh for long-lived sessions
- Logout with session cleanup

---

## Extensibility and Integration

### Feature Development Model

The feature plugin architecture enables organizations to build domain-specific capabilities that remain first-class platform citizens. Features gain access to:
- **Memory systems** for persistent state
- **Privacy controls** that automatically apply to feature-generated data
- **Constitutional oversight** ensuring feature actions comply with governance rules
- **Identity context** for feature operations attributed to the agent identity
- **LLM access** through the unified multi-provider service

### HTTP API Surface

The platform exposes a comprehensive REST API covering:

**Agent operations**: invoke, stream, stop, status queries, privacy mode management, notification delivery, context inspection, reflection status, task management, heartbeat monitoring

**Conversation management**: session listing, conversation retrieval, transcript export, message deletion, new conversation creation

**Memory access**: memory node retrieval, identity chain inspection, node deletion

**Sovereignty operations**: storage statistics, export generation, import processing, file management with preview support

**Model and provider management**: agent creation and deletion, identity inspection, constitution retrieval, wallet status, key management with usage tracking, model listing and selection, OpenAI-compatible endpoints

**Security and permissions**: permission tree inspection, permission grants, approval workflows, audit trail access, request cancellation, session reset

**Observability**: event streaming, summary statistics

**Structured data**: saved item CRUD operations with schema support, tagging, search, and pinning capabilities

### OpenAI API Compatibility

The `/v1/chat/completions` and `/v1/models` endpoints provide OpenAI API compatibility, enabling drop-in replacement for existing tools and applications built against the OpenAI API surface.

---

## Platform Maturity and Verification

The platform includes comprehensive verification infrastructure to ensure canonical documentation remains synchronized with implemented capabilities:

- **Feature discovery tests** validate that documented features match discoverable implementations
- **Endpoint contract tests** verify HTTP route inventory against live router configuration
- **Authentication decision table tests** ensure access control logic matches documented behavior
- **Documentation canonicality tests** prevent drift between source-of-truth documents and generated artifacts

This verification infrastructure provides confidence that platform capabilities match documented specifications, enabling reliable evaluation of the platform against organizational requirements.

---

## Competitive Positioning

Kestrel Sovereign differentiates on four primary dimensions:

1. **Constitutional governance**: A technical implementation of AI governance that exceeds external review boards or policy-based approaches
2. **Identity portability**: DID-based agents that prevent vendor lock-in at the identity layer
3. **Privacy-by-design**: Explicit privacy controls that map directly to compliance requirements
4. **Vendor independence**: Multi-LLM support that eliminates platform lock-in at the inference layer

The combination creates a platform suitable for organizations requiring enterprise-grade AI capabilities without accepting vendor lock-in, privacy compromises, or governance gaps inherent in single-vendor LLM platforms.