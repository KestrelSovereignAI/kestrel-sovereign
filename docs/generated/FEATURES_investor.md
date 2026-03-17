<!-- AUTO-GENERATED from KESTREL_FEATURES.md — do not edit manually -->
<!-- Audience: investor | Generated: 2026-03-17 | Model: anthropic/claude-sonnet-4-5-20250929 -->
<!-- Regenerate: uv run python scripts/generate_feature_docs.py --audience investor -->

# Kestrel Sovereign: Platform Overview for Investors

**Executive Summary**

Kestrel Sovereign is an AI agent platform architected for data sovereignty, constitutional governance, and vendor independence. The system provides portable decentralized identity (DID), multi-provider LLM orchestration with no platform lock-in, and enterprise-grade privacy controls from inception. Its extensible feature architecture and comprehensive API surface position it as infrastructure for the next generation of autonomous AI applications where users—not platforms—own and control their AI agents.

---

## Platform Architecture

### Constitutional Governance Framework

Kestrel implements a governance model with no current industry equivalent. Every agent operates under a formal constitution that defines permissible actions, ethical boundaries, and operational constraints. This constitutional layer:

- Provides legally auditable decision-making frameworks
- Enables transparent governance for regulated industries
- Creates verifiable compliance trails for enterprise deployments
- Establishes clear liability boundaries between platform, operator, and end-user

The constitutional system is not advisory—it is enforceable at the runtime level, providing organizations with unprecedented control over AI agent behavior in production environments.

### Decentralized Identity and Portability

Kestrel agents are born with cryptographically signed decentralized identifiers (DIDs). This architectural choice delivers:

- **True data portability**: Users can export their entire agent state—conversation history, learned preferences, accumulated context—and migrate between hosting providers or self-hosted infrastructure without vendor permission
- **Continuity verification**: Cryptographic signing ensures agent identity and history cannot be forged or tampered with
- **Multi-tenant sovereignty**: Organizations can operate fleets of agents with verified identity chains, enabling audit and compliance requirements that current cloud AI services cannot satisfy

The platform includes complete sovereignty lifecycle management: inception (agent birth with identity), graduation (full autonomy conferral), and retirement (verified data destruction).

### Agent Runtime and Context Management

The core orchestration engine manages:

- **Intelligent context assembly**: Dynamically constructs prompt context from conversation history, retrieved memories, active tool state, and constitutional constraints while respecting token budget limits
- **Streaming inference**: Real-time response generation with server-sent events for low-latency user experience
- **Command routing**: Extensible handler system that dispatches user requests to appropriate feature modules
- **Token budget optimization**: Automatic context pruning and prioritization to maximize response quality within model constraints

This runtime architecture separates concerns between conversation management, context retrieval, tool invocation, and LLM inference—enabling independent scaling and optimization of each subsystem.

---

## AI Capabilities

### Vendor-Independent Multi-LLM Platform

Kestrel's unified LLM service layer eliminates platform lock-in through:

- **Provider abstraction**: Single API surface that routes to OpenAI, Anthropic, Claude Max, Google Gemini, Vertex AI, Ollama (local), OpenRouter, and mock providers for testing
- **Automatic failover**: Provider outages trigger transparent failover to alternative models without service interruption
- **Cost optimization**: Route requests to the most cost-effective provider that meets quality requirements
- **Hybrid deployment**: Simultaneously use cloud LLMs for general tasks and local models for sensitive data processing

The platform includes a model catalog with metadata (context windows, pricing, capability flags), usage tracking across providers, and intelligent retry logic with exponential backoff.

Organizations can add proprietary LLM providers by implementing a standard adapter interface—no core platform modifications required.

### OpenAI-Compatible API Surface

For organizations with existing AI integrations, Kestrel exposes:

- `GET /v1/models` — Model listing in OpenAI format
- `POST /v1/chat/completions` — Drop-in replacement for OpenAI's chat API

This compatibility layer allows gradual migration from OpenAI to Kestrel infrastructure while preserving existing application code, reducing adoption friction for enterprises.

### Extensible Feature Architecture

Kestrel's feature system provides plugin-like extensibility:

- 41 discoverable feature modules in the current platform release
- 36 exported feature classes providing specialized capabilities
- Standard lifecycle hooks (initialization, context injection, command handling, cleanup)
- Isolated permission boundaries enforced by the security subsystem

Current feature domains include:

- **Memory systems**: Short-term conversation memory, long-term semantic memory, and memory agency (agents that curate and organize their own knowledge graphs)
- **External integrations**: GitHub repository access, web search, webhook delivery
- **Infrastructure control**: Cloud compute provisioning (GCP, RunPod, Vast.ai), model deployment, scheduler automation
- **Collaboration**: Peer-to-peer agent communication, council decision-making (multi-agent consensus), bridge protocols for inter-agent messaging
- **Identity and security**: Key management, wallet integration, visual identity generation, consent tracking

This architecture allows organizations to build proprietary features—domain-specific knowledge retrieval, internal API integrations, custom compliance checks—without forking the core platform.

---

## Data Sovereignty and Privacy

### Privacy-First Architecture

Kestrel implements privacy as a spectrum with five distinct operational modes:

| Mode | Data Storage | LLM Location | Export | Use Case |
|------|-------------|--------------|--------|----------|
| **ephemeral** | None | Local only | No | Maximum privacy, no persistence |
| **isolated** | Temporary | Local only | No | Session-based work, auto-purge |
| **anonymous** | PII-scrubbed | Cloud allowed | No | Cloud inference with privacy guarantees |
| **normal** | Full persistent | Cloud allowed | No | Standard production operation |
| **public** | Full persistent | Cloud allowed | Yes | Shareable agents, open collaboration |

Privacy mode enforcement is not configuration—it is architectural:

- `ephemeral` mode disables all storage backends at runtime
- `isolated` mode writes to temporary storage with automatic session cleanup
- `anonymous` mode runs PII scrubbing before cloud LLM requests and storage writes
- Privacy transitions require explicit user consent with audit logging

This design provides **privacy controls from day one** rather than bolting compliance onto an existing cloud-first architecture.

### Data Export and Portability

The sovereignty API provides:

- **Complete data export**: JSON packages containing conversation transcripts, memory graphs, identity chains, and file attachments
- **Selective export**: Filter by date range, privacy level, or conversation topic
- **Import verification**: Cryptographic validation of exported packages to detect tampering
- **File management**: Direct access to stored files with content-addressed retrieval

Organizations can implement retention policies, regulatory compliance workflows, and disaster recovery procedures using these primitives.

---

## Security and Access Control

### Permission System

Kestrel implements fine-grained authorization:

- **Tree-structured permissions**: Features request specific capabilities (filesystem read, network egress, credential access) with hierarchical approval
- **Pending authorization queue**: Non-interactive approval workflow for batch administrative review
- **Session-scoped grants**: Permissions expire with user session, preventing privilege escalation from stale authorizations
- **Audit trail**: Complete history of permission requests, approvals, denials, and revocations

This system allows organizations to operate AI agents in production while maintaining security postures comparable to traditional application infrastructure.

### Authentication Surface

The platform supports multiple authentication modes:

- **API key authentication**: Programmatic access for service-to-service integration
- **OAuth 2.0 flow**: Browser-based user authentication with session management
- **Localhost bootstrap**: Local-only API key generation for initial setup
- **Hybrid authorization**: Routes accept API key or session tokens, enabling both interactive and automated use cases

Critical routes enforce stricter requirements:

- Server-sent event streams support `?api_key=` query parameter for browser EventSource compatibility
- Browser conditional routes serve UI locally but redirect to OAuth when multi-tenant mode is active
- Webhook endpoints validate cryptographic signatures (e.g., Stripe webhook verification)

---

## Deployment Flexibility

### HTTP API Surface

Kestrel exposes a comprehensive REST API organized into functional route families:

**Agent Interaction**
- `/agent/invoke` — Synchronous agent invocation
- `/agent/stream` — Streaming responses via SSE
- `/agent/stop` — Interrupt long-running operations
- `/agent/privacy-mode` — Read and modify privacy settings
- `/agent/notifications` — Notification polling and SSE streams
- `/agent/context-status` — Inspect active context budget and composition
- `/agent/tasks` — Asynchronous task management
- `/agent/heartbeat/*` — Agent lifecycle monitoring

**Data Management**
- `/api/conversations/*` — Session and message CRUD
- `/api/memories/*` — Memory graph access and management
- `/api/saved-items/*` — Structured data persistence with schemas and tags
- `/api/files/*` — Content-addressed file storage

**Sovereignty Operations**
- `/api/sovereignty/export` — Generate portable agent packages
- `/api/sovereignty/import` — Restore from exported packages
- `/api/sovereignty/files/*` — File catalog and preview

**Administration**
- `/api/models` — Model catalog and selection
- `/api/agents` — Multi-agent management
- `/api/keys` — LLM provider credential management
- `/api/security/*` — Permission system controls
- `/api/observability/*` — Event streams and operational summaries

This API design prioritizes:
- RESTful resource modeling for client library generation
- SSE streams for real-time updates without WebSocket complexity
- Content negotiation for both JSON and file downloads

### Multi-Tenant and Single-User Modes

Kestrel operates in two deployment configurations:

**Single-User / Local Mode**
- OAuth optional, API key bootstrap enabled
- Filesystem storage backend
- Local LLM inference via Ollama
- Suitable for: Developer workstations, edge devices, air-gapped deployments

**Multi-Tenant / Cloud Mode**
- OAuth required for user sessions
- Database-backed storage with tenant isolation
- Cloud LLM routing with cost allocation
- Suitable for: SaaS platforms, enterprise internal tools, managed hosting

The same codebase supports both modes, allowing organizations to develop locally and deploy to cloud infrastructure without architectural changes.

---

## Extensibility and Integration

### Feature Plugin System

Features are self-contained modules that register with the platform and receive:

- **Context injection**: Access to conversation history, memory systems, and user preferences
- **Command routing**: Explicit invocation via `/command <feature_name> <args>`
- **Lifecycle hooks**: Initialization, cleanup, periodic tasks
- **Permission requests**: Declarative capability requirements enforced by security subsystem

Example feature categories:

**Infrastructure Automation**
- `gcp_compute`, `runpod`, `vastai` — Cloud resource provisioning
- `deploy` — Application deployment workflows
- `scheduler` — Cron-like task automation

**Knowledge Management**
- `memory` — Core memory storage
- `memory_agency` — Self-organizing knowledge graphs
- `save` — Structured data persistence
- `web_search` — Internet search integration

**Collaboration**
- `peers` — Inter-agent discovery and communication
- `council` — Multi-agent voting and consensus
- `bridge` — Cross-platform agent messaging

**Developer Experience**
- `code_edit` — Codebase modification with diff generation
- `github` — Repository access and pull request automation
- `mcp` — Model Context Protocol integration

Organizations can package proprietary features as Python modules, register them with the feature loader, and distribute them to agents without modifying core platform code.

### Model Context Protocol (MCP) Support

Kestrel implements the Model Context Protocol, enabling:

- **Standardized tool interfaces**: Define tools once, use across multiple LLM providers
- **Vendor-neutral tool calling**: Abstract provider-specific tool schemas (OpenAI functions, Anthropic tools, Gemini function calling)
- **Tool catalog management**: Dynamic tool registration and discovery

This positions Kestrel as infrastructure for the emerging MCP ecosystem rather than a proprietary closed platform.

---

## Operational Maturity

### Observability

The platform provides:

- **Event stream API**: Real-time operational events (request start/end, errors, performance metrics)
- **Summary statistics**: Aggregated views of system health
- **Detailed health endpoints**: Component-level status checks
- **Audit logging**: Immutable record of security-relevant events

These primitives integrate with standard observability stacks (Prometheus, Grafana, DataDog) via the `/api/observability/*` endpoints.

### Storage and Persistence

Kestrel abstracts storage through a pluggable backend system:

- **Async-first**: Non-blocking I/O for high-concurrency workloads
- **Multi-backend**: Filesystem, SQLite, PostgreSQL, or custom implementations
- **Content-addressed files**: Deduplication and integrity verification via cryptographic hashing
- **Privacy-aware**: Storage operations respect active privacy mode constraints

Organizations can implement custom storage backends (S3, Azure Blob, IPFS) by implementing the async storage interface.

### Database Access

Direct database introspection for administrative tooling:

- `GET /api/db/tables` — Schema discovery
- `GET /api/db/tables/{table_name}` — Table inspection and query

This allows organizations to build custom dashboards, reporting tools, and data pipelines without relying on platform-provided UI components.

---

## Market Positioning

### Competitive Differentiation

**vs. OpenAI Assistants API**
- Kestrel: Multi-provider LLM routing, no vendor lock-in
- OpenAI: Single-provider dependency, proprietary infrastructure

**vs. LangChain / LlamaIndex**
- Kestrel: Integrated platform with identity, privacy, and security from day one
- Competitors: Developer libraries requiring integration of separate identity, storage, and authorization systems

**vs. Hugging Face Transformers**
- Kestrel: Production-ready agent runtime with API surface
- Hugging Face: Model inference library requiring custom orchestration

**vs. Microsoft Semantic Kernel**
- Kestrel: Constitutional governance and cryptographic identity
- Semantic Kernel: Enterprise integration focus, weaker data sovereignty story

### Strategic Advantages

1. **Portable Identity**: Users can migrate agents between providers—DID architecture prevents platform lock-in at the identity layer, not just the API layer

2. **Constitutional AI**: Only platform with enforceable governance frameworks—critical for regulated industries (healthcare, finance, legal) where AI decision-making requires audit trails and liability boundaries

3. **Privacy Spectrum**: Five-mode privacy system provides GDPR/CCPA compliance pathways that cloud-first platforms cannot match without architectural rewrites

4. **Vendor Independence**: Multi-LLM routing eliminates single-provider risk—organizations hedge against OpenAI/Anthropic pricing changes, outages, or policy shifts

5. **Extensible Architecture**: Feature plugin system allows proprietary differentiation—organizations build competitive advantages on Kestrel infrastructure rather than being constrained by SaaS feature roadmaps

### Total Addressable Market

Kestrel targets three market segments:

**Enterprise AI Infrastructure**
- Organizations deploying AI agents in regulated industries
- Required capabilities: Audit trails, data sovereignty, governance frameworks
- Current gap: Cloud AI services lack compliance primitives

**AI-Native SaaS Platforms**
- Startups building agent-based applications
- Required capabilities: Multi-tenancy, cost optimization, vendor independence
- Current gap: Building agent infrastructure is undifferentiated heavy lifting

**Edge and Specialized Deployments**
- Defense, healthcare, financial institutions with air-gap requirements
- Required capabilities: Local LLM inference, cryptographic identity, zero data exfiltration
- Current gap: Cloud AI services architecturally incompatible with security requirements

---

## Technical Validation

### Test Coverage and Verification

Kestrel maintains contract tests for critical subsystems:

- **Authentication decision tables**: Verify that every route enforces correct authentication requirements
- **Endpoint contract suite**: API response schemas match documented specifications
- **Feature doc canonicality**: Generated documentation matches actual codebase capabilities
- **Route discovery**: HTTP surface matches registered FastAPI routers

This testing discipline ensures that platform capabilities are verifiable and that documentation drift is detectable via CI/CD pipelines.

### Audit and Compliance Readiness

The platform includes audit working papers and verification tooling that organizations can extend for compliance workflows:

- Constitutional decision logging
- Permission request/approval trails
- Data export/import verification
- Cryptographic identity chain validation

These primitives position Kestrel as **audit-ready infrastructure** rather than a black-box AI service.

---

## Conclusion

Kestrel Sovereign represents a fundamental architectural shift in AI agent platforms: from cloud-centric services where providers own user data and identity, to infrastructure where users and organizations retain sovereignty over their AI agents.

The platform's constitutional governance, portable DID-based identity, multi-provider LLM orchestration, and privacy-first architecture create defensible competitive advantages in markets where data sovereignty, regulatory compliance, and vendor independence are strategic requirements rather than nice-to-have features.

For investors evaluating the AI infrastructure landscape, Kestrel's positioning is clear: it is not competing to be a better OpenAI wrapper—it is building the substrate for the next generation of autonomous AI applications where control, ownership, and portability are non-negotiable.