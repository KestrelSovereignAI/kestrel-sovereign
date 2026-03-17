<!-- AUTO-GENERATED from KESTREL_FEATURES.md — do not edit manually -->
<!-- Audience: investor | Generated: 2026-03-16 | Model: anthropic/claude-sonnet-4-5-20250929 -->
<!-- Regenerate: uv run python scripts/generate_feature_docs.py --audience investor -->

# Kestrel Sovereign: Platform Overview for Investors

**Executive Summary**

Kestrel Sovereign is a Constitutional AI platform architected for enterprise data sovereignty and vendor independence. The platform enables organizations to deploy AI agents with portable identities, multi-provider LLM routing, and enterprise-grade privacy controls—without platform lock-in. Built on decentralized identity standards and a governance framework with no industry equivalent, Kestrel positions customers to own and control their AI infrastructure as regulatory requirements evolve.

---

## Platform Architecture

### Constitutional Governance Framework

Kestrel implements a governance model unprecedented in the AI platform market: a human-readable constitution that defines agent behavior, user rights, and system boundaries. This constitutional layer serves as:

- **Regulatory readiness:** A transparent governance artifact that can be audited, versioned, and evolved to meet emerging AI compliance requirements
- **Trust anchor:** Users interact with agents bound by explicit, published rules rather than opaque corporate policies
- **Differentiation:** No competing platform offers equivalent constitutional governance as a first-class architectural component

The constitution is cryptographically anchored to agent identity and lifecycle events, creating an auditable chain from governance principles to runtime behavior.

### Decentralized Identity (DID) System

Kestrel agents operate with W3C-standard decentralized identifiers (DIDs), enabling:

- **Portability:** Users own their AI agent's identity independent of hosting provider—agents can migrate between infrastructure without data lock-in
- **Continuity:** Cryptographic signing ensures agent identity persists across deployments, providing verifiable history and accountability
- **Multi-party trust:** DIDs enable federated scenarios where agents authenticate across organizational boundaries without central authority

This identity architecture positions Kestrel uniquely for enterprise scenarios where AI agents must operate across legal entities, geographies, or regulatory domains.

### Agent Lifecycle Management

The platform implements full-lifecycle sovereignty ceremonies:

- **Inception:** Agents are instantiated with constitutional binding, identity generation, and initial capability grants
- **Graduation:** Agents transition from supervised to autonomous operation with formal capability elevation
- **Retirement:** Controlled decommissioning with export, archival, and identity revocation

These lifecycle primitives enable compliance scenarios (e.g., GDPR-mandated deletion, regulatory audit trails) that consumer AI platforms cannot address.

---

## AI Capabilities

### Multi-LLM Platform with Unified Routing

Kestrel eliminates vendor lock-in through a unified LLM service layer supporting eight providers in-tree:

- **Cloud providers:** OpenAI, Anthropic, Gemini, Vertex AI, OpenRouter
- **Private deployment:** Ollama (local models), Claude Max
- **Testing/development:** Mock provider for CI/CD

The platform's provider registry and routing layer enable:

- **Cost optimization:** Route queries to least-cost providers that meet quality/latency requirements
- **Regulatory compliance:** Keep sensitive workloads on-premises or in specific geographies
- **Resilience:** Failover between providers without application changes
- **Future-proofing:** Integrate new LLM vendors as commodities rather than platform migrations

Model metadata tracking captures capabilities, pricing, and context limits across providers, enabling intelligent routing without hardcoded vendor assumptions.

### Extensible Feature System

Kestrel's feature module architecture discovered approximately 40 feature families in the current release, including:

- **Enterprise integration:** GitHub, webhooks, channels, compute orchestration
- **AI tooling:** Code editing, web search, reflection, task scheduling
- **Infrastructure:** GCP Compute, RunPod, Vast.ai deployment automation
- **Governance:** Consent management, security permissions, audit anchoring
- **Memory systems:** Persistent memory, memory agency, context management

Features operate as pluggable capabilities rather than monolithic platform services, enabling:

- **Custom deployments:** Enterprises install only required features, reducing attack surface and operational complexity
- **Rapid extension:** New capabilities integrate via standard feature contracts without core platform changes
- **Commercial flexibility:** Feature-level licensing models (e.g., base platform + premium feature packs)

### Streaming and Context Management

The agent runtime implements:

- **Token budget management:** Dynamic context assembly respecting model limits and cost constraints
- **Streaming responses:** Server-sent events (SSE) for real-time agent output without polling
- **Notification system:** Persistent event channels for asynchronous agent operations

These capabilities support production scenarios where agents operate on long-running tasks, require human-in-the-loop approval, or must maintain context across extended conversations.

---

## Data Sovereignty

### Privacy Modes as Competitive Moats

Kestrel implements a privacy preset system with granular control over data persistence and LLM routing:

| Preset | Storage | LLM Location | Shareable | Use Case |
|--------|---------|--------------|-----------|----------|
| **ephemeral** | none | local only | no | Maximum privacy: healthcare intake, legal consultation |
| **isolated** | temporary | local only | no | Session-scoped: exploratory analysis, draft review |
| **anonymous** | PII-scrubbed | cloud allowed | no | Operational data: analytics, pattern detection |
| **normal** | full persistence | cloud allowed | no | Standard business use |
| **public** | full persistence | cloud allowed | yes | Collaborative work, published content |

This granularity enables:

- **HIPAA compliance:** Route patient interactions through `ephemeral` mode with local LLMs only
- **GDPR compliance:** Use `anonymous` mode for EU data analysis, ensuring no PII reaches cloud providers
- **Competitive intelligence:** Prevent sensitive business data leakage to third-party LLM training sets
- **Regulatory defense:** Privacy mode is cryptographically enforced and auditable, not just UI decoration

No competing platform offers equivalent privacy controls as baseline architecture rather than enterprise add-on.

### Export and Import Capabilities

The sovereignty API family enables:

- **Data portability:** Full export of agent state, memories, conversations, and configuration
- **Provider switching:** Import existing agent state into new deployment without vendor dependency
- **Regulatory compliance:** Respond to data access requests with complete, verifiable exports
- **Disaster recovery:** Backup and restore agent state independent of hosting infrastructure

These capabilities transform AI agents from cloud-resident services into portable assets under customer control—critical for regulated industries and enterprise procurement.

### Storage Abstraction

The platform's storage layer supports:

- **Pluggable backends:** Abstract storage interface allows deployment on customer infrastructure
- **Async operations:** Non-blocking persistence for production workloads
- **Content addressing:** File operations use cryptographic hashes, enabling deduplication and integrity verification

Storage abstraction enables scenarios from air-gapped government deployments to multi-region commercial operations without application changes.

---

## Security and Privacy

### Permission System and Consent Management

Kestrel implements feature-level permission controls:

- **Permission tree:** Hierarchical capability grants with inheritance and override semantics
- **Consent workflow:** Explicit user approval for sensitive operations (data access, external API calls, financial transactions)
- **Audit trail:** Immutable log of permission grants, consent decisions, and security events

This security model positions Kestrel for:

- **Zero-trust environments:** Least-privilege access enforced at feature granularity
- **Compliance scenarios:** Demonstrate consent and authorization for regulated operations
- **Multi-tenant SaaS:** Isolate capabilities between customers or organizational units

The pending approval system surfaces security requests to administrators before execution, enabling human-in-the-loop governance for high-risk operations.

### Authentication and Authorization

The platform implements multiple authentication surfaces:

- **API keys:** Service-to-service authentication for programmatic access
- **OAuth integration:** Standard web authentication flow with session management
- **SSE query tokens:** Secure real-time streaming without cookie overhead
- **Localhost bootstrap:** Initial setup with automatic key generation for self-hosted deployments

This flexibility supports scenarios from public SaaS (OAuth) to private enterprise deployment (API keys) to local development (bootstrap mode) without architectural compromises.

### Observability and Audit

The platform instruments:

- **Event stream:** Real-time feed of agent actions, security decisions, and system events
- **Audit anchoring:** Cryptographic binding of events to agent identity and constitution
- **Summary analytics:** Aggregated metrics for operational monitoring

These observability primitives enable:

- **Forensic analysis:** Reconstruct agent behavior for incident response
- **Compliance reporting:** Generate audit trails for regulatory examination
- **Performance optimization:** Identify bottlenecks in agent workflows

---

## Deployment Flexibility

### Multi-Environment Support

Kestrel's architecture supports deployment across:

- **Public cloud:** Standard SaaS offering with OAuth and stripe integration
- **Private cloud:** Customer-managed infrastructure (GCP, AWS, Azure via compute features)
- **On-premises:** Air-gapped deployment with local LLMs and storage
- **Hybrid:** Route public queries to cloud LLMs while keeping sensitive data on-premises

This flexibility addresses enterprise procurement requirements that single-deployment-model platforms cannot satisfy.

### Compute Orchestration Features

The platform includes compute management for:

- **GCP Compute Engine:** Automated instance provisioning and lifecycle management
- **RunPod:** GPU compute orchestration for local model inference
- **Vast.ai:** Spot market GPU allocation for cost-sensitive workloads

These features enable:

- **Cost optimization:** Dynamically provision compute for batch workloads, terminate when idle
- **Resource efficiency:** Scale inference capacity based on demand without overprovisioning
- **Vendor negotiation:** Leverage multiple compute providers competitively

### Health and Reliability

The platform exposes:

- **Health endpoints:** Basic and detailed system status for load balancer integration
- **Heartbeat system:** Agent liveness monitoring with configurable intervals
- **Graceful degradation:** Agent continues operation during provider outages by routing to available alternatives

These operational capabilities support production SLAs and enterprise uptime requirements.

---

## Extensibility and Integration

### Model Context Protocol (MCP) Support

Kestrel integrates the Model Context Protocol standard, enabling:

- **Tool ecosystem:** Agents leverage external tools via standardized protocol
- **Third-party extensions:** Vendors publish MCP-compatible tools without platform integration
- **Future-proofing:** As MCP adoption grows, Kestrel agents gain capabilities without platform updates

### Webhook and Channel Systems

The platform implements:

- **Outbound webhooks:** Notify external systems of agent events (task completion, approval requests)
- **Inbound webhooks:** Trigger agent workflows from external events (payment confirmation, CI/CD status)
- **Channel abstraction:** Route agent notifications to multiple destinations (Slack, email, SMS)

These integration primitives position Kestrel as orchestration hub rather than isolated AI service.

### API Compatibility

The platform exposes:

- **OpenAI-compatible endpoints:** `/v1/models`, `/v1/chat/completions` for drop-in replacement scenarios
- **Native API:** Feature-rich endpoints for Kestrel-specific capabilities (sovereignty, privacy, constitution)
- **RESTful design:** Standard HTTP semantics for enterprise API gateway integration

This dual-API approach enables gradual migration from existing OpenAI-dependent systems while accessing differentiated Kestrel features.

### Database and File Management

The platform provides:

- **Database introspection:** API for exploring agent memory and state schemas
- **File operations:** Content-addressed storage with preview and retrieval
- **Saved items:** Structured data persistence with tagging, schemas, and search

These primitives support:

- **Custom tooling:** Build dashboards, analytics, or integrations against agent data
- **Migration scenarios:** Extract data for transfer to new systems or analysis pipelines
- **Workflow automation:** Agents persist structured outputs for downstream consumption

---

## Market Positioning

### Competitive Advantages

Kestrel's architecture delivers differentiation on four dimensions competitors cannot easily replicate:

1. **Constitutional governance:** No platform offers equivalent transparency and auditability of AI behavior principles
2. **DID-based portability:** Users own agent identity independent of hosting provider—prevents lock-in
3. **Privacy-first architecture:** Granular privacy modes as baseline feature rather than enterprise add-on
4. **Vendor-independent LLM routing:** Eight providers supported with unified interface, eliminating single-vendor risk

### Target Market Segments

The platform's capabilities address acute pain points in:

- **Regulated industries:** Healthcare, financial services, government agencies requiring data sovereignty
- **Enterprise IT:** Organizations seeking AI adoption without vendor lock-in or shadow IT proliferation
- **Privacy-conscious markets:** EU/UK deployments under GDPR, California under CCPA
- **Multi-national corporations:** Operations spanning regulatory regimes with conflicting data residency requirements

### Commercial Model Enablers

The platform's architecture supports multiple monetization strategies:

- **Freemium SaaS:** Public privacy tier free, premium tiers unlock `normal`/`public` modes
- **Enterprise licensing:** Private deployment with support and compliance tooling
- **Feature packs:** Base platform + premium features (compute orchestration, advanced memory, integration connectors)
- **Managed services:** Operated instance with SLA guarantees and professional services

---

## Technical Maturity

### Route Surface and API Stability

The platform exposes a comprehensive HTTP API surface spanning:

- **Agent operations:** Invocation, streaming, task management, notifications
- **Data management:** Conversations, memories, files, saved items
- **Administration:** Security, observability, sovereignty operations
- **Integration:** Webhooks, channels, compute orchestration

Route families are organized by functional area with consistent patterns, suggesting mature API design rather than ad-hoc accumulation.

### Test and Validation Coverage

The codebase includes:

- **Contract tests:** Verify endpoint behavior and authentication requirements
- **Canonicality tests:** Ensure feature documentation matches discovered implementation
- **Auth decision tables:** Validate security model across route families

This test discipline indicates production-grade operational rigor rather than prototype or research project.

### Configuration and Deployment

The platform supports:

- **Environment-based configuration:** 12-factor app principles for cloud-native deployment
- **Feature toggles:** Enable/disable capabilities without code changes
- **Provider configuration:** Hot-swap LLM providers and credentials without restart

These operational capabilities reduce deployment friction and enable continuous delivery practices.

---

## Summary

Kestrel Sovereign addresses the AI platform market's critical gap: enterprise-grade governance, data sovereignty, and vendor independence are absent from consumer-focused AI services, while traditional enterprise software lacks modern AI capabilities.

The platform's constitutional governance framework, DID-based identity, and privacy-first architecture deliver defensible competitive advantages that cannot be easily replicated by cloud hyperscalers (who profit from lock-in) or startups (who lack resources for comprehensive governance engineering).

Target customers face acute pain: regulated industries cannot adopt AI without data sovereignty guarantees, enterprises cannot commit to single-vendor AI platforms given technology uncertainty, and multi-national operations cannot satisfy conflicting regulatory requirements with one-size-fits-all solutions.

Kestrel's technical maturity—comprehensive API surface, test coverage, and operational tooling—indicates production readiness rather than research project. The platform's extensible architecture positions it to capture value as AI capabilities commoditize: Kestrel's moat is governance, portability, and privacy infrastructure rather than model performance.