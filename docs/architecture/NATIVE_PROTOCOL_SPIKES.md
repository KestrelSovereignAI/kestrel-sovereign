---
type: Architecture Spec
title: Native protocol Phase 0 evidence
description: Measured dependency and vendor-discovery evidence, with unverified acceptance gates.
resource: /docs/architecture/NATIVE_PROTOCOL_SPIKES.md
tags:
- docs
- architecture
- mcp
- a2a
timestamp: '2026-10-17T00:00:00Z'
status: draft
owner: architecture
canonical: true
generated: false
privacy: public
---

# Native protocol Phase 0 evidence

Evidence for [#3679](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3679),
measured 2026-10-17. The [design](NATIVE_MCP_A2A.md) is the implementation
contract. A successful resolver or discovery request is not production
acceptance. The OAuth spike remains incomplete until real consent, exchange,
tool execution and refresh are observed.

## Dependency survey

Surveyed all 18 entries of the local host feature manifest before changing
production constraints. Metadata came from PyPI's package JSON API. Three
entries are editable packages with no PyPI distribution; their source manifests
were inspected instead. These observations are a dated survey, not pins to
retain indefinitely.

| Manifest entry | Published/source version | Relevant declared bounds |
| --- | --- | --- |
| kestrel-feature-github | 0.2.0 | SDK >=0.11,<1 |
| kestrel-feature-voice | 0.5.1 | SDK >=0.30,<1 |
| kestrel-voice-openai | 0.4.0 | SDK >=0.30.0,<1 |
| kestrel-voice-xai | 0.1.2 | SDK >=0.30.0,<1 |
| kestrel-feature-reflection | 0.3.1 | SDK >=0.14.1,<1 |
| kestrel-feature-parametric-self[local] | 0.4.0 | SDK >=0.14.1,<1 |
| kestrel-channel-whatsapp | 0.2.2 | SDK >=0.23.0,<1 |
| kestrel-channel-telegram | 0.1.8 | SDK >=0.36,<0.40 |
| kestrel-feature-workflows | 0.7.2 | SDK >=0.36.0,<1 |
| kestrel-feature-talon | 0.2.13 | SDK >=0.37.1,<0.40; FastAPI >=0.115,<1 |
| kestrel-claws | editable 0.4.3 | SDK >=0.37.1,<0.40; core >=0.53.9 |
| kestrel-feature-eye | editable 0.4.4 | SDK >=0.37.1,<0.40; FastAPI >=0.115,<1; workflows >=0.5.0,<0.8 |
| kestrel-feature-flight | editable 0.4.3 | SDK >=0.37.1,<0.40; FastAPI >=0.115,<1; workflows >=0.7.0,<0.8 |
| kestrel-feature-features | 0.3.2 | SDK >=0.36.0,<0.40 |
| kestrel-feature-mcp | 0.4.1 | MCP >=1.26.0,<2; SDK >=0.10,<1 |
| kestrel-feature-observability[fleet] | 0.17.15 | SDK >=0.38.1,<1 |
| arize-phoenix | 20.23.0 | FastAPI >=0.137.0; Pydantic AI >=2.53.0,<3 |
| pydantic-ai-slim[anthropic,bedrock,google,mcp,openai,ui] | 2.57.0 | OpenAI >=3.26.0; MCP integration through fastmcp-slim[client] >=3.3.0,<5 |

Core at `9c69e9bed` pins FastAPI 0.136.3 and idna 3.16, caps OpenAI below
3, and its Phoenix extra caps Phoenix below 18 and Pydantic AI below 2.
MCP 2.3.0 requires httpx2 >=2.10.0, which requires idna >=3.18. Thus changing
only the MCP and Phoenix pins cannot resolve. The source SDK owns the old A2A
wire types; core's versions are re-exports.

The isolated Python 3.13 resolver candidate retained core's other direct
requirements and made these changes:

| Dependency | Candidate |
| --- | --- |
| MCP | 2.3.0 |
| A2A SDK, HTTP server/signing extras | 1.2.2 |
| FastAPI | 0.143.3 |
| idna | 3.20 |
| OpenAI | >=3.26.0,<4 (resolved 3.33.0) |
| Phoenix | 20.23.0 |
| Pydantic AI slim, existing extras | 2.57.0 |

`uv pip compile --python-version 3.13` resolved that candidate, and an isolated
environment installed it. Imports of `mcp`, `pydantic_ai`, `phoenix`, `a2a` and
`fastapi` succeeded. Pydantic remained 2.13.4. Production manifests and locks
have not been changed by this spike. This is not yet a resolve of the complete
host manifest, a Core/Phoenix startup test, or a test of provider behavior after
the OpenAI major upgrade. Those are release gates, including Python/platform
coverage supported by each affected package.

## Three live vendor discovery probes

Sent an unauthenticated `server/discover` JSON-RPC POST to the documented
Streamable HTTP URL with `Accept: application/json, text/event-stream` and
`MCP-Protocol-Version: 2026-07-28`. Each returned 401 with a
`WWW-Authenticate` protected-resource metadata URL. Fetched that metadata and
the authorization-server metadata without credentials or automatic redirects.
These 401 responses do not establish the vendors' negotiated MCP revision.

| Vendor | MCP endpoint | RFC 9728 resource metadata | CIMD advertised | PKCE |
| --- | --- | --- | --- | --- |
| Notion | `https://mcp.notion.com/mcp` | `https://mcp.notion.com/.well-known/oauth-protected-resource/mcp` | true | S256 |
| Linear | `https://mcp.linear.app/mcp` | `https://mcp.linear.app/.well-known/oauth-protected-resource/mcp` | true | S256 |
| Atlassian | `https://mcp.atlassian.com/v2/mcp` | `https://mcp.atlassian.com/.well-known/oauth-protected-resource/v2/mcp` | true | S256 |

Notion's issuer is `https://mcp.notion.com`; Linear's is
`https://mcp.linear.app`. Both advertise authorization-response issuer
validation. Atlassian advertises a path-qualified issuer under
`https://auth.atlassian.com` and did not advertise
`authorization_response_iss_parameter_supported`; a present response `iss`
must still be validated. All three also advertise dynamic registration, which
is not evidence for the CIMD premise and must not silently replace that test.

Vendor setup references: [Notion](https://developers.notion.com/guides/mcp/get-started-with-mcp),
[Linear](https://linear.app/docs/mcp),
[Atlassian](https://support.atlassian.com/atlassian-ai-gateway/docs/get-started-with-the-atlassian-remote-mcp-server/).
Linear documents a read-only endpoint and the `read` scope. Its ordinary
endpoint challenged with `read write`; do not silently accept that broad scope
in a read-only spike. Notion challenged with `default`; Atlassian advertises
both read and write scopes. Determine effective permissions before consenting.

### Remaining acceptance experiment

For each vendor, use a real test account and a reachable Kestrel CIMD document
with a declared callback. Record the following as independent outcomes:

1. Metadata/resource/issuer validation and client identity selection.
2. Acceptance or explicit rejection of the CIMD client ID and loopback redirect.
3. Consent, PKCE exchange and validation of state/issuer/resource.
4. Negotiated wire revision and one authorized read-only tool result.
5. Token refresh, including rotation, after restarting the client.
6. Any vendor allowlist, tenant policy or hosted-callback constraint.

No live login, CIMD URL, callback, authorization code, access token, refresh or
tool call was part of the discovery measurement. Client allowlisting is
therefore **unknown for all three**, not disproved by the metadata. No secret
or consent grant should be added to this document. Console UX selection remains
gated on this experiment; the user has been asked for test accounts and the
public metadata destination.

## Open release gates

- Complete the OAuth acceptance experiment above.
- Resolve and test the whole host manifest after coordinated package changes.
- Complete full-diff worktree review before each PR; record the final verdict.
- Implement/run both protocol conformance suites and real interop acceptance.
- Resolve [#596](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/596)
  before shipping remote connector write scopes. It was OPEN when checked.
- Track the MCP server direction in [#1990](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1990).
