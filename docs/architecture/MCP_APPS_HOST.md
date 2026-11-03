# Authenticated MCP Apps host

Core owns interactive identity, routed-agent membership and privacy policy.
The optional MCP feature owns original tool registrations, native connections,
private result associations and physical Views. The feature creates its host
with `agent.create_mcp_apps_host(feature, configuration=..., is_current=...)`,
then attaches its `AppPublications` instance. This Core component requires the
owned delivery contract in MCP 0.10.1; complete console composition is supplied
by the MCP feature separately.

## Publication and HTTP authority

A chat part contains only `reference` and `session_id`. Those values locate a
result and do not authorize access. Initial publication requires the actual
active HTTP credential owner and the original live turn collector. The
collector snapshots Core's existing privacy-policy epoch. A result staged
before a policy change cannot publish after that change, including after a
round trip back to the same mode. Native execution and its successful SQL
audit remain successful when only UI publication is withheld; Core emits its
existing UI-delivery notice and does not replay the completed tool.

The feature router mounts under `/api/mcp/apps`. Each operation independently
authenticates the current request, checks the original principal, agent and
chat, and retains Core's privacy and authority leases through the actual outer
ASGI send. API-key, expiring JWT and native signed session-cookie requests are
supported. Cookie mutations use Core's canonical CSRF protection. Peer,
internal and EventSource query credentials do not acquire interactive grants.

`POST /publications/{reference}/views?session_id=...` opens a physical View and
returns the associated native resource, original arguments and result without
repeating the initial tool. `GET /views/{id}/tools` lists only the original
connection's App-visible tools. `POST /views/{id}/tools/call` accepts `name` and
`arguments` and uses the original governed Core binding, permission hooks,
one-shot approvals and required SQL audit. `DELETE /views/{id}` revokes and
joins the View; closing remains non-disclosing under restrictive privacy modes.
The body cannot select an owner, agent, tool registration or credential.

## Lifetime and sandbox policy

Each actual transport send has a fresh Core-owned child task. Core owns its
cancellation before it enters the View's delivery borrow; the View owns it
while borrowed. Retirement joins the original borrow, including transports
which suppress cancellation, and never assumes a buffering acknowledgement is
physical delivery. Errors returned to readers contain generic codes, not
credential diagnostics or native private payloads. An uncertain result never
invites an automatic retry of a tool call.

The host configuration requires an explicit public `host_origin` and a
different-origin `sandbox_url`. Origin separation is validated before replacing
an existing host, using operator configuration rather than a TLS proxy's
internal request URL. The sandbox service must use that same public host origin.
HTML, message, request and operation budgets are bounded. Connection policies
are operator configuration indexed by the original connection name; server
resource metadata requests grants but cannot select or increase host policy.
The feature's native AppBridge renderer intersects grants through its canonical
sandbox policy implementation. Raw HTML is not inserted into the host DOM.

Before installing another privacy policy, Core erases private associations and
revokes retained Views under the existing exclusive persistence gate. Earlier
private sends settle before that installation. The feature must maintain idle
retention, drain retired work before transport shutdown, and close its host
before releasing its feature incarnation. Failed cleanup retains its original
owner for retry. Replacing a host before the prior owner retires is refused.

The chat renderer context includes the owning pane's `sessionId` and
`isCurrentChat()` predicate. These are routing/lifetime information, never an
HTTP grant. They allow a View in a background pane to close on navigation and
prevent a stale part from opening into another active conversation.
