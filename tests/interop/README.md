# Native A2A interoperability

Run `python -m pytest tests/interop/test_a2a_go.py tests/interop/test_native_drainer_shutdown.py` from the Core checkout,
using an environment with Core's current dependencies and Go on `PATH`.
`a2a_go_peer/go.mod` declares the toolchain and public SDK release; `go.sum`
authenticates its dependencies. The `a2a-go-interop` CI job runs this gate after
unit tests, including when CI is called for an exact release ref. Missing Go,
build failures, and failed exchanges fail the gate rather than skip it.

The independent Go process uses the public A2A Go SDK's card verifier, JSON-RPC
and HTTP+JSON clients and servers. It implements Kestrel's declared sender
extension with Go's Ed25519 and ML-DSA implementations and an independent JCS
canonicalizer. There is no translation of Kestrel's former protocol.

The test creates temporary hybrid identities and a real Kestrel agent, waits
for initialization, publishes its signed card, and exchanges tasks in both
directions over each binding. It checks durable inbound creator ownership,
outbound dispatch persistence, and wrong-recipient refusal without task writes.
The business handler completes actual TaskManager tasks deterministically.
The test joins its HTTP servers, business handlers and Go process before the
temporary state is removed. It uses no operator agent or provider credentials.

The same required CI lane runs the real-agent SQLite shutdown regressions.
These hold original claim and acknowledgement COMMIT replies across dispatcher
and agent shutdown, verify that shutdown joins those owners, and refuse new
claims after shutdown. The PostgreSQL counterparts run in the integration lane.
The foreign peer also runs past its client startup deadline with an admitted
HTTP request held across SIGTERM. Shutdown must drain that request and write its
receipts using a fresh bounded cleanup context.

This proves the native protocol and persistence on a loopback SQLite agent.
It does not establish LLM quality, public reverse-proxy deployment, PostgreSQL
interoperability, MCP acceptance, or TCK conformance. Published dependency
adoption, exact-head CI and the full worktree review remain release gates.
