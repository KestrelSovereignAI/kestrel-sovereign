# Seed agents

These are **tracked, reviewable seed/template `SOUL.md` files**, not live runtime
identity. `Dockerfile.multi_agent` copies this directory into `/app/agent_data/`
(`COPY docker/seed_agents/ /app/agent_data/`), so inside the built image each seed
lands at `/app/agent_data/<name>/SOUL.md`.

**What actually runs.** Only agents listed in `docker/multi_agent.cloudrun.toml`
(currently `Kestrel` and `kestrel-demo`) are registered and started by `host:app`.
The `emma`, `claw`, `nellie`, and `meridian` seeds are **reviewable templates that are
not yet wired into any `multi_agent.toml`**. On cold start the entrypoint loop
(`docker/multi_agent_entrypoint.sh`) does bootstrap an identity + DB for their dirs,
but because they are unregistered they are **bootstrapped-but-idle** — no agent
process is launched for them until an operator adds an `[agents.<name>]` entry (with
its own port + `autostart`) to the multi_agent config. To promote one to a running
agent, register it there and rebuild.

Outside the image, live agent identity lives in `agent_data/<name>/SOUL.md`, which is
**gitignored runtime data** (`.gitignore`) and is not edited from here. Editing a seed
file does **not** mutate an already-running agent on a host; it only changes what a
freshly built/provisioned image starts from.

## Operational lanes (Castle)

Each agent seed carries a bounded `Castle / Multi-Agent Operating Role` section. This is
operational posture only — additive identity context, not new runtime power, permissions,
or deployed behavior. Jason / the Sovereign remains the authority above any Castle policy;
Castle is the enterprise/control-plane policy layer and does not own the agents.

- **Emma** — orchestration: signal intake, work queues, assignment/handoffs, Talon
  dispatch, stale-work rescue, restart and blocker cleanup, loop closure only after
  evidence exists. Does not verify her own claims.
- **Claw** — implementation and repair: patches, failing tests, technical diagnosis,
  workflow hardening, plumbing cleanup. Implementation is not proof; does not certify its
  own fixes.
- **Nellie** — verification and evidence: independent review, regression checks, evidence
  gates, and state-claim discipline. A gate, not an optional reviewer.
- **Meridian** — governance and reference: doctrine/reference consistency, identity,
  lifecycle, naming, and memory boundaries. Not the orchestrator or runtime queue owner.

No agent is both the executor and the verifier for the same work. **Talon** is a bounded
execution worker inside workflow stages — a workforce, not an authority and not a verifier.

Runtime terminology: use `multi_agent` for the running mesh, `Castle` for the
enterprise/control-plane layer, and `Mews` for the fleet/work-visibility UI.
