# Kestrel Documentation Index

This directory holds detailed documentation for kestrel-sovereign. The top-level [`README.md`](../README.md) is the single best entry point; this index links into the deeper material.

## Start here

- **[`../README.md`](../README.md)** — project entry point, install, quick start, feature stability matrix
- **[`../QUICKSTART.md`](../QUICKSTART.md)** — concise install + first agent
- **[`../KESTREL_FEATURES.md`](../KESTREL_FEATURES.md)** — canonical feature inventory (the live source of truth)
- **[`GETTING_STARTED.md`](GETTING_STARTED.md)** — developer-oriented setup
- **[`PROJECT_VISION.md`](PROJECT_VISION.md)** — what kestrel-sovereign exists to do

## Architecture & engineering

- **[`architecture/README.md`](architecture/README.md)** — full architecture index with status flags (Active / Experimental / Aspirational), grouped by Core / Memory / Security / Features / Cloud / Wallets / Tools / Testing
- **[`audit/README.md`](audit/README.md)** — engineering quality program: feature proof matrices, seam campaigns, sync/async audit, issue-body archive. Actively maintained.
- **[`SOVEREIGNTY.md`](SOVEREIGNTY.md)** — architectural context for the sovereignty model
- **[`development/`](development/)** — developer environment, dev container, internal notes
- **[`guides/BUILDING_FEATURES.md`](guides/BUILDING_FEATURES.md)** — how to write a Kestrel feature

## Principles & governance

- **[`principles/KESTREL_CONSTITUTION.md`](principles/KESTREL_CONSTITUTION.md)** — the canonical constitutional document
- **[`principles/CONTEXT.md`](principles/CONTEXT.md)** — context for the constitutional design
- **[`principles/US_CONSTITUTION.md`](principles/US_CONSTITUTION.md)** — vendored reference document

## Diagrams

- **[`diagrams/README.md`](diagrams/README.md)** — index of architecture diagrams
- **[`diagrams/ARCHITECTURE_DIAGRAMS.md`](diagrams/ARCHITECTURE_DIAGRAMS.md)** — comprehensive visual reference
- **[`diagrams/data-architecture/`](diagrams/data-architecture/)** — storage, IPFS, sovereignty diagrams (DA-01 through DA-11)

## Users & operators

- **[`user-documentation/README.md`](user-documentation/README.md)** — end-user guides, concept explanations, key ceremonies
- **[`use_cases/`](use_cases/)** — concrete deployment scenarios
- **[`deployment/README.md`](deployment/README.md)** — deployment notes
- **[`demos/DEMO_SCRIPT.md`](demos/DEMO_SCRIPT.md)** — narrated demo walkthrough

## Generated documentation

- **[`generated/README.md`](generated/README.md)** — explains the generation pipeline
- **[`generated/FEATURES_developer.md`](generated/FEATURES_developer.md)** — auto-generated developer feature reference
- **[`generated/FEATURES_user.md`](generated/FEATURES_user.md)** — user-facing feature reference
- **[`generated/FEATURES_investor.md`](generated/FEATURES_investor.md)** — investor-facing feature reference

Regenerate via `scripts/generate_feature_docs.py`.

## Research & references

- **[`research/`](research/)** — external references, training notes, LoRA, KV-cache, LLM router

## Design

- **[`design/`](design/)** — brand guide, logo specs, ecosystem visual identity
- **[`design/launch/README.md`](design/launch/README.md)** — launch-page copy drafts (wireframe, one-screen variant, preview-packet language)
- **[`logo-prompts.md`](logo-prompts.md)** — logo prompt history

## Archive

- **[`archive/KESTREL_FEATURES_legacy.md`](archive/KESTREL_FEATURES_legacy.md)** — legacy feature catalog kept for historical reference

## Documentation freshness

Docs may lag code; treat them as guidance and prefer code/tests for current behavior. Each major architecture document carries an explicit status banner indicating whether the described feature is Active, Experimental, or Aspirational. If a doc and the code disagree, the code is authoritative — please file an issue or PR.

**Pre-release cleanup required:** Some docs in `business/`, `outreach/`, and `legal/` may contain identifying information that should be reviewed before public release.
