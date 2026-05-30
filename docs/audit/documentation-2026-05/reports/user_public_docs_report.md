# User Public Docs Lane Report

Source: subagent lane review, read-only, 2026-05-30.

## Scope Reviewed

User/public docs lane: `docs/user-documentation/`, `docs/use_cases/`, `docs/demos/`, `docs/concepts/`, `docs/design/launch/`, `README.md`, `docs/generated/FEATURES_user.md`, and `docs/generated/FEATURES_investor.md`.

## Canonical Doc Recommendation

Use `README.md` plus `KESTREL_FEATURES.md` as the public source of truth, but refresh both first. Public docs should present "core agent + installable packages" clearly, and generated audience docs should be treated as derived only after `KESTREL_FEATURES.md` is corrected.

## Stale Or Conflicting Claims

- `README.md` says `pip install kestrel-sovereign` includes voice, but voice is cataloged as `kestrel-feature-voice`, `core = false`, and no voice feature entry point/class is shipped in the wheel metadata.
- `README.md` still lists v0.1.8 stability while package version is `0.18.0`; it also calls wallet/economics stable while wallet is an external optional feature package.
- `docs/generated/FEATURES_user.md` and `docs/generated/FEATURES_investor.md` are stale: generated 2026-04-13, claim 42 modules, and describe RunPod/Vast/GCP compute, Talon, wallet, GitHub, voice, channels, council, and observability as broadly available/out-of-box.
- `docs/user-documentation/CONSTITUTIONAL_AI_EXPLANATION.md` says users cannot change the constitution and describes founder/company review; current constitution supports Sovereign-authored Book II `[emancipation]` activation and reanchor flows.
- `docs/user-documentation/KEY_CONCEPTS_EXPLAINED.md` overstates user-defined rules and blockchain/distributed storage as default behavior.
- `docs/user-documentation/SOVEREIGNTY_USER_GUIDE.md` promises millisecond Merkle/sharded backups and simple CID+key restore; memory lane already found this should align to current sovereignty export behavior and caveats.
- `docs/user-documentation/EMMA_INCEPTION_GUIDE.md` and `docs/user-documentation/KEY_CEREMONY_GUIDE.md` use root-level `inception_service.py`/`retirement_service.py` commands in places; current code lives under `kestrel_sovereign/`.
- `docs/design/launch/PREVIEW_PACKET_LANGUAGE.md`, `docs/design/launch/PUBLISH_READY_LANDING_PAGE_COPY.md`, and `docs/design/launch/SIMPLE_LAUNCH_PAGE_ONE_SCREEN.md` make public "clinical study / real patients / production" proof claims that are not evidenced by this repo's public docs and conflict with README's production caveats.
- `docs/use_cases/PATIENT_CONTROLLED_HEALTH_RECORDS.md` is a PRD/vision doc but reads like functional availability: FHIR/HL7 adapters, health vault, monetization contracts, and notary service should be marked aspirational or moved out of public user docs.

## Code/Package Evidence

- `pyproject.toml` version is `0.18.0`; package build includes only `kestrel_sovereign`; extracted packages are not bundled.
- `pyproject.toml` has empty `kestrel_sovereign.features`, `cloud_providers`, `voice_providers`, and `storage_providers` entry-point groups for this package.
- `kestrel_sovereign/data/feature_registry.toml` marks voice, wallet, MCP, GitHub, council, visual, legal, and cloud voice providers as non-core package installs.
- `kestrel_sovereign/features/__init__.py` discovers local feature modules; local discovery returned no external feature entry points.
- `kestrel_sovereign/static/index.html` shows current Console tabs include Chat, Identity, Constitution, Memories, Tasks, Sovereignty, Resources, Features, Security. README says 8 tabs and omits Features.
- `kestrel_sovereign/privacy.py`, `kestrel_sovereign/storage/privacy_wrapper.py`, and `kestrel_sovereign/endpoints/agent.py` support the five privacy presets, storage enforcement, local-only LLM switching, and destructive-op gate.
- `kestrel_sovereign/cli_agent_docker.py` and `kestrel_sovereign/cli_demo.py` back the Docker-agent and `kestrel demo run` flows.
- `kestrel_sovereign/features/channels/feature.py` provides channel registry/logging only; no concrete WhatsApp/Telegram/Discord/Slack adapters are included.

## Docs To Update

- `README.md`
- `docs/user-documentation/README.md`
- `docs/user-documentation/CONSTITUTIONAL_AI_EXPLANATION.md`
- `docs/user-documentation/KEY_CONCEPTS_EXPLAINED.md`
- `docs/user-documentation/OLLAMA_EXPLAINED.md`
- `docs/user-documentation/SOVEREIGNTY_USER_GUIDE.md`
- `docs/user-documentation/EMMA_INCEPTION_GUIDE.md`
- `docs/user-documentation/KEY_CEREMONY_GUIDE.md`
- `docs/demos/DEMO_SCRIPT.md`
- launch copy under `docs/design/launch/`
- `docs/use_cases/CONTEXT.md`

## Docs To Archive Or Mark Historical

- `docs/user-documentation/EDUCATION_IMPLEMENTATION_ROADMAP.md`
- `docs/user-documentation/USER_EDUCATION_STRATEGY.md`
- `docs/user-documentation/EXCEL_IMPORT_GUIDE.md`
- `docs/user-documentation/INTEGRITY_AUDIT_SYSTEM_SIMPLE.md`
- `docs/user-documentation/SOVEREIGN_COMPUTING_EXPLANATION.md`
- Mark `docs/use_cases/PATIENT_CONTROLLED_HEALTH_RECORDS.md` as aspirational PRD unless implementation evidence is added.

## Generated Docs To Regenerate

After refreshing `KESTREL_FEATURES.md`, regenerate:

- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_investor.md`
- `docs/generated/FEATURES_developer.md`

## Open Questions

- Should public docs say local voice is bundled, or require `kestrel-feature-voice`?
- Are wallet/economics public-ready, or should all wallet claims be optional-package-only?
- Can the clinical/real-patient launch claim be substantiated publicly, or should it be removed?
- Should public docs lead with Console/API flows over chat bang commands for export/import?
- Is `feature_registry.toml` intentionally using external package names with `core = true` for some in-tree features, or is that drift?

## Suggested First PR Slice

Start with public install truth: update `README.md`, generated docs source inputs, and `docs/user-documentation/README.md` to clearly separate core, built-in local modules, installable feature packages, and aspirational/historical docs. Then regenerate generated feature docs.

