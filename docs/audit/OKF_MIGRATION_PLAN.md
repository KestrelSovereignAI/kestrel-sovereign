---
type: Migration Plan
title: Kestrel Documentation OKF Migration Plan
description: Plan for converting Kestrel documentation to Google Open Knowledge Format and wiring automated upkeep into doc generation and Talon workflows.
resource: /docs/audit/OKF_MIGRATION_PLAN.md
tags: [docs, okf, talon, generated-docs, audit]
timestamp: 2026-06-18T00:00:00Z
status: implemented
---

# Kestrel Documentation OKF Migration Plan

## Purpose

Convert Kestrel's documentation corpus into an Open Knowledge Format (OKF)
bundle while preserving the existing human-readable docs tree, generated
feature docs, and audit-ledger workflow.

OKF v0.1 is intentionally minimal: markdown files with YAML frontmatter,
normal markdown links, optional `index.md` files for progressive disclosure,
and optional `log.md` files for update history. A conformant bundle requires
parseable YAML frontmatter with a non-empty `type` field on every non-reserved
markdown document.

References:

- Google Cloud announcement: https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing
- OKF v0.1 spec: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md

## Current State

Snapshot from 2026-06-18:

- Initial baseline before this migration: `docs/` contained 319 markdown files,
  0 markdown files had YAML frontmatter at the top of the file, and 273
  markdown files had an H1 heading.
- Current corpus after implementation: `docs/` contains 260 markdown files.
- 254 non-reserved Markdown files have OKF frontmatter and validate with
  `uv run python scripts/docs_okf.py validate --all docs`.
- 6 generated OKF views are intentionally reserved and skipped:
  `docs/audit/index.md`, `docs/audit/log.md`, `docs/generated/index.md`,
  `docs/generated/log.md`, `docs/architecture/index.md`, and
  `docs/architecture/log.md`.
- 0 non-reserved Markdown files are missing OKF frontmatter.
- Only about 14 files mention status conventions or status-like metadata.
- Existing navigation and inventory surfaces are already strong:
  - `docs/README.md`
  - `docs/architecture/README.md`
  - `docs/audit/documentation-2026-05/DOCUMENTATION_INVENTORY.md`
  - `docs/audit/documentation-2026-05/CANONICAL_SOURCES.md`
  - `docs/audit/documentation-2026-05/REPORT_INDEX.md`
  - `docs/audit/documentation-2026-05/STALE_ARTIFACTS.md`
  - `docs/audit/REPO_MAP.md`
- The May 2026 audit already defines the right cleanup primitives:
  canonical sources, lane reports, stale-artifact classification, generated
  docs rules, and validation commands.
- `scripts/generate_feature_docs.py` is a prompt-driven generator from
  `KESTREL_FEATURES.md` to three audience docs:
  `FEATURES_developer.md`, `FEATURES_user.md`, and
  `FEATURES_investor.md`.
- `docs/generated/FEATURES_*.md` are stale generated views, not sources of
  truth. They currently use HTML comments for generation metadata rather than
  YAML frontmatter.
- `docs/audit/REPO_MAP.md` is a generated always-on context artifact, refreshed
  by `.github/workflows/repo-map.yml`.
- The sibling `kestrel-talon` project already has useful automation hooks:
  quality gates, integration gates, self-review, issue-declared verification
  gates, PRD batch mode, and fresh-context iteration loops.
- The sibling `kestrel-flight` project provides Playwright demo orchestration:
  narration, ordered screenshots, visual callouts, demo state, and
  dependency-aware demo configs.
- The sibling `kestrel-eye` project reviews screenshots against expectations
  and already integrates with Talon as an `--eye-check` quality gate.

## Target Shape

Use `docs/` as the OKF bundle root after migration.

Reserved files:

- `README.md` files should remain human directory indexes for now. OKF reserves
  `index.md`, but the current repo uses `README.md` everywhere. The migration
  should either generate sibling `index.md` files from each `README.md`, or
  eventually rename directory indexes only after link-impact review.
- `log.md` should be generated initially only at the bundle root and for high
  churn areas such as `docs/audit/`, `docs/generated/`, and
  `docs/architecture/`.

Every non-reserved markdown file should gain frontmatter:

```yaml
---
type: Architecture Spec
title: Signal Dispatcher
description: Runtime signal dispatch model and wake-source architecture.
resource: /docs/architecture/SIGNAL_DISPATCHER.md
tags: [architecture, signals, runtime]
timestamp: 2026-06-18T00:00:00Z
status: active
owner: core-runtime
canonical: true
generated: false
---
```

The OKF-required field is `type`. The Kestrel-specific extension fields should
be:

- `status`: `active`, `experimental`, `design-of-record`, `aspirational`,
  `historical`, `snapshot`, `generated`, `needs-revalidation`, or `private`.
- `owner`: package, subsystem, or lane owner.
- `canonical`: boolean for source-of-truth docs.
- `generated`: boolean for generated artifacts.
- `source`: source file or command for generated docs.
- `audience`: for audience views such as user/developer/investor docs.
- `privacy`: `public`, `internal`, `private`, or `review-before-public`.
- `supersedes` / `superseded_by`: bundle-relative links for archive hygiene.

## Concept Type Taxonomy

Use descriptive, unregistered OKF types. Start with a small taxonomy:

| Type | Example paths |
|---|---|
| `Documentation Index` | `docs/README.md`, directory indexes |
| `Architecture Spec` | `docs/architecture/**` |
| `Runbook` | `docs/deployment/README.md`, operational guides |
| `User Guide` | `docs/user-documentation/**` |
| `Design Note` | `docs/design/**`, `docs/concepts/**` |
| `Research Note` | `docs/research/**` |
| `Audit Ledger` | `docs/audit/DOCUMENTATION_AUDIT_5_2026.md` |
| `Audit Report` | `docs/audit/documentation-2026-05/reports/**` |
| `Issue Body` | `docs/audit/issues/**` |
| `Generated Reference` | `docs/generated/FEATURES_*.md`, `docs/audit/REPO_MAP.md` |
| `Demo Script` | `docs/demos/**`, demo narration transcripts |
| `Demo Evidence` | generated screenshots, transcripts, and eye reports |
| `Historical Snapshot` | `docs/archive/**`, old code reviews |
| `Business Document` | `docs/business/**` |
| `Legal Document` | `docs/legal/**` |
| `Outreach Document` | `docs/outreach/**` |

## Migration Phases

### Phase 0 - Establish OKF Tooling

Add deterministic tooling before editing hundreds of files:

- `scripts/docs_okf.py validate`: parse frontmatter, enforce non-empty `type`,
  report missing H1/title/description/status, tolerate broken links per OKF.
- `scripts/docs_okf.py inventory`: emit a machine-readable inventory from
  frontmatter plus path/H1.
- `scripts/docs_okf.py index`: generate `index.md` files from child
  frontmatter without replacing current `README.md` files.
- `scripts/docs_okf.py log`: update scoped `log.md` files from git changes or
  explicit change summaries.
- `scripts/docs_okf.py check-generated`: verify generated docs identify their
  source command and are up to date.

Add tests around the parser and validation rules before corpus migration.

### Phase 1 - Metadata Pilot

Pilot on a narrow, high-value subset:

- `docs/audit/documentation-2026-05/*`
- `docs/audit/OKF_MIGRATION_PLAN.md`
- `docs/generated/README.md`
- `docs/generated/FEATURES_developer.md`
- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_investor.md`
- `docs/README.md`
- `docs/architecture/README.md`

Definition of done:

- All pilot files have valid frontmatter.
- Generated `index.md` exists for the pilot directory or can be synthesized.
- Validator passes in CI.
- No generated feature doc is edited manually.

### Phase 2 - Canonical Architecture And Audit Docs

Migrate active source-of-truth documents next:

- `README.md`, `QUICKSTART.md`, `KESTREL_FEATURES.md`
- `docs/architecture/**`
- `docs/audit/*.md`
- `docs/audit/documentation-2026-05/**`
- `docs/deployment/README.md`
- `docs/guides/BUILDING_FEATURES.md`

This phase should also apply the May 2026 audit's findings: package-boundary
truth, signal/Talon truth, generated docs staleness, and stale architecture
status banners.

### Phase 3 - Public/User And Sensitive Docs

Migrate user-facing and public-release docs:

- `docs/user-documentation/**`
- `docs/use_cases/**`
- `docs/demos/**`
- `docs/design/launch/**`

Classify privacy explicitly. Business, legal, outreach, and strategy docs
should not silently appear as ordinary public concepts; mark them
`privacy: review-before-public` or `privacy: private` until redacted.

### Phase 4 - Archive, Research, Plans, And Diagrams

Finish the long tail:

- `docs/archive/**`
- `docs/research/**`
- `docs/plans/**`
- `docs/planning/**`
- `docs/diagrams/**`
- `docs/code_reviews/**`

Use `historical`, `snapshot`, or `aspirational` status aggressively. The goal
is not to pretend old docs are current; it is to make their status queryable.

## Generated Feature Docs

The current generator should be updated, not bypassed.

Current behavior:

- Reads all of `KESTREL_FEATURES.md`.
- Uses audience-specific LLM prompts.
- Writes `docs/generated/FEATURES_{audience}.md`.
- Uses HTML comments for generation metadata.
- Has tests for canonical source path, dry-run paths, and model default
  resolution.

Target behavior:

- Preserve `KESTREL_FEATURES.md` as the canonical feature inventory unless a
  later migration splits features into per-feature OKF concepts.
- Emit OKF frontmatter in each generated audience doc:
  - `type: Generated Reference`
  - `generated: true`
  - `source: /KESTREL_FEATURES.md`
  - `audience: developer|user|investor`
  - `generator: scripts/generate_feature_docs.py`
  - `model: provider/model`
  - `timestamp`
- Add `--check` mode that fails if checked-in generated docs are stale.
- Add `--metadata-only` or deterministic render mode for CI-safe validation
  without requiring LLM API keys.
- Add prompt constraints that tell the LLM to preserve OKF frontmatter exactly
  or, preferably, write frontmatter after generation deterministically in
  Python.
- Feed generated docs into `scripts/docs_okf.py validate`.

Longer-term option:

- Split `KESTREL_FEATURES.md` into feature concept documents under
  `docs/features/`.
- Generate `KESTREL_FEATURES.md` and audience docs from those concepts.
- This is more powerful, but it should wait until package-boundary truth is
  clean.

## Demo Documentation

Demo docs should be treated as executable documentation, not static prose.
Kestrel already has the sibling tools needed for this:

- `kestrel-flight`: Playwright demo/test orchestration with narration,
  numbered screenshots, highlights, state persistence, and dependency-aware
  configs.
- `kestrel-eye`: vision review of screenshots against explicit expectations,
  producing JSON/Markdown reports and Talon-compatible exit codes.

Target behavior:

- Keep `docs/demos/DEMO_SCRIPT.md` as a human narrative, but give it OKF
  metadata:
  - `type: Demo Script`
  - `status`
  - `owner`
  - `source_test`
  - `evidence`
  - `last_verified`
- Add a demo evidence directory, for example `docs/generated/demos/` or
  `docs/demo-evidence/`, containing generated transcripts, selected
  screenshots, and eye review reports.
- Mark generated demo evidence:
  - `type: Demo Evidence`
  - `generated: true`
  - `source: demos/<name>.demo.ts`
  - `generator: kestrel-flight`
  - `reviewer: kestrel-eye`
  - `timestamp`
- Keep large screenshot/video artifacts out of the normal docs corpus if they
  become heavy; store stable reports and selected thumbnails in docs, and keep
  bulky run artifacts in CI/build outputs.
- Connect user-facing docs that describe UI workflows to the demo evidence that
  proves the workflow still renders as claimed.

Suggested first demo slice:

- Create one canonical smoke demo for the Sovereign Console:
  identity, constitution, memory, features, security, and sovereignty tabs.
- Use `kestrel-flight` to generate narration and ordered screenshots.
- Use `kestrel-eye` to assert visible UI elements, layout, and absence of stale
  claims.
- Link the resulting report from `docs/demos/DEMO_SCRIPT.md` and relevant user
  docs.

## Talon Workflow Integration

Add OKF upkeep as a first-class gate beside review and testing.

Recommended Talon gates:

- Quality check:

```bash
kestrel-talon claim --repo KestrelSovereignAI/kestrel-sovereign --issue N \
  --backend codex --worktree \
  --quality-check "uv run python scripts/docs_okf.py validate" \
  --quality-check "uv run python scripts/generate_feature_docs.py --all --dry-run"
```

- Demo/visual check for docs or UI changes:

```bash
kestrel-talon claim --repo KestrelSovereignAI/kestrel-sovereign --issue N \
  --backend codex --worktree \
  --integration-check \
  --eye-check --eye-config eye-sovereign-console.toml \
  --quality-check "uv run python scripts/docs_okf.py validate"
```

- Issue-declared hard gate:

```talon-verify
uv run python scripts/docs_okf.py validate
uv run python scripts/generate_feature_docs.py --all --dry-run
empty: git diff -- docs/generated docs/audit/REPO_MAP.md
```

- Demo hard gate for changes that affect documented UI flows:

```talon-verify
npx playwright test --config=demo_config.cjs
kestrel-eye review --screenshot-dir demo-output
```

- PRD batch for the migration:

```json
{
  "stories": [
    {
      "id": "okf-0",
      "title": "Add OKF validation and inventory tooling",
      "description": "Create scripts/docs_okf.py with validate, inventory, index, and check-generated modes plus unit tests.",
      "done": false
    },
    {
      "id": "okf-1",
      "title": "Convert audit pilot docs to OKF frontmatter",
      "description": "Add frontmatter to the May 2026 audit workspace, generated docs README, and this OKF migration plan. Generate pilot indexes.",
      "done": false
    },
    {
      "id": "okf-2",
      "title": "Update generated feature doc creator for OKF metadata",
      "description": "Teach scripts/generate_feature_docs.py to write deterministic OKF frontmatter, add --check, and update tests.",
      "done": false
    },
    {
      "id": "okf-3",
      "title": "Wire OKF validation into CI and Talon quality guidance",
      "description": "Add CI validation and document Talon quality/verification gates for documentation updates.",
      "done": false
    },
    {
      "id": "okf-4",
      "title": "Add executable demo documentation evidence",
      "description": "Add a kestrel-flight Sovereign Console demo, kestrel-eye expectations, and OKF metadata linking demo docs to generated evidence.",
      "done": false
    },
    {
      "id": "okf-5",
      "title": "Migrate active architecture and audit source-of-truth docs",
      "description": "Add OKF metadata and status classifications to active architecture/audit docs while preserving existing links.",
      "done": false
    }
  ]
}
```

Talon should not be used to perform the entire corpus conversion in one issue.
Use small PRD stories and require OKF validation plus doc-generation dry runs
on each slice.

## CI And Automation

Add a docs workflow:

- On PR touching `docs/**`, `README.md`, `KESTREL_FEATURES.md`,
  `scripts/generate_feature_docs.py`, demo specs, eye configs, or
  `scripts/docs_okf.py`:
  - `uv run python scripts/docs_okf.py validate`
  - `uv run python scripts/generate_feature_docs.py --all --dry-run`
  - `uv run python scripts/generate_feature_docs.py --all --check`
    once deterministic check mode exists.
- On PR touching documented UI workflows:
  - run the relevant `kestrel-flight` demo Playwright config
  - run `kestrel-eye` against the generated screenshots
  - publish the Markdown/JSON review report as CI artifacts
  - update checked-in demo evidence only when the workflow intentionally
    changes.
- Nightly:
  - keep existing `scripts/generate_repo_map.py`
  - run OKF inventory generation
  - optionally run a small smoke demo plus Eye review for the main console
  - open or update a bot PR if generated indexes/logs drift

Do not require live LLM calls in ordinary CI. Live regeneration of audience docs
should remain manual, release-gated, or handled by a dedicated workflow with
credentials.

## Open Decisions

- Whether `docs/` itself is the OKF bundle root, or whether generated OKF lives
  under a parallel `docs/okf/` bundle. Recommendation: use `docs/` directly
  after a pilot because the repo already treats it as the knowledge corpus.
- Whether to rename `README.md` indexes to OKF `index.md`. Recommendation:
  generate `index.md` first; avoid mass renames until link impact is clear.
- Whether `KESTREL_FEATURES.md` remains a monolithic canonical concept.
  Recommendation: yes for the first migration; split later.
- Whether business/legal/outreach docs should remain in public repo. OKF makes
  their privacy classification queryable but does not solve publication risk.
- Whether generated `REPO_MAP.md` should become an OKF concept or a separate
  agent-context artifact. Recommendation: mark it `type: Generated Reference`
  and `status: snapshot`, but continue treating it as generated-only.
- Whether demo screenshots should be committed, stored as CI artifacts, or
  summarized into committed Markdown reports. Recommendation: commit lightweight
  reports and selected stable images only; keep bulky artifacts in CI storage.

## Recommended First Issue

The first Talon issue should stay Phase 0 sized when repeated in another repo:

1. Add `scripts/docs_okf.py`.
2. Add parser/validator tests.
3. Validate only files that already opt into OKF frontmatter.
4. Add this plan file as the first conformant concept.
5. Do not mass-edit the corpus yet.

That keeps a first PR reviewable and gives future Talon/documentation work a
hard gate before the larger migration begins. In this worktree, the migration
continued through the full `docs/` corpus after the pilot tooling proved out.

## Initial Worktree Implementation

Implemented in branch `codex/okf-docs-demo`:

- Added `scripts/docs_okf.py` with opt-in OKF validation, inventory output, and
  generated feature-doc metadata checks.
- Added OKF frontmatter to this plan, generated feature audience docs,
  `docs/generated/README.md`, and `docs/demos/DEMO_SCRIPT.md`.
- Updated `scripts/generate_feature_docs.py` to emit deterministic OKF
  frontmatter and support `--check` without live LLM credentials.
- Added `scripts/generate_demo_evidence_docs.py`, which scans existing
  `demos/*/demo.cjs` and `demos/*/eye.toml` files and generates
  `docs/generated/DEMO_EVIDENCE.md`.
- Added `.github/workflows/docs-okf.yml` for CI checks.
- Added `.kestreltalon/quality.yaml` so Talon can run the same OKF,
  generated-doc, and demo-evidence gates.
- Added `docs/audit/issues/okf-docs-demo-prd.json` so follow-up migration work
  can be driven through `kestrel-talon batch`.
- Converted the May 2026 documentation audit workspace
  (`docs/audit/documentation-2026-05/**`) to opt-in OKF frontmatter and added a
  regression test requiring that workspace to remain OKF-complete.
- Converted all non-reserved Markdown files under `docs/` to OKF frontmatter,
  added missing H1 headings where legacy issue/code-review notes lacked one,
  and added a regression test requiring `docs/` to remain OKF-complete.
- Tightened GitHub Actions and `.kestreltalon/quality.yaml` so the validation
  gate now runs `uv run python scripts/docs_okf.py validate --all docs`.
- Added `scripts/docs_okf.py index` and `scripts/docs_okf.py log`, generated
  `index.md` and `log.md` for `docs/audit`, `docs/generated`, and
  `docs/architecture`, and wired `--check` commands into CI/Talon gates.
- Linked user-facing workflow docs and demo docs to the generated demo evidence
  index plus the matching `kestrel-flight` demo scripts and `kestrel-eye`
  configs.
- Added focused tests:
  - `tests/unit/test_docs_okf.py`
  - `tests/unit/test_generate_feature_docs.py`
  - `tests/unit/test_demo_evidence_docs.py`

Verified locally:

```bash
uv run pytest tests/unit/test_docs_okf.py tests/unit/test_generate_feature_docs.py tests/unit/test_demo_evidence_docs.py tests/unit/test_feature_doc_canonicality.py -q
uv run python scripts/docs_okf.py validate --all docs
uv run python scripts/docs_okf.py validate
uv run python scripts/generate_feature_docs.py --check
uv run python scripts/generate_demo_evidence_docs.py --check
uv run python scripts/docs_okf.py index --check
uv run python scripts/docs_okf.py log --check
uv run python scripts/generate_feature_docs.py --all --dry-run
```
