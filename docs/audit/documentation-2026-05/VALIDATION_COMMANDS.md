# Validation Commands

Commands for the May 2026 documentation audit.

## Inventory

```bash
find docs -type f \( -name '*.md' -o -name '*.toml' -o -name '*.yml' -o -name '*.yaml' \) | sort
```

## Stale-Term Search

```bash
rg -n "core = true|feature_features|WorkflowsFeature|TalonCoordinatorFeature|mesh networking|features/|packages/|llm_config|RunPod|Vast|MCP|kestrel-feature|kestrel-cloud|kestrel-voice|kestrel-storage|kestrel-talon" README.md docs KESTREL_FEATURES.md kestrel_sovereign/data/feature_registry.toml
```

## Recent Docs-Affecting History

```bash
git log --oneline --since='2026-04-01' -- README.md docs pyproject.toml kestrel_sovereign/data/feature_registry.toml
```

## Generated Feature Docs

Run only after `KESTREL_FEATURES.md` is corrected:

```bash
uv run python scripts/generate_feature_docs.py --all
```

## Repo Map

```bash
uv run python scripts/generate_repo_map.py
```

