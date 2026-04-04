# kestrel-feature-github — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel-feature-github/
├── pyproject.toml
├── README.md
├── SKILL.md
└── src/
    └── kestrel_feature_github/
        ├── __init__.py
        ├── feature.py         # GitHubFeature entry point
        ├── client.py          # GitHub API client
        ├── models.py          # Data models
        ├── ast_analyzer.py    # AST analysis utilities
        └── cache.py           # Response caching
```

## Entry Points

- `kestrel_sovereign.features`: `GitHubFeature = "kestrel_feature_github.feature:GitHubFeature"`

## Key Files to Read First

1. `src/kestrel_feature_github/feature.py` — Main feature class and tool registration
2. `src/kestrel_feature_github/client.py` — GitHub API client implementation
3. `SKILL.md` — Full skill/command reference

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Uses src layout — imports are from `kestrel_feature_github`, not `src.kestrel_feature_github`
- Requires `GITHUB_TOKEN` environment variable for API access
- API responses are cached via `cache.py` to reduce rate limiting
