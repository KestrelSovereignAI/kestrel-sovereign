# kestrel-feature-github

GitHub integration for Kestrel Sovereign agents — manage issues, pull requests, and repositories directly from agent conversations.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-feature-github.git
```

## Dependencies

- `kestrel-sovereign-sdk`
- `httpx`
- `pyyaml`
- `aiosqlite`

## Usage

Once installed, the `GitHubFeature` is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.features` entry point.

See [SKILL.md](SKILL.md) for the full skill reference.

## Configuration

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub personal access token with repo access |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e ".[test]"
uv run pytest
```
