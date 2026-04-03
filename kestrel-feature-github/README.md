# kestrel-feature-github

Kestrel feature: GitHub integration for issues, PRs, and repository management.

## Installation

```bash
pip install -e .
```

Once installed, the feature is automatically discovered by kestrel-sovereign via the
`kestrel_sovereign.features` entry point.

## Development

```bash
pip install -e ".[test]"
pytest
```

## Usage

After installation, GitHubFeature is available as a tool in any Kestrel agent.

See [SKILL.md](SKILL.md) for the full skill reference.
