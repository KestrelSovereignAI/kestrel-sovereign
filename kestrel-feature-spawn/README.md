# kestrel-feature-spawn

Kestrel feature: multi-agent spawning, delegation, and external bridge gateway.

Includes:
- **SpawnFeature** -- runtime child agent spawning, task delegation, and lifecycle management
- **BridgeFeature** -- external gateway integration (browser extensions, Discord, Slack, etc.)

## Installation

```bash
pip install -e .
```

Once installed, the features are automatically discovered by kestrel-sovereign via the
`kestrel_sovereign.features` entry point.

## Development

```bash
pip install -e ".[test]"
pytest
```

## Usage

After installation, SpawnFeature and BridgeFeature are available as tools in any Kestrel agent.

See [SKILL.md](SKILL.md) for the full skill reference.
