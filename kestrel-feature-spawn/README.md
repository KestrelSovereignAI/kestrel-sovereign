# kestrel-feature-spawn

Multi-agent spawning, delegation, and external bridge gateway for Kestrel Sovereign. **SpawnFeature** handles runtime child agent creation with scoped constitutions and task delegation. **BridgeFeature** provides external gateway integration for browser extensions, Discord, Slack, and other platforms.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-feature-spawn.git
```

With wallet delegation support:

```bash
uv pip install "kestrel-feature-spawn[wallet] @ git+https://github.com/KestrelSovereignAI/kestrel-feature-spawn.git"
```

## Dependencies

- `kestrel-sovereign-sdk`
- `kestrel-sovereign`
- Optional: `kestrel-feature-wallet` (via `[wallet]`)

## Usage

Once installed, both `SpawnFeature` and `BridgeFeature` are automatically discovered by kestrel-sovereign via the `kestrel_sovereign.features` entry point.

## Configuration

No additional environment variables required.

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e ".[test]"
uv run pytest
```
