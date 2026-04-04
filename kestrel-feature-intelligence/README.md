# kestrel-feature-intelligence

Metacognition and multi-model governance for Kestrel Sovereign agents. Provides two features: **Reflection** for agent self-improvement and insight generation, and **Council** for multi-model deliberation where foundation models reach consensus before irreversible actions.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-feature-intelligence.git
```

With optional integrations:

```bash
uv pip install "kestrel-feature-intelligence[github,wallet] @ git+https://github.com/KestrelSovereignAI/kestrel-feature-intelligence.git"
```

## Dependencies

- `kestrel-sovereign-sdk`
- `kestrel-sovereign>=0.1.8`
- Optional: `kestrel-feature-github` (via `[github]`), `kestrel-feature-wallet` (via `[wallet]`)

## Usage

Once installed, both `ReflectionFeature` and `CouncilFeature` are automatically discovered by kestrel-sovereign via the `kestrel_sovereign.features` entry point.

### Council Commands

- `!council-convene` — Deliberate on a question with multiple models
- `!council-status` — View council session history
- `!council-override` — Human override of council decisions
- `!council-members` — List configured council members
- `!council-evidence` — Preview evidence presented to council

## Configuration

No additional environment variables required. Uses the agent's configured LLM providers for council deliberation.

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e ".[test]"
uv run pytest
```
