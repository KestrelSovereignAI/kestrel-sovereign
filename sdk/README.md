# kestrel-sovereign-sdk

Lightweight SDK providing base interfaces, protocols, and utilities for Kestrel Sovereign feature package development. Feature packages depend on this SDK instead of the full framework, keeping dependencies minimal and development fast.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk.git
```

With encryption helpers:

```bash
uv pip install "kestrel-sovereign-sdk[crypto] @ git+https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk.git"
```

## Dependencies

- `pydantic>=2.0`
- Optional: `cryptography>=42.0` (via `[crypto]` extra)

## Usage

```python
from kestrel_sdk.features.base import Feature, Tool

class MyFeature(Feature):
    name = "my-feature"

    def get_tools(self):
        return [Tool(name="my-tool", description="Does something", handler=self.handle)]
```

## Configuration

No environment variables required. This is a development-time dependency only.

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
