# kestrel-sovereign-sdk

Lightweight interfaces for Kestrel Sovereign feature packages.

## Install

```bash
pip install kestrel-sovereign-sdk
```

## Usage

```python
from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.hooks.base import Hook, HookEvent
from kestrel_sdk.tools.base import AgentTool, ToolSchema
```

## What's included

This package contains only abstract base classes, protocols, and data models.
No heavy dependencies (FastAPI, database drivers, LLM clients, etc.).

Feature packages depend on this SDK instead of the full `kestrel-sovereign` framework.

### Modules

| Module | Contents |
|--------|----------|
| `kestrel_sdk.features.base` | `Feature`, `@tool` decorator, `parse_docstring_params` |
| `kestrel_sdk.hooks.base` | `Hook`, `HookEvent`, `HookInput`, `HookOutput`, `PermissionDecision` |
| `kestrel_sdk.tools.base` | `ToolCategory`, `ToolSchema`, `ToolParameter`, `AgentTool` |
| `kestrel_sdk.voice.base` | `TTSProvider`, `STTProvider`, `VoiceInfo`, `VoiceConfig` |
| `kestrel_sdk.storage.providers.base` | `StorageProvider`, `StorageTier`, `CryostasisCapable` |
| `kestrel_sdk.deploy.base` | `DeployProvider` |
| `kestrel_sdk.deploy.models` | `DeploymentProfile`, `DeployManagerError` |
| `kestrel_sdk.a2a.agent_card` | `AgentCard`, `AgentSkill`, `AgentCapabilities` |
| `kestrel_sdk.a2a.types` | `Task`, `TaskState`, `TaskStatus`, etc. |
| `kestrel_sdk.config.constants` | Timeout/interval constants |
| `kestrel_sdk.config.defaults` | Service URL and feature flag defaults |
| `kestrel_sdk.security.encryption` | Fernet helpers (requires `[crypto]` extra) |
| `kestrel_sdk.security.exceptions` | `SecurityError` hierarchy |

## License

Apache-2.0
