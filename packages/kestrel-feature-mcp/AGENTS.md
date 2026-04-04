# kestrel-feature-mcp — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
packages/kestrel-feature-mcp/
├── pyproject.toml
├── README.md
├── kestrel_feature_mcp/
│   ├── __init__.py
│   ├── feature.py     # MCPAgent feature entry point
│   ├── gateway.py     # Docker MCP Gateway management
│   ├── manager.py     # Container lifecycle management
│   └── registry.py    # MCP server catalog/registry
```

## Entry Points

- `kestrel_sovereign.features`: `MCPAgent = "kestrel_feature_mcp.feature:MCPAgent"`

## Key Files to Read First

1. `kestrel_feature_mcp/feature.py` — Main feature class and tool registration
2. `kestrel_feature_mcp/gateway.py` — Docker MCP Gateway implementation
3. `kestrel_feature_mcp/registry.py` — Server catalog and search

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Requires Docker to be running for gateway and container modes
- MCP servers run as Docker containers — manage lifecycle carefully
- The registry provides a searchable catalog of 311+ available servers
