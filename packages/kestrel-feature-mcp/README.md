# kestrel-feature-mcp

Model Context Protocol (MCP) server management and tool bridging for Kestrel Sovereign. Provides unified access to 311+ MCP servers via Docker MCP Toolkit, direct container management for custom servers, and a searchable catalog of available tools.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-feature-mcp.git
```

## Dependencies

- `kestrel-sovereign-sdk`
- `docker>=7.1.0`
- `mcp>=1.26.0`
- `trio>=0.32.0`
- `aiohttp>=3.13.3`

## Usage

Once installed, the `MCPAgent` feature is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.features` entry point.

### Commands

- `!mcp-gateway-start <servers>` — Start Docker MCP Gateway
- `!mcp-gateway-stop` — Stop gateway
- `!mcp-gateway-call <tool> <args>` — Call tool through gateway
- `!mcp-load <image>` — Load MCP server from Docker image
- `!mcp-list` — List running MCP servers
- `!mcp-search <query>` — Search MCP catalog
- `!mcp-catalog` — List all available servers

## Configuration

| Variable | Description |
|----------|-------------|
| `DOCKER_HOST` | Docker daemon socket (optional, defaults to local) |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e ".[test]"
uv run pytest
```
