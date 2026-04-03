# kestrel-feature-mcp

Model Context Protocol (MCP) server management and tool bridging for Kestrel Sovereign.

## Installation

```bash
pip install kestrel-feature-mcp
```

Or as part of kestrel-sovereign:

```bash
pip install kestrel-sovereign[mcp]
```

## Features

- **Gateway Mode**: Unified access to 311+ MCP servers via Docker MCP Toolkit
- **Container Mode**: Direct Docker container management for custom MCP servers
- **Server Registry**: Searchable catalog of available MCP servers
- **Tool Bridging**: Call any MCP tool through a unified interface

## Dependencies

- `docker>=7.1.0`
- `mcp>=1.26.0`
- `trio>=0.32.0`
- `kestrel-sovereign>=0.1.8`

## Usage

This package registers as a Kestrel feature via entry points. Once installed,
the `MCPAgent` feature is automatically discovered and loaded by the agent.

### Commands

- `!mcp-gateway-start <servers>` - Start Docker MCP Gateway
- `!mcp-gateway-stop` - Stop gateway
- `!mcp-gateway-call <tool> <args>` - Call tool through gateway
- `!mcp-load <image>` - Load MCP server from Docker image
- `!mcp-list` - List running MCP servers
- `!mcp-search <query>` - Search MCP catalog
- `!mcp-catalog` - List all available servers
