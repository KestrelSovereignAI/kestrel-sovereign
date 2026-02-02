# Kestrel Control Panel

Multi-agent orchestrator for power users. Discover, start, stop, and manage multiple Kestrel sovereign agents from a single dashboard.

## Quick Start

```bash
# From the kestrel-sovereign root directory
uv run python control-panel/server.py
```

Then open: http://localhost:8899

## Features

- **Agent Discovery**: Automatically scans `agent_data/*/` for valid agents
- **Lifecycle Management**: Start/stop agents on auto-assigned ports (8900+)
- **Status Dashboard**: See which agents are running, their ports, DIDs
- **Direct Links**: One-click to open an agent's chat UI

## Architecture

```
control-panel/
├── server.py      # FastAPI orchestrator
├── index.html     # Dashboard UI
├── README.md
└── *.log          # Per-agent logs (auto-created)
```

The control panel runs on port **8899** and manages agent instances on ports **8900+**.

Each agent runs as a separate process with its own Kestrel server instance.

## API

- `GET /api/agents` - List all discovered agents
- `POST /api/agents/{id}/start` - Start an agent
- `POST /api/agents/{id}/stop` - Stop an agent
- `GET /api/agents/{id}/logs` - Get agent logs

## Notes

- Agents are discovered by scanning for `kestrel_prime.db` files in `agent_data/*/`
- Running agents are tracked in memory (restarting control panel loses track of running agents)
- Logs are written to `control-panel/{agent_id}.log`
