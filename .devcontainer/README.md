# Kestrel Dev Container

This directory contains the Docker-based development container configuration for Kestrel + Kestrel development.

## What's Included

The dev container provides a complete development environment with:

### Services
- **Python 3.11** development environment
- **PostgreSQL 15** with pgvector extension (port 5433)
- **Redis 7** for caching and sessions (port 6380)
- **Ollama** (optional - for local LLM, requires GPU)

### Pre-installed Tools
- **Python tools**: uv, pytest, black, pylint, mypy, ipython
- **Node.js 20**: npm, yarn, pnpm, playwright
- **Database clients**: psql, redis-cli
- **Development utilities**: git, docker, curl, jq, httpie

### VS Code Extensions
- Python (with Pylance)
- Docker
- GitLens
- GitHub Copilot
- YAML, TOML support
- And more...

## Getting Started

### Prerequisites
1. **Docker Desktop** installed and running
2. **VS Code** with "Dev Containers" extension installed
3. Git repository cloned locally

### Open in Dev Container

#### Option 1: VS Code Command Palette
1. Open the Kestrel project in VS Code
2. Press `F1` or `Cmd+Shift+P` (Mac) / `Ctrl+Shift+P` (Windows/Linux)
3. Type: `Dev Containers: Reopen in Container`
4. Wait for container to build and initialize (~5-10 minutes first time)

#### Option 2: VS Code UI
1. Open the project in VS Code
2. Click the green button in the bottom-left corner
3. Select "Reopen in Container"

#### Option 3: Command Line
```bash
# From the project root
code .
# Then use Command Palette as in Option 1
```

### First Build
The first time you open the dev container:
- Docker will build the development image (~3-5 minutes)
- Services (PostgreSQL, Redis) will start
- Post-create script will:
  - Wait for services to be ready
  - Create Python virtual environment
  - Install Kestrel + Kestrel dependencies
  - Set up Git config
  - Run database migrations
  - Create necessary directories

**Total time:** ~5-10 minutes

Subsequent launches are much faster (~30 seconds).

## Environment Variables

The dev container automatically configures:
```bash
PYTHONPATH=/workspace:/workspace/kestrel
KESTREL_DB_PATH=/workspace/agent_data
DATABASE_URL=postgresql://kestrel_user:kestrel_password@postgres:5432/kestrel
REDIS_URL=redis://:redis_password_2024@redis:6379
```

### Adding API Keys
Create or edit `/workspace/kestrel/.env`:
```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
REPLICATE_API_TOKEN=r8_...
```

Or set them in your host `.bashrc`/`.zshrc` before launching VS Code.

## Working in the Dev Container

### Terminal Access
Open integrated terminal in VS Code (`Ctrl+\`` or `Cmd+\``). You'll be in `/workspace`.

### Quick Commands
```bash
# Navigate
k          # cd /workspace (Kestrel root)
f          # cd /workspace/kestrel

# Start services
cd /workspace/kestrel
kestrel start    # Start Kestrel API on port 7777

# Run tests
pytest tests/                           # All tests (fail-fast)
pytest tests/integration/test_auth_e2e.py -v  # Specific test
pytest --cov                            # With coverage

# Database
psql $DATABASE_URL                     # Connect to PostgreSQL
redis-cli -h redis -a redis_password_2024  # Connect to Redis

# Python REPL
ipython                                # Enhanced Python shell
python main.py agent_data/kestrel.db   # Kestrel CLI
```

### File Synchronization
All files in `/workspace` are **live-synced** with your host machine. Edit files in VS Code and they update in real-time.

### Persistent Data
These volumes persist across container rebuilds:
- `kestrel-venv`: Python virtual environment
- `kestrel-agent-data`: Agent databases and files
- `kestrel-logs`: Application logs
- `postgres-data`: PostgreSQL database
- `redis-data`: Redis cache

To reset everything:
```bash
docker volume rm kestrel-venv kestrel-agent-data kestrel-logs postgres-data redis-data
```

## Port Forwarding

Ports automatically forwarded to your host machine:
| Service | Container Port | Host Port | URL |
|---------|---------------|-----------|-----|
| Kestrel API | 8080 | 7777 | http://localhost:7777 |
| PostgreSQL | 5432 | 5433 | localhost:5433 |
| Redis | 6379 | 6380 | localhost:6380 |
| Ollama | 11434 | 11434 | http://localhost:11434 |

Access from your **host machine** browser at http://localhost:7777

## Running Services

### Start Kestrel Server
```bash
cd /workspace/kestrel
kestrel start
```

### Start Kestrel Agent
```bash
cd /workspace
python main.py agent_data/kestrel_prime.db
```

### Check Service Health
```bash
curl http://localhost:7777/health
```

## Database Management

### PostgreSQL
```bash
# Connect
psql postgresql://kestrel_user:kestrel_password@postgres:5432/kestrel

# From host machine
psql postgresql://kestrel_user:kestrel_password@localhost:5433/kestrel

# Run migrations
cd /workspace/kestrel
alembic upgrade head

# Reset database
./scripts/rebuild_db.sh
```

### Redis
```bash
# Connect
redis-cli -h redis -a redis_password_2024

# From host machine
redis-cli -h localhost -p 6380 -a redis_password_2024

# Flush all data
redis-cli -h redis -a redis_password_2024 FLUSHALL
```

## Troubleshooting

### Container won't start
```bash
# Check Docker Desktop is running
docker ps

# Check logs
docker-compose -f .devcontainer/docker-compose.devcontainer.yml logs

# Rebuild container
# In VS Code: Cmd+Shift+P -> "Dev Containers: Rebuild Container"
```

### Services not ready
```bash
# Check PostgreSQL
docker exec -it kestrel-dev-postgres pg_isready -U kestrel_user

# Check Redis
docker exec -it kestrel-dev-redis redis-cli -a redis_password_2024 ping
```

### Python dependencies missing
```bash
# Reinstall in container terminal
cd /workspace
source .venv_kestrel/bin/activate
uv pip install -e .

cd /workspace/kestrel
uv sync
```

### Port already in use
Stop local services before opening dev container:
```bash
# From host machine
cd ./kestrel
kestrel stop
docker-compose down
```

## Customization

### Add VS Code Extensions
Edit `.devcontainer/devcontainer.json`:
```json
"extensions": [
  "ms-python.python",
  "your.extension.id"
]
```

### Add System Packages
Edit `.devcontainer/Dockerfile.devcontainer`:
```dockerfile
RUN apt-get update && apt-get install -y \
    your-package-name
```

### Change Python Version
Edit `.devcontainer/Dockerfile.devcontainer`:
```dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.12
```

## Ollama (Optional GPU Support)

Ollama is configured but disabled by default (requires GPU).

### Enable Ollama
```bash
# Start with GPU profile
docker-compose -f .devcontainer/docker-compose.devcontainer.yml --profile gpu up -d ollama

# Pull a model
docker exec -it kestrel-dev-ollama ollama pull llama3.2:3b

# Test
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Hello"
}'
```

## Best Practices

1. **Always use the virtual environment**: It's activated by default in the container
2. **Run tests before committing**: `pytest -x tests/`
3. **Use fail-fast testing**: The `-x` flag stops at first failure
4. **Check service health**: `curl http://localhost:7777/health`
5. **Keep containers running**: Don't stop PostgreSQL/Redis during development

## Differences from Local Development

| Aspect | Local | Dev Container |
|--------|-------|---------------|
| Port 7777 | Direct | Forwarded from 8080 |
| Database | localhost:5433 | postgres:5432 |
| Redis | localhost:6380 | redis:6379 |
| Python env | .venv_kestrel | /workspace/.venv_kestrel (volume) |
| Git config | Your host config | Set in container |

## Exiting Dev Container

1. **Keep container running**: Just close VS Code
2. **Stop container**: `Dev Containers: Reopen Folder Locally`
3. **Remove containers**:
   ```bash
   docker-compose -f .devcontainer/docker-compose.devcontainer.yml down
   ```

## Resources

- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Dev Container Specification](https://containers.dev/)

---

**Happy coding! 🚀**
