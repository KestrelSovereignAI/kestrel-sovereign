# Dev Container Quick Start

Get a complete Kestrel + Kestrel development environment running in Docker Desktop with one click.

## 🎯 What You Get

A fully configured development environment with:
- ✅ Python 3.11 + all dependencies installed
- ✅ PostgreSQL 15 with pgvector (port 5433)
- ✅ Redis 7 (port 6380)
- ✅ VS Code extensions pre-installed
- ✅ Git configured with your identity
- ✅ Database migrations run automatically
- ✅ No conflicts with your host machine

## 📋 Prerequisites

1. **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)
   - Start Docker Desktop and ensure it's running

2. **VS Code** - [Download here](https://code.visualstudio.com/)
   - Install the "Dev Containers" extension:
     - Open VS Code
     - Press `Cmd+Shift+X` (Mac) or `Ctrl+Shift+X` (Windows/Linux)
     - Search for "Dev Containers"
     - Install the extension from Microsoft

3. **Project cloned locally**
   ```bash
   git clone https://github.com/Kestrel-Sovereign-AI/kestrel.git
   cd kestrel
   ```

## 🚀 Quick Start (3 Steps)

### Step 1: Stop Local Services (if running)
```bash
# From the kestrel directory
cd kestrel
./stop_server.sh
docker-compose down
```

### Step 2: Open in VS Code
```bash
code .
```

### Step 3: Reopen in Container
1. Click the **green button** in the bottom-left corner of VS Code
2. Select **"Reopen in Container"**
3. Wait 5-10 minutes for first-time setup

**That's it!** ✨

## ⏱️ What Happens During Setup

### First Time (~5-10 minutes)
1. **Build Docker image** (~3-5 min)
   - Installs Python 3.11, Node.js, system tools
   - Installs uv, pytest, black, etc.

2. **Start services** (~1 min)
   - PostgreSQL container starts
   - Redis container starts

3. **Post-create setup** (~2-4 min)
   - Waits for services to be ready
   - Creates Python virtual environment
   - Installs Kestrel dependencies
   - Installs Kestrel dependencies
   - Runs database migrations
   - Sets up Git config

### Subsequent Starts (~30 seconds)
Everything is cached, so reopening the container is fast!

## ✅ Verify It's Working

Once the container is ready, open a terminal in VS Code:

```bash
# Check services
curl http://localhost:7777/health
# Should return: {"status":"ok"}

# Check PostgreSQL
psql $DATABASE_URL -c "SELECT version();"

# Check Redis
redis-cli -h redis -a redis_password_2024 ping
# Should return: PONG

# Run tests
pytest tests/ -v
```

## 🛠️ Common Tasks

### Start Kestrel Server
```bash
cd /workspace/kestrel
./start_server.sh

# Access at: http://localhost:7777
```

### Run Tests
```bash
pytest tests/                    # All tests (fail-fast)
pytest tests/integration/        # Integration tests only
pytest --cov                     # With coverage
```

### Database Access
```bash
# PostgreSQL
psql $DATABASE_URL

# Redis
redis-cli -h redis -a redis_password_2024
```

### Python Development
```bash
# Interactive Python
ipython

# Kestrel CLI
python main.py agent_data/kestrel_prime.db

# Run a script
python your_script.py
```

## 🔧 Environment Variables

### Already Configured
```bash
DATABASE_URL=postgresql://kestrel_user:kestrel_password_2024@postgres:5432/kestrel
REDIS_URL=redis://:redis_password_2024@redis:6379
PYTHONPATH=/workspace:/workspace/kestrel
```

### Add Your API Keys
Create `/workspace/kestrel/.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here
REPLICATE_API_TOKEN=r8_your-token-here
```

## 📁 File Structure

```
/workspace/                    # Your project root (synced with host)
├── .devcontainer/            # Dev container config
├── agent_data/               # Kestrel agent databases (persistent volume)
├── kestrel/                    # Kestrel application
│   ├── logs/                 # Application logs (persistent volume)
│   ├── .env                  # Your local config (create this)
│   └── server.py             # FastAPI server
├── storage/                  # Kestrel storage modules
├── llm/                      # Kestrel LLM modules
└── tests/                    # Test suite
```

## 🐛 Troubleshooting

### "Container failed to start"
```bash
# Check Docker Desktop is running
docker ps

# View logs
docker-compose -f .devcontainer/docker-compose.devcontainer.yml logs
```

### "Port already in use"
Stop local services on your host machine:
```bash
# Kill process on port 7777
lsof -ti:7777 | xargs kill -9

# Stop docker containers
cd kestrel && docker-compose down
```

### "PostgreSQL connection failed"
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check health
docker exec -it kestrel-dev-postgres pg_isready -U kestrel_user
```

### "Python dependencies missing"
```bash
# In container terminal
source /workspace/.venv_kestrel/bin/activate
cd /workspace
uv pip install -e .
cd /workspace/kestrel
uv sync
```

### "Rebuild from scratch"
In VS Code:
1. `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type: `Dev Containers: Rebuild Container`
3. Select "Rebuild Without Cache"

## 🔄 Exiting Dev Container

### Keep Container Running (Recommended)
Just close VS Code. The container keeps running in the background.

### Stop Container
1. `Cmd+Shift+P` → "Dev Containers: Reopen Folder Locally"
2. This reopens the folder on your host machine

### Remove Everything
```bash
# Stop and remove containers
docker-compose -f .devcontainer/docker-compose.devcontainer.yml down

# Remove persistent volumes (⚠️ DELETES DATA)
docker volume rm kestrel-venv kestrel-agent-data kestrel-logs postgres-data redis-data
```

## 📚 Learn More

- Full documentation: [.devcontainer/README.md](.devcontainer/README.md)
- Docker status: See root [AGENTS.md](AGENTS.md) or [PROJECT_STATUS.md](PROJECT_STATUS.md)
- VS Code Dev Containers: https://code.visualstudio.com/docs/devcontainers/containers

## 🆘 Get Help

If something isn't working:
1. Check [.devcontainer/README.md](.devcontainer/README.md) troubleshooting section
2. View container logs: `docker-compose -f .devcontainer/docker-compose.devcontainer.yml logs`
3. Rebuild container: `Cmd+Shift+P` → "Rebuild Container"

---

**Enjoy your containerized development environment! 🎉**
