---
type: Guide
title: Getting Started with Kestrel
description: This guide will help you quickly get up and running with the Kestrel
  sovereign AI agent framework.
resource: /docs/GETTING_STARTED.md
tags:
- docs
- guide
timestamp: '2026-06-18T00:00:00Z'
status: active
owner: documentation
canonical: false
generated: false
privacy: public
---

# Getting Started with Kestrel

This guide will help you quickly get up and running with the Kestrel sovereign AI agent framework.

> Canonical setup instructions live in `../README.md`.
> This document is supplementary (quick commands, ops notes, and deeper usage).

## 🏁 Quick Setup for New Sessions

### 1. Environment Setup
```bash
# Navigate to project
cd /path/to/kestrel

# Create a clean virtual environment using uv
uv venv .venv_kestrel

# Activate the new environment
source .venv_kestrel/bin/activate

# Install dependencies in editable mode
# (This will also install pip if it's not already in the venv)
.venv_kestrel/bin/python -m uv pip install -e .
```

### 2. Verify Core Functionality
```bash
# Run the health check script from the virtual environment
.venv_kestrel/bin/python health_check.py

# Test storage system
.venv_kestrel/bin/python -m pytest test_storage.py -x -v

# Check existing agents
ls -la *.db
```

### 3. Configure LLM Providers

The simplest path is the setup wizard, which writes a unified `kestrel.toml`:

```bash
kestrel setup llm
# Or run all wizard steps:
kestrel setup
```

If you prefer to hand-edit, copy the unified template and edit the `[llm]` section:

```bash
cp kestrel.toml.example kestrel.toml
nano kestrel.toml   # edit the [llm] table
```

**Migrating from a legacy `llm_config.toml`?** Run the one-shot migration:

```bash
kestrel migrate-llm-config
```

This merges your existing `llm_config.toml` into `kestrel.toml [llm]`, renames
the source to `llm_config.toml.bak`, and backs up any prior `kestrel.toml`. It
is idempotent — re-running it after a successful migration is a no-op. See
issue #938 for the full deprecation plan.

**Key Configuration Points:**
- **Ollama**: Should work out of the box if running on localhost:11434
- **OpenAI**: Set `OPENAI_API_KEY` environment variable or edit config file
- **Route Priority**: Ollama first for privacy, OpenAI as fallback

### 4. Start Chatting with Existing Agent
```bash
# Use existing agent (in-package module; works from a pip install)
python -m kestrel_sovereign.main kestrel_prime.db
# Source-clone shorthand: `python main.py kestrel_prime.db` still works via the
# repo-root re-export shim.

# Available commands in chat:
# !help     - Show available commands
# !status   - Show agent and privacy status
# !quit     - Exit the interface
# !anchor   - Create cryptographic anchor
# !create-agent <name> - Create new trusted agent
# !export-sovereignty - Export agent state to IPFS/Filecoin
# !import-sovereignty <cid> - Import agent state from IPFS
# !send-mail "<addr>" "<path>" - Send physical mail
# !privacy <mode> - Set privacy mode (ephemeral, isolated, anonymous, normal)
# !model-mandate - Configure LLM preferences
# 
# Physical Interaction Commands (MOCK):
# !notarize-physical <file_path> - Send a document for physical notarization
# !check-mail     - Check for new incoming physical mail
# 
# Privacy Mode Commands:
# !privacy ephemeral    - Go off-the-record (nothing stored, local LLM only)
# !privacy isolated     - Temporary session (user controls saving)
# !privacy anonymous    - Remove PII but preserve learning
# !privacy normal       - Standard persistent storage
# !privacy-status       - Show detailed privacy report
# !privacy-save         - Save isolated session
# !privacy-discard      - Discard isolated session
```

### 5. Create New Agent (if needed)
```bash
# Run inception service
python inception_service.py

# Follow the prompts to create a new agent
```

## 🔒 Privacy Modes - Complete Data Sovereignty

Kestrel gives you complete control over what gets stored and how your data is processed. You can go "off the record" anytime:

### Available Privacy Modes

**🔄 Normal Mode** (`!privacy normal`)
- Standard operation - conversations stored permanently
- Full memory anchoring and learning capabilities
- Default mode for most interactions

**👻 Ephemeral Mode** (`!privacy ephemeral`)
- **True off-the-record** - nothing stored anywhere
- **Local LLM Only** - Forces use of local models (Ollama) for maximum privacy
- Agent can still access existing knowledge for context
- Perfect for sensitive topics, brainstorming, personal matters
- No memory anchoring possible

**🏝️ Isolated Mode** (`!privacy isolated`)
- Conversations stored in temporary session only
- You decide what (if anything) becomes permanent
- Use `!privacy-save` or `!privacy-discard` to control data
- Perfect for complex analysis that might need context. Ideal for human-led storytelling (elderly) or intimate sharing (romantic) before committing to permanence.

**🎭 Anonymous Mode** (`!privacy anonymous`)
- Conversations stored but with PII automatically removed
- Preserves learning while protecting identity
- Emails, SSNs, phone numbers replaced with placeholders
- Still allows memory anchoring for integrity

### Quick Privacy Demo

```bash
# Test privacy modes
python test_privacy_demo.py
```

### Privacy Commands in Chat

```
!privacy-status - Show detailed privacy report
!privacy ephemeral - Switch to ephemeral (off-the-record) mode
!privacy isolated - Switch to isolated session mode
!privacy-save   - Save isolated session to permanent storage
!privacy-discard - Discard isolated session without saving
!privacy anonymous - Switch to anonymous mode (PII removed)
!privacy normal - Return to standard mode
```

### Privacy Mode Philosophy

**Sovereignty means YOU control:**
- What gets stored vs. ephemeral
- What gets learned vs. forgotten  
- What stays local vs. external processing
- What becomes permanent vs. temporary
Human stories persist only as intended, preventing loss while enabling ethical AI evolution.

This is true data sovereignty - not just promises, but cryptographically enforced control.

## 🔧 Troubleshooting

### Common Issues

**1. "Ollama not responding"**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Pull a model if none available
ollama pull phi3:latest
```

**2. "OpenAI API errors"**
```bash
# Set API key (optional - Ollama is primary)
export OPENAI_API_KEY="your-key-here"

# Or disable OpenAI in kestrel.toml under [llm]
# route_priority = ["ollama:local"]
```

**3. "Database not found"**
```bash
# Check existing databases
ls -la *.db

# Create new agent if none exist
python inception_service.py
```

**4. "Import errors"**
```bash
# Ensure your virtual environment is active
source .venv_kestrel/bin/activate

# Re-install dependencies in editable mode
.venv_kestrel/bin/python -m uv pip install -e .

# Check Python version (needs 3.11+)
.venv_kestrel/bin/python --version
```

## 📊 Health Check Script

Run the health check script to verify your setup:

```bash
.venv_kestrel/bin/python health_check.py
```

## 🎯 Typical Workflow

### For Development/Testing
1. Run health check: `.venv_kestrel/bin/python health_check.py`
2. Test with existing agent: `.venv_kestrel/bin/python -m kestrel_sovereign.main kestrel_prime.db`
3. Run specific tests: `.venv_kestrel/bin/python -m pytest test_kestrel_agent.py -x`

### For New Features
1. Create feature branch
2. Run tests: `python -m pytest -x`
3. Test with agent: Chat and verify functionality
4. Update documentation in `features/` directory

### For Demos
1. Ensure Ollama is running with a fast model (phi3:latest)
2. Use existing agent with interesting conversation history
3. Demonstrate key features: memory, reasoning, anchoring

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `kestrel_sovereign/main.py` | Interactive chat interface (root `main.py` is a re-export shim for source clones) |
| `kestrel_agent.py` | Core agent logic |
| `storage.py` | Memory and knowledge management |
| `kestrel.toml` | Unified config (LLM, agents, features). `[llm]` section holds provider config. |
| `inception_service.py` | New agent creation |
| `test_*.py` | Test suites |
| `features/*.md` | Detailed documentation |

## 🔄 Next Steps

After getting started:

1. **Explore Features**: Read `features/` documentation
2. **Test Multi-Model**: Try both Ollama and OpenAI
3. **Create Agents**: Use inception service for new agents
4. **Review Architecture**: Understand the sovereignty model
5. **Contribute**: Check the open issues and project board for current priorities.

---

**Need Help?** Check the project's open issues for current known problems and priorities.