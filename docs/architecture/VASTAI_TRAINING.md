# VastAI LoRA Training Architecture

> **Scope clarification (2026-04-25):** this doc is specifically about *VastAI as a training backend* and that effort is deprioritized — see status banner below. VastAI **as a general compute provider** is a separate, active feature: `features/vastai/` ships with 29 tests and is exercised by the `manage_vastai` tool. If you're looking for the general-purpose VastAI integration, that's not this document.

## Overview

VastAI is a peer-to-peer GPU marketplace where individual providers rent out their GPUs. This document covers using VastAI for FLUX.2 LoRA training with the SimpleTuner HTTP API.

## Current Status: ⏸️ DEPRIORITIZED (December 2025)

**VastAI GCR authentication not yet solved - using other providers instead.**

The VastAI SDK's `login` parameter does not correctly handle GCR (Google Container Registry) service account JSON keys. All authentication formats fail with "docker login failed!" error.

**Working alternatives:**
- ✅ **Google Vertex AI** - Fully working for FLUX.2 LoRA training
- 🔄 **RunPod** - Nearly complete, uses template-based GCR auth

### Formats Tested (All Failed)

```bash
# Format 1: Raw JSON content
-u _json_key -p '{"type":"service_account",...}' gcr.io

# Format 2: Shell-quoted JSON
-u _json_key -p '<shlex.quote(json)>' gcr.io

# Format 3: Base64 encoded key
-u _json_key_base64 -p '<base64_key>' gcr.io

# Format 4: Base64 with https registry
-u _json_key -p '<base64_key>' https://gcr.io
```

### Root Cause

The VastAI SDK passes the `login` parameter to `image_login` in the API. The remote machines then attempt to parse this string and run `docker login`. The JSON service account key contains special characters (quotes, colons, newlines) that break the parsing on the remote machine.

### Comparison with RunPod

RunPod solves this problem with **templates**:
- Create a template in the RunPod web UI
- Configure registry credentials in the UI (not via CLI string parsing)
- Reference template via `template_id` when creating pods
- The template's credentials are securely stored and applied

VastAI has a `create_template` command but it uses the same broken `login` string parsing.

## Workaround Options

### Option 1: Use Vertex AI or RunPod (Recommended)

Both have working GCR authentication:
- **Vertex AI**: Native GCP integration, no auth issues
- **RunPod**: Template-based GCR auth via `template_id`

```python
# RunPod config with template_id for GCR auth
[profiles.training]
template_id = "YOUR_TEMPLATE_ID"  # Create template with GCR credentials
image_name = "gcr.io/YOUR_PROJECT_ID/simpletuner-flux2:latest"
```

### Option 2: Push to Docker Hub

Push the SimpleTuner image to Docker Hub and use Docker Hub credentials:

```bash
# Tag and push to Docker Hub
docker tag gcr.io/YOUR_PROJECT_ID/simpletuner-flux2:latest \
    yourusername/simpletuner-flux2:latest
docker push yourusername/simpletuner-flux2:latest

# VastAI login format for Docker Hub (works)
-u yourusername -p yourpassword docker.io
```

### Option 3: Use Public Images

Use a public Docker Hub image (no authentication needed):

```toml
[profiles.training]
image_name = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
onstart_cmd = "pip install simpletuner && python -m simpletuner.api --port 8000"
```

## Architecture (When Working)

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     VastAI Marketplace                       │
├─────────────────────────────────────────────────────────────┤
│  Provider GPUs (peer-to-peer)                               │
│  - A100 80GB: ~$0.74-1.50/hr                                │
│  - H100 80GB: ~$1.50-2.50/hr                                │
│  - RTX 4090:  ~$0.40-0.60/hr                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   VastAI Manager                             │
│  features/vastai/vastai_manager.py                          │
├─────────────────────────────────────────────────────────────┤
│  - search_offers(): Find available GPUs                     │
│  - start_session(): Create instance                         │
│  - wait_for_api_ready(): Poll /ready endpoint               │
│  - submit_training_job_http(): POST /train                  │
│  - poll_training_status_http(): GET /status/{job_id}        │
│  - download_lora_http(): GET /download/{job_id}             │
│  - generate_image_http(): POST /generate/async              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              SimpleTuner Docker Container                    │
│  gcr.io/YOUR_PROJECT_ID/simpletuner-flux2:latest              │
├─────────────────────────────────────────────────────────────┤
│  HTTP API Endpoints:                                         │
│  - GET  /health              Health check                   │
│  - GET  /ready               Readiness (model loaded)       │
│  - POST /train               Start training (multipart)     │
│  - GET  /status/{job_id}     Training progress              │
│  - GET  /download/{job_id}   Download LoRA weights          │
│  - POST /generate/async      Start generation               │
│  - GET  /generate/status/{id} Generation progress           │
│  - GET  /loras               List available LoRAs           │
└─────────────────────────────────────────────────────────────┘
```

### Training Flow

1. **Search Offers**: Find GPU matching requirements (80GB+ VRAM, reliability > 0.85)
2. **Create Instance**: Rent GPU with SimpleTuner container
3. **Wait for Ready**: Poll `/ready` until model is loaded (~5-10 min first run)
4. **Submit Training**: POST multipart form to `/train` with avatar image
5. **Poll Status**: GET `/status/{job_id}` until `completed`
6. **Download LoRA**: GET `/download/{job_id}` for safetensors file
7. **Generate Images**: POST `/generate/async` with trained LoRA
8. **Cleanup**: Destroy instance to stop billing

### Key Differences from RunPod

| Feature | VastAI | RunPod |
|---------|--------|--------|
| Pricing Model | Peer-to-peer marketplace | Fixed pricing |
| Pod Resume | Not supported | Supported (pause/resume) |
| Network Volumes | Local only (tied to machine) | Persistent across pods |
| Private Registry | Broken for GCR | Works via templates |
| Reliability | Variable by provider | More consistent |
| GPU Availability | Higher (many providers) | Lower (limited capacity) |

## Configuration

### vastai_config.toml

```toml
[manager]
default_ttl_seconds = 3600
max_ttl_seconds = 7200
poll_interval_seconds = 15
readiness_timeout_seconds = 600

[profiles.training]
name = "A100 80GB - FLUX.2 LoRA Training"
task_type = "training"
image_name = "gcr.io/YOUR_PROJECT_ID/simpletuner-flux2:latest"  # BLOCKED
disk_gb = 150
gpu_ram_min = 80
num_gpus = 1
reliability_min = 0.85
compute_cap_min = 800
cuda_vers_min = 12.0
ports = ["8000/http"]
cost_per_hr_max = 1.50
onstart_cmd = "python /app/simpletuner_api.py --port 8000"
```

### Environment Variables

```bash
VASTAI_API_KEY=<your-api-key>
GCR_SERVICE_ACCOUNT_KEY_FILE=/path/to/key.json  # Not working
HF_TOKEN=<huggingface-token>  # For gated FLUX.2-dev model
```

## Code Examples

### VastAI Manager Usage

```python
from features.vastai.vastai_manager import VastAIManager

manager = VastAIManager()

# Start training session
await manager.start_session(
    task_profile="training",
    ttl_seconds=3600,
    metadata={"companion_id": "test-123"}
)

session = manager._session

# Wait for API ready
await manager.wait_for_api_ready(session, timeout=600)

# Submit training job
job_id = await manager.submit_training_job_http(
    session=session,
    avatar_data=image_bytes,
    companion_id="test-123",
    trigger_word="TOKtest123",
    steps=500,
    lora_rank=16,
)

# Poll for completion
while True:
    status = await manager.poll_training_status_http(session, job_id)
    if status["status"] == "completed":
        break
    await asyncio.sleep(15)

# Download LoRA
lora_bytes = await manager.download_lora_http(session, job_id)

# Generate image
result = await manager.generate_image_http(
    session=session,
    prompt="a portrait of TOKtest123 smiling",
    lora_path=status["lora_path"],
    trigger_word="TOKtest123",
)
```

### VastAI Adapter Usage

```python
from features.training.adapters.vastai_adapter import VastAITrainingAdapter
from features.training.types import TrainingConfig

adapter = VastAITrainingAdapter()

# Check availability
if not adapter.is_available():
    raise RuntimeError("VastAI not configured")

# Start training
job = await adapter.start_training(
    companion_id="companion-uuid",
    avatar_data=image_bytes,
    config=TrainingConfig(steps=500, lora_rank=16),
)

# Poll status
status = await adapter.get_status(job.job_id)

# Download weights
lora_bytes = await adapter.download_weights(job.job_id)

# Cleanup (stop billing)
await adapter.cleanup(job.job_id)
```

## Testing

### API Connectivity Tests (No GPU Cost)

```bash
VASTAI_API_KEY=xxx pytest tests/integration/test_vastai_e2e.py -v \
    -k "TestVastAIConnectivity"
```

### Full Training Tests (Costs ~$0.50-1.00)

```bash
VASTAI_API_KEY=xxx pytest tests/integration/test_vastai_e2e.py -v \
    --run-cloud
```

## Files

| File | Purpose |
|------|---------|
| `features/vastai/vastai_manager.py` | VastAI session management and HTTP API |
| `features/vastai/vastai_adapter.py` | TrainingProvider adapter |
| `vastai_config.toml` | GPU profiles and settings |
| `tests/integration/test_vastai_e2e.py` | E2E tests |
| `kestrel/tests/e2e/test_vastai_lora.spec.cjs` | Playwright E2E test |

## Future Work

1. **Fix GCR Authentication**: File issue with VastAI or find working format
2. **Docker Hub Mirror**: Push SimpleTuner image to Docker Hub as workaround
3. **VastAI Templates**: Investigate if web UI template creation has better auth
4. **Volume Support**: Implement local volume caching for model persistence

## References

- [VastAI Documentation](https://docs.vast.ai/)
- [VastAI CLI GitHub](https://github.com/vast-ai/vast-cli)
- [SimpleTuner API](../docker/simpletuner_api.py)
- [RunPod Training (Working)](./RUNPOD_TRAINING.md)
