# LoRA Training & Generation - Operational Guide

**Status:** Active - Dual Model Architecture (FLUX.2 + FLUX.1 Uncensored)
**Last Updated:** 2026-01-04
**Owner:** Kestrel Team
**Source of Truth For:** LoRA training and selfie generation

## Current Architecture: Dual Model Support ✅

We support **two FLUX variants** for different content needs:

| Model | Container | Content Type | Notes |
|-------|-----------|--------------|-------|
| **FLUX.2-dev** | `kestrel-lora:v8` | SFW + Artistic Nudity | Base model has subtle content filtering |
| **FLUX.1-dev + Uncensored LoRA** | `kestrel-lora-flux1:v1` | **Full NSFW** | Multi-LoRA: character + uncensored |

### Key Discovery (January 2026)

**FLUX.2-dev has built-in content filtering** that cannot be bypassed:
- ✅ Works fine for SFW content and artistic nudity (e.g., "topless" scenes)
- ❌ Fails for explicit anatomy prompts - produces "swimsuit with sewn-on vulva" effect
- ❌ No uncensored LoRA adapters exist for FLUX.2-dev

**Solution: FLUX.1-dev with Uncensored LoRA**
- Uses `enhanceaiteam/Flux-Uncensored-V2` LoRA adapter
- Multi-LoRA stacking: character appearance + uncensored capabilities
- Requires training new LoRAs on FLUX.1-dev (FLUX.2 LoRAs incompatible)

### Which Model to Use?

| Content Type | Model | Container |
|--------------|-------|-----------|
| Portrait, Lifestyle, Fashion | FLUX.2-dev | `kestrel-lora:v8` |
| Artistic Nudity (topless) | FLUX.2-dev | `kestrel-lora:v8` |
| **Explicit NSFW** | FLUX.1-dev + Uncensored | `kestrel-lora-flux1:v1` |

---

## FLUX.2-dev Architecture (Current Default)

We use **RunPod for both training AND generation** with FLUX.2-dev:

| Component | Provider | Reason |
|-----------|----------|--------|
| **LoRA Training** | **RunPod** | FLUX.2-dev, sovereign infrastructure |
| **Selfie Generation** | **RunPod** | FLUX.2-dev, same pod, fast |

**Training on RunPod:**
- ✅ FLUX.2-dev with SimpleTuner
- ✅ A100 80GB with int8-quanto quantization (~50GB VRAM)
- ✅ Network volume caches models (no re-download)
- ⚠️ Single-image training takes 2-3 hours (1000+ repeats)
- ⚠️ **Content filtering on explicit scenes** - use FLUX.1 for NSFW

**Generation on RunPod:**
- ✅ Same pod, same cached models
- ✅ ~$0.02/image on warm pod
- ✅ Async pattern avoids Cloudflare timeout

---

## FLUX.1-dev Uncensored Architecture (For NSFW) 🔞

For explicit content generation, we use FLUX.1-dev with the uncensored LoRA adapter.

### Multi-LoRA Stacking

The FLUX.1 container loads **two LoRA adapters simultaneously**:

```python
# 1. Character LoRA (trained appearance)
pipe.load_lora_weights("/tmp/lora", weight_name="pytorch_lora_weights.safetensors", adapter_name="character")

# 2. Uncensored LoRA (removes content filtering)
pipe.load_lora_weights("enhanceaiteam/Flux-Uncensored-V2", weight_name="lora.safetensors", adapter_name="uncensored")

# 3. Combine with weights
pipe.set_adapters(["character", "uncensored"], adapter_weights=[1.0, 0.8])
```

### Files for FLUX.1 Uncensored

| File | Purpose |
|------|---------|
| `docker/Dockerfile.flux1-uncensored` | Docker image for FLUX.1-dev |
| `docker/simpletuner_flux1_api.py` | API with multi-LoRA support |
| `docker/cloudbuild-flux1-uncensored.yaml` | Cloud Build config |

### Key Differences from FLUX.2

| Setting | FLUX.2-dev | FLUX.1-dev |
|---------|------------|------------|
| `model_family` | `flux2` | `flux` |
| `pretrained_model_name_or_path` | `black-forest-labs/FLUX.2-dev` | `black-forest-labs/FLUX.1-dev` |
| Pipeline Class | `Flux2Pipeline` | `FluxPipeline` |
| Uncensored LoRA | ❌ Not available | ✅ `enhanceaiteam/Flux-Uncensored-V2` |
| diffusers version | `git+...diffusers.git` | `diffusers>=0.31.0` (stable) |

### Build FLUX.1 Image

```bash
cd ./
gcloud builds submit --config=docker/cloudbuild-flux1-uncensored.yaml --project=YOUR_PROJECT_ID
```

### Container Registry

| Image | Tag | Purpose |
|-------|-----|---------|
| `gcr.io/YOUR_PROJECT_ID/kestrel-lora` | `:v8` | FLUX.2-dev (SFW + artistic) |
| `gcr.io/YOUR_PROJECT_ID/kestrel-lora-flux1` | `:v1` | FLUX.1-dev uncensored (NSFW) |

### Important: LoRA Compatibility

**FLUX.1 and FLUX.2 LoRAs are NOT interchangeable!**

- LoRAs trained on FLUX.2-dev only work with FLUX.2-dev
- LoRAs trained on FLUX.1-dev only work with FLUX.1-dev
- For NSFW companions, train on FLUX.1-dev from the start

### Trigger Words for Uncensored Content

From the [Flux-Uncensored-V2 model card](https://huggingface.co/enhanceaiteam/Flux-Uncensored-V2):

```
nsfw, naked, pron, kissing, erotic, nude, sensual, adult content, explicit
```

Example prompt:
```
A photo of TOKabc123, nude, explicit, full frontal nudity, photorealistic, 8k
```

### Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPANION CREATION                          │
├─────────────────────────────────────────────────────────────────┤
│  1. User creates companion with physical traits                 │
│  2. Replicate generates avatar image (FLUX.1-schnell, SFW)      │
│  3. Avatar bytes stored in PostgreSQL (avatar_data column)      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LORA TRAINING (RUNPOD)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Resume persistent RunPod pod with FLUX.2-dev cached         │
│  2. Upload avatar bytes via multipart form to /train            │
│  3. SimpleTuner trains LoRA with trigger word (TOK{id})         │
│  4. Training: 2-3 hours for single-image, ~$5-6                 │
│  5. LoRA saved to /workspace/trained_loras/ (persistent)        │
│  6. Upload to IPFS via Lighthouse for sovereign storage         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               SELFIE GENERATION (RUNPOD)                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Same pod, FLUX.2-dev already loaded                         │
│  2. Load companion's LoRA from IPFS or local cache              │
│  3. Generate with prompt: "TOK{id} at the beach"                │
│  4. UNCENSORED output - full creative freedom                   │
│  5. Store selfie in companion_files table                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cost Comparison

| Provider | Training Cost | Training Time | Generation | Notes |
|----------|---------------|---------------|------------|-------|
| **Replicate (training)** | ~$6-7 | 15-20 min | Censored | Fast, reliable |
| **RunPod (training)** | ~$5-6 | 2-3 hours | Uncensored | Complex, TTL issues |
| **RunPod (generation)** | ~$0.02/image | Instant | Uncensored | Our choice |
| **fal.ai** | ~$8+ | ~30 min | Varies | More expensive |

**Bottom Line:** Replicate for training + RunPod for generation = best of both worlds.

---

## Kestrel Integration via TrainingProviderFactory

The `TrainingProviderFactory` provides a unified interface for training and generation across multiple providers. As of December 2025, RunPod is the recommended provider for selfie generation due to uncensored FLUX.2-dev support.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       kestrel SELFIE ENDPOINT                              │
│                POST /api/companions/{id}/selfie/generate                 │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     VisualIdentityFeature                                │
│    _get_training_provider() -> TrainingProviderFactory                   │
│    _generate_with_provider() -> provider.generate_image()                │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  TrainingProviderFactory                                 │
│    Priority: vertex_ai > runpod > replicate > gcp_compute > vastai       │
│    Environment: RUNPOD_API_KEY enables RunPod                            │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  RunPodTrainingAdapter                                   │
│    generate_image(GenerationConfig) -> GenerationResult                  │
│    Uses async polling: /generate/async + /generate/status/{job_id}       │
│    Avoids Cloudflare 100s timeout via async pattern                      │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    RunPod SimpleTuner API                                │
│    POST /generate/async -> {"job_id": "abc123"}                          │
│    GET /generate/status/{job_id} -> {"status": "completed", "images":...}│
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `features/training/factory.py` | TrainingProviderFactory for auto-detection |
| `features/training/adapters/runpod_adapter.py` | RunPod adapter with `generate_image()` |
| `features/training/types.py` | GenerationConfig, GenerationResult, GenerationState |
| `features/visual_identity/feature.py` | VisualIdentityFeature using factory |
| `kestrel/endpoints/selfie.py` | Kestrel selfie API endpoints |

### Generation Types

```python
from features.training import (
    TrainingProviderFactory,
    GenerationConfig,
    GenerationResult,
    GenerationState,
    GenerationError,
)

# Create generation config
config = GenerationConfig(
    prompt="A portrait of TOKabc123 person at the beach",
    lora_path="/workspace/trained_loras/companion123.safetensors",
    trigger_word="TOKabc123",
    num_outputs=1,
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=4.0,
)

# Get provider and generate
provider = TrainingProviderFactory.get_provider("runpod")
result = await provider.generate_image(config)

# Result contains:
# - job_id: Async job ID
# - state: GenerationState (COMPLETED, FAILED, etc.)
# - images: List of base64 data URLs
# - elapsed_seconds: Total generation time
```

### Async Generation Pattern (Solves Cloudflare Timeout)

The RunPod adapter uses async polling to avoid Cloudflare's 100-second timeout:

```
1. POST /generate/async
   → Returns immediately with job_id
   → Pod starts background generation

2. Poll GET /generate/status/{job_id}
   → status: "loading_model" (~20s)
   → status: "loading_lora" (~5s)
   → status: "generating" (~330s with CPU offload)
   → status: "completed" + images[]

3. Return base64 images
```

**Total time with CPU offload on A100 80GB:** ~5.5 minutes per image

### Testing with Playwright

The E2E test at `kestrel/tests/e2e/test_luna_runpod_lora.spec.cjs` validates the full pipeline:

```javascript
test('6. Generate selfie with trained LoRA', async ({ request }) => {
    // Timeout: 10 minutes (async generation takes ~6-7 min)
    const selfieResp = await request.post(`${kestrel_URL}/api/companions/${companionId}/selfie/generate`, {
        headers: { 'Authorization': `Bearer ${authToken}` },
        data: {
            scene: 'stargazing at night with aurora borealis in background',
            style: 'photorealistic',
            provider: 'runpod'
        },
        timeout: 600000
    });

    // Response includes:
    // - image_url: base64 data URL
    // - backend: "runpod"
    // - elapsed_seconds: ~330-400s
});
```

Run with:
```bash
cd kestrel
kestrel_URL=http://localhost:7777 npx playwright test tests/e2e/test_luna_runpod_lora.spec.cjs --timeout=1800000
```

---

## RunPod Self-Hosted Training (Historical Reference)

> **Note:** Self-hosted training on RunPod is documented below for reference.
> We may revisit this for batch training or full sovereignty requirements.
> Current recommendation: Use Replicate for training.

### December 21, 2025 Training Run

**First successful sovereign training** - avatar bytes passed directly from PostgreSQL:

| Metric | Value |
|--------|-------|
| Job ID | `3a55cc5d-94c0-4e95-8bf0-7b548ab36f2e` |
| Companion | `d6822ae1-f12c-40d9-8487-a7c4bdc93c35` (Luna) |
| Trigger Word | `TOKd6822ae1` |
| Steps | 1000 |
| Repeats | 1110 (formula: `int(steps * 1.1) + 10`) |
| Training Time | ~3 hours (reached 99%, then pod terminated) |
| Cost | ~$5.67 ($1.89/hr × 3 hrs) |

**Issues Encountered:**
1. **Pod terminated at 99%** - TTL expired (was set to 1 hour, training took 3+ hours)
2. **Training restarted instead of completing** - SimpleTuner bug at 99.2%
3. **Single-image training is slow** - 1110 repeats required vs multi-image datasets

**Key Fix Applied:** Avatar data now passed as bytes from `avatar_data` column instead of trying to download from local URL path.

### When to Consider RunPod Training

RunPod self-hosted training may be worth revisiting for:
1. **Batch training** - Multiple companions on same pod amortizes overhead
2. **Full data sovereignty** - Avatar never leaves your infrastructure
3. **Custom training configs** - Full SimpleTuner control

## Current Configuration

### RunPod Pod
- **Pod ID:** `4anmjn65oser3r`
- **Pod Name:** kestrel-lora-v3-secure-ca
- **URL:** `https://4anmjn65oser3r-8000.proxy.runpod.net`
- **GPU:** NVIDIA A100 80GB PCIe (SECURE cloud)
- **Network Volume:** `YOUR_VOLUME_ID` (configure in runpod_config.toml)
- **Volume Mount Path:** `/workspace` (REQUIRED)
- **Docker Image:** `gcr.io/YOUR_PROJECT_ID/kestrel-lora:v5`
- **Template ID:** `YOUR_TEMPLATE_ID` (create in RunPod with GCR auth)
- **Datacenter:** CA-MTL-3 (Montreal, Canada)

### Docker Image Tags
| Tag | Purpose | Notes |
|-----|---------|-------|
| `:v8` | **Current (recommended)** | Flux2Pipeline for FLUX.2-dev, diffusers==0.36.0 |
| `:v7` | Previous attempt | Wrong fix: Used FluxPipeline (FLUX.1), not Flux2Pipeline |
| `:v6` | Failed | Wrong fix: FluxPipeline is for FLUX.1, not FLUX.2 |
| `:v5` | Legacy | Stable diffusers==0.36.0, but uses wrong pipeline class |
| `:latest` | Default | Points to v8 |

### Environment Variables (CRITICAL)
```bash
# MUST be set for persistent model caching
HF_HOME=/workspace/huggingface
TRANSFORMERS_CACHE=/workspace/huggingface
TORCH_HOME=/workspace/huggingface
HF_TOKEN=<your-huggingface-token>
```

**WARNING:** If `HF_HOME=/tmp/huggingface`, models download to ephemeral storage and are LOST on pod restart (~25GB FLUX.2 download each time).

### SimpleTuner FLUX.2 Config (CRITICAL)
```python
{
    # FLUX.2 - NOT FLUX.1!
    "model_family": "flux2",        # NOT "flux"!
    "model_flavour": "dev",
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-dev",

    # FLUX.2 guidance settings
    "flux_guidance_mode": "constant",
    "flux_guidance_value": 1.0,

    # Training
    "model_type": "lora",
    "lora_rank": 16,
    "optimizer": "adamw_bf16",
    "max_train_steps": 1000,
    "learning_rate": 1e-4,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "bf16",

    # Checkpointing (for visibility)
    "checkpoint_step_interval": 100,  # Save every 100 steps
    "checkpoints_total_limit": 3,     # Keep last 3

    # Logging
    "report_to": "tensorboard",       # Local monitoring
    "debug_dataset_loader": True,     # Verbose dataset output
    "print_filenames": True,          # Show processed files
}
```

### Environment: Logging
```bash
# Set in Dockerfile for verbose debug.log output
SIMPLETUNER_LOG_LEVEL=DEBUG
```

## Files

| File | Purpose |
|------|---------|
| `docker/simpletuner_api.py` | REST API wrapper for SimpleTuner training |
| `docker/Dockerfile.simpletuner` | Docker image with SimpleTuner + Python 3.12 |
| `docker/cloudbuild-simpletuner.yaml` | GCloud Build config |
| `runpod_config.toml` | RunPod profiles including `[profiles.training]` |
| `features/runpod/runpod_manager.py` | RunPod pod management |
| `kestrel/services/lora_training_service.py` | Kestrel integration for LoRA training |

## API Endpoints

Base URL: `https://<pod-id>-8000.proxy.runpod.net`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check (not training) |
| `/train` | POST | Start LoRA training job |
| `/status/{job_id}` | GET | Get training status |
| `/download/{job_id}` | GET | Download trained LoRA |
| `/generate` | POST | Sync image generation (may timeout on proxy) |
| `/generate/async` | POST | **Async generation (recommended)** - returns job_id |
| `/generate/status/{job_id}` | GET | Poll async generation status/results |
| `/loras` | GET | List available trained LoRAs |
| `/debug/exec` | GET | Debug command execution |
| `/debug/logs/{job_id}` | GET | Get training logs, checkpoints, TensorBoard events |

### Async Generation (Recommended)
```bash
# Start async generation - returns immediately with job_id
curl -X POST "https://<pod>-8000.proxy.runpod.net/generate/async" \
  -F "prompt=A photo of TOK person at the beach" \
  -F "lora_path=test-avatar-001" \
  -F "trigger_word=TOK" \
  -F "num_inference_steps=28"
# Returns: {"job_id": "abc123", "status": "pending", ...}

# Poll for results (status: pending -> loading_model -> generating -> completed)
curl "https://<pod>-8000.proxy.runpod.net/generate/status/abc123"
# When completed: {"status": "completed", "images": ["data:image/png;base64,..."], ...}
```

### Training Request
```bash
curl -X POST "https://rdbmipru7tqs9g-8000.proxy.runpod.net/train" \
  -F "image=@avatar.jpg" \
  -F "companion_id=a86e1284-106c-4aae-b17c-68937f17e96b" \
  -F "trigger_word=TOKa86e1284" \
  -F "steps=1000" \
  -F "lora_rank=16"
```

## Known Issues & Fixes

### Issue 1: FLUX.1 instead of FLUX.2
**Symptom:** Training uses wrong model architecture
**Fix:** Ensure `model_family: "flux2"` (not `"flux"`) in config

### Issue 2: Network Volume Not Mounted
**Symptom:** `HF_HOME=/tmp/huggingface`, models lost on restart
**Diagnosis:**
```bash
curl -s "https://<pod>-8000.proxy.runpod.net/debug/exec?cmd=env%20%7C%20grep%20HF"
curl -s "https://<pod>-8000.proxy.runpod.net/debug/exec?cmd=mount%20%7C%20grep%20workspace"
```
**Fix:** Pod must be created with `network_volume_id` in RunPod config

### Issue 3: "Separator is not found" Error
**Symptom:** Progress tracking breaks due to tqdm progress bars
**Fix:** Use `read(4096)` chunks instead of `readline()` in output parsing

### Issue 4: No Training Progress Logged
**Symptom:** Training runs but progress stays at 0%
**Fix:** Parse tqdm percentage patterns (`50%|████`) in addition to step patterns

### Issue 5: Training Runs Forever
**Symptom:** Training continues past max_train_steps (e.g., 500 → 1000+)
**Cause:** `repeats` in multidatabackend.json was hardcoded to 100, causing 100 epochs
**Fix:** Calculate `repeats = int(steps * 1.1) + 10` to match max_train_steps
**Commit:** Fixed in simpletuner_api.py (December 2025)

### Issue 6: "Request URL is missing http/https protocol"
**Symptom:** Training fails immediately with protocol error
**Cause:** `image_url` in DB is `/api/avatars/{id}` (local API path), not downloadable URL.
The `download_and_sanitize_avatar()` function downloads Replicate URLs and stores bytes
in `avatar_data`, then sets `image_url` to a local path for serving.
**Fix:** Pass `avatar_data` bytes directly from DB instead of trying to download from URL.
- `companions.py`: Fetch `avatar_data` from DB, pass to training service
- `lora_training_service.py`: Accept `avatar_data: bytes` instead of `image_url: str`
- `runpod_manager.py`: Accept `avatar_data: bytes`, detect content type from magic bytes
**Commit:** Fixed December 21, 2025

### Issue 7: RunPod Template Image Caching
**Symptom:** Pod shows using new image tag in API, but runs old code (proven by error signature)
**Error:** `CombinedTimestepTextProjEmbeddings.forward() takes 3 positional arguments but 4 were given` (v5 error)
**Cause:** RunPod aggressively caches Docker image layers, even across different tags
**Diagnosis:**
```bash
# Pod API says it's using v8
curl "https://api.runpod.io/graphql?api_key=$KEY" -d '{"query":"..."}'
# Returns: imageName: "gcr.io/YOUR_PROJECT_ID/kestrel-lora:v8"

# But running code is v5 (proven by error)
curl "https://<pod>-8000.proxy.runpod.net/generate/async" -F "prompt=..."
# Returns: v5 pipeline error
```
**What DOESN'T Work:**
- ❌ Stop/resume pod - uses cached layers
- ❌ Update template image tag - cached layers persist
- ❌ New tag (v6→v7→v8) - cache is at layer level, not tag level

**What WORKS:**
- ✅ **Terminate pod completely** - forces fresh image pull on next creation
- ✅ **Wait for cache expiration** - unknown duration (hours? days?)
- ✅ **Different base image** - change FROM line in Dockerfile

**Recommended Fix:**
```bash
# 1. Update your template to use :v8 in RunPod web UI
# 2. Terminate pod (not stop!)
curl -X POST "https://api.runpod.io/v2/$POD_ID/terminate" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"

# 3. Create new pod from template (waits for template image)
# Pod will pull fresh v8 image
```

### Issue 8: Cannot Generate Selfies While Training
**Symptom:** Selfie generation fails with CUDA OOM error
**Error:** `CUDA out of memory. Tried to allocate 320.00 MiB. GPU 0 has 195.62 MiB free.`
**Cause:** Training uses ~80GB VRAM (FLUX.2-dev + SimpleTuner), leaving no memory for inference.
**Fix:** Wait for training to complete before generating selfies. Training and generation **cannot run simultaneously** on a single A100 80GB pod.
**Workaround Options:**
1. **Wait for training** - Training takes ~3 hours for single-image LoRA
2. **Use separate pod** - Create a dedicated generation pod (adds cost)
3. **Queue requests** - Implement job queue to serialize training/generation

**Important:** The pod status shows `current_job` when training is active. Check `/health` before attempting generation:
```bash
curl -s "https://<pod>-8000.proxy.runpod.net/health"
# If current_job is set, training is in progress - wait before generating
```

## Build & Deploy

### Build Docker Image
```bash
cd ./
gcloud builds submit --config=docker/cloudbuild-simpletuner.yaml --project=YOUR_PROJECT_ID
```

### Create/Update Pod with Network Volume
The pod MUST be created with network volume attached:
```python
# In runpod_config.toml
[profiles.training]
network_volume_id = "YOUR_VOLUME_ID"  # Create in RunPod console
volume_mount_path = "/workspace"
datacenter_id = "CA-MTL-3"  # MUST match volume location

[profiles.training.env]
HF_HOME = "/workspace/huggingface"
```

### Resume Persistent Pod
```bash
# Using runpod CLI or API
runpod pod resume ipjkh8tbnuz098
```

## Debugging Commands

```bash
# Check GPU
curl -s ".../debug/exec?cmd=nvidia-smi"

# Check HF cache location
curl -s ".../debug/exec?cmd=env%20%7C%20grep%20HF"

# Check if workspace mounted
curl -s ".../debug/exec?cmd=df%20-h%20%7C%20grep%20workspace"

# Check training process
curl -s ".../debug/exec?cmd=ps%20aux%20%7C%20grep%20train"

# Check SimpleTuner debug.log (verbose output)
curl -s ".../debug/exec?cmd=tail%20-100%20/app/debug.log"

# Check for LoRA output
curl -s ".../debug/exec?cmd=find%20/app/output%20-name%20%27*.safetensors%27"

# Get full training logs via API (checkpoints, TensorBoard, debug.log)
curl -s ".../debug/logs/{job_id}?lines=200"
```

### Log Interpretation
- **TensorBoard events** in `/app/output/{job_id}/logs/` show training metrics
- **Checkpoints** appear as `checkpoint-100`, `checkpoint-200` directories
- **debug.log** contains verbose SimpleTuner output when `SIMPLETUNER_LOG_LEVEL=DEBUG`
- Progress at 0% for extended time = training likely stuck (check debug.log for errors)

## Training Timeline

### Expected (from SimpleTuner docs)
| Step | Time | Notes |
|------|------|-------|
| Pod start (cold) | 2-5 min | New pod creation |
| Pod resume (warm) | 10-30s | Persistent pod |
| FLUX.2 download | 15-20 min | First time only (25GB) |
| Mistral-3 encoder | 5-10 min | Text encoder download |
| Training (1000 steps) | 30-60 min | A100 80GB |
| **Total (cold)** | ~60-90 min | First training |
| **Total (warm)** | ~30-60 min | Subsequent trainings |

### Observed (December 21, 2025 - Single Image Training)
| Step | Time | Notes |
|------|------|-------|
| Pod resume | ~30s | Persistent pod already had models cached |
| Dataset prep | ~2 min | Single image, caption generation |
| Training loop | ~3 hours | 1110 repeats × 1000 steps, single image |
| **Total** | ~3 hours | Much slower than expected |

**Why so slow?** Single-image training requires many repeats (1110) to prevent overfitting. The SimpleTuner docs assume multi-image datasets where each image is seen fewer times. With 10+ training images, expect closer to 30-60 minutes.

### Repeat Formula
```python
# For single-image training, SimpleTuner needs many repeats
repeats = int(max_train_steps * 1.1) + 10  # e.g., 1000 → 1110 repeats
```

Each repeat cycles through the entire dataset once. With 1 image and 1110 repeats, SimpleTuner processes the image 1110 times across training.

## Cost

- A100 80GB: $1.89/hr
- Network Volume: $0.10/GB/month (150GB = $15/month)
- Training job (1000 steps, single image): ~$5-6 per companion (observed: 3 hours)
- vs Replicate: ~$6-7 (but only 15-20 min)
- vs fal.ai: $8+ per training

### Cost Optimization Tips
1. **Keep pod running** for batch trainings - avoid cold start overhead
2. **Use 500 steps** for quick tests (~1.5 hours, ~$3)
3. **Consider Replicate** for one-off trainings where time matters
4. **Generation is cheap** - ~$0.02/image on warm pod

## RunPod Self-Hosted Architecture (Reference)

This architecture applies when using RunPod for training (not current recommendation):

```
┌─────────────────────────────────────────────────────────────────┐
│                     LORA TRAINING (RUNPOD)                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Fetch avatar_data bytes from PostgreSQL (NOT image_url!)    │
│  2. Detect content type from magic bytes (PNG/JPEG)             │
│  3. Upload bytes to RunPod via multipart form                   │
│  4. SimpleTuner trains with trigger word (TOK{companion_id})    │
│  5. LoRA weights saved to /workspace/trained_loras/             │
└─────────────────────────────────────────────────────────────────┘
```

## FLUX.2-dev Inference on A100 80GB

FLUX.2-dev is a 32B parameter model requiring ~80GB VRAM in BF16. To run inference on A100 80GB, CPU offloading is **required**.

### Code (from HuggingFace blog)
```python
from diffusers import Flux2Pipeline
import torch

pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
    cache_dir="/workspace/huggingface",
)

# REQUIRED for A100 80GB - moves components to CPU when not in use
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="TOKabc123 person at the beach",
    num_inference_steps=28,  # 28 is good trade-off (50 for quality)
    guidance_scale=4,
    height=1024,
    width=1024,
).images[0]
```

### Memory Requirements
| Configuration | VRAM Required |
|---------------|---------------|
| Full BF16 (no offload) | ~80GB+ |
| With CPU offload | ~62GB |
| 4-bit quantization | ~20GB |

**Important:** Use `Flux2Pipeline`, not `FluxPipeline` (which is for FLUX.1).

### CRITICAL: Flux2Pipeline Requires Diffusers from Source

`Flux2Pipeline` is **NOT available in stable diffusers releases** (as of diffusers 0.36.0). It's only in the development branch.

The Dockerfile MUST install diffusers from git:

```dockerfile
# In docker/Dockerfile.simpletuner - REQUIRED for FLUX.2-dev inference
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    python-multipart \
    aiofiles \
    google-cloud-storage \
    optimum-quanto \
    "git+https://github.com/huggingface/diffusers.git"  # REQUIRED - Flux2Pipeline not in stable
```

**Without this, you get:**
```
Pipeline <class 'diffusers.pipelines.flux.pipeline_flux.FluxPipeline'> expected
['feature_extractor', 'image_encoder', 'scheduler', 'text_encoder', 'text_encoder_2',
'tokenizer', 'tokenizer_2', 'transformer', 'vae'], but only
{'tokenizer', 'transformer', 'text_encoder', 'vae', 'scheduler'} were passed.
```

This error means you're using `FluxPipeline` (FLUX.1) which expects dual text encoders (T5 + CLIP), but FLUX.2-dev has a single Mistral-3 text encoder.

**FLUX.2-dev Architecture (from model_index.json):**
- `_class_name`: "Flux2Pipeline"
- `text_encoder`: "Mistral3ForConditionalGeneration" (NOT T5 + CLIP)
- `transformer`: "Flux2Transformer2DModel" (NOT FluxTransformer2DModel)
- `vae`: "AutoencoderKLFlux2" (different VAE)

## Prompt Builder - Critical Fix (January 2026)

### Issue: Scene Prompts Ignored

**Symptom:** All explicit scene prompts generated identical "nurse uniform" images regardless of requested scene.

**Root Cause:** The prompt builder was appending `companion_appearance` to every prompt:
```
# BAD - old code
prompt = f"A photo of {trigger_word}, {scene_description}, {companion_appearance}..."
# Result: "TOK379f40b3, bare breasts visible, nurse uniform" → FLUX renders nurse uniform
```

**The LoRA trigger word ALREADY encodes appearance** from training data. Appending appearance is:
1. Redundant (trigger word has it)
2. Conflicting (scene says "topless", appearance says "nurse uniform")

### Fix Applied

In `features/visual_identity/feature.py`:
```python
# GOOD - new code
# Use ONLY scene description - trigger word already has appearance baked in from LoRA training
base_prompt = f"A photo of TRIGGER_WORD, {scene_description}. High quality, photorealistic, 8k."
```

**Commit:** `f7d7226` - fix(kestrel): Make build_selfie_prompt() use scene parameter

### Lesson Learned

**Never override LoRA-encoded appearance with text prompts.** The trigger word IS the appearance.

---

## References

- [HuggingFace FLUX.2 Blog](https://huggingface.co/blog/flux-2) - Official inference guide
- [SimpleTuner FLUX2.md](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX2.md)
- [Flux-Uncensored-V2](https://huggingface.co/enhanceaiteam/Flux-Uncensored-V2) - Uncensored LoRA for FLUX.1-dev
- [PLAN_RUNPOD_INTEGRATION.md](./PLAN_RUNPOD_INTEGRATION.md) - General RunPod architecture
- [runpod_config.toml](/runpod_config.toml) - Pod configuration profiles
