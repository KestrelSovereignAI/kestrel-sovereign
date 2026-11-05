---
type: Architecture Spec
title: Training Provider Architecture
description: The Training Provider system provides a unified interface for LoRA training
  across multiple GPU providers. This enables automatic provider selection, consistent
  error handling,...
resource: /docs/architecture/TRAINING_PROVIDER_ARCHITECTURE.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-07-24T00:00:00Z'
status: experimental
owner: architecture
canonical: false
generated: false
privacy: public
---

# Training Provider Architecture

## Runpod catalog cutover

The built-in `runpod` adapter is **recovery-only**, not a fresh training or
generation backend. Its public start/generate methods and the legacy
`kestrel runpod deploy lora-trainer` command refuse before provider, model,
task or configuration effects. An old session is not new input/consent
authority, and legacy FLUX configuration is not automatically an SDXL catalog
request. Detailed cross-repository requirements remain in
[KCR#64](https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/issues/64);
this Core implementation is tracked by
[Core#3954](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3954).

`get_provider(name, purpose="training"|"generation")` requires the requested
fresh capability before creating **or returning a cached** provider. Automatic
selectors respect that capability. A forced `GENERATION_PROVIDER` that cannot
serve the request raises clearly instead of switching providers/models.
`list_available_providers()` lists fresh-work backends; use
`include_recovery=True` to include configured historical-only providers.

`get_provider("runpod")` remains the historical accessor. Existing status,
download, cancellation and release retain the SAME session-lifecycle owner.
Unknown state after restart/eviction must not be reported as confirmed cleanup;
reconcile the original durable cloud ledger/resource identity. Pausing a
persistent Pod stops GPU billing, not necessarily disk billing. Original
execution denial, uncertain COMMIT and native retirement debt are propagated,
not converted into ordinary provider unavailability. Failed closes remain
reachable until their custody can be resolved.

New catalog delivery uses the existing external-provider entry-point mechanism
below, under its own name and actual host admission contract. Core neither
imports private Frinz/catalog types nor installs a second queue/worker. The
compatibility release must actually be published before a cloud consumer sets
its dependency floor; a branch version is not a release.

## Overview

The Training Provider system provides a unified interface for LoRA training across multiple GPU providers. This enables automatic provider selection, consistent error handling, and seamless switching between providers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TrainingProviderFactory                  │
│     get_provider("vertex_ai") | get_default_provider()      │
└──────────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────┴──────────────────────┐
    │              TrainingProvider Protocol       │
    │  start_training | get_status | download_weights │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────┬───────────┼───────────┬──────────┐
    ▼          ▼           ▼           ▼          ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌───────┐ ┌────────┐
│Vertex  │ │Replicate│ │GCP       │ │Vast.ai│ │RunPod  │
│AI      │ │         │ │Compute   │ │       │ │        │
│Adapter │ │Adapter  │ │Adapter   │ │Adapter│ │Adapter │
└────────┘ └────────┘ └──────────┘ └───────┘ └────────┘
     │          │           │           │          │
     ▼          ▼           ▼           ▼          ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌───────┐ ┌────────┐
│Vertex  │ │Replicate│ │GCPCompute│ │VastAI │ │RunPod  │
│AI      │ │API      │ │Engine    │ │       │ │        │
│Manager │ │         │ │Manager   │ │Manager│ │Manager │
└────────┘ └────────┘ └──────────┘ └───────┘ └────────┘
```

## Provider Types

### Serverless Providers
Jobs run to completion without instance management:
- **Vertex AI**: Google Cloud Custom Jobs (A100 80GB)
- **Replicate**: Managed training API (FLUX.1 only)

### Session-Based Providers
Require instance lifecycle management:
- **GCP Compute Engine**: VM instances with SSH access
- **Vast.ai**: GPU marketplace instances
- **RunPod**: Managed GPU pods with persistent volumes

## Provider Capabilities Matrix

Each provider has different capabilities. Use `TrainingProviderFactory.get_capabilities()` to query these programmatically. Model/cost notes below are historical, not current quotes or capacity observations.

| Provider | Training | Generation | Uncensored | FLUX Version | Cost/Train | Notes |
|----------|----------|------------|------------|--------------|------------|-------|
| **RunPod (legacy adapter)** | No new work | No new work | Historical metadata | Historical 2.x | Not offered | Status/cancel/download/resource custody only |
| **Vertex AI** | ✅ | ✅ | ✅ | 2.x | ~$5-10 | Serverless, most reliable |
| **Replicate** | ✅ | ✅ | ❌ | 1.x | ~$2-5 | Cheapest, but censored |
| **Vast.ai** | ✅ | ✅ | ✅ | 2.x | ~$2-4 | Cheapest uncensored |
| **GCP Compute** | ✅ | ✅ | ✅ | 2.x | ~$4-8 | VM-based, manual control |

### Capability Details

**Training**: Providers declaring training support can accept their supported inputs; the legacy Runpod adapter cannot.

**Generation**: Requires a declared generation capability and compatible, authorized artifact inputs; availability alone is insufficient.

**Uncensored**: Whether content safety filters are applied:
- ✅ = No filters, NSFW content allowed
- ❌ = Content safety filters applied by provider

**FLUX Version**:
- **1.x**: FLUX.1-dev/schnell - Original models, smaller text encoder
- **2.x**: FLUX.2-dev - Newer models, Mistral-24B text encoder, better quality

### Replicate Limitations

Replicate is the **cheapest** option but has important limitations:

1. **Censored Generation**: Replicate applies content safety filters to all output
2. **FLUX.1 Only**: Training uses FLUX.1-dev, not the newer FLUX.2
3. **Portable Weights**: LoRA weights CAN be downloaded and used elsewhere
4. **⚠️ NO IPFS Support**: Replicate does NOT accept IPFS URLs for external LoRA

#### IPFS Limitation Details

Replicate's `hf_lora` parameter only accepts:
- HuggingFace paths: `username/model-name`
- HuggingFace URLs: `https://huggingface.co/.../file.safetensors`
- Replicate delivery URLs: `https://replicate.delivery/.../trained_model.tar`
- CivitAI URLs: `https://civitai.com/api/download/...`
- Direct `.safetensors` URLs ending in `.safetensors`

**NOT supported:**
- IPFS gateway URLs (e.g., `https://gateway.lighthouse.storage/ipfs/Qm...`)
- Plain IPFS CIDs (e.g., `QmXxx...`)

The Replicate adapter will return a clear error if IPFS URLs are passed.

**Workarounds:**
1. **Use a compatible provider**: verify its current artifact input contract. The legacy Runpod adapter does not admit new generation.
2. **Upload to HuggingFace**: Store LoRA on HuggingFace and use the HF path
3. **Train on Replicate**: If you train on Replicate, the model is hosted there automatically

**Recommended Workflow for Uncensored Content:**
```python
# Train on Replicate (cheap)
replicate_provider = TrainingProviderFactory.get_provider("replicate")
job = await replicate_provider.start_training(companion_id, avatar_data)
# ... wait for completion ...
weights = await replicate_provider.download_weights(job.job_id)

# Select an available generation backend with the required capability.
# This does not choose the retired Runpod adapter or authorize artifact use.
generation_provider = TrainingProviderFactory.get_generation_provider(uncensored=True)
```

### Selecting Providers by Capability

```python
from features.training import TrainingProviderFactory

# Get first available uncensored provider
provider = TrainingProviderFactory.get_uncensored_provider()

# Get provider for generation (optionally require uncensored)
provider = TrainingProviderFactory.get_generation_provider(uncensored=True)

# Check capabilities of specific provider
caps = TrainingProviderFactory.get_capabilities("replicate")
if caps and not caps.uncensored:
    print("Replicate applies content filters")
```

## Key Files

### Core Module (`features/training/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `types.py` | TrainingState, TrainingConfig, TrainingJob, TrainingStatus, ProviderCapabilities, GenerationResult |
| `protocol.py` | TrainingProvider Protocol and error classes |
| `factory.py` | TrainingProviderFactory with auto-detection and capability-based routing |

### Adapters (`features/training/adapters/`)

| File | Wraps | Type |
|------|-------|------|
| `vertex_ai_adapter.py` | VertexAIManager | Serverless |
| `replicate_adapter.py` | Replicate API | Serverless |
| `gcp_compute_adapter.py` | GCPComputeEngineManager | Session-based |
| `vastai_adapter.py` | VastAIManager | Session-based |
| `runpod_adapter.py` | RunPodManager (TODO) | Session-based |

## TrainingProvider Protocol

```python
@runtime_checkable
class TrainingProvider(Protocol):
    """Unified interface for all training providers."""

    @property
    def provider_name(self) -> str:
        """Provider identifier (e.g., 'vertex_ai', 'runpod')."""
        ...

    @property
    def provider_type(self) -> ProviderType:
        """SERVERLESS or SESSION_BASED."""
        ...

    def is_available(self) -> bool:
        """Check if provider is configured and ready."""
        ...

    async def start_training(
        self,
        companion_id: str,
        avatar_data: bytes,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingJob:
        """Start a LoRA training job."""
        ...

    async def get_status(self, job_id: str) -> TrainingStatus:
        """Get current job status and progress."""
        ...

    async def download_weights(self, job_id: str) -> Optional[bytes]:
        """Download trained LoRA weights."""
        ...

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        ...

    async def cleanup(self, job_id: str) -> None:
        """Clean up resources (terminate instances, etc.)."""
        ...
```

## TrainingState Enum

```python
class TrainingState(Enum):
    PENDING = "pending"           # Job created, waiting to start
    PROVISIONING = "provisioning" # Instance/resources being allocated
    PREPARING = "preparing"       # Environment setup (model loading)
    TRAINING = "training"         # Actively training
    COMPLETED = "completed"       # Successfully finished
    FAILED = "failed"             # Error occurred
    CANCELLED = "cancelled"       # User cancelled
```

Each state has mapping methods from provider-specific states:
- `from_vertex_state()`
- `from_gcp_instance_state()`
- `from_runpod_state()`
- `from_vastai_state()`
- `from_replicate_state()`

## TrainingConfig

```python
@dataclass
class TrainingConfig:
    trigger_word: Optional[str] = None  # Default: TOK{companion_id[:8]}
    steps: int = 500              # 500 steps for fast iteration (was 1000)
    lora_rank: int = 16
    learning_rate: float = 1e-4
    batch_size: int = 1
    resolution: str = "512,768,1024"
    profile: str = "training"     # GPU profile name
    use_spot: bool = True         # Use spot/preemptible instances
    ttl_seconds: int = 7200       # Max session duration (2 hours)
    callback_url: Optional[str] = None
```

## FLUX.2 SimpleTuner Configuration

**Reference:** [SimpleTuner FLUX2.md Quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX2.md)

### Recommended Settings for A100 80GB

```python
{
    # Precision & Quantization (REQUIRED for 80GB)
    "mixed_precision": "bf16",
    "base_model_precision": "int8-quanto",  # Quantize transformer
    "text_encoder_1_precision": "int8-quanto",  # Quantize Mistral-24B
    "quantize_via": "accelerator",  # GPU quantization
    "gradient_checkpointing": True,

    # Training (optimized for speed)
    "optimizer": "adamw_bf16",
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,  # Keep at 1 for speed
    "max_train_steps": 500,  # Sufficient for single subject

    # Validation (disabled for speed)
    "validation_steps": 0,  # Validate only after training
    "checkpoint_step_interval": 0,  # Save only final

    # Resolution
    "resolution": 1024,
    "validation_resolution": 1024,
}
```

### Memory Usage
- **int8-quanto on both transformer + text encoder**: ~52GB
- **Fits on 80GB without CPU offload**

### Key FLUX.2 Differences from FLUX.1
- Uses **Mistral-Small-3.1-24B** as text encoder (~48GB download)
- Requires **both** `base_model_precision` and `text_encoder_1_precision` to be quantized
- No CPU offload needed with proper quantization

See also: `/docs/FLUX2_TRAINING_CONFIG.md` for detailed configuration reference.

## Usage Examples

### Auto-Select Best Provider

```python
from features.training import TrainingProviderFactory, TrainingConfig

# Get best available provider (priority: vertex_ai > replicate > gcp_compute > vastai)
provider = TrainingProviderFactory.get_default_provider()
if not provider:
    raise RuntimeError("No training providers available")

# Start training
job = await provider.start_training(
    companion_id="abc123",
    avatar_data=avatar_bytes,
    config=TrainingConfig(steps=1000, lora_rank=16)
)

# Poll status
while True:
    status = await provider.get_status(job.job_id)
    if status.state.is_terminal():
        break
    await asyncio.sleep(60)

# Download weights
if status.state == TrainingState.COMPLETED:
    weights = await provider.download_weights(job.job_id)
    await provider.cleanup(job.job_id)
```

### Request Specific Provider

```python
provider = TrainingProviderFactory.get_provider("vertex_ai", purpose="training")
if not provider:
    available = TrainingProviderFactory.list_available_providers()
    raise RuntimeError(f"Vertex AI not available. Available: {available}")
```

## Implementing a New Adapter

Use the existing external-provider registration described below. Do not copy
the retired Runpod implementation, add another in-tree factory registry, or
invent a FLUX-to-catalog parameter conversion. Session-backed implementations
reuse the canonical `SessionTrainingLifecycle` owner for task and resource
retirement; they still need their own actual host input/admission contract.

## Writing a Training Provider (External Package, Entry Points)

External packages register a provider through the existing
`kestrel_sovereign.training_providers` entry-point group and Kestrel discovers
it at runtime, mirroring how LLM adapters register under
`kestrel_sovereign.llm_providers`.

This is how, for example, an out-of-process generation backend (a worker that
polls a shared queue and produces avatars/LoRA selfies) becomes a first-class
provider on a host where no in-process provider matches — without editing
Kestrel.

### 1. Implement the `TrainingProvider` protocol

Your class satisfies the same [`TrainingProvider` protocol](#trainingprovider-protocol)
the built-in adapters do: `provider_name`, `provider_type`, `is_available()`,
and the async `start_training` / `get_status` / `download_weights` / `cancel` /
`cleanup` methods. `is_available()` reports availability based on **your own**
environment checks — Kestrel does **not** add an entry to `PROVIDER_ENV_VARS`
for external providers.

Two **optional class attributes** control routing (both backward compatible —
omit them and sensible defaults apply). The factory reads them via
``getattr`` — they are deliberately **not** members of the
`@runtime_checkable` `TrainingProvider` protocol, since data members on a
runtime-checkable protocol would become *required* for `isinstance()` checks
and break the built-in adapters that don't declare them:

```python
from kestrel_sovereign.features.training.types import (
    ProviderCapabilities, ProviderType,
)


class CatalogWorkerProvider:
    # Lower = tried first. Interleaves with built-ins on the same numeric
    # scale: built-in provider N sits at N * 10 (local_mps=0, runpod=10,
    # vertex_ai=20, ...). priority=5 slots between local_mps and runpod.
    # Omit to default to the lowest priority (sorts after all built-ins).
    priority = 5

    # Declared once as a class attribute; surfaced via
    # TrainingProviderFactory.get_capabilities("catalog_worker").
    # Omit to default to an all-False ProviderCapabilities().
    capabilities = ProviderCapabilities(
        training=True,
        generation=True,
        uncensored=True,
        flux_version="2.x",
        supports_lora_download=True,
    )

    provider_name = "catalog_worker"
    provider_type = ProviderType.SERVERLESS

    def is_available(self) -> bool:
        # Your own env / reachability check.
        import os
        return bool(os.getenv("FRINZ_CATALOG_DATABASE_URL"))

    async def start_training(self, companion_id, avatar_data, config=None):
        ...
    # get_status / download_weights / cancel / cleanup ...
```

### 2. Register the entry point

In your package's `pyproject.toml`:

```toml
[project.entry-points."kestrel_sovereign.training_providers"]
catalog_worker = "frinz.services.catalog_worker_provider:CatalogWorkerProvider"
```

The entry-point **name** (`catalog_worker`) becomes the provider name used by
`get_provider("catalog_worker")`, `list_available_providers()`, and priority
routing.

### 3. What the factory does

On first use, `TrainingProviderFactory` lazily discovers every registered
entry point, records its class, `capabilities`, and `priority`, then:

- **Instantiates + checks `is_available()`** — an unavailable provider is
  silently skipped (never appears in `list_available_providers()` and
  `get_provider()` returns `None`). Original execution-control failures are
  raised, never normalized into ordinary unavailability.
- **Merges into the priority order** — `_effective_priority()` interleaves
  entry-point providers with the built-in `PROVIDER_PRIORITY` by their declared
  `priority`, so `get_default_provider()` / `get_generation_provider()` /
  `get_uncensored_provider()` all consider them.
- **Gates on capability** — routing respects `capabilities`, not just
  availability: `get_default_provider()` only returns a provider whose
  `capabilities.training` is set, and `get_uncensored_provider()` requires
  **both** `generation` and `uncensored`. A generation-only backend is never
  routed into a training flow, and a training-only backend is never returned as
  an uncensored generator.
- **Never shadows a built-in** — an entry point whose name collides with a
  built-in provider is ignored, and duplicate entry-point names keep the first
  registration (both logged).

No changes to `PROVIDER_PRIORITY`, `PROVIDER_ENV_VARS`, or
`PROVIDER_CAPABILITIES` in `factory.py` are required — external providers are
fully self-describing.

## Docker Image Support

The `docker/simpletuner_api.py` supports two modes:

### API Mode (default)
```bash
python simpletuner_api.py --port 8000
```
Runs FastAPI server for real-time training requests.

### Vertex AI Batch Mode
```bash
python simpletuner_api.py --vertex-mode \
    --avatar-gcs gs://bucket/avatar.png \
    --output-gcs gs://bucket/output/ \
    --companion-id abc123 \
    --trigger-word TOKabc123 \
    --steps 1000 \
    --lora-rank 16
```
Downloads from GCS, trains, uploads to GCS, exits.

## Error Handling

All adapters should raise these exceptions:

```python
class TrainingProviderError(Exception):
    """Base exception for training providers."""

class ProviderNotAvailableError(TrainingProviderError):
    """Provider not configured or unavailable."""

class TrainingSubmissionError(TrainingProviderError):
    """Failed to submit training job."""

class TrainingStatusError(TrainingProviderError):
    """Failed to get training status."""

class DownloadError(TrainingProviderError):
    """Failed to download trained weights."""
```

## Environment Variables

| Variable | Provider | Purpose |
|----------|----------|---------|
| `GCP_PROJECT_ID` | Vertex AI, GCP Compute | GCP project |
| `GOOGLE_APPLICATION_CREDENTIALS` | Vertex AI, GCP Compute | Service account JSON |
| `REPLICATE_API_TOKEN` | Replicate | API authentication |
| `VASTAI_API_KEY` | Vast.ai | API authentication |
| `RUNPOD_API_KEY` | RunPod | API authentication |

## Job ID Management

### Problem: Temporary UUID vs Real Provider Job ID

When a training job is submitted, the system generates a temporary UUID before calling the provider's API. This is necessary to immediately return a job ID to the caller. However, after the provider accepts the job, it returns a real job ID (e.g., Vertex AI's numeric job ID like `2278252740599611392`).

### Solution: `_update_training_job_id()`

After receiving the real job ID from the provider, the database must be updated:

```python
async def _update_training_job_id(companion_id: str, real_job_id: str):
    """
    Update the training job ID with the real provider job ID.

    This is called after the training provider (e.g., Vertex AI) returns
    the actual job ID, replacing the temporary UUID that was initially stored.
    """
    async with app_state.pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT avatar_config FROM companions WHERE id = $1", companion_id
        )
        avatar_config = _parse_avatar_config(row["avatar_config"] if row else None)
        avatar_config["lora_training_job_id"] = real_job_id
        await conn.execute(
            "UPDATE companions SET avatar_config = $1 WHERE id = $2",
            json.dumps(avatar_config), companion_id
        )
```

This is called in all training background tasks:
- `_run_vertex_training()` - After `vertex_manager.submit_training_job()`
- `_run_gcp_training()` - After `gcp_manager.submit_training_job()`
- `_run_vastai_training()` - After `vastai_manager.submit_training_job()`

### Provider Name Consistency

The status endpoint accepts both `"vertex"` and `"vertex_ai"` as valid provider names:

```python
if training_provider in ("vertex", "vertex_ai"):
    # Query Vertex AI for live status
```

This handles legacy data that may use either naming convention.

## Verified Training Jobs (December 2025)

| Companion ID | Vertex AI Job ID | Status | Duration | Notes |
|--------------|------------------|--------|----------|-------|
| `379f40b3-a808-41a1-bb83-29202dd09f00` | `2278252740599611392` | ✅ SUCCEEDED | ~2h51m | 500 steps, A100 80GB |
| `865ffca5-7a9c-41fa-aa5c-0f805609cd96` | `2134753279035310080` | ✅ SUCCEEDED | ~3h | 500 steps |
| `dbcabc51-8c46-4cc8-bc2e-713241e82b6d` | `3507401187336912896` | ✅ SUCCEEDED | ~3h | 500 steps |

LoRA outputs stored in: `gs://kestrel-training/training/{companion_id}/{timestamp}/output/`

---

*Last Updated: December 28, 2025*
