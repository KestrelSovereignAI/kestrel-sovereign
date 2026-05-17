# Training Provider Architecture

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

Each provider has different capabilities. Use `TrainingProviderFactory.get_capabilities()` to query these programmatically.

| Provider | Training | Generation | Unfiltered | FLUX Version | Cost/Train | Notes |
|----------|----------|------------|------------|--------------|------------|-------|
| **RunPod** | ✅ | ✅ | ✅ | 2.x | ~$3-5 | Persistent pod, best for iteration |
| **Vertex AI** | ✅ | ✅ | ✅ | 2.x | ~$5-10 | Serverless, most reliable |
| **Replicate** | ✅ | ✅ | ❌ | 1.x | ~$2-5 | Cheapest, but censored |
| **Vast.ai** | ✅ | ✅ | ✅ | 2.x | ~$2-4 | Lowest-cost marketplace option |
| **GCP Compute** | ✅ | ✅ | ✅ | 2.x | ~$4-8 | VM-based, manual control |

### Capability Details

**Training**: All providers support LoRA fine-tuning with avatar images.

**Generation**: All providers can generate images using trained LoRA weights.

**Unfiltered**: Whether the provider adds extra content filters:
- ✅ = No additional provider filters
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
1. **Use a different provider**: RunPod and Vertex AI both support IPFS
2. **Upload to HuggingFace**: Store LoRA on HuggingFace and use the HF path
3. **Train on Replicate**: If you train on Replicate, the model is hosted there automatically

**Recommended Cross-Provider Workflow:**
```python
# Train on Replicate (cheap)
replicate_provider = TrainingProviderFactory.get_provider("replicate")
job = await replicate_provider.start_training(companion_id, avatar_data)
# ... wait for completion ...
weights = await replicate_provider.download_weights(job.job_id)

# Generate on RunPod
runpod_provider = TrainingProviderFactory.get_unfiltered_provider()
# Upload weights to RunPod and generate
```

### Selecting Providers by Capability

```python
from features.training import TrainingProviderFactory

# Get first available provider without additional generation filters
provider = TrainingProviderFactory.get_unfiltered_provider()

# Get provider for generation (optionally require unfiltered output)
provider = TrainingProviderFactory.get_generation_provider(unfiltered=True)

# Check capabilities of specific provider
caps = TrainingProviderFactory.get_capabilities("replicate")
if caps and not caps.unfiltered_generation:
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
provider = TrainingProviderFactory.get_provider("vertex_ai")
if not provider:
    available = TrainingProviderFactory.list_available_providers()
    raise RuntimeError(f"Vertex AI not available. Available: {available}")
```

## Implementing a New Adapter

To add a new training provider (e.g., RunPod):

### 1. Create the Adapter Class

```python
# features/training/adapters/runpod_adapter.py

from ..protocol import TrainingProvider, ProviderNotAvailableError
from ..types import ProviderType, TrainingConfig, TrainingJob, TrainingStatus, TrainingState

class RunPodTrainingAdapter:
    """Adapter wrapping RunPodManager for TrainingProvider protocol."""

    provider_name = "runpod"
    provider_type = ProviderType.SESSION_BASED

    def __init__(self, manager=None):
        self._manager = manager
        self._active_jobs: dict[str, dict] = {}

    def _get_manager(self):
        """Lazy load the RunPod manager."""
        if self._manager is None:
            from features.runpod.runpod_manager import RunPodManager
            self._manager = RunPodManager()
        return self._manager

    def is_available(self) -> bool:
        """Check for RUNPOD_API_KEY."""
        try:
            manager = self._get_manager()
            return manager is not None and manager._api_key is not None
        except Exception:
            return False

    async def start_training(self, companion_id, avatar_data, config=None):
        # 1. Start/resume pod
        # 2. Upload avatar via API or SSH
        # 3. Call training endpoint
        # 4. Track job internally
        # 5. Return TrainingJob
        ...

    async def get_status(self, job_id):
        # Poll training status from pod API
        # Map to TrainingState using TrainingState.from_runpod_state()
        ...

    async def download_weights(self, job_id):
        # Download from pod or network volume
        ...

    async def cancel(self, job_id):
        # Stop training, optionally pause pod
        ...

    async def cleanup(self, job_id):
        # Pause or terminate pod to stop billing
        ...
```

### 2. Register in Factory

```python
# features/training/factory.py

# Add to _PROVIDER_CLASSES
_PROVIDER_CLASSES = {
    "vertex_ai": ("features.training.adapters.vertex_ai_adapter", "VertexAITrainingAdapter"),
    "replicate": ("features.training.adapters.replicate_adapter", "ReplicateTrainingAdapter"),
    "gcp_compute": ("features.training.adapters.gcp_compute_adapter", "GCPComputeTrainingAdapter"),
    "vastai": ("features.training.adapters.vastai_adapter", "VastAITrainingAdapter"),
    "runpod": ("features.training.adapters.runpod_adapter", "RunPodTrainingAdapter"),  # Add this
}

# Add to _PROVIDER_PRIORITY (higher priority = tried first)
_PROVIDER_PRIORITY = ["vertex_ai", "runpod", "replicate", "gcp_compute", "vastai"]
```

### 3. Add State Mapping

If RunPod has unique states, add mapping in `types.py`:

```python
@classmethod
def from_runpod_state(cls, state: str) -> "TrainingState":
    """Map RunPod pod state to unified state."""
    mapping = {
        "offline": cls.PENDING,
        "provisioning": cls.PROVISIONING,
        "loading": cls.PREPARING,
        "ready": cls.TRAINING,
        "running": cls.TRAINING,
        "terminating": cls.CANCELLED,
        "error": cls.FAILED,
    }
    return mapping.get(state.lower(), cls.PENDING)
```

### 4. Export from `__init__.py`

```python
# features/training/adapters/__init__.py
from .runpod_adapter import RunPodTrainingAdapter

__all__ = [
    # ... existing
    "RunPodTrainingAdapter",
]
```

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
