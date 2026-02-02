# FLUX.2 LoRA Training Configuration

**Source**: [SimpleTuner FLUX2.md Quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX2.md)

## Recommended Settings for A100 80GB GPU

### Memory & Precision
```python
{
    "mixed_precision": "bf16",              # BF16 for transformer
    "base_model_precision": "int8-quanto",  # Quantize transformer to int8
    "text_encoder_1_precision": "int8-quanto",  # Quantize Mistral-24B text encoder
    "quantize_via": "accelerator",          # GPU-based quantization
    "gradient_checkpointing": True,         # Reduce memory usage
}
```

**Memory Usage**: ~52GB with int8 quantization → fits on 80GB without CPU offload

### Training Parameters
```python
{
    "optimizer": "adamw_bf16",
    "learning_rate": 1e-4,          # Baseline, try 5e-5 if unstable
    "lr_scheduler": "constant",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,  # Keep at 1 for faster training
    "max_train_steps": 500,         # 500 steps is sufficient for single subject
    "validation_steps": 0,          # Disable for speed (validate after training)
    "checkpoint_step_interval": 0,  # Disable for speed (save at end only)
}
```

### Resolution
```python
{
    "validation_resolution": 1024,   # Standard 1024x1024
    "resolution": 1024,              # Training resolution
}
```

### FLUX.2 Specific Settings
```python
{
    "flux_guidance_mode": "constant",
    "flux_guidance_value": 1.0,
}
```

## Key Differences from FLUX.1

1. **Text Encoder**: FLUX.2 uses Mistral-Small-3.1-24B (~48GB) instead of T5-XXL
2. **Quantization Required**: Must quantize both transformer AND text encoder for 80GB GPU
3. **No CPU Offload Needed**: With int8 quantization, everything fits in VRAM

## Common Issues

### Disk Space
- Mistral-Small-3.1-24B download: ~48GB
- Ensure network volume has 100GB+ free space

### Memory
If OOM occurs:
1. Verify `base_model_precision: int8-quanto` is set
2. Verify `text_encoder_1_precision: int8-quanto` is set
3. Reduce batch size to 1
4. Enable gradient checkpointing

### Training Too Slow
If using CPU offload unnecessarily:
- Remove any CPU offload settings
- With int8-quanto on both models, GPU-only is faster

## Implementation Reference

See `/docker/simpletuner_api.py` function `create_simpletuner_config()` for the actual configuration used.

## Version Compatibility

- SimpleTuner 3.3.3+: Requires Mistral text encoder
- This config tested on: SimpleTuner 3.3.3, A100 80GB PCIe

---

## Image Generation via Vertex AI

### Generation Mode
Docker images support both training and generation via `--generate-mode` flag.

**Image Tags:**
- `gcr.io/YOUR_PROJECT_ID/kestrel-lora:latest` - GCS-only LoRA source (stable)
- `gcr.io/YOUR_PROJECT_ID/kestrel-lora:ipfs-v1` - IPFS + GCS support (sovereign)

### Generation Job Submission

**From IPFS (Recommended - Sovereign Storage):**
```python
from features.vertex_ai.vertex_ai_manager import VertexAIManager

manager = VertexAIManager()
job = await manager.submit_generation_job(
    lora_ipfs_cid="QmWfQLVhZb1ysotcZMKTWQQUs93iZMzKxX1Aumroj8FBCN",
    prompt="A photo of TOK379f40b3 woman, professional nurse in blue scrubs...",
    trigger_word="TOK379f40b3",
    output_gcs_prefix="gs://kestrel-training/generation/{companion_id}/selfie_001",
    image_tag="ipfs-v1",  # Use IPFS-enabled image
)
```

**From GCS (Legacy):**
```python
job = await manager.submit_generation_job(
    lora_gcs_path="gs://kestrel-training/training/{companion_id}/pytorch_lora_weights.safetensors",
    prompt="A photo of TOK{id} woman...",
    trigger_word="TOK{id}",
    output_gcs_prefix="gs://kestrel-training/generation/{companion_id}/selfie_001",
    image_tag="latest",  # GCS-only image
)
```

### Expected Timing (A100 80GB with int8-quanto)
- Job submission: ~10s
- Container startup: ~30s
- Model loading: ~2-3 min
- LoRA loading: ~10s
- Image generation: ~30-60s per image
- **Total: ~4-5 min per job**

### Generation Jobs History

| Date | Companion | Job ID | Status | Source | Notes |
|------|-----------|--------|--------|--------|-------|
| 2025-12-30 | Lisa | (earlier test) | ✅ Complete | GCS | Selfie in gallery |
| 2025-12-30 | Leah | 8140047195924594688 | ❌ Failed | GCS | LoRA not in GCS (only IPFS) |
| 2025-12-30 | Lela | 8822342539471224832 | ❌ Failed | GCS | LoRA not in GCS (only IPFS) |
| 2025-12-30 | Lila | 4722237291215454208 | ❌ Failed | GCS | LoRA not in GCS (only IPFS) |
| 2025-12-30 | Leah | 5996193035807883264 | ❌ Failed | IPFS | dweb.link gateway timeout |
| 2025-12-30 | Lela | 4391785668557144064 | ❌ Failed | IPFS | dweb.link gateway timeout |
| 2025-12-30 | Lila | 4776843436697321472 | ❌ Failed | IPFS | dweb.link gateway timeout |
| 2025-12-30 | Leah | 7852063110377504768 | ⏳ Pending | Lighthouse | files.lighthouse.storage gateway |
| 2025-12-30 | Lela | 340058931923517440 | ⏳ Pending | Lighthouse | files.lighthouse.storage gateway |
| 2025-12-30 | Lila | 932282282922737664 | ⏳ Pending | Lighthouse | files.lighthouse.storage gateway |

### Monitoring Jobs
```bash
# Check job status via gcloud
gcloud ai custom-jobs describe {job_id} --region=us-central1 --project=YOUR_PROJECT_ID

# Or via Python
status = await manager.get_job_status(job_id)
print(status['state'])  # pending, running, completed, failed
```

---
*Last Updated: December 30, 2025*
