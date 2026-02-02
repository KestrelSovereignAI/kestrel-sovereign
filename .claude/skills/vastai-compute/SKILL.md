---
name: vastai-compute
description: Use when provisioning GPU instances on Vast.ai for training, inference, or compute tasks. Handles instance search, creation, SSH connection, and management via the vastai CLI.
---

# Vast.ai GPU Compute Skill

This skill helps provision and manage GPU instances on Vast.ai for ML training, inference, and compute workloads.

## Trigger Keywords
- "vast.ai", "vastai", "vast ai"
- "GPU instance", "rent GPU", "cloud GPU"
- "training instance", "inference server"
- "cheap GPU", "spot GPU"

## Prerequisites

### Installation

**CLI only:**
```bash
# Install via uv
uv pip install --upgrade vastai

# Or download directly
wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast
chmod +x vast
```

**Python SDK (for programmatic access):**
```bash
uv pip install vastai-sdk
```

### Authentication
1. Get API key from: https://cloud.vast.ai/cli/
2. Set the key:
```bash
# CLI method (saves to ~/.vast_api_key)
vastai set api-key <your-api-key>

# Or environment variable
export VASTAI_API_KEY=<your-api-key>
```

### Private Docker Registry (GCR)

For images from Google Container Registry (gcr.io), set the service account key:

```bash
# Option 1: Set the raw JSON key as env var
export GCR_SERVICE_ACCOUNT_KEY='{"type":"service_account","project_id":"...",...}'

# Option 2: Load from file
export GCR_SERVICE_ACCOUNT_KEY=$(cat /path/to/service-account.json)
```

The VastAIManager will automatically detect GCR images and authenticate using:
- Username: `_json_key`
- Password: Service account JSON from `GCR_SERVICE_ACCOUNT_KEY`
- Registry: `gcr.io`

For other private registries, set `docker_login` in the profile config:
```toml
[profiles.myprofile]
docker_login = "-u myuser -p mypassword docker.io"
```

### Python SDK Usage
```python
from vastai_sdk import VastAI

vast = VastAI(api_key='your-api-key')

# Search offers
offers = vast.search_offers(query='gpu_ram >= 24 reliability > 0.9')

# Create instance
result = vast.create_instance(
    ID=offer_id,
    image='pytorch/pytorch',
    disk=50,
    onstart='pip install transformers'
)

# Manage instances
vast.show_instances()
vast.start_instance(ID=12345)
vast.stop_instance(ID=12345)
vast.destroy_instance(id=12345)
```

## Common Operations

### Search for Instances

**Basic search:**
```bash
vastai search offers
```

**Find high-reliability instances with multiple GPUs:**
```bash
vastai search offers 'reliability > 0.99 num_gpus>=4' -o 'num_gpus-'
```

**Find Ampere GPUs (compute capability 8.0+):**
```bash
vastai search offers 'compute_cap > 800'
```

**Find cheap instances sorted by price:**
```bash
vastai search offers 'gpu_ram >= 24' -o 'dph+'
```

**Common filters:**
- `reliability > 0.99` - High reliability score (0-1)
- `num_gpus >= 2` - Minimum GPU count
- `gpu_ram >= 24` - Minimum GPU memory (GB)
- `compute_cap > 800` - Ampere or newer (RTX 30xx, A100, etc.)
- `cuda_vers >= 12.0` - CUDA version
- `inet_down > 100` - Download speed (Mbps)
- `disk_space >= 50` - Available disk (GB)
- `rentable = true` - Only rentable instances (recommended)
- `verified = true` - Only verified hosts
- `dph <= 0.50` - Max price per hour
- `gpu_name in [RTX_3090, RTX_4090]` - Specific GPU models

**Sort options (-o):**
- `dph+` - Price ascending (cheapest first)
- `dph-` - Price descending
- `num_gpus-` - GPU count descending
- `gpu_ram-` - GPU memory descending
- `reliability-` - Reliability descending

### Create Instance

**Basic instance:**
```bash
vastai create instance <offer_id> --image pytorch/pytorch --disk 32
```

**With custom Docker image:**
```bash
vastai create instance <offer_id> \
  --image nvcr.io/nvidia/pytorch:24.01-py3 \
  --disk 100 \
  --onstart-cmd "pip install transformers accelerate"
```

**Common images:**
- `pytorch/pytorch` - Official PyTorch
- `nvcr.io/nvidia/pytorch:24.01-py3` - NVIDIA PyTorch container
- `huggingface/transformers-pytorch-gpu` - HuggingFace
- `tensorflow/tensorflow:latest-gpu` - TensorFlow

### Connect to Instance

**SSH connection:**
```bash
# Get SSH command
ssh $(vastai ssh-url <instance_id>)

# Or manually construct
vastai show instance <instance_id>
# Then use the ssh_host and ssh_port from output
```

### Instance Management

**List your instances:**
```bash
vastai show instances
```

**Stop an instance:**
```bash
vastai stop instance <instance_id>
```

**Destroy an instance:**
```bash
vastai destroy instance <instance_id>
```

**Execute commands on inactive instance:**
```bash
vastai execute <instance_id> 'ls -l'
vastai execute <instance_id> 'rm filename.txt'
vastai execute <instance_id> 'du -d1 -h'
```

### Additional Commands

**Data Transfer:**
```bash
# Copy between local and instance
vastai copy /local/path instance_id:/remote/path
vastai copy instance_id:/remote/path /local/path

# Cloud sync (S3, Google Drive)
vastai cloud copy --src s3://bucket/path --dst /workspace
```

**Instance Control:**
```bash
vastai reboot instance <instance_id>    # Restart without losing GPU
vastai label instance <instance_id> "my-label"
vastai change bid <instance_id> <new_price>  # Adjust spot pricing
```

## Storage Types

**Container Storage (default):**
- Fixed at creation, cannot resize
- Deleted when instance is destroyed
- Minimum 10GB
- Use for: temporary files, cache, build artifacts

**Local Volumes:**
- Persistent across instance restarts
- Tied to physical machine (cannot migrate)
- Survives instance destruction
- Use for: trained models, datasets, important data

**IMPORTANT:** Vast.ai volumes are LOCAL only - they cannot be moved between machines like RunPod network volumes. Use cloud sync (S3, GCS) for data that needs to persist across different hosts.

## Kestrel-Specific Workflows

### LoRA Training Instance

For training LoRA adapters:
```bash
# Find A100/H100 with good reliability
vastai search offers 'gpu_name in [A100, H100] reliability > 0.95 gpu_ram >= 40' -o 'dph+'

# Create with training image
vastai create instance <offer_id> \
  --image nvcr.io/nvidia/pytorch:24.01-py3 \
  --disk 100 \
  --onstart-cmd "pip install transformers peft accelerate bitsandbytes"
```

### Inference Server

For running inference:
```bash
# Find cheaper GPU with good memory
vastai search offers 'gpu_ram >= 24 reliability > 0.9' -o 'dph+'

# Create with vLLM
vastai create instance <offer_id> \
  --image vllm/vllm-openai:latest \
  --disk 50 \
  --env HUGGING_FACE_HUB_TOKEN=<token>
```

## Cost Management

- Use `--disk` to minimize storage costs
- Destroy instances when not in use
- Consider interruptible instances for training (cheaper)
- Monitor with `vastai show instances`

## Help & Documentation

```bash
vastai --help
vastai search offers --help
vastai create instance --help
```

Full CLI docs: https://vast.ai/docs/cli/commands

## Vast.ai vs RunPod Comparison

| Feature | Vast.ai | RunPod |
|---------|---------|--------|
| **Pricing** | Marketplace (variable, often cheaper) | Fixed pricing, predictable |
| **Reliability** | Variable (peer-to-peer hosts) | Generally higher (managed) |
| **API** | CLI + Python SDK (`vastai-sdk`) | Python SDK (`runpod`) + CLI |
| **Best For** | Cost-sensitive batch jobs | Production inference |
| **Instance Resume** | No (must create new) | Yes, fast (~10-30s) |
| **Persistent Storage** | Local volumes only (machine-tied) | Network volumes (portable) |
| **Templates** | Docker images only | Templates + Pods |
| **SSH Access** | Yes | Yes |
| **Spot/Bid Pricing** | Yes (`change bid`) | Community cloud |

**When to use Vast.ai:**
- Budget-conscious training runs
- Batch processing jobs
- Exploratory work / prototyping
- When you need specific GPU configurations

**When to use RunPod:**
- Production inference servers
- Time-sensitive workloads
- Need reliable uptime
- Using Kestrel's built-in RunPod integration

## Integration with Kestrel

Kestrel has both RunPod and Vast.ai integration:

**RunPod:** `features/runpod/` - Production-ready, pod resume support
**Vast.ai:** `features/vastai/` - Cost-optimized, marketplace model

### VastAIFeature Commands

```bash
# Check status
!vastai status

# Search for GPU offers
!vastai search query="gpu_ram >= 24"
!vastai search profile=training

# Start an instance
!vastai on profile=training
!vastai on profile=ollama

# Stop and destroy instance
!vastai off

# List all your instances
!vastai list

# Get SSH connection URL
!vastai ssh
```

### Configuration

Edit `vastai_config.toml` to customize GPU profiles:
- `[profiles.training]` - LoRA/FLUX training
- `[profiles.inference]` - Fast inference
- `[profiles.ollama]` - Ollama server
- `[profiles.llm]` - Large LLM inference
- `[profiles.budget]` - Cheap testing

## Recommendation

For complex provisioning workflows, consider creating automation scripts in `scripts/vastai/` that handle:
1. Instance search with project-specific requirements
2. Automatic SSH key configuration
3. Environment setup and dependency installation
4. Job monitoring and cleanup
