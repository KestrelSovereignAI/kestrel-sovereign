# Lane Brief: Cloud Training Deployment

Goal: reconcile Cloud Run deployment docs with external cloud provider packages and research-era GPU/training docs.

Start with:

- `docs/deployment/README.md`
- `docs/architecture/TRAINING_PROVIDER_ARCHITECTURE.md`
- `docs/architecture/PLAN_RUNPOD_INTEGRATION.md`
- `docs/architecture/RUNPOD_LORA_TRAINING.md`
- `docs/architecture/VASTAI_TRAINING.md`
- `docs/research/LoRA/`
- `pyproject.toml`

Check for:

- old module paths such as `features.runpod.*`
- provider integrations described as in-core when extracted
- Cloud Run commands that disagree with the current `kestrel deploy` CLI
- research notes presented as operational runbooks
- training adapters whose ownership moved to external packages

Report to: `reports/cloud_training_deployment_report.md`

