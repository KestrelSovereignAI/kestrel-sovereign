#!/usr/bin/env python3
"""
SimpleTuner API Wrapper for FLUX.1-dev Training

BACKWARDS COMPATIBILITY WRAPPER - This file now imports the refactored FLUX.1 implementation.
The actual implementation has been moved to flux1_api.py as part of consolidation effort.

Wraps SimpleTuner's FLUX.1-dev training in a REST API for RunPod/Vertex AI deployment.
This version uses FLUX.1-dev (not FLUX.2-dev) and supports optional
operator-supplied auxiliary LoRA composition.

Key Features:
  - FLUX.1-dev base model (black-forest-labs/FLUX.1-dev)
  - Character LoRA training via SimpleTuner
  - Optional multi-LoRA support: combines character + auxiliary adapters

API Endpoints:
  POST /train      - Start LoRA training
  GET  /status     - Get training status
  GET  /health     - Health check
  GET  /download   - Download trained LoRA

Environment Variables:
  FLUX_AUX_LORA_REPO      - Optional auxiliary LoRA repository
  FLUX_AUX_LORA_FILENAME  - Auxiliary LoRA filename (default: lora.safetensors)
  FLUX_AUX_LORA_WEIGHT    - Auxiliary LoRA blend weight (default: 0.8)
  HF_TOKEN                - HuggingFace token for gated models

IMPORTANT: RunPod mode requires network volume mounted at /workspace.
           NO FALLBACKS - if /workspace is not mounted, the service will fail.
           This is intentional to avoid silent failures with ephemeral storage.
"""

# Import the refactored FLUX.1 API implementation
from base_simpletuner_api import setup_paths
from flux1_api import Flux1SimpleTunerAPI, run_main
import logging
import sys

logger = logging.getLogger(__name__)

# Create API instance
api = None

if __name__ == "__main__":
    # Initialize the API instance
    api = Flux1SimpleTunerAPI()

    # Use the main execution from the FLUX1API class
    # Check if any arguments are provided for batch mode
    if len(sys.argv) > 1:
        # Delegate to the refactored implementation with sys.argv
        # This preserves the original command-line interface
        original_name = sys.argv[0]
        sys.argv[0] = "flux1_api.py"  # Set expected name for argument parsing
        try:
            run_main(Flux1SimpleTunerAPI, "Kestrel FLUX.1 LoRA Training API")
        except SystemExit:
            # This is expected when batch modes complete
            pass
        finally:
            sys.argv[0] = original_name
    else:
        # Default API server mode - setup paths and run
        setup_paths(is_vertex_mode=False)
        api.run_server(host="0.0.0.0", port=8000)
