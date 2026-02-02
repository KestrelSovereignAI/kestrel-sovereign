#!/usr/bin/env python3
"""
SimpleTuner API Wrapper for FLUX.1-dev Training with Uncensored Support

BACKWARDS COMPATIBILITY WRAPPER - This file now imports the refactored FLUX.1 implementation.
The actual implementation has been moved to flux1_api.py as part of consolidation effort.

Wraps SimpleTuner's FLUX.1-dev training in a REST API for RunPod/Vertex AI deployment.
This version uses FLUX.1-dev (not FLUX.2-dev) to enable uncensored content generation
via the enhanceaiteam/Flux-Uncensored-V2 LoRA adapter.

Key Features:
  - FLUX.1-dev base model (black-forest-labs/FLUX.1-dev)
  - Character LoRA training via SimpleTuner
  - Uncensored LoRA (enhanceaiteam/Flux-Uncensored-V2) for NSFW generation
  - Multi-LoRA support: combines character + uncensored adapters

API Endpoints:
  POST /train      - Start LoRA training
  GET  /status     - Get training status
  GET  /health     - Health check
  GET  /download   - Download trained LoRA

Environment Variables:
  FLUX_UNCENSORED=true  - Enable uncensored LoRA (default: true)
  HF_TOKEN              - HuggingFace token for gated models

IMPORTANT: RunPod mode requires network volume mounted at /workspace.
           NO FALLBACKS - if /workspace is not mounted, the service will fail.
           This is intentional to avoid silent failures with ephemeral storage.
"""

# Import the refactored FLUX.1 API implementation
from flux1_api import FLUX1API, setup_paths
import asyncio
import logging
import sys

logger = logging.getLogger(__name__)

# Create API instance
api = None

if __name__ == "__main__":
    # Initialize the API instance
    api = FLUX1API()

    # Use the main execution from the FLUX1API class
    import argparse

    # Check if any arguments are provided for batch mode
    if len(sys.argv) > 1:
        # Delegate to FLUX1API implementation with sys.argv
        # This preserves the original command-line interface
        original_name = sys.argv[0]
        sys.argv[0] = "flux1_api.py"  # Set expected name for argument parsing
        try:
            # Import and run the main from flux1_api
            from flux1_api import main
            asyncio.run(main())
        except SystemExit:
            # This is expected when batch modes complete
            pass
        finally:
            sys.argv[0] = original_name
    else:
        # Default API server mode - setup paths and run
        setup_paths(is_vertex_mode=False)
        api.run_server(host="0.0.0.0", port=8000)