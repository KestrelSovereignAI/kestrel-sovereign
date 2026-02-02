#!/usr/bin/env python3
"""
SimpleTuner API Wrapper for Kestrel LoRA Training

BACKWARDS COMPATIBILITY WRAPPER - This file now imports the refactored FLUX.2 implementation.
The actual implementation has been moved to flux2_api.py as part of consolidation effort.

Wraps SimpleTuner's FLUX.2 training in a REST API for RunPod deployment.
SimpleTuner handles all the FLUX.2 architecture details correctly.

API Endpoints:
  POST /train      - Start LoRA training
  GET  /status     - Get training status
  GET  /health     - Health check
  GET  /download   - Download trained LoRA

IMPORTANT: RunPod mode requires network volume mounted at /workspace.
           NO FALLBACKS - if /workspace is not mounted, the service will fail.
           This is intentional to avoid silent failures with ephemeral storage.
"""

# Import the refactored FLUX.2 API implementation
from flux2_api import FLUX2API, setup_paths
import asyncio
import logging

logger = logging.getLogger(__name__)

# Create API instance
api = None

if __name__ == "__main__":
    # Initialize the API instance
    api = FLUX2API()

    # Use the main execution from the FLUX2API class
    import argparse
    import sys

    # Check if any arguments are provided for batch mode
    if len(sys.argv) > 1:
        # Delegate to FLUX2API implementation with sys.argv
        # This preserves the original command-line interface
        original_name = sys.argv[0]
        sys.argv[0] = "flux2_api.py"  # Set expected name for argument parsing
        try:
            # Import and run the main from flux2_api
            from flux2_api import main
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
