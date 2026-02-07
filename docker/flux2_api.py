#!/usr/bin/env python3
"""
FLUX.2 SimpleTuner API — FLUX.2 [dev]-specific configuration.

All shared logic lives in base_simpletuner_api.py. This file only contains:
- Model identifiers and pipeline classes
- Model-specific config overrides
"""

import logging

from base_simpletuner_api import BaseSimpleTunerAPI, run_main, WORKSPACE_PATH

from kestrel_sovereign.kestrel_config.constants import (
    TRAINING_TIMEOUT,
)

logger = logging.getLogger(__name__)


class Flux2SimpleTunerAPI(BaseSimpleTunerAPI):
    """FLUX.2 [dev] implementation."""

    def __init__(self):
        super().__init__(
            app_title="Kestrel FLUX.2 LoRA Training API",
            service_name="kestrel-flux2-simpletuner",
        )
        # Override quantized cache paths for FLUX.2
        self.quantized_cache_dir = f"{WORKSPACE_PATH}/quantized_flux2"
        self.quantized_transformer_path = f"{self.quantized_cache_dir}/transformer_int8"
        self.quantized_text_encoder_path = f"{self.quantized_cache_dir}/text_encoder_int8"
        self.quantized_marker_path = f"{self.quantized_cache_dir}/.quantized_complete"

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def get_model_family(self) -> str:
        return "flux"

    def get_model_name(self) -> str:
        return "black-forest-labs/FLUX.2-dev"

    def get_display_name(self) -> str:
        return "FLUX.2 [dev]"

    def get_pipeline_class(self):
        from diffusers import FluxPipeline
        return FluxPipeline

    def get_transformer_class(self):
        from diffusers import FluxTransformer2DModel
        return FluxTransformer2DModel

    def get_text_encoder_class(self):
        from transformers import T5EncoderModel
        return T5EncoderModel

    def get_quantized_cache_dir(self) -> str:
        return f"{WORKSPACE_PATH}/quantized_flux2"

    def get_gcs_cache_prefix(self) -> str:
        return "quantized-cache/flux2-dev-int8"

    def get_training_timeout(self) -> int:
        return TRAINING_TIMEOUT

    def get_cached_model_filename(self) -> str:
        return "transformer/config.json"

    def create_model_specific_config(self, base_config: dict) -> dict:
        base_config["model_family"] = "flux"
        base_config["model_flavour"] = "dev"
        return base_config


if __name__ == "__main__":
    run_main(Flux2SimpleTunerAPI, "Kestrel FLUX.2 LoRA Training API")
