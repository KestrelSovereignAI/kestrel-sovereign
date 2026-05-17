#!/usr/bin/env python3
"""
FLUX.1 SimpleTuner API — FLUX.1 [dev]-specific configuration and overrides.

All shared logic lives in base_simpletuner_api.py. This file only contains:
- Model identifiers and pipeline classes
- Optional auxiliary LoRA composition (FLUX.1-specific)
- Model-specific config overrides
"""

import os
import logging

from base_simpletuner_api import BaseSimpleTunerAPI, run_main, WORKSPACE_PATH

from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_MODEL_PULL,
)

logger = logging.getLogger(__name__)

# FLUX.1-specific: optional auxiliary LoRA composition.
AUXILIARY_LORA_REPO = os.getenv("FLUX_AUX_LORA_REPO", "")
AUXILIARY_LORA_FILENAME = os.getenv("FLUX_AUX_LORA_FILENAME", "lora.safetensors")
AUXILIARY_LORA_WEIGHT = float(os.getenv("FLUX_AUX_LORA_WEIGHT", "0.8"))


class Flux1SimpleTunerAPI(BaseSimpleTunerAPI):
    """FLUX.1 [dev] implementation with optional auxiliary LoRA support."""

    def __init__(self):
        super().__init__(
            app_title="Kestrel FLUX.1 LoRA Training API",
            service_name="kestrel-flux1-simpletuner",
        )
        # Override quantized cache paths for FLUX.1
        self.quantized_cache_dir = f"{WORKSPACE_PATH}/quantized_flux"
        self.quantized_transformer_path = f"{self.quantized_cache_dir}/transformer_int8"
        self.quantized_text_encoder_path = f"{self.quantized_cache_dir}/text_encoder_int8"
        self.quantized_marker_path = f"{self.quantized_cache_dir}/.quantized_complete"

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def get_model_family(self) -> str:
        return "flux"

    def get_model_name(self) -> str:
        return "black-forest-labs/FLUX.1-dev"

    def get_display_name(self) -> str:
        return "FLUX.1 [dev]"

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
        return f"{WORKSPACE_PATH}/quantized_flux"

    def get_gcs_cache_prefix(self) -> str:
        return "quantized-cache/flux1-dev-int8"

    def get_training_timeout(self) -> int:
        return HTTP_TIMEOUT_MODEL_PULL

    def get_cached_model_filename(self) -> str:
        return "transformer/config.json"

    def create_model_specific_config(self, base_config: dict) -> dict:
        base_config["model_family"] = "flux"
        base_config["model_flavour"] = "dev"
        return base_config

    # =========================================================================
    # FLUX.1-specific: optional auxiliary LoRA support
    # =========================================================================

    def load_generation_loras(self, pipe, local_lora_path: str):
        """Load LoRA adapters with optional auxiliary LoRA composition."""
        # Load primary (companion) LoRA as "companion" adapter
        pipe.load_lora_weights(
            os.path.dirname(local_lora_path),
            weight_name=os.path.basename(local_lora_path),
            adapter_name="companion",
        )

        if not AUXILIARY_LORA_REPO:
            pipe.set_adapters(["companion"], adapter_weights=[1.0])
            return

        # Optionally load an operator-specified auxiliary LoRA.
        try:
            logger.info(f"Loading auxiliary LoRA: {AUXILIARY_LORA_REPO}")
            pipe.load_lora_weights(
                AUXILIARY_LORA_REPO,
                weight_name=AUXILIARY_LORA_FILENAME,
                adapter_name="auxiliary",
            )
            # Compose both adapters
            pipe.set_adapters(
                ["companion", "auxiliary"],
                adapter_weights=[1.0, AUXILIARY_LORA_WEIGHT],
            )
            logger.info(f"Composed LoRAs: companion=1.0, auxiliary={AUXILIARY_LORA_WEIGHT}")
        except Exception as e:
            logger.warning(f"Failed to load auxiliary LoRA, using companion only: {e}")
            pipe.set_adapters(["companion"], adapter_weights=[1.0])

    def get_generation_metadata_extras(self) -> dict:
        if not AUXILIARY_LORA_REPO:
            return {}
        return {
            "auxiliary_lora": AUXILIARY_LORA_REPO,
            "auxiliary_weight": AUXILIARY_LORA_WEIGHT,
        }


if __name__ == "__main__":
    run_main(Flux1SimpleTunerAPI, "Kestrel FLUX.1 LoRA Training API")
