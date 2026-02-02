"""
SimpleTuner Training Logic.

Contains SimpleTuner configuration generation and training execution.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_MODEL_PULL
from . import config as app_config

logger = logging.getLogger(__name__)


def create_simpletuner_config(
    job_id: str,
    trigger_word: str,
    dataset_path: str,
    output_path: str,
    cache_path: str,
    steps: int = 1000,
    lora_rank: int = 16,
) -> dict:
    """
    Create SimpleTuner config for FLUX.2 training.

    Based on SimpleTuner's FLUX2.md quickstart guide.
    Optimized for A100 80GB.
    """
    return {
        # Model config - FLUX.2 requires model_family="flux2" per SimpleTuner docs
        "model_family": "flux2",
        "model_flavour": "dev",
        "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-dev",

        # Training type
        "model_type": "lora",
        "lora_rank": lora_rank,
        "lora_alpha": lora_rank,  # Typically same as rank

        # Training params - REQUIRED
        "optimizer": "adamw_bf16",
        "num_train_epochs": 0,  # Required when using max_train_steps
        "max_train_steps": steps,
        "learning_rate": 1e-4,
        "lr_scheduler": "constant",
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,  # Reduced from 4 for faster iteration

        # Data backend config path - REQUIRED
        "data_backend_config": f"{output_path}/config/multidatabackend.json",

        # Memory optimization for A100 80GB - FLUX.2 quickstart recommended settings
        # With int8 quantization, total memory ~52GB - fits on 80GB without CPU offload
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,

        # Quantize both transformer and text encoder to int8
        # This is the recommended config for 80GB GPU per FLUX2.md quickstart
        "base_model_precision": "int8-quanto",  # Quantize transformer
        "text_encoder_1_precision": "int8-quanto",  # Quantize Mistral-24B text encoder
        "quantize_via": "accelerator",  # GPU quantization to avoid OOM

        # FLUX.2 guidance settings
        "flux_guidance_mode": "constant",
        "flux_guidance_value": 1.0,

        # Output
        "output_dir": output_path,

        # Validation - disabled during training for speed, final only
        "validation_prompt": f"a portrait photo of {trigger_word}, high quality",
        "validation_steps": 0,  # Disable mid-training validation for speed
        "validation_resolution": 1024,

        # Checkpointing - save at end of training
        # Set to steps value so checkpoint saves once when max_train_steps is reached
        "checkpoint_step_interval": steps,  # Save final checkpoint
        "checkpoints_total_limit": 1,  # Keep only final checkpoint

        # Logging - TensorBoard for local monitoring
        "logging_dir": f"{output_path}/logs",
        "report_to": "tensorboard",

        # Debug options - enable verbose logging
        "debug_dataset_loader": True,
        "print_filenames": True,

        # Cache paths
        "cache_dir_vae": f"{cache_path}/vae",
    }


def create_multidatabackend_config(
    trigger_word: str,
    dataset_path: str,
    cache_path: str,
    steps: int = 500,
) -> list:
    """
    Create SimpleTuner multidatabackend.json config.

    Repeats is set to match training steps so training completes at max_train_steps.
    For single-image training, repeats should be >= steps to ensure enough samples.
    """
    # Calculate repeats to match training steps
    # For 1 image, repeats should be >= steps to have enough training samples
    # Add 10% buffer to avoid running out of samples before max_train_steps
    repeats = int(steps * 1.1) + 10

    return [
        {
            "id": "main_dataset",
            "type": "local",
            "instance_data_dir": dataset_path,
            "caption_strategy": "instanceprompt",
            "instance_prompt": f"a photo of {trigger_word}",
            "resolution": 1024,
            "minimum_image_size": 512,
            "maximum_image_size": 2048,
            "target_downsample_size": 1024,
            "resolution_type": "pixel_area",
            "prepend_instance_prompt": False,
            "cache_dir_vae": f"{cache_path}/vae",
            "cache_dir_text": f"{cache_path}/text",
            "disabled": False,
            "skip_file_discovery": "",  # Empty string = don't skip (SimpleTuner expects string, not bool)
            "preserve_data_backend_cache": True,
            # Repeat image to have enough samples for max_train_steps
            "repeats": repeats,
        }
    ]


async def run_training(job_id: str, config: dict, dataset_path: str):
    """
    Run SimpleTuner training in background.

    Args:
        job_id: Unique job identifier
        config: SimpleTuner configuration dict
        dataset_path: Path to training dataset
    """
    app_config.current_job = job_id

    job = app_config.training_jobs[job_id]
    job["status"] = "preparing"
    job["progress"] = 0.05

    try:
        output_path = config["output_dir"]
        cache_path = f"{output_path}/cache"

        # Create directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(cache_path, exist_ok=True)
        os.makedirs(f"{cache_path}/vae", exist_ok=True)
        os.makedirs(f"{cache_path}/text", exist_ok=True)

        # Write config files
        config_dir = f"{output_path}/config"
        os.makedirs(config_dir, exist_ok=True)

        # Main config as JSON (SimpleTuner accepts JSON or env)
        config_path = f"{config_dir}/config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Multidatabackend config - pass steps so repeats matches max_train_steps
        multidata_config = create_multidatabackend_config(
            trigger_word=job["trigger_word"],
            dataset_path=dataset_path,
            cache_path=cache_path,
            steps=config["max_train_steps"],
        )
        multidata_path = f"{config_dir}/multidatabackend.json"
        with open(multidata_path, "w") as f:
            json.dump(multidata_config, f, indent=2)

        logger.info(f"Config written to {config_path}")
        logger.info(f"Multidatabackend written to {multidata_path}")

        job["status"] = "training"
        job["progress"] = 0.1

        env = os.environ.copy()

        # SimpleTuner CLI: `simpletuner train --env <config_dir>`
        # This runs accelerate launch internally and captures output
        cmd = [
            "simpletuner", "train",
            "--env", config_dir,
        ]

        logger.info(f"Starting SimpleTuner: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        # Monitor progress by reading output
        # Use read() with size limit instead of readline() to avoid "separator not found" errors
        # tqdm progress bars can produce very long lines that exceed readline's buffer
        steps_total = config["max_train_steps"]
        current_step = 0
        buffer = ""

        while True:
            try:
                # Read chunks instead of lines to handle tqdm progress bars
                chunk = await asyncio.wait_for(
                    process.stdout.read(4096),  # Read up to 4KB at a time
                    timeout=HTTP_TIMEOUT_MODEL_PULL
                )
                if not chunk:
                    # EOF - process finished
                    break

                # Decode and add to buffer
                buffer += chunk.decode(errors='replace')

                # Process complete lines from buffer
                while '\n' in buffer:
                    line_str, buffer = buffer.split('\n', 1)
                    line_str = line_str.strip()

                    # Skip empty lines and carriage returns (tqdm updates)
                    if not line_str or line_str == '\r':
                        continue

                    # Clean up tqdm carriage returns within line
                    if '\r' in line_str:
                        # Take the last segment after carriage return (final tqdm state)
                        line_str = line_str.split('\r')[-1].strip()
                        if not line_str:
                            continue

                    logger.info(f"[SimpleTuner] {line_str}")

                    # Parse step progress from output
                    # SimpleTuner outputs: "Step 100/1000 - loss: 0.xxx"
                    if "Step " in line_str and "/" in line_str:
                        try:
                            step_part = line_str.split("Step ")[1].split(" ")[0]
                            current_step = int(step_part.split("/")[0])
                            progress = 0.1 + (current_step / steps_total) * 0.8
                            job["progress"] = min(progress, 0.9)
                            job["current_step"] = current_step
                        except (IndexError, ValueError):
                            pass

                    # Also check for percentage-based progress (e.g., "50%|████")
                    if "%" in line_str and "|" in line_str:
                        try:
                            pct_str = line_str.split("%")[0].strip()
                            # Get the last number before %
                            pct = int(pct_str.split()[-1])
                            if pct > 0:
                                progress = 0.1 + (pct / 100) * 0.8
                                job["progress"] = min(progress, 0.9)
                                job["current_step"] = int(pct * steps_total / 100)
                        except (IndexError, ValueError):
                            pass

            except asyncio.TimeoutError:
                # Check if process is still running
                if process.returncode is not None:
                    break
                logger.warning("No output for 10 minutes, continuing to wait...")
                continue
            except Exception as e:
                logger.warning(f"Error reading output: {e}")
                # Check if process is still alive before breaking
                if process.returncode is not None:
                    break
                continue

        await process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"SimpleTuner exited with code {process.returncode}")

        # Find the output LoRA file
        job["status"] = "finalizing"
        job["progress"] = 0.95

        # SimpleTuner outputs to output_dir/pytorch_lora_weights.safetensors
        lora_file = Path(output_path) / "pytorch_lora_weights.safetensors"
        if not lora_file.exists():
            # Check for checkpoint directories
            for checkpoint_dir in Path(output_path).glob("checkpoint-*"):
                potential_lora = checkpoint_dir / "pytorch_lora_weights.safetensors"
                if potential_lora.exists():
                    lora_file = potential_lora
                    break

        if not lora_file.exists():
            raise RuntimeError(f"LoRA file not found in {output_path}")

        job["lora_path"] = str(lora_file)
        job["status"] = "completed"
        job["progress"] = 1.0
        job["completed_at"] = datetime.utcnow().isoformat()

        logger.info(f"Training completed! LoRA at: {lora_file}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.utcnow().isoformat()

    finally:
        app_config.current_job = None
