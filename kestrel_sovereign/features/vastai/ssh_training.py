"""
Vast.ai SSH-Based Training and Inference.

Provides LoRA training and image generation via SSH for Kohya instances.
These methods execute commands directly on GPU instances via SSH.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from kestrel_sovereign.kestrel_config.constants import (
    SSH_COMMAND_TIMEOUT_SHORT,
    SSH_COMMAND_TIMEOUT_DEFAULT,
    SSH_COMMAND_TIMEOUT_MEDIUM,
    SSH_COMMAND_TIMEOUT_SETUP,
    SSH_COMMAND_TIMEOUT_LONG,
    SSH_COMMAND_TIMEOUT_GENERATION,
)
from .models import VastAIManagerError, VastAISession

logger = logging.getLogger(__name__)


class VastAISSHTrainingMixin:
    """SSH-based training and inference methods for Vast.ai instances."""

    _lock: asyncio.Lock
    _session: Optional[VastAISession]

    async def run_ssh_command(
        self,
        command: str,
        session: Optional[VastAISession] = None,
        timeout: int = 300,
    ) -> str:
        """
        Execute a command on the instance via SSH.

        Args:
            command: Shell command to execute
            session: Session to use (defaults to current session)
            timeout: Command timeout in seconds

        Returns:
            Command stdout

        Raises:
            VastAIManagerError: If SSH fails or command times out
        """
        if session is None:
            async with self._lock:
                session = self._session

        if not session:
            raise VastAIManagerError("No active session for SSH command")

        if not session.ssh_host or not session.ssh_port:
            raise VastAIManagerError(
                f"SSH not available for instance {session.instance_id}"
            )

        ssh_cmd = (
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
            f"-p {session.ssh_port} root@{session.ssh_host} "
            f'"{command}"'
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            if proc.returncode != 0:
                err_msg = stderr.decode() if stderr else "Unknown error"
                raise VastAIManagerError(f"SSH command failed: {err_msg}")

            return stdout.decode()

        except asyncio.TimeoutError:
            raise VastAIManagerError(f"SSH command timed out after {timeout}s")
        except Exception as e:
            raise VastAIManagerError(f"SSH failed: {e}") from e

    async def submit_training_job(
        self,
        session: VastAISession,
        image_url: str,
        companion_id: str,
        trigger_word: str = "sks person",
        num_repeats: int = 10,
        max_train_epochs: int = 1,
        network_dim: int = 4,
        learning_rate: float = 1e-4,
    ) -> str:
        """
        Submit a LoRA training job to a Kohya instance.

        Args:
            session: Active Vast.ai session with Kohya
            image_url: URL of training image(s)
            companion_id: Companion ID for output naming
            trigger_word: Trigger word for LoRA (default: "sks person")
            num_repeats: Number of repeats per image
            max_train_epochs: Training epochs
            network_dim: LoRA rank (4 = small/fast, 16 = more detail)
            learning_rate: Learning rate

        Returns:
            Job ID (companion_id used as job ID)
        """
        from kestrel_sovereign.kestrel_config.constants import DEFAULT_TRAINING_BATCH_SIZE

        job_id = companion_id

        # Create training data directory structure
        setup_cmd = f"""
mkdir -p /workspace/training_data/{job_id}
mkdir -p /workspace/lora_output

# Download training image
cd /workspace/training_data/{job_id}
wget -q -O image_001.png '{image_url}'

# Create caption file with trigger word
echo '{trigger_word}' > image_001.txt

# Create dataset config
cat > /workspace/training_data/dataset_{job_id}.toml << 'DATASET_EOF'
[general]
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = 1024
batch_size = {DEFAULT_TRAINING_BATCH_SIZE}

[[datasets.subsets]]
image_dir = '/workspace/training_data/{job_id}'
caption_extension = '.txt'
num_repeats = {num_repeats}
DATASET_EOF
"""

        logger.info(f"Setting up training data for {job_id}")
        await self.run_ssh_command(setup_cmd, session, timeout=SSH_COMMAND_TIMEOUT_SETUP)

        # Submit training job
        train_cmd = f"""
cd /opt/workspace-internal/kohya_ss
source /venv/main/bin/activate

nohup python sd-scripts/flux_train_network.py \\
    --pretrained_model_name_or_path /workspace/models/flux1-dev/flux1-dev.safetensors \\
    --clip_l /workspace/models/flux1-dev/text_encoder/model.safetensors \\
    --t5xxl /workspace/models/text_encoders/t5xxl_fp16.safetensors \\
    --ae /workspace/models/flux1-dev/ae.safetensors \\
    --dataset_config /workspace/training_data/dataset_{job_id}.toml \\
    --output_dir /workspace/lora_output \\
    --output_name {job_id} \\
    --network_module networks.lora_flux \\
    --network_dim {network_dim} \\
    --network_train_unet_only \\
    --optimizer_type adamw8bit \\
    --learning_rate {learning_rate} \\
    --cache_latents_to_disk \\
    --cache_text_encoder_outputs \\
    --cache_text_encoder_outputs_to_disk \\
    --max_train_epochs {max_train_epochs} \\
    --timestep_sampling shift \\
    --discrete_flow_shift 3.1582 \\
    --model_prediction_type raw \\
    --guidance_scale 1.0 \\
    --gradient_checkpointing \\
    --seed 42 \\
    > /tmp/training_{job_id}.log 2>&1 &

echo $! > /tmp/training_{job_id}.pid
echo "Training started with PID $(cat /tmp/training_{job_id}.pid)"
"""

        logger.info(f"Starting LoRA training for {job_id}")
        result = await self.run_ssh_command(train_cmd, session, timeout=SSH_COMMAND_TIMEOUT_MEDIUM)
        logger.info(f"Training submitted: {result}")

        return job_id

    async def poll_training_status(
        self,
        session: VastAISession,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Poll training job status.

        Args:
            session: Active Vast.ai session
            job_id: Training job ID

        Returns:
            {"status": str, "progress": float, "error": str}
        """
        # Check if process is still running
        check_cmd = f"""
if [ -f /tmp/training_{job_id}.pid ]; then
    PID=$(cat /tmp/training_{job_id}.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "RUNNING"
        # Extract progress from log if available
        tail -5 /tmp/training_{job_id}.log 2>/dev/null | grep -oP '\\d+%' | tail -1 || echo "0%"
    else
        # Process finished - check if output exists
        if [ -f /workspace/lora_output/{job_id}.safetensors ]; then
            echo "COMPLETED"
            ls -la /workspace/lora_output/{job_id}.safetensors
        else
            echo "FAILED"
            tail -20 /tmp/training_{job_id}.log 2>/dev/null
        fi
    fi
else
    echo "NOT_STARTED"
fi
"""

        result = await self.run_ssh_command(check_cmd, session, timeout=SSH_COMMAND_TIMEOUT_DEFAULT)
        lines = result.strip().split("\n")

        if not lines:
            return {"status": "unknown", "progress": 0.0, "error": "No output"}

        status_line = lines[0].strip()

        if status_line == "RUNNING":
            progress = 0.0
            if len(lines) > 1:
                try:
                    progress_str = lines[1].replace("%", "")
                    progress = float(progress_str) / 100.0
                except ValueError:
                    pass
            return {"status": "running", "progress": progress}

        if status_line == "COMPLETED":
            return {"status": "completed", "progress": 1.0}

        if status_line == "FAILED":
            error = "\n".join(lines[1:]) if len(lines) > 1 else "Unknown error"
            return {"status": "failed", "progress": 0.0, "error": error}

        return {"status": "not_started", "progress": 0.0}

    async def download_lora(
        self,
        session: VastAISession,
        job_id: str,
    ) -> bytes:
        """
        Download trained LoRA file from instance.

        Args:
            session: Active Vast.ai session
            job_id: Training job ID

        Returns:
            LoRA safetensors file content

        Raises:
            VastAIManagerError: If download fails
        """
        import tempfile

        if not session.ssh_host or not session.ssh_port:
            raise VastAIManagerError("SSH not available for download")

        remote_path = f"/workspace/lora_output/{job_id}.safetensors"

        # Download via SCP
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
            local_path = f.name

        scp_cmd = (
            f"scp -o StrictHostKeyChecking=no -P {session.ssh_port} "
            f"root@{session.ssh_host}:{remote_path} {local_path}"
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=SSH_COMMAND_TIMEOUT_LONG)

            if proc.returncode != 0:
                raise VastAIManagerError(f"SCP failed: {stderr.decode()}")

            with open(local_path, "rb") as f:
                data = f.read()

            # Cleanup
            os.unlink(local_path)

            logger.info(f"Downloaded LoRA: {len(data)} bytes")
            return data

        except Exception as e:
            # Cleanup on error
            if os.path.exists(local_path):
                os.unlink(local_path)
            raise VastAIManagerError(f"Download failed: {e}") from e

    async def generate_with_lora(
        self,
        session: VastAISession,
        prompt: str,
        lora_path: str,
        num_outputs: int = 1,
        width: int = 512,
        height: int = 512,
        steps: int = 10,
        guidance: float = 3.5,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate image using trained LoRA on Kohya instance.

        Args:
            session: Active Vast.ai session with Kohya
            prompt: Generation prompt
            lora_path: Path to LoRA on instance (e.g., /workspace/lora_output/xxx.safetensors)
            num_outputs: Number of images to generate
            width: Output width
            height: Output height
            steps: Inference steps
            guidance: Guidance scale
            seed: Random seed (random if None)

        Returns:
            {"images": [url1, url2, ...], "success": True}
        """
        import random

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Ensure lora_path is an absolute path on the instance
        if not lora_path.startswith("/"):
            lora_path = f"/workspace/lora_output/{lora_path}"
        if not lora_path.endswith(".safetensors"):
            lora_path = f"{lora_path}.safetensors"

        output_dir = f"/workspace/inference_output_{seed}"

        # Generate image(s)
        gen_cmd = f"""
cd /opt/workspace-internal/kohya_ss
source /venv/main/bin/activate

mkdir -p {output_dir}

python sd-scripts/flux_minimal_inference.py \\
    --ckpt /workspace/models/flux1-dev/flux1-dev.safetensors \\
    --clip_l /workspace/models/flux1-dev/text_encoder/model.safetensors \\
    --t5xxl /workspace/models/text_encoders/t5xxl_fp16.safetensors \\
    --ae /workspace/models/flux1-dev/ae.safetensors \\
    --lora {lora_path} \\
    --prompt '{prompt}' \\
    --output {output_dir}/output.png \\
    --width {width} \\
    --height {height} \\
    --steps {steps} \\
    --guidance {guidance} \\
    --seed {seed} \\
    2>&1

# List generated files
ls -la {output_dir}/*.png 2>/dev/null | head -5
"""

        logger.info(f"Generating with LoRA: {lora_path[:50]}...")
        result = await self.run_ssh_command(gen_cmd, session, timeout=SSH_COMMAND_TIMEOUT_GENERATION)

        # Check if generation succeeded by looking for output files
        if "Saved image to" not in result and ".png" not in result:
            raise VastAIManagerError(f"Generation failed: {result}")

        # Download the generated image(s)
        images = []
        for i in range(num_outputs):
            try:
                # Find the generated file
                list_cmd = f"ls -t {output_dir}/*.png 2>/dev/null | head -1"
                img_path = (await self.run_ssh_command(list_cmd, session, timeout=SSH_COMMAND_TIMEOUT_SHORT)).strip()

                if img_path:
                    # Read and encode image
                    read_cmd = f"base64 {img_path}"
                    img_b64 = await self.run_ssh_command(read_cmd, session, timeout=SSH_COMMAND_TIMEOUT_MEDIUM)

                    # Return as data URL
                    data_url = f"data:image/png;base64,{img_b64.strip()}"
                    images.append(data_url)

            except Exception as e:
                logger.warning(f"Failed to download image {i}: {e}")

        if not images:
            raise VastAIManagerError("No images generated")

        return {"images": images, "success": True}

    # Legacy method name for backwards compatibility
    async def _ssh_command(
        self,
        session: VastAISession,
        command: str,
        timeout: int = 300,
    ) -> str:
        """Execute command via SSH (legacy method)."""
        return await self.run_ssh_command(command, session, timeout)
