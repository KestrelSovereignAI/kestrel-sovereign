"""
GCP Compute SSH-based Training Methods.

Contains SSH command execution and LoRA training methods
for GCP Compute Engine instances.
"""

import asyncio
import logging
import os
import shlex
import tempfile
from typing import Any, Dict, Optional

from kestrel_sdk.config.constants import (
    SSH_COMMAND_TIMEOUT_DEFAULT,
    SSH_COMMAND_TIMEOUT_MEDIUM,
    SSH_COMMAND_TIMEOUT_SETUP,
    SSH_COMMAND_TIMEOUT_LONG,
    SSH_COMMAND_TIMEOUT_GENERATION,
)
from .models import GCPComputeManagerError, GCPComputeSession

logger = logging.getLogger(__name__)


class GCPSSHTrainingMixin:
    """
    Mixin for SSH-based training operations on GCP Compute instances.

    Requires GCPComputeEngineManagerCore as base class.
    """

    async def run_ssh_command(
        self,
        command: str,
        session: Optional[GCPComputeSession] = None,
        timeout: int = 60,
    ) -> str:
        """
        Run a command on the instance via SSH using gcloud compute ssh.

        Uses gcloud compute ssh with explicit --project flag to ensure
        we always use the correct project and service account credentials.

        Args:
            command: Shell command to run
            session: Session to use (defaults to current)
            timeout: Command timeout in seconds

        Returns:
            Command output
        """
        session = session or self._session
        if not session:
            raise GCPComputeManagerError("No active session")

        # Build gcloud compute ssh command with explicit project
        ssh_cmd = [
            "gcloud",
            "compute",
            "ssh",
            f"--project={self.project_id}",
            f"--zone={session.zone}",
            session.instance_name,
            f"--command={command}",
            "--quiet",
        ]

        # Add service account credentials if available
        if self._credentials_file:
            env = os.environ.copy()
            env["GOOGLE_APPLICATION_CREDENTIALS"] = self._credentials_file
        else:
            env = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            if proc.returncode != 0:
                logger.warning(f"SSH command failed: {stderr.decode()}")

            return stdout.decode()

        except asyncio.TimeoutError:
            raise GCPComputeManagerError(f"SSH command timed out after {timeout}s")
        except Exception as e:
            raise GCPComputeManagerError(f"SSH command failed: {e}") from e

    async def submit_training_job(
        self,
        session: GCPComputeSession,
        image_url: str,
        companion_id: str,
        trigger_word: str = "sks person",
        num_repeats: int = 10,
        max_train_epochs: int = 1,
        network_dim: int = 4,
        learning_rate: float = 1e-4,
    ) -> str:
        """
        Submit a LoRA training job.

        This implementation mirrors the VastAI manager's approach, using SSH
        to submit training jobs to the container.

        Args:
            session: Active GCP session
            image_url: URL of training image(s)
            companion_id: Companion ID for output naming
            trigger_word: Trigger word for LoRA
            num_repeats: Number of repeats per image
            max_train_epochs: Training epochs
            network_dim: LoRA rank (4 = small/fast, 16 = more detail)
            learning_rate: Learning rate

        Returns:
            Job ID (companion_id)
        """
        from kestrel_sdk.config.constants import DEFAULT_TRAINING_BATCH_SIZE

        job_id = companion_id
        mount_path = self.disk_config.get("mount_path", "/workspace")

        # Create training data directory structure
        mount_path_quoted = shlex.quote(mount_path)
        setup_cmd = f"""
mkdir -p {mount_path_quoted}/training_data/{shlex.quote(job_id)}
mkdir -p {mount_path_quoted}/lora_output

# Download training image
cd {mount_path_quoted}/training_data/{shlex.quote(job_id)}
wget -q -O image_001.png {shlex.quote(image_url)}

# Create caption file with trigger word
echo {shlex.quote(trigger_word)} > image_001.txt

# Create dataset config
cat > {mount_path_quoted}/training_data/dataset_{shlex.quote(job_id)}.toml << 'DATASET_EOF'
[general]
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = 1024
batch_size = {DEFAULT_TRAINING_BATCH_SIZE}

[[datasets.subsets]]
image_dir = '{mount_path}/training_data/{shlex.quote(job_id)}'
caption_extension = '.txt'
num_repeats = {num_repeats}
DATASET_EOF
"""

        logger.info(f"Setting up training data for {job_id}")
        await self.run_ssh_command(setup_cmd, session, timeout=SSH_COMMAND_TIMEOUT_SETUP)

        # Submit training job via docker exec
        mount_path_quoted = shlex.quote(mount_path)
        train_cmd = f"""
docker exec kestrel-workload bash -c '
cd /workspace
nohup python train_lora.py \\
    --pretrained_model_name_or_path {mount_path_quoted}/models/flux1-dev/flux1-dev.safetensors \\
    --clip_l {mount_path_quoted}/models/flux1-dev/text_encoder/model.safetensors \\
    --t5xxl {mount_path_quoted}/models/text_encoders/t5xxl_fp16.safetensors \\
    --ae {mount_path_quoted}/models/flux1-dev/ae.safetensors \\
    --dataset_config {mount_path_quoted}/training_data/dataset_{shlex.quote(job_id)}.toml \\
    --output_dir {mount_path_quoted}/lora_output \\
    --output_name {shlex.quote(job_id)} \\
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
    > /tmp/training_{shlex.quote(job_id)}.log 2>&1 &

echo $! > /tmp/training_{shlex.quote(job_id)}.pid
echo "Training started with PID $(cat /tmp/training_{shlex.quote(job_id)}.pid)"
'
"""

        logger.info(f"Starting LoRA training for {job_id}")
        result = await self.run_ssh_command(train_cmd, session, timeout=SSH_COMMAND_TIMEOUT_MEDIUM)
        logger.info(f"Training submitted: {result}")

        return job_id

    async def poll_training_status(
        self,
        session: GCPComputeSession,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Poll training job status.

        Args:
            session: Active GCP session
            job_id: Training job ID

        Returns:
            {"status": str, "progress": float, "error": str}
        """
        mount_path = self.disk_config.get("mount_path", "/workspace")
        mount_path_quoted = shlex.quote(mount_path)

        check_cmd = f"""
docker exec kestrel-workload bash -c '
if [ -f /tmp/training_{shlex.quote(job_id)}.pid ]; then
    PID=$(cat /tmp/training_{shlex.quote(job_id)}.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "RUNNING"
        tail -5 /tmp/training_{shlex.quote(job_id)}.log 2>/dev/null | grep -oP "\\d+%" | tail -1 || echo "0%"
    else
        if [ -f {mount_path_quoted}/lora_output/{shlex.quote(job_id)}.safetensors ]; then
            echo "COMPLETED"
            ls -la {mount_path_quoted}/lora_output/{shlex.quote(job_id)}.safetensors
        else
            echo "FAILED"
            tail -20 /tmp/training_{shlex.quote(job_id)}.log 2>/dev/null
        fi
    fi
else
    echo "NOT_STARTED"
fi
'
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
        session: GCPComputeSession,
        job_id: str,
    ) -> bytes:
        """
        Download trained LoRA file from instance.

        Args:
            session: Active GCP session
            job_id: Training job ID

        Returns:
            LoRA safetensors file content

        Raises:
            GCPComputeManagerError: If download fails
        """
        if not session.ssh_host:
            raise GCPComputeManagerError("SSH not available for download")

        mount_path = self.disk_config.get("mount_path", "/workspace")
        remote_path = f"{mount_path}/lora_output/{shlex.quote(job_id)}.safetensors"

        # Download via SCP
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
            local_path = f.name

        scp_cmd = (
            f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"-i {shlex.quote(self.ssh_key_file)} "
            f"{shlex.quote(self.ssh_user)}@{shlex.quote(session.ssh_host)}:{remote_path} {shlex.quote(local_path)}"
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=SSH_COMMAND_TIMEOUT_LONG)

            if proc.returncode != 0:
                raise GCPComputeManagerError(f"SCP failed: {stderr.decode()}")

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
            raise GCPComputeManagerError(f"Download failed: {e}") from e

    async def generate_with_lora(
        self,
        session: GCPComputeSession,
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
        Generate image using trained LoRA.

        Args:
            session: Active GCP session
            prompt: Generation prompt
            lora_path: Path to LoRA on instance
            num_outputs: Number of images to generate
            width: Output width
            height: Output height
            steps: Inference steps
            guidance: Guidance scale
            seed: Random seed

        Returns:
            {"images": [base64_data, ...], "success": True}
        """
        import base64
        import random

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        mount_path = self.disk_config.get("mount_path", "/workspace")
        mount_path_quoted = shlex.quote(mount_path)

        # Ensure lora_path is absolute
        if not lora_path.startswith("/"):
            lora_path = f"{mount_path}/lora_output/{lora_path}"
        if not lora_path.endswith(".safetensors"):
            lora_path = f"{lora_path}.safetensors"

        output_dir = f"/tmp/inference_output_{seed}"

        gen_cmd = f"""
docker exec kestrel-workload bash -c '
mkdir -p {shlex.quote(output_dir)}

python generate_image.py \\
    --ckpt {mount_path_quoted}/models/flux1-dev/flux1-dev.safetensors \\
    --clip_l {mount_path_quoted}/models/flux1-dev/text_encoder/model.safetensors \\
    --t5xxl {mount_path_quoted}/models/text_encoders/t5xxl_fp16.safetensors \\
    --ae {mount_path_quoted}/models/flux1-dev/ae.safetensors \\
    --lora {shlex.quote(lora_path)} \\
    --prompt {shlex.quote(prompt)} \\
    --output {shlex.quote(output_dir)}/output.png \\
    --width {width} \\
    --height {height} \\
    --steps {steps} \\
    --guidance {guidance} \\
    --seed {seed} \\
    2>&1

# Encode output to base64
if [ -f {shlex.quote(output_dir)}/output.png ]; then
    echo "BASE64_START"
    base64 {shlex.quote(output_dir)}/output.png
    echo "BASE64_END"
fi
'
"""

        logger.info(f"Generating with LoRA: {lora_path[:50]}...")
        result = await self.run_ssh_command(gen_cmd, session, timeout=SSH_COMMAND_TIMEOUT_GENERATION)

        # Extract base64 image data
        if "BASE64_START" in result and "BASE64_END" in result:
            start = result.index("BASE64_START") + len("BASE64_START")
            end = result.index("BASE64_END")
            b64_data = result[start:end].strip()

            return {
                "success": True,
                "images": [f"data:image/png;base64,{b64_data}"],
            }

        return {
            "success": False,
            "error": "Failed to generate image",
            "output": result[:500],
        }
