"""
Vertex AI Custom Job Manager for LoRA Training.

Uses Vertex AI Custom Jobs for serverless GPU training. Unlike VM-based
approaches (RunPod, GCP Compute), jobs run to completion without TTL issues.

Key Features:
- Serverless: No instance lifecycle management
- Reliable: Jobs run to completion, automatic retries
- Cost-effective: Pay only for compute time
- Same Docker image: gcr.io/YOUR_PROJECT_ID/kestrel-lora:latest

Usage:
    manager = VertexAIManager()
    job = await manager.submit_training_job(
        companion_id="...",
        avatar_data=bytes,
        trigger_word="TOKabc123"
    )

    # Poll for completion
    while True:
        status = await manager.get_job_status(job.job_name)
        if status["state"] == "completed":
            break
        await asyncio.sleep(POLL_INTERVAL_DEFAULT)

    # Download trained LoRA
    lora_bytes = await manager.download_lora(job.job_name)
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_SHORT,
    HTTP_TIMEOUT_DEFAULT,
    HTTP_TIMEOUT_MEDIUM,
    HTTP_TIMEOUT_DOWNLOAD,
    POLL_INTERVAL_DEFAULT,
)
from kestrel_sovereign.kestrel_config.defaults import get_lighthouse_gateway_url

logger = logging.getLogger(__name__)


class JobState(Enum):
    """Vertex AI job states."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_vertex_state(cls, state: str) -> "JobState":
        """Convert Vertex AI job state string to enum."""
        mapping = {
            "JOB_STATE_PENDING": cls.PENDING,
            "JOB_STATE_QUEUED": cls.QUEUED,
            "JOB_STATE_RUNNING": cls.RUNNING,
            "JOB_STATE_SUCCEEDED": cls.COMPLETED,
            "JOB_STATE_FAILED": cls.FAILED,
            "JOB_STATE_CANCELLED": cls.CANCELLED,
            "JOB_STATE_CANCELLING": cls.CANCELLED,
        }
        return mapping.get(state, cls.PENDING)


@dataclass
class VertexAITrainingJob:
    """Tracks a Vertex AI training job."""
    job_name: str  # Full resource name: projects/.../locations/.../customJobs/...
    job_id: str  # Short ID
    companion_id: str
    trigger_word: str
    state: JobState
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    gcs_output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_name": self.job_name,
            "job_id": self.job_id,
            "companion_id": self.companion_id,
            "trigger_word": self.trigger_word,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "gcs_output_path": self.gcs_output_path,
        }


class VertexAIManagerError(Exception):
    """Custom exception for Vertex AI operations."""


class VertexAIManager:
    """
    Manages Vertex AI Custom Jobs for LoRA training.

    Uses the same Docker image as RunPod/GCP Compute:
    gcr.io/YOUR_PROJECT_ID/kestrel-lora:latest

    The image runs simpletuner_api.py which accepts training requests.
    For Vertex AI, we pass training config via environment variables
    and the avatar image via GCS.
    """

    # Default configuration
    DEFAULT_PROJECT = "YOUR_PROJECT_ID"
    DEFAULT_REGION = "us-central1"
    DEFAULT_MACHINE_TYPE = "a2-ultragpu-1g"  # A100 80GB
    DEFAULT_ACCELERATOR = "NVIDIA_A100_80GB"
    DEFAULT_GCS_BUCKET = "kestrel-training"

    # Container images for different FLUX versions
    # FLUX.2-dev: Standard container with content filtering (SFW + artistic nudity)
    # FLUX.1-dev: Uncensored container with multi-LoRA support for explicit content
    CONTAINER_IMAGES = {
        "flux2": "gcr.io/YOUR_PROJECT_ID/kestrel-lora:v8",
        "flux1": "gcr.io/YOUR_PROJECT_ID/kestrel-lora-flux1:v1",
    }
    DEFAULT_IMAGE = CONTAINER_IMAGES["flux2"]  # Default to FLUX.2-dev

    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        service_account: Optional[str] = None,
        flux_version: str = "flux2",
    ):
        """
        Initialize Vertex AI manager.

        Args:
            project_id: GCP project ID (default: YOUR_PROJECT_ID)
            region: GCP region (default: us-central1)
            service_account: Service account email for jobs
            flux_version: FLUX version to use ("flux1" or "flux2", default: "flux2")
                         flux1 = FLUX.1-dev with optional auxiliary LoRA support
                         flux2 = FLUX.2-dev with standard content filtering
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID", self.DEFAULT_PROJECT)
        self.region = region or os.getenv("GCP_REGION", self.DEFAULT_REGION)
        self.service_account = service_account or os.getenv(
            "GCP_SERVICE_ACCOUNT",
            f"kestrel-agent-admin@{self.project_id}.iam.gserviceaccount.com"
        )

        # Set container image based on FLUX version
        self.flux_version = flux_version
        self.container_image = self.CONTAINER_IMAGES.get(flux_version, self.DEFAULT_IMAGE)

        self.gcs_bucket = os.getenv("GCS_TRAINING_BUCKET", self.DEFAULT_GCS_BUCKET)
        self.hf_token = os.getenv("HF_TOKEN", "")

        # Vertex AI API endpoint
        self.api_endpoint = f"https://{self.region}-aiplatform.googleapis.com/v1"
        self.parent = f"projects/{self.project_id}/locations/{self.region}"

        # Track active jobs
        self._jobs: Dict[str, VertexAITrainingJob] = {}

        # Auth token (lazy loaded)
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        logger.info(f"VertexAIManager initialized: project={self.project_id}, region={self.region}, flux={self.flux_version}, image={self.container_image}")

    def _build_env_vars(self) -> list:
        """Build environment variables for container, excluding empty values."""
        env_vars = [
            {"name": "HF_HOME", "value": "/tmp/huggingface"},
            {"name": "VERTEX_AI_MODE", "value": "true"},
            # GCS bucket for quantized model cache (reduces cold start from ~10min to ~2-3min)
            {"name": "GCS_TRAINING_BUCKET", "value": self.gcs_bucket},
        ]
        # Only add HF_TOKEN if it's set
        if self.hf_token:
            env_vars.append({"name": "HF_TOKEN", "value": self.hf_token})
        return env_vars

    async def _get_access_token(self) -> str:
        """Get GCP access token for API calls."""
        # Check if we have a valid cached token
        if self._access_token and self._token_expiry:
            if datetime.now(timezone.utc) < self._token_expiry:
                return self._access_token

        # Get token from metadata server (when on GCP), service account file, or gcloud
        from datetime import timedelta

        # Try metadata server first (fastest when on GCP)
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SHORT) as client:
                response = await client.get(
                    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                    headers={"Metadata-Flavor": "Google"}
                )
                if response.status_code == 200:
                    data = response.json()
                    self._access_token = data["access_token"]
                    # Token typically valid for 1 hour, refresh after 50 min
                    self._token_expiry = datetime.now(timezone.utc) + timedelta(minutes=50)
                    return self._access_token
        except Exception:
            pass  # Not on GCP, try service account file

        # Try service account file via google-auth library
        try:
            import google.auth
            from google.oauth2 import service_account
            import os

            creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_file and os.path.exists(creds_file):
                credentials = service_account.Credentials.from_service_account_file(
                    creds_file,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                # Refresh to get access token
                import google.auth.transport.requests
                request = google.auth.transport.requests.Request()
                credentials.refresh(request)
                self._access_token = credentials.token
                self._token_expiry = datetime.now(timezone.utc) + timedelta(minutes=50)
                return self._access_token
        except Exception as e:
            logger.debug(f"Service account auth failed: {e}")

        # Fall back to gcloud command
        import subprocess
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=HTTP_TIMEOUT_SHORT
            )
            if result.returncode == 0:
                self._access_token = result.stdout.strip()
                self._token_expiry = datetime.now(timezone.utc) + timedelta(minutes=50)
                return self._access_token
        except Exception as e:
            logger.error(f"Failed to get access token via gcloud: {e}")

        raise VertexAIManagerError(
            "Failed to get GCP access token. Ensure you're authenticated with gcloud "
            "or running on GCP with appropriate service account."
        )

    async def _upload_to_gcs(self, data: bytes, blob_name: str) -> str:
        """
        Upload data to GCS bucket.

        Returns:
            GCS URI: gs://bucket/blob_name
        """
        token = await self._get_access_token()

        # Use JSON API for upload
        upload_url = f"https://storage.googleapis.com/upload/storage/v1/b/{self.gcs_bucket}/o"

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_MEDIUM) as client:
            response = await client.post(
                upload_url,
                params={"uploadType": "media", "name": blob_name},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/octet-stream",
                },
                content=data
            )

            if response.status_code not in (200, 201):
                raise VertexAIManagerError(
                    f"Failed to upload to GCS: {response.status_code} {response.text}"
                )

        gcs_uri = f"gs://{self.gcs_bucket}/{blob_name}"
        logger.info(f"Uploaded {len(data)} bytes to {gcs_uri}")
        return gcs_uri

    async def _download_from_gcs(self, gcs_uri: str) -> bytes:
        """Download data from GCS URI."""
        token = await self._get_access_token()

        # Parse gs://bucket/path
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")

        parts = gcs_uri[5:].split("/", 1)
        bucket = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""

        # URL encode the blob name
        import urllib.parse
        encoded_blob = urllib.parse.quote(blob_name, safe="")

        download_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{encoded_blob}"

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DOWNLOAD) as client:
            response = await client.get(
                download_url,
                params={"alt": "media"},
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code != 200:
                raise VertexAIManagerError(
                    f"Failed to download from GCS: {response.status_code} {response.text}"
                )

            return response.content

    async def submit_training_job(
        self,
        companion_id: str,
        avatar_data: bytes,
        trigger_word: Optional[str] = None,
        steps: int = 1000,
        lora_rank: int = 16,
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
    ) -> VertexAITrainingJob:
        """
        Submit a LoRA training job to Vertex AI.

        Args:
            companion_id: Companion UUID
            avatar_data: Avatar image bytes (JPEG/PNG)
            trigger_word: Trigger word for LoRA (default: TOK{companion_id[:8]})
            steps: Training steps (default: 1000)
            lora_rank: LoRA rank (default: 16)
            machine_type: GCP machine type (default: a2-ultragpu-1g)
            accelerator_type: GPU type (default: NVIDIA_A100_80GB)

        Returns:
            VertexAITrainingJob with job details
        """
        token = await self._get_access_token()

        # Generate trigger word if not provided
        if not trigger_word:
            trigger_word = f"TOK{companion_id[:8]}"

        # Upload avatar to GCS
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        avatar_blob = f"training/{companion_id}/{timestamp}/avatar.jpg"
        avatar_gcs_uri = await self._upload_to_gcs(avatar_data, avatar_blob)

        # Output path for trained LoRA
        output_blob = f"training/{companion_id}/{timestamp}/output"
        output_gcs_uri = f"gs://{self.gcs_bucket}/{output_blob}"

        # Build job spec
        job_display_name = f"lora-{companion_id[:8]}-{timestamp}"

        job_spec = {
            "displayName": job_display_name,
            "jobSpec": {
                "workerPoolSpecs": [
                    {
                        "machineSpec": {
                            "machineType": machine_type or self.DEFAULT_MACHINE_TYPE,
                            "acceleratorType": accelerator_type or self.DEFAULT_ACCELERATOR,
                            "acceleratorCount": 1,
                        },
                        "replicaCount": 1,
                        "containerSpec": {
                            "imageUri": self.container_image,
                            "command": ["python3", "/app/simpletuner_api.py"],
                            "args": [
                                "--vertex-mode",  # Signal to run as batch job
                                f"--avatar-gcs={avatar_gcs_uri}",
                                f"--output-gcs={output_gcs_uri}",
                                f"--companion-id={companion_id}",
                                f"--trigger-word={trigger_word}",
                                f"--steps={steps}",
                                f"--lora-rank={lora_rank}",
                            ],
                            "env": self._build_env_vars(),
                        },
                    }
                ],
                "serviceAccount": self.service_account,
            },
        }

        # Submit job
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_MEDIUM) as client:
            response = await client.post(
                f"{self.api_endpoint}/{self.parent}/customJobs",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=job_spec
            )

            if response.status_code not in (200, 201):
                raise VertexAIManagerError(
                    f"Failed to submit job: {response.status_code} {response.text}"
                )

            result = response.json()

        # Extract job info
        job_name = result["name"]
        job_id = job_name.split("/")[-1]

        job = VertexAITrainingJob(
            job_name=job_name,
            job_id=job_id,
            companion_id=companion_id,
            trigger_word=trigger_word,
            state=JobState.PENDING,
            created_at=datetime.now(timezone.utc),
            gcs_output_path=output_gcs_uri,
        )

        self._jobs[job_id] = job

        logger.info(f"Submitted Vertex AI job: {job_id} for companion {companion_id}")
        return job

    async def submit_generation_job(
        self,
        prompt: str,
        trigger_word: str,
        output_gcs_prefix: str,
        lora_gcs_path: Optional[str] = None,
        lora_ipfs_cid: Optional[str] = None,
        ipfs_gateway: Optional[str] = None,
        num_outputs: int = 1,
        width: int = 1024,
        height: int = 1024,
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        image_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit an image generation job to Vertex AI.

        Uses the same Docker image as training, but with --generate-mode flag.
        The container will:
        1. Download LoRA from IPFS or GCS
        2. Load FLUX model with int8-quanto quantization
        3. Generate images (with optional auxiliary LoRA stacking if using FLUX.1)
        4. Upload results to GCS
        5. Exit

        Args:
            prompt: Generation prompt
            trigger_word: LoRA trigger word
            output_gcs_prefix: GCS prefix for output (gs://bucket/generation/{id}/{timestamp})
            lora_gcs_path: GCS path to LoRA (gs://bucket/.../pytorch_lora_weights.safetensors)
            lora_ipfs_cid: IPFS CID for LoRA (e.g., QmXxx...)
            ipfs_gateway: IPFS gateway URL (default: https://ipfs.io/ipfs)
            num_outputs: Number of images to generate (default: 1)
            width: Image width (default: 1024)
            height: Image height (default: 1024)
            machine_type: GCP machine type (default: a2-ultragpu-1g)
            accelerator_type: GPU type (default: NVIDIA_A100_80GB)
            image_uri: Docker image URI (default: uses self.container_image based on flux_version)

        Returns:
            Dict with job_id, job_name, output_gcs_path
        """
        if not lora_gcs_path and not lora_ipfs_cid:
            raise VertexAIManagerError("Either lora_gcs_path or lora_ipfs_cid is required")
        if lora_gcs_path and lora_ipfs_cid:
            raise VertexAIManagerError("Specify either lora_gcs_path or lora_ipfs_cid, not both")

        token = await self._get_access_token()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_display_name = f"gen-{timestamp}"

        # Build container args based on LoRA source
        container_args = [
            "--generate-mode",
            f"--output-gcs={output_gcs_prefix}",
            f"--prompt={prompt}",
            f"--trigger-word={trigger_word}",
            f"--num-outputs={num_outputs}",
            f"--width={width}",
            f"--height={height}",
        ]

        if lora_ipfs_cid:
            container_args.append(f"--lora-ipfs={lora_ipfs_cid}")
            container_args.append(f"--ipfs-gateway={ipfs_gateway}")
        else:
            container_args.append(f"--lora-gcs={lora_gcs_path}")

        # Use configured container image or provided override
        final_image_uri = image_uri or self.container_image

        job_spec = {
            "displayName": job_display_name,
            "jobSpec": {
                "workerPoolSpecs": [
                    {
                        "machineSpec": {
                            "machineType": machine_type or self.DEFAULT_MACHINE_TYPE,
                            "acceleratorType": accelerator_type or self.DEFAULT_ACCELERATOR,
                            "acceleratorCount": 1,
                        },
                        "replicaCount": 1,
                        "containerSpec": {
                            "imageUri": final_image_uri,
                            "command": ["python3", "/app/simpletuner_api.py"],
                            "args": container_args,
                            "env": self._build_env_vars(),
                        },
                    }
                ],
                "serviceAccount": self.service_account,
            },
        }

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_MEDIUM) as client:
            response = await client.post(
                f"{self.api_endpoint}/{self.parent}/customJobs",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=job_spec
            )

            if response.status_code not in (200, 201):
                raise VertexAIManagerError(
                    f"Failed to submit generation job: {response.status_code} {response.text}"
                )

            result = response.json()

        job_name = result["name"]
        job_id = job_name.split("/")[-1]

        logger.info(f"Submitted generation job: {job_id}, output: {output_gcs_prefix}")

        return {
            "job_id": job_id,
            "job_name": job_name,
            "output_gcs_path": output_gcs_prefix,
        }

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get current status of a training job.

        Args:
            job_id: Job ID or full job name

        Returns:
            Dict with state, progress, error info
        """
        token = await self._get_access_token()

        # Handle both short ID and full name
        if job_id.startswith("projects/"):
            job_name = job_id
        else:
            job_name = f"{self.parent}/customJobs/{job_id}"

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            response = await client.get(
                f"{self.api_endpoint}/{job_name}",
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code != 200:
                raise VertexAIManagerError(
                    f"Failed to get job status: {response.status_code} {response.text}"
                )

            result = response.json()

        state = JobState.from_vertex_state(result.get("state", "JOB_STATE_PENDING"))

        # Update cached job if we have it
        short_id = job_name.split("/")[-1]
        if short_id in self._jobs:
            job = self._jobs[short_id]
            job.state = state
            if state == JobState.RUNNING and not job.started_at:
                job.started_at = datetime.now(timezone.utc)
            if state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                job.completed_at = datetime.now(timezone.utc)
            if result.get("error"):
                job.error_message = result["error"].get("message", str(result["error"]))

        # Calculate progress estimate
        progress = 0.0
        if state == JobState.PENDING:
            progress = 0.05
        elif state == JobState.QUEUED:
            progress = 0.1
        elif state == JobState.RUNNING:
            # Estimate based on elapsed time (assuming ~60 min total)
            start_time = result.get("startTime")
            if start_time:
                # Use module-level datetime import (no local import needed)
                start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                # Estimate 60 minutes for training
                progress = min(0.1 + (elapsed / 3600) * 0.8, 0.9)
            else:
                progress = 0.15
        elif state == JobState.COMPLETED:
            progress = 1.0

        return {
            "job_id": short_id,
            "job_name": job_name,
            "state": state.value,
            "progress": progress,
            "error": result.get("error", {}).get("message"),
            "create_time": result.get("createTime"),
            "start_time": result.get("startTime"),
            "end_time": result.get("endTime"),
            "gcs_output_path": self._jobs.get(short_id, {}).gcs_output_path if short_id in self._jobs else None,
        }

    async def download_lora(self, job_id: str) -> Optional[bytes]:
        """
        Download trained LoRA weights from completed job.

        Args:
            job_id: Job ID

        Returns:
            LoRA weights as bytes, or None if not found
        """
        # Get job info
        short_id = job_id.split("/")[-1] if "/" in job_id else job_id
        job = self._jobs.get(short_id)

        if not job or not job.gcs_output_path:
            # Try to get from status
            status = await self.get_job_status(job_id)
            if status.get("state") != "completed":
                logger.warning(f"Job {job_id} not completed, cannot download LoRA")
                return None
            output_path = status.get("gcs_output_path")
            if not output_path:
                logger.error(f"No output path for job {job_id}")
                return None
        else:
            output_path = job.gcs_output_path

        # Look for .safetensors file in output
        lora_uri = f"{output_path}/lora.safetensors"

        try:
            lora_bytes = await self._download_from_gcs(lora_uri)
            logger.info(f"Downloaded LoRA: {len(lora_bytes)} bytes from {lora_uri}")
            return lora_bytes
        except Exception as e:
            logger.error(f"Failed to download LoRA from {lora_uri}: {e}")

            # Try alternative paths (including nested companion_id subdirectory)
            alt_paths = [
                "pytorch_lora_weights.safetensors",
                "adapter_model.safetensors",
            ]

            # Also try nested paths (SimpleTuner creates {companion_id}/ subdirectory)
            if job:
                companion_id = job.companion_id
                alt_paths.extend([
                    f"{companion_id}/pytorch_lora_weights.safetensors",
                    f"{companion_id}/adapter_model.safetensors",
                    f"{companion_id}/lora.safetensors",
                ])

            for alt_name in alt_paths:
                try:
                    alt_uri = f"{output_path}/{alt_name}"
                    lora_bytes = await self._download_from_gcs(alt_uri)
                    logger.info(f"Downloaded LoRA from alternate path: {alt_uri}")
                    return lora_bytes
                except Exception:
                    continue

            return None

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled successfully
        """
        token = await self._get_access_token()

        if job_id.startswith("projects/"):
            job_name = job_id
        else:
            job_name = f"{self.parent}/customJobs/{job_id}"

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            response = await client.post(
                f"{self.api_endpoint}/{job_name}:cancel",
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code not in (200, 201):
                logger.error(f"Failed to cancel job: {response.status_code} {response.text}")
                return False

        # Update cached job
        short_id = job_name.split("/")[-1]
        if short_id in self._jobs:
            self._jobs[short_id].state = JobState.CANCELLED
            self._jobs[short_id].completed_at = datetime.now(timezone.utc)

        logger.info(f"Cancelled job: {job_id}")
        return True

    async def list_jobs(
        self,
        companion_id: Optional[str] = None,
        state_filter: Optional[JobState] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List training jobs.

        Args:
            companion_id: Filter by companion ID
            state_filter: Filter by job state
            limit: Max jobs to return

        Returns:
            List of job info dicts
        """
        token = await self._get_access_token()

        # Build filter
        filters = []
        if state_filter:
            vertex_state = f"JOB_STATE_{state_filter.value.upper()}"
            if state_filter == JobState.COMPLETED:
                vertex_state = "JOB_STATE_SUCCEEDED"
            filters.append(f'state="{vertex_state}"')

        params = {"pageSize": limit}
        if filters:
            params["filter"] = " AND ".join(filters)

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            response = await client.get(
                f"{self.api_endpoint}/{self.parent}/customJobs",
                params=params,
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code != 200:
                raise VertexAIManagerError(
                    f"Failed to list jobs: {response.status_code} {response.text}"
                )

            result = response.json()

        jobs = []
        for job_data in result.get("customJobs", []):
            job_name = job_data["name"]
            job_id = job_name.split("/")[-1]

            # Filter by companion_id if specified
            if companion_id:
                display_name = job_data.get("displayName", "")
                if companion_id[:8] not in display_name:
                    continue

            jobs.append({
                "job_id": job_id,
                "job_name": job_name,
                "display_name": job_data.get("displayName"),
                "state": JobState.from_vertex_state(job_data.get("state", "")).value,
                "create_time": job_data.get("createTime"),
                "start_time": job_data.get("startTime"),
                "end_time": job_data.get("endTime"),
                "error": job_data.get("error", {}).get("message"),
            })

        return jobs
