"""
Vast.ai HTTP API Methods for SimpleTuner Docker Image.

Provides training and inference via HTTP endpoints exposed by
the SimpleTuner container running on Vast.ai instances.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_DEFAULT,
    HTTP_TIMEOUT_MEDIUM,
    HTTP_TIMEOUT_DOWNLOAD,
)
from .models import VastAIManagerError, VastAISession

logger = logging.getLogger(__name__)


class VastAIHTTPAPIMixin:
    """HTTP API methods for SimpleTuner Docker container on Vast.ai."""

    async def wait_for_api_ready(
        self,
        session: VastAISession,
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> bool:
        """
        Wait for the SimpleTuner API to be ready to accept jobs.

        Args:
            session: Active Vast.ai session
            timeout: Maximum wait time in seconds
            poll_interval: Time between checks

        Returns:
            True if ready, False if timeout
        """
        import httpx

        if not session.backend_base_url:
            logger.error("No backend URL available for session")
            return False

        base_url = session.backend_base_url.rstrip("/")
        deadline = time.monotonic() + timeout

        logger.info(f"Waiting for SimpleTuner API at {base_url}/ready...")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            while time.monotonic() < deadline:
                try:
                    response = await client.get(f"{base_url}/ready")
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("can_accept_job", False):
                            logger.info(f"SimpleTuner API ready at {base_url}")
                            return True
                except Exception as e:
                    logger.debug(f"API check failed: {e}")

                await asyncio.sleep(poll_interval)

        logger.error(f"SimpleTuner API did not become ready within {timeout}s")
        return False

    async def submit_training_job_http(
        self,
        session: VastAISession,
        avatar_data: bytes,
        companion_id: str,
        trigger_word: Optional[str] = None,
        steps: int = 500,
        lora_rank: int = 16,
        callback_url: Optional[str] = None,
        wait_for_ready: bool = True,
    ) -> str:
        """
        Submit a LoRA training job via HTTP API.

        Args:
            session: Active Vast.ai session with SimpleTuner container
            avatar_data: Training image bytes (JPEG/PNG)
            companion_id: Companion UUID
            trigger_word: Custom trigger word (default: TOK{companion_id[:8]})
            steps: Training steps
            lora_rank: LoRA rank
            callback_url: Optional webhook for completion
            wait_for_ready: Wait for API to be ready before submitting

        Returns:
            Job ID from the training API
        """
        import httpx

        if not session.backend_base_url:
            raise VastAIManagerError("No backend URL available for session")

        base_url = session.backend_base_url.rstrip("/")

        # Wait for API to be ready
        if wait_for_ready:
            ready = await self.wait_for_api_ready(session)
            if not ready:
                raise VastAIManagerError("SimpleTuner API not ready")

        # Generate trigger word if not provided
        if not trigger_word:
            clean_id = companion_id.replace("-", "")[:8]
            trigger_word = f"TOK{clean_id}"

        logger.info(f"Submitting training job for {companion_id} to {base_url}/train")

        # Prepare multipart form data
        files = {
            "image": ("avatar.png", avatar_data, "image/png"),
        }
        data = {
            "companion_id": companion_id,
            "trigger_word": trigger_word,
            "steps": str(steps),
            "lora_rank": str(lora_rank),
        }
        if callback_url:
            data["callback_url"] = callback_url

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_MEDIUM) as client:
            response = await client.post(
                f"{base_url}/train",
                files=files,
                data=data,
            )

            if response.status_code != 200:
                raise VastAIManagerError(
                    f"Training submission failed: {response.status_code} - {response.text}"
                )

            result = response.json()
            job_id = result.get("job_id")
            if not job_id:
                raise VastAIManagerError(f"No job_id in response: {result}")

            logger.info(f"Training job submitted: {job_id}")
            return job_id

    async def poll_training_status_http(
        self,
        session: VastAISession,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Poll training job status via HTTP API.

        Args:
            session: Active Vast.ai session
            job_id: Training job ID

        Returns:
            {"status": str, "progress": float, "error": str, ...}
        """
        import httpx

        if not session.backend_base_url:
            raise VastAIManagerError("No backend URL available")

        base_url = session.backend_base_url.rstrip("/")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            response = await client.get(f"{base_url}/status/{job_id}")

            if response.status_code == 404:
                return {"status": "not_found", "progress": 0.0, "error": "Job not found"}

            if response.status_code != 200:
                return {"status": "error", "progress": 0.0, "error": response.text}

            return response.json()

    async def download_lora_http(
        self,
        session: VastAISession,
        job_id: str,
    ) -> bytes:
        """
        Download trained LoRA weights via HTTP API.

        Args:
            session: Active Vast.ai session
            job_id: Training job ID

        Returns:
            LoRA safetensors file content

        Raises:
            VastAIManagerError: If download fails
        """
        import httpx

        if not session.backend_base_url:
            raise VastAIManagerError("No backend URL available")

        base_url = session.backend_base_url.rstrip("/")

        logger.info(f"Downloading LoRA for job {job_id} from {base_url}/download/{job_id}")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DOWNLOAD) as client:
            response = await client.get(f"{base_url}/download/{job_id}")

            if response.status_code == 400:
                raise VastAIManagerError(f"Training not complete: {response.text}")

            if response.status_code == 404:
                raise VastAIManagerError(f"LoRA not found for job {job_id}")

            if response.status_code != 200:
                raise VastAIManagerError(
                    f"Download failed: {response.status_code} - {response.text}"
                )

            lora_data = response.content
            logger.info(f"Downloaded LoRA: {len(lora_data)} bytes")
            return lora_data

    async def generate_image_http(
        self,
        session: VastAISession,
        prompt: str,
        lora_path: str,
        trigger_word: str = "TOK",
        num_outputs: int = 1,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.0,
        timeout: int = 900,
    ) -> Dict[str, Any]:
        """
        Generate images using FLUX.2-dev with a trained LoRA via HTTP API.

        Uses async generation to avoid timeouts:
        1. POST /generate/async to start generation
        2. Poll /generate/status/{job_id} until completed

        Args:
            session: Active Vast.ai session
            prompt: Generation prompt
            lora_path: Path to LoRA on the instance
            trigger_word: LoRA trigger word
            num_outputs: Number of images to generate
            width: Output width
            height: Output height
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            timeout: Maximum wait time in seconds

        Returns:
            {"images": [base64_data_urls], "success": True}
        """
        import httpx

        if not session.backend_base_url:
            raise VastAIManagerError("No backend URL available")

        base_url = session.backend_base_url.rstrip("/")
        start_time = datetime.now(timezone.utc)

        # Step 1: Start async generation
        form_data = {
            "prompt": prompt,
            "lora_path": lora_path,
            "trigger_word": trigger_word,
            "num_outputs": str(num_outputs),
            "width": str(width),
            "height": str(height),
            "num_inference_steps": str(num_inference_steps),
            "guidance_scale": str(guidance_scale),
        }

        logger.info(f"Starting async generation: {prompt[:50]}...")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            response = await client.post(
                f"{base_url}/generate/async",
                data=form_data,
            )

            if response.status_code != 200:
                raise VastAIManagerError(
                    f"Failed to start generation: {response.status_code} - {response.text}"
                )

            result = response.json()
            gen_job_id = result["job_id"]
            logger.info(f"Generation job started: {gen_job_id}")

        # Step 2: Poll for completion
        poll_interval = 10
        elapsed = 0

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            while elapsed < timeout:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                response = await client.get(
                    f"{base_url}/generate/status/{gen_job_id}"
                )

                if response.status_code != 200:
                    logger.warning(f"Status check failed: {response.status_code}")
                    continue

                status = response.json()
                gen_status = status.get("status", "unknown")

                logger.info(f"[{elapsed}s] Generation status: {gen_status}")

                if gen_status == "completed":
                    images = status.get("images", [])
                    total_elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                    logger.info(f"Generation completed in {total_elapsed:.1f}s: {len(images)} images")
                    return {
                        "images": images,
                        "success": True,
                        "job_id": gen_job_id,
                        "elapsed_seconds": total_elapsed,
                    }

                if gen_status == "failed":
                    error = status.get("error", "Unknown error")
                    raise VastAIManagerError(f"Generation failed: {error}")

        raise VastAIManagerError(f"Generation timed out after {timeout}s")

    async def list_loras_http(self, session: VastAISession) -> List[Dict[str, Any]]:
        """
        List available trained LoRAs on the instance.

        Args:
            session: Active Vast.ai session

        Returns:
            List of {"id": str, "path": str, "size_mb": float}
        """
        import httpx

        if not session.backend_base_url:
            return []

        base_url = session.backend_base_url.rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
                response = await client.get(f"{base_url}/loras")

                if response.status_code != 200:
                    return []

                data = response.json()
                return data.get("loras", [])

        except Exception as e:
            logger.warning(f"Failed to list LoRAs: {e}")
            return []
