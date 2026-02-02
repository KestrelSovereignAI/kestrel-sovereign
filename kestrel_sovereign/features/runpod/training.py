"""
RunPod LoRA Training Methods.

Contains SSH-based and HTTP-based training methods for
LoRA training on RunPod GPU instances.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_DEFAULT,
    HTTP_TIMEOUT_DOWNLOAD,
    HTTP_TIMEOUT_UPLOAD,
    POD_READY_TIMEOUT,
    BACKEND_URL_TIMEOUT,
    BACKEND_URL_TIMEOUT_SHORT,
)
from .models import GPUProfile, PodStatus, RunPodManagerError, RunPodSession
from .providers import DirectRunPodProvider

logger = logging.getLogger(__name__)


class RunPodTrainingMixin:
    """
    Mixin for LoRA training operations on RunPod.

    Requires RunPodManagerCore as base class.
    """

    async def _wait_for_training_ready(
        self,
        session: RunPodSession,
        timeout: int = 600,  # 10 minutes default (model loading can take 5-10 min)
        poll_interval: int = 15
    ) -> None:
        """
        Wait for training pod's /ready endpoint to return 200.

        This is critical because:
        - The FLUX.2 model is ~24GB and takes 5-10 minutes to download on first run
        - Even cached, loading to GPU takes 1-2 minutes
        - /health returns OK while model is still loading
        - /ready returns 503 until model is fully loaded and GPU-ready

        Args:
            session: Active RunPod session with backend_base_url
            timeout: Max seconds to wait (default 10 minutes)
            poll_interval: Seconds between /ready checks

        Raises:
            RunPodManagerError: If not ready within timeout
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")

        ready_url = f"{session.backend_base_url}/ready"
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)

        logger.info(f"Waiting for training model to load at {ready_url} (may take 5-10 min on first run)...")

        attempts = 0
        last_status = None
        last_detail = None

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            while datetime.now(timezone.utc) < deadline:
                attempts += 1
                try:
                    response = await client.get(ready_url)

                    if response.status_code == 200:
                        data = response.json()
                        gpu = data.get("gpu", "unknown")
                        gpu_memory = data.get("gpu_memory_gb", "?")
                        logger.info(f"Training pod ready! GPU: {gpu} ({gpu_memory}GB)")
                        return

                    elif response.status_code == 503:
                        # Could be loading OR training in progress
                        try:
                            data = response.json()
                            detail = data.get("detail", "loading")
                        except Exception:
                            detail = response.text[:100]

                        # Check if another training is already running
                        if "Training in progress" in str(detail):
                            # Extract the job ID from the message
                            # Format: "Training in progress: {job_id}"
                            existing_job = str(detail).split(":")[-1].strip() if ":" in str(detail) else "unknown"
                            raise RunPodManagerError(
                                f"Cannot start training - another job is already running on this pod: {existing_job}. "
                                f"Wait for it to complete or cancel it first."
                            )

                        if detail != last_detail:
                            logger.info(f"Training pod not ready (attempt {attempts}): {detail}")
                            last_detail = detail

                    elif response.status_code == 404:
                        # /ready endpoint doesn't exist - SimpleTuner loads model on-demand
                        # Skip the wait and proceed directly to training
                        logger.info("Training pod has no /ready endpoint - SimpleTuner loads model on-demand, proceeding...")
                        return

                    else:
                        # Unexpected status
                        logger.warning(f"Unexpected /ready response: {response.status_code}")

                except httpx.ConnectError:
                    if last_status != "connect_error":
                        logger.info(f"Training pod not yet reachable (attempt {attempts})")
                        last_status = "connect_error"

                except httpx.TimeoutException:
                    if last_status != "timeout":
                        logger.warning(f"Training pod /ready timed out (attempt {attempts})")
                        last_status = "timeout"

                await asyncio.sleep(poll_interval)

        # Timeout reached
        raise RunPodManagerError(
            f"Training pod model not ready after {timeout}s ({attempts} attempts). "
            f"The FLUX model may still be downloading. Check pod logs."
        )

    async def start_training_pod(self, companion_id: str) -> Optional[RunPodSession]:
        """
        Start a pod for LoRA training using the training profile.

        If persistent_pod_id is configured:
        - Resume the existing pod if stopped (~10-30s)
        - Use the existing pod if already running (instant)
        - After training, pod should be stopped (paused) not terminated

        Otherwise tries profiles in order:
        1. "training" - A100 80GB in US-TX-3 (has network volume cache)
        2. "training-h100" - H100 80GB in US-TX-3 (faster but more expensive)
        3. "training-flex" - A100 80GB any datacenter (no network volume)

        Args:
            companion_id: Companion being trained (for naming/tracking)

        Returns:
            RunPodSession if started successfully, None otherwise
        """
        ttl_seconds = 3600  # 1 hour max for training
        profiles_to_try = ["training", "training-h100", "training-flex"]

        # Check if training profile has a persistent pod configured
        if "training" in self.profiles:
            profile = self.profiles["training"]
            # Expand env var at RUNTIME (not at load time) so server doesn't need restart
            persistent_pod_id = self._expand_single_env_var(profile.persistent_pod_id)
            if persistent_pod_id:
                logger.info(f"Using persistent training pod: {persistent_pod_id}")
                return await self._use_persistent_pod(persistent_pod_id, profile, ttl_seconds)

        # Try to resume a stopped training pod first (much faster)
        stopped_pod = await self.find_stopped_pod("lora_training", "training")
        if stopped_pod:
            try:
                profile = self._select_profile("training")
                logger.info("Resuming stopped training pod (10-30s vs 2-5min for new)")
                return await self.resume_stopped_pod(stopped_pod, profile, ttl_seconds)
            except RunPodManagerError as e:
                logger.warning(f"Failed to resume stopped pod: {e}")

        # Try each profile in order
        last_error = None
        for profile_name in profiles_to_try:
            if profile_name not in self.profiles:
                continue

            try:
                logger.info(f"Trying training profile: {profile_name}")
                result = await self.start_session(
                    task_profile=profile_name,
                    model_name="flux-lora-trainer",
                    ttl_seconds=ttl_seconds,
                    metadata={
                        "name": f"kestrel-lora-{companion_id[:8]}",
                        "companion_id": companion_id,
                        "purpose": "lora_training"
                    }
                )

                async with self._lock:
                    session = self._session
                    if session:
                        # Verify backend URL is available (required for training)
                        if not session.backend_base_url:
                            logger.error(f"Pod started but no backend URL after ready - check RunPod pod ports")
                            # Try to stop the pod that has no URL
                            try:
                                await self.stop_session()
                            except Exception:
                                pass
                            last_error = RunPodManagerError(f"Pod {session.pod_id} has no backend URL - ports not assigned")
                            continue
                        logger.info(f"Training pod started with profile {profile_name}, URL: {session.backend_base_url}")
                        return session

            except RunPodManagerError as e:
                logger.warning(f"Profile {profile_name} failed: {e}")
                last_error = e
                # Continue to next profile
                continue

        # All profiles failed
        if last_error:
            logger.error(f"All training profiles failed. Last error: {last_error}")
        return None

    async def _use_persistent_pod(self, pod_id: str, profile: GPUProfile, ttl_seconds: int) -> Optional[RunPodSession]:
        """
        Use an existing persistent pod - resume if stopped, connect if running.

        This is the preferred mode for training pods:
        - No startup delay if already running
        - ~10-30s resume time if stopped
        - Models stay cached on network volume
        """
        try:
            # Get current pod status
            pod_info = await asyncio.to_thread(self.provider.get_status, pod_id)
            if not pod_info:
                logger.error(f"Persistent pod {pod_id} not found")
                return None

            status = pod_info.get("desiredStatus") or pod_info.get("status")
            logger.info(f"Persistent pod {pod_id} status: {status}")

            # Create session object
            now = datetime.now(timezone.utc)
            session = RunPodSession(
                pod_id=pod_id,
                profile=profile,
                task_profile="training",
                model_name="flux-lora-trainer",
                pod_type=profile.pod_type,
                status=self._map_status(status),
                ttl_seconds=ttl_seconds,
                started_at=now,
                expires_at=now + timedelta(seconds=ttl_seconds),
            )

            # Resume if stopped/exited
            if status in ("EXITED", "exited", "STOPPED", "stopped"):
                logger.info(f"Resuming stopped persistent pod {pod_id} (status={status})")
                gpu_count = 1
                try:
                    result = await asyncio.to_thread(self.provider.resume_pod, pod_id, gpu_count)
                    logger.info(f"Resume API call returned: {result}")
                except Exception as resume_err:
                    logger.error(f"Failed to resume pod {pod_id}: {resume_err}")
                    raise
                # Wait for it to be ready
                logger.info(f"Waiting for pod {pod_id} to become ready (up to {POD_READY_TIMEOUT}s)...")
                await self._wait_for_pod_ready(session, timeout=POD_READY_TIMEOUT)
                logger.info(f"Pod {pod_id} is now ready")

            # If running, just wait for backend URL
            elif status in ("RUNNING", "running"):
                logger.info(f"Persistent pod {pod_id} already running")
                # Update runtime info to get ports
                self._update_session_from_runtime(session, pod_info)
                # If no backend URL yet, wait for it
                if not session.backend_base_url:
                    await self._wait_for_backend_url(session, timeout=BACKEND_URL_TIMEOUT_SHORT)

            else:
                logger.warning(f"Persistent pod {pod_id} in unexpected state: {status}")
                return None

            # Store session
            async with self._lock:
                self._session = session

            if not session.backend_base_url:
                logger.error(f"Persistent pod {pod_id} has no backend URL")
                return None

            logger.info(f"Using persistent pod {pod_id}, URL: {session.backend_base_url}")
            return session

        except Exception as e:
            logger.error(f"Failed to use persistent pod {pod_id}: {e}")
            return None

    async def _wait_for_pod_ready(self, session: RunPodSession, timeout: int = 300) -> None:
        """Wait for a pod to reach RUNNING status."""
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        while datetime.now(timezone.utc) < deadline:
            pod_info = await asyncio.to_thread(self.provider.get_status, session.pod_id)
            status = pod_info.get("desiredStatus") or pod_info.get("status")
            session.status = self._map_status(status)
            self._update_session_from_runtime(session, pod_info)

            if status in ("RUNNING", "running"):
                logger.info(f"Pod {session.pod_id} is now running")
                # Wait for backend URL
                await self._wait_for_backend_url(session, timeout=BACKEND_URL_TIMEOUT)
                return

            logger.debug(f"Pod {session.pod_id} status: {status}, waiting...")
            await asyncio.sleep(5)

        raise RunPodManagerError(f"Pod {session.pod_id} did not become ready within {timeout}s")

    async def _wait_for_backend_url(self, session: RunPodSession, timeout: int = 120) -> None:
        """Wait for backend URL to be populated (ports assigned)."""
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        while datetime.now(timezone.utc) < deadline:
            if session.backend_base_url:
                return

            # Refresh pod info
            pod_info = await asyncio.to_thread(self.provider.get_status, session.pod_id)
            self._update_session_from_runtime(session, pod_info)

            if session.backend_base_url:
                logger.info(f"Backend URL ready: {session.backend_base_url}")
                return

            logger.debug(f"Waiting for backend URL on pod {session.pod_id}...")
            await asyncio.sleep(5)

        logger.warning(f"Pod {session.pod_id} has no backend URL after {timeout}s")

    async def start_inference_pod(self, companion_id: str) -> Optional[RunPodSession]:
        """
        Start a pod for LoRA-based image generation.

        Tries to resume a stopped pod first (~10-30s) before creating new (~2-5min).

        Args:
            companion_id: Companion for tracking

        Returns:
            RunPodSession if started successfully, None otherwise
        """
        try:
            profile = self._select_profile("image")
            ttl_seconds = 600  # 10 min for inference

            # Try to resume a stopped inference pod first (much faster)
            stopped_pod = await self.find_stopped_pod("lora_inference", "image")
            if stopped_pod:
                logger.info(f"Resuming stopped inference pod (10-30s vs 2-5min for new)")
                return await self.resume_stopped_pod(stopped_pod, profile, ttl_seconds)

            # No stopped pod found, create new one
            result = await self.start_session(
                task_profile="image",
                model_name="flux-with-lora",
                ttl_seconds=ttl_seconds,
                metadata={
                    "name": f"kestrel-selfie-{companion_id[:8]}",
                    "companion_id": companion_id,
                    "purpose": "lora_inference"
                }
            )

            async with self._lock:
                return self._session

        except RunPodManagerError as e:
            logger.error(f"Failed to start inference pod for {companion_id}: {e}")
            return None

    async def submit_training_job(
        self,
        session: RunPodSession,
        avatar_data: bytes,
        companion_id: str,
        callback_url: Optional[str] = None,
        wait_for_model_ready: bool = True
    ) -> str:
        """
        Submit training job to pod's /train endpoint.

        IMPORTANT: The training pod needs 5-10 minutes to load the FLUX model
        after startup. We wait for the /ready endpoint before submitting.

        Args:
            session: Active RunPod session
            avatar_data: Avatar image bytes from sovereign storage
            companion_id: Companion ID
            callback_url: Optional webhook for completion
            wait_for_model_ready: If True, wait for /ready endpoint before submitting.
                                  Model loading can take 5-10 minutes on first run.

        Returns:
            Training job ID from the pod
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")

        # Wait for the model to be loaded before submitting training
        # This is critical - /health returns OK while model is still loading!
        if wait_for_model_ready:
            await self._wait_for_training_ready(session)

        train_url = f"{session.backend_base_url}/train"
        logger.info(f"Submitting training job to {train_url}")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_UPLOAD) as client:
            try:
                # Use avatar bytes directly from sovereign storage
                logger.info(f"Using avatar data from sovereign storage: {len(avatar_data)} bytes")

                # Detect content type from magic bytes
                if avatar_data[:8] == b'\x89PNG\r\n\x1a\n':
                    content_type = "image/png"
                    filename = "avatar.png"
                else:
                    content_type = "image/jpeg"
                    filename = "avatar.jpg"

                # Submit as multipart form with image file
                files = {"image": (filename, avatar_data, content_type)}
                data = {
                    "companion_id": companion_id,
                }
                if callback_url:
                    data["callback_url"] = callback_url

                response = await client.post(train_url, files=files, data=data)
                response.raise_for_status()
                result = response.json()
            except httpx.ConnectError as e:
                raise RunPodManagerError(f"Cannot connect to training pod at {train_url}: {e}") from e
            except httpx.HTTPStatusError as e:
                error_body = e.response.text[:500] if e.response else "No response body"
                raise RunPodManagerError(f"Training pod returned HTTP {e.response.status_code}: {error_body}") from e
            except httpx.TimeoutException as e:
                raise RunPodManagerError(f"Timeout connecting to training pod at {train_url}") from e

        job_id = result.get("job_id")
        if not job_id:
            raise RunPodManagerError(f"Training job did not return job_id: {result}")

        logger.info(f"Training job submitted: {job_id}")
        return job_id

    async def get_current_job(self, session: RunPodSession) -> Optional[Dict[str, Any]]:
        """
        Check if a training job is currently running on the pod.

        Args:
            session: Active RunPod session

        Returns:
            Job info dict if training in progress, None if idle
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")

        url = f"{session.backend_base_url}/current-job"

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            try:
                response = await client.get(url)
                if response.status_code == 404:
                    # Endpoint not available in older container versions
                    return None
                response.raise_for_status()
                data = response.json()
                if data.get("current_job"):
                    return data
                return None
            except httpx.HTTPStatusError:
                return None

    async def cancel_training_job(self, session: RunPodSession, job_id: str) -> Dict[str, Any]:
        """
        Cancel a training job.

        Note: This marks the job as cancelled but may not stop the actual
        training process. For stuck jobs, pod restart may be needed.

        Args:
            session: Active RunPod session
            job_id: Job ID to cancel

        Returns:
            Cancellation result
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")

        url = f"{session.backend_base_url}/cancel/{job_id}"

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            response = await client.post(url)
            response.raise_for_status()
            return response.json()

    async def clear_current_job(self, session: RunPodSession) -> Dict[str, Any]:
        """
        Force-clear the current job lock on the pod.

        USE WITH CAUTION: Only use when a job is stuck and unresponsive.
        This clears the lock but does NOT kill any running processes.

        Args:
            session: Active RunPod session

        Returns:
            Result with cleared_job info
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")

        url = f"{session.backend_base_url}/clear-current-job"

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            response = await client.post(url)
            response.raise_for_status()
            result = response.json()
            logger.warning(f"Force-cleared job lock on pod: {result}")
            return result

    async def poll_training_status(self, session: RunPodSession, job_id: str) -> Dict[str, Any]:
        """
        Get training job status from pod.

        Args:
            session: Active RunPod session
            job_id: Training job ID

        Returns:
            Status dict with: status, progress, error, output_path
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")

        status_url = f"{session.backend_base_url}/status/{job_id}"

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            response = await client.get(status_url)
            response.raise_for_status()
            return response.json()

    async def download_lora(self, session: RunPodSession, job_id: str) -> bytes:
        """
        Download trained LoRA file from pod.

        Args:
            session: Active RunPod session
            job_id: Completed training job ID

        Returns:
            LoRA file bytes (.safetensors)
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")

        download_url = f"{session.backend_base_url}/download/{job_id}"
        logger.info(f"Downloading LoRA from {download_url}")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DOWNLOAD) as client:
            response = await client.get(download_url)
            response.raise_for_status()
            return response.content

    async def generate_with_lora(
        self,
        session: RunPodSession,
        prompt: str,
        lora_path: str,
        num_outputs: int = 1
    ) -> Dict[str, Any]:
        """
        Generate images using loaded LoRA model.

        Args:
            session: Active RunPod session
            prompt: Image generation prompt
            lora_path: Path to LoRA file (in Kestrel storage)
            num_outputs: Number of images to generate

        Returns:
            Dict with "images" list of URLs/base64
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")

        # Use the image generation endpoint
        generate_url = f"{session.backend_base_url}/generate"
        logger.info(f"Generating with LoRA at {generate_url}")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_UPLOAD) as client:
            response = await client.post(
                generate_url,
                json={
                    "prompt": prompt,
                    "lora_path": lora_path,
                    "num_outputs": num_outputs,
                    "aspect_ratio": "1:1",
                    "output_format": "jpg"
                }
            )
            response.raise_for_status()
            return response.json()
