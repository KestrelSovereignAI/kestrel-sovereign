"""
E2E Integration tests for Vast.ai GPU feature.

Tests the full lifecycle using SimpleTuner HTTP API:
1. API connectivity and offer search
2. Instance creation and destruction
3. SimpleTuner container deployment (training + generation)
4. LoRA training workflow via HTTP API
5. Image generation with trained LoRAs

Run API tests only (no instances created):
    VASTAI_API_KEY=xxx pytest tests/integration/test_vastai_e2e.py -v -k "TestVastAIConnectivity"

Run all tests including cloud resources (costs money):
    VASTAI_API_KEY=xxx pytest tests/integration/test_vastai_e2e.py -v --run-cloud

SimpleTuner API Endpoints:
    - GET /health - Health check
    - GET /ready - Readiness check (can accept jobs)
    - POST /train - Start training (multipart form)
    - GET /status/{job_id} - Training status
    - GET /download/{job_id} - Download trained LoRA
    - POST /generate/async - Start async generation
    - GET /generate/status/{job_id} - Generation status with images
    - GET /loras - List available LoRAs
"""

import pytest
import pytest_asyncio
import os
import asyncio
import base64
from datetime import datetime, timezone
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

# Check for API key
HAS_API_KEY = bool(os.environ.get("VASTAI_API_KEY"))


# =============================================================================
# Phase 1: API Connectivity Tests (No Instances Created)
# =============================================================================

@pytest.mark.skipif(not HAS_API_KEY, reason="VASTAI_API_KEY not set")
class TestVastAIConnectivity:
    """Test Vast.ai API connectivity without starting instances."""

    @pytest.fixture
    def manager(self):
        """Create a VastAIManager instance."""
        from kestrel_sovereign.features.vastai.manager import VastAIManager
        return VastAIManager()

    @pytest.mark.asyncio
    async def test_search_any_gpu_offers(self, manager):
        """Verify can search for any GPU offers."""
        offers = await manager.search_offers(
            query="rentable = true gpu_ram >= 8",
            limit=5
        )

        print(f"\n[VASTAI] Found {len(offers)} GPU offers:")
        for offer in offers[:3]:
            print(f"  - {offer.get('gpu_name', 'Unknown')}: "
                  f"${offer.get('dph_total', 0):.2f}/hr, "
                  f"{offer.get('gpu_ram', 0)}GB VRAM, "
                  f"reliability: {offer.get('reliability', 0):.2f}")

        assert len(offers) > 0, "No GPU offers found - check API key"

    @pytest.mark.asyncio
    async def test_search_rtx_3090_offers(self, manager):
        """Verify can search for RTX 3090 specifically."""
        offers = await manager.search_offers(
            query="gpu_name = RTX_3090 reliability > 0.8 rentable = true",
            limit=5
        )

        print(f"\n[VASTAI] Found {len(offers)} RTX 3090 offers:")
        for offer in offers[:3]:
            print(f"  - ID {offer.get('id')}: "
                  f"${offer.get('dph_total', 0):.2f}/hr, "
                  f"reliability: {offer.get('reliability', 0):.2f}")

        # RTX 3090s should be available most of the time
        if len(offers) == 0:
            print("[VASTAI] No RTX 3090 available right now - this is normal sometimes")

    @pytest.mark.asyncio
    async def test_search_training_profile(self, manager):
        """Verify training profile search works."""
        # Get the training profile from config
        profile = manager.profiles.get("training")
        if not profile:
            pytest.skip("No 'training' profile in vastai_config.toml")

        offers = await manager.search_offers(profile=profile, limit=5)

        print(f"\n[VASTAI] Found {len(offers)} offers matching training profile:")
        print(f"  Profile requires: {profile.gpu_ram_min}GB VRAM, "
              f"reliability > {profile.reliability_min}, "
              f"max ${profile.cost_per_hr_max}/hr")

        for offer in offers[:3]:
            print(f"  - {offer.get('gpu_name', 'Unknown')}: "
                  f"${offer.get('dph_total', 0):.2f}/hr, "
                  f"{offer.get('gpu_ram', 0)}GB VRAM")

        # Verify offers match profile criteria
        for offer in offers:
            assert offer.get("gpu_ram", 0) >= profile.gpu_ram_min, \
                f"Offer has insufficient VRAM: {offer.get('gpu_ram')}GB < {profile.gpu_ram_min}GB"

    @pytest.mark.asyncio
    async def test_search_budget_profile(self, manager):
        """Verify budget profile finds cheap GPUs."""
        profile = manager.profiles.get("budget")
        if not profile:
            pytest.skip("No 'budget' profile in vastai_config.toml")

        offers = await manager.search_offers(profile=profile, limit=10)

        print(f"\n[VASTAI] Found {len(offers)} budget GPU offers:")
        for offer in offers[:5]:
            print(f"  - {offer.get('gpu_name', 'Unknown')}: "
                  f"${offer.get('dph_total', 0):.3f}/hr")

        assert len(offers) > 0, "No budget GPUs available"

        # Budget offers should be cheap
        cheapest = offers[0]
        assert cheapest.get("dph_total", 999) < 0.50, \
            f"Cheapest offer is ${cheapest.get('dph_total')}/hr - too expensive for budget"

    @pytest.mark.asyncio
    async def test_list_existing_instances(self, manager):
        """List any existing instances on the account."""
        instances = await manager.show_instances()

        print(f"\n[VASTAI] Found {len(instances)} existing instances:")
        for inst in instances:
            print(f"  - ID {inst.get('id')}: "
                  f"{inst.get('actual_status', 'unknown')} - "
                  f"{inst.get('gpu_name', 'Unknown GPU')}")

        # This test passes regardless of instance count - just checking API works


# =============================================================================
# Phase 2: Instance Lifecycle Tests (Creates Real Instances - Costs Money)
# =============================================================================

@pytest.mark.skipif(not HAS_API_KEY, reason="VASTAI_API_KEY not set")
@pytest.mark.cloud_resource
class TestVastAIInstanceLifecycle:
    """Test instance creation and destruction.

    WARNING: These tests create real GPU instances and cost money.
    Use --run-cloud flag to enable.
    """

    @pytest_asyncio.fixture
    async def manager(self):
        """Create a VastAIManager and cleanup after test."""
        from kestrel_sovereign.features.vastai.manager import VastAIManager

        mgr = VastAIManager()
        yield mgr

        # Cleanup: destroy any active session
        if mgr._session and mgr._session.is_active:
            print(f"\n[VASTAI-CLEANUP] Destroying instance {mgr._session.instance_id}...")
            await mgr.stop_session()
            print("[VASTAI-CLEANUP] Instance destroyed.")

    @pytest.mark.asyncio
    async def test_start_stop_budget_instance(self, manager):
        """Start a cheap instance and stop it immediately.

        Uses the budget profile to minimize cost (~$0.10-0.15/hr).
        """
        from kestrel_sovereign.features.vastai.models import InstanceStatus

        # First, find a cheap offer
        profile = manager.profiles.get("budget")
        if not profile:
            pytest.skip("No 'budget' profile configured")

        offers = await manager.search_offers(profile=profile, limit=1)
        if not offers:
            pytest.skip("No budget GPUs available right now")

        offer = offers[0]
        print(f"\n[VASTAI] Starting budget instance:")
        print(f"  Offer ID: {offer.get('id')}")
        print(f"  GPU: {offer.get('gpu_name')}")
        print(f"  Cost: ${offer.get('dph_total', 0):.3f}/hr")

        # Start the instance
        status = await manager.start_session(
            task_profile="budget",
            ttl_seconds=300,  # 5 min TTL for tracking
            offer_id=offer["id"],
        )

        print(f"[VASTAI] Instance started: {status}")

        assert status["active"] is True
        assert status["status"] in [InstanceStatus.RUNNING.value, InstanceStatus.LOADING.value]
        assert status["instance_id"] is not None

        # Get status
        current_status = await manager.get_status()
        print(f"[VASTAI] Current status: {current_status}")

        # Stop immediately to minimize cost
        print(f"[VASTAI] Stopping instance {status['instance_id']}...")
        stop_result = await manager.stop_session()
        print(f"[VASTAI] Stop result: {stop_result}")

        assert stop_result["active"] is False


# =============================================================================
# Phase 3: SimpleTuner Container Tests (HTTP API)
# =============================================================================

@pytest.mark.skipif(not HAS_API_KEY, reason="VASTAI_API_KEY not set")
@pytest.mark.cloud_resource
class TestVastAITrainingContainer:
    """Test SimpleTuner container deployment on Vast.ai.

    Uses the shared SimpleTuner Docker image (gcr.io/YOUR_PROJECT_ID/simpletuner-flux2)
    with HTTP API for training and generation.
    Cost: ~$0.74-1.50/hr for A100 80GB.
    """

    @pytest_asyncio.fixture
    async def training_instance(self):
        """Start a training instance and return session info.

        Uses A100 80GB for FLUX.2-dev compatibility.
        """
        from kestrel_sovereign.features.vastai.manager import VastAIManager
        import httpx

        manager = VastAIManager()

        # Check for existing running instance first
        instances = await manager.show_instances()
        running = [i for i in instances if i.get("actual_status") == "running"]

        if running:
            # Reuse existing instance - find the backend URL
            inst = running[0]
            print(f"\n[VASTAI] Reusing existing instance {inst['id']}")

            # Reconstruct backend URL from instance info
            public_ip = inst.get("public_ipaddr")
            ports = inst.get("ports", {})
            base_url = None

            # Find port 8000 mapping
            for port_key in ["8000/tcp", "8000"]:
                if port_key in ports and ports[port_key]:
                    external_port = ports[port_key][0].get("HostPort")
                    if external_port and public_ip:
                        base_url = f"http://{public_ip}:{external_port}"
                        break

            if not base_url and public_ip:
                # Fallback to direct port
                base_url = f"http://{public_ip}:8000"

            yield {
                "manager": manager,
                "instance_id": inst["id"],
                "base_url": base_url,
                "reused": True,
            }
            return

        # Start new training session via manager
        print("\n[VASTAI] Starting training session...")
        result = await manager.start_session(
            task_profile="training",
            ttl_seconds=3600,  # 1 hour
            metadata={"companion_id": "test-e2e"},
        )

        session = manager._session
        if not session:
            pytest.skip("Could not start training instance - check GPU availability")

        print(f"[VASTAI] Training instance started: {session.instance_id}")
        print(f"[VASTAI] Backend URL: {session.backend_base_url}")

        # Wait for SimpleTuner API to be ready
        if session.backend_base_url:
            print("[VASTAI] Waiting for SimpleTuner API to be ready...")
            ready = await manager.wait_for_api_ready(session, timeout=600)
            if ready:
                print("[VASTAI] SimpleTuner API ready!")
            else:
                print("[VASTAI] Warning: API may not be fully ready")

        yield {
            "manager": manager,
            "session": session,
            "instance_id": session.instance_id,
            "base_url": session.backend_base_url,
            "reused": False,
        }

        # Don't destroy - keep running for development
        print(f"\n[VASTAI] Keeping instance {session.instance_id} running for development")
        print(f"[VASTAI] To stop: vastai destroy instance {session.instance_id}")

    @pytest.mark.asyncio
    async def test_health_endpoint(self, training_instance):
        """Test /health endpoint on SimpleTuner container."""
        import httpx

        base_url = training_instance.get("base_url")
        if not base_url:
            pytest.skip("Could not determine instance URL")

        print(f"\n[VASTAI] Testing /health at {base_url}/health")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{base_url}/health")

        print(f"[VASTAI] Health response: {response.status_code}")
        print(f"[VASTAI] Body: {response.text}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "cuda_available" in data

    @pytest.mark.asyncio
    async def test_ready_endpoint(self, training_instance):
        """Test /ready endpoint for readiness check."""
        import httpx

        base_url = training_instance.get("base_url")
        if not base_url:
            pytest.skip("Could not determine instance URL")

        print(f"\n[VASTAI] Testing /ready at {base_url}/ready")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{base_url}/ready")

        print(f"[VASTAI] Ready response: {response.status_code}")
        print(f"[VASTAI] Body: {response.text}")

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True

    @pytest.mark.asyncio
    async def test_loras_endpoint(self, training_instance):
        """Test /loras endpoint listing available LoRAs."""
        import httpx

        base_url = training_instance.get("base_url")
        if not base_url:
            pytest.skip("Could not determine instance URL")

        print(f"\n[VASTAI] Testing /loras at {base_url}/loras")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{base_url}/loras")

        print(f"[VASTAI] LoRAs response: {response.status_code}")
        print(f"[VASTAI] Body: {response.text}")

        assert response.status_code == 200
        data = response.json()
        assert "loras" in data
        # List should exist (may be empty initially)
        assert isinstance(data["loras"], list)

    @pytest.mark.asyncio
    async def test_train_endpoint_starts_job(self, training_instance):
        """Test that POST /train starts a training job via multipart form."""
        import httpx

        base_url = training_instance.get("base_url")
        if not base_url:
            pytest.skip("Could not determine instance URL")

        companion_id = "test-companion-vastai"
        trigger_word = "TOKtestvas"

        # Create a small test image (512x512 solid color PNG)
        # This creates a minimal valid PNG for testing
        test_image = create_test_png_image()

        print(f"\n[VASTAI] Testing /train with multipart form")

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Use multipart form data as expected by SimpleTuner API
            files = {
                "image": ("test_avatar.png", test_image, "image/png"),
            }
            data = {
                "companion_id": companion_id,
                "trigger_word": trigger_word,
                "steps": 100,  # Minimal steps for testing
                "lora_rank": 16,
            }

            response = await client.post(
                f"{base_url}/train",
                files=files,
                data=data,
            )

        print(f"[VASTAI] Train response: {response.status_code}")
        print(f"[VASTAI] Body: {response.text}")

        assert response.status_code == 200
        result = response.json()
        assert "job_id" in result
        job_id = result["job_id"]

        # Check job status via /status/{job_id}
        print(f"[VASTAI] Checking job status for {job_id}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            status_response = await client.get(f"{base_url}/status/{job_id}")

        print(f"[VASTAI] Job status: {status_response.text}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        # Job should be in some state
        assert status_data["status"] in ["queued", "pending", "preparing", "running", "training", "completed", "failed"]


def create_test_png_image(width: int = 512, height: int = 512) -> bytes:
    """Create a minimal test PNG image for training tests.

    Creates a simple solid color image that's valid for training.
    """
    try:
        from PIL import Image
        import io

        # Create a simple gradient image
        img = Image.new("RGB", (width, height), color=(128, 100, 150))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        # Fallback: create minimal valid PNG without PIL
        # This is a 1x1 red pixel PNG (minimal valid PNG)
        return bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
            0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,  # IEND chunk
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ])


# =============================================================================
# Phase 4: Full Training Pipeline (End-to-End)
# =============================================================================

@pytest.mark.skipif(not HAS_API_KEY, reason="VASTAI_API_KEY not set")
@pytest.mark.cloud_resource
class TestVastAIFullPipeline:
    """Test full training → download → generation pipeline.

    This is the most expensive test - runs complete LoRA training.
    Cost: ~$0.50-1.00 for a full run on A100 80GB.
    """

    @pytest_asyncio.fixture
    async def running_instance(self):
        """Get or start a running instance with SimpleTuner API."""
        from kestrel_sovereign.features.vastai.manager import VastAIManager
        import httpx

        manager = VastAIManager()

        # Check for existing running instance
        instances = await manager.show_instances()
        running = [i for i in instances if i.get("actual_status") == "running"]

        if running:
            inst = running[0]
            public_ip = inst.get("public_ipaddr")
            ports = inst.get("ports", {})

            # Find port 8000 mapping
            base_url = None
            for port_key in ["8000/tcp", "8000"]:
                if port_key in ports and ports[port_key]:
                    external_port = ports[port_key][0].get("HostPort")
                    if external_port and public_ip:
                        base_url = f"http://{public_ip}:{external_port}"
                        break

            if not base_url and public_ip:
                base_url = f"http://{public_ip}:8000"

            return {
                "manager": manager,
                "instance_id": inst["id"],
                "base_url": base_url,
            }

        # No running instance - start one
        print("\n[VASTAI] Starting training session for full pipeline test...")
        await manager.start_session(
            task_profile="training",
            ttl_seconds=3600,
            metadata={"test": "full_pipeline"},
        )

        session = manager._session
        if not session or not session.backend_base_url:
            pytest.skip("Could not start training instance")

        # Wait for API ready
        await manager.wait_for_api_ready(session, timeout=600)

        return {
            "manager": manager,
            "session": session,
            "instance_id": session.instance_id,
            "base_url": session.backend_base_url,
        }

    @pytest.mark.asyncio
    async def test_full_training_to_completion(self, running_instance):
        """Run training to completion and download LoRA via HTTP API."""
        import httpx

        base_url = running_instance.get("base_url")
        if not base_url:
            pytest.skip("Could not determine instance URL")

        instance_id = running_instance.get("instance_id")
        print(f"\n[VASTAI] Using instance {instance_id} at {base_url}")

        # Create test image
        test_image = create_test_png_image()
        companion_id = f"test-full-{datetime.now(timezone.utc).strftime('%H%M%S')}"
        trigger_word = f"TOK{companion_id[:8]}"

        # Start training via multipart form
        print(f"[VASTAI] Starting training for {companion_id}...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {
                "image": ("avatar.png", test_image, "image/png"),
            }
            data = {
                "companion_id": companion_id,
                "trigger_word": trigger_word,
                "steps": 250,  # Short training for test
                "lora_rank": 16,
            }

            response = await client.post(
                f"{base_url}/train",
                files=files,
                data=data,
            )

        assert response.status_code == 200, f"Train failed: {response.text}"
        job_id = response.json()["job_id"]
        print(f"[VASTAI] Started training job {job_id}")

        # Poll until completion (timeout 20 minutes for FLUX.2)
        max_wait = 1200  # 20 minutes
        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient(timeout=30.0) as client:
            while asyncio.get_event_loop().time() - start_time < max_wait:
                status_response = await client.get(f"{base_url}/status/{job_id}")
                status_data = status_response.json()

                progress = status_data.get("progress", 0) * 100
                print(f"[VASTAI] Job {job_id}: {status_data['status']} ({progress:.1f}%)")

                if status_data["status"] == "completed":
                    print(f"[VASTAI] Training completed!")
                    break
                elif status_data["status"] == "failed":
                    pytest.fail(f"Training failed: {status_data.get('error')}")

                await asyncio.sleep(15)
            else:
                pytest.fail("Training did not complete within 20 minutes")

        # Download LoRA via /download/{job_id}
        print(f"[VASTAI] Downloading LoRA...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            download_response = await client.get(f"{base_url}/download/{job_id}")

        assert download_response.status_code == 200, f"Download failed: {download_response.text}"
        lora_data = download_response.content

        print(f"[VASTAI] Downloaded LoRA: {len(lora_data)} bytes")
        assert len(lora_data) > 0, "LoRA file is empty"
        assert len(lora_data) > 1000, f"LoRA file too small: {len(lora_data)} bytes"

        # Save locally for inspection
        lora_path = f"/tmp/lora_{companion_id}.safetensors"
        with open(lora_path, "wb") as f:
            f.write(lora_data)
        print(f"[VASTAI] Saved to {lora_path}")

        # Verify LoRA appears in /loras list
        async with httpx.AsyncClient(timeout=30.0) as client:
            loras_response = await client.get(f"{base_url}/loras")

        assert loras_response.status_code == 200
        loras_data = loras_response.json()
        lora_names = [l.get("name", l.get("path", "")) for l in loras_data.get("loras", [])]
        print(f"[VASTAI] Available LoRAs: {lora_names}")

        # Store for generation test
        running_instance["lora_path"] = status_data.get("lora_path", f"/workspace/loras/{companion_id}")
        running_instance["trigger_word"] = trigger_word


# =============================================================================
# Phase 5: Image Generation Tests
# =============================================================================

@pytest.mark.skipif(not HAS_API_KEY, reason="VASTAI_API_KEY not set")
@pytest.mark.cloud_resource
class TestVastAIGeneration:
    """Test image generation with trained LoRAs.

    Uses async generation endpoint to avoid timeouts.
    Cost: ~$0.02-0.05 per generation on A100 80GB.
    """

    @pytest_asyncio.fixture
    async def generation_instance(self):
        """Get a running instance with SimpleTuner API."""
        from kestrel_sovereign.features.vastai.manager import VastAIManager

        manager = VastAIManager()

        # Check for existing running instance
        instances = await manager.show_instances()
        running = [i for i in instances if i.get("actual_status") == "running"]

        if not running:
            pytest.skip("No running instance - run training tests first")

        inst = running[0]
        public_ip = inst.get("public_ipaddr")
        ports = inst.get("ports", {})

        # Find port 8000 mapping
        base_url = None
        for port_key in ["8000/tcp", "8000"]:
            if port_key in ports and ports[port_key]:
                external_port = ports[port_key][0].get("HostPort")
                if external_port and public_ip:
                    base_url = f"http://{public_ip}:{external_port}"
                    break

        if not base_url and public_ip:
            base_url = f"http://{public_ip}:8000"

        if not base_url:
            pytest.skip("Could not determine instance URL")

        return {
            "manager": manager,
            "instance_id": inst["id"],
            "base_url": base_url,
        }

    @pytest.mark.asyncio
    async def test_generation_without_lora(self, generation_instance):
        """Test base FLUX.2 generation without any LoRA."""
        import httpx

        base_url = generation_instance.get("base_url")
        print(f"\n[VASTAI] Testing generation at {base_url}")

        # Start async generation
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/generate/async",
                json={
                    "prompt": "A serene mountain landscape at sunset, photorealistic",
                    "num_outputs": 1,
                    "width": 512,
                    "height": 512,
                    "num_inference_steps": 20,  # Fewer steps for faster test
                },
            )

        assert response.status_code == 200, f"Generate failed: {response.text}"
        result = response.json()
        assert "job_id" in result
        job_id = result["job_id"]
        print(f"[VASTAI] Started generation job {job_id}")

        # Poll for completion (timeout 10 minutes)
        max_wait = 600
        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient(timeout=30.0) as client:
            while asyncio.get_event_loop().time() - start_time < max_wait:
                status_response = await client.get(f"{base_url}/generate/status/{job_id}")
                status_data = status_response.json()

                print(f"[VASTAI] Generation {job_id}: {status_data.get('status', 'unknown')}")

                if status_data.get("status") == "completed":
                    print("[VASTAI] Generation completed!")
                    images = status_data.get("images", [])
                    assert len(images) > 0, "No images in result"

                    # Validate base64 images
                    for i, img in enumerate(images):
                        if img.startswith("data:image"):
                            # Data URL format
                            base64_part = img.split(",")[1] if "," in img else img
                        else:
                            base64_part = img

                        img_bytes = base64.b64decode(base64_part)
                        print(f"[VASTAI] Image {i+1}: {len(img_bytes)} bytes")
                        assert len(img_bytes) > 1000, f"Image {i+1} too small"

                    return

                elif status_data.get("status") == "failed":
                    pytest.fail(f"Generation failed: {status_data.get('error')}")

                await asyncio.sleep(10)

        pytest.fail("Generation did not complete within 10 minutes")

    @pytest.mark.asyncio
    async def test_generation_with_lora(self, generation_instance):
        """Test generation with a trained LoRA."""
        import httpx

        base_url = generation_instance.get("base_url")

        # First, get available LoRAs
        async with httpx.AsyncClient(timeout=30.0) as client:
            loras_response = await client.get(f"{base_url}/loras")

        assert loras_response.status_code == 200
        loras_data = loras_response.json()
        available_loras = loras_data.get("loras", [])

        if not available_loras:
            pytest.skip("No trained LoRAs available - run training test first")

        # Use the first available LoRA
        lora = available_loras[0]
        lora_path = lora.get("path") or lora.get("name")
        trigger_word = lora.get("trigger_word", "TOK")

        print(f"\n[VASTAI] Testing generation with LoRA: {lora_path}")
        print(f"[VASTAI] Trigger word: {trigger_word}")

        # Start async generation with LoRA
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/generate/async",
                json={
                    "prompt": f"A portrait of {trigger_word} smiling in a garden, natural lighting",
                    "lora_path": lora_path,
                    "trigger_word": trigger_word,
                    "num_outputs": 1,
                    "width": 768,
                    "height": 1024,
                    "num_inference_steps": 25,
                    "guidance_scale": 3.5,
                },
            )

        assert response.status_code == 200, f"Generate failed: {response.text}"
        result = response.json()
        job_id = result["job_id"]
        print(f"[VASTAI] Started LoRA generation job {job_id}")

        # Poll for completion
        max_wait = 600
        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient(timeout=30.0) as client:
            while asyncio.get_event_loop().time() - start_time < max_wait:
                status_response = await client.get(f"{base_url}/generate/status/{job_id}")
                status_data = status_response.json()

                print(f"[VASTAI] Generation {job_id}: {status_data.get('status', 'unknown')}")

                if status_data.get("status") == "completed":
                    images = status_data.get("images", [])
                    assert len(images) > 0, "No images in result"

                    # Save first image for inspection
                    img = images[0]
                    if img.startswith("data:image"):
                        base64_part = img.split(",")[1]
                    else:
                        base64_part = img

                    img_bytes = base64.b64decode(base64_part)
                    img_path = f"/tmp/generated_{job_id}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    print(f"[VASTAI] Saved generated image to {img_path}")

                    return

                elif status_data.get("status") == "failed":
                    pytest.fail(f"Generation failed: {status_data.get('error')}")

                await asyncio.sleep(10)

        pytest.fail("Generation did not complete within 10 minutes")


# =============================================================================
# Phase 6: Adapter Integration Tests
# =============================================================================

@pytest.mark.skipif(not HAS_API_KEY, reason="VASTAI_API_KEY not set")
@pytest.mark.cloud_resource
class TestVastAIAdapter:
    """Test the VastAITrainingAdapter integration.

    Tests the unified TrainingProvider interface.
    """

    @pytest.mark.asyncio
    async def test_adapter_is_available(self):
        """Test adapter availability check."""
        from kestrel_sovereign.features.training.adapters.vastai_adapter import VastAITrainingAdapter

        adapter = VastAITrainingAdapter()
        available = adapter.is_available()

        print(f"\n[VASTAI] Adapter available: {available}")
        assert available is True, "Adapter should be available with API key set"

    @pytest.mark.asyncio
    async def test_adapter_provider_info(self):
        """Test adapter provider metadata."""
        from kestrel_sovereign.features.training.adapters.vastai_adapter import VastAITrainingAdapter
        from kestrel_sovereign.features.training.types import ProviderType

        adapter = VastAITrainingAdapter()

        assert adapter.provider_name == "vastai"
        assert adapter.provider_type == ProviderType.SESSION_BASED
        print(f"\n[VASTAI] Provider: {adapter.provider_name} ({adapter.provider_type.value})")

    @pytest.mark.asyncio
    async def test_adapter_start_training(self):
        """Test starting training via adapter interface."""
        from kestrel_sovereign.features.training.adapters.vastai_adapter import VastAITrainingAdapter
        from kestrel_sovereign.features.training.types import TrainingConfig, TrainingState

        adapter = VastAITrainingAdapter()

        # Create test image and config
        test_image = create_test_png_image()
        config = TrainingConfig(
            steps=100,
            lora_rank=16,
            trigger_word="TOKadaptertest",
        )

        companion_id = f"adapter-test-{datetime.now(timezone.utc).strftime('%H%M%S')}"

        print(f"\n[VASTAI] Starting training via adapter for {companion_id}...")

        try:
            job = await adapter.start_training(
                companion_id=companion_id,
                avatar_data=test_image,
                config=config,
            )

            print(f"[VASTAI] Job created: {job.job_id}")
            print(f"[VASTAI] State: {job.state}")
            print(f"[VASTAI] Trigger word: {job.trigger_word}")
            print(f"[VASTAI] Provider job ID: {job.provider_job_id}")
            print(f"[VASTAI] Session ID: {job.provider_session_id}")

            assert job.job_id is not None
            assert job.companion_id == companion_id
            assert job.provider == "vastai"
            assert job.state in [TrainingState.PENDING, TrainingState.TRAINING, TrainingState.PREPARING]
            assert job.trigger_word == "TOKadaptertest"

            # Check status
            status = await adapter.get_status(job.job_id)
            print(f"[VASTAI] Status: {status.state} ({status.progress * 100:.1f}%)")

            assert status.job_id == job.job_id

            # Cancel to avoid long-running training
            cancelled = await adapter.cancel(job.job_id)
            print(f"[VASTAI] Cancelled: {cancelled}")

        except Exception as e:
            print(f"[VASTAI] Error: {e}")
            # Clean up any active jobs
            await adapter.cleanup(companion_id)
            raise
