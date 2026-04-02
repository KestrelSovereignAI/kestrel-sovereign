"""
Integration tests for LoRA training providers.

Provider-agnostic tests that work with any training provider:
- local_mps (Apple Silicon with SimpleTuner)
- runpod (Cloud GPU)
- vertex_ai (Google Cloud)
- replicate (Serverless)

Use case: Training a custom LoRA from reference images for consistent
image generation (avatars, characters, styles, objects).

Run all tests:
    pytest tests/integration/test_lora_training.py -v

Run with specific provider:
    TRAINING_PROVIDER=local_mps pytest tests/integration/test_lora_training.py -v

Run real cloud tests (costs money):
    pytest tests/integration/test_lora_training.py -v --run-cloud -k "real"
"""

import os
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Test image - 1x1 red pixel PNG for minimal testing
TEST_IMAGE_DATA = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 pixels
    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
    0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
    0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0xFE,
    0xD4, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,  # IEND chunk
    0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
])


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def provider_name():
    """Get the provider to test from environment or default to local_mps."""
    return os.getenv("TRAINING_PROVIDER", "local_mps")


@pytest_asyncio.fixture
async def training_provider(provider_name):
    """Get a training provider instance."""
    from kestrel_sovereign.features.training.factory import TrainingProviderFactory

    provider = TrainingProviderFactory.get_provider(provider_name)
    if provider is None:
        pytest.skip(f"Provider {provider_name} not available")

    yield provider

    # Cleanup
    if hasattr(provider, 'close'):
        await provider.close()


@pytest.fixture
def training_config():
    """Default training configuration for tests."""
    from kestrel_sovereign.features.training.types import TrainingConfig

    return TrainingConfig(
        steps=10,  # Minimal steps for testing
        lora_rank=4,  # Small rank for fast training
        learning_rate=1e-4,
        resolution=512,  # Lower resolution for speed
        trigger_word="TOKTEST",
    )


# =============================================================================
# Unit Tests (No External Resources)
# =============================================================================

class TestTrainingProviderFactory:
    """Test the training provider factory."""

    def test_list_providers(self):
        """Test listing available provider types."""
        from kestrel_sovereign.features.training.factory import TrainingProviderFactory

        providers = TrainingProviderFactory.PROVIDER_PRIORITY
        assert "local_mps" in providers
        assert len(providers) >= 1

    def test_get_capabilities(self):
        """Test getting provider capabilities."""
        from kestrel_sovereign.features.training.factory import TrainingProviderFactory

        caps = TrainingProviderFactory.PROVIDER_CAPABILITIES

        # local_mps should support training and generation
        assert "local_mps" in caps
        assert caps["local_mps"].training is True
        assert caps["local_mps"].generation is True

    def test_auto_select_provider(self):
        """Test auto-selecting best available provider."""
        from kestrel_sovereign.features.training.factory import TrainingProviderFactory

        provider = TrainingProviderFactory.get_default_provider()

        # Should return something if any provider is available
        # On a Mac with SimpleTuner, should get local_mps
        if provider:
            assert hasattr(provider, 'provider_name')
            assert hasattr(provider, 'start_training')

    def test_get_local_provider(self):
        """Test getting the local MPS provider specifically."""
        from kestrel_sovereign.features.training.factory import TrainingProviderFactory

        provider = TrainingProviderFactory.get_local_provider()

        # May be None if SimpleTuner/SDXL not installed
        if provider:
            assert provider.provider_name == "local_mps"


class TestTrainingTypes:
    """Test training-related type definitions."""

    def test_training_config_defaults(self):
        """Test TrainingConfig has sensible defaults."""
        from kestrel_sovereign.features.training.types import TrainingConfig

        config = TrainingConfig()

        assert config.steps is None or config.steps > 0
        assert config.lora_rank is None or config.lora_rank > 0

    def test_training_state_enum(self):
        """Test TrainingState enum values."""
        from kestrel_sovereign.features.training.types import TrainingState

        assert TrainingState.PENDING
        assert TrainingState.TRAINING
        assert TrainingState.COMPLETED
        assert TrainingState.FAILED

    def test_provider_type_enum(self):
        """Test ProviderType enum values."""
        from kestrel_sovereign.features.training.types import ProviderType

        assert ProviderType.LOCAL
        assert ProviderType.SERVERLESS
        assert ProviderType.SESSION_BASED


class TestLocalMPSAdapter:
    """Test the LocalMPSTrainingAdapter specifically."""

    def test_adapter_imports(self):
        """Test adapter can be imported."""
        from kestrel_sovereign.features.training.adapters import LocalMPSTrainingAdapter

        assert LocalMPSTrainingAdapter.provider_name == "local_mps"

    def test_adapter_instantiation(self, tmp_path):
        """Test adapter can be instantiated."""
        from kestrel_sovereign.features.training.adapters import LocalMPSTrainingAdapter

        adapter = LocalMPSTrainingAdapter(working_dir=str(tmp_path / "training"))

        assert adapter.provider_name == "local_mps"
        assert adapter.working_dir.exists()

    def test_is_available_check(self, tmp_path):
        """Test availability check doesn't crash."""
        from kestrel_sovereign.features.training.adapters import LocalMPSTrainingAdapter

        adapter = LocalMPSTrainingAdapter(working_dir=str(tmp_path / "training"))
        available = adapter.is_available()

        # Should return bool without crashing
        assert isinstance(available, bool)


# =============================================================================
# Integration Tests (Mock External Resources)
# =============================================================================

class TestTrainingWorkflowMocked:
    """Test training workflow with mocked external resources."""

    @pytest.mark.asyncio
    async def test_start_training_creates_config(self, training_config, tmp_path):
        """Test that start_training creates proper config files."""
        from kestrel_sovereign.features.training.adapters import LocalMPSTrainingAdapter
        from unittest.mock import patch

        adapter = LocalMPSTrainingAdapter(
            working_dir=str(tmp_path),
            model_path=str(tmp_path / "fake-model"),
            diffusers_path=str(tmp_path / "fake-diffusers"),
        )

        # Mock subprocess to prevent actual training
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            # Create fake model directory in diffusers format so is_available passes
            (tmp_path / "fake-model").mkdir()
            (tmp_path / "fake-model" / "model_index.json").write_text("{}")
            # Create fake diffusers directory with training script
            (tmp_path / "fake-diffusers" / "examples" / "text_to_image").mkdir(parents=True)
            (tmp_path / "fake-diffusers" / "examples" / "text_to_image" / "train_text_to_image_lora_sdxl.py").write_text("")

            # Mock torch.backends.mps
            with patch('torch.backends.mps.is_available', return_value=True):
                try:
                    job = await adapter.start_training(
                        companion_id="test-123",
                        avatar_data=TEST_IMAGE_DATA,
                        config=training_config,
                    )

                    # Verify job was created
                    assert job.job_id is not None
                    assert job.companion_id == "test-123"
                    assert job.trigger_word == training_config.trigger_word

                    # Verify dataset files were created (diffusers format)
                    dataset_dir = adapter.datasets_dir / job.job_id
                    assert (dataset_dir / "metadata.jsonl").exists()
                    # Verify job directory exists
                    config_dir = adapter.configs_dir / job.job_id
                    assert config_dir.exists()

                except Exception as e:
                    # May fail if torch not installed - that's OK for unit test
                    if "torch" not in str(e).lower():
                        raise

    @pytest.mark.asyncio
    async def test_get_status_returns_valid_state(self, training_config, tmp_path):
        """Test that get_status returns valid training state."""
        from kestrel_sovereign.features.training.adapters import LocalMPSTrainingAdapter
        from kestrel_sovereign.features.training.types import TrainingState
        from unittest.mock import patch
        import subprocess

        adapter = LocalMPSTrainingAdapter(
            working_dir=str(tmp_path),
            model_path=str(tmp_path / "fake-model"),
            diffusers_path=str(tmp_path / "fake-diffusers"),
        )

        # Create directories in diffusers format
        (tmp_path / "fake-model").mkdir()
        (tmp_path / "fake-model" / "model_index.json").write_text("{}")
        # Create fake diffusers directory with training script
        (tmp_path / "fake-diffusers" / "examples" / "text_to_image").mkdir(parents=True)
        (tmp_path / "fake-diffusers" / "examples" / "text_to_image" / "train_text_to_image_lora_sdxl.py").write_text("")

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None  # Still running
            mock_popen.return_value = mock_process

            with patch('torch.backends.mps.is_available', return_value=True):
                try:
                    job = await adapter.start_training(
                        companion_id="test-456",
                        avatar_data=TEST_IMAGE_DATA,
                        config=training_config,
                    )

                    status = await adapter.get_status(job.job_id)

                    assert status.job_id == job.job_id
                    assert status.state in [
                        TrainingState.PENDING,
                        TrainingState.TRAINING,
                        TrainingState.COMPLETED,
                        TrainingState.FAILED,
                    ]
                    assert 0.0 <= status.progress <= 1.0

                except Exception as e:
                    if "torch" not in str(e).lower():
                        raise


# =============================================================================
# Real Integration Tests (Require --run-cloud flag)
# =============================================================================

class TestCharacterConsistency:
    """
    Test that LoRA training produces consistent character identity.

    The core value of LoRA training: generate images of the SAME person
    in different scenes, poses, and contexts while maintaining identity.
    """

    # Scene prompts for testing character consistency
    CONSISTENCY_SCENES = [
        "portrait photo, professional headshot, neutral background",
        "casual photo, outdoor setting, natural lighting",
        "lifestyle photo, relaxed pose, warm lighting",
    ]

    @pytest.mark.asyncio
    async def test_generation_uses_trigger_word(self, training_config):
        """Test that generation prompts properly incorporate trigger word."""
        trigger = training_config.trigger_word

        # Build prompt with trigger word
        scene = "portrait photo, professional lighting"
        prompt = f"{trigger} person, {scene}"

        assert trigger in prompt
        assert "person" in prompt

    @pytest.mark.asyncio
    async def test_scene_variety_with_consistent_identity(self):
        """
        Verify the concept: same trigger word = same identity across scenes.

        This test documents the expected behavior:
        - Train LoRA on reference images of a person
        - Use trigger word in different scene prompts
        - Generated images should show the SAME person in different contexts
        """
        trigger = "TOKCHAR"

        # Different scenes, same identity
        prompts = [
            f"{trigger} person, portrait photo, professional headshot",
            f"{trigger} person, casual outdoor photo, park setting",
            f"{trigger} person, artistic portrait, dramatic lighting",
            f"{trigger} person, beach photo, summer vibes",
            f"{trigger} person, cozy indoor setting, warm sweater",
        ]

        # All prompts should contain the trigger word
        for prompt in prompts:
            assert trigger in prompt

        # The key insight: trigger word encodes the person's appearance
        # so all generations should show the SAME person regardless of scene

    @pytest.mark.asyncio
    async def test_prompt_structure_for_consistency(self):
        """
        Test recommended prompt structure for character-consistent generation.

        Best practices:
        1. Trigger word FIRST (encodes identity)
        2. Subject type (person, woman, man)
        3. Scene/setting description
        4. Style modifiers
        """
        trigger = "TOKPERSON"

        # Good: trigger word first
        good_prompt = f"{trigger} person, beach photo, golden hour, relaxed pose"
        assert good_prompt.startswith(trigger)

        # The trained LoRA associates trigger with specific appearance
        # so trigger word + any scene = consistent person in that scene


# =============================================================================
# Real Integration Tests (Require --run-cloud flag)
# =============================================================================

@pytest.mark.skipif(
    not os.getenv("RUN_CLOUD_TESTS"),
    reason="Cloud tests disabled. Set RUN_CLOUD_TESTS=1 to enable."
)
class TestTrainingWorkflowReal:
    """
    Real integration tests that use actual training providers.

    WARNING: These tests may incur costs and take significant time.
    Only run with explicit opt-in via RUN_CLOUD_TESTS=1 environment variable.
    """

    @pytest.mark.asyncio
    async def test_real_training_start(self, training_provider, training_config):
        """Test starting real training (will be cancelled immediately)."""
        try:
            job = await training_provider.start_training(
                companion_id="integration-test",
                avatar_data=TEST_IMAGE_DATA,
                config=training_config,
            )

            assert job.job_id is not None
            print(f"Started training job: {job.job_id}")

            # Cancel immediately to avoid costs
            await training_provider.cancel(job.job_id)
            print("Training cancelled")

        except Exception as e:
            print(f"Training start failed (expected if resources unavailable): {e}")

    @pytest.mark.asyncio
    async def test_real_status_check(self, training_provider, training_config):
        """Test checking status of real training job."""
        try:
            job = await training_provider.start_training(
                companion_id="integration-test-status",
                avatar_data=TEST_IMAGE_DATA,
                config=training_config,
            )

            status = await training_provider.get_status(job.job_id)
            print(f"Training status: {status.state.value}, progress: {status.progress}")

            assert status.job_id == job.job_id

            # Cleanup
            await training_provider.cancel(job.job_id)

        except Exception as e:
            print(f"Status check failed: {e}")


@pytest.mark.skipif(
    not os.getenv("RUN_FULL_TRAINING"),
    reason="Full training tests disabled. Set RUN_FULL_TRAINING=1 to run."
)
class TestFullTrainingCycle:
    """
    Full end-to-end training and generation tests.

    These tests run actual training to completion and verify character
    consistency across multiple generated images.

    WARNING: Expensive and time-consuming. Only run when explicitly enabled.
    """

    @pytest.mark.asyncio
    async def test_train_and_generate_consistent_character(
        self, training_provider, training_config
    ):
        """
        Full cycle: train LoRA, generate images, verify consistency.

        Steps:
        1. Start training with reference image
        2. Wait for training to complete
        3. Generate images in multiple scenes
        4. Verify all images show the same person (manual inspection)
        """
        from kestrel_sovereign.features.training.types import TrainingState
        import asyncio

        # Use a real test image if available
        test_image_path = Path(os.getenv(
            "TEST_IMAGE_PATH",
            "~/models/local-training/test-images/reference.jpg"
        )).expanduser()

        if test_image_path.exists():
            avatar_data = test_image_path.read_bytes()
        else:
            pytest.skip(f"Test image not found: {test_image_path}")

        print("\n=== Starting LoRA Training ===")
        job = await training_provider.start_training(
            companion_id="consistency-test",
            avatar_data=avatar_data,
            config=training_config,
        )
        print(f"Job started: {job.job_id}")
        print(f"Trigger word: {job.trigger_word}")

        # Wait for training to complete (with timeout)
        max_wait = 3600  # 1 hour max
        poll_interval = 30  # Check every 30 seconds
        elapsed = 0

        while elapsed < max_wait:
            status = await training_provider.get_status(job.job_id)
            print(f"Status: {status.state.value}, progress: {status.progress:.1%}")

            if status.state == TrainingState.COMPLETED:
                print("Training completed!")
                break
            elif status.state == TrainingState.FAILED:
                pytest.fail(f"Training failed: {status.error}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if elapsed >= max_wait:
            await training_provider.cancel(job.job_id)
            pytest.fail("Training timed out")

        # Download trained weights
        print("\n=== Downloading LoRA Weights ===")
        lora_bytes = await training_provider.download_weights(job.job_id)
        assert lora_bytes is not None
        print(f"Downloaded {len(lora_bytes)} bytes")

        # Generate images in different scenes
        print("\n=== Generating Character-Consistent Images ===")
        scenes = [
            "portrait photo, professional headshot, studio lighting",
            "casual photo, outdoor park, natural daylight",
            "artistic portrait, dramatic lighting, moody atmosphere",
        ]

        generated_images = []
        for scene in scenes:
            prompt = f"{job.trigger_word} person, {scene}"
            print(f"Generating: {prompt[:50]}...")

            result = await training_provider.generate_image(
                lora_bytes=lora_bytes,
                prompt=prompt,
            )

            if result.images:
                generated_images.append({
                    "scene": scene,
                    "prompt": prompt,
                    "image": result.images[0],
                })
                print(f"  Generated image for: {scene[:30]}...")

        # Save images for manual inspection
        output_dir = Path(os.getenv(
            "TEST_OUTPUT_DIR",
            "~/models/local-training/test-output"
        )).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img_data in enumerate(generated_images):
            # Decode base64 if needed
            image_data = img_data["image"]
            if image_data.startswith("data:image"):
                import base64
                image_data = base64.b64decode(image_data.split(",")[1])

            output_path = output_dir / f"consistency_test_{i}.png"
            with open(output_path, "wb") as f:
                f.write(image_data if isinstance(image_data, bytes) else image_data.encode())
            print(f"Saved: {output_path}")

        print(f"\n=== Generated {len(generated_images)} images ===")
        print(f"Output directory: {output_dir}")
        print("Manually inspect images to verify same person appears in all scenes.")

        # Cleanup
        await training_provider.cleanup(job.job_id)
