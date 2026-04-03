"""
Visual Identity Feature - Companion Image Generation

Single source of truth for all companion visual generation:
- Avatar generation (initial creation)
- Selfie generation (during chat)
- LoRA training for character consistency (sovereign selfies)
- Scene variations

Backend: ImageGenerationService
    - Replicate API (FLUX.1-schnell for initial avatar)
    - RunPod on-demand with trained LoRA (sovereign selfies)

Storage: Kestrel content-addressable storage
    - Avatar stored as part of agent identity (avatar_hash on agent node)
    - LoRA models stored in IPFS (travels with sovereignty exports)
    - Encrypted at rest

Lazy Training Flow:
    1. First !selfie request checks for lora_model_path
    2. If no LoRA exists, triggers training (~15-20 min)
    3. Subsequent selfies use trained LoRA for character consistency
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional

import httpx

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory
from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_DEFAULT,
    TRAINING_TIMEOUT_EXTENDED,
    TRAINING_POLL_INTERVAL,
)

logger = logging.getLogger(__name__)

# Import TrainingProviderFactory for unified provider access
try:
    from kestrel_sovereign.features.training import (
        TrainingProviderFactory,
        GenerationConfig,
        GenerationError,
    )
    TRAINING_FACTORY_AVAILABLE = True
except ImportError:
    TRAINING_FACTORY_AVAILABLE = False
    logger.warning("TrainingProviderFactory not available")

# ImageGenerationService requires platform integration
IMAGE_SERVICE_AVAILABLE = False


class VisualIdentityFeature(Feature):
    """
    Feature for companion visual identity (image generation).

    Used by KestrelAgent for generating images during conversations.
    Can also be used standalone (agent=None) for platform integration.
    """

    def __init__(self, agent=None):
        """
        Initialize the feature.

        Args:
            agent: KestrelAgent instance (optional for standalone usage)
        """
        if agent is not None:
            super().__init__(agent)
        else:
            # Standalone mode for external integration
            self.agent = None
            self.name = self.__class__.__name__

    @property
    def tool_description(self) -> str:
        return (
            "Generate visual content - create avatar portraits from descriptions, "
            "generate selfies and photos in various scenes, maintain visual consistency"
        )

    async def initialize(self):
        """Initialize the image generation service and LoRA training support"""
        # Image generation service (Replicate + RunPod)
        if IMAGE_SERVICE_AVAILABLE:
            self.service = ImageGenerationService()
            self.enabled = self.service.enabled
            if not self.enabled:
                logger.warning("VisualIdentityFeature: Replicate not available")
        else:
            self.service = None
            self.enabled = False
            logger.info("VisualIdentityFeature: ImageGenerationService unavailable, checking training providers")

        # LoRA training via unified TrainingProviderFactory
        self._training_provider = None  # Lazy-loaded via TrainingProviderFactory
        self._lora_initialized = False
        self.db_pool = None  # Direct db_pool reference for companion lookups

        # Enable feature if a training provider with generation capability exists
        # (e.g., local_mps can generate selfies without Replicate)
        if not self.enabled and TRAINING_FACTORY_AVAILABLE:
            gen_provider = TrainingProviderFactory.get_generation_provider()
            if gen_provider:
                self.enabled = True
                logger.info(f"VisualIdentityFeature enabled via generation provider: {gen_provider.provider_name}")

    def _ensure_lora_services(self) -> bool:
        """
        Lazy-initialize LoRA training services via TrainingProviderFactory.

        Uses unified factory for provider selection with priority:
        1. RunPod (uncensored FLUX.2, supports training + generation)
        2. Vertex AI (serverless FLUX.2, training only)
        3. Replicate (serverless FLUX.1, censored)
        4. GCP Compute (VM-based)
        5. Vast.ai (marketplace)

        Returns True if LoRA training is available.
        """
        if self._lora_initialized:
            return self._training_provider is not None

        self._lora_initialized = True

        # Use unified TrainingProviderFactory for provider selection
        if TRAINING_FACTORY_AVAILABLE:
            self._training_provider = TrainingProviderFactory.get_default_provider()
            if self._training_provider:
                logger.info(f"✅ Training provider initialized: {self._training_provider.provider_name}")
                return True
            else:
                available = TrainingProviderFactory.list_available_providers()
                logger.warning(f"No training providers available. Checked: {available or 'none'}")
        else:
            logger.warning("TrainingProviderFactory not available")

        logger.info("LoRA training disabled (no providers configured)")
        return False

    def _get_training_provider(self, provider_name: Optional[str] = None):
        """
        Get the unified training provider via TrainingProviderFactory.

        This is the new recommended approach for getting a provider.
        The factory handles availability checks and caching.

        Args:
            provider_name: Optional specific provider ("runpod", "vertex_ai", "vastai").
                          If None, uses default provider priority.

        Returns:
            TrainingProvider or None if no providers available
        """
        if not TRAINING_FACTORY_AVAILABLE:
            return None

        # If specific provider requested, get it directly (bypass cache)
        if provider_name:
            provider = TrainingProviderFactory.get_provider(provider_name)
            if provider:
                logger.info(f"✅ Using requested provider: {provider.provider_name}")
                return provider
            else:
                logger.warning(f"⚠️ Requested provider '{provider_name}' not available, falling back to default")

        # Default behavior - use cached provider
        if self._training_provider is None:
            self._training_provider = TrainingProviderFactory.get_default_provider()
            if self._training_provider:
                logger.info(f"✅ Using training provider: {self._training_provider.provider_name}")

        return self._training_provider

    async def _generate_with_provider(
        self,
        prompt: str,
        lora_path: str,
        trigger_word: str,
        companion_id: str,
        lora_ipfs_cid: Optional[str] = None,
        provider_name: Optional[str] = None,
        flux_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate image using the unified TrainingProviderFactory.

        This is the new approach that uses the provider's generate_image() method
        (only supported by RunPod adapter currently).

        Args:
            prompt: Generation prompt
            lora_path: Path to LoRA model (GCS path or local)
            trigger_word: LoRA trigger word
            companion_id: Companion ID for logging
            lora_ipfs_cid: Optional IPFS CID for LoRA (preferred over lora_path)
            provider_name: Optional specific provider to use ("runpod", "vertex_ai", etc.)
            flux_version: Optional FLUX version ("flux1" or "flux2") for container selection

        Returns:
            {"success": True, "images": [data_url, ...], "backend": "provider_name"}

        Raises:
            RuntimeError: If generation fails
        """
        training_provider = self._get_training_provider(provider_name)
        if not training_provider:
            raise RuntimeError("No training provider available via factory")

        # Check if provider has generate_image method (RunPod and Vertex AI both support this)
        if not hasattr(training_provider, 'generate_image'):
            raise RuntimeError(
                f"Provider {training_provider.provider_name} doesn't support image generation. "
                f"Use RunPod or Vertex AI for selfie generation."
            )

        try:
            config = GenerationConfig(
                prompt=prompt,
                lora_path=lora_path,
                trigger_word=trigger_word,
            )

            # Log which LoRA source we're using
            if lora_ipfs_cid:
                logger.info(f"🎨 Generating via {training_provider.provider_name} with IPFS LoRA: {lora_ipfs_cid[:16]}...")
            else:
                logger.info(f"🎨 Generating via {training_provider.provider_name} with GCS LoRA: {lora_path[:50]}...")

            # Pass IPFS CID to provider if available (preferred over GCS path)
            # Pass flux_version to select correct container (flux1 = uncensored, flux2 = standard)
            result = await training_provider.generate_image(config, lora_ipfs_cid=lora_ipfs_cid, flux_version=flux_version)

            if result.state.value != "completed":
                raise RuntimeError(f"Generation failed: {result.error or 'Unknown error'}")

            if not result.images:
                raise RuntimeError("Generation completed but no images returned")

            logger.info(f"✅ Generated {len(result.images)} images via {training_provider.provider_name}")

            return {
                "success": True,
                "images": result.images,
                "backend": training_provider.provider_name,
                "elapsed_seconds": result.elapsed_seconds,
            }

        except GenerationError as e:
            logger.error(f"Provider generation failed: {e}")
            raise RuntimeError(str(e))

    def set_db_pool(self, db_pool):
        """Set the database pool for companion lookups."""
        self.db_pool = db_pool

    def set_runpod_manager(self, runpod_manager):
        """Set RunPod manager (for sharing with external server)."""
        # RunPod manager is now managed by TrainingProviderFactory
        # This method kept for backward compatibility
        if self.service:
            self.service.runpod_manager = runpod_manager

    def _get_subagent_prompt(self) -> str:
        """Get the system prompt for visual identity subagent."""
        return """You are the Visual Identity subagent within Kestrel, specializing in image generation.

Your capabilities: Generate selfies, portraits, and avatars for companions.

Available tools:
- generate_selfie: Generate a selfie in various scenes (casual, portrait, glamour, flirty, cozy, adventure, mysterious)
- generate_avatar: Generate avatar portraits from descriptions

CRITICAL INSTRUCTIONS:
1. When you successfully generate an image, ALWAYS include the image URL in your response as a markdown image
2. Format images like this: ![Selfie](https://the-image-url.com/image.png)
3. Add a brief, friendly message about the image
4. If generation fails, explain why and suggest alternatives

Example response when image_url is returned:
"Here's your casual selfie! 📸

![Selfie](https://replicate.delivery/xxx.png)

Looking good! Want another one in a different style?"
"""

    # Scene-specific prompt enhancements (shared across methods)
    # Each scene should include: setting, clothing/attire, pose, lighting
    SCENE_PROMPTS = {
        "portrait": "professional headshot, studio lighting, neutral background, business attire",
        "casual": "casual selfie at home, comfortable clothes, natural lighting, relaxed smile",
        "glamour": "glamorous evening setting, elegant black dress, sophisticated pose, studio lighting",
        "flirty": "playful smile, flirtatious expression, slight head tilt, soft lighting",
        "cozy": "cozy home setting, comfortable sweater, warm atmosphere, soft natural light",
        "adventure": "outdoor hiking setting, athletic wear, dynamic pose, bright daylight",
        "mysterious": "dramatic shadows, dark elegant attire, enigmatic expression, moody lighting",
        "romantic": "soft romantic candlelit setting, elegant dress, intimate atmosphere, warm colors",
        "playful": "fun playful expression, colorful casual outfit, bright colors, dynamic pose",
        "dreamy": "dreamy soft focus, flowing white dress, ethereal lighting, pastel colors",
        "confident": "confident powerful pose, professional attire, strong lighting, bold composition",
        # Beach and swimwear scenes
        "beach": "at the beach, bikini swimsuit, golden hour sunset lighting, ocean waves in background, beautiful smile, selfie angle",
        "swimsuit": "poolside setting, stylish bikini, bright sunny day, relaxed pose, tropical vibes",
        "tropical": "tropical beach paradise, colorful bikini, palm trees, crystal clear water, vacation selfie",
        "pool": "luxury pool setting, designer swimwear, sunglasses, lounge chair, summer vibes",
        # Additional lifestyle scenes
        "fitness": "gym or yoga studio, athletic sports bra and leggings, energetic pose, natural lighting",
        "nightout": "nightclub or bar setting, sexy cocktail dress, glamorous makeup, neon lights",
        "lingerie": "elegant bedroom setting, tasteful lingerie, soft boudoir lighting, confident pose",
        "summer": "sunny outdoor cafe, sundress, bright daylight, happy relaxed expression",
        # Professional/occupational scenes
        "nurse": "healthcare setting, nurse scrubs with stethoscope, hospital or clinic background, professional caring expression, soft clinical lighting",
        # Adult scenes (for sovereign companions - requires uncensored model variant)
        # Note: Base FLUX models have content filtering. For explicit content,
        # use fine-tuned uncensored variants (e.g., FLUX.1-dev uncensored)
        "topless": "artistic portrait, bare breasts visible, tasteful nude photography, studio lighting, sensual pose",
        "nude": "full nude portrait, artistic nude photography, studio setting, tasteful pose, natural lighting",
    }

    @tool(
        name="generate_selfie",
        description="Generate a selfie or portrait of the companion character.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!selfie"
    )
    async def generate_selfie(
        self,
        scene: str = "casual",
        reference_image: Optional[str] = None,
        companion_id: Optional[str] = None,
        lora_model_path: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        style: str = "photorealistic",
        allow_training: bool = True,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a selfie of the companion. REQUIRES LoRA - no censored fallback.

        If companion has a trained LoRA model, uses Vast.ai/RunPod with FLUX.1-dev.
        If no LoRA exists and allow_training=True, triggers lazy training (~15-20 min first time).
        If no LoRA exists and allow_training=False, FAILS - we don't fall back to censored models.

        Args:
            scene: Style of photo (casual, portrait, glamour, flirty, cozy, adventure, mysterious, romantic, playful, dreamy, confident)
            reference_image: Ignored (kept for API compatibility)
            companion_id: Companion UUID for LoRA lookup (required for selfies)
            lora_model_path: Direct path to LoRA model (optional, overrides lookup)
            style: Art style (photorealistic, anime, artistic)
            allow_training: If True and no LoRA, train one. If False and no LoRA, fail.
            provider: Force specific provider (runpod, vertex_ai, vastai). None = auto-select.

        Returns:
            {"success": bool, "image_url": str, "scene": str, "used_lora": bool, "trained_this_request": bool, "error": str}
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Image generation not available (no providers configured)"
            }

        # AUTO-FILL companion_id from agent's companion_context if not provided
        # This enables "send me a selfie" to work without the user providing IDs
        if not companion_id and self.agent and hasattr(self.agent, 'companion_context'):
            companion_context = getattr(self.agent, 'companion_context', {})
            companion_id = companion_context.get('companion_id')
            if companion_id:
                logger.info(f"Auto-filled companion_id from agent context: {companion_id}")

        scene = scene.lower()
        scene_description = self.SCENE_PROMPTS.get(scene, self.SCENE_PROMPTS["casual"])

        # Look up companion appearance and trigger word if we have a companion_id and db_pool
        companion_appearance = ""
        companion_trigger_word = None  # Will be set from DB if available
        if companion_id and self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT avatar_config FROM companions WHERE id = $1",
                        companion_id
                    )
                    if row and row["avatar_config"]:
                        import json
                        config = json.loads(row["avatar_config"]) if isinstance(row["avatar_config"], str) else row["avatar_config"]
                        # Use appearance, description, or prompt from avatar_config
                        companion_appearance = config.get("appearance", "") or config.get("description", "") or config.get("prompt", "")
                        # Get trigger word from avatar_config (set during training)
                        companion_trigger_word = config.get("trigger_word")
                        if companion_appearance:
                            logger.info(f"Using companion appearance: {companion_appearance[:50]}...")
                        if companion_trigger_word:
                            logger.info(f"Using trigger word from DB: {companion_trigger_word}")
            except Exception as e:
                logger.warning(f"Failed to lookup companion appearance: {e}")

        # Build enhanced prompt - trigger word will be prepended later when we have it
        # NOTE: The LoRA trigger word ALREADY encodes appearance (face, hair, body, clothing from training).
        # We should NOT append companion_appearance as it's redundant and can conflict with scene requests.
        # If custom_prompt provided, use it directly (for censorship testing, etc.)
        if custom_prompt:
            # Replace TRIGGER_WORD placeholder if present, otherwise prepend it
            if "TRIGGER_WORD" in custom_prompt:
                base_prompt = custom_prompt
            else:
                base_prompt = f"TRIGGER_WORD, {custom_prompt}"
            logger.info(f"Using custom prompt: {custom_prompt[:80]}...")
        else:
            # Use ONLY scene description - trigger word already has appearance baked in from LoRA training
            base_prompt = f"A photo of TRIGGER_WORD, {scene_description}. High quality, photorealistic, 8k."
            logger.info(f"Using scene '{scene}' with trigger word only (no appearance override)")

            if style == "anime":
                base_prompt = f"anime style illustration, {base_prompt}"
            elif style == "artistic":
                base_prompt = f"artistic portrait painting style, {base_prompt}"

        trained_this_request = False
        used_lora = False

        # Track IPFS CID for generation (preferred over GCS path)
        lora_ipfs_cid = None
        # Track FLUX version for container selection (flux1 = uncensored, flux2 = standard)
        flux_version = None

        try:
            # =========================================================
            # SOVEREIGN PATH: LoRA is REQUIRED - no censored fallback
            # =========================================================
            # Check if we have a provider via factory OR legacy services
            has_provider = self._get_training_provider() is not None
            has_legacy_services = self._ensure_lora_services()

            if lora_model_path or (companion_id and (has_provider or has_legacy_services)):
                # If no direct path provided, look up or train
                if not lora_model_path and companion_id:
                    # First check for existing LoRA (get full info including IPFS CID)
                    lora_info = await self._lookup_lora_info(companion_id)
                    if lora_info and lora_info.get("lora_model_path"):
                        lora_model_path = lora_info["lora_model_path"]
                        lora_ipfs_cid = lora_info.get("lora_ipfs_cid")  # May be None
                        flux_version = lora_info.get("flux_version")  # "flux1" or "flux2"
                        # Use trigger word from lookup if available and not already set
                        if not companion_trigger_word and lora_info.get("trigger_word"):
                            companion_trigger_word = lora_info["trigger_word"]
                        if lora_ipfs_cid:
                            logger.info(f"Found IPFS CID for LoRA: {lora_ipfs_cid[:16]}...")
                        if flux_version:
                            logger.info(f"Using FLUX version from DB: {flux_version}")
                    elif allow_training:
                        # No existing LoRA - train one (this takes 15-20 min)
                        lora_model_path = await self._get_or_train_lora(companion_id)
                        if lora_model_path and not lora_model_path.startswith("existing:"):
                            trained_this_request = True
                        elif lora_model_path and lora_model_path.startswith("existing:"):
                            lora_model_path = lora_model_path.replace("existing:", "")
                    else:
                        # No LoRA and not allowed to train - FAIL LOUD
                        return {
                            "success": False,
                            "error": f"No LoRA model for companion {companion_id}. Train one first with /train-lora or set allow_training=true",
                            "needs_training": True,
                            "companion_id": companion_id
                        }

                # Generate with LoRA if we have a path
                if lora_model_path:
                    if lora_ipfs_cid:
                        logger.info(f"🎨 Generating sovereign selfie with IPFS LoRA: {lora_ipfs_cid[:16]}...")
                    else:
                        logger.info(f"🎨 Generating sovereign selfie with GCS LoRA: {lora_model_path[:50]}...")

                    # Use trigger word from DB if available (set during training)
                    # Fall back to generated trigger word only if DB doesn't have one
                    if companion_trigger_word:
                        trigger_word = companion_trigger_word
                        logger.info(f"Using trigger word from avatar_config: {trigger_word}")
                    else:
                        # Legacy fallback - generate trigger word from companion_id
                        trigger_word = "TOK"
                        if companion_id and len(companion_id) >= 8:
                            trigger_word = f"TOK{companion_id[:8].replace('-', '')}"
                        logger.warning(f"No trigger_word in DB, using generated: {trigger_word}")

                    # NEW: Try unified provider approach first (TrainingProviderFactory)
                    # This uses the RunPod adapter's generate_image() method with async polling
                    training_provider = self._get_training_provider(provider)
                    if training_provider and hasattr(training_provider, 'generate_image'):
                        try:
                            # Replace TRIGGER_WORD placeholder with actual trigger word
                            final_prompt = base_prompt.replace("TRIGGER_WORD", trigger_word)
                            logger.info(f"Final prompt: {final_prompt[:100]}...")

                            result = await self._generate_with_provider(
                                prompt=final_prompt,
                                lora_path=lora_model_path,
                                trigger_word=trigger_word,
                                companion_id=companion_id or "unknown",
                                lora_ipfs_cid=lora_ipfs_cid,  # Pass IPFS CID (preferred)
                                provider_name=provider,  # Pass explicit provider selection
                                flux_version=flux_version,  # "flux1" or "flux2" for container selection
                            )
                            if result.get("success") and result.get("images"):
                                return {
                                    "success": True,
                                    "image_url": result["images"][0],
                                    "scene": scene,
                                    "prompt": final_prompt,  # Include full prompt for gallery storage
                                    "used_lora": True,
                                    "trained_this_request": trained_this_request,
                                    "reference_used": False,
                                    "backend": result.get("backend", "provider"),
                                    "elapsed_seconds": result.get("elapsed_seconds"),
                                    "lora_source": "ipfs" if lora_ipfs_cid else "gcs",
                                }
                        except RuntimeError as e:
                            logger.error(f"Provider generation failed: {e}")
                            return {
                                "success": False,
                                "error": f"Generation failed: {e}",
                                "companion_id": companion_id
                            }

            # =========================================================
            # NO FALLBACK - LoRA is REQUIRED for uncensored generation
            # =========================================================
            # We do NOT use Replicate's schnell - it's censored.
            # FLUX.1-dev on our own infrastructure is the only path.
            logger.error(f"No LoRA available for companion {companion_id} - cannot generate uncensored selfie")
            return {
                "success": False,
                "error": "LoRA model required for selfie generation. Please train a LoRA first using /train-lora endpoint.",
                "needs_training": True,
                "companion_id": companion_id
            }

        except RuntimeError as e:
            # RuntimeError from generate_with_lora means RunPod unavailable but LoRA exists
            logger.error(f"LoRA generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Selfie generation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_or_train_lora(self, companion_id: str) -> str:
        """
        Get existing LoRA path or trigger lazy training via unified TrainingProviderFactory.

        This is the "sovereign selfie" lazy training path.

        Args:
            companion_id: Companion UUID

        Returns:
            LoRA model path (prefixed with "existing:" if already existed)

        Raises:
            RuntimeError: If no LoRA exists and training fails or is unavailable
        """
        # Check for existing LoRA first
        existing_path = await self._lookup_lora_path(companion_id)
        if existing_path:
            return f"existing:{existing_path}"

        # Need to train - use unified provider
        provider = self._get_training_provider()
        if not provider:
            raise RuntimeError(
                f"No LoRA training provider available. "
                f"Set RUNPOD_API_KEY, GCP_PROJECT_ID, REPLICATE_API_TOKEN, or VASTAI_API_KEY. "
                f"Companion {companion_id} cannot generate selfies without trained LoRA."
            )

        # Get avatar data from database
        if not self.db_pool:
            raise RuntimeError("Database pool not configured on VisualIdentityFeature")

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT avatar_data, image_url FROM companions WHERE id = $1",
                companion_id
            )

        if not row:
            raise RuntimeError(f"Companion {companion_id} not found")

        avatar_data = row.get("avatar_data")
        if not avatar_data:
            raise RuntimeError(
                f"Companion {companion_id} has no avatar data. "
                f"Generate and SET an avatar first using /generate-avatar and PATCH /avatar"
            )

        # Import TrainingConfig
        from kestrel_sovereign.features.training import TrainingConfig

        trigger_word = f"TOK{companion_id[:8]}"
        config = TrainingConfig(trigger_word=trigger_word)

        try:
            job = await provider.start_training(
                companion_id=companion_id,
                avatar_data=bytes(avatar_data),
                config=config,
            )
            logger.info(f"Started LoRA training via {provider.provider_name}: {job.job_id}")

            # Return job info - actual path will be stored when training completes
            return f"training:{job.job_id}"

        except Exception as e:
            raise RuntimeError(f"LoRA training failed via {provider.provider_name}: {e}")

    async def _lookup_lora_path(self, companion_id: str) -> Optional[str]:
        """Look up existing LoRA path from companion's avatar_config."""
        result = await self._lookup_lora_info(companion_id)
        return result.get("lora_model_path") if result else None

    async def _lookup_lora_info(self, companion_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up LoRA info from companion's avatar_config.

        Returns dict with:
            - lora_model_path: GCS path to LoRA (e.g., gs://bucket/path/pytorch_lora_weights.safetensors)
            - lora_ipfs_cid: IPFS CID for LoRA (e.g., QmXxx...) - preferred for generation
            - trigger_word: Trigger word for LoRA activation

        Returns None if companion not found or no LoRA configured.
        """
        if not self.db_pool:
            return None

        try:
            import json
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT avatar_config FROM companions WHERE id = $1",
                    companion_id
                )
                if not row:
                    return None

                raw_config = row["avatar_config"]
                if raw_config is None:
                    return None
                elif isinstance(raw_config, str):
                    try:
                        config = json.loads(raw_config)
                    except json.JSONDecodeError:
                        return None
                else:
                    config = raw_config

                # Return all LoRA-related info
                return {
                    "lora_model_path": config.get("lora_model_path"),
                    "lora_ipfs_cid": config.get("lora_ipfs_cid"),
                    "trigger_word": config.get("trigger_word"),
                    "flux_version": config.get("flux_version"),  # "flux1" or "flux2"
                }

        except Exception as e:
            logger.error(f"Failed to lookup LoRA info for {companion_id}: {e}")
            return None

    async def _train_lora_for_companion(self, companion_id: str) -> Optional[str]:
        """
        Train a LoRA model for a companion using the unified TrainingProviderFactory.

        Provider priority (configured in factory):
        1. RunPod (uncensored FLUX.2, supports training + generation)
        2. Vertex AI (serverless FLUX.2, training only)
        3. Replicate (serverless FLUX.1, censored)
        4. GCP Compute (VM-based)
        5. Vast.ai (marketplace)

        Args:
            companion_id: Companion UUID

        Returns:
            LoRA model path if successful, None if failed
        """
        import httpx

        if not self._training_provider:
            logger.warning(f"Cannot train LoRA for {companion_id}: no training provider available")
            return None

        if not self.db_pool:
            logger.warning(f"Cannot train LoRA for {companion_id}: no database pool configured")
            return None

        try:
            # Get companion's avatar data from database
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT image_url, avatar_data, name, user_id FROM companions WHERE id = $1",
                    companion_id
                )
                if not row:
                    logger.error(f"Companion {companion_id} not found for lazy training")
                    return None

                image_url = row["image_url"]
                avatar_data = row.get("avatar_data")  # Binary avatar data if stored
                companion_name = row["name"]

                if not image_url and not avatar_data:
                    logger.error(f"Companion {companion_id} has no avatar image for training")
                    return None

            # Get avatar bytes
            if avatar_data:
                avatar_bytes = avatar_data
            elif image_url:
                # Download avatar from URL
                try:
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
                        resp = await client.get(image_url)
                        if resp.status_code != 200:
                            logger.error(f"Failed to download avatar for {companion_id}: HTTP {resp.status_code}")
                            return None
                        avatar_bytes = resp.content
                except Exception as e:
                    logger.error(f"Failed to download avatar: {e}")
                    return None
            else:
                logger.error(f"No avatar data or URL for {companion_id}")
                return None

            logger.info(f"🎨 Starting LoRA training for {companion_id} ({companion_name}) via {self._training_provider.provider_name}")

            # Start training via unified provider
            from kestrel_sovereign.features.training.types import TrainingConfig, TrainingState
            config = TrainingConfig(trigger_word=f"TOK{companion_name[:8]}")
            job = await self._training_provider.start_training(
                companion_id=companion_id,
                avatar_data=avatar_bytes,
                config=config
            )

            logger.info(f"📊 Training job started: {job.job_id} via {job.provider}")

            # Poll for completion (training takes ~15-20 min)
            max_wait = TRAINING_TIMEOUT_EXTENDED  # 30 minutes max
            poll_interval = TRAINING_POLL_INTERVAL  # Check every 30 seconds
            elapsed = 0

            while elapsed < max_wait:
                status = await self._training_provider.get_status(job.job_id)

                if status.state == TrainingState.COMPLETED:
                    # Determine lora_path based on provider
                    lora_path = job.output_path or f"{job.provider}:{job.job_id}"
                    logger.info(f"✅ Training completed: {lora_path}")

                    # Update companion's avatar_config with LoRA info
                    try:
                        async with self.db_pool.acquire() as conn:
                            await conn.execute("""
                                UPDATE companions
                                SET avatar_config = COALESCE(avatar_config, '{}'::jsonb) || $1::jsonb
                                WHERE id = $2
                            """, json.dumps({
                                "lora_model_path": lora_path,
                                "lora_training_status": "completed",
                                "lora_trigger_word": job.trigger_word,
                                "lora_provider": job.provider,
                                "lora_job_id": job.job_id
                            }), companion_id)
                    except Exception as e:
                        logger.error(f"Failed to update avatar_config: {e}")

                    # Cleanup resources (important for session-based providers)
                    await self._training_provider.cleanup(job.job_id)
                    return lora_path

                if status.state == TrainingState.FAILED:
                    error = status.error or "Unknown error"
                    logger.error(f"Training failed: {error}")
                    await self._training_provider.cleanup(job.job_id)
                    return None

                if status.state == TrainingState.CANCELLED:
                    logger.warning(f"Training was cancelled")
                    await self._training_provider.cleanup(job.job_id)
                    return None

                logger.info(f"⏳ Training progress: {status.progress*100:.0f}% ({status.state.value})")
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            logger.error(f"Training timed out after {max_wait}s")
            await self._training_provider.cancel(job.job_id)
            await self._training_provider.cleanup(job.job_id)
            return None

        except Exception as e:
            logger.error(f"LoRA training failed for {companion_id}: {e}", exc_info=True)
            return None

    @tool(
        name="train_lora",
        description="Train a LoRA model for character-consistent selfie generation. If called without arguments, uses the current companion's ID and avatar.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!train-lora"
    )
    async def train_lora(
        self,
        companion_id: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explicitly trigger LoRA training for a companion.

        Usually called automatically via lazy training on first !selfie,
        but can be triggered manually.

        Args:
            companion_id: Companion UUID (auto-filled from context if not provided)
            image_url: Avatar image URL (auto-filled from context if not provided)

        Returns:
            {"success": bool, "job_id": str, "status": str, "error": str}
        """
        # AUTO-FILL from agent's companion_context if not provided
        # This enables "train my LoRA" to work without the user providing IDs
        if self.agent and hasattr(self.agent, 'companion_context'):
            companion_context = getattr(self.agent, 'companion_context', {})
            if not companion_id:
                companion_id = companion_context.get('companion_id')
                if companion_id:
                    logger.info(f"Auto-filled companion_id from agent context: {companion_id}")
            if not image_url:
                image_url = companion_context.get('image_url')
                if image_url:
                    logger.info(f"Auto-filled image_url from agent context: {image_url[:50]}...")

        if not companion_id:
            return {
                "success": False,
                "error": "No companion_id provided and couldn't determine from context. Please specify your companion ID."
            }

        if not self._ensure_lora_services():
            return {
                "success": False,
                "error": "LoRA training not available (RUNPOD_API_KEY not set)"
            }

        try:
            # Check for existing LoRA
            existing = await self._lookup_lora_path(companion_id)
            if existing:
                return {
                    "success": True,
                    "status": "already_trained",
                    "lora_path": existing
                }

            # Start training
            lora_path = await self._train_lora_for_companion(companion_id)

            if lora_path:
                return {
                    "success": True,
                    "status": "completed",
                    "lora_path": lora_path
                }
            else:
                return {
                    "success": False,
                    "error": "Training failed"
                }

        except Exception as e:
            logger.error(f"LoRA training error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @tool(
        name="generate_avatar",
        description="Generate an avatar portrait from a description and store it as part of agent identity.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!avatar"
    )
    async def generate_avatar(
        self,
        description: str,
        num_outputs: int = 2
    ) -> Dict[str, Any]:
        """
        Generate avatar options from a description and store in Kestrel storage.

        The primary avatar is stored as part of the agent's identity (like constitution_hash),
        ensuring it travels with sovereignty exports.

        Args:
            description: Physical description for the avatar
            num_outputs: Number of options to generate (1-4)

        Returns:
            {"success": bool, "image_urls": list, "stored_url": str, "error": str}
        """
        if not self.enabled or not self.service:
            return {
                "success": False,
                "error": "Image generation not available (missing REPLICATE_API_TOKEN)"
            }

        try:
            logger.info(f"Generating {num_outputs} avatar options: {description[:50]}...")

            image_urls = self.service.generate_character_portrait(
                prompt=description,
                num_outputs=min(num_outputs, 4)  # Cap at 4
            )

            if not image_urls:
                return {
                    "success": False,
                    "error": "Avatar generation returned no results"
                }

            logger.info(f"✅ Generated {len(image_urls)} avatar options")

            # Download and store the primary avatar in Kestrel storage
            stored_url = None
            stored_hash = None
            if self.agent and hasattr(self.agent, 'storage') and self.agent.storage:
                try:
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
                        response = await client.get(image_urls[0])
                        if response.status_code == 200:
                            image_data = response.content

                            # Store as part of agent identity
                            stored_hash = await self.agent.storage.files.store_avatar(
                                image_data=image_data,
                                agent_id=self.agent.agent_id,
                                avatar_type="primary",
                                source_url=image_urls[0]
                            )
                            stored_url = f"/api/files/{stored_hash}"
                            logger.info(f"✅ Avatar stored in Kestrel: {stored_hash[:16]}...")

                            # Store additional variants if generated
                            for i, url in enumerate(image_urls[1:], start=1):
                                try:
                                    resp = await client.get(url)
                                    if resp.status_code == 200:
                                        await self.agent.storage.files.store_avatar(
                                            image_data=resp.content,
                                            agent_id=self.agent.agent_id,
                                            avatar_type=f"variant_{i}",
                                            source_url=url
                                        )
                                except Exception as e:
                                    logger.warning(f"Failed to store variant {i}: {e}")

                except Exception as e:
                    logger.error(f"Failed to store avatar in Kestrel: {e}")
                    # Continue - return Replicate URLs as fallback

            return {
                "success": True,
                "image_urls": image_urls,
                "stored_url": stored_url,
                "stored_hash": stored_hash
            }

        except Exception as e:
            logger.error(f"Avatar generation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
