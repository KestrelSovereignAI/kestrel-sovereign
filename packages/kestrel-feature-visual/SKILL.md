# VisualIdentityFeature

> Companion image generation - avatars, selfies, LoRA training for character consistency, scene variations.

## Skills

### generate_avatar
- **Description**: Generate an avatar portrait from a description and store it as part of agent identity
- **Category**: communication
- **Command**: `!avatar`
- **Parameters**:
  - `description` (string, required): Physical description for the avatar
  - `num_outputs` (int, optional): Number of options to generate (1-4, default 2)
- **Returns**: `{"success": bool, "image_urls": list, "stored_url": str, "stored_hash": str}`

### generate_selfie
- **Description**: Generate a selfie or portrait of the companion character
- **Category**: communication
- **Command**: `!selfie`
- **Parameters**:
  - `scene` (string, optional): Style of photo (casual, portrait, glamour, flirty, cozy, adventure, mysterious, romantic, playful, dreamy, confident, beach, swimsuit, tropical, pool, fitness, nightout, lingerie, summer, nurse, topless, nude)
  - `companion_id` (string, optional): Companion UUID for LoRA lookup
  - `lora_model_path` (string, optional): Direct path to LoRA model
  - `custom_prompt` (string, optional): Custom generation prompt
  - `style` (string, optional): Art style (photorealistic, anime, artistic)
  - `allow_training` (bool, optional): If True and no LoRA, train one (default True)
  - `provider` (string, optional): Force specific provider (runpod, vertex_ai, vastai)
- **Returns**: `{"success": bool, "image_url": str, "scene": str, "used_lora": bool}`

### train_lora
- **Description**: Train a LoRA model for character-consistent selfie generation
- **Category**: communication
- **Command**: `!train-lora`
- **Parameters**:
  - `companion_id` (string, optional): Companion UUID (auto-filled from context)
  - `image_url` (string, optional): Avatar image URL (auto-filled from context)
- **Returns**: `{"success": bool, "job_id": str, "status": str}`

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `REPLICATE_API_TOKEN` | env | - | Replicate API token for avatar generation |
| `RUNPOD_API_KEY` | env | - | RunPod API key for LoRA training/generation |

## Dependencies

- Requires: kestrel-sovereign, replicate, httpx
- Optional: runpod (for LoRA training)
