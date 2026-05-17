"""
Unified types for TrainingProvider protocol.

These types are shared across all training providers to ensure consistent
data structures regardless of the underlying infrastructure (Vertex AI,
GCP Compute Engine, RunPod, Vast.ai, Replicate).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class TrainingState(Enum):
    """
    Unified training job states across all providers.

    State flow:
        PENDING -> PROVISIONING -> PREPARING -> TRAINING -> COMPLETED
                                                         -> FAILED
                                                         -> CANCELLED
    """
    PENDING = "pending"              # Job created, waiting to start
    PROVISIONING = "provisioning"    # Instance/resources being allocated
    PREPARING = "preparing"          # Environment setup (model loading, etc.)
    TRAINING = "training"            # Actively training
    COMPLETED = "completed"          # Successfully finished
    FAILED = "failed"                # Error occurred
    CANCELLED = "cancelled"          # User cancelled

    @classmethod
    def from_vertex_state(cls, state: str) -> "TrainingState":
        """Map Vertex AI job state to unified state."""
        mapping = {
            "pending": cls.PENDING,
            "queued": cls.PENDING,
            "running": cls.TRAINING,
            "completed": cls.COMPLETED,
            "succeeded": cls.COMPLETED,
            "failed": cls.FAILED,
            "cancelled": cls.CANCELLED,
            "cancelling": cls.CANCELLED,
        }
        return mapping.get(state.lower().replace("job_state_", ""), cls.PENDING)

    @classmethod
    def from_gcp_instance_state(cls, state: str) -> "TrainingState":
        """Map GCP Compute instance state to unified state."""
        mapping = {
            "offline": cls.PENDING,
            "provisioning": cls.PROVISIONING,
            "staging": cls.PROVISIONING,
            "running": cls.TRAINING,
            "stopping": cls.CANCELLED,
            "terminated": cls.CANCELLED,
            "suspended": cls.CANCELLED,
            "error": cls.FAILED,
        }
        return mapping.get(state.lower(), cls.PENDING)

    @classmethod
    def from_runpod_state(cls, state: str) -> "TrainingState":
        """Map RunPod pod state to unified state."""
        mapping = {
            "offline": cls.PENDING,
            "provisioning": cls.PROVISIONING,
            "loading": cls.PREPARING,
            "ready": cls.TRAINING,
            "running": cls.TRAINING,
            "terminating": cls.CANCELLED,
            "error": cls.FAILED,
        }
        return mapping.get(state.lower(), cls.PENDING)

    @classmethod
    def from_vastai_state(cls, state: str) -> "TrainingState":
        """Map Vast.ai instance state to unified state."""
        mapping = {
            "offline": cls.PENDING,
            "creating": cls.PROVISIONING,
            "loading": cls.PREPARING,
            "running": cls.TRAINING,
            "stopping": cls.CANCELLED,
            "exited": cls.CANCELLED,
            "error": cls.FAILED,
        }
        return mapping.get(state.lower(), cls.PENDING)

    @classmethod
    def from_replicate_state(cls, state: str) -> "TrainingState":
        """Map Replicate training state to unified state."""
        mapping = {
            "preparing": cls.PREPARING,
            "creating_zip": cls.PREPARING,
            "submitting": cls.PROVISIONING,
            "training": cls.TRAINING,
            "downloading": cls.COMPLETED,  # Still processing but training done
            "storing": cls.COMPLETED,
            "completed": cls.COMPLETED,
            "succeeded": cls.COMPLETED,
            "failed": cls.FAILED,
            "canceled": cls.CANCELLED,
        }
        return mapping.get(state.lower(), cls.PENDING)

    def is_terminal(self) -> bool:
        """Return True if this is a terminal state (no further progress)."""
        return self in (self.COMPLETED, self.FAILED, self.CANCELLED)

    def is_active(self) -> bool:
        """Return True if job is actively running."""
        return self in (self.PROVISIONING, self.PREPARING, self.TRAINING)


class ProviderType(Enum):
    """Type of training provider lifecycle model."""
    SERVERLESS = "serverless"       # Job runs to completion (Vertex AI, Replicate)
    SESSION_BASED = "session_based"  # Requires instance management (GCP, RunPod, Vast.ai)
    LOCAL = "local"                 # Local training on owned hardware


@dataclass
class ProviderCapabilities:
    """
    Capabilities and limitations of each training provider.

    Used by factory to help users select appropriate providers based on needs.
    """
    training: bool = False              # Supports LoRA training
    generation: bool = False            # Supports image generation with trained LoRA
    unfiltered_generation: bool = False  # Provider does not enforce additional content filters
    flux_version: str = "1.x"           # FLUX model version: "1.x" or "2.x"
    supports_lora_download: bool = True  # Can download weights for use elsewhere

    def to_dict(self) -> Dict[str, Any]:
        return {
            "training": self.training,
            "generation": self.generation,
            "unfiltered_generation": self.unfiltered_generation,
            "flux_version": self.flux_version,
            "supports_lora_download": self.supports_lora_download,
        }


@dataclass
class TrainingConfig:
    """Configuration for LoRA training.

    Common parameters across all providers.
    """
    # Training parameters
    trigger_word: Optional[str] = None  # Default: TOK{companion_id[:8]}
    steps: int = 1000
    lora_rank: int = 16
    learning_rate: float = 1e-4
    batch_size: int = 1
    resolution: str = "512,768,1024"

    # Infrastructure parameters
    profile: str = "training"  # GPU profile name
    use_spot: bool = True      # Use spot/preemptible instances (60-90% cheaper)
    ttl_seconds: int = 7200    # Max session duration (2 hours default)

    # Callback
    callback_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_word": self.trigger_word,
            "steps": self.steps,
            "lora_rank": self.lora_rank,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "resolution": self.resolution,
            "profile": self.profile,
            "use_spot": self.use_spot,
            "ttl_seconds": self.ttl_seconds,
            "callback_url": self.callback_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(
            trigger_word=data.get("trigger_word"),
            steps=data.get("steps", 1000),
            lora_rank=data.get("lora_rank", 16),
            learning_rate=data.get("learning_rate", 1e-4),
            batch_size=data.get("batch_size", 1),
            resolution=data.get("resolution", "512,768,1024"),
            profile=data.get("profile", "training"),
            use_spot=data.get("use_spot", True),
            ttl_seconds=data.get("ttl_seconds", 7200),
            callback_url=data.get("callback_url"),
        )


@dataclass
class TrainingJob:
    """
    Represents a LoRA training job across any provider.

    This is the canonical representation returned by start_training().
    """
    job_id: str                    # Our internal UUID
    companion_id: str              # Companion being trained
    provider: str                  # Provider name: 'vertex_ai', 'gcp_compute', etc.
    state: TrainingState           # Current state
    trigger_word: str              # LoRA trigger word (e.g., TOKabc123)
    created_at: datetime           # When job was created
    config: TrainingConfig = field(default_factory=TrainingConfig)

    # Provider-specific identifiers (for internal tracking)
    provider_job_id: Optional[str] = None     # e.g., Vertex job name, Replicate training ID
    provider_session_id: Optional[str] = None  # e.g., GCP instance name, RunPod pod_id

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Output location
    output_path: Optional[str] = None  # GCS URI, local path, etc.

    # Error info
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "companion_id": self.companion_id,
            "provider": self.provider,
            "state": self.state.value,
            "trigger_word": self.trigger_word,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_path": self.output_path,
            "error_message": self.error_message,
            "provider_job_id": self.provider_job_id,
            "provider_session_id": self.provider_session_id,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        return cls(
            job_id=data["job_id"],
            companion_id=data["companion_id"],
            provider=data["provider"],
            state=TrainingState(data["state"]),
            trigger_word=data["trigger_word"],
            created_at=datetime.fromisoformat(data["created_at"]),
            config=TrainingConfig.from_dict(data.get("config", {})),
            provider_job_id=data.get("provider_job_id"),
            provider_session_id=data.get("provider_session_id"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            output_path=data.get("output_path"),
            error_message=data.get("error_message"),
        )


@dataclass
class TrainingStatus:
    """
    Status snapshot of a training job.

    Returned by get_status() for polling job progress.
    """
    job_id: str
    state: TrainingState
    progress: float  # 0.0 to 1.0
    message: Optional[str] = None
    error: Optional[str] = None

    # Timing info
    elapsed_seconds: Optional[float] = None
    estimated_remaining_seconds: Optional[float] = None

    # Provider-specific details (optional, for debugging)
    provider_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingStatus":
        return cls(
            job_id=data["job_id"],
            state=TrainingState(data["state"]),
            progress=data.get("progress", 0.0),
            message=data.get("message"),
            error=data.get("error"),
            elapsed_seconds=data.get("elapsed_seconds"),
            estimated_remaining_seconds=data.get("estimated_remaining_seconds"),
            provider_details=data.get("provider_details", {}),
        )


class GenerationState(Enum):
    """
    Unified generation job states across all providers.
    """
    PENDING = "pending"              # Job created, waiting to start
    LOADING_MODEL = "loading_model"  # Model being loaded (with CPU offload)
    LOADING_LORA = "loading_lora"    # LoRA weights being loaded
    GENERATING = "generating"        # Actively generating image
    COMPLETED = "completed"          # Successfully finished
    FAILED = "failed"                # Error occurred

    def is_terminal(self) -> bool:
        """Return True if this is a terminal state."""
        return self in (self.COMPLETED, self.FAILED)

    def is_active(self) -> bool:
        """Return True if generation is actively running."""
        return self in (self.LOADING_MODEL, self.LOADING_LORA, self.GENERATING)


@dataclass
class GenerationConfig:
    """Configuration for image generation with LoRA."""
    prompt: str                       # Generation prompt
    lora_path: str                    # Path to LoRA safetensors
    trigger_word: str = "TOK"         # Trigger word for LoRA
    num_outputs: int = 1              # Number of images (1-4)
    width: int = 1024                 # Image width
    height: int = 1024                # Image height
    num_inference_steps: int = 28     # Denoising steps
    guidance_scale: float = 4.0       # CFG scale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "lora_path": self.lora_path,
            "trigger_word": self.trigger_word,
            "num_outputs": self.num_outputs,
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }


@dataclass
class GenerationResult:
    """Result of an image generation job."""
    job_id: str
    state: GenerationState
    images: list[str] = field(default_factory=list)  # base64 data URLs
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "images": self.images,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }
