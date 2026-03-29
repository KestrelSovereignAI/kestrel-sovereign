"""Voice HTTP endpoints — TTS synthesis, STT transcription, voice config."""
import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------


class VoiceConfigRequest(BaseModel):
    tts_provider: str = ""
    tts_voice_id: str = ""
    tts_model: str = ""
    stt_provider: str = ""
    stt_model: str = ""
    output_format: str = "opus"


class TTSRequest(BaseModel):
    text: str
    voice_id: str = ""  # Override agent's default voice
    format: str = "opus"  # opus, mp3, wav, pcm


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_FORMAT_CONTENT_TYPES = {
    "opus": "audio/opus",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def _get_voice_feature(agent):
    """Retrieve VoiceFeature from agent, raising 503 if unavailable."""
    features = getattr(agent, "features", {})
    vf = features.get("VoiceFeature")
    if vf is None:
        raise HTTPException(status_code=503, detail="VoiceFeature not available on this agent.")
    return vf


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.get("/voices")
async def list_voices(request: Request, provider: str = "") -> dict:
    """List available TTS voices, filtered by current privacy mode."""
    agent = get_agent(request)
    vf = _get_voice_feature(agent)
    try:
        return await vf.list_voices(provider=provider)
    except Exception as e:
        logger.error("Error listing voices: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing voices.")


@router.get("/config")
async def get_voice_config(request: Request) -> dict:
    """Get the agent's current voice configuration."""
    agent = get_agent(request)
    vf = _get_voice_feature(agent)
    return vf._voice_config.to_dict()


@router.post("/config")
async def set_voice_config(request: Request, body: VoiceConfigRequest) -> dict:
    """Set the agent's voice configuration."""
    agent = get_agent(request)
    vf = _get_voice_feature(agent)

    from kestrel_sovereign.voice.base import VoiceConfig

    new_config = VoiceConfig(
        tts_provider=body.tts_provider or vf._voice_config.tts_provider,
        tts_voice_id=body.tts_voice_id or vf._voice_config.tts_voice_id,
        tts_model=body.tts_model or vf._voice_config.tts_model,
        stt_provider=body.stt_provider or vf._voice_config.stt_provider,
        stt_model=body.stt_model or vf._voice_config.stt_model,
        output_format=body.output_format or vf._voice_config.output_format,
    )
    vf._voice_config = new_config

    # Persist to agent identity if available
    identity = getattr(agent, "identity", None)
    if identity and hasattr(identity, "voice_config"):
        identity.voice_config = new_config.to_dict()

    # Emit observability event for audit trail
    obs = agent.features.get("ObservabilityFeature") if hasattr(agent, "features") else None
    if obs and hasattr(obs, "emit"):
        await obs.emit("voice_config_changed", new_config.to_dict())

    return {"success": True, "config": new_config.to_dict()}


@router.post("/tts")
async def synthesize_speech(request: Request, body: TTSRequest) -> Response:
    """Synthesize speech from text. Returns audio bytes."""
    agent = get_agent(request)
    vf = _get_voice_feature(agent)

    try:
        tts = await vf._get_tts_provider()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    voice_id = body.voice_id or vf._voice_config.tts_voice_id
    if not voice_id:
        voices = await tts.list_voices()
        if not voices:
            raise HTTPException(status_code=503, detail="No voices available on the current TTS provider.")
        voice_id = voices[0].voice_id

    output_format = body.format or vf._voice_config.output_format or "opus"

    try:
        audio_bytes = await tts.synthesize(
            text=body.text,
            voice_id=voice_id,
            model=vf._voice_config.tts_model,
            output_format=output_format,
        )
    except Exception as e:
        logger.error("TTS synthesis error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="TTS synthesis failed.")

    content_type = _FORMAT_CONTENT_TYPES.get(output_format, "application/octet-stream")
    return Response(content=audio_bytes, media_type=content_type)


@router.post("/tts/stream")
async def synthesize_speech_stream(request: Request, body: TTSRequest) -> StreamingResponse:
    """Streaming TTS. Returns chunked audio."""
    agent = get_agent(request)
    vf = _get_voice_feature(agent)

    try:
        tts = await vf._get_tts_provider()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    voice_id = body.voice_id or vf._voice_config.tts_voice_id
    if not voice_id:
        voices = await tts.list_voices()
        if not voices:
            raise HTTPException(status_code=503, detail="No voices available on the current TTS provider.")
        voice_id = voices[0].voice_id

    output_format = body.format or vf._voice_config.output_format or "opus"

    async def audio_generator():
        try:
            async for chunk in tts.synthesize_stream(
                text=body.text,
                voice_id=voice_id,
                model=vf._voice_config.tts_model,
                output_format=output_format,
            ):
                yield chunk
        except Exception as e:
            logger.error("TTS streaming error: %s", e, exc_info=True)

    content_type = _FORMAT_CONTENT_TYPES.get(output_format, "application/octet-stream")
    return StreamingResponse(
        audio_generator(),
        media_type=content_type,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/stt")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    language: str = "",
) -> dict:
    """Transcribe uploaded audio to text."""
    agent = get_agent(request)
    vf = _get_voice_feature(agent)

    try:
        stt = await vf._get_stt_provider()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # Infer format from filename extension
    audio_format = "opus"
    if file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext in ("mp3", "wav", "opus", "ogg", "flac", "webm", "m4a"):
            audio_format = ext

    try:
        transcript = await stt.transcribe(
            audio=audio_bytes,
            language=language,
            audio_format=audio_format,
        )
    except Exception as e:
        logger.error("STT transcription error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Transcription failed.")

    return {
        "text": transcript,
        "language": language or "auto",
        "duration_seconds": None,  # Provider-dependent; not all return this
    }
