"""Voice HTTP endpoints and WebSocket real-time voice chat."""
import asyncio
import json
import logging
import secrets

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from endpoints.agent_helpers import get_agent
from kestrel_sovereign.features.voice.feature import VoicePrivacyError

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


class TTSStreamRequest(BaseModel):
    text: str = ""  # Mode 1: full text to synthesize incrementally
    request_id: str = ""  # Mode 2: piggyback on active agent stream
    voice_id: str = ""
    format: str = "opus"


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

    # If a specific cloud provider is requested and privacy blocks it, return 403
    if provider and not vf.is_provider_allowed(provider, "tts"):
        mode_name = vf._get_privacy_mode_name()
        raise HTTPException(
            status_code=403,
            detail=f"Cannot list voices from '{provider}' in {mode_name} privacy mode. "
                   f"Only local providers are allowed.",
        )

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
    config = vf._voice_config.to_dict()
    config["audio_storage_policy"] = vf._get_audio_storage_policy()
    return config


@router.post("/config")
async def set_voice_config(request: Request, body: VoiceConfigRequest) -> dict:
    """Set the agent's voice configuration."""
    agent = get_agent(request)
    vf = _get_voice_feature(agent)

    # Validate requested providers against privacy mode before applying
    if body.tts_provider and not vf.is_provider_allowed(body.tts_provider, "tts"):
        mode_name = vf._get_privacy_mode_name()
        raise HTTPException(
            status_code=403,
            detail=f"Cannot use '{body.tts_provider}' TTS provider in {mode_name} privacy mode. "
                   f"Install piper-tts for local TTS, or switch to 'anonymous' or higher privacy mode.",
        )
    if body.stt_provider and not vf.is_provider_allowed(body.stt_provider, "stt"):
        mode_name = vf._get_privacy_mode_name()
        raise HTTPException(
            status_code=403,
            detail=f"Cannot use '{body.stt_provider}' STT provider in {mode_name} privacy mode. "
                   f"Install faster-whisper for local STT, or switch to 'anonymous' or higher privacy mode.",
        )

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
    except VoicePrivacyError as e:
        raise HTTPException(status_code=403, detail=str(e))
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
async def synthesize_speech_stream(request: Request, body: TTSStreamRequest) -> StreamingResponse:
    """Streaming TTS with chunked audio output.

    Two modes:
    1. Full text mode: ``body.text`` provided — split into sentences,
       synthesize each incrementally and stream audio chunks.
    2. Agent response mode: ``body.request_id`` provided — tap into an
       active agent streaming response and synthesize sentences as they
       complete.
    """
    if not body.text and not body.request_id:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'request_id'.")

    agent = get_agent(request)
    vf = _get_voice_feature(agent)

    try:
        tts = await vf._get_tts_provider()
    except VoicePrivacyError as e:
        raise HTTPException(status_code=403, detail=str(e))
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
            if body.request_id:
                # Mode 2: tap into active agent stream
                async for chunk in vf.stream_speak(
                    request_id=body.request_id,
                    voice_id=voice_id,
                    output_format=output_format,
                ):
                    yield chunk
            else:
                # Mode 1: split text into sentences, synthesize each
                from kestrel_sovereign.voice.base import split_sentences

                sentences = split_sentences(body.text)
                if not sentences:
                    sentences = [body.text]

                for sentence in sentences:
                    async for chunk in tts.synthesize_stream(
                        text=sentence,
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
    except VoicePrivacyError as e:
        raise HTTPException(status_code=403, detail=str(e))
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


# ------------------------------------------------------------------
# WebSocket binary framing helpers
# ------------------------------------------------------------------

# 1-byte header distinguishes JSON control messages from audio data.
FRAME_JSON: int = 0x00
FRAME_AUDIO: int = 0x01


def encode_control(payload: dict) -> bytes:
    """Encode a JSON control message with the 0x00 header byte."""
    return bytes([FRAME_JSON]) + json.dumps(payload).encode("utf-8")


def encode_audio(chunk: bytes) -> bytes:
    """Encode an audio chunk with the 0x01 header byte."""
    return bytes([FRAME_AUDIO]) + chunk


# ------------------------------------------------------------------
# WebSocket voice chat
# ------------------------------------------------------------------

# Voice chat states
_STATE_LISTENING = "listening"
_STATE_THINKING = "thinking"
_STATE_SPEAKING = "speaking"


def _ws_get_agent(websocket: WebSocket):
    """Get agent from WebSocket app state (mirrors get_agent for HTTP)."""
    agent = getattr(websocket.state, "agent", None) if hasattr(websocket, "state") else None
    if agent is not None:
        return agent
    agent = getattr(websocket.app.state, "agent", None)
    if agent is not None:
        return agent
    return None


def _ws_authenticate(websocket: WebSocket) -> bool:
    """Authenticate WebSocket via query parameter api_key or cookie session."""
    import os
    expected_key = os.environ.get("KESTREL_API_KEY", "")
    if not expected_key:
        return True  # No key configured — open access

    # Check query parameter
    api_key = websocket.query_params.get("api_key", "")
    if api_key and secrets.compare_digest(api_key, expected_key):
        return True

    # Check cookie-based session (OAuth flow)
    if hasattr(websocket, "session"):
        user_email = websocket.session.get("user_email")
        if user_email:
            return True

    return False


@router.websocket("/chat")
async def voice_chat(websocket: WebSocket):
    """Real-time bidirectional voice chat over WebSocket.

    Protocol:
    - Client → Server: raw audio bytes (opus frames)
    - Server → Client: mixed binary/JSON messages distinguished by 1-byte header:
      - 0x00 + JSON bytes = control message (transcript, status, error)
      - 0x01 + audio bytes = TTS audio chunk

    Control messages:
    - {"type": "transcript", "text": "what the user said", "final": true}
    - {"type": "response", "text": "agent's text response"}
    - {"type": "status", "state": "listening|thinking|speaking"}
    - {"type": "error", "message": "..."}

    Auth: query parameter ?api_key=... or session cookie.
    """
    # --- Auth check before accept ---
    if not _ws_authenticate(websocket):
        await websocket.close(code=4401, reason="Authentication required")
        return

    # --- Resolve agent ---
    agent = _ws_get_agent(websocket)
    if agent is None:
        await websocket.close(code=4503, reason="Agent not initialized")
        return

    # --- Resolve VoiceFeature ---
    features = getattr(agent, "features", {})
    vf = features.get("VoiceFeature")
    if vf is None:
        await websocket.close(code=4503, reason="VoiceFeature not available")
        return

    # --- Privacy gate: check at connection time ---
    try:
        stt = await vf._get_stt_provider()
    except VoicePrivacyError as e:
        await websocket.close(code=4403, reason=str(e))
        return
    except Exception as e:
        await websocket.close(code=4503, reason=f"STT provider unavailable: {e}")
        return

    try:
        tts = await vf._get_tts_provider()
    except VoicePrivacyError as e:
        await websocket.close(code=4403, reason=str(e))
        return
    except Exception as e:
        await websocket.close(code=4503, reason=f"TTS provider unavailable: {e}")
        return

    # Resolve voice_id for TTS
    voice_id = vf._voice_config.tts_voice_id
    if not voice_id:
        try:
            voices = await tts.list_voices()
            if voices:
                voice_id = voices[0].voice_id
        except Exception:
            pass
    output_format = vf._voice_config.output_format or "opus"

    # --- Initialize VAD (optional — degrades gracefully if webrtcvad missing) ---
    vad = None
    try:
        from kestrel_sovereign.voice.vad import VoiceActivityDetector, load_vad_config

        # Load VAD config from agent's kestrel.toml if available
        vad_config_dict = {}
        agent_config = getattr(agent, "config", None)
        if agent_config and hasattr(agent_config, "config_file"):
            import toml as _toml
            try:
                cfg_path = getattr(agent_config, "config_file", None)
                if cfg_path and hasattr(cfg_path, "exists") and cfg_path.exists():
                    vad_config_dict = _toml.load(str(cfg_path))
            except Exception:
                pass
        vad_cfg = load_vad_config(vad_config_dict or None)
        vad = VoiceActivityDetector(
            aggressiveness=vad_cfg["aggressiveness"],
            silence_threshold_ms=vad_cfg["silence_threshold_ms"],
            pre_speech_padding_ms=vad_cfg["pre_speech_padding_ms"],
        )
        logger.info("VAD enabled (aggressiveness=%d, silence=%dms, padding=%dms)",
                     vad_cfg["aggressiveness"], vad_cfg["silence_threshold_ms"],
                     vad_cfg["pre_speech_padding_ms"])
    except ImportError:
        logger.info("webrtcvad not installed — VAD disabled, using direct STT passthrough")
    except Exception as e:
        logger.warning("VAD initialization failed — falling back to direct STT: %s", e)

    # --- Accept the connection ---
    await websocket.accept()

    async def send_control(payload: dict):
        """Send a framed JSON control message."""
        await websocket.send_bytes(encode_control(payload))

    async def send_status(state: str):
        """Send a status transition control message."""
        await send_control({"type": "status", "state": state})

    # Notify client we are ready
    await send_status(_STATE_LISTENING)

    async def _process_utterance(transcript_text: str):
        """Run agent processing and TTS for a final transcript."""
        # --- Thinking ---
        await send_status(_STATE_THINKING)
        await send_control({"type": "transcript", "text": transcript_text, "final": True})

        # Collect the full agent response via streaming
        full_response = []
        try:
            if hasattr(agent, "process_input_streaming"):
                async for chunk in agent.process_input_streaming(transcript_text):
                    full_response.append(chunk)
                    # Send incremental text to client
                    await send_control({"type": "response", "text": chunk})
            else:
                result = await agent.process_input(transcript_text)
                full_response.append(result)
                await send_control({"type": "response", "text": result})
        except Exception as e:
            logger.error("Agent processing error in voice chat: %s", e, exc_info=True)
            await send_control({"type": "error", "message": "Agent processing failed."})
            await send_status(_STATE_LISTENING)
            return

        response_text = "".join(full_response)
        if not response_text.strip():
            await send_status(_STATE_LISTENING)
            return

        # --- Speaking ---
        await send_status(_STATE_SPEAKING)

        try:
            if voice_id:
                async for audio_chunk in tts.synthesize_stream(
                    text=response_text,
                    voice_id=voice_id,
                    model=vf._voice_config.tts_model,
                    output_format=output_format,
                ):
                    await websocket.send_bytes(encode_audio(audio_chunk))
            else:
                logger.warning("No TTS voice_id configured; skipping speech synthesis.")
        except Exception as e:
            logger.error("TTS streaming error in voice chat: %s", e, exc_info=True)
            await send_control({"type": "error", "message": "TTS synthesis failed."})

        # Back to listening
        await send_status(_STATE_LISTENING)

    async def _transcribe_and_process(audio_data: bytes):
        """Transcribe audio data via STT and process the result."""
        try:
            if hasattr(stt, "transcribe_stream"):
                async def _single_chunk_stream(data=audio_data):
                    yield data

                async for text_segment in stt.transcribe_stream(
                    _single_chunk_stream(), language=""
                ):
                    if not text_segment.strip():
                        continue
                    is_final = True
                    await send_control({
                        "type": "transcript",
                        "text": text_segment,
                        "final": is_final,
                    })
                    if is_final:
                        await _process_utterance(text_segment)
            else:
                transcript = await stt.transcribe(
                    audio=audio_data, language="", audio_format="opus"
                )
                if transcript and transcript.strip():
                    await send_control({
                        "type": "transcript",
                        "text": transcript,
                        "final": True,
                    })
                    await _process_utterance(transcript)
        except Exception as e:
            logger.error("STT error in voice chat: %s", e, exc_info=True)
            await send_control({"type": "error", "message": "Transcription failed."})

    # --- VAD-based receive loop ---
    if vad is not None:
        # VAD mode: feed audio through VAD, accumulate utterances,
        # then send complete utterances to STT.
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def _vad_audio_stream():
            """Yield audio frames from the queue for VAD processing."""
            while True:
                frame = await audio_queue.get()
                if frame is None:
                    break
                yield frame

        async def _vad_processor():
            """Run VAD on queued audio and process detected utterances."""
            try:
                async for event_type, audio_data in vad.detect_utterances(_vad_audio_stream()):
                    if event_type == "speech_start":
                        await send_status(_STATE_LISTENING)
                    elif event_type == "speech_end":
                        await _transcribe_and_process(audio_data)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("VAD processor error: %s", e, exc_info=True)

        vad_task = asyncio.create_task(_vad_processor())

        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"]:
                    await audio_queue.put(message["bytes"])

                elif "text" in message and message["text"]:
                    try:
                        client_msg = json.loads(message["text"])
                        msg_type = client_msg.get("type", "")

                        if msg_type == "config":
                            logger.debug("Voice chat config update: %s", client_msg)
                        elif msg_type == "end":
                            break
                        else:
                            logger.debug("Unknown client message type: %s", msg_type)
                    except json.JSONDecodeError:
                        await send_control({"type": "error", "message": "Invalid JSON."})

        except WebSocketDisconnect:
            logger.debug("Voice chat WebSocket disconnected")
        except Exception as e:
            logger.error("Unexpected error in voice chat: %s", e, exc_info=True)
            try:
                await send_control({"type": "error", "message": "Internal server error."})
            except Exception:
                pass
        finally:
            await audio_queue.put(None)  # Signal VAD stream end
            vad_task.cancel()
            try:
                await vad_task
            except asyncio.CancelledError:
                pass

    else:
        # --- Fallback: direct STT passthrough (no VAD) ---
        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"]:
                    await _transcribe_and_process(message["bytes"])

                elif "text" in message and message["text"]:
                    try:
                        client_msg = json.loads(message["text"])
                        msg_type = client_msg.get("type", "")

                        if msg_type == "config":
                            logger.debug("Voice chat config update: %s", client_msg)
                        elif msg_type == "end":
                            break
                        else:
                            logger.debug("Unknown client message type: %s", msg_type)
                    except json.JSONDecodeError:
                        await send_control({"type": "error", "message": "Invalid JSON."})

        except WebSocketDisconnect:
            logger.debug("Voice chat WebSocket disconnected")
        except Exception as e:
            logger.error("Unexpected error in voice chat: %s", e, exc_info=True)
            try:
                await send_control({"type": "error", "message": "Internal server error."})
            except Exception:
                pass
