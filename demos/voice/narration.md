# Voice Vignette — Narration

## Why this feature exists

Voice was Kestrel's most-requested capability for months — the ability to
hold a *spoken* conversation with the agent, with the same constitutional
governance and privacy modes as the chat surface.  Squack II (epic #721,
shipped late April 2026) added it.

This vignette walks the eight beats of an end-to-end voice session:

1. **Mic button arrives** in the chat header.
2. **Voice picker** opens; discovered voices are listed.
3. User picks a **voice + writes a session directive**.
4. Click the mic — **session engages**.
5. **Path badge** shows the active route (Realtime or Pipeline).
6. **Transcript drawer** renders the user + agent turns live.
7. **Esc** returns to idle.
8. Settings **persist** across reload.

## Architecture quick reference

* **Realtime route** — OpenAI Realtime API; STT + LLM + TTS as one pipe.
* **Pipeline route** — STT (whisper) → text agent → TTS (ElevenLabs/etc).
  Cheaper, slower, supports any LLM. Default for non-OpenAI agents.
* **Voice picker** — discovered from the active TTS provider; persists in
  localStorage so it survives reload.
* **Path badge** — visible signal of which route is in use.
* **Transcript drawer** — left side; live user + agent turns.

## Running the vignette

```bash
demos/run.sh voice
kestrel-eye review --config demos/voice/eye.toml
```

> The voice demo records audio if `KESTREL_VOICE_AUDIO=1`; otherwise it
> verifies the visual surface only.  CI gates live in
> `tests/e2e/test_voice_ui.spec.cjs` which mocks the network boundary.
