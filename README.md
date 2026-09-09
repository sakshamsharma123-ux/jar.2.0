# Real-Time Voice AI Agent (Stage 1)

A local, low-latency, real-time voice conversation loop built using the **Pipecat** framework, **Deepgram Nova-2** (STT), **Groq Llama 3.1 8B Instant** (LLM), **Deepgram Aura** (TTS), and **Silero VAD** for sub-250ms barge-in interruptions.

---

## 🏗️ Architecture

```
[System Microphone]
       │
       ▼
[LocalAudioTransport (Input)]
       │
       ▼
[SileroVADAnalyzer] ──────(Speech Start / Barge-In)─────► Truncates TTS output (<250ms)
       │
       ▼
[DeepgramSTTService (Nova-2)] ──(Streaming WebSocket)──► Emits User Transcripts
       │
       ▼
[LLM Context Aggregator] ────(System prompt + User Turn)
       │
       ▼
[GroqLLMService (Llama-3.1-8b-instant)] ──(Fast token streaming)
       │
       ▼
[DeepgramTTSService (Aura: aura-asteria-en)] ──(Streaming synthesis)
       │
       ▼
[LocalAudioTransport (Output)]
       │
       ▼
[System Speakers]
```

---

## 📋 Prerequisites

- Python 3.10, 3.11, or 3.12
- Working microphone and speakers / headphones

---

## 🚀 Quick Start

### 1. Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Pre-flight check
Run the diagnostic script to verify audio devices and configuration:
```powershell
python verify_voice.py
```

### 4. Configure API Keys (When ready)
Edit the `.env` file generated in the project root:
```env
DEEPGRAM_API_KEY=your_actual_deepgram_api_key
GROQ_API_KEY=your_actual_groq_api_key
```

### 5. Launch the Voice Agent
```powershell
python src/voice_agent.py
```

---

## 🎙️ Real-Time Lifecycle Events

As you speak with the agent, the console streams visual lifecycle indicators:

- `[LISTENING]` — Microphone is open, VAD is actively listening for your voice.
- `[USER TRANSCRIPT]` — Real-time transcription of your speech from Deepgram.
- `[BOT THINKING]` — User turn completed, Groq LLM is generating tokens.
- `[BOT SPEAKING]` — Deepgram Aura is playing synthesized speech through speakers.
- `[BARGE-IN DETECTED]` — You spoke while the bot was speaking; outgoing audio and generation stopped immediately (<250ms).

---

## ⚙️ Configuration & Tuning (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DEEPGRAM_API_KEY` | *(Required)* | Deepgram API key (https://console.deepgram.com) |
| `GROQ_API_KEY` | *(Required)* | Groq API key (https://console.groq.com/keys) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model for ultra-low latency inference |
| `DEEPGRAM_TTS_VOICE` | `aura-asteria-en` | Deepgram Aura voice model |
| `VAD_CONFIDENCE` | `0.7` | Silero speech detection confidence threshold |
| `VAD_START_SECS` | `0.15` | Minimum seconds of speech before triggering turn start |
| `VAD_STOP_SECS` | `0.25` | Silence duration before triggering turn completion |
| `AUDIO_INPUT_DEVICE_INDEX` | *(Auto)* | PyAudio device index for microphone |
| `AUDIO_OUTPUT_DEVICE_INDEX` | *(Auto)* | PyAudio device index for speakers |
