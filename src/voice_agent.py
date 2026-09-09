"""
src/voice_agent.py
================================================================================
Real-Time Voice AI Agent (Stage 1)
--------------------------------------------------------------------------------
Architecture:
  [Microphone] -> LocalAudioTransport (Input)
               -> Silero VAD (Speech detection & low-latency barge-in)
               -> Deepgram STT (Streaming Nova-2 WebSockets)
               -> LLM Context Aggregator (System Prompt & Conversation history)
               -> Groq LLM (Llama 3.1 8B Instant token generation)
               -> Deepgram TTS (Aura Asteria streaming synthesis)
               -> LocalAudioTransport (Output) -> [Speakers]

Lifecycle Events Logged:
  - [LISTENING]          : Microphone open, awaiting user speech
  - [USER TRANSCRIPT]    : User speech recognized in real time
  - [BOT THINKING]       : User finished speaking, LLM is streaming tokens
  - [BOT SPEAKING]       : TTS audio output playing through speakers
  - [BARGE-IN DETECTED]  : User spoke during bot playback; audio immediately cut (<250ms)
================================================================================
"""

import asyncio
import os
import signal
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

# Terminal color styling
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    MAGENTA = Fore.MAGENTA
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL
except ImportError:
    CYAN = GREEN = YELLOW = RED = MAGENTA = BOLD = RESET = ""

# Pipecat framework imports
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    InterruptionFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams


# ============================================================================
# 1. Configuration & Environment Validation
# ============================================================================

def load_agent_config():
    """Load configuration from .env file with validated fallbacks."""
    # Find .env relative to project root
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    # Model & Voice tuning
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
    tts_voice = os.getenv("DEEPGRAM_TTS_VOICE", "aura-asteria-en").strip()

    # VAD Latency Tuning
    vad_confidence = float(os.getenv("VAD_CONFIDENCE", "0.7"))
    vad_start_secs = float(os.getenv("VAD_START_SECS", "0.15"))
    vad_stop_secs = float(os.getenv("VAD_STOP_SECS", "0.25"))

    # Audio Device Overrides
    audio_in_idx = os.getenv("AUDIO_INPUT_DEVICE_INDEX", "").strip()
    audio_out_idx = os.getenv("AUDIO_OUTPUT_DEVICE_INDEX", "").strip()

    input_device = int(audio_in_idx) if audio_in_idx.isdigit() else None
    output_device = int(audio_out_idx) if audio_out_idx.isdigit() else None

    return {
        "deepgram_key": deepgram_key,
        "groq_key": groq_key,
        "groq_model": groq_model,
        "tts_voice": tts_voice,
        "vad_confidence": vad_confidence,
        "vad_start_secs": vad_start_secs,
        "vad_stop_secs": vad_stop_secs,
        "input_device_index": input_device,
        "output_device_index": output_device,
    }


def validate_api_keys(config: dict) -> bool:
    """Validate that API keys are set and not placeholder values."""
    dg_key = config["deepgram_key"]
    g_key = config["groq_key"]

    is_dg_missing = (not dg_key) or ("your_deepgram" in dg_key)
    is_groq_missing = (not g_key) or ("your_groq" in g_key)

    if is_dg_missing or is_groq_missing:
        print(f"\n{RED}{BOLD}[CONFIGURATION REQUIRED]{RESET}")
        print(f"{YELLOW}The real-time voice loop requires valid API keys in your .env file:{RESET}")
        if is_dg_missing:
            print(f"  - {BOLD}DEEPGRAM_API_KEY{RESET}: Set your Deepgram key (https://console.deepgram.com)")
        if is_groq_missing:
            print(f"  - {BOLD}GROQ_API_KEY{RESET}: Set your Groq API key (https://console.groq.com/keys)")
        print(f"\n{CYAN}Once you add the keys to .env, re-run: python src/voice_agent.py{RESET}\n")
        return False
    return True


def resolve_audio_devices(requested_in: int | None, requested_out: int | None) -> tuple[int | None, int | None]:
    """Auto-detect working audio devices if not explicitly specified."""
    try:
        import pyaudio
        p = pyaudio.PyAudio()

        # Input device resolution
        final_in = requested_in
        if final_in is None:
            try:
                def_in = p.get_default_input_device_info()
                final_in = def_in["index"]
            except Exception:
                # Find first input device with input channels (preferring mic)
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev.get("maxInputChannels", 0) > 0:
                        if "mic" in dev.get("name", "").lower():
                            final_in = i
                            break
                        elif final_in is None:
                            final_in = i

        # Output device resolution
        final_out = requested_out
        if final_out is None:
            try:
                def_out = p.get_default_output_device_info()
                final_out = def_out["index"]
            except Exception:
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev.get("maxOutputChannels", 0) > 0:
                        final_out = i
                        break

        p.terminate()
        return final_in, final_out
    except Exception:
        return requested_in, requested_out


# ============================================================================
# 2. Console Lifecycle Event Logger (FrameProcessor)
# ============================================================================

class ConsoleLifecycleLogger(FrameProcessor):
    """
    Monitors frames passing through the Pipecat pipeline and logs real-time lifecycle states:
      - [LISTENING]
      - [USER TRANSCRIPT]
      - [BOT THINKING]
      - [BOT SPEAKING]
      - [BARGE-IN DETECTED]
    """

    def __init__(self):
        super().__init__()
        self._is_bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # 1. User Transcript
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip() if frame.text else ""
            if text:
                print(f"\n{CYAN}{BOLD}[USER TRANSCRIPT]{RESET} {text}")

        # 2. Bot Thinking (LLM turn started)
        elif isinstance(frame, LLMFullResponseStartFrame):
            print(f"{YELLOW}{BOLD}[BOT THINKING]{RESET} ...")

        # 3. Bot Speaking (TTS playback begins)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._is_bot_speaking = True
            print(f"{GREEN}{BOLD}[BOT SPEAKING]{RESET} ", end="", flush=True)

        # 4. Bot Finished Speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._is_bot_speaking:
                print()  # newline after streamed response
            self._is_bot_speaking = False
            print(f"{MAGENTA}{BOLD}[LISTENING]{RESET} (Speak to assistant...)")

        # 5. Natural Barge-in Interruption
        elif isinstance(frame, InterruptionFrame):
            print(f"\n{RED}{BOLD}[BARGE-IN DETECTED]{RESET} Interrupted assistant! Outgoing audio stopped.")
            self._is_bot_speaking = False

        await self.push_frame(frame, direction)


# ============================================================================
# 3. System Prompt Definition
# ============================================================================

SYSTEM_PROMPT = """You are a low-latency, real-time voice AI assistant.
Your responses are spoken aloud over voice synthesis.
Follow these rules strictly:
1. Keep answers concise: exactly 1 to 2 sentences per turn.
2. Be direct, natural, and conversational.
3. Do not use bullet points, markdown symbols, asterisks, URLs, or emoji, because your text is read directly by a text-to-speech engine.
4. If you do not know something, state that directly without guessing.
"""


# ============================================================================
# 4. Pipeline Assembly & Runner
# ============================================================================

async def run_voice_agent():
    config = load_agent_config()

    print(f"""
{CYAN}{BOLD}================================================================
          Real-Time Voice AI Agent (Stage 1)
================================================================{RESET}
  * Framework : Pipecat AI
  * STT Engine: Deepgram Nova-2 (Streaming WebSocket)
  * LLM Engine: Groq ({config['groq_model']})
  * TTS Engine: Deepgram Aura ({config['tts_voice']})
  * VAD Engine: Silero VAD (Low Latency Barge-in)
================================================================
""")

    if not validate_api_keys(config):
        return

    # Auto-detect or resolve input/output device indices
    input_idx, output_idx = resolve_audio_devices(
        config["input_device_index"],
        config["output_device_index"]
    )
    print(f"  * Audio Input Device Index : {input_idx}")
    print(f"  * Audio Output Device Index: {output_idx}")

    # 1. Initialize Local Audio Transport
    transport_params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        input_device_index=input_idx,
        output_device_index=output_idx,
    )
    transport = LocalAudioTransport(transport_params)

    # 2. Silero VAD (Tuned for ultra-fast turn taking and immediate barge-in)
    vad = SileroVADAnalyzer(
        params=VADParams(
            confidence=config["vad_confidence"],
            start_secs=config["vad_start_secs"],
            stop_secs=config["vad_stop_secs"],
            min_volume=0.6,
        )
    )
    vad_processor = VADProcessor(vad_analyzer=vad)

    # 3. Deepgram STT (Streaming Nova-2 WebSockets)
    stt = DeepgramSTTService(
        api_key=config["deepgram_key"],
        live_options=None,  # Uses optimized streaming defaults
    )

    # 4. Groq LLM (Ultra-low latency token generation)
    llm = GroqLLMService(
        api_key=config["groq_key"],
        settings=GroqLLMService.Settings(
            model=config["groq_model"],
            temperature=0.6,
            max_tokens=150,
        ),
    )

    # 5. Deepgram TTS (Aura Asteria streaming synthesis)
    tts = DeepgramTTSService(
        api_key=config["deepgram_key"],
        settings=DeepgramTTSService.Settings(
            voice=config["tts_voice"],
        ),
    )

    # 6. Conversation Context & Lifecycle Logger
    initial_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(initial_messages)
    context_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad),
    )
    lifecycle_logger = ConsoleLifecycleLogger()

    # 7. Pipeline Construction
    # Flow: Audio Input -> Silero VAD -> STT -> User Context -> Groq LLM -> TTS -> Audio Output
    pipeline = Pipeline([
        transport.input(),              # Captures mic frames
        vad_processor,                  # Analyzes speech & emits start/stop events
        stt,                            # Transcribes audio to text frames
        context_pair.user(),            # Aggregates user transcripts into context
        lifecycle_logger,               # Displays [USER TRANSCRIPT] & [BOT THINKING]
        llm,                            # Streams response tokens from Groq
        tts,                            # Synthesizes speech from tokens
        transport.output(),             # Streams audio frames to speakers
        context_pair.assistant(),       # Aggregates assistant response into context
    ])

    # 8. Pipeline Task with Automatic Interruption (Barge-in < 250ms)
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,    # Crucial: enables instant speech interruption
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    runner = PipelineRunner()

    # Handle graceful exit on Ctrl+C (SIGINT)
    stop_event = asyncio.Event()

    def handle_signal():
        print(f"\n{YELLOW}[SHUTTING DOWN] Gracefully stopping voice agent...{RESET}")
        stop_event.set()
        asyncio.create_task(task.cancel())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            # Windows workaround for add_signal_handler
            pass

    print(f"{GREEN}{BOLD}Pipeline assembled successfully!{RESET}")
    print(f"{MAGENTA}{BOLD}[LISTENING]{RESET} (Speak to assistant, or interrupt at any time. Press Ctrl+C to exit)\n")

    try:
        await runner.run(task)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"\n{RED}{BOLD}[PIPELINE ERROR]{RESET} {e}")
    finally:
        print(f"{GREEN}Voice agent session ended.{RESET}")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    try:
        asyncio.run(run_voice_agent())
    except KeyboardInterrupt:
        print(f"\n{GREEN}Agent stopped by user.{RESET}")


if __name__ == "__main__":
    main()
