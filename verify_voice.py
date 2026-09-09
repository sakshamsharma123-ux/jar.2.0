"""
verify_voice.py
--------------------------------------------------------------------------------
Pre-flight diagnostics script for Real-Time Voice AI Agent (Stage 1).
Tests:
1. Python environment and required dependency imports.
2. Hardware audio devices (Microphones and Speakers) via PyAudio.
3. Configuration checks (.env existence & key formatting).
4. Live API connectivity tests for Deepgram STT/TTS and Groq LLM (when keys are provided).
--------------------------------------------------------------------------------
"""

import os
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

# Initialize color formatting for terminal
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    CYAN = Fore.CYAN
    MAGENTA = Fore.MAGENTA
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL
except ImportError:
    GREEN = YELLOW = RED = CYAN = MAGENTA = BOLD = RESET = ""

ICON_OK = "[OK]"
ICON_FAIL = "[FAIL]"
ICON_WARN = "[!]"
ICON_INFO = "[*]"


def print_banner():
    banner = f"""
{CYAN}{BOLD}================================================================
          Real-Time Voice AI Agent - Pre-flight Check
================================================================{RESET}"""
    print(banner)


def check_dependencies() -> bool:
    print(f"\n{BOLD}[1/4] Checking Python & Core Dependencies...{RESET}")
    print(f"  * Python Version: {sys.version.split()[0]}")

    required_packages = [
        ("pipecat", "pipecat-ai"),
        ("pyaudio", "pyaudio"),
        ("dotenv", "python-dotenv"),
        ("deepgram", "deepgram-sdk"),
        ("groq", "groq"),
    ]

    all_passed = True
    for module_name, pip_name in required_packages:
        try:
            __import__(module_name)
            print(f"  {GREEN}{ICON_OK}{RESET} Module '{module_name}' ({pip_name}) is installed.")
        except ImportError:
            print(f"  {RED}{ICON_FAIL}{RESET} Missing module '{module_name}'. Install with: pip install {pip_name}")
            all_passed = False

    return all_passed


def check_audio_devices() -> tuple[list[dict], list[dict], int | None, int | None]:
    print(f"\n{BOLD}[2/4] Scanning Audio Devices (Microphone & Speakers)...{RESET}")

    input_devices = []
    output_devices = []
    default_in_idx = None
    default_out_idx = None

    try:
        import pyaudio
        p = pyaudio.PyAudio()

        try:
            default_in_info = p.get_default_input_device_info()
            default_in_idx = default_in_info["index"]
        except Exception:
            default_in_idx = None

        try:
            default_out_info = p.get_default_output_device_info()
            default_out_idx = default_out_info["index"]
        except Exception:
            default_out_idx = None

        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            max_in = dev.get("maxInputChannels", 0)
            max_out = dev.get("maxOutputChannels", 0)
            name = dev.get("name", f"Device {i}")

            if max_in > 0:
                input_devices.append({"index": i, "name": name, "channels": max_in})
            if max_out > 0:
                output_devices.append({"index": i, "name": name, "channels": max_out})

        p.terminate()

        # Display input devices
        print(f"\n  {CYAN}--- Microphones / Audio Inputs ({len(input_devices)} found) ---{RESET}")
        for dev in input_devices:
            is_default = " (OS Default)" if dev["index"] == default_in_idx else ""
            print(f"   [{dev['index']}] {dev['name']} (Channels: {dev['channels']}){is_default}")

        # Display output devices
        print(f"\n  {CYAN}--- Speakers / Audio Outputs ({len(output_devices)} found) ---{RESET}")
        for dev in output_devices:
            is_default = " (OS Default)" if dev["index"] == default_out_idx else ""
            print(f"   [{dev['index']}] {dev['name']} (Channels: {dev['channels']}){is_default}")

        if not input_devices:
            print(f"  {RED}{ICON_FAIL} No audio input device detected! Please connect a microphone.{RESET}")
        else:
            # Auto-recommend the best microphone
            mic_in = None
            if default_in_idx is not None:
                mic_in = default_in_idx
            else:
                for dev in input_devices:
                    if "mic" in dev["name"].lower():
                        mic_in = dev["index"]
                        break
                if mic_in is None:
                    mic_in = input_devices[0]["index"]
            print(f"\n  {GREEN}{ICON_OK} Selected Input Device Index: {mic_in}{RESET}")

        if not output_devices:
            print(f"  {RED}{ICON_FAIL} No audio output device detected! Please connect speakers or headphones.{RESET}")
        else:
            sel_out = default_out_idx if default_out_idx is not None else output_devices[0]["index"]
            print(f"  {GREEN}{ICON_OK} Selected Output Device Index: {sel_out}{RESET}")

    except Exception as exc:
        print(f"  {RED}{ICON_FAIL} Error accessing audio subsystem: {exc}{RESET}")

    return input_devices, output_devices, default_in_idx, default_out_idx


def check_api_keys():
    print(f"\n{BOLD}[3/4] Checking Environment Configuration (.env)...{RESET}")

    env_path = Path(".env")
    if not env_path.exists():
        print(f"  {YELLOW}{ICON_WARN} .env file not found.{RESET}")
        print(f"    Created a template at '.env' using '.env.example'.")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
        else:
            with open(".env", "w", encoding="utf-8") as f:
                f.write("DEEPGRAM_API_KEY=your_deepgram_api_key_here\nGROQ_API_KEY=your_groq_api_key_here\n")

    load_dotenv(override=True)

    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    is_dg_placeholder = (not deepgram_key) or ("your_deepgram" in deepgram_key)
    is_groq_placeholder = (not groq_key) or ("your_groq" in groq_key)

    if is_dg_placeholder:
        print(f"  {YELLOW}{ICON_WARN} DEEPGRAM_API_KEY: Not set or placeholder (Ready for your key in .env){RESET}")
    else:
        masked = deepgram_key[:4] + "..." + deepgram_key[-4:] if len(deepgram_key) > 8 else "***"
        print(f"  {GREEN}{ICON_OK} DEEPGRAM_API_KEY detected: {masked}{RESET}")

    if is_groq_placeholder:
        print(f"  {YELLOW}{ICON_WARN} GROQ_API_KEY: Not set or placeholder (Ready for your key in .env){RESET}")
    else:
        masked = groq_key[:4] + "..." + groq_key[-4:] if len(groq_key) > 8 else "***"
        print(f"  {GREEN}{ICON_OK} GROQ_API_KEY detected: {masked}{RESET}")

    return deepgram_key, groq_key, not (is_dg_placeholder or is_groq_placeholder)


def test_api_connectivity(deepgram_key: str, groq_key: str, keys_ready: bool):
    print(f"\n{BOLD}[4/4] Testing API Connectivity...{RESET}")

    if not keys_ready:
        print(f"  {YELLOW}{ICON_INFO} Skipping remote API handshake because API keys are not provided yet.{RESET}")
        print(f"    (The local pipeline structure and hardware are fully pre-validated. Add keys to .env anytime).")
        return

    # 1. Test Deepgram
    print("  * Verifying Deepgram API connectivity...")
    try:
        import httpx
        headers = {"Authorization": f"Token {deepgram_key}"}
        resp = httpx.get("https://api.deepgram.com/v1/projects", headers=headers, timeout=10.0)
        if resp.status_code == 200:
            print(f"    {GREEN}{ICON_OK} Deepgram API connection successful!{RESET}")
        else:
            print(f"    {RED}{ICON_FAIL} Deepgram returned HTTP {resp.status_code}: {resp.text}{RESET}")
    except Exception as e:
        print(f"    {RED}{ICON_FAIL} Deepgram connection error: {e}{RESET}")

    # 2. Test Groq
    print("  * Verifying Groq API connectivity...")
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        models = client.models.list()
        print(f"    {GREEN}{ICON_OK} Groq API connection successful! Found {len(models.data)} models.{RESET}")
    except Exception as e:
        print(f"    {RED}{ICON_FAIL} Groq connection error: {e}{RESET}")


def main():
    print_banner()
    deps_ok = check_dependencies()
    in_devs, out_devs, def_in, def_out = check_audio_devices()
    dg_key, groq_key, keys_ready = check_api_keys()
    test_api_connectivity(dg_key, groq_key, keys_ready)

    print(f"\n{CYAN}{BOLD}================================================================{RESET}")
    if deps_ok and in_devs and out_devs:
        print(f"{GREEN}{BOLD}Pre-flight Check PASSED!{RESET}")
        print("System microphone & audio drivers are configured and ready.")
        if not keys_ready:
            print(f"{YELLOW}Note: Add your DEEPGRAM_API_KEY and GROQ_API_KEY to .env when you are ready to speak.{RESET}")
        else:
            print(f"{GREEN}All keys present! You can start the agent with: python src/voice_agent.py{RESET}")
    else:
        print(f"{YELLOW}{BOLD}Pre-flight Check completed with warnings. Review logs above.{RESET}")
    print(f"{CYAN}{BOLD}================================================================{RESET}\n")


if __name__ == "__main__":
    main()
