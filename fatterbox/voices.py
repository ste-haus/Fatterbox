"""Voice discovery and Wyoming info generation."""
import logging
from pathlib import Path

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice

_LOGGER = logging.getLogger(__name__)


def load_voices(voices_dir: Path) -> dict:
    """Scan voices directory and return mapping of voice name -> file path.

    Supports .wav (reference audio) and .pt (pre-conditioned) files.
    When both exist for the same voice name, .pt takes precedence.
    """
    voices = {}

    if not voices_dir.exists():
        _LOGGER.warning(f"Voices directory does not exist: {voices_dir}")
        return voices

    for wav_file in voices_dir.glob("*.wav"):
        voice_name = wav_file.stem
        voices[voice_name] = str(wav_file)
        _LOGGER.info(f"Loaded voice: {voice_name} (reference audio: {wav_file.name})")

    for pt_file in voices_dir.glob("*.pt"):
        voice_name = pt_file.stem
        if voice_name in voices:
            _LOGGER.info(f"Voice '{voice_name}': .pt overrides .wav (using conditioned voice)")
        else:
            _LOGGER.info(f"Loaded voice: {voice_name} (conditioned: {pt_file.name})")
        voices[voice_name] = str(pt_file)

    if not voices:
        _LOGGER.warning("No voice files found in voices directory")

    return voices


def create_wyoming_info(voices_dir: Path, sample_rate: int) -> Info:
    """Create Wyoming Info with available voices."""
    voices = []
    
    # Discover voices from directory (.wav and .pt, with .pt taking precedence)
    if voices_dir.exists():
        voice_files = {}

        for wav_file in voices_dir.glob("*.wav"):
            voice_files[wav_file.stem] = wav_file

        for pt_file in voices_dir.glob("*.pt"):
            voice_files[pt_file.stem] = pt_file

        for voice_name, voice_file in voice_files.items():
            is_conditioned = voice_file.suffix == ".pt"
            description = (
                f"Conditioned voice from {voice_file.name}"
                if is_conditioned
                else f"Cloned voice from {voice_file.name}"
            )
            voices.append(
                TtsVoice(
                    name=voice_name,
                    description=description,
                    attribution=Attribution(
                        name="Chatterbox",
                        url="https://github.com/resemble-ai/chatterbox"
                    ),
                    installed=True,
                    version="1.0",
                    languages=["en"],
                )
            )
    
    return Info(
        tts=[
            TtsProgram(
                name="chatterbox",
                description="Chatterbox TTS with voice cloning",
                attribution=Attribution(
                    name="Resemble AI",
                    url="https://github.com/resemble-ai/chatterbox"
                ),
                installed=True,
                version="1.0",
                voices=voices,
                supports_synthesize_streaming=True,  # Enable streaming support!
            )
        ]
    )
