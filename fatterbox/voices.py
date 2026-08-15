"""Voice discovery, startup conditioning, and Wyoming info generation."""
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice

_LOGGER = logging.getLogger(__name__)

SOURCE_SUFFIX = ".mp3"
REFERENCE_SUFFIX = ".wav"
CONDITIONED_SUFFIX = ".pt"
PARTIAL_SUFFIX = ".partial"
ARCHIVE_DATE_FORMAT = "%Y%m%d"

SOURCE_GLOB = f"*{SOURCE_SUFFIX}"
REFERENCE_GLOB = f"*{REFERENCE_SUFFIX}"
CONDITIONED_GLOB = f"*{CONDITIONED_SUFFIX}"

# Chatterbox reads its reference audio as mono; transcoding to the model's own
# sample rate makes its internal resample a no-op without discarding detail.
FFMPEG_BINARY = "ffmpeg"
TRANSCODE_FORMAT = "wav"
TRANSCODE_CODEC = "pcm_s16le"
TRANSCODE_CHANNELS = 1


def conditioned_path_for(wav_file: Path) -> Path:
    """Return the .pt path that pairs with a given .wav reference."""
    return wav_file.with_name(wav_file.stem + CONDITIONED_SUFFIX)


def reference_path_for(source_file: Path) -> Path:
    """Return the .wav path that pairs with a given .mp3 source."""
    return source_file.with_name(source_file.stem + REFERENCE_SUFFIX)


def _archive_superseded(target: Path) -> Path:
    """Move a superseded file aside, stamped with the date it was generated.

    The archive name keeps the full original name, so the date lands after the
    extension and the archive is not picked up by the voice globs.
    """
    stamp = datetime.fromtimestamp(target.stat().st_mtime).strftime(ARCHIVE_DATE_FORMAT)
    archive = target.with_name(f"{target.name}.{stamp}")

    collision = 1
    while archive.exists():
        archive = target.with_name(f"{target.name}.{stamp}-{collision}")
        collision += 1

    target.rename(archive)

    return archive


def _outdated_against(candidates, derive_target) -> list:
    """Return (source, target, target_existed) for targets missing or older than their source."""
    pending = []

    for source in sorted(candidates):
        target = derive_target(source)

        if not target.exists():
            pending.append((source, target, False))
        elif source.stat().st_mtime > target.stat().st_mtime:
            pending.append((source, target, True))

    return pending


def _transcode_sources(voices_dir: Path, sample_rate: int) -> int:
    """Convert .mp3 voice sources into the .wav references the conditioner reads.

    Returns the number of references written. A source whose .wav is already
    current is skipped, so this is a no-op on every restart after the first.
    """
    pending = _outdated_against(voices_dir.glob(SOURCE_GLOB), reference_path_for)

    if not pending:
        return 0

    _LOGGER.info(f"Transcoding {len(pending)} {SOURCE_SUFFIX} source(s) to mono {sample_rate} Hz {REFERENCE_SUFFIX}")

    transcoded = 0

    for source, target, target_existed in pending:
        partial = target.with_name(target.name + PARTIAL_SUFFIX)

        command = [
            FFMPEG_BINARY,
            "-nostdin",
            "-loglevel", "error",
            "-y",
            "-i", str(source),
            "-ac", str(TRANSCODE_CHANNELS),
            "-ar", str(sample_rate),
            "-c:a", TRANSCODE_CODEC,
            "-f", TRANSCODE_FORMAT,
            str(partial),
        ]

        try:
            subprocess.run(command, check=True, capture_output=True)

            if target_existed:
                archive = _archive_superseded(target)
                _LOGGER.info(f"Archived superseded reference audio: {target.name} -> {archive.name}")

            partial.replace(target)

            transcoded += 1
            _LOGGER.info(f"Transcoded '{source.stem}': {source.name} -> {target.name}")
        except FileNotFoundError:
            _LOGGER.error(f"{FFMPEG_BINARY} not found on PATH — cannot transcode {SOURCE_SUFFIX} sources")
            partial.unlink(missing_ok=True)
            break
        except subprocess.CalledProcessError as e:
            _LOGGER.error(f"Failed to transcode '{source.stem}' from {source.name}: {e.stderr.decode(errors='replace').strip()}")
            partial.unlink(missing_ok=True)

    return transcoded


def precondition_voices(voices_dir: Path, model, exaggeration: float) -> int:
    """Bring every voice up to date: .mp3 -> .wav -> .pt.

    Runs at startup, before either server accepts traffic, so it can safely
    borrow the serving model. A derived file older than its source is
    regenerated and the superseded file archived. Returns the number of
    voices conditioned.
    """
    if not voices_dir.exists():
        return 0

    _transcode_sources(voices_dir, model.sr)

    pending = _outdated_against(voices_dir.glob(REFERENCE_GLOB), conditioned_path_for)

    if not pending:
        _LOGGER.info("All reference voices have an up-to-date conditioned file")
        return 0

    stale_count = sum(1 for _, _, is_stale in pending if is_stale)
    _LOGGER.info(
        f"Conditioning {len(pending)} voice(s) — {len(pending) - stale_count} new, {stale_count} outdated "
        f"(exaggeration={exaggeration}, baked in permanently)"
    )

    # prepare_conditionals() overwrites the model's active conditionals, which
    # would otherwise leak into requests that specify no voice.
    original_conds = getattr(model, "conds", None)
    conditioned = 0

    for wav_file, output, is_stale in pending:
        partial = output.with_name(output.name + PARTIAL_SUFFIX)

        try:
            model.prepare_conditionals(str(wav_file), exaggeration=exaggeration)

            # Write, archive, then rename, so neither a crash mid-write nor a
            # failed conditioning can leave the voice without a usable .pt.
            model.conds.save(partial)

            if is_stale:
                archive = _archive_superseded(output)
                _LOGGER.info(f"Archived superseded conditioned voice: {output.name} -> {archive.name}")

            partial.replace(output)

            conditioned += 1
            _LOGGER.info(f"Conditioned voice '{wav_file.stem}': {wav_file.name} -> {output.name}")
        except Exception as e:
            _LOGGER.error(f"Failed to condition voice '{wav_file.stem}' from {wav_file.name}: {e}", exc_info=True)
            partial.unlink(missing_ok=True)

    model.conds = original_conds

    return conditioned


def load_voices(voices_dir: Path) -> dict:
    """Scan voices directory and return mapping of voice name -> file path.

    Supports .wav (reference audio) and .pt (pre-conditioned) files.
    When both exist for the same voice name, .pt takes precedence.
    """
    voices = {}

    if not voices_dir.exists():
        _LOGGER.warning(f"Voices directory does not exist: {voices_dir}")
        return voices

    for wav_file in voices_dir.glob(REFERENCE_GLOB):
        voice_name = wav_file.stem
        voices[voice_name] = str(wav_file)
        _LOGGER.info(f"Loaded voice: {voice_name} (reference audio: {wav_file.name})")

    for pt_file in voices_dir.glob(CONDITIONED_GLOB):
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

        for wav_file in voices_dir.glob(REFERENCE_GLOB):
            voice_files[wav_file.stem] = wav_file

        for pt_file in voices_dir.glob(CONDITIONED_GLOB):
            voice_files[pt_file.stem] = pt_file

        for voice_name, voice_file in voice_files.items():
            is_conditioned = voice_file.suffix == CONDITIONED_SUFFIX
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
