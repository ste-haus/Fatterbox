"""Voice discovery, conditioning, and Wyoming info generation."""
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from chatterbox.tts import Conditionals
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice

from .model import MODEL_LOCK

_LOGGER = logging.getLogger(__name__)

SOURCE_SUFFIX = ".mp3"
REFERENCE_SUFFIX = ".wav"
CONDITIONED_SUFFIX = ".pt"
PARTIAL_SUFFIX = ".partial"
ARCHIVE_DIRNAME = "archived"
ARCHIVE_DATE_FORMAT = "%Y%m%d"

SOURCE_GLOB = f"*{SOURCE_SUFFIX}"
REFERENCE_GLOB = f"*{REFERENCE_SUFFIX}"
CONDITIONED_GLOB = f"*{CONDITIONED_SUFFIX}"

# Conditioning stores the exaggeration as the T3 emotion_adv tensor. Everything
# else it derives comes from the reference audio, so one .pt serves every
# exaggeration -- generate() swaps that one tensor per request.
BAKED_EXAGGERATION_ATTR = "emotion_adv"
CONDITIONING_DEVICE = "cpu"

# Chatterbox reads its reference audio as mono; transcoding to the model's own
# sample rate makes its internal resample a no-op without discarding detail.
FFMPEG_BINARY = "ffmpeg"
TRANSCODE_FORMAT = "wav"
TRANSCODE_CODEC = "pcm_s16le"
TRANSCODE_CHANNELS = 1


class VoiceRegistry:
    """Holds the current voice snapshot behind the mapping methods callers use.

    Reloading rebinds the whole dict rather than mutating it, so a reader
    iterating one generation never sees it change underneath.
    """

    def __init__(self, voices: Optional[dict] = None):
        self.voices = voices if voices is not None else {}

    def get(self, name):
        return self.voices.get(name)

    def keys(self):
        return self.voices.keys()

    def values(self):
        return self.voices.values()

    def items(self):
        return self.voices.items()

    def __contains__(self, name):
        return name in self.voices

    def __iter__(self):
        return iter(self.voices)

    def __len__(self):
        return len(self.voices)

    def replace(self, voices: dict):
        self.voices = voices


@dataclass
class Voice:
    """A discovered voice and the files backing it."""

    name: str
    path: str
    reference_path: Optional[str] = None
    exaggeration: Optional[float] = None

    @property
    def is_conditioned(self) -> bool:
        return self.path.endswith(CONDITIONED_SUFFIX)


def conditioned_path_for(wav_file: Path) -> Path:
    """Return the .pt path that pairs with a given .wav reference."""
    return wav_file.with_name(wav_file.stem + CONDITIONED_SUFFIX)


def reference_path_for(source_file: Path) -> Path:
    """Return the .wav path that pairs with a given .mp3 source."""
    return source_file.with_name(source_file.stem + REFERENCE_SUFFIX)


def read_baked_exaggeration(pt_file: Path) -> Optional[float]:
    """Return the exaggeration a .pt was conditioned with, or None if unreadable."""
    try:
        conds = Conditionals.load(str(pt_file), map_location=CONDITIONING_DEVICE)
        return float(getattr(conds.t3, BAKED_EXAGGERATION_ATTR).item())
    except Exception as e:
        _LOGGER.warning(f"Could not read baked exaggeration from {pt_file.name}: {e}")
        return None


def _archive_superseded(target: Path) -> Path:
    """Move a superseded file into the archive directory, stamped with the date it was generated.

    The archive sits one level down, so it is out of the way of the voice
    globs regardless of what it is named.
    """
    archive_dir = target.parent / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.fromtimestamp(target.stat().st_mtime).strftime(ARCHIVE_DATE_FORMAT)
    archive = archive_dir / f"{target.name}.{stamp}"

    collision = 1
    while archive.exists():
        archive = archive_dir / f"{target.name}.{stamp}-{collision}"
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
                _LOGGER.info(f"Archived superseded reference audio: {target.name} -> {ARCHIVE_DIRNAME}/{archive.name}")

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


def condition_reference(model, wav_file: Path, exaggeration: float, output: Path, replacing: bool) -> bool:
    """Condition a .wav reference and write the resulting .pt to output.

    Holds the model lock for the duration, since prepare_conditionals() mutates
    the conditionals that in-flight generation reads.
    """
    partial = output.with_name(output.name + PARTIAL_SUFFIX)

    try:
        started = time.time()

        with MODEL_LOCK:
            original_conds = getattr(model, "conds", None)
            try:
                model.prepare_conditionals(str(wav_file), exaggeration=exaggeration)
                model.conds.save(partial)
            finally:
                model.conds = original_conds

        elapsed = time.time() - started

        if replacing:
            archive = _archive_superseded(output)
            _LOGGER.info(f"Archived superseded conditioned voice: {output.name} -> {ARCHIVE_DIRNAME}/{archive.name}")

        partial.replace(output)

        _LOGGER.info(f"Conditioned '{wav_file.stem}': {wav_file.name} -> {output.name} in {elapsed:.2f}s")

        return True
    except Exception as e:
        _LOGGER.error(f"Failed to condition voice '{wav_file.stem}' from {wav_file.name}: {e}", exc_info=True)
        partial.unlink(missing_ok=True)
        return False


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

    _LOGGER.info(f"Conditioning {len(pending)} voice(s) at exaggeration {exaggeration}")

    started = time.time()
    conditioned = sum(1 for wav_file, output, replacing in pending
                      if condition_reference(model, wav_file, exaggeration, output, replacing))

    _LOGGER.info(f"Conditioned {conditioned}/{len(pending)} voice(s) in {time.time() - started:.2f}s")

    return conditioned


def load_voices(voices_dir: Path) -> dict:
    """Scan voices directory and return mapping of voice name -> Voice.

    Supports .wav (reference audio) and .pt (pre-conditioned) files.
    When both exist for the same voice name, .pt takes precedence and the .wav
    is retained so the .pt can be regenerated from it.
    """
    voices = {}

    if not voices_dir.exists():
        _LOGGER.warning(f"Voices directory does not exist: {voices_dir}")
        return voices

    references = {wav_file.stem: wav_file for wav_file in voices_dir.glob(REFERENCE_GLOB)}

    for voice_name, wav_file in references.items():
        voices[voice_name] = Voice(name=voice_name, path=str(wav_file), reference_path=str(wav_file))
        _LOGGER.info(f"Loaded voice: {voice_name} (reference audio: {wav_file.name})")

    for pt_file in voices_dir.glob(CONDITIONED_GLOB):
        voice_name = pt_file.stem
        reference = references.get(voice_name)
        baked = read_baked_exaggeration(pt_file)

        voices[voice_name] = Voice(
            name=voice_name,
            path=str(pt_file),
            reference_path=str(reference) if reference else None,
            exaggeration=baked,
        )

        baked_label = f"conditioned at exaggeration {baked}" if baked is not None else "conditioned, exaggeration unreadable"
        _LOGGER.info(f"Loaded voice: {voice_name} ({pt_file.name}, {baked_label})")

    if not voices:
        _LOGGER.warning("No voice files found in voices directory")

    return voices


def create_wyoming_info(voices: dict) -> Info:
    """Create Wyoming Info from the discovered voices."""
    tts_voices = []

    for voice in voices.values():
        source = Path(voice.path).name
        description = f"Conditioned voice from {source}" if voice.is_conditioned else f"Cloned voice from {source}"

        tts_voices.append(
            TtsVoice(
                name=voice.name,
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
                voices=tts_voices,
                supports_synthesize_streaming=True,  # Enable streaming support!
            )
        ]
    )


def reload_voices(voices_dir: Path, model, registry: VoiceRegistry, exaggeration: float, precondition: bool) -> dict:
    """Rescan the voices directory and swap the registry to the result.

    Blocking: conditioning holds the model lock and runs on the calling thread.
    Callers on an event loop must dispatch this to an executor.
    """
    before = set(registry.keys())

    started = time.time()
    conditioned = precondition_voices(voices_dir, model, exaggeration) if precondition else 0

    registry.replace(load_voices(voices_dir))

    after = set(registry.keys())
    summary = {
        "voices": sorted(after),
        "added": sorted(after - before),
        "removed": sorted(before - after),
        "conditioned": conditioned,
        "elapsed": round(time.time() - started, 2),
    }

    _LOGGER.info(f"Reloaded voices: {len(after)} total, added {summary['added']}, removed {summary['removed']}")

    return summary
