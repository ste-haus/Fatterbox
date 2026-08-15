## Overview

Fatterbox is built on [rsxdalv's optimized Chatterbox implementation](https://github.com/rsxdalv/chatterbox/tree/faster), exposing both Wyoming protocol and OpenAPI endpoints with streaming support for minimal time-to-first-word latency. The streaming architecture splits text into sentence chunks and generates audio progressively, allowing playback to begin before the entire text is synthesized.

## Requirements

- Docker with NVIDIA GPU support ([install nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- NVIDIA GPU with CUDA capability
- Voice files: `.mp3` (source audio), `.wav` (reference audio), and/or `.pt` (pre-conditioned)

## Quick Start

1. **Prepare voice files**: Place voice files in a `voices` directory. Each file becomes a voice named after the file stem (e.g., `Jake.wav` or `Jake.pt` → voice name "Jake").

   - **`.mp3`** — Source audio. Not usable directly; must be transcoded to `.wav` first (see preconditioning below).
   - **`.wav`** — Reference audio. Speaker conditioning is extracted from the audio on each generation call.
   - **`.pt`** — Pre-conditioned voice (a serialized Chatterbox `Conditionals` object). Faster at generation time since conditioning is pre-computed. If both `Jake.wav` and `Jake.pt` exist, the `.pt` takes precedence.

### Preconditioning

   Set `FATTERBOX_PRECONDITION_ON_START=true` (or pass `--precondition-on-start`) to bring the voices directory up to date at startup, before either server accepts traffic. **Off by default.** The pipeline is `.mp3` → `.wav` → `.pt`:

   - Every `.mp3` without a current `.wav` is transcoded with `ffmpeg` to mono 16-bit PCM at the model's native sample rate (24 kHz), which is the rate Chatterbox reads reference audio at — so nothing is resampled twice and no detail is discarded.
   - Every `.wav` without a matching `.pt` is conditioned in place.
   - A derived file older than its source is regenerated. The superseded file is renamed to `Jake.pt.YYYYMMDD` (or `Jake.wav.YYYYMMDD`), stamped with the date it was originally generated, and is ignored by voice discovery from then on. Touching a `.mp3` therefore cascades through both stages.

   Conditioning reuses the already-loaded model, so it costs no extra VRAM — but it does add a few seconds of startup time per new voice. Each stage writes to a `.partial` file and renames on success, so an interrupted or failed run never leaves a truncated `.wav`/`.pt` behind, and a voice always keeps its last working file. Failures are logged and skipped; they never block startup.

   > **Note:** `ffmpeg` must be on `PATH` for `.mp3` transcoding. The Docker image already includes it.

   The startup pass bakes in the current `FATTERBOX_EXAGGERATION` value. To condition a voice out-of-band with a different value, use the included `scripts/condition_voice.py` script (dependencies are declared inline — `uv` handles the rest):
   ```bash
   # Generate conditioned voice (outputs voices/Jake.pt)
   uv run --no-project scripts/condition_voice.py voices/Jake.wav

   # With custom exaggeration (default: 0.5)
   uv run --no-project scripts/condition_voice.py voices/Jake.wav -e 0.8

   # Custom output path
   uv run --no-project scripts/condition_voice.py voices/Jake.wav -o voices/alt-Jake.pt
   ```
   > **Note:** The `exaggeration` value is baked into the `.pt` file at creation time. The runtime `FATTERBOX_EXAGGERATION` setting has no effect on `.pt` voices.

2. **Pull the prebuilt image** (or build your own with `docker build -t fatterbox .`):
```bash
docker pull docker.io/justinlime/fatterbox:v0.1.0
```

3. **Run the container**:
```bash
docker run --gpus all \
  -v ./voices:/chatter/voices \
  -p 10200:10200 \
  -p 8000:8000 \
  docker.io/justinlime/fatterbox:v0.1.0
```

## Servers

Two servers run simultaneously:

- **Wyoming protocol**: `tcp://0.0.0.0:10200` (Home Assistant integration)
- **OpenAPI REST**: `http://0.0.0.0:8000` (OpenAI-compatible)

## Configuration

Configure via environment variables (all prefixed with `FATTERBOX_`):

### Server Configuration
```bash
FATTERBOX_WYOMING_HOST=0.0.0.0
FATTERBOX_WYOMING_PORT=10200
FATTERBOX_OPENAPI_HOST=0.0.0.0
FATTERBOX_OPENAPI_PORT=8000
FATTERBOX_VOICES_DIR=./voices
FATTERBOX_PRECONDITION_ON_START=false # transcode .mp3 and generate missing/outdated .pt at startup
```

### Model Configuration
```bash
FATTERBOX_DEVICE=cuda              # cuda or cpu
FATTERBOX_DTYPE=bf16               # float32, fp16, bf16 (bf16 recommended)
FATTERBOX_BACKEND=cudagraphs-manual # fastest option
```

**Minimum VRAM Required:**
- FP32: ~4.5 GB
- FP16/BF16: ~3.5 GB (recommended)

Estimates based on my tests with a RTX 3090. You might experience memory spikes if generating
large sentences without punctuations.
BF16 offers the best balance of speed and memory efficiency on modern GPUs.

### Generation Parameters
```bash
FATTERBOX_EXAGGERATION=0.5      # Emotional expressiveness (0.0-2.0)
FATTERBOX_CFG_WEIGHT=0.5        # Voice adherence (0.0-1.0)
FATTERBOX_TEMPERATURE=0.8       # Randomness (0.05-5.0)
FATTERBOX_SEED=0                # Random seed (0=random)
FATTERBOX_TOP_P=1.0             # Nucleus sampling (0.0-1.0)
FATTERBOX_MIN_P=0.0             # Min probability (0.0-1.0)
FATTERBOX_MAX_NEW_TOKENS=4096   # Max audio tokens (~25 per second)
FATTERBOX_N_TIMESTEPS=10        # Diffusion steps
FATTERBOX_FLOW_CFG_SCALE=1.0    # Mel decoder CFG scale
FATTERBOX_DEBUG=false           # Enable debug logging
```

### Example with custom settings:
```bash
docker run --gpus all \
  -v ./voices:/chatter/voices \
  -p 10200:10200 \
  -p 8000:8000 \
  -e FATTERBOX_DTYPE=bf16 \
  -e FATTERBOX_EXAGGERATION=0.7 \
  -e FATTERBOX_CFG_WEIGHT=0.4 \
  fatterbox
```

## API Usage

### OpenAPI Endpoint
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test.",
    "voice": "Jake"
  }' \
  --output speech.wav
```

### List Available Voices
```bash
curl http://localhost:8000/v1/voices
```

## Wyoming Protocol

Compatible with Home Assistant's Wyoming protocol. Configure in Home Assistant using:
- Host: `<docker-host-ip>`
- Port: `10200`

## Performance Tips

- Use `bf16` dtype (recommended) for best balance of speed and VRAM efficiency
- RTX 30xx/40xx GPUs have native BF16 support for optimal performance
- Use `cudagraphs-manual` backend (default) for fastest generation
- Lower `EXAGGERATION` and `CFG_WEIGHT` for more expressive speech
