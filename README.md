# Fatterbox

Self-hosted text-to-speech with voice cloning, speaking both the **Wyoming protocol** (for Home Assistant) and an **OpenAI-compatible HTTP API** — from a single process, on one GPU.

Built on [rsxdalv's optimized Chatterbox implementation](https://github.com/rsxdalv/chatterbox/tree/faster).

- **Streaming by default** — text is split into sentences and synthesized progressively, so playback starts before the full passage is rendered. Minimal time-to-first-word.
- **Clone a voice from a single audio file** — drop a `.wav` or `.mp3` in a directory and it becomes a named voice.
- **Precondition for speed** — cache the expensive part of cloning to disk, so it isn't recomputed on every request.

---

## Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Voices](#voices)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Home Assistant](#home-assistant)
- [Performance](#performance)

---

## Requirements

- An NVIDIA GPU with CUDA capability, and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- Docker
- At least one voice file (see [Voices](#voices))

**VRAM**

| Precision | Minimum |
|---|---|
| FP32 | ~4.5 GB |
| FP16 / BF16 | ~3.5 GB *(recommended)* |

Measured on an RTX 3090. Expect spikes when generating long sentences without punctuation, since chunking has nothing to split on.

---

## Quick Start

**1. Add a voice.** Any clear speech sample works — a few seconds is enough.

```bash
mkdir -p voices
cp ~/somewhere/Jake.wav voices/
```

**2. Run it.**

```bash
docker run --gpus all \
  -v ./voices:/chatter/voices \
  -p 10200:10200 \
  -p 8000:8000 \
  ghcr.io/ste-haus/fatterbox:latest
```

**3. Say something.**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, this is a test.", "voice": "Jake"}' \
  --output speech.wav
```

That's it — `Jake.wav` became the voice `"Jake"`. Two servers are now listening:

| Server | Address | For |
|---|---|---|
| Wyoming | `tcp://0.0.0.0:10200` | Home Assistant |
| HTTP | `http://0.0.0.0:8000` | OpenAI-compatible clients, plus `/docs` |

To build the image yourself instead: `docker build -t fatterbox .`

---

## Voices

Every file in the voices directory becomes a voice **named after its stem** — `Jake.wav` → `"Jake"`.

### File types

| Extension | What it is | Notes |
|---|---|---|
| `.mp3` | Source audio | Must be transcoded to `.wav` before use — see [Preconditioning](#preconditioning) |
| `.wav` | Reference audio | Usable directly. Conditioning is re-extracted on **every** generation call |
| `.pt` | Preconditioned voice | A serialized Chatterbox `Conditionals`. Conditioning already done — the fast path |

When both `Jake.wav` and `Jake.pt` exist, the `.pt` wins and the `.wav` is kept so the `.pt` can be regenerated from it.

```
voices/
  Jake.wav          reference audio
  Jake.pt           conditioned — used for generation
  archived/         superseded files, ignored by discovery
```

> **Why preconditioning is worth it.** `.wav` generation re-extracts the speaker conditioning on every call — and because streaming synthesizes sentence by sentence, a five-sentence request conditions five times. A `.pt` does it zero times.

### One `.pt` covers every exaggeration

Conditioning derives the speaker embedding and prompt tokens from the audio; none of that depends on exaggeration. Exaggeration is stored separately as a single `emotion_adv` scalar, which Chatterbox swaps per request.

So a voice never needs more than one `.pt`, and the value it was conditioned at is **informational only** — it's logged at startup and reported by `/v1/voices`, but it doesn't limit what a request can ask for.

### Preconditioning

Set `FATTERBOX_PRECONDITION_ON_START=true` to bring the voices directory up to date at startup, before either server accepts traffic. **Off by default.**

The pipeline is `.mp3` → `.wav` → `.pt`:

1. Every `.mp3` without a current `.wav` is transcoded with `ffmpeg` to **mono 16-bit PCM at 24 kHz** — the model's native rate and the rate Chatterbox reads reference audio at, so nothing is resampled twice.
2. Every `.wav` without a matching `.pt` is conditioned in place.
3. Any derived file older than its source is regenerated, and the file it replaces is moved to `voices/archived/` with a date stamp. Touching a `.mp3` cascades through both stages.

It reuses the already-loaded model, so it costs no extra VRAM and is a handful of forward passes over a short clip — not another model load. Elapsed time is logged per voice.

Each stage writes to a `.partial` file and renames on success, so an interrupted run never leaves a truncated `.wav` or `.pt`, and a voice always keeps its last working file. Failures are logged and skipped; they never block startup.

> `ffmpeg` must be on `PATH` for `.mp3` transcoding. The Docker image includes it.

### Adding voices without a restart

Voices are discovered once at startup. To pick up files added since:

```bash
curl -X POST http://localhost:8000/v1/voices/reload
```

Call it **after the file finishes copying.** Fatterbox deliberately does not watch the directory: a partially copied `.wav` is indistinguishable from a complete one, and conditioning it would write a corrupt `.pt` that then takes precedence over the `.wav`. Your calling the endpoint *is* the signal that the file is whole.

### Conditioning by hand

`scripts/condition_voice.py` produces a `.pt` out-of-band. Dependencies are declared inline, so `uv` handles the rest:

```bash
uv run --no-project scripts/condition_voice.py voices/Jake.wav
uv run --no-project scripts/condition_voice.py voices/Jake.wav -o voices/alt-Jake.pt
```

> `-e/--exaggeration` sets the `emotion_adv` value stored in the file, but Chatterbox overrides it per request, so it has no effect on output. It exists for parity with the on-disk format.

---

## Configuration

Everything is an environment variable prefixed with `FATTERBOX_`. Each also has a matching `--kebab-case` CLI flag.

### Servers

| Variable | Default | Description |
|---|---|---|
| `FATTERBOX_WYOMING_HOST` | `0.0.0.0` | Wyoming bind address |
| `FATTERBOX_WYOMING_PORT` | `10200` | Wyoming port |
| `FATTERBOX_OPENAPI_HOST` | `0.0.0.0` | HTTP bind address |
| `FATTERBOX_OPENAPI_PORT` | `8000` | HTTP port |
| `FATTERBOX_VOICES_DIR` | `./voices` | Where voice files live |
| `FATTERBOX_PRECONDITION_ON_START` | `false` | Transcode `.mp3` and generate missing `.pt` at startup |
| `FATTERBOX_DEBUG` | `false` | Debug logging |

### Model

| Variable | Default | Description |
|---|---|---|
| `FATTERBOX_DEVICE` | `cuda` | `cuda` or `cpu` |
| `FATTERBOX_DTYPE` | `float32` | `float32`, `fp16`, `bf16` — **`bf16` recommended** |
| `FATTERBOX_BACKEND` | `cudagraphs-manual` | Generation backend; the default is fastest |

### Generation

| Variable | Default | Range | Description |
|---|---|---|---|
| `FATTERBOX_EXAGGERATION` | `0.5` | 0.0–2.0 | Emotional expressiveness |
| `FATTERBOX_CFG_WEIGHT` | `0.5` | 0.0–1.0 | Voice adherence and pacing |
| `FATTERBOX_TEMPERATURE` | `0.8` | 0.05–5.0 | Randomness |
| `FATTERBOX_SEED` | `0` | | Random seed; `0` means random |
| `FATTERBOX_TOP_P` | `1.0` | 0.0–1.0 | Nucleus sampling |
| `FATTERBOX_MIN_P` | `0.0` | 0.0–1.0 | Minimum probability |
| `FATTERBOX_MAX_NEW_TOKENS` | `4096` | | Max audio tokens (~25 per second) |
| `FATTERBOX_N_TIMESTEPS` | `10` | | Diffusion steps for flow matching |
| `FATTERBOX_FLOW_CFG_SCALE` | `1.0` | | Mel decoder CFG scale |

### Example

```bash
docker run --gpus all \
  -v ./voices:/chatter/voices \
  -p 10200:10200 \
  -p 8000:8000 \
  -e FATTERBOX_DTYPE=bf16 \
  -e FATTERBOX_EXAGGERATION=0.7 \
  -e FATTERBOX_CFG_WEIGHT=0.4 \
  -e FATTERBOX_PRECONDITION_ON_START=true \
  ghcr.io/ste-haus/fatterbox:latest
```

---

## API Reference

Interactive docs are served at `http://localhost:8000/docs`.

### `POST /v1/audio/speech`

Compatible with OpenAI's endpoint. Responds with streaming WAV via chunked transfer encoding.

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | *required* | Text to synthesize. Also accepted as `text` |
| `voice` | string | first voice | Voice name |
| `exaggeration` | float | `FATTERBOX_EXAGGERATION` | Per-request expressiveness |
| `response_format` | string | `wav` | Only `wav` is supported |
| `model`, `speed` | | | Accepted for compatibility; ignored |

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello there.", "voice": "Jake", "exaggeration": 0.9}' \
  --output speech.wav
```

Any voice serves any exaggeration, `.pt` or `.wav`. Wyoming has no per-request exaggeration field, so Wyoming requests always use `FATTERBOX_EXAGGERATION`.

### `GET /v1/voices`

```json
{
  "voices": [
    {"name": "Jake", "exaggeration": 0.5},
    {"name": "Solo", "exaggeration": null}
  ]
}
```

`exaggeration` is what the `.pt` was conditioned at, or `null` for a `.wav`-only voice. Informational — see [One `.pt` covers every exaggeration](#one-pt-covers-every-exaggeration).

### `GET /v1/info`

Sample rate, plus every voice with its backing file path.

### `POST /v1/voices/reload`

Rescans the voices directory. See [Adding voices without a restart](#adding-voices-without-a-restart).

```json
{
  "voices": ["Jake", "Jess"],
  "added": ["Jess"],
  "removed": [],
  "conditioned": 1,
  "elapsed": 3.4
}
```

New files are conditioned only when `FATTERBOX_PRECONDITION_ON_START` is enabled; with it off, a dropped-in `.wav` still becomes a usable voice immediately, just without a `.pt`. Conditioning runs in a worker thread, so in-flight streams keep playing. Both servers pick up the result — Wyoming rebuilds its voice list per `Describe`, so no reconnect is needed.

> **Not part of the OpenAI spec:** `/v1/voices`, `/v1/voices/reload`, and `/v1/info` are Fatterbox conveniences — OpenAI has no voice-discovery endpoint. So are the `exaggeration` field and the `text` alias for `input`. Clients written against them won't port to other OpenAI-compatible servers.

---

## Home Assistant

Add the Wyoming integration and point it at Fatterbox:

- **Host:** your Docker host's IP
- **Port:** `10200`

Voices appear by name, and streaming is advertised so playback starts early.

---

## Performance

- **Use `bf16`.** Best balance of speed and VRAM on RTX 30xx/40xx, which have native BF16 support.
- **Keep `cudagraphs-manual`** (the default backend) — it's the fastest.
- **Precondition your voices.** The single biggest win for repeat requests; see [Preconditioning](#preconditioning).
- **Punctuate long text.** Chunking splits on sentence boundaries; without punctuation there's nothing to split on, which delays first audio and spikes VRAM.
- **Lower `EXAGGERATION` and `CFG_WEIGHT`** for more expressive, less literal delivery.
