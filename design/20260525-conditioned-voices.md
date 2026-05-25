# Support pre-conditioned voice files (.pt) alongside .wav references

**Date**: 2026-05-25
**Status**: Approved

## Context

Fatterbox currently only supports `.wav` reference audio files for Chatterbox TTS voice cloning. Each request re-extracts speaker conditioning from the .wav at generation time. Chatterbox also supports pre-computed "conditionals" — a `Conditionals` object containing speaker embeddings, prompt tokens, and emotion data — saved as `.pt` files via `torch.save()`. Loading a `.pt` is faster than re-extracting from audio every call.

The goal is to support both formats: drop either a `Jake.wav` or `Jake.pt` into the `voices/` directory and reference it as voice `"Jake"`. If both exist for the same name, `.pt` takes precedence (it's pre-conditioned and skips extraction).

## Terminology

- Chatterbox calls these **"conditionals"** (the `Conditionals` class from `chatterbox.tts`)
- The `.pt` extension is the standard PyTorch tensor format and matches Chatterbox's internal convention (`conds.pt`)
- "Pretrained" is incorrect — nothing is trained. "Conditioned voice" or "pre-conditioned voice" is the accurate term

## Key API details

- `model.prepare_conditionals(wav_path, exaggeration=0.5)` extracts conditioning from a .wav and stores it in `model.conds`
- `model.conds.save(path)` serializes the Conditionals object to a `.pt` file
- `Conditionals.load(path, map_location=device)` deserializes it back
- `model.generate(text, audio_prompt_path=...)` internally calls `prepare_conditionals()` for .wav files
- For .pt files: load into `model.conds` directly, then call `generate()` **without** `audio_prompt_path`

**Caveat**: The `exaggeration` parameter is baked into the `.pt` at creation time (stored as `emotion_adv` tensor). For `.pt` voices, the runtime `exaggeration` setting has no effect.

## Changes

### 1. `fatterbox/voices.py` — scan for `.pt` files alongside `.wav`

- `load_voices()`: after scanning `*.wav`, also scan `*.pt`. If both exist for the same stem, `.pt` wins. Log which type was loaded for each voice.
- `create_wyoming_info()`: same scan logic for building the Wyoming voice list. Update the description to distinguish "Cloned voice" vs "Conditioned voice".

### 2. `fatterbox/handler.py` — handle `.pt` in `_generate_audio()`

- Check if the resolved voice path ends with `.pt`
- If `.pt`: load `Conditionals` from the file, assign to `model.conds`, call `generate()` **without** `audio_prompt_path`
- If `.wav`: existing behavior (pass `audio_prompt_path=`)
- Import `Conditionals` from `chatterbox.tts`
- Log a debug note when using a conditioned voice that exaggeration is baked-in

### 3. `fatterbox/openapi.py` — same `.pt` handling in `_generate_audio_sync()`

Mirror the handler.py changes in the standalone sync generation function.

### 4. `README.md` — update documentation

- Update "Prepare voice files" to mention both `.wav` and `.pt` formats
- Explain that `.pt` files are pre-conditioned voices (created via `model.prepare_conditionals()` + `model.conds.save()`)
- Note the exaggeration caveat for `.pt` files
- Update requirements line to mention both formats

## Verification

1. Confirm `Conditionals` is importable from `chatterbox.tts` (grep the installed package or check import)
2. Place a `.wav` in voices/ — confirm it still works as before
3. Generate a `.pt` from a `.wav` using `prepare_conditionals()` + `conds.save()` — confirm it loads and generates speech
4. Place both `Jake.wav` and `Jake.pt` — confirm `.pt` takes precedence and logs accordingly
5. Hit both Wyoming and OpenAPI endpoints with a `.pt` voice
