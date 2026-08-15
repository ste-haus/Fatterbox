#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tts-webui-chatterbox-tts @ git+https://github.com/rsxdalv/chatterbox.git@faster",
#     "torch",
#     "torchaudio",
#     "transformers>=4.46,<5",
#     "torchcodec",
# ]
# ///
"""Generate a TTS audio file from a voice and text input.

Accepts both .wav (reference audio) and .pt (pre-conditioned) voice files.

Usage:

    uv run --no-project scripts/test_voice.py voices/Jake.wav "Hello, this is a test."
    uv run --no-project scripts/test_voice.py voices/Jake.pt "Hello, this is a test."
    uv run --no-project scripts/test_voice.py voices/Jake.pt "Hello!" -o custom_output.wav
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS, Conditionals

DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5


def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio from a voice file and text")
    parser.add_argument("voice", type=Path, help="Path to voice file (.wav or .pt)")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.wav"),
                        help="Output .wav path (default: output.wav)")
    parser.add_argument("-e", "--exaggeration", type=float, default=DEFAULT_EXAGGERATION,
                        help=f"Emotional expressiveness (default: {DEFAULT_EXAGGERATION}, ignored for .pt voices)")
    parser.add_argument("-c", "--cfg-weight", type=float, default=DEFAULT_CFG_WEIGHT,
                        help=f"Voice adherence/pacing (default: {DEFAULT_CFG_WEIGHT})")
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to run on (default: cpu)")
    args = parser.parse_args()

    if not args.voice.exists():
        print(f"Error: {args.voice} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model on {args.device}...")
    model = ChatterboxTTS.from_pretrained(device=args.device)

    backend = "eager" if args.device == "cpu" else "cudagraphs-manual"
    t3_params = {"generate_token_backend": backend}

    print(f"Generating speech: \"{args.text}\"")
    start = time.time()

    if args.voice.suffix == ".pt":
        print(f"Using conditioned voice: {args.voice.name}")
        model.conds = Conditionals.load(str(args.voice), map_location=args.device)
        wav = model.generate(
            args.text,
            cfg_weight=args.cfg_weight,
            t3_params=t3_params,
        )
    else:
        print(f"Using reference audio: {args.voice.name} (exaggeration={args.exaggeration})")
        wav = model.generate(
            args.text,
            audio_prompt_path=str(args.voice),
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            t3_params=t3_params,
        )

    elapsed = time.time() - start
    print(f"Generated in {elapsed:.2f}s")

    torchaudio.save(str(args.output), wav.squeeze().unsqueeze(0).cpu(), model.sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
