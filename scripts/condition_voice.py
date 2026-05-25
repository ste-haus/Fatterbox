#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tts-webui-chatterbox-tts @ git+https://github.com/rsxdalv/chatterbox.git@faster",
#     "torch",
#     "torchaudio",
#     "transformers>=4.46,<5",
# ]
# ///
"""Generate a pre-conditioned voice file (.pt) from a reference audio file (.wav).

Conditioning extracts speaker embeddings from a reference audio clip and saves
them as a serialized Chatterbox Conditionals object. This is a one-time
operation — the resulting .pt file can be used in place of the .wav for faster
TTS generation (skips re-extraction on every call).

Usage:

    uv run --no-project condition_voice.py voices/Jake.wav                        # -> voices/Jake.pt
    uv run --no-project condition_voice.py voices/Jake.wav -e 0.8                 # custom exaggeration
    uv run --no-project condition_voice.py voices/Jake.wav -o voices/alt-Jake.pt  # custom output path
"""
import argparse
import sys
from pathlib import Path

from chatterbox.tts import ChatterboxTTS

DEFAULT_EXAGGERATION = 0.5


def main():
    parser = argparse.ArgumentParser(description="Generate a .pt conditioned voice from a .wav reference")
    parser.add_argument("input", type=Path, help="Path to input .wav file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output .pt path (default: same name as input with .pt extension)")
    parser.add_argument("-e", "--exaggeration", type=float, default=DEFAULT_EXAGGERATION,
                        help=f"Emotional expressiveness baked into the voice (default: {DEFAULT_EXAGGERATION})")
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to run on (default: cpu)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.with_suffix(".pt")

    print(f"Loading model on {args.device}...")
    model = ChatterboxTTS.from_pretrained(device=args.device)

    print(f"Extracting conditioning from {args.input} (exaggeration={args.exaggeration})...")
    model.prepare_conditionals(str(args.input), exaggeration=args.exaggeration)

    model.conds.save(output)
    print(f"Saved conditioned voice to {output}")


if __name__ == "__main__":
    main()
