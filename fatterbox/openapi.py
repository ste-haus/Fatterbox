"""FastAPI REST API server for Chatterbox TTS."""
import asyncio
import logging
import math
import struct
import time
from pathlib import Path
from typing import Optional

import torch
from chatterbox.tts import Conditionals
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from .model import MODEL_LOCK
from .utils import split_text
from .voices import reload_voices

_LOGGER = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    """Request model for TTS generation (OpenAPI compatible)."""
    input: str = Field(..., description="Text to synthesize", alias="text")
    voice: Optional[str] = Field(None, description="Voice name (from voices directory)")
    model: Optional[str] = Field("tts-1", description="Model identifier (for compatibility)")
    response_format: Optional[str] = Field("wav", description="Audio format (only wav supported)")
    speed: Optional[float] = Field(1.0, description="Playback speed (not implemented)")
    exaggeration: Optional[float] = Field(None, description="Emotional expressiveness (0.0-2.0). Defaults to FATTERBOX_EXAGGERATION. Applied per request regardless of what a .pt was conditioned at.")
    
    class Config:
        populate_by_name = True  # Allow both 'input' and 'text' field names


class VoiceInfo(BaseModel):
    """Voice information model."""
    name: str
    path: str
    conditioned: bool = False
    exaggeration: Optional[float] = None


class ServerInfo(BaseModel):
    """Server information model."""
    name: str = "Fatterbox TTS API"
    version: str = "1.0.0"
    sample_rate: int
    voices: list[VoiceInfo]


def create_api(model, voices, voices_dir: Path, precondition: bool = False) -> FastAPI:
    """Create FastAPI application with OpenAPI-compatible endpoints."""
    app = FastAPI(
        title="Fatterbox TTS API",
        description="OpenAPI-compatible Chatterbox TTS with voice cloning and streaming",
        version="1.0.0"
    )
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/v1/info", response_model=ServerInfo)
    async def get_info():
        """Get server information and available voices."""
        voice_list = [
            VoiceInfo(
                name=voice.name,
                path=voice.path,
                conditioned=voice.is_conditioned,
                exaggeration=voice.exaggeration,
            )
            for voice in voices.values()
        ]
        return ServerInfo(
            sample_rate=model.sr,
            voices=voice_list
        )
    
    @app.get("/v1/voices")
    async def get_voices():
        """Get available voices.

        `exaggeration` reports the value a .pt was conditioned at, which is
        informational only -- any voice can serve any requested exaggeration.
        """
        return {
            "voices": [
                {"name": voice.name, "exaggeration": voice.exaggeration}
                for voice in voices.values()
            ]
        }
    
    @app.post("/v1/voices/reload")
    async def reload():
        """Rescan the voices directory, picking up files added since startup.

        Conditioning is blocking work that holds the model lock, so it runs in
        a worker thread rather than stalling in-flight streams.
        """
        exaggeration = getattr(model, "_wyoming_gen_params", {}).get("exaggeration", 0.5)

        _LOGGER.info(f"Reloading voices from {voices_dir}")

        return await asyncio.get_event_loop().run_in_executor(
            None,
            reload_voices,
            voices_dir,
            model,
            voices,
            exaggeration,
            precondition,
        )

    @app.post("/v1/audio/speech")
    async def synthesize(request: TTSRequest):
        """
        Synthesize speech from text (OpenAPI-compatible endpoint).
        Automatically streams audio using chunked transfer encoding and sentence splitting.
        
        Compatible with OpenAI's /v1/audio/speech endpoint.
        """
        # Accept both 'input' and 'text' field names for compatibility
        text = request.input
        
        _LOGGER.info(f"TTS request: '{text[:50]}...' with voice: {request.voice}")
        
        voice = voices.get(request.voice) if request.voice else None

        if request.voice and not voice:
            raise HTTPException(status_code=404, detail=f"Voice '{request.voice}' not found")

        gen_params = getattr(model, "_wyoming_gen_params", {})

        voice_path = voice.path if voice else None
        exaggeration = request.exaggeration if request.exaggeration is not None else gen_params.get("exaggeration", 0.5)
        
        async def generate_chunks():
            """Generate audio chunks using sentence splitting (streaming)."""
            try:
                # Split text into chunks for streaming (same as Wyoming handler)
                chunks = split_text(text)
                
                # Send WAV header first
                header = _create_wav_header(model.sr)
                yield header
                
                start_time = time.time()
                first_chunk = True
                
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    
                    chunk_start = time.time()
                    
                    # Generate audio for this chunk
                    audio = await asyncio.get_event_loop().run_in_executor(
                        None,
                        _generate_audio_sync,
                        model,
                        chunk,
                        voice_path,
                        exaggeration
                    )
                    
                    chunk_time = time.time() - chunk_start
                    
                    if first_chunk:
                        ttfa = time.time() - start_time
                        _LOGGER.info(f"First chunk generated (TTFA: {ttfa:.2f}s)")
                        first_chunk = False
                    
                    _LOGGER.info(f"Chunk {i+1}/{len(chunks)}: {chunk_time:.2f}s")
                    
                    # Convert to PCM bytes and yield
                    audio_bytes = (audio * 32767).numpy().astype("int16").tobytes()
                    yield audio_bytes
                
                total_time = time.time() - start_time
                _LOGGER.info(f"Streaming complete - Total time: {total_time:.2f}s")
            
            except Exception as e:
                _LOGGER.error(f"Error during streaming synthesis: {e}", exc_info=True)
                raise
            finally:
                # Clean up CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return StreamingResponse(
            generate_chunks(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "Cache-Control": "no-cache",
            }
        )
    
    return app


def _generate_audio_sync(model, text: str, voice_path: str = None, exaggeration: float = None) -> torch.Tensor:
    """Generate audio synchronously (same logic as Wyoming handler)."""
    with MODEL_LOCK, torch.no_grad():
        backend = getattr(model, '_wyoming_backend', 'cudagraphs-manual')
        gen_params = getattr(model, '_wyoming_gen_params', {})

        t3_params = {
            "benchmark_t3": True,
            "generate_token_backend": backend,
            "skip_when_1": True,
        }

        if gen_params.get("seed"):
            t3_params["seed"] = gen_params["seed"]

        if exaggeration is None:
            exaggeration = gen_params.get("exaggeration", 0.5)

        if voice_path and voice_path.endswith(".pt"):
            device = next(model.t3.parameters()).device
            model.conds = Conditionals.load(voice_path, map_location=device)
            _LOGGER.debug(f"Using conditioned voice at exaggeration {exaggeration}")
            wav = model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=gen_params.get("cfg_weight", 0.5),
                t3_params=t3_params,
            )
        elif voice_path:
            wav = model.generate(
                text,
                audio_prompt_path=voice_path,
                exaggeration=exaggeration,
                cfg_weight=gen_params.get("cfg_weight", 0.5),
                t3_params=t3_params,
            )
        else:
            wav = model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=gen_params.get("cfg_weight", 0.5),
                t3_params=t3_params,
            )

        result = wav.squeeze().cpu()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return result


def _create_wav_header(sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Create a minimal WAV header for streaming (data size will be unknown)."""
    # RIFF header
    riff = b'RIFF'
    # File size (unknown for streaming, use max value)
    file_size = struct.pack('<I', 0xFFFFFFFF - 8)
    wave = b'WAVE'
    
    # fmt subchunk
    fmt = b'fmt '
    fmt_size = struct.pack('<I', 16)  # PCM
    audio_format = struct.pack('<H', 1)  # PCM
    num_channels = struct.pack('<H', channels)
    sample_rate_bytes = struct.pack('<I', sample_rate)
    byte_rate = struct.pack('<I', sample_rate * channels * sample_width)
    block_align = struct.pack('<H', channels * sample_width)
    bits_per_sample = struct.pack('<H', sample_width * 8)
    
    # data subchunk
    data = b'data'
    data_size = struct.pack('<I', 0xFFFFFFFF)  # Unknown size for streaming
    
    return (riff + file_size + wave +
            fmt + fmt_size + audio_format + num_channels + 
            sample_rate_bytes + byte_rate + block_align + bits_per_sample +
            data + data_size)
