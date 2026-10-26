"""FastAPI application for the laboratory 5 synthesis service."""


import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .errors import ArtifactError, InputTextError
from .pipeline import TTSPipeline
from .settings import ServiceSettings


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


_pipeline: TTSPipeline | None = None
_load_lock = threading.Lock()


def get_pipeline() -> TTSPipeline:
    global _pipeline
    if _pipeline is None:
        with _load_lock:
            if _pipeline is None:
                _pipeline = TTSPipeline.from_config(os.getenv("TTS_LAB5_CONFIG"))
    return _pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = ServiceSettings.load(os.getenv("TTS_LAB5_CONFIG"))
    if settings.preload_models:
        await run_in_threadpool(get_pipeline)
    yield


app = FastAPI(
    title="RUSLAN FastSpeech 2 synthesis service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    settings = ServiceSettings.load(os.getenv("TTS_LAB5_CONFIG"))
    missing = [str(path) for path in settings.required_artifacts() if not path.is_file()]
    return {
        "status": "ok" if not missing else "not_ready",
        "models_loaded": _pipeline is not None,
        "device": _pipeline.device if _pipeline is not None else settings.device,
        "missing_artifacts": missing,
    }


@app.post("/analyze")
async def analyze(request: SynthesisRequest) -> dict:
    try:
        frontend = await run_in_threadpool(get_pipeline().analyze, request.text)
    except InputTextError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ArtifactError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "original_text": frontend.original_text,
        "normalized_text": frontend.normalized_text,
        "words": [
            {"word": word.spelling, "normalized": word.normalized, "phones": list(word.phones)}
            for word in frontend.words
        ],
        "phones": list(frontend.phones),
        "pauses": list(frontend.pauses),
        "duration_overrides": frontend.duration_overrides,
    }


@app.post("/synthesize", response_class=Response)
async def synthesize(request: SynthesisRequest) -> Response:
    try:
        result = await run_in_threadpool(get_pipeline().synthesize, request.text)
    except InputTextError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ArtifactError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(
        content=result.wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="synthesis.wav"',
            "X-Synthesis-Time": f"{result.synthesis_time_s:.6f}",
            "X-Audio-Duration": f"{result.audio_duration_s:.6f}",
            "X-Real-Time-Factor": f"{result.real_time_factor:.6f}",
        },
    )

