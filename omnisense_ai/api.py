"""
OmniSense AI — Production API Server
=====================================
REST API exposing multimodal analysis and voice narration capabilities.
Built with FastAPI for async, scalable, production-ready deployment.

Endpoints:
  POST /analyse          → Text + image analysis (returns JSON)
  POST /analyse/voice    → Text + image analysis + MP3 audio
  POST /batch            → Multiple analysis requests
  GET  /modes            → Available analysis modes
  GET  /health           → System health check

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

Author : Pranay Sai Vanam
"""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from src.multimodal_engine import AnalysisMode, LLMProvider, MultimodalEngine
from src.voice_narrator import OmniNarrator, VoiceProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Startup ──────────────────────────────────────────────────────────────────

engine  : Optional[MultimodalEngine] = None
narrator: Optional[OmniNarrator]     = None
UPLOAD_DIR = "/tmp/omnisense_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, narrator
    provider = LLMProvider(os.getenv("LLM_PROVIDER", "anthropic"))
    engine   = MultimodalEngine(provider=provider)
    narrator = OmniNarrator(preferred=VoiceProvider.GTTS, output_dir="./outputs/audio")
    logger.info(f"OmniSense API started — provider: {provider}")
    yield
    logger.info("OmniSense API shutting down.")


app = FastAPI(
    title       = "OmniSense AI — Multimodal Analysis API",
    description = "Vision + Language + Voice powered by Claude / GPT-4o. Built by Pranay Sai Vanam.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins =["*"],
    allow_methods =["*"],
    allow_headers =["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    text_input   : str         = Field(..., description="User's question or context")
    mode         : AnalysisMode= Field(AnalysisMode.DESCRIBE, description="Analysis mode")
    extra_context: str         = Field("", description="Optional extra system context")


class AnalysisResponse(BaseModel):
    mode        : str
    provider    : str
    analysis    : str
    confidence  : str
    tokens_used : int
    latency_ms  : float
    has_image   : bool


class HealthResponse(BaseModel):
    status  : str
    provider: str
    version : str


class ModeInfo(BaseModel):
    mode       : str
    description: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """API health check with provider info."""
    return HealthResponse(
        status  = "healthy",
        provider= engine.provider if engine else "not initialized",
        version = "1.0.0",
    )


@app.get("/modes", response_model=list[ModeInfo], tags=["System"])
async def list_modes():
    """List all available analysis modes with descriptions."""
    descriptions = {
        "describe"      : "General visual + text description and analysis",
        "compare"       : "Verify whether image supports or contradicts a text claim",
        "extract"       : "Extract structured data (tables, text, numbers) from images",
        "sentiment"     : "Analyse mood, emotion, and tonal context",
        "accessibility" : "Generate WCAG-compliant alt-text for the image",
        "summarize"     : "Summarise documents, charts, and slide images",
    }
    return [ModeInfo(mode=m, description=d) for m, d in descriptions.items()]


@app.post("/analyse", response_model=AnalysisResponse, tags=["Analysis"])
async def analyse(
    text_input   : str         = Form(...),
    mode         : AnalysisMode= Form(AnalysisMode.DESCRIBE),
    extra_context: str         = Form(""),
    image        : Optional[UploadFile] = File(None),
):
    """
    Analyse text and/or image together.
    Supports multipart form with optional image upload.
    Returns structured JSON with analysis + metadata.
    """
    if not engine:
        raise HTTPException(503, "Engine not initialized.")

    image_path = None
    if image and image.filename:
        img_bytes  = await image.read()
        image_path = f"{UPLOAD_DIR}/{int(time.time())}_{image.filename}"
        with open(image_path, "wb") as f:
            f.write(img_bytes)

    try:
        result = engine.analyse(
            text_input    = text_input,
            image_path    = image_path,
            mode          = mode,
            extra_context = extra_context,
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(500, str(e))

    return AnalysisResponse(
        mode       = result.mode,
        provider   = result.provider,
        analysis   = result.analysis,
        confidence = result.confidence,
        tokens_used= result.tokens_used,
        latency_ms = result.latency_ms,
        has_image  = image_path is not None,
    )


@app.post("/analyse/voice", tags=["Analysis"])
async def analyse_with_voice(
    text_input   : str          = Form(...),
    mode         : AnalysisMode = Form(AnalysisMode.DESCRIBE),
    extra_context: str          = Form(""),
    language     : str          = Form("english"),
    image        : Optional[UploadFile] = File(None),
):
    """
    Full multimodal pipeline: Text + Image → LLM Analysis → Voice Narration.
    Returns MP3 audio stream of the analysis result.
    """
    if not engine or not narrator:
        raise HTTPException(503, "Engine not initialized.")

    image_path = None
    if image and image.filename:
        img_bytes  = await image.read()
        image_path = f"{UPLOAD_DIR}/{int(time.time())}_{image.filename}"
        with open(image_path, "wb") as f:
            f.write(img_bytes)

    try:
        result     = engine.analyse(text_input, image_path, mode, extra_context)
        audio_bytes= narrator.narrate_to_bytes(result.analysis)
    except Exception as e:
        logger.error(f"Voice pipeline error: {e}")
        raise HTTPException(500, str(e))

    return Response(
        content     = audio_bytes,
        media_type  = "audio/mpeg",
        headers     = {
            "X-Analysis-Mode"      : result.mode,
            "X-Confidence"         : result.confidence,
            "X-Tokens-Used"        : str(result.tokens_used),
            "X-Latency-Ms"         : str(result.latency_ms),
            "Content-Disposition"  : "inline; filename=omnisense_analysis.mp3",
        },
    )


@app.post("/batch", tags=["Analysis"])
async def batch_analyse(
    requests: list[AnalysisRequest],
):
    """Process multiple text-only analysis requests in a single call."""
    if not engine:
        raise HTTPException(503, "Engine not initialized.")
    if len(requests) > 10:
        raise HTTPException(400, "Max 10 requests per batch.")

    results = engine.batch_analyse([r.model_dump() for r in requests])
    return [
        {
            "mode"      : r.mode,
            "analysis"  : r.analysis,
            "confidence": r.confidence,
            "tokens"    : r.tokens_used,
        }
        for r in results
    ]
