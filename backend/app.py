import os
import sys
import uuid
import shutil
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Modules.speech_module import transcribe_audio
from Models.predict import predict

UPLOAD_DIR = PROJECT_ROOT / "temp_uploads"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(exist_ok=True)
    logger.info("✓ VoiceGuard API started")
    logger.info("✓ speech_module loaded")
    logger.info("✓ hybrid_model_soft loaded")
    yield
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(title="VoiceGuard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResult(BaseModel):
    type: str
    title: str
    desc: str
    scam: float
    warn: float
    safe: float
    transcript: Optional[str] = None


class TextRequest(BaseModel):
    text: str


def classify(text):
    result = predict(text)
    scam_pct = round(result["scam"] * 100, 1)
    warn_pct = round(result["slightly_suspicious"] * 100, 1)
    safe_pct = round(result["neutral"] * 100, 1)
    label = result["label"]

    if label == "scam":
        return {
            "type": "danger",
            "title": "Scam Detected",
            "desc": "High-confidence vishing indicators detected. This matches known scam patterns including authority impersonation and financial pressure.",
            "scam": scam_pct,
            "warn": warn_pct,
            "safe": safe_pct,
        }
    elif label == "slightly_suspicious":
        return {
            "type": "warn",
            "title": "Slightly Suspicious",
            "desc": "Some risk indicators present. Exercise caution and verify the caller's identity independently.",
            "scam": scam_pct,
            "warn": warn_pct,
            "safe": safe_pct,
        }
    else:
        return {
            "type": "safe",
            "title": "Call is Safe",
            "desc": "No significant scam indicators detected. The conversation appears to be from a legitimate source.",
            "scam": scam_pct,
            "warn": warn_pct,
            "safe": safe_pct,
        }

@app.get("/api/health")
async def health():
    return {"status": "ok", "modules": {"speech": True, "model": True}}


@app.post("/api/analyze/audio", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext != ".wav":
        raise HTTPException(400, "Only .wav files are supported")

    audio_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        transcript = transcribe_audio(str(audio_path))
        if transcript is None:
            raise HTTPException(500, "Transcription failed — check AssemblyAI API key")

        logger.info(f"Transcribed: {transcript[:100]}...")
        scores = classify(transcript)
        return AnalysisResult(transcript=transcript, **scores)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))
    finally:
        if audio_path.exists():
            audio_path.unlink()

@app.post("/api/analyze/text", response_model=AnalysisResult)
async def analyze_text(request: TextRequest):
    try:
        scores = classify(request.text)
        return AnalysisResult(transcript=request.text, **scores)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))

if FRONTEND_DIST.exists():
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend"
    )