import os
import sys
import uuid
import shutil
import logging
import re
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

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


class FlagItem(BaseModel):
    type: str
    label: str
    text: str


class AnalysisResult(BaseModel):
    type: str
    title: str
    desc: str
    scam: float
    warn: float
    safe: float
    transcript: Optional[str] = None
    flags: List[FlagItem] = []


class TextRequest(BaseModel):
    text: str


INDICATOR_RULES = [
    {
        "patterns": [r"\b(police|officer|sergeant|inspector|bukit aman|cybercrime|polis|pdrm|interpol|fbi|cia)\b"],
        "type": "danger",
        "label": "Authority Impersonation",
        "text": "Claims to be law enforcement or government authority",
    },
    {
        "patterns": [r"\b(arrest|jail|prison|warrant|legal action|court order|sue you|detained|handcuff)\b"],
        "type": "danger",
        "label": "Arrest/Legal Threat",
        "text": "Threatens arrest or legal consequences",
    },
    {
        "patterns": [r"\b(immediately|urgent|right now|within .* hour|deadline|expires today|act fast|hurry|quickly)\b"],
        "type": "danger",
        "label": "Urgency & Pressure",
        "text": "Creates artificial time pressure to force quick action",
    },
    {
        "patterns": [r"\b(tac|otp|pin|password|cvv|security code|verification code|login credential)\b"],
        "type": "danger",
        "label": "Credential Request",
        "text": "Requests sensitive authentication codes or passwords",
    },
    {
        "patterns": [r"\b(bank account|account number|transfer .* (rm|usd|inr|money)|wire transfer|pay .*immediately|send money)\b"],
        "type": "danger",
        "label": "Financial Demand",
        "text": "Demands money transfer or requests bank account details",
    },
    {
        "patterns": [r"\b(don.t tell|don.t inform|classified|confidential|secret|between us|do not share)\b"],
        "type": "warn",
        "label": "Secrecy Demand",
        "text": "Instructs victim not to inform anyone about the call",
    },
    {
        "patterns": [r"\b(remote access|anydesk|teamviewer|install .* app|download .* software|screen share)\b"],
        "type": "danger",
        "label": "Remote Access Request",
        "text": "Asks to install remote access software",
    },
    {
        "patterns": [r"\b(refund|compensation|reward|prize|lottery|won|claim your|free gift)\b"],
        "type": "warn",
        "label": "Reward/Refund Lure",
        "text": "Offers unexpected reward or refund as bait",
    },
    {
        "patterns": [r"\b(survey|research|feedback|questionnaire)\b.*\b(saving|investment|banking|financial|income)\b"],
        "type": "warn",
        "label": "Information Harvesting",
        "text": "Collecting personal financial information through survey",
    },
    {
        "patterns": [r"\b(mykad|ic number|nric|passport number|social security|ssn|aadhaar)\b"],
        "type": "warn",
        "label": "Identity Document Request",
        "text": "Requests national ID or identity document numbers",
    },
    {
        "patterns": [r"\b(drug trafficking|money laundering|terrorism|fraud case|criminal case|investigation)\b"],
        "type": "danger",
        "label": "Criminal Accusation",
        "text": "Falsely accuses victim of involvement in criminal activity",
    },
    {
        "patterns": [r"\b(follow.up|following up|your (recent|previous) (application|request|order|appointment))\b"],
        "type": "safe",
        "label": "Legitimate Follow-up",
        "text": "References a prior application or request by the caller",
    },
    {
        "patterns": [r"\b(confirm .* (details|identity|appointment|address)|verify your (name|email|employment))\b"],
        "type": "safe",
        "label": "Standard Verification",
        "text": "Performs routine identity or appointment confirmation",
    },
    {
        "patterns": [r"\b(help desk|helpdesk|it support|technical support|customer service|service desk)\b"],
        "type": "safe",
        "label": "Support Context",
        "text": "Conversation involves a standard IT or customer support request",
    },
    {
        "patterns": [r"\b(printer|laptop|vpn|wifi|wi-fi|software|network|email .* (issue|problem|not working))\b"],
        "type": "safe",
        "label": "Technical Issue",
        "text": "Discusses a routine technical or equipment issue",
    },
    {
        "patterns": [r"\b(reference number|ticket number|case id|tracking|txn)\b"],
        "type": "safe",
        "label": "Reference Provided",
        "text": "Provides a tracking or reference number for follow-up",
    },
]


def detect_flags(text):
    lower = text.lower()
    flags = []
    seen_labels = set()
    for rule in INDICATOR_RULES:
        if rule["label"] in seen_labels:
            continue
        for pattern in rule["patterns"]:
            if re.search(pattern, lower):
                flags.append(FlagItem(
                    type=rule["type"],
                    label=rule["label"],
                    text=rule["text"],
                ))
                seen_labels.add(rule["label"])
                break
    flags.sort(key=lambda f: {"danger": 0, "warn": 1, "safe": 2}.get(f.type, 3))
    return flags


def classify(text):
    result = predict(text)
    scam_pct = round(result["scam"] * 100, 1)
    warn_pct = round(result["slightly_suspicious"] * 100, 1)
    safe_pct = round(result["neutral"] * 100, 1)
    label = result["label"]
    flags = detect_flags(text)

    if label == "scam":
        return {
            "type": "danger",
            "title": "Scam Detected",
            "desc": "High-confidence vishing indicators detected. This matches known scam patterns including authority impersonation and financial pressure.",
            "scam": scam_pct,
            "warn": warn_pct,
            "safe": safe_pct,
            "flags": flags,
        }
    elif label == "slightly_suspicious":
        return {
            "type": "warn",
            "title": "Slightly Suspicious",
            "desc": "Some risk indicators present. Exercise caution and verify the caller's identity independently.",
            "scam": scam_pct,
            "warn": warn_pct,
            "safe": safe_pct,
            "flags": flags,
        }
    else:
        return {
            "type": "safe",
            "title": "Call is Safe",
            "desc": "No significant scam indicators detected. The conversation appears to be from a legitimate source.",
            "scam": scam_pct,
            "warn": warn_pct,
            "safe": safe_pct,
            "flags": flags,
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