"""
Serveur FastAPI – AiSecurity
Endpoints :
  POST /api/plate   → OCR de la plaque d'immatriculation
  POST /api/speed   → Estimation de vitesse via optical flow (2 frames)
  GET  /api/health  → Status
"""
import io
import base64
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from plate_reader import read_plate
from speed_estimator import SpeedEstimator

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aisecurity")

speed_estimator = SpeedEstimator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AiSecurity backend démarré ✓")
    yield
    logger.info("AiSecurity backend arrêté")


app = FastAPI(title="AiSecurity API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/plate")
async def detect_plate(image: UploadFile = File(...)):
    """
    Reçoit une image (crop du véhicule) et retourne la plaque lue.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")

    raw = await image.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Image vide")
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image trop grande (max 5 Mo)")

    result = read_plate(raw)
    logger.info("Plaque détectée: %s (%.0f%%)", result.get("plate"), (result.get("confidence", 0) * 100))
    return JSONResponse(content=result)


@app.post("/api/speed")
async def estimate_speed(
    vehicle_id: str = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    bbox_w: float = Form(...),
    bbox_h: float = Form(...),
    frame: UploadFile = File(...),
):
    """
    Reçoit une frame en niveaux de gris (JPEG) + position du véhicule,
    retourne la vitesse estimée.
    """
    raw = await frame.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise HTTPException(status_code=400, detail="Frame invalide")

    # Mise à jour du frame_width de l'estimateur si besoin
    if img.shape[1] != speed_estimator.frame_width:
        speed_estimator.frame_width = img.shape[1]

    speed = speed_estimator.update(vehicle_id, img, cx, cy, bbox_w, bbox_h)
    speed_estimator.cleanup_stale()

    return JSONResponse(content={
        "vehicle_id": vehicle_id,
        "speed_kmh": round(speed, 1) if speed is not None else None,
    })
