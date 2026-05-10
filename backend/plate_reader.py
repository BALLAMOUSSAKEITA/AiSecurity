"""
Utilitaire de lecture de plaques d'immatriculation.
Stratégie :
  1. Si PLATE_RECOGNIZER_API_KEY est défini → Plate Recognizer API (haute précision)
  2. Sinon → Tesseract OCR local (précision moindre, aucune clé requise)
"""
import os
import re
import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


PLATE_RECOGNIZER_API_KEY = os.getenv("PLATE_RECOGNIZER_API_KEY", "")
PLATE_RECOGNIZER_URL = "https://api.platerecognizer.com/v1/plate-reader/"


def _preprocess_for_ocr(img_array: np.ndarray) -> np.ndarray:
    """Améliore l'image pour une meilleure détection OCR."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # Upscale si trop petit
    h, w = gray.shape
    if w < 200:
        scale = 200 / w
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    # Débruitage + binarisation
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _clean_plate_text(text: str) -> str:
    """Nettoie le texte OCR pour ne garder que les caractères de plaque."""
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9\-]', '', text)
    return text


def read_plate_tesseract(image_bytes: bytes) -> dict:
    """Lecture OCR via Tesseract (fallback local)."""
    if not TESSERACT_AVAILABLE:
        return {"plate": None, "confidence": 0.0, "engine": "tesseract", "error": "Tesseract non installé"}

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        processed = _preprocess_for_ocr(img_bgr)

        config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)

        texts = []
        confs = []
        for txt, conf in zip(data['text'], data['conf']):
            if txt.strip() and int(conf) > 30:
                texts.append(txt.strip())
                confs.append(int(conf))

        if not texts:
            return {"plate": None, "confidence": 0.0, "engine": "tesseract"}

        plate = _clean_plate_text(' '.join(texts))
        confidence = sum(confs) / len(confs) / 100.0

        return {"plate": plate if len(plate) >= 4 else None, "confidence": confidence, "engine": "tesseract"}
    except Exception as e:
        return {"plate": None, "confidence": 0.0, "engine": "tesseract", "error": str(e)}


def read_plate_recognizer(image_bytes: bytes) -> dict:
    """Lecture via l'API Plate Recognizer (haute précision)."""
    if not REQUESTS_AVAILABLE or not PLATE_RECOGNIZER_API_KEY:
        return {"plate": None, "confidence": 0.0, "engine": "plate_recognizer", "error": "API key manquante"}

    try:
        response = _requests.post(
            PLATE_RECOGNIZER_URL,
            files={"upload": ("vehicle.jpg", image_bytes, "image/jpeg")},
            headers={"Authorization": f"Token {PLATE_RECOGNIZER_API_KEY}"},
            timeout=5,
        )
        if response.status_code != 201:
            return {"plate": None, "confidence": 0.0, "engine": "plate_recognizer",
                    "error": f"HTTP {response.status_code}"}

        data = response.json()
        results = data.get("results", [])
        if not results:
            return {"plate": None, "confidence": 0.0, "engine": "plate_recognizer"}

        best = results[0]
        plate = best.get("plate", "").upper()
        score = best.get("score", 0.0)
        return {"plate": plate, "confidence": score, "engine": "plate_recognizer"}
    except Exception as e:
        return {"plate": None, "confidence": 0.0, "engine": "plate_recognizer", "error": str(e)}


def read_plate(image_bytes: bytes) -> dict:
    """Point d'entrée principal : utilise Plate Recognizer si clé disponible, sinon Tesseract."""
    if PLATE_RECOGNIZER_API_KEY:
        result = read_plate_recognizer(image_bytes)
        if result.get("plate"):
            return result
    return read_plate_tesseract(image_bytes)
