# AiSecurity – Détection de véhicules, plaques & vitesse

Application web mobile-first permettant de :
- **Accéder à la caméra** du téléphone (caméra arrière par défaut)
- **Détecter les véhicules** en temps réel via TensorFlow.js (COCO-SSD)
- **Lire les plaques** d'immatriculation (OCR via Tesseract ou Plate Recognizer API)
- **Estimer la vitesse** par optical flow (OpenCV)

---

## Architecture

```
frontend/     React + Vite + TypeScript + TensorFlow.js
backend/      FastAPI + Python + OpenCV + Tesseract/Plate Recognizer
```

---

## Lancement rapide

### 1. Backend Python

```bash
cd backend

# Copier et remplir le fichier d'environnement
cp .env.example .env

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

# Démarrer le serveur
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

> **Windows** : installez Tesseract OCR depuis https://github.com/UB-Mannheim/tesseract/wiki
> puis ajoutez son chemin dans `.env` avec `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`

### 2. Frontend React

```bash
cd frontend
npm install
npm run dev
```

L'app sera disponible sur `http://localhost:5173`.

---

## Accès depuis le téléphone

Le serveur Vite expose l'app sur le réseau local (`--host true`).
Depuis votre téléphone (même réseau Wi-Fi) :

```
http://<IP-de-votre-PC>:5173
```

> **HTTPS requis** sur certains téléphones pour accéder à la caméra.
> Utilisez `ngrok` ou configurez un certificat auto-signé si nécessaire :
> ```bash
> npx ngrok http 5173
> ```

---

## Améliorer la précision des plaques

| Moteur | Précision | Prérequis |
|--------|-----------|-----------|
| Tesseract (défaut) | ~60% | Installation locale |
| Plate Recognizer | ~95% | Clé API gratuite à https://platerecognizer.com |

Ajoutez votre clé dans `backend/.env` :
```
PLATE_RECOGNIZER_API_KEY=votre_clé_ici
```

---

## Limitations connues

- La vitesse est une **estimation indicative** (±30% sans calibration fixe)
- Précision optimale : caméra fixée sur trépied, angle constant vers la route
- Plaques floues (haute vitesse) = OCR dégradé
- Usage soumis au **RGPD** : ne pas collecter de plaques sans base légale

---

## Stack technique

- **Frontend** : React 18, TypeScript, Vite, TensorFlow.js 4, COCO-SSD
- **Backend** : FastAPI, Python 3.11+, OpenCV, Tesseract, Pillow, python-dotenv
