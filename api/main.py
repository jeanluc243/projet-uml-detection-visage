import base64
import io
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from scripts.extract_faces import extract_face

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "mobilenet_best.keras"
SUMMARY_PATH = BASE_DIR / "reports" / "summary.json"
IMG_SIZE = (224, 224)
DEFAULT_UNKNOWN_THRESHOLD = 0.60

app = FastAPI(
    title="Face Recognition API",
    description="API de reconnaissance faciale pour identifier une personne à partir d'une photo.",
    version="1.0.0",
)


class PredictBase64Request(BaseModel):
    image_base64: str = Field(..., description="Image encodée en base64, avec ou sans data URL prefix.")


class PredictionResponse(BaseModel):
    identity: str
    confidence: float
    known: bool
    scores: dict[str, float]


def _model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()


def _unknown_threshold() -> float:
    value = os.getenv("UNKNOWN_THRESHOLD", str(DEFAULT_UNKNOWN_THRESHOLD))
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError("UNKNOWN_THRESHOLD doit être un nombre décimal, ex: 0.60") from exc


@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    model_path = _model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable: {model_path}. Définis MODEL_PATH ou ajoute le fichier .keras au déploiement."
        )
    return tf.keras.models.load_model(model_path)


@lru_cache(maxsize=1)
def load_class_names() -> list[str]:
    env_classes = os.getenv("CLASS_NAMES")
    if env_classes:
        return [name.strip() for name in env_classes.split(",") if name.strip()]

    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r", encoding="utf-8") as file:
            summary: dict[str, Any] = json.load(file)
        classes = summary.get("classes")
        if isinstance(classes, list) and all(isinstance(item, str) for item in classes):
            return classes

    dataset_train = BASE_DIR / "images" / "dataset" / "train"
    if dataset_train.exists():
        return sorted(path.name for path in dataset_train.iterdir() if path.is_dir())

    raise FileNotFoundError(
        "Classes introuvables. Ajoute reports/summary.json ou définis CLASS_NAMES=bujiriri,kabulu,mateo."
    )


def _read_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image valide.") from exc


def _prepare_image(image: Image.Image) -> np.ndarray:
    cache_dir = BASE_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".jpg", dir=cache_dir, delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        image.save(temp_path, format="JPEG", quality=95)
        face = extract_face(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    if face is None:
        raise HTTPException(status_code=422, detail="Aucun visage détecté dans l'image.")

    face = face.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(face, dtype=np.float32)
    return np.expand_dims(pixels, axis=0)


def _predict_from_bytes(image_bytes: bytes) -> PredictionResponse:
    model = load_model()
    class_names = load_class_names()
    image = _read_image(image_bytes)
    batch = _prepare_image(image)

    probabilities = model.predict(batch, verbose=0)[0]
    if len(probabilities) != len(class_names):
        raise HTTPException(
            status_code=500,
            detail=f"Incohérence modèle/classes: {len(probabilities)} sorties pour {len(class_names)} classes.",
        )

    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    threshold = _unknown_threshold()
    known = confidence >= threshold
    identity = class_names[best_index] if known else "unknown"

    return PredictionResponse(
        identity=identity,
        confidence=confidence,
        known=known,
        scores={class_name: float(probabilities[index]) for index, class_name in enumerate(class_names)},
    )


@app.get("/health")
def health() -> dict[str, Any]:
    model_path = _model_path()
    return {
        "status": "ok",
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "classes": load_class_names(),
        "unknown_threshold": _unknown_threshold(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envoie une image avec multipart/form-data, champ 'file'.")
    image_bytes = await file.read()
    return _predict_from_bytes(image_bytes)


@app.post("/predict-base64", response_model=PredictionResponse)
def predict_base64(payload: PredictBase64Request) -> PredictionResponse:
    encoded = payload.image_base64
    if "," in encoded and encoded.strip().lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="image_base64 n'est pas un base64 valide.") from exc

    return _predict_from_bytes(image_bytes)
