"""Image classification utilities.

This module tries to use a saved Keras model (if available) and maps predictions
to labels defined in LABELS. If no model is available, it falls back to a
deterministic filename-based classifier so behavior is reproducible for testing.
"""
import os
import random
from typing import Tuple
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

try:
    # Optional deep learning model (if present)
    

    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'convnextnet_model.h5')
    if os.path.exists(MODEL_PATH):
        try:
            DL_MODEL = load_model(MODEL_PATH)
        except Exception as e:
            print(f"[ImageClass] Failed to load DL model: {e}")
            DL_MODEL = None
    else:
        DL_MODEL = None
except Exception:
    DL_MODEL = None

# Default labels (must match how model was trained). Adjust if model class order differs.
LABELS = [
    "Bele", "Chela", "Guchi", "Kachki", "Kata Phasa",
    "Mola", "Nama Chanda", "Pabda", "Puti", "Tengra"
]


def _predict_with_model(image_path: str) -> Tuple[str, float]:
    """Predict using the DL model if available. Returns (label, confidence).

    If the model is missing or outputs unexpected shape, raises RuntimeError.
    """
    if DL_MODEL is None:
        raise RuntimeError("No DL model available")

    # Load and preprocess
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = DL_MODEL.predict(img_array)

    # Expecting a vector of class probabilities
    if preds.ndim == 2 and preds.shape[1] == len(LABELS):
        prob_vector = preds[0]
        idx = int(prob_vector.argmax())
        label = LABELS[idx]
        confidence = float(prob_vector[idx])
        return label, round(confidence, 3)
    else:
        raise RuntimeError(f"Unexpected prediction shape: {preds.shape}")


def _deterministic_fallback(image_path: str) -> Tuple[str, float]:
    """Deterministic fallback classifier based on filename."""
    basename = os.path.basename(image_path)
    h = sum(ord(c) for c in basename) % len(LABELS)
    label = LABELS[h]
    random.seed(basename)
    confidence = round(0.7 + random.random() * 0.3, 3)
    return label, confidence


def classify_image(file_path: str) -> Tuple[str, float, str]:
    """Return (label, confidence, method) for the provided image file path.

    Tries DL model first (method='dl'), falls back to deterministic method
    (method='fallback') otherwise. The extra 'method' value is useful for
    debugging and for the frontend to show which path was used.
    """
    try:
        label, confidence = _predict_with_model(file_path)
        return label, confidence, 'dl'
    except Exception as e:
        # Print debug but continue with fallback
        print(f"[ImageClass] Model predict failed: {e}; using fallback classifier")
        label, confidence = _deterministic_fallback(file_path)
        return label, confidence, 'fallback'

