"""
Food Freshness Prediction Module
---------------------------------
Dual approach:
  1. If a trained Keras model (.h5) exists → CNN prediction
  2. Otherwise → OpenCV-based heuristic analysis (color, texture, edges)
"""

import os
import cv2
import numpy as np
from PIL import Image

# Path to the trained model (created by train.py)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "food_freshness_model.h5")
IMG_SIZE = (224, 224)

_model = None


def _load_keras_model():
    """Load the trained Keras model once."""
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model  # lazy import
        _model = load_model(MODEL_PATH)
    return _model


# ──────────────────────────────────────────────
#  CNN-based prediction
# ──────────────────────────────────────────────

def _predict_with_model(image_path: str) -> dict:
    """Run inference using the trained CNN model."""
    model = _load_keras_model()
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr, verbose=0)[0][0]
    # Model output: 0 → expired, 1 → fresh  (sigmoid)
    is_fresh = prediction >= 0.5
    confidence = float(prediction) if is_fresh else float(1 - prediction)

    return {
        "label": "Fresh" if is_fresh else "Expired",
        "confidence": round(confidence * 100, 1),
        "method": "CNN (MobileNetV2)",
        "details": {
            "color_score": None,
            "texture_score": None,
            "edge_score": None,
            "overall_score": round(confidence * 100, 1),
        },
    }


# ──────────────────────────────────────────────
#  OpenCV heuristic analysis (fallback)
# ──────────────────────────────────────────────

def _analyze_color(hsv: np.ndarray) -> dict:
    """
    Analyse colour distribution.
    Fresh foods tend toward vivid greens / reds / yellows.
    Expired foods shift toward browns / dark greens / greys.
    """
    h, s, v = cv2.split(hsv)

    # Saturation – vivid colours = fresh
    mean_sat = float(np.mean(s))
    sat_score = min(mean_sat / 120.0, 1.0)  # higher = fresher

    # Brightness – very dark patches hint at rot
    mean_val = float(np.mean(v))
    val_score = min(mean_val / 140.0, 1.0)

    # Brown-pixel ratio  (hue 10-25, low-medium sat)
    brown_mask = cv2.inRange(hsv, (8, 40, 20), (25, 200, 180))
    brown_ratio = float(np.sum(brown_mask > 0)) / brown_mask.size
    brown_penalty = min(brown_ratio * 4, 1.0)  # more brown = worse

    # Dark-spot ratio
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))
    dark_ratio = float(np.sum(dark_mask > 0)) / dark_mask.size
    dark_penalty = min(dark_ratio * 3, 1.0)

    score = (sat_score * 0.35 + val_score * 0.25
             + (1 - brown_penalty) * 0.25 + (1 - dark_penalty) * 0.15)
    return {"score": round(score * 100, 1),
            "mean_saturation": round(mean_sat, 1),
            "mean_brightness": round(mean_val, 1),
            "brown_ratio": round(brown_ratio * 100, 1),
            "dark_ratio": round(dark_ratio * 100, 1)}


def _analyze_texture(gray: np.ndarray) -> dict:
    """
    Use Laplacian variance to gauge surface texture.
    Smooth, uniform surfaces → fresh.
    Rough / blotchy → potential mould / decay.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())

    # Very high variance can indicate spots / mould; moderate is normal
    if variance < 500:
        tex_score = 0.85
    elif variance < 1500:
        tex_score = 0.70
    elif variance < 3000:
        tex_score = 0.50
    else:
        tex_score = 0.30

    return {"score": round(tex_score * 100, 1),
            "laplacian_variance": round(variance, 1)}


def _analyze_edges(gray: np.ndarray) -> dict:
    """
    Canny edge density.
    Excessive edges may indicate wrinkles, cracks, mould spots.
    """
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size

    if edge_density < 0.05:
        edge_score = 0.90
    elif edge_density < 0.12:
        edge_score = 0.75
    elif edge_density < 0.22:
        edge_score = 0.55
    else:
        edge_score = 0.30

    return {"score": round(edge_score * 100, 1),
            "edge_density": round(edge_density * 100, 1)}


def _predict_with_opencv(image_path: str) -> dict:
    """Full OpenCV-based heuristic pipeline."""
    img = cv2.imread(image_path)
    if img is None:
        return {"label": "Error", "confidence": 0,
                "method": "OpenCV", "details": {"error": "Cannot read image"}}

    img = cv2.resize(img, IMG_SIZE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color = _analyze_color(hsv)
    texture = _analyze_texture(gray)
    edges = _analyze_edges(gray)

    overall = (color["score"] * 0.50
               + texture["score"] * 0.30
               + edges["score"] * 0.20)

    is_fresh = overall >= 50
    confidence = overall if is_fresh else (100 - overall)

    return {
        "label": "Fresh" if is_fresh else "Expired",
        "confidence": round(confidence, 1),
        "method": "OpenCV Heuristic Analysis",
        "details": {
            "color_score": color["score"],
            "texture_score": texture["score"],
            "edge_score": edges["score"],
            "overall_score": round(overall, 1),
        },
    }


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def predict(image_path: str) -> dict:
    """Return freshness prediction for the given image file."""
    if _load_keras_model() is not None:
        return _predict_with_model(image_path)
    return _predict_with_opencv(image_path)
