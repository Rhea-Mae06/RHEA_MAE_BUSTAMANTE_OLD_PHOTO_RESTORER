from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"

LAMA_MODEL_PATH = Path(os.environ.get("LAMA_MODEL", MODELS_DIR / "big-lama.pt"))
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
HAAR_FRONTAL_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
HAAR_PROFILE_PATH = MODELS_DIR / "haarcascade_profileface.xml"

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
HAAR_FRONTAL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)
HAAR_PROFILE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_profileface.xml"
)

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 60 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 120_000_000))
FULL_IMAGE_MAX_SIDE = int(os.environ.get("FULL_IMAGE_MAX_SIDE", 1600))
TILE_SIZE = int(os.environ.get("RESTORE_TILE_SIZE", 960))
TILE_PADDING = int(os.environ.get("RESTORE_TILE_PADDING", 128))
MAX_TILES = int(os.environ.get("RESTORE_MAX_TILES", 16))
