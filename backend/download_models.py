from __future__ import annotations

from config import (
    HAAR_FRONTAL_PATH,
    HAAR_FRONTAL_URL,
    HAAR_PROFILE_PATH,
    HAAR_PROFILE_URL,
    LAMA_MODEL_PATH,
    LAMA_MODEL_URL,
    YUNET_MODEL_PATH,
    YUNET_MODEL_URL,
)
from downloads import ensure_file

MODELS = [
    ("LaMa inpainting model", LAMA_MODEL_PATH, LAMA_MODEL_URL, 100_000_000),
    ("YuNet face detector", YUNET_MODEL_PATH, YUNET_MODEL_URL, 100_000),
    ("Haar frontal face", HAAR_FRONTAL_PATH, HAAR_FRONTAL_URL, 100_000),
    ("Haar profile face", HAAR_PROFILE_PATH, HAAR_PROFILE_URL, 100_000),
]


def main() -> int:
    failed = False
    for label, path, url, minimum in MODELS:
        print(f"Checking {label}...")
        if ensure_file(path, url, min_bytes=minimum):
            print(f"  ready: {path} ({path.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            failed = True
            print(f"  could not download {url}")
            print(f"  place the file at: {path}")
    if failed:
        print("Some models are missing. AI inpainting may fall back to OpenCV.")
        return 1
    print("All models are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
