from __future__ import annotations

import io
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import MAX_IMAGE_PIXELS
from damage_detection import detect_damage
from inpainting import inpaint_regions


class RestorationError(Exception):
    pass


@dataclass
class RestorationResult:
    image_bytes: bytes
    meta: dict


def _load_image(data: bytes) -> Image.Image:
    if not data:
        raise RestorationError("The uploaded file is empty.")
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        image = Image.open(io.BytesIO(data))
        image.seek(0)
        image.load()
    except Image.DecompressionBombError as exc:
        raise RestorationError(
            "The image is too large to process safely. Please use an image under 120 megapixels."
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RestorationError(
            "The file is not a readable image. Please upload a JPG, PNG, WebP, BMP or TIFF."
        ) from exc

    try:
        transposed = ImageOps.exif_transpose(image)
        if transposed is not None:
            image = transposed
    except Exception:
        pass
    return image


def _tone_and_grain(
    original: np.ndarray, restored: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    height, width = original.shape[:2]
    short_side = min(height, width)
    sigma = max(5.0, min(28.0, short_side * 0.016))
    low_original = cv2.GaussianBlur(original, (0, 0), sigma).astype(np.float32)
    low_restored = cv2.GaussianBlur(restored, (0, 0), sigma).astype(np.float32)
    correction = (low_original - low_restored) * 0.75
    mask_float = (mask > 0).astype(np.float32)[:, :, None]
    output = restored.astype(np.float32) + correction * mask_float

    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    high_pass = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    valid = mask == 0
    if np.any(valid):
        sample = high_pass[valid]
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        noise_sigma = min(max(1.4826 * mad, 0.0), 8.0)
        if noise_sigma > 1.2:
            rng = np.random.default_rng()
            noise = rng.normal(0.0, noise_sigma * 0.55, (height, width)).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (0, 0), 0.5)
            output += noise[:, :, None] * mask_float

    return np.clip(output, 0, 255).astype(np.uint8)


def _composite(
    original: np.ndarray, restored: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    alpha = (mask > 0).astype(np.float32)
    feathered = cv2.GaussianBlur(alpha, (0, 0), 0.8) * 1.5
    feathered = np.minimum(feathered, alpha)[:, :, None]
    output = (
        original.astype(np.float32) * (1.0 - feathered)
        + restored.astype(np.float32) * feathered
    )
    output = np.clip(output, 0, 255).astype(np.uint8)
    output[mask == 0] = original[mask == 0]
    return output


def _save_png(
    image: Image.Image, icc_profile: bytes | None, dpi: tuple | None
) -> bytes:
    buffer = io.BytesIO()
    save_options = {"format": "PNG", "compress_level": 6}
    if icc_profile:
        save_options["icc_profile"] = icc_profile
    if dpi:
        try:
            save_options["dpi"] = dpi
        except Exception:
            pass
    image.save(buffer, **save_options)
    return buffer.getvalue()


def restore_image(data: bytes, sensitivity: float = 0.5) -> RestorationResult:
    started = time.perf_counter()
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    image = _load_image(data)
    original_mode = image.mode
    icc_profile = image.info.get("icc_profile")
    dpi = image.info.get("dpi")

    has_alpha = "A" in image.getbands() or (
        image.mode in ("P", "PA") and "transparency" in image.info
    )
    gray_input = original_mode in ("L", "LA", "1")
    alpha_channel = None
    if has_alpha:
        rgba = image.convert("RGBA")
        alpha_channel = np.array(rgba.getchannel("A"))
        rgb = np.array(rgba.convert("RGB"))
    else:
        rgb = np.array(image.convert("RGB"))

    damage = detect_damage(rgb, sensitivity)
    if not np.any(damage.mask):
        restored_rgb = rgb.copy()
        engine = "none"
        tiles = 0
    else:
        restored_rgb, engine, tiles = inpaint_regions(rgb, damage.mask)
        restored_rgb = _tone_and_grain(rgb, restored_rgb, damage.mask)
        restored_rgb = _composite(rgb, restored_rgb, damage.mask)

    if gray_input:
        gray_original = np.array(image.convert("L"))
        if np.any(damage.mask):
            gray_restored = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2GRAY)
            gray_restored = _composite_gray(gray_original, gray_restored, damage.mask)
        else:
            gray_restored = gray_original
        output_image = Image.fromarray(gray_restored, mode="L")
    else:
        output_image = Image.fromarray(restored_rgb, mode="RGB")

    if alpha_channel is not None:
        output_image = output_image.convert("RGBA")
        output_image.putalpha(Image.fromarray(alpha_channel, mode="L"))

    png = _save_png(output_image, icc_profile, dpi)
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    width, height = output_image.size
    meta = {
        "engine": engine,
        "damage_ratio": float(damage.stats["damage_ratio"]),
        "damage_pixels": int(damage.stats["damage_pixels"]),
        "crack_pixels": int(damage.stats["crack_pixels"]),
        "large_crack_pixels": int(damage.stats.get("large_crack_pixels", 0)),
        "scratch_pixels": int(damage.stats["scratch_pixels"]),
        "mark_pixels": int(damage.stats["mark_pixels"]),
        "faces": int(damage.stats["faces"]),
        "protected_face_pixels": int(damage.stats["protected_face_pixels"]),
        "width": int(width),
        "height": int(height),
        "mode": output_image.mode,
        "grayscale": bool(gray_input),
        "sensitivity": sensitivity,
        "tiles": int(tiles),
        "processing_ms": elapsed_ms,
    }
    return RestorationResult(image_bytes=png, meta=meta)


def _composite_gray(
    original: np.ndarray, restored: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    alpha = (mask > 0).astype(np.float32)
    feathered = cv2.GaussianBlur(alpha, (0, 0), 0.8) * 1.5
    feathered = np.minimum(feathered, alpha)
    output = (
        original.astype(np.float32) * (1.0 - feathered)
        + restored.astype(np.float32) * feathered
    )
    output = np.clip(output, 0, 255).astype(np.uint8)
    output[mask == 0] = original[mask == 0]
    return output
