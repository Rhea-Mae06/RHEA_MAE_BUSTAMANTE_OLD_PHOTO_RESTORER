from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np

from config import (
    FULL_IMAGE_MAX_SIDE,
    LAMA_MODEL_PATH,
    LAMA_MODEL_URL,
    MAX_TILES,
    TILE_PADDING,
    TILE_SIZE,
)
from downloads import ensure_file


def _pad_to_modulo(image: np.ndarray, mask: np.ndarray, modulo: int = 8):
    height, width = image.shape[:2]
    target_h = ((height + modulo - 1) // modulo) * modulo
    target_w = ((width + modulo - 1) // modulo) * modulo
    pad_h = target_h - height
    pad_w = target_w - width
    if pad_h == 0 and pad_w == 0:
        return image, mask
    padded_image = np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="symmetric",
    )
    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="symmetric")
    return padded_image, padded_mask


class LaMaInpainter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._model = None
        self._load_attempted = False
        self._load_error: Optional[str] = None
        self._torch = None

    def _load(self) -> None:
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                import torch

                torch.set_grad_enabled(False)
                try:
                    threads = max(1, (os.cpu_count() or 4) - 1)
                    torch.set_num_threads(threads)
                except Exception:
                    pass
                if not ensure_file(
                    LAMA_MODEL_PATH, LAMA_MODEL_URL, min_bytes=100_000_000
                ):
                    self._load_error = "LaMa model could not be downloaded."
                    return
                model = torch.jit.load(str(LAMA_MODEL_PATH), map_location="cpu")
                model.eval()
                self._torch = torch
                self._model = model
            except Exception as exc:
                self._load_error = str(exc)
                self._model = None

    @property
    def ready(self) -> bool:
        self._load()
        return self._model is not None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model_present(self) -> bool:
        return LAMA_MODEL_PATH.exists() and LAMA_MODEL_PATH.stat().st_size > 100_000_000

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._load()
        if self._model is None:
            raise RuntimeError(self._load_error or "LaMa model is unavailable.")
        torch = self._torch
        if torch is None:
            raise RuntimeError("PyTorch is unavailable.")

        padded_image, padded_mask = _pad_to_modulo(rgb, mask, 8)
        height, width = padded_image.shape[:2]
        image_tensor = (
            torch.from_numpy(padded_image.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        mask_tensor = (
            torch.from_numpy((padded_mask > 0).astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        with self._inference_lock, torch.inference_mode():
            output = self._model(image_tensor, mask_tensor)
        result = output[0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        return result[: rgb.shape[0], : rgb.shape[1]]


_INPAINTER: Optional[LaMaInpainter] = None


def get_inpainter() -> LaMaInpainter:
    global _INPAINTER
    if _INPAINTER is None:
        _INPAINTER = LaMaInpainter()
    return _INPAINTER


def _tile_plan(
    height: int, width: int, mask: np.ndarray
) -> list[tuple[int, int, int, int, int, int, int, int]]:
    tile = TILE_SIZE
    pad = TILE_PADDING
    stride = max(128, tile - 2 * pad)
    plan = []
    for y0 in range(0, height, stride):
        y1 = min(height, y0 + tile)
        if y1 - y0 < 32:
            continue
        core_y0 = 0 if y0 == 0 else min(y1, y0 + pad)
        core_y1 = height if y1 == height else max(core_y0, y1 - pad)
        for x0 in range(0, width, stride):
            x1 = min(width, x0 + tile)
            if x1 - x0 < 32:
                continue
            core_x0 = 0 if x0 == 0 else min(x1, x0 + pad)
            core_x1 = width if x1 == width else max(core_x0, x1 - pad)
            if core_x1 <= core_x0 or core_y1 <= core_y0:
                continue
            if not np.any(mask[core_y0:core_y1, core_x0:core_x1]):
                continue
            crop_x0 = max(0, x0 - pad)
            crop_y0 = max(0, y0 - pad)
            crop_x1 = min(width, x1 + pad)
            crop_y1 = min(height, y1 + pad)
            plan.append(
                (crop_x0, crop_y0, crop_x1, crop_y1, core_x0, core_y0, core_x1, core_y1)
            )
    return plan


def _inpaint_tiles(
    rgb: np.ndarray,
    mask: np.ndarray,
    plan: list[tuple[int, int, int, int, int, int, int, int]],
    inpainter: LaMaInpainter,
) -> np.ndarray:
    result = rgb.copy()
    for crop_x0, crop_y0, crop_x1, crop_y1, core_x0, core_y0, core_x1, core_y1 in plan:
        crop = rgb[crop_y0:crop_y1, crop_x0:crop_x1]
        crop_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]
        restored = inpainter.inpaint(crop, crop_mask)
        local_core = (
            slice(core_y0 - crop_y0, core_y1 - crop_y0),
            slice(core_x0 - crop_x0, core_x1 - crop_x0),
        )
        core_mask = crop_mask[local_core] > 0
        target = result[crop_y0:crop_y1, crop_x0:crop_x1]
        target_core = target[local_core]
        target_core[core_mask] = restored[local_core][core_mask]
        target[local_core] = target_core
        result[crop_y0:crop_y1, crop_x0:crop_x1] = target
    return result


def _inpaint_resized(
    rgb: np.ndarray, mask: np.ndarray, inpainter: LaMaInpainter
) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, FULL_IMAGE_MAX_SIDE / float(max(height, width)))
    if scale >= 1.0:
        return inpainter.inpaint(rgb, mask)
    small_w = max(8, int(round(width * scale)))
    small_h = max(8, int(round(height * scale)))
    small_rgb = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    restored_small = inpainter.inpaint(small_rgb, small_mask)
    restored = cv2.resize(restored_small, (width, height), interpolation=cv2.INTER_LINEAR)
    result = rgb.copy()
    damage = mask > 0
    result[damage] = restored[damage]
    return result


def inpaint_regions(
    rgb: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, str, int]:
    if not np.any(mask):
        return rgb.copy(), "none", 0

    inpainter = get_inpainter()
    if inpainter.ready:
        try:
            height, width = rgb.shape[:2]
            if max(height, width) <= FULL_IMAGE_MAX_SIDE:
                return inpainter.inpaint(rgb, mask), "lama", 1
            plan = _tile_plan(height, width, mask)
            if plan and len(plan) <= MAX_TILES:
                return _inpaint_tiles(rgb, mask, plan, inpainter), "lama-tiled", len(plan)
            return _inpaint_resized(rgb, mask, inpainter), "lama-resized", 0
        except Exception:
            pass

    fallback = cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA)
    result = rgb.copy()
    damage = mask > 0
    result[damage] = fallback[damage]
    return result, "opencv-telea", 0
