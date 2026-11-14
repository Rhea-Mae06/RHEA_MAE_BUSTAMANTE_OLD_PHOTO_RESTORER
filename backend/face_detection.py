from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import (
    HAAR_FRONTAL_PATH,
    HAAR_FRONTAL_URL,
    HAAR_PROFILE_PATH,
    HAAR_PROFILE_URL,
    YUNET_MODEL_PATH,
    YUNET_MODEL_URL,
)
from downloads import ensure_file


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 0.0
    landmarks: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)


def _iou(a: FaceBox, b: FaceBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / float(a.area + b.area - inter)


def _clip_box(box: FaceBox, width: int, height: int) -> FaceBox:
    x1 = max(0, min(width - 1, box.x))
    y1 = max(0, min(height - 1, box.y))
    x2 = max(x1 + 1, min(width, box.x2))
    y2 = max(y1 + 1, min(height, box.y2))
    return FaceBox(x1, y1, x2 - x1, y2 - y1, box.score, box.landmarks)


class FaceDetector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._yunet = None
        self._haar_frontal = None
        self._haar_profile = None

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if ensure_file(YUNET_MODEL_PATH, YUNET_MODEL_URL):
                try:
                    self._yunet = cv2.FaceDetectorYN.create(
                        str(YUNET_MODEL_PATH), "", (320, 320), 0.6, 0.3, 5000
                    )
                except Exception:
                    self._yunet = None
            if ensure_file(HAAR_FRONTAL_PATH, HAAR_FRONTAL_URL):
                try:
                    self._haar_frontal = cv2.CascadeClassifier(str(HAAR_FRONTAL_PATH))
                    if self._haar_frontal.empty():
                        self._haar_frontal = None
                except Exception:
                    self._haar_frontal = None
            if ensure_file(HAAR_PROFILE_PATH, HAAR_PROFILE_URL):
                try:
                    self._haar_profile = cv2.CascadeClassifier(str(HAAR_PROFILE_PATH))
                    if self._haar_profile.empty():
                        self._haar_profile = None
                except Exception:
                    self._haar_profile = None

    def available(self) -> bool:
        self._load()
        return self._yunet is not None or self._haar_frontal is not None

    def detect(self, rgb: np.ndarray) -> list[FaceBox]:
        self._load()
        if rgb.ndim != 3:
            return []
        height, width = rgb.shape[:2]
        scale = min(1.0, 1280.0 / max(height, width))
        if scale < 1.0:
            sw = max(32, int(round(width * scale)))
            sh = max(32, int(round(height * scale)))
            small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small = rgb
            sw, sh = width, height

        faces: list[FaceBox] = []
        if self._yunet is not None:
            try:
                bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
                self._yunet.setInputSize((sw, sh))
                _, detected = self._yunet.detect(bgr)
                if detected is not None:
                    inv = 1.0 / scale
                    for row in detected:
                        score = float(row[-1])
                        if score < 0.6:
                            continue
                        x, y, w, h = (float(v) * inv for v in row[:4])
                        landmarks = np.array(row[4:14], dtype=np.float32).reshape(5, 2) * inv
                        faces.append(
                            FaceBox(
                                int(round(x)),
                                int(round(y)),
                                int(round(w)),
                                int(round(h)),
                                score,
                                landmarks,
                            )
                        )
            except Exception:
                pass

        gray_small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray_small = cv2.equalizeHist(gray_small)
        min_side = max(24, int(min(sw, sh) * 0.035))
        haar_boxes: list[FaceBox] = []
        for cascade, flipped in (
            (self._haar_frontal, False),
            (self._haar_profile, False),
            (self._haar_profile, True),
        ):
            if cascade is None:
                continue
            source = cv2.flip(gray_small, 1) if flipped else gray_small
            try:
                found = cascade.detectMultiScale(
                    source, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
                )
            except Exception:
                continue
            for x, y, w, h in found:
                if flipped:
                    x = sw - x - w
                inv = 1.0 / scale
                haar_boxes.append(
                    FaceBox(
                        int(round(x * inv)),
                        int(round(y * inv)),
                        int(round(w * inv)),
                        int(round(h * inv)),
                        0.5,
                        None,
                    )
                )

        for box in haar_boxes:
            if not any(_iou(box, existing) > 0.4 for existing in faces):
                faces.append(box)

        merged: list[FaceBox] = []
        for box in sorted(faces, key=lambda b: b.score, reverse=True):
            box = _clip_box(box, width, height)
            if box.w < 8 or box.h < 8:
                continue
            if any(_iou(box, kept) > 0.55 for kept in merged):
                continue
            merged.append(box)

        return merged


_DETECTOR: Optional[FaceDetector] = None


def get_face_detector() -> FaceDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = FaceDetector()
    return _DETECTOR


def detect_faces(rgb: np.ndarray) -> list[FaceBox]:
    return get_face_detector().detect(rgb)


def build_feature_mask(shape: tuple[int, int], faces: list[FaceBox]) -> np.ndarray:
    height, width = shape[:2]
    feature_mask = np.zeros((height, width), dtype=np.uint8)
    for face in faces:
        if face.landmarks is not None and len(face.landmarks) >= 5:
            points = face.landmarks[:5]
            eye_rx = max(5, int(round(face.w * 0.12)))
            eye_ry = max(4, int(round(face.h * 0.085)))
            for cx, cy in points[0:2]:
                cv2.ellipse(
                    feature_mask,
                    (int(round(cx)), int(round(cy - face.h * 0.02))),
                    (eye_rx, eye_ry),
                    0,
                    0,
                    360,
                    255,
                    -1,
                )
            nx, ny = points[2]
            cv2.ellipse(
                feature_mask,
                (int(round(nx)), int(round(ny))),
                (max(4, int(round(face.w * 0.07))), max(4, int(round(face.h * 0.06)))),
                0,
                0,
                360,
                255,
                -1,
            )
            mx = int(round((points[3][0] + points[4][0]) * 0.5))
            my = int(round((points[3][1] + points[4][1]) * 0.5))
            mouth_rx = max(5, int(round(abs(points[3][0] - points[4][0]) * 0.55)))
            mouth_ry = max(4, int(round(face.h * 0.08)))
            cv2.ellipse(
                feature_mask,
                (mx, my),
                (mouth_rx, mouth_ry),
                0,
                0,
                360,
                255,
                -1,
            )
        else:
            x1 = int(round(face.x + face.w * 0.18))
            y1 = int(round(face.y + face.h * 0.18))
            x2 = int(round(face.x + face.w * 0.82))
            y2 = int(round(face.y + face.h * 0.78))
            cv2.rectangle(feature_mask, (x1, y1), (x2, y2), 255, -1)
    return feature_mask


def build_protection_masks(
    shape: tuple[int, int], faces: list[FaceBox]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape[:2]
    face_mask = np.zeros((height, width), dtype=np.uint8)
    core_mask = np.zeros((height, width), dtype=np.uint8)

    for face in faces:
        pad_x = max(2, int(face.w * 0.08))
        pad_y = max(2, int(face.h * 0.08))
        x1 = max(0, face.x - pad_x)
        y1 = max(0, face.y - pad_y)
        x2 = min(width - 1, face.x2 + pad_x)
        y2 = min(height - 1, face.y2 + pad_y)
        cv2.rectangle(face_mask, (x1, y1), (x2, y2), 255, -1)

        if face.landmarks is not None and len(face.landmarks) >= 5:
            points = face.landmarks[:5]
            cx1 = float(np.min(points[:, 0]))
            cy1 = float(np.min(points[:, 1]))
            cx2 = float(np.max(points[:, 0]))
            cy2 = float(np.max(points[:, 1]))
            spread_x = max(4.0, (cx2 - cx1) * 0.35)
            spread_y = max(4.0, (cy2 - cy1) * 0.55)
            core_x1 = max(0, int(cx1 - spread_x))
            core_y1 = max(0, int(cy1 - spread_y))
            core_x2 = min(width - 1, int(cx2 + spread_x))
            core_y2 = min(height - 1, int(cy2 + spread_y))
        else:
            core_x1 = max(0, int(face.x + face.w * 0.16))
            core_y1 = max(0, int(face.y + face.h * 0.15))
            core_x2 = min(width - 1, int(face.x + face.w * 0.84))
            core_y2 = min(height - 1, int(face.y + face.h * 0.82))
        cv2.rectangle(core_mask, (core_x1, core_y1), (core_x2, core_y2), 255, -1)

    return face_mask, core_mask
