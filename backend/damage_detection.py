from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from face_detection import (
    FaceBox,
    build_feature_mask,
    detect_faces,
)


@dataclass
class DamageResult:
    mask: np.ndarray
    crack_mask: np.ndarray
    scratch_mask: np.ndarray
    mark_mask: np.ndarray
    face_mask: np.ndarray
    core_mask: np.ndarray
    faces: list[FaceBox]
    protected_face_pixels: int
    stats: dict


def _odd(value: float, minimum: int = 3, maximum: int = 41) -> int:
    value = int(round(value))
    if value % 2 == 0:
        value += 1
    return max(minimum, min(maximum, value))


def _line_kernel(length: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    half = (length - 1) / 2.0
    angle = math.radians(angle_degrees)
    dx = int(round(math.cos(angle) * half))
    dy = int(round(math.sin(angle) * half))
    cv2.line(
        kernel,
        (center - dx, center - dy),
        (center + dx, center + dy),
        1,
        1,
        cv2.LINE_8,
    )
    return kernel


def _robust_threshold(
    values: np.ndarray, sensitivity: float, floor_value: float
) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    k = 4.2 - 2.4 * sensitivity
    threshold = median + k * sigma
    floor = floor_value * (1.0 - 0.35 * sensitivity)
    return max(threshold, floor)


def _component_shape(labels: np.ndarray, index: int) -> tuple[float, float, float]:
    ys, xs = np.where(labels == index)
    if xs.size < 3:
        return 0.0, 0.0, 0.0
    points = np.column_stack((xs, ys)).astype(np.float32)
    points -= points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 1e-6))
    length = math.sqrt(float(eigenvalues[-1])) * 3.46
    width = math.sqrt(float(eigenvalues[0])) * 3.46
    elongation = length / max(width, 0.5)
    return length, width, elongation


def _side_stats(
    gray: np.ndarray, labels: np.ndarray, index: int, width: float
) -> tuple[float, float]:
    ys, xs = np.where(labels == index)
    if xs.size < 4:
        return 999.0, 0.0
    points = np.column_stack((xs, ys)).astype(np.float32)
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    offset = max(3, int(round(width * 1.5 + 3)))
    height, image_width = gray.shape[:2]
    x1 = np.clip(np.round(xs + normal[0] * offset).astype(int), 0, image_width - 1)
    y1 = np.clip(np.round(ys + normal[1] * offset).astype(int), 0, height - 1)
    x2 = np.clip(np.round(xs - normal[0] * offset).astype(int), 0, image_width - 1)
    y2 = np.clip(np.round(ys - normal[1] * offset).astype(int), 0, height - 1)
    valid = (x1 != x2) | (y1 != y2)
    if not np.any(valid):
        return 999.0, 0.0
    side_a = gray[y1[valid], x1[valid]].astype(np.float32)
    side_b = gray[y2[valid], x2[valid]].astype(np.float32)
    line_pixels = gray[ys, xs].astype(np.float32)
    side_mean = (side_a + side_b) * 0.5
    side_difference = float(np.median(np.abs(side_a - side_b)))
    line_contrast = float(np.median(np.abs(line_pixels - side_mean)))
    return side_difference, line_contrast


def _filter_components(
    binary: np.ndarray,
    min_area: int,
    min_length: float,
    min_elongation: float,
    max_width: float,
    gray: np.ndarray | None = None,
    check_sides: bool = False,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    output = np.zeros_like(binary)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        box_w = int(stats[index, cv2.CC_STAT_WIDTH])
        box_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if max(box_w, box_h) < min_length * 0.5:
            continue
        length, width, elongation = _component_shape(labels, index)
        if length < min_length or elongation < min_elongation or width > max_width:
            continue
        if check_sides and gray is not None:
            side_difference, line_contrast = _side_stats(gray, labels, index, width)
            limit = max(16.0, min(34.0, line_contrast * 0.55))
            if side_difference > limit:
                continue
        output[labels == index] = 255
    return output


def _filter_blobs(
    binary: np.ndarray,
    gray: np.ndarray,
    gradient: np.ndarray,
    min_area: int,
    max_area: float,
    max_dimension: float,
    max_std: float,
    max_gradient: float,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    output = np.zeros_like(binary)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        box_w = int(stats[index, cv2.CC_STAT_WIDTH])
        box_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if box_w > max_dimension or box_h > max_dimension:
            continue
        component = labels == index
        if float(np.std(gray[component])) > max_std:
            continue
        if float(np.mean(gradient[component])) > max_gradient:
            continue
        output[component] = 255
    return output


def _detect_isolated_specks(gray: np.ndarray, sensitivity: float) -> np.ndarray:
    median = cv2.medianBlur(gray, 3)
    difference = cv2.absdiff(gray, median).astype(np.float32)
    threshold = _robust_threshold(difference, sensitivity, floor_value=34.0)
    binary = (difference >= threshold).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    output = np.zeros_like(binary)
    ring_kernel = np.ones((5, 5), np.uint8)
    inner_kernel = np.ones((3, 3), np.uint8)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 2 or area > 18:
            continue
        box_w = int(stats[index, cv2.CC_STAT_WIDTH])
        box_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if box_w > 7 or box_h > 7:
            continue
        component = (labels == index).astype(np.uint8)
        ring = cv2.dilate(component, ring_kernel) > 0
        inner = cv2.dilate(component, inner_kernel) > 0
        ring &= ~inner
        if not np.any(ring):
            continue
        if float(np.std(gray[ring])) <= 14.0:
            output[component > 0] = 255
    return output


def _shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _ridge_response(gray: np.ndarray) -> np.ndarray:
    offsets = (1.5, 2.5, 3.5)
    directions = 16
    source = gray.astype(np.float32)
    response = np.zeros_like(source)
    for offset in offsets:
        for step in range(directions):
            angle = math.pi * step / directions
            dx = math.cos(angle) * offset
            dy = math.sin(angle) * offset
            side_a = _shift(source, dx, dy)
            side_b = _shift(source, -dx, -dy)
            bright = source - np.maximum(side_a, side_b)
            dark = np.minimum(side_a, side_b) - source
            response = np.maximum(response, np.maximum(bright, dark))
    return response


def _ridge_threshold(response: np.ndarray, sensitivity: float) -> float:
    median = float(np.median(response))
    mad = float(np.median(np.abs(response - median)))
    sigma = 1.4826 * mad
    k = 9.5 - 2.0 * sensitivity
    floor = 20.0 - 6.0 * sensitivity
    return max(median + k * sigma, floor)


def detect_large_cracks(
    gray: np.ndarray, sensitivity: float
) -> np.ndarray:
    height, width = gray.shape[:2]
    short_side = min(height, width)
    work_max = 1536
    scale = min(1.0, work_max / float(max(height, width)))
    if scale < 1.0:
        work_w = max(32, int(round(width * scale)))
        work_h = max(32, int(round(height * scale)))
        work = cv2.resize(gray, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        work = gray
        work_w, work_h = width, height

    work_short = min(work_h, work_w)
    response = _ridge_response(work)
    threshold = _ridge_threshold(response, sensitivity)
    binary = (response >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    min_area = max(6, int(round(work_short * work_short * 0.00006)))
    min_length = max(12.0, work_short * 0.05)
    max_thickness = max(6.0, work_short * 0.016)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        length, _, _ = _component_shape(labels, index)
        if length < min_length:
            continue
        if area / max(length, 1.0) > max_thickness:
            continue
        cleaned[labels == index] = 255

    radius = max(1, min(3, int(round(work_short * 0.004))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    cleaned = cv2.dilate(cleaned, kernel)

    if scale < 1.0:
        cleaned = cv2.resize(cleaned, (width, height), interpolation=cv2.INTER_NEAREST)
        full_radius = max(1, min(4, int(round(short_side * 0.003))))
        cleaned = cv2.dilate(
            cleaned,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (full_radius * 2 + 1, full_radius * 2 + 1)
            ),
        )
    return cleaned


def _line_response(gray: np.ndarray, sensitivity: float) -> np.ndarray:
    height, width = gray.shape[:2]
    short_side = min(height, width)
    base = _odd(short_side * 0.014, minimum=7, maximum=19)
    lengths = sorted({base, _odd(base * 2.1, minimum=11, maximum=33)})
    angles = (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    response = np.zeros_like(gray, dtype=np.float32)
    for length in lengths:
        kernels = [_line_kernel(length, angle) for angle in angles]
        for kernel in kernels:
            black = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
            white = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, kernel)
            response = np.maximum(response, np.maximum(black, white).astype(np.float32))
    return response


def detect_cracks(
    gray: np.ndarray, sensitivity: float
) -> tuple[np.ndarray, np.ndarray]:
    height, width = gray.shape[:2]
    short_side = min(height, width)
    response = _line_response(gray, sensitivity)
    threshold = _robust_threshold(response, sensitivity, floor_value=15.0)
    binary = (response >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    min_length = max(10.0, min(36.0, short_side * 0.008))
    max_width = min(12.0, max(3.5, short_side * 0.012))
    mask = _filter_components(
        binary,
        min_area=6,
        min_length=min_length,
        min_elongation=3.0,
        max_width=max_width,
        gray=gray,
        check_sides=True,
    )
    return mask, response


def detect_scratches(
    gray: np.ndarray, sensitivity: float
) -> tuple[np.ndarray, np.ndarray]:
    height, width = gray.shape[:2]
    short_side = min(height, width)
    response = _line_response(gray, sensitivity)
    threshold = _robust_threshold(response, sensitivity, floor_value=16.0)
    binary = (response >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    min_length = max(12.0, min(44.0, short_side * 0.012))
    max_width = min(10.0, max(3.5, short_side * 0.013))
    lines = _filter_components(
        binary,
        min_area=8,
        min_length=min_length,
        min_elongation=3.4,
        max_width=max_width,
        gray=gray,
        check_sides=True,
    )
    specks = _detect_isolated_specks(gray, sensitivity)
    return cv2.bitwise_or(lines, specks), response


def detect_marks(
    rgb: np.ndarray, gray: np.ndarray, sensitivity: float
) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    short_side = min(height, width)
    total = height * width
    response = np.zeros((height, width), dtype=np.float32)
    marks = np.zeros((height, width), dtype=np.uint8)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )

    local = cv2.medianBlur(gray, 41)
    diff = cv2.absdiff(gray, local).astype(np.float32)
    threshold = _robust_threshold(diff, sensitivity, floor_value=44.0)
    binary = (diff >= threshold).astype(np.uint8) * 255
    luminance_marks = _filter_blobs(
        binary,
        gray,
        gradient,
        min_area=40,
        max_area=max(80.0, total * 0.008),
        max_dimension=max(12.0, short_side * 0.15),
        max_std=16.0,
        max_gradient=7.0,
    )
    marks = cv2.bitwise_or(marks, luminance_marks)
    response = np.maximum(response, diff)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    for channel in (1, 2):
        plane = lab[:, :, channel]
        local = cv2.medianBlur(plane, 31)
        diff = cv2.absdiff(plane, local).astype(np.float32)
        threshold = _robust_threshold(diff, sensitivity, floor_value=26.0)
        binary = (diff >= threshold).astype(np.uint8) * 255
        chroma_marks = _filter_blobs(
            binary,
            gray,
            gradient,
            min_area=30,
            max_area=max(60.0, total * 0.005),
            max_dimension=max(10.0, short_side * 0.12),
            max_std=16.0,
            max_gradient=6.0,
        )
        marks = cv2.bitwise_or(marks, chroma_marks)
        response = np.maximum(response, diff)

    return marks, response


def _linked_mask(mask: np.ndarray) -> np.ndarray:
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    output = np.zeros_like(closed)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= 3:
            output[labels == index] = 255
    return output


def protect_faces(
    mask: np.ndarray,
    line_mask: np.ndarray,
    face_mask: np.ndarray,
    core_mask: np.ndarray,
    faces: list[FaceBox],
) -> tuple[np.ndarray, int]:
    if not faces or np.count_nonzero(face_mask) == 0:
        return mask, 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = np.zeros_like(mask)
    removed = 0

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area == 0:
            continue
        component = labels == index
        face_overlap = int(np.count_nonzero(component & (face_mask > 0)))
        if face_overlap == 0:
            output[component] = 255
            continue

        line_overlap = int(np.count_nonzero(component & (line_mask > 0)))
        line_fraction = line_overlap / float(area)
        core_fraction = int(np.count_nonzero(component & (core_mask > 0))) / float(area)
        length, width, elongation = _component_shape(labels, index)

        overlapping_faces = []
        x, y, w, h = (
            int(stats[index, cv2.CC_STAT_LEFT]),
            int(stats[index, cv2.CC_STAT_TOP]),
            int(stats[index, cv2.CC_STAT_WIDTH]),
            int(stats[index, cv2.CC_STAT_HEIGHT]),
        )
        for face in faces:
            if (
                x < face.x2
                and x + w > face.x
                and y < face.y2
                and y + h > face.y
            ):
                overlapping_faces.append(face)
        face_size = max((max(f.w, f.h) for f in overlapping_faces), default=0)

        keep = False
        if face_size > 0 and line_fraction >= 0.30:
            long_enough = length >= max(10.0, face_size * 0.14)
            thin_enough = width <= max(3.5, face_size * 0.028)
            line_like = elongation >= 3.2 and long_enough and thin_enough
            crosses_core = core_fraction < 0.80
            if line_like and crosses_core:
                keep = True
            short_edge_mark = (
                core_fraction < 0.12
                and elongation >= 3.0
                and width <= max(3.0, face_size * 0.02)
            )
            if short_edge_mark:
                keep = True

        if keep:
            output[component] = 255
        else:
            removed += face_overlap

    return output, removed


def _cap_mask(
    mask: np.ndarray,
    crack_mask: np.ndarray,
    scratch_mask: np.ndarray,
    mark_mask: np.ndarray,
    sensitivity: float,
) -> np.ndarray:
    total = mask.size
    ratio = np.count_nonzero(mask) / float(total)
    max_ratio = 0.10 + 0.10 * sensitivity
    if ratio <= max_ratio:
        return mask

    priority = np.zeros(mask.shape, dtype=np.uint8)
    priority[crack_mask > 0] = 3
    priority[scratch_mask > 0] = np.maximum(priority[scratch_mask > 0], 2)
    priority[mark_mask > 0] = np.maximum(priority[mark_mask > 0], 1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area == 0:
            continue
        component = labels == index
        mean_priority = float(np.mean(priority[component]))
        components.append((mean_priority, area, index))
    components.sort(key=lambda item: (item[0], item[1]), reverse=True)

    output = np.zeros_like(mask)
    budget = int(total * max_ratio)
    used = 0
    for _, area, index in components:
        if used + area > budget:
            continue
        output[labels == index] = 255
        used += area
    return output


def detect_damage(rgb: np.ndarray, sensitivity: float = 0.5) -> DamageResult:
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = detect_faces(rgb)
    feature_mask = build_feature_mask(gray.shape, faces)

    crack_mask, _ = detect_cracks(gray, sensitivity)
    scratch_mask, _ = detect_scratches(gray, sensitivity)
    mark_mask, _ = detect_marks(rgb, gray, sensitivity)
    large_crack_mask = detect_large_cracks(gray, sensitivity)
    crack_mask = cv2.bitwise_or(crack_mask, large_crack_mask)

    combined = cv2.bitwise_or(cv2.bitwise_or(crack_mask, scratch_mask), mark_mask)
    combined = _linked_mask(combined)

    protected = cv2.bitwise_and(combined, cv2.bitwise_not(feature_mask))
    protected_pixels = int(np.count_nonzero(combined & feature_mask))
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8), iterations=1)
    protected = cv2.bitwise_and(protected, cv2.bitwise_not(feature_mask))
    protected = _cap_mask(
        protected, crack_mask, scratch_mask, mark_mask, sensitivity
    )

    damage_pixels = int(np.count_nonzero(protected))
    stats = {
        "crack_pixels": int(np.count_nonzero(crack_mask)),
        "large_crack_pixels": int(np.count_nonzero(large_crack_mask)),
        "scratch_pixels": int(np.count_nonzero(scratch_mask)),
        "mark_pixels": int(np.count_nonzero(mark_mask)),
        "damage_pixels": damage_pixels,
        "damage_ratio": damage_pixels / float(protected.size),
        "faces": len(faces),
        "protected_face_pixels": int(protected_pixels),
        "sensitivity": sensitivity,
    }
    return DamageResult(
        mask=protected,
        crack_mask=crack_mask,
        scratch_mask=scratch_mask,
        mark_mask=mark_mask,
        face_mask=feature_mask,
        core_mask=feature_mask,
        faces=faces,
        protected_face_pixels=protected_pixels,
        stats=stats,
    )
