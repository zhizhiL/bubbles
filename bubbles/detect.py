"""Hollow-bubble detection: HoughCircles + hollowness post-filter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class DetectionParams:
    min_radius: int = 9
    max_radius: int = 70
    min_dist: int = 60
    dp: float = 1.0
    canny_high: int = 80
    accumulator_threshold: int = 25
    blur_ksize: int = 5
    blur_sigma: float = 1.0
    hollowness_min: float = 8.0
    ring_uniformity_max: float = 1.0
    ring_thickness: int = 3


@dataclass
class Detection:
    cx: int
    cy: int
    r: int
    hollowness: float
    uniformity: float


def _ring_stats(img: np.ndarray, cx: int, cy: int, r: int) -> tuple[float, float, float]:
    """Return (inner_mean, ring_mean, ring_uniformity).

    `ring_mean` is the mean along the best-fit dark annulus (slid in r ∈ [0.85r,1.0r]).
    `ring_uniformity` = std of intensities sampled at N angles on that annulus, divided
    by (inner_mean - ring_mean). Low values mean a clean, uniformly dark ring (bubble).
    High values mean the dark pixels cluster on one side (irregular debris).
    """
    H, W = img.shape
    inner_r = max(2.0, 0.5 * r)
    pad = 2
    y0, y1 = max(0, cy - r - pad), min(H, cy + r + pad + 1)
    x0, x1 = max(0, cx - r - pad), min(W, cx + r + pad + 1)
    patch = img[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    inner_m = d < inner_r
    if not inner_m.any():
        return 0.0, 0.0, np.inf
    inner_mean = float(patch[inner_m].mean())

    # Find the radius r* in [0.85r, 1.0r] with darkest annular mean
    best = (np.inf, r)
    for rr in range(max(1, int(0.85 * r)), int(1.0 * r) + 1):
        m = (d >= rr - 0.5) & (d < rr + 0.5)
        if m.any():
            v = float(patch[m].mean())
            if v < best[0]:
                best = (v, rr)
    ring_mean, r_star = best
    if not np.isfinite(ring_mean):
        return inner_mean, inner_mean, np.inf

    # Sample at N angles on that annulus to assess angular uniformity
    n = max(36, int(2 * np.pi * r_star))
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    sx = np.round(cx + r_star * np.cos(theta)).astype(int)
    sy = np.round(cy + r_star * np.sin(theta)).astype(int)
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    if ok.sum() < n * 0.8:
        return inner_mean, ring_mean, np.inf
    vals = img[sy[ok], sx[ok]].astype(np.float32)
    contrast = max(inner_mean - ring_mean, 1e-3)
    uniformity = float(vals.std()) / contrast
    return inner_mean, ring_mean, uniformity


def _hollowness(img: np.ndarray, cx: int, cy: int, r: int, thickness: int = 0) -> float:
    inner_mean, ring_mean, _ = _ring_stats(img, cx, cy, r)
    return inner_mean - ring_mean


def _preprocess(img: np.ndarray, p: DetectionParams) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = img.astype(np.uint8)
    k = p.blur_ksize | 1
    blurred = cv2.GaussianBlur(g, (k, k), p.blur_sigma)
    return cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def detect_bubbles(img: np.ndarray, params: DetectionParams | None = None) -> list[Detection]:
    p = params or DetectionParams()
    raw_gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    work = _preprocess(img, p)

    circles = cv2.HoughCircles(
        work,
        cv2.HOUGH_GRADIENT,
        dp=p.dp,
        minDist=p.min_dist,
        param1=p.canny_high,
        param2=p.accumulator_threshold,
        minRadius=p.min_radius,
        maxRadius=p.max_radius,
    )
    if circles is None:
        return []

    out: list[Detection] = []
    for c in circles[0]:
        cx, cy, r = int(round(c[0])), int(round(c[1])), int(round(c[2]))
        inner_mean, ring_mean, uniformity = _ring_stats(raw_gray, cx, cy, r)
        h = inner_mean - ring_mean
        if h < p.hollowness_min:
            continue
        if uniformity > p.ring_uniformity_max:
            continue
        out.append(Detection(cx, cy, r, h, uniformity))
    return out


def draw_overlay(img: np.ndarray, dets: Sequence[Detection]) -> np.ndarray:
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    for d in dets:
        cv2.circle(out, (d.cx, d.cy), d.r, (0, 255, 0), 2)
        cv2.circle(out, (d.cx, d.cy), 2, (0, 0, 255), -1)
    return out
