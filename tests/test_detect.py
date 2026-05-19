"""Smoke tests against the two example images."""
from __future__ import annotations

from pathlib import Path

import tifffile

from bubbles.detect import detect_bubbles

EX = Path(__file__).resolve().parents[1] / "temp"


def test_0097_finds_two_bubbles_left_side():
    img = tifffile.imread(EX / "test_00_run_0097.tif")
    dets = detect_bubbles(img)
    left = [d for d in dets if d.cx < img.shape[1] // 4]
    assert len(left) == 2, f"expected 2 left-side bubbles, got {len(left)}: {dets}"
    for d in left:
        assert 30 <= d.r <= 90
        assert d.hollowness > 15
        assert d.uniformity < 1.0


def test_0202_finds_multiple_bubbles():
    img = tifffile.imread(EX / "test_00_run_0202.tif")
    dets = detect_bubbles(img)
    assert len(dets) >= 5, f"expected at least 5 bubbles, got {len(dets)}: {dets}"
    for d in dets:
        assert d.hollowness > 8
        assert d.uniformity < 1.0
