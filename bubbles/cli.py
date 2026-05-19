"""Batch detect bubbles across a glob of .tif files, write CSV + overlays."""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2
import tifffile

from .detect import DetectionParams, detect_bubbles, draw_overlay


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect hollow bubbles in TIFF frames.")
    ap.add_argument("inputs", nargs="+", help="TIFF files or glob patterns")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--min-radius", type=int, default=9)
    ap.add_argument("--max-radius", type=int, default=70)
    ap.add_argument("--min-dist", type=int, default=60)
    ap.add_argument("--canny-high", type=int, default=80)
    ap.add_argument("--acc-threshold", type=int, default=25)
    ap.add_argument("--hollowness-min", type=float, default=8.0)
    ap.add_argument("--uniformity-max", type=float, default=1.0)
    args = ap.parse_args()

    params = DetectionParams(
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_dist=args.min_dist,
        canny_high=args.canny_high,
        accumulator_threshold=args.acc_threshold,
        hollowness_min=args.hollowness_min,
        ring_uniformity_max=args.uniformity_max,
    )

    paths: list[str] = []
    for pat in args.inputs:
        matched = sorted(glob.glob(pat))
        paths.extend(matched if matched else [pat])

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "detections.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cx", "cy", "r", "hollowness", "uniformity"])
        for p in paths:
            img = tifffile.imread(p)
            dets = detect_bubbles(img, params)
            name = os.path.splitext(os.path.basename(p))[0]
            for d in dets:
                w.writerow([name, d.cx, d.cy, d.r, f"{d.hollowness:.2f}", f"{d.uniformity:.2f}"])
            if not args.no_overlay:
                overlay = draw_overlay(img, dets)
                cv2.imwrite(os.path.join(args.out, f"{name}_overlay.png"), overlay)
            print(f"{name}: {len(dets)} bubbles")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
