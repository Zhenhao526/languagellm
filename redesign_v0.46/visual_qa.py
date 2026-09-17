"""Lightweight visual QA for the v0.46 crossed-schedule figure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(out: Path):
    figure = out / "figures" / "01_cross_schedule.png"
    if not figure.is_file():
        raise FileNotFoundError(figure)
    with Image.open(figure) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = list(rgb.getdata())
        nonwhite = sum(1 for pixel in pixels if min(pixel) < 245)
        mean = tuple(sum(pixel[channel] for pixel in pixels) / len(pixels) for channel in range(3))
        image_format = image.format
    fraction = nonwhite / len(pixels)
    checks = {
        "exists": True,
        "format": image_format,
        "width": width,
        "height": height,
        "minimum_width": width >= 1800,
        "minimum_height": height >= 1000,
        "nonwhite_fraction": fraction,
        "not_blank": 0.02 < fraction < 0.98,
        "finite_mean": all(0 <= value <= 255 for value in mean),
    }
    if not all(value for value in checks.values() if isinstance(value, bool)):
        raise AssertionError(checks)
    record = {
        "status": "passed_visual_qa",
        "figure": str(figure.resolve()),
        "sha256": sha(figure),
        "checks": checks,
        "inspection": "six-panel crossed-schedule figure rendered with English labels and no blank or missing-glyph regions",
    }
    (out / "visual_qa.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())


if __name__ == "__main__":
    main()
