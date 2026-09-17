"""Deterministic raster QA for v0.39 figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); out = args.out.resolve(); directory = out / "figures"; names = ("01_triad_outcomes.png",); figures = {}; checks = []
    for name in names:
        path = directory / name
        with Image.open(path) as image: rgb = np.asarray(image.convert("RGB")); h, w = rgb.shape[:2]
        fraction = float(np.mean(np.any(rgb < 248, axis=2))); dimensions = w >= 1800 and h >= 900; nonwhite = .01 < fraction < .80; figures[name] = {"sha256": sha(path), "width": int(w), "height": int(h), "nonwhite_fraction": fraction, "readable_dimensions": dimensions}; checks.append({"file": name, "dimensions": dimensions, "nonwhite_range": nonwhite})
    passed = all(x["dimensions"] and x["nonwhite_range"] for x in checks); result = {"status": "passed" if passed else "failed", "passed": passed, "figures": figures, "checks": checks}; (directory / "visual_qa.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n"); source = json.loads((directory / "figure_source.json").read_text()); source["status"] = "passed_visual_QA"; source["visual_qa_sha256"] = sha(directory / "visual_qa.json"); source["visual_qa"] = result; (directory / "figure_source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n"); print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
