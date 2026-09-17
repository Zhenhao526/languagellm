"""Small deterministic raster QA for v0.35 figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import numpy as np


ROOT = Path(__file__).resolve().parent


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve(); figure_dir = out / "figures"
    names = ("01_adaptation_outcomes.png",)
    figures = {}; checks = []
    for name in names:
        path = figure_dir / name
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
            h, w = rgb.shape[:2]
        nonwhite = np.mean(np.any(rgb < 248, axis=2))
        dimensions = w >= 1800 and h >= 900
        nonwhite_range = .01 < nonwhite < .80
        figures[name] = {"sha256": sha(path), "width": int(w), "height": int(h), "nonwhite_fraction": float(nonwhite), "readable_dimensions": bool(dimensions)}
        checks.append({"file": name, "dimensions": bool(dimensions), "nonwhite_range": bool(nonwhite_range)})
    passed = all(item["dimensions"] and item["nonwhite_range"] for item in checks)
    result = {"status": "passed" if passed else "failed", "passed": passed, "figures": figures, "checks": checks}
    (figure_dir / "visual_qa.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    source = json.loads((figure_dir / "figure_source.json").read_text()); source["status"] = "passed_visual_QA"; source["visual_qa_sha256"] = sha(figure_dir / "visual_qa.json"); source["visual_qa"] = result
    (figure_dir / "figure_source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
