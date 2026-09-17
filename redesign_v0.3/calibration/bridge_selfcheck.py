"""Short real-replay bridge check, never a Minecraft competence measurement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

from policy_client import PolicyClient, DEFAULT_PYTHON


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("At least two frames are needed")
    video = Path(__file__).resolve().parent.parent / "deployment/weights/official_minecraft_sample.mp4"
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("Cannot decode official replay")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    frames = []
    for _ in range(args.frames):
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError("Not enough replay frames")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape != (360, 640, 3):
            rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
        frames.append(rgb)
    cap.release()
    small = [cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR) for frame in frames]
    started = time.perf_counter()
    with PolicyClient(python=args.worker_python, device=args.device) as policy:
        sequences = []
        all_finite = True
        for seed, sequence in [(1701, frames), (1701, frames), (1701, small), (1702, frames)]:
            policy.reset(seed)
            actions = []
            for i, frame in enumerate(sequence):
                actions.append(policy.act(frame))
                assert policy.last_info["step"] == i + 1
                all_finite = all_finite and policy.last_info["finite_policy"]
                assert all(np.isfinite(value).all() for value in actions[-1].values())
            sequences.append(actions)
        assert sequences[0] == sequences[1], "Reset with same seed was not repeatable"
        assert sequences[0] == sequences[2], "Transport resolution changed the official resized input"
        assert sequences[0] != sequences[3], "Different seeds unexpectedly produced identical sequences"
        assert all_finite
        report = {
            "scope": "Official replay pixel inference and process transport only; no live world or resource comprehension claim",
            "client_python": platform.python_version(), "client_numpy": np.__version__,
            "torch_imported_in_client": "torch" in sys.modules,
            "worker": policy.metadata,
            "frames_per_sequence": len(frames), "sequences": 4,
            "same_seed_reset_identical": sequences[0] == sequences[1],
            "native_and_preresized_rgb_identical": sequences[0] == sequences[2],
            "different_seed_changes_samples": sequences[0] != sequences[3],
            "all_finite": all_finite,
            "first_native_action": sequences[0][0],
            "total_seconds_including_load": time.perf_counter() - started,
        }
    report["worker_closed"] = policy.process.poll() == 0
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
