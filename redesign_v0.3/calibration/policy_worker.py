"""Official VPT inference in its own Python/PyTorch process.

Wire protocol: UTF-8 JSON lines. The only policy observation accepted is uint8
RGB. Reset seeds and protocol IDs are control data, not model observations.
No world state, reward, textual instruction, or scripted action is accepted.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
from pathlib import Path
import random
import sys
import time

BASE = Path(__file__).resolve().parent
DEPLOYMENT = BASE.parent / "deployment"
sys.path.insert(0, str(DEPLOYMENT / "runtime"))

import cv2
import numpy as np
import torch
from gym3.types import DictType
from lib.action_mapping import CameraHierarchicalMapping
from lib.actions import ActionTransformer, Buttons
from lib.policy import MinecraftAgentPolicy
from lib.torch_util import set_default_torch_device

PROTOCOL = "vpt-rgb-jsonl-v1"
MAX_LINE_BYTES = 2_000_000
SUPPORTED_SHAPES = {(128, 128, 3), (360, 640, 3)}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PolicyWorker:
    def __init__(self, device: str, weights: Path, config_path: Path):
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS is unavailable; no silent CPU fallback")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        self.device = device
        self.mapper = CameraHierarchicalMapping(n_camera_bins=11)
        self.transformer = ActionTransformer(
            camera_binsize=2, camera_maxval=10, camera_mu=10,
            camera_quantization_scheme="mu_law",
        )
        config = json.loads(config_path.read_text())
        policy_kwargs = config["model"]["args"]["net"]["args"]
        pi_head_kwargs = config["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        set_default_torch_device("cpu")
        start = time.perf_counter()
        self.policy = MinecraftAgentPolicy(
            policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs,
            action_space=DictType(**self.mapper.get_action_space_update()),
        )
        tensors = torch.load(weights, map_location="cpu", weights_only=True)
        self.policy.load_state_dict(tensors, strict=True)
        del tensors
        set_default_torch_device(device)
        self.policy.to(device).eval()
        self.policy.requires_grad_(False)
        # Official agent.py uses a false first flag and explicit initial_state().
        self.first = torch.tensor([False], dtype=torch.bool, device=device)
        self.state = None
        self.step = 0
        self.seed = None
        self.ready = {
            "type": "ready", "protocol": PROTOCOL,
            "model": "OpenAI VPT foundation-model-2x",
            "parameters": sum(p.numel() for p in self.policy.parameters()),
            "device": device, "torch": torch.__version__,
            "numpy": np.__version__, "python": sys.version.split()[0],
            "strict_state_dict_load": True, "stochastic": True,
            "temperature": pi_head_kwargs["temperature"],
            "input": "uint8 RGB only; INTER_LINEAR resize to 128x128",
            "policy_action_keys": Buttons.ALL + ["camera"],
            "environment_keys_not_predicted": ["ESC", "pickItem", "swapHands"],
            "warmup_frames": 0, "load_seconds": time.perf_counter() - start,
            "weights": str(weights.resolve()), "config": str(config_path.resolve()),
            "checkpoint_sha256": sha256(weights), "config_sha256": sha256(config_path),
        }

    def reset(self, seed: int):
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "mps":
            torch.mps.manual_seed(seed)
        self.state = self.policy.initial_state(1)
        self.seed = seed
        self.step = 0
        return {"seed": seed, "step": 0}

    @torch.no_grad()
    def act(self, payload: dict):
        if self.state is None:
            raise ValueError("Call reset(seed) before act(frame)")
        if type(payload) is not dict or set(payload) != {"shape", "rgb_base64"}:
            raise ValueError("frame must contain only shape and rgb_base64")
        shape = payload["shape"]
        if not isinstance(shape, list) or any(type(x) is not int for x in shape):
            raise ValueError("shape must be a list of integers")
        shape = tuple(shape)
        if shape not in SUPPORTED_SHAPES:
            raise ValueError("RGB shape must be [128,128,3] or [360,640,3]")
        if type(payload["rgb_base64"]) is not str:
            raise ValueError("rgb_base64 must be a string")
        raw = base64.b64decode(payload["rgb_base64"], validate=True)
        if len(raw) != int(np.prod(shape)):
            raise ValueError("RGB byte count does not match shape")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        # Match the official agent, including its exact image interpolation.
        frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(frame[None]).to(self.device)
        action, next_state, info = self.policy.act(
            {"img": image}, self.first, self.state,
            stochastic=True, return_pd=True,
        )
        finite = bool(torch.isfinite(info["log_prob"]).all()) and all(
            bool(torch.isfinite(value).all()) for value in info["pd"].values()
        )
        if not finite:
            raise FloatingPointError("VPT emitted non-finite policy data")
        # CPU NumPy conversion before mapping follows official agent.py.
        numpy_action = {key: value.cpu().numpy() for key, value in action.items()}
        native = self.transformer.policy2env(self.mapper.to_factored(numpy_action))
        # Remove only the known batch dimension; camera remains a length-2 list.
        native = {key: np.asarray(value)[0].tolist() for key, value in native.items()}
        if set(native) != set(Buttons.ALL + ["camera"]):
            raise RuntimeError("Unexpected official policy action keys")
        if not all(native[key] in (0, 1) for key in Buttons.ALL):
            raise RuntimeError("Non-binary native button output")
        if len(native["camera"]) != 2 or not np.isfinite(native["camera"]).all():
            raise RuntimeError("Invalid native camera output")
        self.state = next_state
        self.step += 1
        return {
            "action": native, "step": self.step,
            "finite_policy": finite,
            "log_prob": float(info["log_prob"].cpu().item()),
        }

    def handle(self, request: dict):
        if type(request) is not dict:
            raise ValueError("Request must be an object")
        op = request.get("op")
        allowed = {"reset": {"id", "op", "seed"},
                   "act": {"id", "op", "frame"}, "close": {"id", "op"}}
        if op not in allowed or set(request) != allowed[op]:
            raise ValueError("Only reset(seed), act(RGB frame), and close are accepted")
        if op == "reset":
            return self.reset(request["seed"])
        if op == "act":
            return self.act(request["frame"])
        return {"closed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--weights", type=Path,
                        default=DEPLOYMENT / "weights/foundation-model-2x.weights")
    parser.add_argument("--config", type=Path, default=DEPLOYMENT / "weights/2x.config.json")
    args = parser.parse_args()
    protocol_stdout = sys.stdout

    def send(value):
        protocol_stdout.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
        protocol_stdout.flush()

    # Library logs must never corrupt the JSON-lines transport.
    with contextlib.redirect_stdout(sys.stderr):
        worker = PolicyWorker(args.device, args.weights, args.config)
    send(worker.ready)
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
        if not line:
            break
        if len(line) > MAX_LINE_BYTES:
            send({"id": None, "ok": False, "error": "Request exceeds protocol size limit"})
            break
        request = None
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = worker.handle(request)
            send({"id": request["id"], "ok": True, **result})
            if request["op"] == "close":
                break
        except Exception as exc:
            print(f"VPT request error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            send({"id": request.get("id") if isinstance(request, dict) else None,
                  "ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
