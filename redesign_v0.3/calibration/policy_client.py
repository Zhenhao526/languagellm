"""NumPy-only client usable from the separate Python 3.10 MineRL runtime.

Example::
    with PolicyClient() as policy:
        policy.reset(seed=7)
        action = policy.act(observation["pov"])

Only pixels cross into the policy process. The caller can fill missing environment
keys from env.action_space.noop(); VPT does not predict ESC/pickItem/swapHands.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import selectors
import subprocess

import numpy as np

BASE = Path(__file__).resolve().parent
DEFAULT_PYTHON = BASE.parent / "deployment/.venv/bin/python"


class PolicyClient:
    def __init__(self, python=DEFAULT_PYTHON, device="mps", timeout=120.0):
        if device not in ("cpu", "mps"):
            raise ValueError("device must be cpu or mps")
        self.timeout = float(timeout)
        self._counter = 0
        self.last_info = None
        self._closed = False
        self.process = subprocess.Popen(
            [str(python), "-u", str(BASE / "policy_worker.py"), "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # Inherit stderr: logs remain visible and cannot fill an unread pipe.
            stderr=None, bufsize=0,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            self.metadata = self._read()
            if (self.metadata.get("type") != "ready" or
                    self.metadata.get("protocol") != "vpt-rgb-jsonl-v1"):
                raise RuntimeError(f"Invalid worker handshake: {self.metadata}")
        except BaseException:
            self._terminate()
            raise

    def _read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                self._terminate()
                raise TimeoutError(f"VPT worker did not reply within {self.timeout}s")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"VPT worker exited (code {self.process.poll()}); see stderr")
        return json.loads(line)

    def _request(self, op, **payload):
        if self._closed:
            raise RuntimeError("VPT client is closed")
        self._counter += 1
        request = {"id": self._counter, "op": op, **payload}
        encoded = (json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        # FileIO on a pipe may write fewer bytes than requested for 640x360 RGB.
        view = memoryview(encoded)
        while view:
            written = self.process.stdin.write(view)
            if not written:
                raise BrokenPipeError("VPT worker stopped accepting requests")
            view = view[written:]
        response = self._read()
        if response.get("id") != self._counter:
            raise RuntimeError("VPT response ID does not match request")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown VPT worker error"))
        return response

    def reset(self, seed=0):
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.last_info = None
        return self._request("reset", seed=seed)

    def act(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise TypeError("frame must be a uint8 NumPy RGB array")
        if frame.shape not in ((128, 128, 3), (360, 640, 3)):
            raise ValueError("frame must have shape (128,128,3) or (360,640,3)")
        payload = {"shape": list(frame.shape),
                   "rgb_base64": base64.b64encode(np.ascontiguousarray(frame).tobytes()).decode("ascii")}
        response = self._request("act", frame=payload)
        self.last_info = {key: value for key, value in response.items()
                          if key not in ("action", "id", "ok")}
        return response["action"]

    def _terminate(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is not None:
                pipe.close()
        self._closed = True

    def close(self):
        if self._closed:
            return
        try:
            if self.process.poll() is None:
                self._request("close")
                self.process.wait(timeout=5)
        finally:
            self._terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
