"""Load the official pretrained VPT policy and measure local replay inference.

This verifies loading and action inference, not closed-loop Minecraft competence.
No model training, simulated survival result, or emergent-language score is made.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'runtime'))

import cv2
import numpy as np
import torch
from gym3.types import DictType
from lib.action_mapping import CameraHierarchicalMapping
from lib.actions import ActionTransformer
from lib.policy import MinecraftAgentPolicy
from lib.torch_util import set_default_torch_device


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def frames_from_video(path, n):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode official replay: {path}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Replay has fewer frames than requested')
        # This is the same RGB, INTER_LINEAR, 128x128 input format as official VPT.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR))
    cap.release()
    return np.stack(frames)


@torch.no_grad()
def measure(policy, frames, device, mapper, transformer):
    set_default_torch_device(device)
    policy.to(device).eval()
    inputs = torch.from_numpy(frames).to(device)
    state = policy.initial_state(1)
    for i in range(min(4, len(inputs))):
        _, state, _ = policy.act({'img': inputs[i:i+1]},
            torch.tensor([i == 0], device=device), state, stochastic=False)
    sync(device)
    state = policy.initial_state(1)
    durations = []
    actions = []
    decoded = []
    finite = True
    for i in range(len(inputs)):
        sync(device)
        start = time.perf_counter()
        action, state, info = policy.act({'img': inputs[i:i+1]},
            torch.tensor([i == 0], device=device), state, stochastic=False, return_pd=True)
        sync(device)
        durations.append(time.perf_counter() - start)
        finite = finite and bool(torch.isfinite(info['log_prob']).all())
        finite = finite and all(bool(torch.isfinite(v).all()) for v in info['pd'].values())
        numpy_action = {k: v.cpu().numpy() for k, v in action.items()}
        actions.append({k: v.tolist() for k, v in numpy_action.items()})
        env_action = transformer.policy2env(mapper.to_factored(numpy_action))
        decoded.append({k: np.asarray(v).tolist() for k, v in env_action.items()})
    result = {
        'device': device, 'frames': len(frames), 'seconds': sum(durations),
        'median_ms': float(np.median(durations) * 1000),
        'p95_ms': float(np.quantile(durations, .95) * 1000),
        'frames_per_second': len(frames) / sum(durations),
        'finite_log_prob_and_distribution': finite,
        'policy_actions': actions, 'decoded_actions': decoded,
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if device == 'mps':
        result['mps_driver_allocated_bytes_at_end'] = torch.mps.driver_allocated_memory()
        result['mps_tensor_allocated_bytes_at_end'] = torch.mps.current_allocated_memory()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, default=BASE/'weights/official_minecraft_sample.mp4')
    parser.add_argument('--weights', type=Path, default=BASE/'weights/foundation-model-2x.weights')
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--device', choices=['cpu', 'mps', 'both'], default='both')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = json.loads((BASE/'weights/2x.config.json').read_text())
    policy_kwargs = config['model']['args']['net']['args']
    pi_head_kwargs = config['model']['args']['pi_head_opts']
    pi_head_kwargs['temperature'] = float(pi_head_kwargs['temperature'])
    mapper = CameraHierarchicalMapping(n_camera_bins=11)
    transformer = ActionTransformer(camera_binsize=2, camera_maxval=10,
                                    camera_mu=10, camera_quantization_scheme='mu_law')
    set_default_torch_device('cpu')
    start = time.perf_counter()
    policy = MinecraftAgentPolicy(policy_kwargs=policy_kwargs,
                                  pi_head_kwargs=pi_head_kwargs,
                                  action_space=DictType(**mapper.get_action_space_update()))
    weights = torch.load(args.weights, map_location='cpu', weights_only=True)
    # Strictly require every tensor: no silently random replacement of the policy.
    policy.load_state_dict(weights, strict=True)
    del weights
    load_seconds = time.perf_counter() - start
    frames = frames_from_video(args.video, args.frames)
    report = {
        'model': 'OpenAI VPT foundation-model-2x',
        'parameters': sum(p.numel() for p in policy.parameters()),
        'strict_state_dict_load': True, 'load_seconds': load_seconds,
        'weights_sha256': sha(args.weights), 'video_sha256': sha(args.video),
        'input': 'Published contractor Minecraft replay, source frames 100 onward, RGB128x128',
        'source_video': 'https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4',
        'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
        'dtype': 'float32', 'cpu_threads': torch.get_num_threads(),
        'mps_available': torch.backends.mps.is_available(),
        'scope': 'Recorded-image action inference only; not a live Minecraft task, no training, no communication experiment.',
        'measurements': [],
    }
    devices = ['cpu', 'mps'] if args.device == 'both' else [args.device]
    for device in devices:
        if device == 'mps' and not torch.backends.mps.is_available():
            report['measurements'].append({'device': device, 'error': 'MPS unavailable'})
            continue
        try:
            result = measure(policy, frames, device, mapper, transformer)
            report['measurements'].append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ('policy_actions','decoded_actions')}), flush=True)
        except Exception as exc:
            report['measurements'].append({'device': device, 'error': repr(exc)})
            print(json.dumps({'device': device, 'error': repr(exc)}), flush=True)
    success = [r for r in report['measurements'] if 'policy_actions' in r]
    if len(success) == 2:
        report['cpu_mps_identical_greedy_action_fraction'] = sum(
            a == b for a, b in zip(success[0]['policy_actions'], success[1]['policy_actions'])) / len(frames)
    (BASE/'inference_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'report': str(BASE/'inference_report.json'), 'parameters': report['parameters']}))
    if not success:
        raise RuntimeError('No backend completed pretrained policy inference')


if __name__ == '__main__':
    main()
