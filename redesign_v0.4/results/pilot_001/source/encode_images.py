"""Cache frozen official DINOv2-L visual features; never use category labels."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def preprocess(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    w, h = im.size
    size = (round(w * 256 / min(w, h)), round(h * 256 / min(w, h)))
    im = im.resize(size, Image.Resampling.BICUBIC)
    x, y = (size[0] - 224) // 2, (size[1] - 224) // 2
    im = im.crop((x, y, x + 224, y + 224))
    arr = np.asarray(im).astype(np.float32) / 255
    arr = (arr - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(ROOT / 'data/manifest.json'))
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest['images'] if isinstance(manifest, dict) else manifest
    repo = ROOT / 'vendor/dinov2'
    sys.path.insert(0, str(repo))
    from dinov2.hub.backbones import dinov2_vitl14
    torch.set_num_threads(4)
    model = dinov2_vitl14(pretrained=False)
    weights = ROOT / 'weights/dinov2_vitl14_pretrain.pth'
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False).eval().to(args.device)
    outputs = []
    start = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(entries), 8):
            batch = entries[offset:offset + 8]
            tensors = []
            for row in batch:
                path = Path(row['path'])
                if not path.is_absolute():
                    path = ROOT / path
                tensors.append(preprocess(path))
            z = model(torch.stack(tensors).to(args.device))
            outputs.append(z.float().cpu().numpy())
            print(json.dumps({'encoded': offset + len(batch), 'total': len(entries)}), flush=True)
    features = np.concatenate(outputs)
    assert features.shape == (len(entries), 1024) and np.isfinite(features).all()
    np.savez_compressed(ROOT / 'data/features.npz', features=features)
    report = {
        'model': 'official DINOv2 ViT-L/14, backbone only',
        'parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters': 0,
        'repository': 'https://github.com/facebookresearch/dinov2',
        'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'weight_url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        'weight_sha256': sha(weights), 'weight_bytes': weights.stat().st_size,
        'weight_hash_status': 'locally computed fingerprint; no publisher SHA256 comparison',
        'strict_load': True, 'device': args.device, 'torch': torch.__version__,
        'images': len(entries), 'features_shape': list(features.shape),
        'manifest_sha256': sha(args.manifest), 'features_sha256': sha(ROOT / 'data/features.npz'),
        'seconds': time.monotonic() - start,
        'preprocess': 'RGB; resize short side 256 bicubic; center crop 224; ImageNet RGB normalization',
        'input': 'pixels only, no file names, category IDs, image descriptions, or text prompt',
    }
    (ROOT / 'data/encoder_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
