"""Fetch a revision-pinned MLX checkpoint, without executing repository code."""
import fnmatch
import argparse
import json
from pathlib import Path
import time
from huggingface_hub import HfApi, snapshot_download

BASE = Path(__file__).resolve().parent
REPO = 'mlx-community/Qwen3.5-9B-8bit'
PATTERNS = ['*.json', '*.safetensors', '*.jinja', '*.model', '*.txt', 'README.md', 'LICENSE*']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', help='Explicit checkpoint commit; otherwise reuse the existing manifest revision')
    args = parser.parse_args()
    out = BASE / 'models/Qwen3.5-9B-8bit'
    manifest_path = BASE/'model_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = args.revision or previous.get('revision')
    info = HfApi().model_info(REPO, revision=revision, files_metadata=True)
    files = [{'name': x.rfilename, 'bytes': x.size,
              'sha256': x.lfs.sha256 if x.lfs else None}
             for x in info.siblings if any(fnmatch.fnmatch(x.rfilename, p) for p in PATTERNS)]
    record = {'repository': REPO, 'revision': info.sha, 'files': files,
              'local_path': str(out), 'total_bytes': sum(x['bytes'] or 0 for x in files),
              'download_started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (BASE/'model_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False), flush=True)
    snapshot_download(REPO, revision=info.sha, local_dir=out,
                      allow_patterns=PATTERNS, max_workers=3)
    record['download_completed'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (BASE/'model_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print('DOWNLOAD_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
