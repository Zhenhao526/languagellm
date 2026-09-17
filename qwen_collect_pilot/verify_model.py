"""Verify downloaded size and upstream SHA-256 before executing the pilot."""
import hashlib
import json
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent

def verify():
    manifest = json.loads((BASE/'model_manifest.json').read_text())
    model_dir = Path(manifest['local_path'])
    results = []
    for entry in manifest['files']:
        path = model_dir/entry['name']
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise RuntimeError(f'Missing or incomplete model file: {path.name}')
        result = {'file':entry['name'], 'bytes':path.stat().st_size}
        if entry['sha256']:
            h = hashlib.sha256()
            with path.open('rb') as f:
                while block := f.read(8*1024*1024):
                    h.update(block)
            result['sha256'] = h.hexdigest()
            if result['sha256'] != entry['sha256']:
                raise RuntimeError(f'SHA-256 mismatch: {path.name}')
        results.append(result)
        print(f'Verified {path.name}', flush=True)
    record = {'verified_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'repository':manifest['repository'], 'revision':manifest['revision'],
              'files':results}
    (BASE/'model_verification.json').write_text(json.dumps(record,indent=2))
    return record

if __name__ == '__main__':
    verify()
