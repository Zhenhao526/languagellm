"""Read-only verification of this completed finite batch; no network or models."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = read(HERE / 'artifact_manifest.json')
    checks = 0
    for row in manifest['artifacts']:
        assert sha(row['path']) == row['sha256'], row['path']
        checks += 1
    for path, field in [
        (HERE/'source_001/freeze.json', 'input_hashes'),
        (HERE/'pixel_001/freeze.json', 'input_hashes'),
        (HERE/'source_decisions.json', 'evidence_hashes'),
        (HERE/'curation_result.json', 'evidence_hashes'),
    ]:
        for target, digest in read(path)[field].items():
            assert sha(target) == digest, target
            checks += 1
    result = read(HERE/'curation_result.json')
    assert result['limited_workflow_usable'] == 3
    assert result['model_calls'] == result['new_training_runs'] == 0
    assert result['reserve_pixels'] == result['confirmation_v1_pixels'] == 0
    assert sum(r['limited_workflow_usable'] for r in result['rows']) == 3
    assert not read(HERE/'pixel_001/independent_technical_qa.json')['failures']
    print(json.dumps(dict(status='passed', artifacts=len(manifest['artifacts']),
        hash_checks=checks, limited_workflow_usable=3, network_requests=0, model_calls=0)))


if __name__ == '__main__':
    main()
