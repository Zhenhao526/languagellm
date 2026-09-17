"""Seal the completed v0.31 formal batch without modifying its evidence."""
import argparse, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    out = args.out.resolve(); manifest = out / 'completion_manifest.json'
    assert not manifest.exists()
    invocation = read(out / 'invocation.json'); complete = read(out / 'training_complete.json')
    assert invocation['formal'] and complete['status'] == 'complete'
    for key in ('source_hashes', 'input_hashes'):
        assert invocation[key] == complete[key]
        for path, digest in invocation[key].items():
            assert sha(path) == digest, path
    for rel, digest in complete['files'].items():
        assert sha(out / rel) == digest, rel
    assert read(out / 'analysis.json')['formal'] and read(out / 'raw_validation.json')['passed']
    assert read(out / 'audit_execution.json')['passed'] and read(out / 'figures/visual_qa.json')['passed']
    assert read(ROOT / '结果审查.json')['passed_with_stated_limits']
    report = out / '固定伙伴与轮换伙伴中的共同符号形成研究报告.md'; assert report.is_file()

    # Keep the formal package self-contained enough to audit: include the formal
    # output, the smoke batch used by preflight, frozen source code and all
    # hash-bound inherited inputs/dependencies. Exclude abandoned partial smoke
    # directories and Python bytecode.
    files = set()
    for path in ROOT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name not in ('completion_manifest.json',):
            if 'results' in path.parts and any(part in ('smoke_001', 'smoke_002') for part in path.parts):
                continue
            files.add(path.resolve())
    files.update(Path(path) for path in invocation['source_hashes'])
    files.update(Path(path) for path in invocation['input_hashes'])
    audit_deps = read(out / 'audit_execution.json').get('audit_dependencies', {})
    files.update(Path(path) for path in audit_deps if Path(path).exists())
    for match in re.finditer(r'\]\((<?)([^)>]+)>?\)', report.read_text()):
        target = Path(match.group(2))
        if target.is_absolute():
            assert target.exists(), target
            files.add(target)
    artifacts = [{'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size} for path in sorted(files)]
    record = {
        'status': 'complete_development_batch', 'version': 'v0.31',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'study': 'fixed versus rotating partners in a multi-partner target task',
        'overall_research_goal_complete': False,
        'terminal': read(out / 'terminal_receipt.json'),
        'counts': {key: complete[key] for key in ('social_runs', 'pair_updates', 'messages', 'actions', 'new_private_fits', 'new_dino_inferences')},
        'report': str(report), 'artifacts': artifacts,
        'limitations': [
            'Four population members reuse two frozen private encoders; partner rotation is a controlled schedule, not natural demographic or generational change.',
            'The target is a symbolic resource-location protocol, not a reconstruction of human ecology.',
            'The bounded audit does not replay every gradient and Adam state transition; inherited visual and private inputs are hash-bound.',
            'Multi-partner simultaneous correctness remains low, and offline recombination cannot establish autonomous composition.',
            'The four initialization sources are the independent units; panels, directions, masks and photos are nested repeats.'
        ]
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': record['status'], 'artifacts': len(artifacts), 'manifest_sha256': sha(manifest)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
