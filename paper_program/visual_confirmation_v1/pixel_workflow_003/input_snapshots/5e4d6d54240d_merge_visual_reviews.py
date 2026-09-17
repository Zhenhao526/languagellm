"""Mechanical merging of two sealed AI reviews; no visual adjudication."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(folder, sha_a, sha_b):
    root = HERE/folder
    a_path, b_path = root/'review_a.json', root/'review_b.json'
    assert sha(a_path) == sha_a and sha(b_path) == sha_b, 'Review differs from communicated seal'
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    packet = json.loads((root/'packet.json').read_text())
    packet_hash = sha(root/'packet.json')
    assert a['packet_sha256'] == b['packet_sha256'] == packet_hash

    def parse(obj):
        rows = obj.get('decisions', obj.get('reviews'))
        parsed = {}
        for r in rows:
            c = r['criteria']
            criteria = {x['criterion']: x['value'] for x in c} if isinstance(c, list) else c
            assert set(criteria) == set(packet['criteria'])
            passed = all(value is True for value in criteria.values())
            if 'pass_visual' in r:
                assert r['pass_visual'] is passed
            parsed[r['id']] = {'criteria': criteria, 'pass': passed}
        assert len(parsed) == len(rows) and set(parsed) == set(packet['ids'])
        return parsed

    ra, rb = parse(a), parse(b)
    merged = []
    for ident in packet['ids']:
        ca, cb = ra[ident], rb[ident]
        merged.append({'id': ident, 'a_pass': ca['pass'], 'b_pass': cb['pass'],
            'dual_visual_pass': ca['pass'] and cb['pass'],
            'decision': 'dual_visual_pass_pending_identity_license' if ca['pass'] and cb['pass'] else 'conservative_visual_exclusion',
            'pass_disagreement': ca['pass'] != cb['pass'],
            'criterion_disagreements': [c for c in packet['criteria'] if ca['criteria'][c] != cb['criteria'][c]],
            'a_criteria': ca['criteria'], 'b_criteria': cb['criteria']})
    result = {'status': 'sealed_reviews_mechanically_merged', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'review_kind': 'Two independent AI visual reviews, not human judgments',
        'source_hashes': {str(a_path): sha_a, str(b_path): sha_b, str(root/'packet.json'): packet_hash},
        'merger_source_sha256': sha(__file__), 'records': merged, 'reviewed': len(merged),
        'dual_pass': sum(r['dual_visual_pass'] for r in merged),
        'pass_disagreements': sum(r['pass_disagreement'] for r in merged),
        'rule': 'All5 criteria must literally be true in both sealed reviews. Uncertain or disagreement excludes; no original judgment rewritten.',
        'source_identity_license_inferred': False, 'new_pixel_views': 0, 'new_model_calls': 0}
    path = root/'visual_merge.json'
    assert not path.exists(), 'Do not overwrite sealed merge'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    assert sha(a_path) == sha_a and sha(b_path) == sha_b
    print(json.dumps({k: result[k] for k in ('status', 'reviewed', 'dual_pass', 'pass_disagreements')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--sha-a', required=True)
    parser.add_argument('--sha-b', required=True)
    args = parser.parse_args()
    main(args.folder, args.sha_a, args.sha_b)
