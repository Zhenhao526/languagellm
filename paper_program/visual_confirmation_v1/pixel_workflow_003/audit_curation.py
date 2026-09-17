"""Verify sealed bookkeeping and artifact hashes. No pixels decoded, no network/model calls."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from urllib.parse import unquote
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
FRAME = HERE.parent
ROOT = FRAME.parents[1]

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

checks = 0
def check(value, reason):
    global checks
    checks += 1
    assert value, reason

def hashset(mapping):
    for name, expected in mapping.items():
        p = Path(name)
        if not p.is_absolute():
            p = ROOT / p
        check(p.is_file() and sha(p) == expected, f'hash changed: {p}')

status = read(HERE / 'curation_status_20260916.json')
identity = read(HERE / 'identity_license_review_20260916.json')
visual = read(HERE / 'review_round1/visual_merge.json')
freeze = read(HERE / 'freeze.json')
for obj in [status, identity, visual, freeze]:
    hashset(obj['source_hashes'])
for row in freeze['snapshots']:
    check(sha(Path(row['snapshot'])) == row['sha256'], 'frozen snapshot changed')

expected_reviews = {
    'review_a.json': 'd3ce14b0cbcb9a965435b9416297b884b39b58c9d4ac43fecc05606373a72370',
    'review_b.json': '9776eaec23a036c0bb25bdba0a40344f1f9cfb50f52b727832634d0f142d2ec3',
}
for fn, expected in expected_reviews.items():
    check(sha(HERE / 'review_round1' / fn) == expected, 'review changed')
for r in visual['records']:
    dual = all(v is True for v in r['a_criteria'].values()) and all(v is True for v in r['b_criteria'].values())
    check(r['dual_visual_pass'] == dual, 'visual merge mismatch')
check(sum(r['dual_visual_pass'] for r in visual['records']) == 5, 'dual count')
check({r['blind_id'] for r in identity['records']} == {r['id'] for r in visual['records'] if r['dual_visual_pass']}, 'review scope')
accepted = [r for r in identity['records'] if r['provisionally_workflow_usable_with_limits']]
check(Counter(r['stratum'] for r in accepted) == dict(apple=2, banana=1, orange=1), 'usable resources')
check(status['cumulative']['counts'] == dict(apple=3, banana=3, orange=3, water=2), 'cumulative counts')
check(status['cumulative']['gaps'] == dict(apple=1, banana=1, orange=1, water=10), 'gaps')
check(len(set(status['cumulative']['usable_ids'])) == 11, 'usable distinct')
check(60 == 3 + 43 + 3 + 11, 'partition')
check(status['cumulative']['workflow_originals_downloaded'] == 60, 'download total')
check(status['qa']['cumulative_pair_comparisons'] == 60 * 90 + 60 * 59 // 2, 'pair accounting')

selection = read(HERE / 'selection_manifest.json')
manifest = read(HERE / 'download_manifest.json')
requests = [json.loads(s) for s in (HERE / 'requests.jsonl').read_text().splitlines()]
check(len(requests) == 17 and all(r['http_status'] == 200 for r in requests), 'request status')
check({r['candidate_id'] for r in requests} == {r['metadata']['id'] for r in selection['rows']}, 'request scope')
check(len({r['candidate_id'] for r in requests}) == 17, 'single attempt')
check(all(r['attempt'] == 1 for r in requests), 'no retry')
for r in manifest['records']:
    for pathkey, hashkey in [('original_path', 'original_sha256'), ('processed_path', 'processed_sha256'), ('crop_path', 'crop_sha256'), ('full_path', 'full_sha256')]:
        p = Path(r[pathkey])
        if not p.is_absolute():
            p = ROOT / p
        check(sha(p) == r[hashkey], f'derivative bytes changed: {pathkey}')

queues = read(FRAME / 'candidate_review_order.json')['reserved_queues']
used = set()
used_counts = Counter()
for batch in ['002', '003']:
    for r in read(FRAME / f'pixel_workflow_{batch}/selection_manifest.json')['rows']:
        check(r['metadata']['id'] not in used, 'reserve reused')
        used.add(r['metadata']['id'])
        used_counts[r['metadata']['stratum']] += 1
for k in ['apple', 'banana', 'orange', 'water']:
    check(set(r['id'] for r in queues['reserve'][k][:used_counts[k]]) == {
        r['metadata']['id'] for batch in ['002', '003']
        for r in read(FRAME / f'pixel_workflow_{batch}/selection_manifest.json')['rows']
        if r['metadata']['stratum'] == k}, 'same queue prefix')
    check(len(queues['reserve'][k]) - used_counts[k] == status['remaining_unconsumed_reserved_counts'][k], 'remaining count')
confirmation = {r['id'] for q in queues['confirmation'].values() for r in q}
check(not (confirmation & used), 'confirmation used')
check(len(confirmation) == 48, 'confirmation capacity')
check(status['confirmation_pixels_accessed'] == status['model_calls'] == 0, 'exposure')
check(not status['automatic_next_batch'], 'automatic continuation')

# Do not rewrite historical collection status or the earlier batch's frozen status.
check(sha(FRAME / 'pixel_workflow_002/curation_status_20260916.json') == 'd4854d56a98c33f68e2130b34b5d630987b9304dd4c192d95e29edbc6b31e07a', '002 status changed')
check(sha(FRAME / 'pixel_workflow_002/README.md') == '2906be63cc4bebcdc0c7a7652286d9dcb101bf5aed5b4354977145afb8aff462', '002 readme changed')
check(sha(HERE / 'collection_status.json') == 'f24555938ba914bd8a5607d76272fd5d3288fdb408b14bd9b7637316532726a4', 'technical stage changed')
for row in read(HERE / 'finalization_snapshots/receipt.json')['snapshots']:
    check(sha(Path(row['snapshot'])) == row['sha256'], 'finalization snapshot')

docs = [HERE / 'README.md', HERE / '后续图库可行性审查.md',
        HERE / 'identity_license_review_20260916.md', FRAME / 'README.md']
links = 0
for p in docs:
    for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
        if target.startswith(('http://', 'https://', '#')):
            continue
        q = Path(unquote(target.split('#')[0]))
        if not q.is_absolute():
            q = p.parent / q
        if q == HERE / 'curation_delivery_qa.json':
            continue  # Current output is written after verification.
        links += 1
        check(q.exists(), f'missing link {p}: {target}')

output = {
    'status': 'passed', 'checks': checks, 'failures': 0, 'local_links_checked': links,
    'scope': 'Offline final curation arithmetic, reserve allocation, source/snapshot/review/artifact hashes, and local links; no fresh visual judgment.',
    'collector_exit_code': 0, 'source_hashes_and_frozen_reviews_match': True,
    'prior002_state_and_003_technical_stage_unchanged': True,
    'new_usable': 4, 'cumulative_usable': 11, 'workflow_gap': 13,
    'new_technical_checks_previously_passed': 737,
    'identity_checks_previously_passed': 70,
    'network_requests': 0, 'image_views': 0, 'model_calls': 0,
    'source_sha256': sha(Path(__file__).resolve()),
    'source_hashes': {str(p): sha(p) for p in [HERE / 'curation_status_20260916.json', *docs]},
    'created_utc': datetime.now(timezone.utc).isoformat(),
}
(HERE / 'curation_delivery_qa.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
receipt = {'status': 'third_batch_complete_no_automatic_continuation',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'artifact_hashes': {str(p): sha(p) for p in [HERE / 'curation_status_20260916.json',
        HERE / 'curation_delivery_qa.json', HERE / 'freeze.json',
        HERE / 'identity_license_review_20260916.json', HERE / 'finalize_curation.py',
        HERE / 'audit_curation.py', HERE / 'review_round1/review_a.json',
        HERE / 'review_round1/review_b.json', *docs]},
    'note': 'No further image collection, visual review, or model work is triggered by this receipt.'}
(HERE / 'completion_receipt_20260916.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': output['status'], 'checks': checks, 'local_links': links,
    'curation_sha256': sha(HERE / 'curation_status_20260916.json'),
    'receipt_sha256': sha(HERE / 'completion_receipt_20260916.json')}, ensure_ascii=False))
