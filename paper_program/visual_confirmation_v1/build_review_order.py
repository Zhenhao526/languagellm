"""Expose the already fixed metadata candidate order; no network or image reads.

This does not allocate more images or change existing reservations. Unallocated
clusters are listed for provenance only, not as an expanded replacement pool.
"""
from pathlib import Path
import collections
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SALT = 'visual_confirmation_v1|2026-09-15|prospective|no_model_scores'
STRATA = ('apple', 'banana', 'orange', 'water')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    state = read('status.json')
    assert state['status'] == 'metadata_collection_complete'
    groups = collections.defaultdict(list)
    for row in read('manifest.json')['images']:
        groups[row['cluster_id']].append(row)
    representatives = []
    for cluster_id, members in groups.items():
        usable = [row for row in members if not row['exclusion_reasons']]
        if usable:
            row = min(usable, key=lambda x: (x['frame_rank_hash'], x['id']))
            assert row['cluster_rank'] == hashlib.sha256((SALT + '|cluster|' + cluster_id).encode()).hexdigest()
            representatives.append(row)
    representatives.sort(key=lambda x: x['cluster_rank'])
    expected, used = {}, set()
    for split, targets in state['allocation']['targets'].items():
        counts = collections.Counter()
        for row in representatives:
            if row['cluster_id'] not in used and counts[row['stratum']] < targets[row['stratum']]:
                counts[row['stratum']] += 1
                used.add(row['cluster_id'])
                expected[row['id']] = split
    for members in groups.values():
        for row in members:
            assert row['provisional_assignment'] == expected.get(row['id'])
    keys = ('id', 'pageid', 'title', 'stratum', 'cluster_id', 'cluster_rank', 'frame_rank_hash',
            'source_page', 'provisional_assignment')
    entries = [dict(global_cluster_rank=i, **{k: row[k] for k in keys})
               for i, row in enumerate(representatives, 1)]
    queues = {split: {stratum: [row for row in entries
                               if row['stratum'] == stratum and row['provisional_assignment'] == split]
                      for stratum in STRATA}
              for split in state['allocation']['targets']}
    write('candidate_review_order.json', dict(
        status='fixed_metadata_reservations_only_pixel_identity_license_review_pending',
        plan_sha256=state['plan_sha256'], manifest_sha256=sha(HERE / 'manifest.json'), salt=SALT,
        order_rule='Eligible representative per cluster: minimum (frame_rank_hash, id); globally sort cluster_rank.',
        eligible_cluster_count=len(entries), reserved_cluster_count=len(used),
        unallocated_cluster_count=len(entries) - len(used),
        reserved_queues=queues, all_eligible_cluster_representatives=entries,
        replacement_rule='Only the already reserved same-stratum reserve queue is a replacement pool. '
                         'Unallocated entries are provenance, not permission to expand candidate quotas. '
                         'Pixel/identity review may exclude or merge clusters; record every decision without reseeding.',
        image_downloads=0, image_views=0, model_calls=0))
    previous_manifest = ROOT / 'redesign_v0.11/results/anchoring_001/completion_manifest.json'
    frozen = json.loads(previous_manifest.read_text())
    snapshots = []
    for name in ('README.md', 'status.json', 'metadata_qa.json'):
        previous_key = str((HERE / name).relative_to(ROOT))
        snapshot = HERE / 'resume_001' / name
        digest = sha(snapshot)
        assert digest == frozen['artifacts'][previous_key]
        snapshots.append(dict(dynamic_path=previous_key, archive_path=str(snapshot.relative_to(ROOT)),
                              archived_sha256=digest, matches_v11_completion_manifest=True))
    write('historical_status_receipt.json', dict(
        status='all_three_prior_state_files_preserved_byte_for_byte',
        previous_completion_manifest=str(previous_manifest.relative_to(ROOT)),
        previous_completion_manifest_sha256=sha(previous_manifest), snapshots=snapshots,
        note='Current README/status/metadata_qa are mutable stage status files and have advanced to 800 metadata rows. '
             'The v11 report and completion manifest were not edited; their recorded hashes refer to these archived bytes.'))
    state.update(candidate_order='candidate_review_order.json', candidate_order_sha256=sha(HERE / 'candidate_review_order.json'),
                 historical_status_receipt='historical_status_receipt.json',
                 historical_status_receipt_sha256=sha(HERE / 'historical_status_receipt.json'))
    write('status.json', state)
    print(json.dumps(dict(eligible_clusters=len(entries), reserved=len(used), unallocated=len(entries)-len(used),
                          historical_snapshots='3/3 exact'), ensure_ascii=False))


if __name__ == '__main__':
    main()
