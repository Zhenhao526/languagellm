"""Read-only checks of saved metadata; writes new receipts, no network/image APIs."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CURATION = HERE.parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open('x') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write('\n')


def main():
    prepared = HERE / 'offline_001'
    manifest = read(prepared / 'manifest.json')
    sources = {}
    for path, expected in manifest['source_sha256'].items():
        actual = digest(Path(path))
        assert actual == expected, path
        sources[path] = actual
    outputs = {}
    for name, expected in manifest['output_sha256'].items():
        actual = digest(prepared / name)
        assert actual == expected, name
        outputs[name] = actual
    originals = read(CURATION / 'ranking_001/candidate_order.json')
    ordered = [r for label in ('apple', 'banana', 'orange', 'water')
               for r in originals[label]]
    candidates = read(prepared / 'candidates.json')
    assert [r['legacy_nomination'] for r in candidates] == ordered
    assert len(candidates) == len({r['pageid'] for r in ordered}) == 232
    assert len({r['component_id'] for r in ordered}) == 232
    counts = Counter()
    widths = Counter()
    reasons = Counter()
    eligible_ids = set()
    for row in candidates:
        old = row['legacy_nomination']
        w, h = old['width'], old['height']
        possible = [d for d in (960, 1280, 1920, 3840)
                    if d < w and min(d, h * d // w) >= 512]
        selected = min(possible) if possible else None
        assert row['recommended_width'] == selected
        assert row['technical_metadata_eligible'] == bool(possible)
        assert row['actual_thumbnail_received'] is False
        assert row['source_orientation_verified'] is False
        assert row['visual_acceptance'] is row['source_acceptance'] is row['material_split'] is None
        if possible:
            assert row['predicted_thumbnail_width'] == selected
            assert row['predicted_thumbnail_height_floor'] == h * selected // w
            assert row['predicted_thumbnail_height_ceil'] == (h * selected + w - 1) // w
            counts[old['label']] += 1
            widths[selected] += 1
            eligible_ids.add(old['pageid'])
        else:
            reasons[row['technical_metadata_exclusion']] += 1
    assert counts == {'apple': 37, 'banana': 36, 'orange': 40, 'water': 96}
    assert widths == {960: 207, 1280: 2}
    assert sum(reasons.values()) == 23
    batches = read(prepared / 'metadata_api_batches.json')
    batch_ids = [pid for batch in batches for pid in batch['pageids']]
    assert len(batch_ids) == len(set(batch_ids)) == 209
    assert set(batch_ids) == eligible_ids
    selected_by_id = {r['legacy_nomination']['pageid']: r['recommended_width'] for r in candidates}
    for i, batch in enumerate(batches):
        assert batch['batch_index'] == i
        assert 1 <= len(batch['pageids']) <= 25
        assert batch['network_requested'] is False
        assert all(selected_by_id[pid] == batch['requested_width'] for pid in batch['pageids'])
        assert batch['parameters']['pageids'] == '|'.join(map(str, batch['pageids']))
        assert batch['parameters']['iiurlwidth'] == batch['requested_width']
        pool = [r['pageid'] for r in ordered
                if ('water' if r['label'] == 'water' else 'food') == batch['stage']
                and selected_by_id[r['pageid']] == batch['requested_width']]
        positions = [pool.index(pid) for pid in batch['pageids']]
        assert positions == sorted(positions)
    receipt = {
        'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'audit_source_sha256': digest(Path(__file__)),
        'source_sha256_verified': sources, 'output_sha256_verified': outputs,
        'unchanged_nominations': 232, 'unchanged_source_components': 232,
        'strict_metadata_eligible': dict(counts), 'selected_widths': dict(widths),
        'technical_exclusions': dict(reasons), 'planned_api_batches': len(batches),
        'all_eligible_ids_in_exactly_one_batch': True,
        'scope': 'Pure saved-metadata self-check, not independent implementation/visual acceptance.',
        'new_network_requests': 0, 'image_bytes_read': 0, 'pixel_views': 0, 'model_calls': 0,
    }
    save_new(HERE / 'offline_selfcheck.json', receipt)

    cross = CURATION / 'cross_frame_overlap_001'
    expected = read(cross / 'overlap.json')
    checked_sources = {}
    loaded = {}
    for key, item in expected['sources'].items():
        actual = digest(Path(item['path']))
        assert actual == item['sha256'], key
        checked_sources[key] = dict(item)
        loaded[key] = read(Path(item['path']))
    other = loaded['other_manifest']['images']
    requested = loaded['other_frame']['requested_metadata']
    assert Counter((r['pageid'], r['stratum']) for r in requested) == Counter((r['pageid'], r['stratum']) for r in other)
    assert len(other) == 800 and len({r['pageid'] for r in other}) == 799
    rows = []
    for a in ordered:
        for b in other:
            same_id = a['pageid'] == b['pageid']
            same_sha = bool(a['original_file_sha1']) and a['original_file_sha1'] == b['original_sha1']
            if same_id or same_sha:
                rows.append({
                    'own_pageid': a['pageid'], 'other_pageid': b['pageid'],
                    'own_review_id': a['review_id'], 'own_label': a['label'],
                    'own_original_role': a['role'], 'other_stratum': b['stratum'],
                    'other_provisional_assignment': b.get('provisional_assignment'),
                    'other_exclusion_reasons': b['exclusion_reasons'],
                    'same_pageid': same_id, 'same_original_sha1': same_sha,
                })
    assert rows == expected['overlaps']
    label_counts = dict(Counter(r['own_label'] for r in rows))
    assignment_counts = dict(Counter(str(r['other_provisional_assignment']) for r in rows))
    assert label_counts == expected['by_own_label'] == {'apple': 16, 'banana': 20, 'orange': 13, 'water': 59}
    assert assignment_counts == expected['by_other_provisional_assignment'] == {'None': 83, 'confirmation': 9, 'reserve': 10, 'workflow': 6}
    assert len(rows) == len({r['own_pageid'] for r in rows}) == 108
    assert all(r['same_pageid'] and r['same_original_sha1'] for r in rows)
    duplicate = [r for r in other if r['pageid'] == 40608211]
    assert len(duplicate) == 2
    assert {r['stratum'] for r in duplicate} == {'orange', 'water'}
    assert all(r.get('provisional_assignment') is None for r in duplicate)
    assert 40608211 not in {r['pageid'] for r in ordered}
    status = loaded['other_status']
    assert status['status'] == loaded['other_manifest']['status'] == 'metadata_collection_complete'
    assert status['manifest_sha256'] == checked_sources['other_manifest']['sha256']
    linked_hashes = {}
    other_dir = Path(checked_sources['other_status']['path']).parent
    for key, filename in [('plan_sha256', 'prospective_sampling_plan.md'), ('source_code_sha256', 'collect_metadata.py'),
                          ('resume_record_sha256', status['resume_record']),
                          ('candidate_status_sha256', 'candidate_status_manifest.json'),
                          ('candidate_order_sha256', status['candidate_order']),
                          ('historical_status_receipt_sha256', status['historical_status_receipt'])]:
        p = other_dir / filename
        actual = digest(p)
        assert actual == status[key], filename
        linked_hashes[str(p)] = actual
    result = {
        'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'audit_source_sha256': digest(Path(__file__)),
        'audited_overlap_sha256': digest(cross / 'overlap.json'),
        'source_hashes_verified': checked_sources, 'status_linked_hashes_verified': linked_hashes,
        'other_rows': 800, 'other_unique_pageids': 799, 'overlapping_candidates_and_pairs': 108,
        'by_own_label': label_counts, 'by_other_provisional_assignment': assignment_counts,
        'confirmation_pageids': [r['own_pageid'] for r in rows if r['other_provisional_assignment'] == 'confirmation'],
        'overlap_rows_exact_match': True,
        'same_page_different_sha_pairs': 0, 'different_page_same_sha_pairs': 0,
        'other_duplicate': {'pageid': 40608211, 'strata': ['orange', 'water'],
                            'assignments': [None, None], 'in_own_fixed_candidates': False},
        'other_status': status['status'], 'other_finished_at_utc': status['finished_at_utc'],
        'scope': 'Exact pageid/original-SHA1 overlap and saved source hashes only; not pixel, author/series, provenance or full other-task execution audit. No process observation by this script.',
        'new_network_requests': 0, 'image_bytes_read': 0, 'pixel_views': 0, 'model_calls': 0,
    }
    save_new(cross / 'independent_audit.json', result)
    print(json.dumps({'thumbnail_check': receipt['status'], 'cross_frame_check': result['status'],
                      'eligible': dict(counts), 'overlap': len(rows), 'assignments': assignment_counts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
