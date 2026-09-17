"""Read-only integrity and exclusion checks for the metadata collection."""
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import collections
import hashlib
import json
import datetime as dt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    freeze, state = read('plan_freeze.json'), read('status.json')
    assert freeze['plan_sha256'] == sha(HERE / 'prospective_sampling_plan.md') == state['plan_sha256']
    assert state['source_code_sha256'] == sha(HERE / 'collect_metadata.py')
    assert state['status'] in ('metadata_collection_complete', 'partial_metadata_rate_limited')
    complete = state['status'] == 'metadata_collection_complete'
    old = read('old_exclusion_registry.json')
    assert old['count'] == len(old['images']) == 101
    for filename, digest in old['source_hashes'].items():
        assert sha(ROOT / 'redesign_v0.4/data' / filename) == digest
    requests = [json.loads(x) for x in (HERE / 'requests.jsonl').read_text().splitlines()]
    assert freeze['fixed_at_utc'] < min(r['requested_at_utc'] for r in requests)
    for request in requests:
        assert sha(HERE / request['response_file']) == request['response_sha256']
        url = urlparse(request['url'])
        assert url.hostname == 'commons.wikimedia.org' and url.path == '/w/api.php'
        query = parse_qs(url.query)
        assert query['action'] == ['query'] and 'iiurlwidth' not in query
        assert query.get('list') == ['categorymembers'] or query.get('prop') == ['imageinfo|info|revisions']
    resume_checks = None
    if state.get('resume_record'):
        resume = read(state['resume_record'])
        assert sha(HERE / state['resume_record']) == state['resume_record_sha256']
        assert sha(HERE / 'sampling_frame.json') == resume['sampling_frame_sha256']
        current = [r for r in requests if r.get('resume_id') == resume['resume_id']]
        assert current and resume['recorded_at_utc'] < current[0]['requested_at_utc']
        gaps = [(dt.datetime.fromisoformat(b['requested_at_utc']) - dt.datetime.fromisoformat(a['requested_at_utc'])).total_seconds()
                for a, b in zip(current, current[1:])]
        assert all(x >= resume['normal_minimum_interval_seconds'] for x in gaps)
        retry_checks = []
        for a, b in zip(current, current[1:]):
            if a.get('status') in (429, 503) and a['url'] == b['url']:
                value = {k.lower(): v for k, v in a.get('headers', {}).items()}.get('retry-after')
                gap = (dt.datetime.fromisoformat(b['requested_at_utc']) - dt.datetime.fromisoformat(a['requested_at_utc'])).total_seconds()
                if value and value.isdigit():
                    assert gap >= int(value)
                    retry_checks.append(dict(response=a['response_file'], required=int(value), request_start_gap=gap))
        resume_checks = dict(resume_id=resume['resume_id'], source_frame_unchanged=True,
                             request_count=len(current), minimum_request_start_gap=min(gaps) if gaps else None,
                             retry_after_checks=retry_checks,
                             historical_timing_defects_disclosed=resume['historical_retry_after_timing_defects'])
    frame = read('sampling_frame.json')
    rows = read('manifest.json')['images']
    assert len(rows) == state['metadata_count']
    if complete:
        assert len(rows) == len(frame['requested_metadata'])
    else:
        index = read('candidate_status_manifest.json')
        assert len(index['candidates']) == index['expected'] == len(frame['requested_metadata']) == 800
        assert index['obtained'] == len(rows) and index['pending'] == 800 - len(rows)
        assert state['assignments_made'] == 0
    old_titles = {r['title_key'] for r in old['images']}
    old_sha1 = {r['original_sha1'] for r in old['images']}
    old_authors = set(old['author_identity_keys'])
    requested = {(r['pageid'], r['stratum'], r['frame_rank_hash']) for r in frame['requested_metadata']}
    groups, assigned, reasons = collections.defaultdict(list), [], collections.Counter()
    for row in rows:
        assert (row['pageid'], row['stratum'], row['frame_rank_hash']) in requested
        assert row['title_key'] not in old_titles
        assert row['pixel_review'] == row['near_duplicate_review'] == row['attribution_review'] == 'not_performed'
        assert row['model_exposure'] == 'none'
        assert sha(HERE / row['metadata_response']) == row['metadata_response_sha256']
        for reason in row['exclusion_reasons']:
            reasons[reason] += 1
        if not row['exclusion_reasons']:
            assert row['original_sha1'] not in old_sha1
            assert not old_authors.intersection(row['author_keys'])
            assert row['author_keys'] and row['license_metadata_allowlist_passed']
            assert row['original_version_timestamp'] <= '2026-09-14T23:59:59Z'
        if row.get('provisional_assignment'):
            assert not row['exclusion_reasons']
            assigned.append(row)
            groups[row['provisional_assignment']].append(row)
    assert len(assigned) == len({r['cluster_id'] for r in assigned})
    for i, a in enumerate(assigned):
        for b in assigned[i + 1:]:
            assert not set(a['author_keys']).intersection(b['author_keys'])
            assert not set(a['original_source_keys']).intersection(b['original_source_keys'])
            assert a['original_sha1'] != b['original_sha1']
    counts = {split: {s: sum(r['stratum'] == s for r in members) for s in ('apple', 'banana', 'orange', 'water')}
              for split, members in groups.items()}
    if complete:
        for split in state['allocation']['counts']:
            assert counts.get(split, {s: 0 for s in ('apple', 'banana', 'orange', 'water')}) == state['allocation']['counts'][split]
    else:
        assert not assigned
    order_checks = snapshot_checks = None
    if complete and state.get('candidate_order'):
        assert sha(HERE / state['candidate_order']) == state['candidate_order_sha256']
        order = read(state['candidate_order'])
        assert order['manifest_sha256'] == sha(HERE / 'manifest.json')
        representatives = order['all_eligible_cluster_representatives']
        assert len(representatives) == order['eligible_cluster_count'] == state['allocation']['available_author_clusters']
        assert [r['cluster_rank'] for r in representatives] == sorted(r['cluster_rank'] for r in representatives)
        assert len({r['cluster_id'] for r in representatives}) == len(representatives)
        reserved = [r for r in representatives if r['provisional_assignment']]
        assert {r['id'] for r in reserved} == {r['id'] for r in assigned}
        for split, strata in order['reserved_queues'].items():
            for stratum, queue in strata.items():
                assert queue == [r for r in representatives if r['provisional_assignment'] == split and r['stratum'] == stratum]
        order_checks = dict(eligible_clusters=len(representatives), reserved=len(reserved),
                            unallocated=len(representatives)-len(reserved), same_stratum_reserve_order_fixed=True,
                            unallocated_is_not_an_expanded_replacement_pool=True)
    if state.get('historical_status_receipt'):
        assert sha(HERE / state['historical_status_receipt']) == state['historical_status_receipt_sha256']
        receipt = read(state['historical_status_receipt'])
        assert sha(ROOT / receipt['previous_completion_manifest']) == receipt['previous_completion_manifest_sha256']
        for snapshot in receipt['snapshots']:
            assert sha(ROOT / snapshot['archive_path']) == snapshot['archived_sha256']
            assert snapshot['matches_v11_completion_manifest']
        snapshot_checks = dict(archived_files=len(receipt['snapshots']), all_match_prior_frozen_hashes=True,
                               current_status_files_have_advanced=True, frozen_v11_report_or_manifest_modified=False)
    result = dict(status='passed_partial_integrity_checks' if not complete else 'passed', collection_complete=complete,
                  plan_before_all_candidate_requests=True, plan_sha256=freeze['plan_sha256'],
                  official_api_only=True, requests=len(requests),
                  continuation_checks=resume_checks,
                  retained_api_errors=[dict(file=r['response_file'], http_status=r.get('status'), error=r.get('api_error', r.get('error')))
                                       for r in requests if r.get('api_error') or r.get('status') != 200],
                  original_old_records=101, normalized_old_title_keys=len(old_titles),
                  frame_unique_normalized_titles=frame['unique_file_count'],
                  metadata_rows=len(rows), metadata_unique_page_ids=len({r['pageid'] for r in rows}),
                  eligible_pending_review=sum(not r['exclusion_reasons'] for r in rows),
                  exclusion_reason_counts=dict(reasons),
                  original_sha1_and_author_group_overlap_with_old_among_eligible=0,
                  provisional_assignment_counts=counts,
                  fixed_review_order_checks=order_checks, historical_status_snapshot_checks=snapshot_checks,
                  author_clusters_across_assignments_disjoint=True,
                  image_downloads=0, image_views=0, model_calls=0,
                  limitations=['Pixel and near-duplicate review not performed.',
                               'Author identities are metadata proxies, not fully disambiguated people.',
                               'Reservations are not approved/ready confirmation images.',
                               'No access-control seal is claimed in this shared workspace.'])
    (HERE / 'metadata_qa.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
