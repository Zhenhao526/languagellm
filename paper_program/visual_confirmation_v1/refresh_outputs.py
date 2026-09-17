"""Refresh local metadata indexes after a completed or interrupted collector run.

Reads this directory only. No network, image reads, or model calls.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    state, frame, manifest = read('status.json'), read('sampling_frame.json'), read('manifest.json')
    rows = manifest['images']
    lookup = {(row['pageid'], row['stratum']): i for i, row in enumerate(rows)}
    requests = [json.loads(x) for x in (HERE / 'requests.jsonl').read_text().splitlines()]
    failed_batch = None
    if state['status'] != 'metadata_collection_complete':
        recent = requests[-1]
        if recent.get('status') == 429 and recent.get('label', '').startswith('metadata_'):
            failed_batch = int(recent['label'].split('_')[1])
    index = []
    for rank, candidate in enumerate(frame['requested_metadata']):
        row_index = lookup.get((candidate['pageid'], candidate['stratum']))
        if row_index is not None:
            status = 'metadata_obtained'
        elif failed_batch is not None and failed_batch <= rank < failed_batch + 10:
            status = 'rate_limited_after_bounded_retry'
        else:
            status = 'not_requested_after_collection_stop'
        index.append(dict(metadata_request_rank=rank + 1, pageid=candidate['pageid'], title=candidate['title'],
                          stratum=candidate['stratum'], frame_rank_hash=candidate['frame_rank_hash'],
                          status=status, manifest_row=row_index))
    write('candidate_status_manifest.json', dict(status=state['status'], expected=len(index),
                                               obtained=len(rows), pending=len(index) - len(rows), candidates=index))
    with (HERE / 'license_chain.tsv').open('w') as file:
        fields = ['pageid', 'stratum', 'title', 'source_page', 'original_sha1', 'original_version_timestamp',
                  'artist_text', 'license', 'license_url', 'attribution_required',
                  'metadata_response', 'metadata_response_sha256']
        writer = csv.DictWriter(file, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    if state['status'] == 'metadata_collection_complete':
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row['cluster_id']].append(row)
        write('author_groups.json', dict(status='metadata_graph_complete_pixel_identity_review_pending',
            groups=[dict(cluster_id=key, cluster_rank=members[0]['cluster_rank'],
                         identity_keys=sorted({k for row in members for k in row['author_keys'] + row['original_source_keys']}),
                         members=[dict(pageid=row['pageid'], stratum=row['stratum'],
                                       metadata_status=row['metadata_status'],
                                       provisional_assignment=row['provisional_assignment']) for row in members])
                    for key, members in sorted(groups.items())]))
        write('provisional_allocations.json', dict(status='reservations_only_not_approved_images',
            allocation=state['allocation'],
            records=[{key: row[key] for key in ('pageid', 'title', 'stratum', 'cluster_id', 'cluster_rank',
                                                'provisional_assignment', 'pixel_review', 'near_duplicate_review',
                                                'attribution_review', 'model_exposure')}
                     for row in rows if row.get('provisional_assignment')]))
        state['metadata_author_groups'] = len(groups)
    elif failed_batch is not None:
        state['status'] = 'partial_metadata_rate_limited'
    state.update(sampling_frame_complete_within_fixed_scope=True,
                 planned_metadata_rows=len(index), pending_metadata_rows=len(index) - len(rows),
                 eligible_pending_review=sum(not row['exclusion_reasons'] for row in rows),
                 assignments_made=sum(bool(row.get('provisional_assignment')) for row in rows),
                 metadata_count_by_stratum=dict(collections.Counter(row['stratum'] for row in rows)),
                 no_alternate_origin_used=True,
                 manifest_sha256=sha(HERE / 'manifest.json'),
                 candidate_status_sha256=sha(HERE / 'candidate_status_manifest.json'))
    if state.get('resume_record'):
        resume = read(state['resume_record'])
        assert sha(HERE / 'sampling_frame.json') == resume['sampling_frame_sha256']
        state['historical_retry_after_timing_defects'] = resume['historical_retry_after_timing_defects']
        state['resume_http_status_counts'] = dict(collections.Counter(str(r.get('status')) for r in requests
                                                                     if r.get('resume_id') == resume['resume_id']))
    write('status.json', state)
    print(json.dumps({key: value for key, value in state.items() if key not in ('allocation',)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
