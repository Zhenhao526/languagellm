"""Read existing metadata, preserve exclusion scope, and freeze before network."""
from pathlib import Path
from datetime import datetime, timezone
import json
from metadata_helpers import HERE, sha, all_keys, components, norm, title_key

V1 = HERE.parent / 'visual_confirmation_v1'
def read(p): return json.loads(p.read_text())
def write(p, d): p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')

def main():
    assert not (HERE / 'requests.jsonl').exists(), 'No refreeze after requests.'
    assert not (HERE / 'freeze.json').exists(), 'Already frozen.'
    files = [V1 / p for p in ['old_exclusion_registry.json', 'manifest.json', 'author_groups.json',
        'provisional_allocations.json', 'candidate_review_order.json', 'README.md',
        'pixel_workflow_001/download_manifest.json', 'pixel_workflow_002/download_manifest.json',
        'pixel_workflow_003/download_manifest.json', 'pixel_workflow_003/curation_status_20260916.json',
        'pixel_workflow_003/author_resolution_001/resolution_supplement.json',
        'pixel_workflow_003/后续图库可行性审查.md']]
    old = read(files[0]); images = read(files[1])['images']; groups = read(files[2])['groups']
    reserved = read(files[3])['records']; reserved_clusters = {r['cluster_id'] for r in reserved}
    reserved_pageids = {m['pageid'] for g in groups if g['cluster_id'] in reserved_clusters for m in g['members']}
    downloaded = [r for n in ['001', '002', '003'] for r in read(V1 / f'pixel_workflow_{n}/download_manifest.json')['records']]
    downloaded_ids = {int(r['candidate_id'].split(':')[-1]) for r in downloaded}
    assert len(downloaded_ids) == 60 and downloaded_ids <= reserved_pageids
    blocked = set(old['author_identity_keys']) | set(old['original_source_keys'])
    for row in old['images']: blocked.update(all_keys(row))
    # An unresolved alias is conservatively blocked; this does not assert that the accounts are identical.
    alias = read(V1 / 'pixel_workflow_003/author_resolution_001/resolution_supplement.json')
    for name in alias['official_redirect'].values():
        name = name.split(':', 1)[-1]
        blocked.update(['commons-user:' + norm(name), 'text:' + norm(name)])
    blocked.update(all_keys(alias['old101_potential_alias_record']))
    for row in images:
        if row['pageid'] in reserved_pageids: blocked.update(all_keys(row))
    grouped = components(images)
    changed = True
    while changed:
        before = len(blocked)
        for g in grouped:
            keys = set().union(*(all_keys(r) for r in g))
            if keys & blocked: blocked.update(keys)
        changed = len(blocked) != before
    blocked_rows = [r for r in images if all_keys(r) & blocked]
    exclusion = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        old_record_count=101, downloaded_count=60, reserved_cluster_count=144,
        reserved_member_page_count=len(reserved_pageids),
        blocked_keys=sorted(blocked),
        blocked_pageids=sorted({r['pageid'] for r in blocked_rows} | reserved_pageids),
        blocked_titles=sorted({title_key(r['title']) for r in old['images'] + blocked_rows}),
        unresolved_old_author_count=sum(not r['author_resolved'] for r in old['images']),
        old_unavailable_pixels=11, pending_alias_preserved=True,
        note='Conservative metadata-key/connected-component exclusion, not proof of complete author independence; no new pixels read.',
        source_hashes={str(p): sha(p.read_bytes()) for p in files})
    write(HERE / 'exclusion_registry.json', exclusion)
    inputs = files + [HERE / 'exclusion_registry.json', HERE / 'config.json', HERE / '固定元数据可行性计划.md',
        HERE / 'metadata_helpers.py', HERE / 'prepare_freeze.py', HERE / 'collect_metadata.py']
    snapshots = HERE / 'input_snapshots'; snapshots.mkdir(exist_ok=True)
    entries = []
    for p in inputs:
        data = p.read_bytes(); h = sha(data); dest = snapshots / (h[:12] + '_' + p.name)
        dest.write_bytes(data)
        entries.append(dict(original=str(p), snapshot=str(dest), sha256=h))
    write(HERE / 'freeze.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        before_first_request=True, source_hashes={str(p): sha(p.read_bytes()) for p in inputs},
        snapshots=entries, original_v1_inputs_unchanged=True, pixel_requests=0, model_calls=0))
    print(json.dumps({'blocked_keys': len(blocked), 'blocked_pageids': len(exclusion['blocked_pageids']),
        'freeze_sha256': sha((HERE / 'freeze.json').read_bytes())}))

if __name__ == '__main__': main()
