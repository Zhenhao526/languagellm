"""Offline independent byte/pixel/timing audit, without image viewing or models."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import sys
import numpy as np
from PIL import Image, ImageOps

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    count = 0

    def check(test, message):
        nonlocal count
        count += 1
        assert test, message

    frozen = read(OUT/'freeze.json')
    inputs = read(OUT/'comparison_inputs.json')
    selection = read(OUT/'selection_manifest.json')
    manifest = read(OUT/'download_manifest.json')
    packet = read(OUT/'review_round1/packet.json')
    requests = [json.loads(s) for s in (OUT/'requests.jsonl').read_text().splitlines()]
    check(manifest['status'] != 'running', 'Batch must be terminal')
    for path, digest in frozen['source_hashes'].items():
        check(sha(path) == digest, 'Frozen source/input differs')
    for row in frozen['snapshots']:
        check(sha(row['snapshot']) == row['sha256'], 'Input snapshot differs')
    queue = read(BASE/'candidate_review_order.json')['reserved_queues']
    expected = {r['id'] for key, n in selection['quotas'].items() for r in queue['reserve'][key][:n]}
    chosen = {r['metadata']['id']: r for r in selection['rows']}
    check(set(chosen) == expected and len(expected) == 19, 'Selection differs from fixed first19')
    check(len({r['metadata']['cluster_id'] for r in chosen.values()}) == 19, 'Duplicate reserved cluster')
    forbidden = {r['id'] for key in ('workflow', 'confirmation') for q in queue[key].values() for r in q}
    check(not forbidden & expected, 'Forbidden reservation selected')
    check(datetime.fromisoformat(frozen['frozen_utc']) < datetime.fromisoformat(requests[0]['started']['utc']),
          'Freeze must precede first request')
    previous = None
    per_candidate = {}
    minimum_serial_m = minimum_serial_u = None
    retry_checks = []
    for request in requests:
        check(request['candidate_id'] in chosen, 'Out-of-batch request')
        meta = chosen[request['candidate_id']]['metadata']
        check(request['url'] == meta['original_url'], 'Request URL changed')
        check(request['timeout_seconds'] == 60 and request['read_limit_bytes'] == 50*1024*1024, 'Budget changed')
        check(sha(request['raw_path']) == request['raw_sha256'], 'Raw response digest')
        check(Path(request['raw_path']).stat().st_size == request['raw_bytes'] <= 50*1024*1024, 'Raw response length')
        attempts = per_candidate.setdefault(request['candidate_id'], [])
        attempts.append(request['attempt'])
        check(attempts == list(range(1, len(attempts)+1)) and len(attempts) <= 2, 'Retry count changed')
        if previous:
            check(not previous['explicit_refusal'], 'Request after explicit refusal')
            check(not (previous['http_status'] in (429, 503) and previous['attempt'] == 2), 'Request after repeated rate limit')
            gap_m = request['started']['monotonic']-previous['finished']['monotonic']
            gap_u = (datetime.fromisoformat(request['started']['utc'])-datetime.fromisoformat(previous['finished']['utc'])).total_seconds()
            check(gap_m >= 10 and gap_u >= 10, 'Serial interval below10 seconds')
            minimum_serial_m = gap_m if minimum_serial_m is None else min(minimum_serial_m, gap_m)
            minimum_serial_u = gap_u if minimum_serial_u is None else min(minimum_serial_u, gap_u)
            if previous['http_status'] in (429, 503):
                check(request['candidate_id'] == previous['candidate_id'], 'Skipped retry-before-next candidate')
                value = next((v for k, v in previous['header_items'] if k.lower() == 'retry-after'), None)
                if value is None:
                    delay = 30.0
                else:
                    try:
                        delay = float(value)
                    except ValueError:
                        target = parsedate_to_datetime(value)
                        if target.tzinfo is None:
                            target = target.replace(tzinfo=timezone.utc)
                        delay = max(0.0, (target-datetime.fromisoformat(previous['finished']['utc'])).total_seconds())
                check(gap_m >= delay+1 and gap_u >= delay+1, 'Retry-After plus1 not met')
                retry_checks.append({'required_with_margin': delay+1, 'monotonic': gap_m, 'utc': gap_u})
        previous = request
    request_ids = list(per_candidate)
    expected_order = [r['metadata']['id'] for r in selection['rows']]
    check(request_ids == expected_order[:len(request_ids)], 'Skipped/reordered fixed candidate queue')
    sys.path.insert(0, str(BASE))
    import collect_workflow_pixels as legacy
    sys.path.insert(0, str(ROOT/'redesign_v0.4'))
    from encode_images import preprocess
    successes = []
    processing_errors = []
    for record in manifest['records']:
        if record['status'] not in ('downloaded', 'quarantined_format'):
            processing_errors.append({'status': record['status'], 'error': record.get('error')})
            continue
        successes.append(record)
        meta = chosen[record['candidate_id']]['metadata']
        raw = Path(record['original_path']).read_bytes()
        check(hashlib.sha1(raw).hexdigest() == meta['original_sha1'], 'Commons original SHA1')
        check(sha(record['original_path']) == record['original_sha256'], 'Original SHA256')
        decoded = Image.open(io.BytesIO(raw))
        full = ImageOps.exif_transpose(decoded).convert('RGB')
        check((meta['width'], meta['height']) in (decoded.size, full.size), 'Source dimensions')
        ordinary = decoded.format == {'image/jpeg': 'JPEG', 'image/png': 'PNG'}[meta['mime']] and getattr(decoded, 'n_frames', 1) == 1
        check(ordinary == record['ordinary_single_image'], 'Actual format gate')
        check(np.array_equal(np.asarray(Image.open(record['full_path'])), np.asarray(full)), 'EXIF full image')
        im = full.copy()
        im.thumbnail((640, 640), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        im.save(buffer, format='JPEG', quality=95)
        check(buffer.getvalue() == Path(record['processed_path']).read_bytes(), '640 JPEG bytes independently rebuilt')
        crop = legacy.crop_image(Image.open(io.BytesIO(buffer.getvalue())))
        check(np.array_equal(np.asarray(crop), np.asarray(Image.open(record['crop_path']))), '224 crop independently rebuilt')
        arr = np.asarray(crop).astype(np.float32)/255
        arr = (arr-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
        check(np.array_equal(arr.transpose(2, 0, 1), preprocess(record['processed_path']).numpy()), 'Legacy tensor exact')
        check(legacy.hash_image(Image.open(record['processed_path'])) == record['hashes'], 'Legacy pixel hashes')
    expected_packet = sorted(r['blind_id'] for r in successes if r['ordinary_single_image'])
    check(packet['ids'] == expected_packet, 'Anonymous packet membership/order')
    for key in ('image_sha256', 'individual_image_sha256'):
        for path, digest in packet[key].items():
            check(sha(path) == digest, 'Packet image digest')
    text = json.dumps(packet, ensure_ascii=False)
    for row in chosen.values():
        meta = row['metadata']
        for key in ('id', 'title', 'source_page', 'original_url', 'artist_html', 'license_url', 'cluster_id'):
            value = meta.get(key)
            if value:
                check(value not in text, 'Identity/source leaked into anonymous packet')
    check(not any(k in packet for k in ('stratum', 'license', 'artist', 'model_scores', 'quarantined')), 'Packet reveals source/format data')
    dup = read(OUT/'near_duplicate_flags.json')
    check(dup['old_available_pixels'] == 90 and len(dup['old_unavailable']) == 11 and dup['previous_batch_pixels'] == 24, 'Historical pixel coverage')
    n = len(successes)
    check(dup['comparisons'] == 114*n+n*(n-1)//2, 'Pair coverage')
    hashes = dup['pixel_hash_rows']
    reproduced = []
    for i, item in enumerate(hashes[114:], 114):
        for prior in hashes[:i]:
            pd = (int(item['phash'], 16)^int(prior['phash'], 16)).bit_count()
            dd = (int(item['dhash'], 16)^int(prior['dhash'], 16)).bit_count()
            exact = bool(prior.get('original_sha1')) and item['original_sha1'] == prior['original_sha1']
            if pd <= 8 or dd <= 6 or exact or item['processed_sha256'] == prior['processed_sha256']:
                reproduced.append((item['id'], prior['id']))
    check(reproduced == [(f['a'], f['b']) for f in dup['flags']], 'Duplicate recall independently enumerated')
    qa = {'status': 'passed', 'checks': count, 'failures': 0, 'scope': 'Offline technical audit, not visual or license acceptance',
        'planned': 19, 'processed': len(manifest['records']), 'downloaded': n, 'ordinary': len(expected_packet),
        'processing_or_download_errors': processing_errors, 'request_count': len(requests),
        'minimum_serial_monotonic_seconds': minimum_serial_m, 'minimum_serial_utc_seconds': minimum_serial_u,
        'retry_after_checks': retry_checks, 'old_pixels': 90, 'old_pixel_gaps': 11, 'previous_batch_pixels': 24,
        'comparison_count': dup['comparisons'], 'duplicate_flags': len(dup['flags']),
        'new_network_requests': 0, 'image_views': 0, 'model_calls': 0,
        'source_sha256': sha(__file__), 'manifest_sha256': sha(OUT/'download_manifest.json'),
        'freeze_sha256': sha(OUT/'freeze.json'), 'packet_sha256': sha(OUT/'review_round1/packet.json'),
        'finished_utc': datetime.now(timezone.utc).isoformat()}
    (OUT/'independent_collection_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: qa[k] for k in ('status', 'checks', 'downloaded', 'ordinary', 'request_count', 'duplicate_flags')}))


if __name__ == '__main__':
    main()
