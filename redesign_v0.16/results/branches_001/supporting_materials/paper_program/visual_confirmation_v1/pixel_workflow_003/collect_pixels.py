"""Fixed reserve-only workflow acquisition. No inference or visual adjudication."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import argparse
import collections
import hashlib
import io
import json
import math
import os
import platform
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[1]
SALT = 'visual_confirmation_v1|2026-09-15|prospective|no_model_scores'
QUOTAS = {'apple': 3, 'banana': 2, 'orange': 2, 'water': 10}
PLANNED = sum(QUOTAS.values())
REFERENCE_COUNT = 133
LIMIT = 50 * 1024 * 1024
USER_AGENT = 'LanguageEmergenceResearch/0.16 (local academic image curation; serialized requests)'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc)


def stamp():
    return {'utc': now().isoformat(), 'monotonic': time.monotonic()}


def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def hash_image(im):
    gray = ImageOps.exif_transpose(im).convert('RGB').convert('L')
    a = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    x = np.arange(32)
    k = np.arange(32)
    c = np.cos(np.pi * (2*x[None, :] + 1) * k[:, None] / 64) * np.sqrt(2 / 32)
    c[0] /= np.sqrt(2)
    low = (c @ a @ c.T)[:8, :8]
    pb = (low > np.median(low)).reshape(-1)
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    db = (small[:, 1:] > small[:, :-1]).reshape(-1)
    return {'phash': np.packbits(pb).tobytes().hex(), 'dhash': np.packbits(db).tobytes().hex()}


def crop_image(im):
    im = ImageOps.exif_transpose(im).convert('RGB')
    w, h = im.size
    size = (round(w * 256 / min(w, h)), round(h * 256 / min(w, h)))
    im = im.resize(size, Image.Resampling.BICUBIC)
    x, y = (size[0] - 224) // 2, (size[1] - 224) // 2
    return im.crop((x, y, x+224, y+224))


def retry_delay(headers, finished):
    headers = {k.lower(): v for k, v in headers.items()}
    value = headers.get('retry-after')
    if value is None:
        return 30.0, 'missing_default_30'
    try:
        delay = float(value)
        if not math.isfinite(delay) or delay < 0:
            raise ValueError('Retry-After is not finite nonnegative')
        return delay, 'seconds'
    except ValueError:
        target = parsedate_to_datetime(value)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - finished).total_seconds()), 'http_date'


def verify_inputs(frozen):
    for path, expected in frozen['source_hashes'].items():
        assert sha(path) == expected, 'Frozen input/source changed: ' + path


def prepare():
    assert not (OUT/'freeze.json').exists(), 'Do not overwrite a frozen batch'
    assert (platform.python_version(), PIL.__version__, np.__version__) == ('3.12.14', '12.3.0', '1.26.4')
    queues = read(BASE/'candidate_review_order.json')
    assert sha(BASE/'manifest.json') == queues['manifest_sha256']
    assert sha(BASE/'prospective_sampling_plan.md') == queues['plan_sha256']
    assert queues['reserved_cluster_count'] == 144
    metadata = {r['id']: r for r in read(BASE/'manifest.json')['images']}
    status = read(OUT/'prerequisite_status.json')
    assert status['counts'] == {'apple': 1, 'banana': 2, 'orange': 2, 'water': 2}
    assert status['gaps'] == QUOTAS and status['planned'] == PLANNED and status['usable_total'] == 7
    consumed = {r['metadata']['id'] for r in read(BASE/'pixel_workflow_002/selection_manifest.json')['rows']}
    assert len(consumed) == 19
    chosen = []
    for stratum, count in QUOTAS.items():
        q = queues['reserved_queues']['reserve'][stratum]
        assert q == sorted(q, key=lambda r: r['cluster_rank'])
        unused = [(i, r) for i, r in enumerate(q, 1) if r['id'] not in consumed]
        assert len(unused) >= count
        for ordinal, r in unused[:count]:
            m = metadata[r['id']]
            assert m['provisional_assignment'] == 'reserve'
            assert m['stratum'] == stratum and not m['exclusion_reasons']
            assert m['cluster_id'] == r['cluster_id'] and m['license_metadata_allowlist_passed']
            assert m['original_url'].startswith('https://upload.wikimedia.org/')
            ident = hashlib.sha256((SALT+'|pixel-blind|'+m['id']).encode()).hexdigest()[:12]
            chosen.append({'blind_id': ident, 'same_stratum_reserve_ordinal': ordinal,
                           'batch_purpose': 'workflow_replacement', 'metadata': m})
    chosen.sort(key=lambda r: r['metadata']['cluster_rank'])
    assert len(chosen) == len({r['metadata']['cluster_id'] for r in chosen}) == PLANNED
    assert len({r['blind_id'] for r in chosen}) == PLANNED
    excluded = {r['cluster_id'] for kind in ['workflow', 'confirmation']
                for q in queues['reserved_queues'][kind].values() for r in q}
    assert not excluded & {r['metadata']['cluster_id'] for r in chosen}
    previous = []
    for batch, expected in [('pixel_workflow_001', 24), ('pixel_workflow_002', 19)]:
        prior = read(BASE/batch/'download_manifest.json')['records']
        assert len(prior) == expected and all(r['status'] in ('downloaded', 'quarantined_format') for r in prior)
        previous += [dict(r, prior_batch=batch) for r in prior]
    assert len(previous) == 43
    assert not {r['blind_id'] for r in previous} & {r['blind_id'] for r in chosen}
    # Pixel availability is an engineering fact, not a new content exclusion.
    old = read(ROOT/'redesign_v0.4/data/candidates_downloaded.json')['images']
    available, unavailable = [], []
    for r in old:
        p = ROOT/'redesign_v0.4'/r['path']
        if r.get('download_status') != 'ok' or not p.is_file():
            unavailable.append({'id': r['id'], 'download_status': r.get('download_status'), 'path': str(p)})
        else:
            assert sha(p) == r['sha256']
            available.append({'id': r['id'], 'path': str(p), 'sha256': r['sha256'],
                              'original_sha1': r.get('original_sha1')})
    assert len(old) == 101 and len(available) == 90 and len(unavailable) == 11
    write(OUT/'selection_manifest.json', {'created_utc': now().isoformat(), 'quotas': QUOTAS,
        'planned': PLANNED, 'selection_rule': 'First unused reserve entries per frozen same-stratum queue; global cluster_rank request order',
        'original_reservation_unchanged': True, 'rows': chosen,
        'confirmation_pixel_access': 0, 'unallocated_candidate_access': 0})
    write(OUT/'comparison_inputs.json', {'old_available': available, 'old_unavailable': unavailable,
        'previous_batch': previous, 'confirmation_pixels_compared': False})
    sources = [Path(__file__), OUT/'audit_collection.py', OUT/'make_review_batch.py', OUT/'merge_visual_reviews.py', OUT/'冻结执行方案.md', OUT/'selection_manifest.json', OUT/'comparison_inputs.json',
        BASE/'pixel_workflow_plan.md', BASE/'prospective_sampling_plan.md', BASE/'manifest.json',
        BASE/'candidate_review_order.json', BASE/'author_groups.json',
        OUT/'prerequisite_status.json', OUT/'author_resolution_001/resolution_supplement.json',
        OUT/'author_resolution_001/requests.json', OUT/'author_resolution_001/user_page.response',
        OUT/'author_resolution_001/redirects_api.response', OUT/'author_resolution_001/resolve_author.py',
        BASE/'pixel_workflow_002/curation_status_20260916.json', BASE/'pixel_workflow_002/selection_manifest.json',
        BASE/'pixel_workflow_002/download_manifest.json', BASE/'pixel_workflow_002/independent_collection_qa.json',
        BASE/'pixel_workflow_002/collect_pixels.py',
        BASE/'pixel_workflow_001/download_manifest.json', BASE/'pixel_workflow_001/offline_finish_qa.json',
        BASE/'collect_workflow_pixels.py', ROOT/'redesign_v0.4/encode_images.py',
        ROOT/'redesign_v0.4/data/candidates_downloaded.json']
    sources += [BASE/r['metadata']['metadata_response'] for r in chosen]
    sources += [Path(r['path']) for r in available]
    sources += [Path(r[k]) for r in previous for k in ('original_path', 'processed_path')]
    sources = sorted(set(p.resolve() for p in sources))
    hashes = {str(p): sha(p) for p in sources}
    snapshots = OUT/'input_snapshots'
    snapshots.mkdir(exist_ok=False)
    snapshot_rows = []
    for p in sources:
        if p.suffix in ('.json', '.md', '.py', '.response'):
            dest = snapshots/(hashes[str(p)][:12]+'_'+p.name)
            shutil.copy2(p, dest)
            snapshot_rows.append({'original': str(p), 'snapshot': str(dest), 'sha256': sha(dest)})
    for child in ('responses', 'originals', 'processed', 'blind'):
        (OUT/child).mkdir(exist_ok=False)
    write(OUT/'freeze.json', {'frozen_utc': now().isoformat(), 'before_first_request': True,
        'source_hashes': hashes, 'snapshots': snapshot_rows, 'planned': PLANNED, 'quotas': QUOTAS,
        'runtime': {'python': platform.python_version(), 'pillow': PIL.__version__, 'numpy': np.__version__,
                    'numpy_path': np.__file__, 'executable': sys.executable},
        'model_calls': 0, 'confirmation_requests': 0})
    write(OUT/'live_status.json', {'status': 'frozen_before_requests', 'planned': PLANNED, 'processed': 0,
        'downloaded': 0, 'model_calls': 0, 'confirmation_requests': 0})
    print(json.dumps({'status': 'frozen', 'planned': PLANNED, 'quotas': QUOTAS,
                      'selection_sha256': sha(OUT/'selection_manifest.json'), 'freeze_sha256': sha(OUT/'freeze.json')}))


def wait_until(reason, deadline_mono, deadline_utc, context):
    start = stamp()
    event = {'reason': reason, 'started': start, 'deadline_monotonic': deadline_mono,
             'deadline_utc': deadline_utc.isoformat()}
    write(OUT/'live_status.json', dict(context, status='waiting_'+reason,
         earliest_retry_utc=deadline_utc.isoformat(), wait_started=start))
    while True:
        remaining = max(deadline_mono-time.monotonic(), (deadline_utc-now()).total_seconds())
        if remaining <= 0:
            break
        time.sleep(min(remaining, 15.0))
    end = stamp()
    event.update(finished=end, actual_monotonic_seconds=end['monotonic']-start['monotonic'],
                 actual_utc_seconds=(datetime.fromisoformat(end['utc'])-datetime.fromisoformat(start['utc'])).total_seconds(),
                 monotonic_deadline_met=end['monotonic'] >= deadline_mono,
                 utc_deadline_met=datetime.fromisoformat(end['utc']) >= deadline_utc)
    append(OUT/'waits.jsonl', event)
    assert event['monotonic_deadline_met'] and event['utc_deadline_met']
    return event


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_once(row, attempt, index):
    meta, ident = row['metadata'], row['blind_id']
    req_headers = {'User-Agent': USER_AGENT}
    result = {'candidate_id': meta['id'], 'blind_id': ident, 'url': meta['original_url'],
              'attempt': attempt, 'request_headers': req_headers, 'started': stamp(),
              'http_status': None, 'headers': {}, 'header_items': [], 'error': None,
              'body_complete': False, 'body_capped': False, 'response_obtained': False,
              'transient_network_error': False, 'timeout_seconds': 60, 'read_limit_bytes': LIMIT}
    body = bytearray()
    response = None
    try:
        req = urllib.request.Request(meta['original_url'], headers=req_headers)
        try:
            response = urllib.request.build_opener(NoRedirect).open(req, timeout=60)
        except urllib.error.HTTPError as e:
            response = e
        result['response_obtained'] = True
        result['http_status'] = response.code
        result['headers'] = {k.lower(): v for k, v in response.headers.items()}
        result['header_items'] = list(response.headers.items())
        raw_len = result['headers'].get('content-length')
        content_length = int(raw_len) if raw_len and raw_len.isdigit() else None
        result['content_length'] = content_length
        while len(body) < LIMIT:
            chunk = response.read(min(1024*1024, LIMIT-len(body)))
            if not chunk:
                result['body_complete'] = True
                break
            body.extend(chunk)
        if content_length is not None and len(body) == content_length:
            result['body_complete'] = True
        result['body_capped'] = not result['body_complete'] and len(body) >= LIMIT
        if content_length is not None and len(body) < content_length:
            result['body_complete'] = False
        if not result['body_complete']:
            result['error'] = 'response_body_capped_or_incomplete'
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
        result['error'] = type(e).__name__ + ': ' + str(e)
        result['transient_network_error'] = True
    except Exception as e:
        result['error'] = type(e).__name__ + ': ' + str(e)
    finally:
        if response is not None:
            response.close()
    path = OUT/'responses'/f'{index:02d}_{ident}_a{attempt}.response'
    path.write_bytes(body)
    result.update(finished=stamp(), raw_path=str(path), raw_sha256=sha(path), raw_bytes=len(body))
    mime = result['headers'].get('content-type', '').lower()
    denied_text = any(s in bytes(body[:262144]).lower() for s in
        (b'captcha', b'access denied', b'access to this resource has been denied', b'robot verification', b'cf-chl-'))
    result['explicit_refusal'] = result['http_status'] in (401, 403) or ('text/html' in mime and denied_text)
    append(OUT/'requests.jsonl', result)
    return result


def process_image(row, request):
    meta, ident = row['metadata'], row['blind_id']
    raw = Path(request['raw_path']).read_bytes()
    digest = hashlib.sha1(raw).hexdigest()
    assert digest == meta['original_sha1'], 'Original version SHA1 changed'
    suffix = '.jpg' if meta['mime'] == 'image/jpeg' else '.png'
    dest = OUT/'originals'/f'{ident}{suffix}'
    dest.write_bytes(raw)
    im = Image.open(io.BytesIO(raw))
    oriented = ImageOps.exif_transpose(im).convert('RGB')
    assert (meta['width'], meta['height']) in (im.size, oriented.size), 'Metadata dimensions differ'
    record = {'candidate_id': meta['id'], 'blind_id': ident, 'status': 'downloaded',
        'original_path': str(dest), 'original_sha256': sha(dest), 'original_sha1': digest,
        'source_page': meta['source_page'], 'license': meta['license'], 'license_url': meta['license_url'],
        'artist_html': meta['artist_html'], 'credit_html': meta['credit_html'],
        'decoded_format': im.format, 'frames': getattr(im, 'n_frames', 1), 'raw_size': list(im.size),
        'oriented_size': list(oriented.size), 'orientation': im.getexif().get(274), 'bytes': len(raw),
        'modification': 'EXIF orientation/RGB; max640 LANCZOS/JPEG95; fixed224crop'}
    ordinary = im.format == {'image/jpeg': 'JPEG', 'image/png': 'PNG'}[meta['mime']] and record['frames'] == 1
    record['ordinary_single_image'] = ordinary
    # First frame derivatives are retained for repeatable duplicate detection only when quarantined.
    processed_im = oriented.copy()
    processed_im.thumbnail((640, 640), Image.Resampling.LANCZOS)
    processed = OUT/'processed'/f'{ident}.jpg'
    processed_im.save(processed, format='JPEG', quality=95)
    crop = crop_image(Image.open(processed))
    crop_path = OUT/'blind'/f'{ident}_crop.png'
    full_path = OUT/'blind'/f'{ident}_full.png'
    crop.save(crop_path)
    oriented.save(full_path)
    sys.path.insert(0, str(ROOT/'redesign_v0.4'))
    from encode_images import preprocess
    arr = np.asarray(crop).astype(np.float32)/255
    arr = (arr-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
    assert np.array_equal(arr.transpose(2, 0, 1), preprocess(processed).numpy())
    record.update(processed_path=str(processed), processed_sha256=sha(processed),
        crop_path=str(crop_path), crop_sha256=sha(crop_path), full_path=str(full_path), full_sha256=sha(full_path),
        hashes=hash_image(Image.open(processed)), preprocess_matches_legacy=True,
        status='downloaded' if ordinary else 'quarantined_format', visual_review='pending')
    return record


def finalize(records, frozen, terminal_status):
    inputs = read(OUT/'comparison_inputs.json')
    prior = []
    for r in inputs['old_available']:
        prior.append({'id': r['id'], 'group': 'old_available90', 'original_sha1': r.get('original_sha1'),
            'processed_sha256': sha(r['path']), **hash_image(Image.open(r['path']))})
    for r in inputs['previous_batch']:
        hashes = hash_image(Image.open(r['processed_path']))
        assert hashes == r['hashes']
        prior.append({'id': r['blind_id'], 'group': r['prior_batch']+'_all',
            'original_sha1': r['original_sha1'], 'processed_sha256': sha(r['processed_path']), **hashes})
    new = [r for r in records if r['status'] in ('downloaded', 'quarantined_format')]
    flags, comparisons = [], 0
    for r in new:
        for other in prior:
            comparisons += 1
            pd = (int(r['hashes']['phash'], 16)^int(other['phash'], 16)).bit_count()
            dd = (int(r['hashes']['dhash'], 16)^int(other['dhash'], 16)).bit_count()
            exact_original = bool(other.get('original_sha1')) and r['original_sha1'] == other['original_sha1']
            exact_processed = r['processed_sha256'] == other['processed_sha256']
            if pd <= 8 or dd <= 6 or exact_original or exact_processed:
                flags.append({'a': r['blind_id'], 'b': other['id'], 'b_group': other['group'],
                    'phash_distance': pd, 'dhash_distance': dd, 'original_sha1_equal': exact_original,
                    'processed_sha256_equal': exact_processed, 'human_or_independent_review': 'pending'})
        prior.append({'id': r['blind_id'], 'group': 'workflow_003', 'original_sha1': r['original_sha1'],
                      'processed_sha256': r['processed_sha256'], **r['hashes']})
    assert comparisons == len(new)*REFERENCE_COUNT + len(new)*(len(new)-1)//2
    write(OUT/'near_duplicate_flags.json', {'old_records': 101, 'old_available_pixels': 90,
        'old_unavailable': inputs['old_unavailable'], 'previous_batch_pixels': 43, 'previous_batches': {'pixel_workflow_001': 24, 'pixel_workflow_002': 19}, 'new_pixels': len(new),
        'comparisons': comparisons, 'flags': flags, 'confirmation_pixels_compared': False,
        'review_pending': True, 'pixel_hash_rows': prior})
    ordered = sorted([r for r in new if r['ordinary_single_image']], key=lambda r: r['blind_id'])
    packet_dir = OUT/'review_round1'
    packet_dir.mkdir(exist_ok=False)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 18)
    sheets = []
    image_paths = []
    for start in range(0, len(ordered), 4):
        canvas = Image.new('RGB', (1200, 650), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, r in enumerate(ordered[start:start+4]):
            ident = r['blind_id']
            x, y = j % 2 * 600, j // 2 * 325
            draw.text((x+10, y+8), ident, fill='black', font=font)
            full = Image.open(r['full_path']).convert('RGB')
            full.thumbnail((350, 270))
            canvas.paste(full, (x+8+(350-full.width)//2, y+38+(270-full.height)//2))
            canvas.paste(Image.open(r['crop_path']), (x+370, y+55))
            image_paths.append({'id': ident, 'full': r['full_path'], 'crop': r['crop_path']})
        path = packet_dir/f'sheet_{start//4+1:02d}.png'
        canvas.save(path)
        sheets.append(str(path))
    packet = {'ids': [r['blind_id'] for r in ordered], 'sheets': sheets, 'images': image_paths,
        'image_sha256': {p: sha(p) for p in sheets},
        'individual_image_sha256': {r[k]: sha(r[k]) for r in ordered for k in ('full_path', 'crop_path')},
        'criteria': ['real photograph', 'food or visible water in transparent container identifiable',
            'crop retains resource', 'no prominent text or watermark', 'no mixed foods or ambiguous liquid'],
        'scope': 'AI visual review; anonymous full image at left and actual224crop at right. No source or model information.',
        'model_scores_available': False}
    write(packet_dir/'packet.json', packet)
    requests = [json.loads(s) for s in (OUT/'requests.jsonl').read_text().splitlines()]
    waits = [json.loads(s) for s in (OUT/'waits.jsonl').read_text().splitlines()] if (OUT/'waits.jsonl').exists() else []
    intervals = []
    retries = []
    for i, r in enumerate(requests):
        assert sha(r['raw_path']) == r['raw_sha256'] and r['raw_bytes'] <= LIMIT
        if i:
            prev = requests[i-1]
            gap_m = r['started']['monotonic']-prev['finished']['monotonic']
            gap_u = (datetime.fromisoformat(r['started']['utc'])-datetime.fromisoformat(prev['finished']['utc'])).total_seconds()
            assert gap_m >= 10 and gap_u >= 10
            intervals.append({'monotonic': gap_m, 'utc': gap_u})
            if prev['http_status'] in (429, 503):
                assert r['candidate_id'] == prev['candidate_id'] and r['attempt'] == 2
                delay, kind = retry_delay(prev['headers'], datetime.fromisoformat(prev['finished']['utc']))
                assert gap_m >= delay+1 and gap_u >= delay+1
                retries.append({'header_kind': kind, 'required_seconds': delay, 'required_with_margin': delay+1,
                                'monotonic_seconds': gap_m, 'utc_seconds': gap_u})
    verify_inputs(frozen)
    write(OUT/'download_manifest.json', {'status': terminal_status, 'records': records, 'planned': PLANNED,
        'downloaded_and_integrity_checked': len(new), 'ordinary_review_candidates': len(ordered),
        'format_quarantined': len(new)-len(ordered), 'finished_utc': now().isoformat(),
        'model_calls': 0, 'confirmation_requests': 0, 'freeze_sha256': sha(OUT/'freeze.json')})
    qa = {'data_integrity_passed': True, 'request_timing_passed': True,
        'processed_candidates': len(records), 'planned_candidates': PLANNED, 'request_count': len(requests),
        'downloaded': len(new), 'ordinary_candidates': len(ordered), 'quarantine': [r['blind_id'] for r in new if not r['ordinary_single_image']],
        'old_available_pixels': 90, 'old_unavailable_pixels': 11, 'previous_batch_pixels': 43, 'previous_batches': {'pixel_workflow_001': 24, 'pixel_workflow_002': 19},
        'near_duplicate_comparisons': comparisons, 'near_duplicate_flags': len(flags),
        'retry_after_waits': retries, 'normal_request_intervals': intervals,
        'wait_deadlines_passed': all(w['monotonic_deadline_met'] and w['utc_deadline_met'] for w in waits),
        'all_response_bytes_archived_with_caps': True, 'legacy_preprocessing_equality_checks': len(new),
        'packet_sha256': sha(packet_dir/'packet.json'), 'sheet_count': len(sheets),
        'visual_review': 'not_performed_by_collector', 'license_identity_review': 'pending',
        'model_calls': 0, 'confirmation_requests': 0, 'source_hashes_unchanged': True,
        'old_batch_timing_failure_not_rewritten': True}
    write(OUT/'collection_qa.json', qa)
    write(OUT/'live_status.json', {'status': terminal_status, 'planned': PLANNED, 'processed': len(records),
        'downloaded': len(new), 'ordinary': len(ordered), 'quarantined': len(new)-len(ordered),
        'request_count': len(requests), 'packet_ready': True, 'model_calls': 0, 'confirmation_requests': 0,
        'finished_utc': now().isoformat()})
    print(json.dumps({'status': terminal_status, 'processed': len(records), 'downloaded': len(new),
        'ordinary': len(ordered), 'quarantined': len(new)-len(ordered), 'packet_ready': True}), flush=True)


def collect():
    frozen = read(OUT/'freeze.json')
    verify_inputs(frozen)
    assert (platform.python_version(), PIL.__version__, np.__version__) == ('3.12.14', '12.3.0', '1.26.4')
    assert not (OUT/'requests.jsonl').exists(), 'No automatic restart/retry after a stopped batch'
    write(OUT/'process_handle.json', {'pid': os.getpid(), 'started_utc': now().isoformat(),
        'source_sha256': sha(__file__), 'status_path': str(OUT/'live_status.json'),
        'command': 'PYTHONPATH=/Users/xia/.cache/uv/archive-v0/S3z4TeuYuWgwVQ-G .venv/bin/python '+str(Path(__file__))+' collect'})
    rows = read(OUT/'selection_manifest.json')['rows']
    records = []
    previous = None
    terminal = 'download_phase_complete_review_pending'
    stop = False
    for index, row in enumerate(rows, 1):
        record = {'candidate_id': row['metadata']['id'], 'blind_id': row['blind_id'], 'status': 'pending'}
        for attempt in (1, 2):
            context = {'pid': os.getpid(), 'planned': PLANNED, 'processed': len(records),
                'downloaded': sum(r['status'] in ('downloaded', 'quarantined_format') for r in records),
                'current_ordinal': index, 'attempt': attempt, 'model_calls': 0, 'confirmation_requests': 0}
            if previous is not None:
                wait_until('serial_interval', previous['finished']['monotonic']+10,
                           datetime.fromisoformat(previous['finished']['utc'])+timedelta(seconds=10), context)
            write(OUT/'live_status.json', dict(context, status='requesting', started_utc=now().isoformat()))
            req = request_once(row, attempt, index)
            previous = req
            code = req['http_status']
            if req['explicit_refusal'] or (code is not None and 300 <= code < 400):
                record.update(status='error', error='explicit_access_refusal_or_redirect')
                stop = True
                terminal = 'stopped_access_refusal'
                break
            if code == 200 and req['error'] is None and req['body_complete']:
                try:
                    record = process_image(row, req)
                except Exception as e:
                    record.update(status='error', error=type(e).__name__+': '+str(e), raw_path=req['raw_path'])
                break
            record.update(status='error', error=req['error'] or f'HTTP {code}', raw_path=req['raw_path'])
            if code in (429, 503):
                if attempt == 2:
                    stop, terminal = True, 'stopped_rate_limit_after_allowed_retry'
                    break
                try:
                    delay, kind = retry_delay(req['headers'], datetime.fromisoformat(req['finished']['utc']))
                except Exception:
                    stop, terminal = True, 'stopped_unparseable_retry_after'
                    break
                context['retry_after_kind'] = kind
                context['required_wait_seconds_with_margin'] = delay+1
                deadline_utc = datetime.fromisoformat(req['finished']['utc'])+timedelta(seconds=delay+1)
                print(json.dumps(dict(status='waiting_retry_after', processed=len(records), planned=PLANNED,
                    required_wait_seconds_with_margin=delay+1, earliest_retry_utc=deadline_utc.isoformat())), flush=True)
                wait_until('retry_after', req['finished']['monotonic']+delay+1, deadline_utc, context)
                continue
            if attempt == 1 and (req['transient_network_error'] or (code is not None and code >= 500)):
                wait_until('transient_retry', req['finished']['monotonic']+5,
                    datetime.fromisoformat(req['finished']['utc'])+timedelta(seconds=5), context)
                continue
            break
        records.append(record)
        write(OUT/'download_manifest.json', {'status': 'running', 'records': records, 'planned': PLANNED,
            'model_calls': 0, 'confirmation_requests': 0, 'freeze_sha256': sha(OUT/'freeze.json')})
        print(json.dumps({'processed': len(records), 'planned': PLANNED, 'status': record['status']}), flush=True)
        if stop:
            break
    finalize(records, frozen, terminal)


def selftest():
    when = datetime(2026, 9, 16, 0, 0, 0, tzinfo=timezone.utc)
    assert retry_delay({'ReTrY-AfTeR': '600'}, when) == (600.0, 'seconds')
    assert retry_delay({'retry-after': 'Wed, 16 Sep 2026 00:10:00 GMT'}, when) == (600.0, 'http_date')
    assert retry_delay({}, when) == (30.0, 'missing_default_30')
    assert retry_delay({'Retry-After': 'Tue, 15 Sep 2026 23:00:00 GMT'}, when)[0] == 0
    for value in ('-1', 'NaN', 'not-a-date'):
        try:
            retry_delay({'Retry-After': value}, when)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('Invalid retry header was accepted')
    sys.path.insert(0, str(BASE))
    import collect_workflow_pixels as old
    rng = np.random.default_rng(160926)
    for shape in ((256, 391), (432, 277), (640, 640)):
        im = Image.fromarray(rng.integers(0, 256, (*shape, 3), dtype=np.uint8))
        assert old.hash_image(im) == hash_image(im)
        assert np.array_equal(np.asarray(old.crop_image(im)), np.asarray(crop_image(im)))
    print(json.dumps({'selftest_passed': True, 'network_requests': 0, 'synthetic_images': 3,
                      'retry_cases': 7, 'hash_and_crop_match_frozen_algorithm': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'collect', 'selftest'])
    args = parser.parse_args()
    {'prepare': prepare, 'collect': collect, 'selftest': selftest}[args.mode]()
