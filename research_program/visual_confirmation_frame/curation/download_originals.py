"""Bounded, single-attempt acquisition of a frozen metadata nomination stage.

Preparation and verification make no network requests. Original bytes are kept;
this script performs no visual judgement, feature extraction or model inference.
"""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError
from urllib.parse import urlparse
import argparse
import hashlib
import json
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parent
MAX_FILE = 16 * 1024**2
MAX_TOTAL = 2 * 1024**3


def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def prepare(ranking, out, stage):
    ranking, out = Path(ranking).resolve(), Path(out).resolve()
    require(not out.exists(), 'Do not overwrite an existing acquisition')
    order = read(ranking / 'candidate_order.json')
    require(set(order) == {'apple', 'banana', 'orange', 'water'}, 'Four frozen category arrays required')
    all_jobs = [r for key in ('water', 'apple', 'banana', 'orange') for r in order[key]]
    require(all(len(order[k]) >= (84 if k == 'water' else 28) for k in order), 'Metadata quotas are not feasible')
    require(len(all_jobs) <= 252 and len({r['pageid'] for r in all_jobs}) == len(all_jobs), 'Bounded unique files')
    require(len({r['component_id'] for r in all_jobs}) == len(all_jobs), 'One candidate per global component')
    require(len({r['review_id'] for r in all_jobs}) == len(all_jobs), 'Unique anonymous IDs')
    require(sum(r['size'] for r in all_jobs) <= MAX_TOTAL, 'Declared full-pool bytes exceed budget')
    for row in all_jobs:
        parsed = urlparse(row['original_url'])
        require(parsed.scheme == 'https' and parsed.netloc == 'upload.wikimedia.org', 'Only archived Commons original URLs')
        require(0 < row['size'] <= MAX_FILE, 'Declared file size outside frozen budget')
        require(len(row['original_file_sha1']) == 40, 'Original SHA1 missing')
        require(row['review_id'].isalnum() or all(c.isalnum() or c in '_-' for c in row['review_id']), 'Unsafe anonymous file ID')
    keys = ('water',) if stage == 'water' else ('apple', 'banana', 'orange')
    jobs = [dict(row, request_index=i) for i, row in enumerate(r for key in keys for r in order[key])]
    sources = {str(p): sha(p) for p in sorted(ranking.rglob('*')) if p.is_file()}
    for p in (Path(__file__), ROOT / 'plan.md', ROOT / 'test_download_originals.py'):
        sources[str(p.resolve())] = sha(p)
    plan = dict(status='prepared_not_executed', prepared_utc=now(), stage=stage, jobs=jobs,
                source_sha256=sources, fixed_download_order=list(keys), original_get_limit=len(jobs),
                full_pool_get_limit=252, full_pool_declared_bytes=sum(r['size'] for r in all_jobs),
                stage_declared_bytes=sum(r['size'] for r in jobs), file_byte_limit=MAX_FILE,
                full_pool_byte_limit=MAX_TOTAL, timeout_seconds=20,
                delay_after_completion_seconds=5, retries=0, redirects=False,
                stop_round_on_request_error=True, model_calls=0, pixel_review=False)
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out / 'download_originals.py')
    shutil.copyfile(ROOT / 'plan.md', out / 'curation_plan.md')
    shutil.copyfile(ROOT / 'test_download_originals.py', out / 'test_download_originals.py')
    write(out / 'plan.json', plan)
    write(out / 'freeze.json', {'plan_sha256': sha(out / 'plan.json')})
    return {k: plan[k] for k in ('stage', 'original_get_limit', 'stage_declared_bytes', 'full_pool_declared_bytes')}


def verify(out):
    out = Path(out).resolve()
    plan = read(out / 'plan.json')
    require(sha(out / 'plan.json') == read(out / 'freeze.json')['plan_sha256'], 'Frozen acquisition plan changed')
    for path, digest in plan['source_sha256'].items():
        require(sha(path) == digest, 'Frozen source changed: ' + path)
    require(sha(out / 'download_originals.py') == sha(__file__), 'Acquisition snapshot differs')
    require(sha(out / 'curation_plan.md') == sha(ROOT / 'plan.md'), 'Curation snapshot differs')
    return plan


def stream_body(response, path, limit, *, monotonic=time.monotonic):
    """Never read or save more than the declared remaining byte ceiling."""
    started = monotonic()
    total = 0
    reached_eof = False
    with Path(path).open('xb') as stream:
        while total < limit:
            require(monotonic() - started <= 20, 'Response streaming time limit exceeded')
            block = response.read(min(65536, limit - total))
            if not block:
                reached_eof = True
                break
            total += len(block)
            require(total <= limit, 'Reader returned more than requested')
            stream.write(block)
    return total, reached_eof


def execute(out, opener=None):
    out = Path(out).resolve()
    plan = verify(out)
    require(plan['stage'] == 'water', 'Food stage requires separately frozen successful-water authorization')
    target = out / 'execution'
    target.mkdir(exist_ok=False)
    (target / 'originals').mkdir()
    (target / 'receipts').mkdir()
    write(target / 'started.json', {'started_utc': now(), 'plan_sha256': sha(out / 'plan.json')})
    opener = opener or build_opener(NoRedirect())
    receipts, total_bytes, previous_completed, stop = [], 0, None, None
    try:
        for job in plan['jobs']:
            if previous_completed is not None:
                delay = 5 - (time.monotonic() - previous_completed)
                if delay > 0:
                    time.sleep(delay)
            started = time.monotonic()
            receipt = dict(request_index=job['request_index'], pageid=job['pageid'], review_id=job['review_id'],
                           url=job['original_url'], requested_utc=now(),
                           gap_after_previous_completion_seconds=None if previous_completed is None else started - previous_completed)
            stem = f"{job['request_index']:03d}"
            write(target / 'receipts' / (stem + '_started.json'), receipt)
            raw = target / 'originals' / (job['review_id'] + '.bin')
            limit = min(MAX_FILE, MAX_TOTAL - total_bytes)
            try:
                require(limit > 0, 'Total byte budget exhausted')
                request = Request(job['original_url'], headers={
                    'User-Agent': 'LanguageFormationResearch/1.0 (bounded academic image curation)',
                    'Accept': 'image/jpeg,image/png,image/tiff,image/webp'})
                with opener.open(request, timeout=20) as response:
                    receipt.update(http_status=response.status, response_headers=dict(response.headers), final_url=response.geturl())
                    require(response.status == 200 and response.geturl() == job['original_url'], 'Unexpected status or redirect')
                    declared = response.headers.get('Content-Length')
                    if declared is not None:
                        require(declared.isdigit() and 0 < int(declared) <= limit, 'Response Content-Length exceeds budget')
                    read_limit = min(limit, int(declared)) if declared is not None else limit
                    count, eof = stream_body(response, raw, read_limit)
                receipt.update(response_bytes=count, reached_eof=eof, original_sha1=sha(raw, 'sha1'), sha256=sha(raw), file=str(raw))
                # At an exact byte cap, trust only a matching declared length and
                # archived SHA; never make an extra read to test for overflow.
                require(eof or (declared is not None and int(declared) == count), 'Response reached byte cap without verifiable endpoint')
                require(declared is None or count == int(declared), 'Response ended before declared Content-Length')
                receipt['status'] = 'downloaded' if count == job['size'] and receipt['original_sha1'] == job['original_file_sha1'] else 'metadata_mismatch_rejected'
            except Exception as error:
                receipt.update(status='request_failed', error=repr(error), traceback=traceback.format_exc())
                if isinstance(error, HTTPError):
                    receipt.update(http_status=error.code, response_headers=dict(error.headers))
                    error_path = target / 'receipts' / (stem + '_error.bin')
                    try:
                        stream_body(error, error_path, min(1024**2, limit))
                    except Exception as body_error:
                        receipt['error_body_error'] = repr(body_error)
                    if error_path.exists():
                        receipt.update(error_body=str(error_path), error_body_bytes=error_path.stat().st_size,
                                       error_body_sha256=sha(error_path))
                if raw.exists():
                    receipt.update(partial_file=str(raw), partial_bytes=raw.stat().st_size, partial_sha256=sha(raw))
                stop = {'request_index': job['request_index'], 'error': repr(error)}
            total_bytes += raw.stat().st_size if raw.exists() else 0
            total_bytes += receipt.get('error_body_bytes', 0)
            require(total_bytes <= MAX_TOTAL, 'Actual aggregate byte ceiling exceeded')
            previous_completed = time.monotonic()
            receipt.update(completed_utc=now(), elapsed_seconds=previous_completed - started, cumulative_response_bytes=total_bytes)
            write(target / 'receipts' / (stem + '_receipt.json'), receipt)
            receipts.append(receipt)
            print(json.dumps({k: receipt.get(k) for k in ('request_index', 'review_id', 'status', 'response_bytes', 'http_status')}), flush=True)
            if stop:
                break
        verify(out)
        result = dict(status='stopped_on_error' if stop else 'download_stage_complete', completed_utc=now(),
                      requests=len(receipts), planned_requests=len(plan['jobs']), response_bytes=total_bytes,
                      downloaded=sum(r['status'] == 'downloaded' for r in receipts),
                      metadata_mismatch_rejected=sum(r['status'] == 'metadata_mismatch_rejected' for r in receipts),
                      stopped=stop, not_requested_pageids=[r['pageid'] for r in plan['jobs'][len(receipts):]],
                      visual_acceptance_completed=False, model_calls=0,
                      minimum_observed_gap_seconds=min((r['gap_after_previous_completion_seconds'] for r in receipts[1:]), default=None))
        write(target / 'result.json', result)
        return result
    except BaseException as error:
        write(target / 'failure.json', {'error': repr(error), 'traceback': traceback.format_exc(), 'completed_utc': now()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'execute'))
    parser.add_argument('--ranking', type=Path, default=ROOT / 'ranking_001')
    parser.add_argument('--out', type=Path, default=ROOT / 'water_download_001')
    parser.add_argument('--stage', choices=('water', 'food'), default='water')
    args = parser.parse_args()
    print(json.dumps(prepare(args.ranking, args.out, args.stage) if args.command == 'prepare'
                     else execute(args.out) if args.command == 'execute' else verify(args.out), ensure_ascii=False, indent=2))
