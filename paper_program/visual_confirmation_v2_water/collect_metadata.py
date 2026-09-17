"""Bounded official Commons API metadata collector. No image URL fetches."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
import json
import re
import time
from metadata_helpers import HERE, CFG, sha, rank, title_key, all_keys, components, parse_page

def now(): return datetime.now(timezone.utc).isoformat()
def stamp(): return dict(utc=now(), unix=time.time(), monotonic=time.monotonic())
def write(name, value): (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def append(name, value):
    with (HERE / name).open('a') as f: f.write(json.dumps(value, ensure_ascii=False) + '\n')

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): return None

def retry_delay(headers, received_unix, is429=False):
    value = next((str(v) for k, v in headers if k.casefold() == 'retry-after'), None)
    base = CFG['missing_429_retry_after_seconds'] if is429 else CFG['transient_retry_seconds']
    kind = 'fallback'
    if value is not None:
        try:
            base = max(0.0, float(value)); kind = 'seconds'
        except ValueError:
            try:
                dt = parsedate_to_datetime(value)
                if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                base = max(0.0, dt.timestamp() - received_unix); kind = 'http_date'
            except (ValueError, TypeError, OverflowError): pass
    return base + CFG['retry_safety_margin_seconds'], kind

class Client:
    def __init__(self):
        self.records = []; self.logical = 0; self.extra_retries = 0; self.last = None
        self.opener = build_opener(NoRedirect())
        (HERE / 'raw').mkdir(exist_ok=True)

    def wait(self, seconds, reason, origin=None):
        origin = origin or stamp()
        start = stamp()
        targets = dict(unix=origin['unix'] + seconds, monotonic=origin['monotonic'] + seconds)
        write('live_status.json', dict(status='waiting', reason=reason, started=start,
            required_seconds=seconds, targets=targets, http_requests=len(self.records),
            logical_requests=self.logical, pixel_requests=0, model_calls=0))
        while True:
            remaining = max(targets['unix'] - time.time(), targets['monotonic'] - time.monotonic())
            if remaining <= 0: break
            time.sleep(min(remaining, 1.0))
        end = stamp()
        append('waits.jsonl', dict(reason=reason, origin=origin, entered=start, finished=end,
            required_seconds=seconds, elapsed_utc=end['unix'] - origin['unix'],
            elapsed_monotonic=end['monotonic'] - origin['monotonic']))

    def fetch(self, params, label):
        self.logical += 1
        assert self.logical <= CFG['logical_request_limit'], 'logical budget exhausted'
        params = dict(action='query', format='json', formatversion=2, maxlag=5, **params)
        url = CFG['endpoint'] + '?' + urlencode(params)
        assert urlparse(url).netloc == 'commons.wikimedia.org' and urlparse(url).path == '/w/api.php'
        for attempt in range(2):
            assert len(self.records) < CFG['http_request_limit'], 'HTTP budget exhausted'
            if self.last: self.wait(CFG['response_interval_seconds'], 'serial_response_gap', self.last)
            started = stamp(); body = bytearray(); headers = []; code = None; err = None; capped = False
            try:
                request = Request(url, headers={'User-Agent': CFG['user_agent'], 'Accept': 'application/json'})
                try: response = self.opener.open(request, timeout=CFG['timeout_seconds'])
                except HTTPError as exc: response = exc
                with response:
                    code = response.code; headers = list(response.headers.items())
                    while True:
                        chunk = response.read(min(65536, CFG['response_limit_bytes'] + 1 - len(body)))
                        if not chunk: break
                        body.extend(chunk)
                        if len(body) > CFG['response_limit_bytes']:
                            capped = True; break
            except Exception as exc: err = repr(exc)
            finished = stamp(); self.last = finished
            name = f'raw/{len(self.records):04d}_{label}_a{attempt + 1}.response'
            (HERE / name).write_bytes(body)
            data = None; parse_error = None
            if code == 200 and not err and not capped:
                try: data = json.loads(body)
                except Exception as exc: parse_error = repr(exc)
            record = dict(url=url, params=params, label=label, logical_request=self.logical, attempt=attempt + 1,
                started=started, finished=finished, http_status=code, header_items=headers,
                exception=err, parse_error=parse_error, response_capped=capped, response_bytes=len(body),
                response_file=name, response_sha256=sha(body), api_error=data.get('error') if isinstance(data, dict) else None)
            self.records.append(record); append('requests.jsonl', record)
            write('live_status.json', dict(status='request_complete', http_requests=len(self.records),
                logical_requests=self.logical, last_label=label, last_http_status=code, pixel_requests=0, model_calls=0))
            print(json.dumps({'request': len(self.records), 'label': label, 'status': code, 'api_error': record['api_error']}, ensure_ascii=False), flush=True)
            if code == 200 and not err and not capped and isinstance(data, dict) and not data.get('error'):
                return data, record
            if code in (401, 403, 407) or (code is not None and 300 <= code < 400):
                raise RuntimeError(f'access refusal or redirect; stop:HTTP{code}')
            api_code = (record['api_error'] or {}).get('code')
            transient = code in (429, 503) or (code is None and err is not None) or api_code == 'maxlag'
            if not transient or attempt == 1 or self.extra_retries >= CFG['extra_retry_limit']:
                raise RuntimeError(f'bounded request failed; stop:{label} HTTP{code} {api_code} {err or parse_error}')
            self.extra_retries += 1
            seconds, kind = retry_delay(headers, finished['unix'], code == 429)
            self.wait(seconds, 'retry_after_' + kind, finished)
        raise AssertionError('unreachable')

def summarize(rows):
    groups = []
    for group in components(rows):
        keys = set().union(*(all_keys(r) for r in group))
        old_hit = any(r['blocked_identity_or_source_keys'] or 'old_or_reserved_exact_file' in r['exclusion_reasons'] for r in group)
        if old_hit:
            for r in group:
                if 'connected_component_contains_excluded_identity' not in r['exclusion_reasons']:
                    r['exclusion_reasons'].append('connected_component_contains_excluded_identity')
                r['metadata_provisionally_eligible'] = False
        ident = sha('|'.join(sorted(keys)).encode())
        for r in group: r['cluster_id'] = ident
        eligible = sorted([r for r in group if r['metadata_provisionally_eligible']], key=lambda r: (r['frame_rank_hash'], r['id']))
        groups.append(dict(cluster_id=ident, cluster_rank=rank('cluster', ident), identity_keys=sorted(keys),
            members=[r['id'] for r in group], blocked_by_old_or_reserved=old_hit,
            eligible_representative=eligible[0]['id'] if eligible else None))
    groups.sort(key=lambda g: g['cluster_rank'])
    write('author_groups.json', dict(status='metadata_clusters_only_not_verified_people', groups=groups,
        provisionally_eligible_clusters=sum(g['eligible_representative'] is not None for g in groups),
        pixel_allocation_performed=False))
    return groups

def main():
    assert not (HERE / 'requests.jsonl').exists(), 'Single bounded execution; do not restart.'
    freeze = json.loads((HERE / 'freeze.json').read_text())
    from pathlib import Path
    for name, h in freeze['source_hashes'].items(): assert sha(Path(name).read_bytes()) == h, name
    exclusion = json.loads((HERE / 'exclusion_registry.json').read_text())
    client = Client(); rows = []; categories = []; candidates = {}; selected = []; error = None
    state = dict(status='running', started_utc=now(), freeze_sha256=sha((HERE / 'freeze.json').read_bytes()),
        pixel_requests=0, image_views=0, model_calls=0, confirmation_pixel_access=0)
    write('status.json', state)
    try:
        exact, request = client.fetch(dict(titles='|'.join(CFG['exact_categories']), redirects=1, prop='info|categoryinfo'), 'exact_categories')
        exact_pages = exact.get('query', {}).get('pages', [])
        valid_exact = [p['title'] for p in exact_pages if not p.get('missing', False) and p.get('ns') == 14 and re.search(CFG['category_name_pattern'], p['title'], re.I)]
        search_categories = []
        discovery = [dict(kind='exact', response=request['response_file'], pages=exact_pages, redirects=exact.get('query', {}).get('redirects', []))]
        for i, q in enumerate(CFG['category_queries']):
            data, request = client.fetch(dict(list='search', srsearch=q, srnamespace=14, srlimit=CFG['category_search_limit'], srprop='snippet'), f'category_search_{i}')
            found = data.get('query', {}).get('search', [])
            discovery.append(dict(kind='search', query=q, response=request['response_file'], results=found,
                searchinfo=data.get('query', {}).get('searchinfo'), truncated='continue' in data))
            search_categories.extend(p['title'] for p in found if re.search(CFG['category_name_pattern'], p['title'], re.I))
        categories = sorted(set(valid_exact), key=lambda s: rank('category', s))
        categories += sorted(set(search_categories) - set(categories), key=lambda s: rank('category', s))
        categories = categories[:CFG['category_limit']]
        write('category_discovery.json', dict(discovery=discovery, selected_categories=categories,
            rule='Water+glass category-name filter, exact-existing priority then deterministic hash; direct files only'))
        def add(p, source):
            if p.get('ns') != 6: return
            row = candidates.setdefault(p['pageid'], dict(pageid=p['pageid'], title=p['title'], sources=[]))
            if source not in row['sources']: row['sources'].append(source)
        roots = []
        for i, cat in enumerate(categories):
            continuation = {}; pages = []
            for pn in range(CFG['category_pages_per_category']):
                data, request = client.fetch(dict(list='categorymembers', cmtitle=cat, cmtype='file',
                    cmlimit=CFG['category_page_size'], **continuation), f'category_{i}_page_{pn}')
                members = data.get('query', {}).get('categorymembers', [])
                for p in members: add(p, dict(kind='category', category=cat, response=request['response_file']))
                pages.append(dict(response=request['response_file'], count=len(members), continuation=data.get('continue')))
                continuation = data.get('continue', {})
                if not continuation: break
            roots.append(dict(category=cat, pages=pages, truncated=bool(continuation)))
        search_logs = []
        for i, q in enumerate(CFG['file_queries']):
            data, request = client.fetch(dict(list='search', srsearch=q, srnamespace=6, srlimit=CFG['file_search_limit'], srprop='snippet'), f'file_search_{i}')
            found = data.get('query', {}).get('search', [])
            for p in found: add(p, dict(kind='file_search', query=q, response=request['response_file']))
            search_logs.append(dict(query=q, count=len(found), response=request['response_file'],
                searchinfo=data.get('query', {}).get('searchinfo'), truncated='continue' in data))
        exact_excluded = []; eligible_frame = []
        for r in candidates.values():
            r['frame_rank_hash'] = rank('file', r['pageid'])
            if r['pageid'] in exclusion['blocked_pageids'] or title_key(r['title']) in exclusion['blocked_titles']:
                exact_excluded.append(r)
            else: eligible_frame.append(r)
        selected = sorted(eligible_frame, key=lambda r: (r['frame_rank_hash'], r['pageid']))[:CFG['metadata_file_limit']]
        write('sampling_frame.json', dict(created_utc=now(), roots=roots, file_searches=search_logs,
            unique_file_count=len(candidates), candidates=list(candidates.values()),
            excluded_exact_old_or_reserved=exact_excluded, requested_metadata=selected,
            capped_at300=len(eligible_frame) > CFG['metadata_file_limit']))
        # The complete detail list is saved before the first detail request.
        write('metadata_selection_freeze.json', dict(frozen_utc=now(), count=len(selected),
            sampling_frame_sha256=sha((HERE / 'sampling_frame.json').read_bytes()),
            ordered_pageids=[r['pageid'] for r in selected], pixel_requests=0))
        for offset in range(0, len(selected), CFG['metadata_batch_size']):
            batch = selected[offset:offset + CFG['metadata_batch_size']]
            data, request = client.fetch(dict(pageids='|'.join(str(r['pageid']) for r in batch),
                prop='imageinfo|info|revisions', iiprop='timestamp|user|userid|url|size|sha1|mime|extmetadata',
                rvprop='ids|timestamp', iiextmetadatalanguage='en'), f'metadata_{offset:04d}')
            pages = {p['pageid']: p for p in data.get('query', {}).get('pages', [])}
            rows.extend(parse_page(pages.get(r['pageid'], {}), r, request, exclusion) for r in batch)
            write('manifest.json', dict(status='collecting', metadata_count=len(rows), images=rows))
        state['status'] = 'metadata_collection_complete'
    except Exception as exc:
        error = repr(exc); state['status'] = 'bounded_collection_stopped'; state['error'] = error
    groups = summarize(rows)
    unchanged = all(sha(Path(p).read_bytes()) == h for p, h in exclusion['source_hashes'].items())
    state.update(finished_utc=now(), actual_http_requests=len(client.records), logical_requests=client.logical,
        extra_retries=client.extra_retries, category_count=len(categories), unique_frame_files=len(candidates),
        selected_metadata_files=len(selected), metadata_count=len(rows),
        provisionally_eligible_records=sum(r['metadata_provisionally_eligible'] for r in rows),
        provisionally_eligible_clusters=sum(g['eligible_representative'] is not None for g in groups),
        unchanged_v1_inputs=unchanged, pixels_ready=False, allocations_performed=False,
        threshold80_met_in_this_bounded_metadata_sample=sum(g['eligible_representative'] is not None for g in groups) >= 80)
    write('manifest.json', dict(status=state['status'], metadata_count=len(rows), images=rows))
    write('status.json', state); write('live_status.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    if error or not unchanged: raise SystemExit(1)

if __name__ == '__main__': main()
