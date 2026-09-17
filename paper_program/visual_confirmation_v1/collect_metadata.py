"""Frozen-plan, metadata-only Commons sampling. Never fetch image URLs."""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse, unquote, urljoin
import datetime as dt
import argparse
from email.utils import parsedate_to_datetime
import hashlib
import html
import json
import re
import time
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'redesign_v0.4/data'
SALT = 'visual_confirmation_v1|2026-09-15|prospective|no_model_scores'
API = 'https://commons.wikimedia.org/w/api.php'
UA = 'VisualConfirmationMetadata/1.0 (local academic reproducibility study; metadata-only)'
STRATA = {'apple': ['Category:Apples'], 'banana': ['Category:Bananas'],
          'orange': ['Category:Oranges'],
          'water': ['Category:Glasses of water', 'Category:Water bottles']}
CUTOFF = '2026-09-14T23:59:59Z'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def clean(value):
    return re.sub(r'\s+', ' ', html.unescape(re.sub('<[^>]*>', '', str(value)))).strip()


def norm(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', clean(value)).replace('_', ' ')).strip().casefold()


def title_key(value):
    return norm(re.sub(r'^File:', '', value, flags=re.I))


def links(value):
    return [urljoin('https://commons.wikimedia.org', html.unescape(v))
            for v in re.findall(r'href=[\"\']([^\"\']+)', str(value))]


def author_keys(value):
    text = norm(value)
    bad = not text or text in ('[1]', 'unknown', 'anonymous') or bool(re.search(r'\{\{|unknown author|author unknown|not provided', text))
    if bad:
        return [], False
    keys = {'text:' + text}
    for link in links(value):
        u = urlparse(link)
        path = unquote(u.path)
        if u.hostname in ('commons.wikimedia.org', 'en.wikipedia.org'):
            m = re.search(r'/wiki/(User|Creator):([^/#?]+)', path, re.I)
            if m:
                keys.add('commons-' + m[1].lower() + ':' + norm(m[2]))
                keys.add('text:' + norm(m[2]))
        if u.hostname in ('flickr.com', 'www.flickr.com'):
            m = re.match(r'/(?:photos|people)/([^/]+)', path)
            if m:
                keys.add('flickr-author:' + m[1].casefold())
    return sorted(keys), True


def source_keys(value):
    result = set()
    for link in links(value):
        u = urlparse(link)
        if u.hostname in ('flickr.com', 'www.flickr.com'):
            m = re.search(r'/photos/[^/]+/(\d+)', u.path)
            if m:
                result.add('flickr-photo:' + m[1])
        if u.hostname == 'commons.wikimedia.org' and '/wiki/File:' in unquote(u.path):
            result.add('commons-title:' + title_key(unquote(u.path).split('/wiki/', 1)[1]))
    return sorted(result)


class Client:
    def __init__(self, interval=1., resume_id=None):
        self.log = HERE / 'requests.jsonl'
        self.records = [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []
        self.cache = {r['url']: r for r in self.records if r.get('status') == 200 and not r.get('api_error')}
        self.last = 0.
        self.stop = False
        self.interval = interval
        self.resume_id = resume_id

    def fetch(self, params, label, endpoint=API):
        url = endpoint + '?' + urlencode(params)
        if url in self.cache:
            r = self.cache[url]
            data = (HERE / r['response_file']).read_bytes()
            assert sha(data) == r['response_sha256']
            return json.loads(data), r
        if self.stop:
            raise RuntimeError('Official access refusal; client stopped')
        for attempt in range(2):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last)))
            r = dict(url=url, requested_at_utc=now(), label=label, attempt=attempt + 1)
            if self.resume_id:
                r['resume_id'] = self.resume_id
                r['minimum_interval_seconds'] = self.interval
            data = b''
            try:
                with urlopen(Request(url, headers={'User-Agent': UA}), timeout=45) as f:
                    data = f.read()
                    r.update(status=f.status, headers=dict(f.headers), final_url=f.url)
            except HTTPError as e:
                data = e.read()
                r.update(status=e.code, headers=dict(e.headers), error=str(e))
            except Exception as e:
                r.update(status=None, error=repr(e), headers={})
            self.last = time.monotonic()
            filename = f'raw/{len(self.records):04d}_{label}_a{attempt + 1}.response'
            (HERE / filename).write_bytes(data)
            r.update(response_file=filename, response_sha256=sha(data), bytes=len(data))
            try:
                parsed = json.loads(data)
            except (ValueError, UnicodeError):
                parsed = None
            if parsed and 'error' in parsed:
                r['api_error'] = parsed['error']
            with self.log.open('a') as f:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            self.records.append(r)
            if r['status'] == 200 and parsed is not None and 'error' not in parsed:
                self.cache[url] = r
                return parsed, r
            if r['status'] in (401, 403) or (r['status'] == 200 and parsed is None):
                self.stop = True
                raise RuntimeError('Access refusal or non-JSON verification response: ' + filename)
            if attempt == 0:
                headers = {k.lower(): v for k, v in r.get('headers', {}).items()}
                retry = headers.get('retry-after')
                wait = 5
                if r['status'] in (429, 503):
                    try:
                        requested = float(retry)
                    except (TypeError, ValueError):
                        try:
                            requested = (parsedate_to_datetime(retry) - dt.datetime.now(dt.timezone.utc)).total_seconds()
                        except (TypeError, ValueError):
                            requested = 30
                    wait = max(30, requested + 2)
                if wait > 60:
                    raise RuntimeError('Retry-After exceeds bounded task wait; defer request: ' + filename)
                time.sleep(wait)
        raise RuntimeError('Request failed after bounded retry: ' + filename)


def old_registry():
    names = ('candidates_downloaded.json', 'candidates.json', 'additional_downloaded.json', 'manifest.json')
    sources = {name: sha((OLD / name).read_bytes()) for name in names}
    rows = json.loads((OLD / names[0]).read_text())['images']
    assert len(rows) == 101
    lookup = {title_key(r['title']) for r in rows}
    for name in names[1:]:
        assert all(title_key(r['title']) in lookup for r in json.loads((OLD / name).read_text())['images'])
    raw = {}
    for path in sorted((OLD / 'metadata').glob('*.json')):
        sources[str(path.relative_to(OLD))] = sha(path.read_bytes())
        for page in json.loads(path.read_text()).get('query', {}).get('pages', {}).values():
            if page.get('imageinfo'):
                raw[page['imageinfo'][0].get('sha1')] = page
    records, authors, sourceids = [], set(), set()
    for row in rows:
        title = title_key(row['title'])
        meta = raw.get(row.get('original_sha1'), {}).get('imageinfo', [{}])[0].get('extmetadata', {})
        artist = meta.get('Artist', {}).get('value', row.get('author', ''))
        identities, resolved = author_keys(artist)
        authors.update(identities)
        keys = source_keys(' '.join(str(x.get('value', '')) for x in meta.values()))
        sourceids.update(keys)
        records.append(dict(title=row['title'], title_key=title, original_sha1=row.get('original_sha1'),
                            downloaded_sha256=row.get('downloaded_sha256'), processed_sha256=row.get('sha256'),
                            source_url=row.get('source_url'), artist_html=artist, author_keys=identities,
                            author_resolved=resolved, original_source_keys=keys))
    result = dict(source_hashes=sources, count=101, images=records,
                  author_identity_keys=sorted(authors), original_source_keys=sorted(sourceids))
    write('old_exclusion_registry.json', result)
    return result


def frame(client, old):
    by_title = {}
    frame_sources = []
    for stratum, categories in STRATA.items():
        for category in categories:
            cont = {}
            for page_index in range(2):
                params = dict(action='query', format='json', formatversion=2, list='categorymembers',
                              cmtitle=category, cmnamespace=6, cmtype='file', cmprop='ids|title|timestamp',
                              cmsort='sortkey', cmdir='asc', cmlimit=500, maxlag=5, **cont)
                data, r = client.fetch(params, stratum + '_category_' + str(page_index))
                members = data.get('query', {}).get('categorymembers', [])
                frame_sources.append(dict(stratum=stratum, category=category, page=page_index,
                                          count=len(members), response=r['response_file'],
                                          continuation=data.get('continue')))
                for member in members:
                    key = title_key(member['title'])
                    entry = by_title.setdefault(key, dict(title_key=key, title=member['title'], pageid=member['pageid'],
                                                         categories=[], strata=[], frame_responses=[]))
                    entry['categories'].append(category)
                    entry['strata'].append(stratum)
                    entry['frame_responses'].append(r['response_file'])
                cont = data.get('continue')
                if not cont:
                    break
    old_titles = {r['title_key'] for r in old['images']}
    candidates = list(by_title.values())
    selected = []
    for stratum in STRATA:
        pool = [r for r in candidates if stratum in r['strata'] and r['title_key'] not in old_titles]
        pool.sort(key=lambda r: sha((SALT + '|frame|' + stratum + '|' + r['title_key']).encode()))
        for row in pool[:320 if stratum == 'water' else 160]:
            selected.append(dict(row, stratum=stratum,
                                 frame_rank_hash=sha((SALT + '|frame|' + stratum + '|' + row['title_key']).encode())))
    # If a file belongs to multiple resource strata, retain metadata but hold for review.
    write('sampling_frame.json', dict(collected_at_utc=now(), source='Commons official categorymembers API',
                                      roots=frame_sources, unique_file_count=len(candidates),
                                      old_title_excluded=sum(r['title_key'] in old_titles for r in candidates),
                                      all_candidates=candidates, requested_metadata=selected))
    return selected


def record(page, row, request, old):
    info = page.get('imageinfo', [{}])[0]
    meta = info.get('extmetadata', {})
    val = lambda k: clean(meta.get(k, {}).get('value', ''))
    artist_html = meta.get('Artist', {}).get('value', '')
    identities, resolved = author_keys(artist_html)
    original_keys = source_keys(' '.join(str(x.get('value', '')) for x in meta.values()))
    reasons = []
    if not info:
        reasons.append('missing_imageinfo')
    if info.get('sha1') in {x['original_sha1'] for x in old['images']}:
        reasons.append('exact_old_original_sha1')
    if set(identities) & set(old['author_identity_keys']):
        reasons.append('old_author_identity_or_name')
    if set(original_keys) & set(old['original_source_keys']):
        reasons.append('old_original_source_or_derivative_link')
    if not resolved:
        reasons.append('unknown_or_unresolved_author')
    if info.get('mime') not in ('image/jpeg', 'image/png'):
        reasons.append('unsupported_mime')
    if min(info.get('width', 0), info.get('height', 0)) < 256:
        reasons.append('dimensions_below_256')
    if not info.get('timestamp') or info['timestamp'] > CUTOFF:
        reasons.append('version_outside_fixed_cutoff_or_unknown')
    lic, licurl = val('LicenseShortName'), val('LicenseUrl')
    parsed = urlparse(licurl)
    matched = re.fullmatch(r'CC (BY(?:-SA)?) (2\.0|2\.5|3\.0|4\.0)', lic)
    license_ok = parsed.hostname == 'creativecommons.org' and (
        (lic == 'CC0' and parsed.path.startswith('/publicdomain/zero/1.0')) or
        (matched is not None and parsed.path.startswith('/licenses/' + matched[1].lower() + '/' + matched[2])))
    if not license_ok:
        reasons.append('license_not_exactly_verified_allowlist')
    if len(set(row['strata'])) > 1:
        reasons.append('multiple_resource_strata')
    if any(s in val('Categories').casefold() for s in ('deletion request', 'copyright violation', 'no permission', 'disputed')):
        reasons.append('copyright_dispute_metadata')
    return dict(id='commons:' + str(row['pageid']), title=page.get('title', row['title']),
                title_key=row['title_key'], pageid=row['pageid'], stratum=row['stratum'],
                frame_rank_hash=row['frame_rank_hash'], frame_categories=row['categories'],
                source_page=info.get('descriptionurl', 'https://commons.wikimedia.org/wiki/' + row['title'].replace(' ', '_')),
                page_revision=page.get('revisions', []), original_url=info.get('url'),
                original_sha1=info.get('sha1'), original_version_timestamp=info.get('timestamp'),
                mime=info.get('mime'), width=info.get('width'), height=info.get('height'),
                uploader=info.get('user'), uploader_id=info.get('userid'),
                artist_html=artist_html, artist_text=clean(artist_html), artist_links=links(artist_html),
                author_keys=identities, author_identity_status='metadata_resolved_provisionally' if resolved else 'unresolved',
                original_source_keys=original_keys, credit_html=meta.get('Credit', {}).get('value', ''),
                description_html=meta.get('ImageDescription', {}).get('value', ''),
                license=lic, license_url=licurl, attribution_required=val('AttributionRequired'),
                license_metadata_allowlist_passed=license_ok, extmetadata=meta,
                metadata_request_url=request['url'], metadata_response=request['response_file'],
                metadata_response_sha256=request['response_sha256'], frame_responses=row['frame_responses'],
                exclusion_reasons=reasons, metadata_status='eligible_pending_manual_checks' if not reasons else 'excluded_or_unresolved',
                pixel_review='not_performed', near_duplicate_review='not_performed',
                attribution_review='not_performed', model_exposure='none')


def allocate(rows):
    # Connected components are deliberately conservative: shared author text can merge people.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        keys = row['author_keys'] + row['original_source_keys'] + ['sha1:' + str(row['original_sha1'])]
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    clusters = []
    for group in groups.values():
        if any(set(row['exclusion_reasons']) & {'old_author_identity_or_name',
                'old_original_source_or_derivative_link', 'exact_old_original_sha1'} for row in group):
            for row in group:
                reason = 'connected_component_contains_old_source_or_author'
                if reason not in row['exclusion_reasons']:
                    row['exclusion_reasons'].append(reason)
                row['metadata_status'] = 'excluded_or_unresolved'
        identities = sorted({k for row in group for k in row['author_keys'] + row['original_source_keys']})
        cluster_id = sha('|'.join(identities).encode())
        rank = sha((SALT + '|cluster|' + cluster_id).encode())
        for row in group:
            row['cluster_id'] = cluster_id; row['cluster_rank'] = rank
        usable = [r for r in group if not r['exclusion_reasons']]
        if usable:
            usable.sort(key=lambda r: (r['frame_rank_hash'], r['id']))
            clusters.append((rank, usable[0]))
    clusters.sort(key=lambda x: x[0])
    targets = {'workflow': {'apple': 4, 'banana': 4, 'orange': 4, 'water': 12},
               'confirmation': {'apple': 8, 'banana': 8, 'orange': 8, 'water': 24},
               'reserve': {'apple': 12, 'banana': 12, 'orange': 12, 'water': 36}}
    counts = {k: {s: 0 for s in STRATA} for k in targets}
    for row in rows:
        row['provisional_assignment'] = None
    used = set()
    for split in targets:
        for rank, row in clusters:
            if row['cluster_id'] in used:
                continue
            s = row['stratum']
            if counts[split][s] < targets[split][s]:
                counts[split][s] += 1; used.add(row['cluster_id'])
                row['provisional_assignment'] = split
    return dict(targets=targets, counts=counts,
                shortfalls={k: {s: targets[k][s] - counts[k][s] for s in STRATA} for k in targets},
                available_author_clusters=len(clusters),
                note='Reservations only. Pixel, perceptual duplicate, attribution and identity review remain pending; no confirmation image exposure.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-record')
    args = parser.parse_args()
    freeze = json.loads((HERE / 'plan_freeze.json').read_text())
    assert sha((HERE / 'prospective_sampling_plan.md').read_bytes()) == freeze['plan_sha256']
    resume = json.loads((HERE / args.resume_record).read_text()) if args.resume_record else None
    if resume:
        assert resume['plan_sha256'] == freeze['plan_sha256']
        assert resume['sampling_frame_sha256'] == sha((HERE / 'sampling_frame.json').read_bytes())
    client = Client(interval=resume['normal_minimum_interval_seconds'] if resume else 1.,
                    resume_id=resume['resume_id'] if resume else None)
    old = old_registry()
    rows = []
    state = dict(started_at_utc=now(), plan_sha256=freeze['plan_sha256'], source_code_sha256=sha(Path(__file__).read_bytes()),
                 images_downloaded=0, image_views=0, model_calls=0, old_candidates_excluded=101)
    if resume:
        state['resume_record'] = args.resume_record
        state['resume_record_sha256'] = sha((HERE / args.resume_record).read_bytes())
    try:
        selected = json.loads((HERE / 'sampling_frame.json').read_text())['requested_metadata'] if resume else frame(client, old)
        for offset in range(0, len(selected), 10):
            group = selected[offset:offset + 10]
            params = dict(action='query', format='json', formatversion=2, prop='imageinfo|info|revisions',
                          titles='|'.join(r['title'] for r in group),
                          iiprop='timestamp|user|userid|url|size|sha1|mime|extmetadata',
                          rvprop='ids|timestamp', iiextmetadatalanguage='en', maxlag=5)
            data, request = client.fetch(params, 'metadata_' + str(offset).zfill(4))
            pages = {title_key(p['title']): p for p in data.get('query', {}).get('pages', [])}
            for row in group:
                rows.append(record(pages.get(row['title_key'], {}), row, request, old))
            write('manifest.json', dict(status='collecting', metadata_count=len(rows), images=rows))
            print(f'METADATA {len(rows)}/{len(selected)}', flush=True)
        allocation = allocate(rows)
        state.update(status='metadata_collection_complete', metadata_count=len(rows),
                     eligible_pending_review=sum(not r['exclusion_reasons'] for r in rows), allocation=allocation)
    except Exception as exc:
        state.update(status='partial_or_failed', error=repr(exc), metadata_count=len(rows))
    state.update(finished_at_utc=now(), request_count=len(client.records),
                 successful_response_count=sum(r.get('status') == 200 and not r.get('api_error') for r in client.records))
    write('manifest.json', dict(status=state['status'], metadata_count=len(rows), images=rows))
    write('status.json', state)
    print(json.dumps({k: v for k, v in state.items() if k != 'allocation'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
