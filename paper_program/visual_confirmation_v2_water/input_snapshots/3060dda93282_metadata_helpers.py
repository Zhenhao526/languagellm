"""Pure metadata operations. No requests, image decoding or model imports."""
from pathlib import Path
from urllib.parse import urlparse, unquote, urljoin, parse_qs
import html
import re
import json
import hashlib
import unicodedata

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())

def sha(data):
    return hashlib.sha256(data).hexdigest()

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
    bad = not text or text in ('[1]', 'unknown', 'anonymous') or bool(re.search(r'\{\{|unknown author|author unknown|not provided|various authors|multiple authors', text))
    if bad:
        return [], False
    keys = {'text:' + text}
    for link in links(value):
        u = urlparse(link)
        path = unquote(u.path)
        if u.hostname in ('commons.wikimedia.org', 'en.wikipedia.org'):
            title = path.split('/wiki/', 1)[-1] if '/wiki/' in path else parse_qs(u.query).get('title', [''])[0]
            m = re.fullmatch(r'(User|Creator):([^/#?]+)', title, re.I)
            if m:
                keys.update(['commons-' + m[1].lower() + ':' + norm(m[2]), 'text:' + norm(m[2])])
        if u.hostname in ('flickr.com', 'www.flickr.com'):
            m = re.match(r'/(?:photos|people)/([^/]+)', path)
            if m:
                keys.add('flickr-author:' + m[1].casefold())
    composite = bool(re.search(r'\s(?:and|&)\s|;', text))
    return sorted(keys), not composite

def source_keys(value):
    keys = set()
    for link in links(value):
        u = urlparse(link)
        if u.hostname in ('flickr.com', 'www.flickr.com'):
            m = re.search(r'/photos/[^/]+/(\d+)', u.path)
            if m:
                keys.add('flickr-photo:' + m[1])
        if u.hostname == 'commons.wikimedia.org' and '/wiki/File:' in unquote(u.path):
            keys.add('commons-title:' + title_key(unquote(u.path).split('/wiki/', 1)[1]))
    return sorted(keys)

def all_keys(row):
    result = set(row.get('author_keys', [])) | set(row.get('original_source_keys', []))
    a, _ = author_keys(row.get('artist_html', ''))
    result.update(a)
    if row.get('original_sha1'):
        result.add('sha1:' + row['original_sha1'])
    if row.get('title'):
        result.add('commons-title:' + title_key(row['title']))
    return result

def rank(kind, value):
    return sha((CFG['salt'] + '|' + kind + '|' + str(value)).encode())

def license_ok(name, url):
    parsed = urlparse(url)
    matched = re.fullmatch(r'CC (BY(?:-SA)?) (2\.0|2\.5|3\.0|4\.0)', name)
    return parsed.hostname == 'creativecommons.org' and (
        (name == 'CC0' and parsed.path.startswith('/publicdomain/zero/1.0')) or
        (matched is not None and parsed.path.startswith('/licenses/' + matched[1].lower() + '/' + matched[2])))

def components(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for k in all_keys(row):
            if k in seen:
                parent[find(i)] = find(seen[k])
            else:
                seen[k] = i
    grouped = {}
    for i, row in enumerate(rows):
        grouped.setdefault(find(i), []).append(row)
    return list(grouped.values())

def parse_page(page, selected, request, exclusion):
    info = page.get('imageinfo', [{}])[0]
    meta = info.get('extmetadata', {})
    val = lambda k: clean(meta.get(k, {}).get('value', ''))
    artist = meta.get('Artist', {}).get('value', '')
    keys, resolved = author_keys(artist)
    description = meta.get('ImageDescription', {}).get('value', '')
    text = clean(page.get('title', selected['title']) + ' ' + description)
    phrases = [m.group() for m in re.finditer(CFG['scene_phrase_pattern'], text, re.I)]
    category_evidence = [s['category'] for s in selected['sources'] if s['kind'] == 'category']
    negative = sorted(set(m.group().casefold() for m in re.finditer(CFG['negative_pattern'], text, re.I)))
    row = dict(id='commons:' + str(selected['pageid']), pageid=selected['pageid'],
        title=page.get('title', selected['title']), title_key=title_key(selected['title']),
        frame_rank_hash=selected['frame_rank_hash'], frame_sources=selected['sources'],
        source_page=info.get('descriptionurl'), page_revision=page.get('revisions', []),
        original_url=info.get('url'), original_sha1=info.get('sha1'),
        original_version_timestamp=info.get('timestamp'), mime=info.get('mime'),
        width=info.get('width', 0), height=info.get('height', 0),
        artist_html=artist, artist_text=clean(artist), author_keys=keys,
        author_metadata_parseable=resolved,
        uploader_not_inferred_as_author=info.get('user'),
        original_source_keys=source_keys(' '.join(str(v.get('value', '')) for v in meta.values())),
        credit_html=meta.get('Credit', {}).get('value', ''), description_html=description,
        license=val('LicenseShortName'), license_url=val('LicenseUrl'),
        attribution_required=val('AttributionRequired'), extmetadata=meta,
        scene_phrase_matches=phrases, scene_category_sources=category_evidence,
        scene_negative_terms=negative,
        metadata_response=request['response_file'], metadata_response_sha256=request['response_sha256'],
        metadata_request_url=request['url'], pixel_review='not_performed',
        source_identity_manual_review='not_performed', model_exposure='none')
    reasons = []
    if not info: reasons.append('missing_imageinfo')
    if not row['original_sha1']: reasons.append('missing_original_sha1')
    if not resolved: reasons.append('unknown_or_composite_author_pending')
    if row['mime'] not in CFG['allowed_mime']: reasons.append('unsupported_mime')
    if min(row['width'], row['height']) < CFG['minimum_short_edge']: reasons.append('short_edge_below256')
    row['license_metadata_supported'] = license_ok(row['license'], row['license_url'])
    if not row['license_metadata_supported']: reasons.append('license_not_exact_allowlist')
    if any(s in val('Categories').casefold() for s in ['deletion request', 'copyright violation', 'no permission', 'disputed']):
        reasons.append('copyright_dispute_metadata')
    row['scene_text_or_category_supported'] = bool(phrases or category_evidence)
    if not row['scene_text_or_category_supported']: reasons.append('scene_not_supported_by_fixed_text_rules')
    if negative: reasons.append('explicit_scene_exclusion_terms')
    blocked = sorted(all_keys(row) & set(exclusion['blocked_keys']))
    row['blocked_identity_or_source_keys'] = blocked
    if blocked: reasons.append('old_or_reserved_identity_or_source')
    if row['pageid'] in exclusion['blocked_pageids'] or row['title_key'] in exclusion['blocked_titles']:
        reasons.append('old_or_reserved_exact_file')
    row['exclusion_reasons'] = reasons
    row['metadata_provisionally_eligible'] = not reasons
    return row
