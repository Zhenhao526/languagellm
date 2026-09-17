"""Official metadata only: one request per fixed batch, no pixels or models."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse, unquote
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from html.parser import HTMLParser
from collections import defaultdict, Counter
import hashlib
import html
import json
import math
import re
import shutil
import time
import traceback
import unicodedata

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
FRAME = ROOT/'metadata_001'
OUTPUT = ROOT/'details_001'
ENDPOINT = 'https://commons.wikimedia.org/w/api.php'
BATCH_SIZE, TIMEOUT = 25, 20
FIELDS = ('Artist', 'Credit', 'Source', 'LicenseShortName', 'LicenseUrl', 'UsageTerms',
          'AttributionRequired', 'Copyrighted', 'Restrictions', 'ImageDescription',
          'DateTimeOriginal', 'ObjectName', 'Categories')
UNKNOWN = {'', 'unknown', 'unknown author', 'anonymous', 'anonymous author', 'not provided', 'n/a'}
RASTER = {'image/jpeg', 'image/png', 'image/tiff', 'image/webp'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def norm(s): return unicodedata.normalize('NFKC', str(s)).replace('_', ' ').strip().casefold()
def read(path): return json.loads(Path(path).read_text())
def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def require(value, message):
    if not value: raise AssertionError(message)


class ExtractHTML(HTMLParser):
    def __init__(self, value):
        super().__init__(convert_charrefs=True); self.text=[]; self.links=[]; self.feed(str(value))
    def handle_data(self, data): self.text.append(data)
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for name, val in attrs:
                if name == 'href' and val: self.links.append(val)


def plain(value): return ' '.join(' '.join(ExtractHTML(value).text).split())
def extract(info, key): return str(info.get('extmetadata', {}).get(key, {}).get('value', ''))


def author_keys(info):
    raw = extract(info, 'Artist'); parsed = ExtractHTML(raw)
    text = norm(plain(raw)); keys=[]
    if text not in UNKNOWN: keys.append('artist_text:'+text)
    for url in parsed.links:
        url = 'https:'+url if url.startswith('//') else url
        parsed_url=urlparse(url); host=parsed_url.netloc.casefold(); path=unquote(parsed_url.path)
        if host in ('commons.wikimedia.org', 'en.wikipedia.org') and re.match(r'/wiki/(User|Creator):', path, re.I):
            keys.append('creator_url:'+host+norm(path))
        if host in ('www.flickr.com', 'flickr.com'):
            parts=path.strip('/').split('/')
            if len(parts)>=2 and parts[0] in ('people', 'photos'):
                keys.append('creator_flickr:'+norm(parts[1]))
    return sorted(set(keys))


def source_keys(info):
    keys=[]
    for key in ('Credit', 'Source'):
        for url in ExtractHTML(extract(info, key)).links:
            url='https:'+url if url.startswith('//') else url
            parsed=urlparse(url); host=parsed.netloc.casefold(); path=unquote(parsed.path)
            if host in ('www.flickr.com', 'flickr.com'):
                parts=path.strip('/').split('/')
                if len(parts)>=3 and parts[0]=='photos' and parts[2].isdigit(): keys.append('source_flickr_photo:'+parts[2])
            if host=='commons.wikimedia.org' and path.startswith('/wiki/File:'):
                keys.append('source_commons_file:'+norm(path[6:]))
    return sorted(set(keys))


def allowed_license(info):
    short=plain(extract(info, 'LicenseShortName')).strip()
    url=html.unescape(extract(info, 'LicenseUrl')).strip()
    known = bool(re.fullmatch(r'CC BY(?:-SA)? (?:1\.0|2\.0|2\.5|3\.0|4\.0)', short, re.I)) or norm(short) in {'cc0', 'cc0 1.0', 'public domain'}
    known_url=bool(re.match(r'https?://creativecommons\.org/(?:licenses/(?:by|by-sa)/(?:1\.0|2\.0|2\.5|3\.0|4\.0)(?:/|$)|publicdomain/(?:zero|mark)/1\.0(?:/|$))', url, re.I))
    return known or known_url


def original_sha(value):
    value=str(value or '').strip().casefold()
    if re.fullmatch('[0-9a-f]{40}', value): return value
    if re.fullmatch('[0-9a-z]{31}', value): return f'{int(value,36):040x}'
    return None


def old_index(invocation):
    rows=[]; hashes={}
    for text, expected in invocation['old_source_sha256'].items():
        path=Path(text); require(sha(path)==expected, 'Old source changed: '+text); hashes[text]=expected
        data=read(path)
        for pagekey, page in data.get('query', {}).get('pages', {}).items():
            title=page.get('title', '')
            if not title.startswith('File:'): continue
            infos=page.get('imageinfo', [])
            rows.append(dict(source=text, json_pointer=f'/query/pages/{pagekey}', title=title,
                source_kind='api_metadata', sha1s=sorted({s for i in infos if (s:=original_sha(i.get('sha1')))}),
                author_keys=sorted({k for i in infos for k in author_keys(i)}),
                source_keys=sorted({k for i in infos for k in source_keys(i)})))
        for ix, row in enumerate(data.get('images', [])):
            title=row.get('title', '')
            if not title: continue
            if not title.startswith('File:'): title='File:'+title
            s=original_sha(row.get('original_sha1'))
            rows.append(dict(source=text, json_pointer=f'/images/{ix}', title=title, source_kind='candidate_or_manifest',
                             sha1s=[s] if s else [], author_keys=[], source_keys=[]))
    raw=sorted({r['title'] for r in rows}); grouped=defaultdict(set)
    for title in raw: grouped[norm(title)].add(title)
    normalized=sorted(grouped)
    require(normalized == invocation['old_normalized_titles'], 'Old normalized title extraction mismatch')
    api_rows=[r for r in rows if r['source_kind']=='api_metadata']
    return dict(source_files_sha256=hashes, records=rows, exact_titles=raw, normalized_titles=normalized,
        original_file_sha1s=sorted({s for r in rows for s in r['sha1s']}),
        author_keys=sorted({s for r in rows for s in r['author_keys']}), source_keys=sorted({s for r in rows for s in r['source_keys']}),
        counts=dict(api_page_records=len(api_rows), api_exact_titles=len({r['title'] for r in api_rows}),
                    exact_titles=len(raw), normalized_titles=len(normalized)),
        collisions=[dict(normalized=k, exact_titles=sorted(v), records=[r for r in rows if r['title'] in v])
                    for k,v in sorted(grouped.items()) if len(v)>1],
        omitted_normalized_titles=[])


def build_frame():
    result=read(FRAME/'result.json'); invocation=read(FRAME/'invocation.json')
    require(result['status']=='complete' and len(result['categories'])==4, 'Complete four-category frame')
    hashes={str(p):sha(p) for p in FRAME.iterdir() if p.is_file()}
    files={}
    for ix, category in enumerate(result['categories']):
        require(category==read(FRAME/f'category_{ix}_inventory.json') and category['complete'] and
                not category['failure'] and not category['continuation'], 'Incomplete category')
        for request in category['requests']:
            path=FRAME/f'category_{ix}_page_{request["page"]}.json'
            require(sha(path)==request['response_sha256'], 'Category raw response hash')
        for row in category['files']:
            pid=row['pageid']
            if pid not in files: files[pid]=dict(pageid=pid, title=row['title'], categories=[])
            require(files[pid]['title']==row['title'], 'Page ID title disagreement')
            files[pid]['categories'].append(category['category'])
    rows=[files[k] for k in sorted(files)]
    require(len(rows)==1876 and sum(len(r['categories']) for r in rows)==1895, 'Frozen list size changed')
    return rows, hashes, old_index(invocation)


def grouping(rows):
    parent={r['pageid']:r['pageid'] for r in rows}; owners={}; edges=[]
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for row in rows:
        keys=(['sha1:'+row['original_file_sha1']] if row['original_file_sha1'] else [])+row['author_keys']+row['source_keys']
        for key in keys:
            if key in owners:
                a,b=find(row['pageid']),find(owners[key]); parent[a]=b
                edges.append(dict(pageids=[row['pageid'],owners[key]], key=key))
            else: owners[key]=row['pageid']
    groups=defaultdict(list)
    for row in rows: groups[find(row['pageid'])].append(row['pageid'])
    return dict(count=len(groups), groups=[sorted(v) for _,v in sorted(groups.items())], edges=edges)


def summarize(records, old):
    answers={}
    for category in ('Apples', 'Bananas', 'Oranges', 'food_union', 'Glasses of water'):
        subset=[r for r in records if (any(c in ('Apples','Bananas','Oranges') for c in r['categories']) if category=='food_union' else category in r['categories'])]
        resolved=[r for r in subset if r['status']=='resolved']
        fresh=[r for r in resolved if not r['old_title_normalized_match'] and not r['old_sha1_match']]
        raster=[r for r in fresh if r['mime'] in RASTER]
        licensed=[r for r in raster if r['license_pattern_supported']]
        authored=[r for r in licensed if r['author_keys']]
        newauthors=[r for r in authored if not r['old_author_key_match'] and not r['old_source_key_match']]
        group=grouping(authored); group_new=grouping(newauthors)
        answers[category]=dict(listed=len(subset), resolved=len(resolved), unresolved=len(subset)-len(resolved),
            old_exact_title_matches=sum(r['old_title_exact_match'] for r in resolved),
            old_normalized_title_matches=sum(r['old_title_normalized_match'] for r in resolved),
            old_sha1_matches=sum(r['old_sha1_match'] for r in resolved),
            old_sha1_without_old_title=sum(r['old_sha1_match'] and not r['old_title_normalized_match'] for r in resolved),
            fresh_files=len(fresh), fresh_distinct_sha1=len({r['original_file_sha1'] for r in fresh if r['original_file_sha1']}),
            fresh_static_raster_mime=len(raster), metadata_license_pattern_supported=len(licensed),
            supported_with_author_field=len(authored), supported_without_author_field=len(licensed)-len(authored),
            distinct_artist_text_fields=len({r['artist_normalized_text'] for r in authored if r['artist_normalized_text'] not in UNKNOWN}),
            metadata_components=group, excluding_old_author_or_source_components=group_new,
            exclusions_old_author_or_source=len(authored)-len(newauthors),
            license_labels=dict(Counter(r['license_short_name'] for r in resolved)), mime_counts=dict(Counter(r['mime'] for r in resolved)),
            target_per_resource=84 if category in ('food_union','Glasses of water') else None,
            confirmed_independent_photographers=None, visually_accepted_images=0)
    return answers


def main():
    rows, source_hashes, old=build_frame()
    OUTPUT.mkdir(exist_ok=False); (OUTPUT/'responses').mkdir()
    for path in (Path(__file__), ROOT/'metadata_detail_plan.md'): shutil.copyfile(path, OUTPUT/path.name)
    source_hashes.update(old['source_files_sha256'])
    source_hashes.update({str(p):sha(p) for p in (Path(__file__), ROOT/'metadata_detail_plan.md')})
    requests=[]; frame=[]
    for ix in range(0,len(rows),BATCH_SIZE):
        part=rows[ix:ix+BATCH_SIZE]
        params=dict(action='query', format='json', formatversion=2, prop='imageinfo',
            pageids='|'.join(str(r['pageid']) for r in part), iilimit=1,
            iiprop='url|size|mime|sha1|extmetadata|user|timestamp|canonicaltitle',
            iiextmetadatalanguage='en', iiextmetadatafilter='|'.join(FIELDS))
        frame.append(dict(batch=ix//BATCH_SIZE, pageids=[r['pageid'] for r in part], url=ENDPOINT+'?'+urlencode(params)))
    write(OUTPUT/'old_index_evidence.json',old); write(OUTPUT/'request_frame.json',dict(files=rows,batches=frame))
    invocation=dict(started_utc=now(), source_files_sha256=source_hashes, frame_sha256=sha(OUTPUT/'request_frame.json'),
        old_index_sha256=sha(OUTPUT/'old_index_evidence.json'), unique_files=len(rows), requests_planned=len(frame),
        batch_size=BATCH_SIZE, timeout_seconds=TIMEOUT, automatic_retry=False, pixel_downloads=0, model_calls=0)
    write(OUTPUT/'invocation.json',invocation)
    record_map={}; context={}
    try:
        for job in frame:
            context=job; start=time.monotonic(); receipt=dict(job, requested_utc=now())
            response_path=OUTPUT/'responses'/f'batch_{job["batch"]:03d}.bin'
            try:
                request=Request(job['url'],headers={'User-Agent':'LanguageFormationResearch/1.0 (metadata-only local research)','Accept':'application/json'})
                with urlopen(request,timeout=TIMEOUT) as response:
                    require(urlparse(response.geturl()).netloc=='commons.wikimedia.org', 'Unexpected endpoint redirect')
                    receipt.update(http_status=response.status, final_url=response.geturl(), response_headers=dict(response.headers))
                    blob=response.read()
                with response_path.open('xb') as f: f.write(blob)
                receipt.update(response_file=str(response_path),response_sha256=sha(response_path),response_bytes=len(blob))
                data=json.loads(blob)
                require('error' not in data, 'API error: '+json.dumps(data.get('error')))
                pages=data['query']['pages']; ids={p.get('pageid') for p in pages}
                require(len(pages)==len(ids) and ids.issubset(set(job['pageids'])), 'Returned page identity')
                receipt.update(status='received',warnings=data.get('warnings'),returned_pages=len(pages),
                               continuation_not_followed_latest_revision_only=data.get('continue'))
                for p in pages: record_map[p.get('pageid')]=dict(page=p,batch=job['batch'])
            except Exception as error:
                receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
                if isinstance(error,HTTPError):
                    receipt.update(http_status=error.code,response_headers=dict(error.headers))
                    blob=error.read()
                    if not response_path.exists():
                        with response_path.open('xb') as f:f.write(blob)
                    receipt.update(response_file=str(response_path),response_sha256=sha(response_path),response_bytes=len(blob))
            receipt.update(completed_utc=now(),elapsed_seconds=time.monotonic()-start)
            write(OUTPUT/'responses'/f'batch_{job["batch"]:03d}_receipt.json',receipt); requests.append(receipt)
            print(json.dumps({k:receipt[k] for k in ('batch','status','elapsed_seconds')},ensure_ascii=False),flush=True)
        output=[]
        for row in rows:
            source=record_map.get(row['pageid']); page=source['page'] if source else {}
            infos=page.get('imageinfo',[]); info=infos[0] if len(infos)==1 else {}
            current_title=page.get('title',row['title']); digest=original_sha(info.get('sha1'))
            authors=author_keys(info); sources=source_keys(info)
            resolved=bool(info and digest and info.get('url') and info.get('mime'))
            output.append(dict(row,status='resolved' if resolved else 'unresolved',current_title=current_title,
                api_batch=source['batch'] if source else None,original_file_sha1=digest,
                old_title_exact_match=any(t in old['exact_titles'] for t in (row['title'],current_title)),
                old_title_normalized_match=any(norm(t) in old['normalized_titles'] for t in (row['title'],current_title)),
                old_sha1_match=bool(digest and digest in old['original_file_sha1s']),
                old_author_key_match=bool(set(authors)&set(old['author_keys'])),old_source_key_match=bool(set(sources)&set(old['source_keys'])),
                author_keys=authors,source_keys=sources,artist_raw=extract(info,'Artist'),artist_normalized_text=norm(plain(extract(info,'Artist'))),
                upload_user=info.get('user'),license_short_name=plain(extract(info,'LicenseShortName')),
                license_url=extract(info,'LicenseUrl'),license_pattern_supported=allowed_license(info),
                mime=info.get('mime'),width=info.get('width'),height=info.get('height'),size=info.get('size'),
                original_url=info.get('url'),description_url=info.get('descriptionurl'),timestamp=info.get('timestamp'),
                extmetadata=info.get('extmetadata',{}),page_flags={k:v for k,v in page.items() if k!='imageinfo'}))
        summaries=summarize(output,old)
        write(OUTPUT/'records.json',output); write(OUTPUT/'feasibility.json',summaries)
        require(all(sha(p)==h for p,h in source_hashes.items()),'Source changed during run')
        result=dict(status='complete' if all(r['status']=='received' for r in requests) and all(r['status']=='resolved' for r in output) else 'partial',
            completed_utc=now(),requests=len(requests),requests_successful=sum(r['status']=='received' for r in requests),
            unique_files=len(rows),resolved=sum(r['status']=='resolved' for r in output),
            source_files_sha256=source_hashes,records_sha256=sha(OUTPUT/'records.json'),feasibility_sha256=sha(OUTPUT/'feasibility.json'),
            pixel_downloads=0,model_calls=0,automatic_retry=False,summary={k:{x:v[x] for x in ('listed','resolved','fresh_files','metadata_license_pattern_supported','supported_with_author_field')} for k,v in summaries.items()})
        write(OUTPUT/'result.json',result); print(json.dumps(result,ensure_ascii=False,indent=2))
    except BaseException as error:
        write(OUTPUT/'failure.json',dict(status='failed',context=context,error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__': main()
