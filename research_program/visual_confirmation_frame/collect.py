"""Bounded official metadata-only inventory; no pixels or model calls."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import hashlib
import json
import unicodedata
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
OUTPUT = ROOT/'metadata_001'
CATEGORIES = ('Apples', 'Bananas', 'Oranges', 'Glasses of water')
ENDPOINT = 'https://commons.wikimedia.org/w/api.php'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s): return unicodedata.normalize('NFKC', s).replace('_', ' ').strip().casefold()
def now(): return datetime.now(timezone.utc).isoformat()
def write(p, value):
    with Path(p).open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    OUTPUT.mkdir(exist_ok=False)
    old = WORK/'redesign_v0.4/data'
    files = sorted((old/'metadata').glob('*.json')) + [old/'candidates.json', old/'manifest.json']
    old_titles = set()
    hashes = {}
    for p in files:
        if not p.exists(): continue
        hashes[str(p)] = sha(p)
        data = json.loads(p.read_text())
        for page in data.get('query', {}).get('pages', {}).values():
            if page.get('title', '').startswith('File:'): old_titles.add(norm(page['title']))
        for row in data.get('images', []):
            title = row.get('title', '')
            if title: old_titles.add(norm(title if title.startswith('File:') else 'File:'+title))
    write(OUTPUT/'invocation.json', dict(started_utc=now(), source_sha256=sha(__file__),
        plan_sha256=sha(ROOT/'plan.md'), categories=CATEGORIES, old_source_sha256=hashes,
        old_normalized_titles=sorted(old_titles), request_timeout_seconds=15,
        per_category_page_limit=3, page_size=500, pixel_downloads=0, model_calls=0))
    results=[]
    for ix, category in enumerate(CATEGORIES):
        rows=[]; continuation={}; requests=[]; complete=False; failure=None
        for page in range(3):
            params=dict(action='query', format='json', list='categorymembers',
                cmtitle='Category:'+category, cmtype='file', cmlimit=500, cmsort='sortkey', cmdir='ascending',
                **continuation)
            url=ENDPOINT+'?'+urlencode(params)
            name=f'category_{ix}_page_{page}'
            request=dict(url=url, requested_utc=now(), category=category, page=page)
            try:
                with urlopen(Request(url, headers={'User-Agent':'LanguageFormationResearch/1.0 (metadata-only local research)'}), timeout=15) as response:
                    blob=response.read(); request['http_status']=response.status
                (OUTPUT/(name+'.json')).write_bytes(blob)
                request['response_sha256']=sha(OUTPUT/(name+'.json'))
                data=json.loads(blob)
                if 'error' in data: raise RuntimeError(json.dumps(data['error']))
                members=data['query']['categorymembers']
                assert all(r['ns']==6 and r['title'].startswith('File:') for r in members)
                rows.extend(members); continuation=data.get('continue', {})
                complete=not continuation
            except Exception as error:
                failure=repr(error); request['error']=failure
                if isinstance(error, HTTPError):
                    blob=error.read(); (OUTPUT/(name+'_error.bin')).write_bytes(blob)
                    request['http_status']=error.code; request['error_body_sha256']=sha(OUTPUT/(name+'_error.bin'))
                requests.append(request); break
            requests.append(request)
            if complete: break
        assert len({r['pageid'] for r in rows}) == len(rows)
        result=dict(category=category, complete=complete, failure=failure, continuation=continuation,
            requests=requests, listed_files=len(rows), old_title_matches=sum(norm(r['title']) in old_titles for r in rows),
            files=[dict(r, old_title_match=norm(r['title']) in old_titles) for r in rows])
        write(OUTPUT/f'category_{ix}_inventory.json', result); results.append(result)
        print(json.dumps({k:result[k] for k in ('category','complete','failure','listed_files','old_title_matches')}),flush=True)
    write(OUTPUT/'result.json', dict(status='complete' if all(r['complete'] for r in results) else 'partial',
        finished_utc=now(), categories=results, pixel_downloads=0, model_calls=0,
        scope='Metadata-only direct-category frame; not a licensed/visually validated or independently sampled dataset.'))


if __name__ == '__main__': main()
