"""At most one original user page request and one official redirect API request."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, time, urllib.request, urllib.error

OUT = Path(__file__).resolve().parent
URLS = [
    ('user_page', 'https://commons.wikimedia.org/wiki/User:GTSPACE'),
    ('redirects_api', 'https://commons.wikimedia.org/w/api.php?action=query&format=json&formatversion=2&titles=User%3AGTSPACE&redirects=1&prop=info'),
]

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    assert not (OUT/'requests.json').exists(), 'Bounded lookup must not be repeated'
    rows = []
    for index, (kind, url) in enumerate(URLS):
        if index:
            remaining = rows[-1]['finished_monotonic']+10-time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        record = {'kind': kind, 'url': url, 'started_utc': now(), 'started_monotonic': time.monotonic(),
                  'http_status': None, 'headers': {}, 'error': None}
        body = b''
        response = None
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'LanguageEmergenceResearch/0.16 (bounded source identity verification; no media fetch)'})
            try:
                response = urllib.request.build_opener(NoRedirect).open(request, timeout=60)
            except urllib.error.HTTPError as error:
                response = error
            record['http_status'] = response.code
            record['headers'] = dict(response.headers.items())
            body = response.read(8*1024*1024)
        except Exception as error:
            record['error'] = type(error).__name__+': '+str(error)
        finally:
            if response is not None:
                response.close()
        path = OUT/(kind+'.response')
        path.write_bytes(body)
        record.update(finished_utc=now(), finished_monotonic=time.monotonic(), raw_path=str(path),
            raw_bytes=len(body), raw_sha256=hashlib.sha256(body).hexdigest())
        rows.append(record)
        (OUT/'requests.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'kind': kind, 'http_status': record['http_status'], 'bytes': len(body), 'error': record['error']}), flush=True)
        if record['http_status'] in (401,403,429,503):
            break
    (OUT/'execution_receipt.json').write_text(json.dumps({'finished_utc':now(),'requests':len(rows),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'image_requests':0,'model_calls':0,'no_automatic_retry':True,'http_redirects_not_automatically_followed':True},indent=2)+'\n')

if __name__ == '__main__':
    main()
