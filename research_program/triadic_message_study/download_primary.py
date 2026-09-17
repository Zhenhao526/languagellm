"""One bounded download attempt per official paper URL; retain raw responses."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import time
import urllib.request
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PAPERS = [
    ('2019_Lowe_On_the_Pitfalls_of_Measuring_Emergent_Communication_AAMAS',
     'https://ifaamas.org/Proceedings/aamas2019/pdfs/p693.pdf', 9, ['On the Pitfalls', 'Ryan Lowe', 'Joelle Pineau']),
    ('2019_Eccles_Biases_for_Emergent_Communication_NeurIPS',
     'https://papers.neurips.cc/paper/9470-biases-for-emergent-communication-in-multi-agent-reinforcement-learning.pdf',
     11, ['Biases for Emergent Communication', 'Tom Eccles', 'Thore Graepel']),
    ('2019_Jaques_Social_Influence_as_Intrinsic_Motivation_ICML',
     'https://proceedings.mlr.press/v97/jaques19a/jaques19a.pdf', 10,
     ['Social Influence as Intrinsic Motivation', 'Natasha Jaques', 'Angeliki Lazaridou']),
]


def compact(text):
    return re.sub(r'[^a-z0-9]', '', text.casefold())


def main():
    out = HERE / 'literature/download_001'
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for stem, url, pages, identity in PAPERS:
        started = time.perf_counter()
        row = {'filename': stem+'.pdf', 'url': url, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'attempts': 1, 'expected_pages': pages, 'identity_checks': identity}
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'LanguageEmergenceResearch/0.1 (academic literature)'})
            with urllib.request.urlopen(req, timeout=30) as response:
                body = response.read(16*1024*1024+1)
                row.update(http_status=response.status, final_url=response.url,
                    headers={k:v for k,v in response.headers.items() if k.casefold() in
                        ('content-type','content-length','etag','last-modified','retry-after')})
            raw = out/(stem+'.response.bin')
            raw.write_bytes(body)
            row.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), raw_response=raw.name)
            if len(body)>16*1024*1024 or not body.startswith(b'%PDF'):
                raise ValueError('Response not a PDF within 16MiB limit')
            reader = PdfReader(raw)
            first = reader.pages[0].extract_text() or ''
            (out/(stem+'.first_page.txt')).write_text(first)
            row['pages'] = len(reader.pages)
            if len(reader.pages)!=pages or not all(compact(x) in compact(first) for x in identity):
                raise ValueError('Page count or whitespace-normalized first-page identity check failed')
            pdf = out/(stem+'.pdf'); pdf.write_bytes(body)
            text = '\n\n'.join(f'=== PDF PAGE {i+1} ===\n{page.extract_text()}' for i,page in enumerate(reader.pages))
            (out/(stem+'.pages.txt')).write_text(text)
            row.update(status='downloaded_and_identity_checked', text_sha256=hashlib.sha256(text.encode()).hexdigest())
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error))
        row['elapsed_seconds'] = time.perf_counter()-started
        rows.append(row)
        with (out/'requests.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps({k:row.get(k) for k in ('filename','status','pages','bytes','error')}, ensure_ascii=False), flush=True)
    receipt = {'status':'completed_attempts','rows':rows,'pdf_requests':len(PAPERS),'automatic_retries':0,
               'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'prior_browser_fetch_note':'AAMAS web parser timed out before this independent local download attempt.'}
    (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
