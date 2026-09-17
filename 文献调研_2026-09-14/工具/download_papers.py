"""Download openly accessible research PDFs from curated, verified source URLs.
Usage: bundled-python download_papers.py records.json output_dir
Preserves failures as metadata; never stores an HTML error as a PDF.
"""
import sys, json, re, hashlib, time, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from pypdf import PdfReader

def safe(s):
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', '', str(s)).strip().rstrip('.')

def download(record, outdir):
    r = dict(record)
    name = safe(f"{r['year']}_{r['first_author']}_{r['title']}")[:210] + '.pdf'
    target = outdir / name
    r['local_filename'] = name
    attempts = []
    urls = r.get('pdf_urls', [])
    if isinstance(urls, str): urls = [urls]
    for url in urls:
        tmp = target.with_suffix('.part')
        try:
            if target.exists():
                data = target.read_bytes()
            else:
                result = subprocess.run(['curl','--fail','--location','--silent','--show-error','--connect-timeout','15','--max-time','60','--output',str(tmp),url], capture_output=True, text=True)
                if result.returncode: raise RuntimeError(result.stderr.strip())
                data = tmp.read_bytes()
                if not data.lstrip().startswith(b'%PDF-'): raise ValueError('Response is not a PDF')
            reader = PdfReader(str(target if target.exists() else tmp))
            if len(reader.pages) == 0: raise ValueError('PDF has no pages')
            preview = '\n'.join((p.extract_text() or '') for p in reader.pages[:2])
            if not target.exists(): tmp.replace(target)
            r.update(download_status='downloaded', downloaded_url=url, local_path=str(target.resolve()), pages=len(reader.pages), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), first_pages_preview=preview[:1200])
            break
        except Exception as e:
            attempts.append({'url':url,'error':str(e)[:250]})
            if tmp.exists(): tmp.unlink()
    else:
        r.update(download_status='unavailable', local_path=None)
    r['download_attempts'] = attempts
    print(r.get('id',name), r['download_status'], flush=True)
    return r

if __name__ == '__main__':
    src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    records = json.loads(src.read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda r: download(r, outdir), records))
    src.with_name(src.stem+'_downloads.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
