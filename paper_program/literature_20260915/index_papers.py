from pathlib import Path
from pypdf import PdfReader
import json,hashlib,os
B=Path(__file__).resolve().parent;P=B/'primary'
rename={'2407.17960':'2024_Kouwenhoven_Representational_Alignment_CMCL_arXiv2407.17960v1.pdf','2601.10169':'2026upload_Carmeli_CtD_ICLR2025_arXiv2601.10169v1.pdf'}
for aid,name in rename.items():
    old=B/'radar_data/pdfs'/f'{aid}.pdf';dest=P/name
    if not dest.exists(): os.link(old,dest)
records=[]
for path in sorted(P.glob('*.pdf')):
    try:
        assert path.read_bytes().startswith(b'%PDF')
        reader=PdfReader(path);texts=[p.extract_text() or '' for p in reader.pages]
        out=P/(path.stem+'.pages.txt')
        if not out.exists(): out.write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n{t}' for i,t in enumerate(texts)))
        records.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,pages=len(texts),page_chars=list(map(len,texts)),valid_pdf=True,text=str(out)))
    except Exception as e: records.append(dict(file=str(path),valid_pdf=False,error=repr(e)))
(B/'download_verification.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print([(Path(r['file']).name,r.get('pages'),r['valid_pdf']) for r in records])
