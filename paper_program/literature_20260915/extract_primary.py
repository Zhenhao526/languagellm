from pathlib import Path
from pypdf import PdfReader
import urllib.request,hashlib,json
B=Path(__file__).resolve().parent;P=B/'primary';P.mkdir(exist_ok=True)
items=[('lee2024',Path('文献调研_2026-09-14/论文/02_神经多智能体/2024_Lee_One-to-Many Communication and Compositionality in Emergent Communication.pdf').resolve(),'https://aclanthology.org/2024.emnlp-main.1157.pdf'),('ctd2025',Path('research_program/literature_audit_2026-09-15/papers/2025_Carmeli_CtD_Composition through Decomposition in Emergent Communication_ICLR2025.pdf').resolve(),'https://proceedings.iclr.cc/paper_files/paper/2025/file/fb9d01fb202e360b2f78510c47e0fa3e-Paper-Conference.pdf'),('celebi2025',P/'2025_Elberg_CELEBI_NeurIPS2025.pdf','https://proceedings.neurips.cc/paper_files/paper/2025/file/3310034c97fab48fdbcba18f90fd5364-Paper-Conference.pdf'),('cope2024',P/'2024_Cope_Learning_Translations_IJCAI2024.pdf','https://www.ijcai.org/proceedings/2024/0005.pdf')]
records=[]
for key,path,url in items:
    new=not path.exists()
    if new: path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
    assert path.read_bytes().startswith(b'%PDF')
    reader=PdfReader(path);texts=[p.extract_text() or '' for p in reader.pages]
    (P/f'{key}.txt').write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n{t}' for i,t in enumerate(texts)))
    records.append(dict(key=key,pdf=str(path),source=url,new_download=new,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pages=len(texts),chars_per_page=list(map(len,texts))))
    print(key,len(texts),flush=True)
(P/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
