"""Fixed two missing primary PDFs; no broad search, no fallback paper substitution."""
from pathlib import Path
import urllib.request, json, hashlib, datetime
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
new=[('2019_Graesser_Emergent_Linguistic_Phenomena_EMNLP.pdf','https://aclanthology.org/D19-1384.pdf','EMNLP-IJCNLP 2019 proceedings','Emergent Linguistic Phenomena','Graesser'),('2021_Patel_Interpretation_of_Emergent_Communication_ICCV.pdf','https://openaccess.thecvf.com/content/ICCV2021/papers/Patel_Interpretation_of_Emergent_Communication_in_Heterogeneous_Collaborative_Embodied_Agents_ICCV_2021_paper.pdf','ICCV 2021 proceedings','Interpretation of Emergent Communication','Patel')]
old=[('2018_Mordatch','文献调研_2026-09-14/论文/02_神经多智能体/2018_Mordatch_Emergence of Grounded Compositional Language in Multi-Agent Populations.pdf','https://ojs.aaai.org/index.php/AAAI/article/view/11492','AAAI 2018'),('2024_Lee','文献调研_2026-09-14/论文/02_神经多智能体/2024_Lee_One-to-Many Communication and Compositionality in Emergent Communication.pdf','https://aclanthology.org/2024.emnlp-main.1157/','EMNLP 2024'),('2025_Piriyajitakonkij','paper_program/literature_20260915/primary/2025_Piriyajitakonkij_From_Grunts_to_Lexicons_arXiv2505.12872v2.pdf','https://arxiv.org/pdf/2505.12872v2','arXiv v2'),('2022_Rita','文献调研_2026-09-14/论文/02_神经多智能体/2022_Rita_On the role of population heterogeneity in emergent communication.pdf','https://arxiv.org/abs/2204.12982','arXiv v1')]
(ROOT/'papers').mkdir(exist_ok=True);(ROOT/'text').mkdir(exist_ok=True)
rows=[]
def inspect(path,key,url,version,status):
 b=path.read_bytes(); assert b.startswith(b'%PDF-')
 reader=PdfReader(path);pages=[p.extract_text() or '' for p in reader.pages]; txt='\n'.join(f'\n=== PDF PAGE {i+1} ===\n{x}' for i,x in enumerate(pages)); assert len(txt)>2000
 (ROOT/'text'/f'{key}.pages.txt').write_text(txt)
 return dict(key=key,status=status,path=str(path.relative_to(REPO)),official_url=url,version=version,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),pages=len(pages),extracted_chars=len(txt),pdf_header=True,first_page=pages[0],text_path=str((ROOT/'text'/f'{key}.pages.txt').relative_to(REPO)))
for name,url,version,title,author in new:
 try:
  path=ROOT/'papers'/name
  if path.exists(): raise FileExistsError(path)
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research literature retrieval contact-independent local archive'}),timeout=45) as resp: data=resp.read()
  if not data.startswith(b'%PDF-'): raise ValueError('Not PDF header')
  path.write_bytes(data); row=inspect(path,path.stem,url,version,'downloaded')
  row['title_author_validated']=title.lower() in row['first_page'].lower() and author.lower() in row['first_page'].lower(); assert row['title_author_validated']
  rows.append(row)
 except Exception as e: rows.append(dict(path=name,official_url=url,version=version,status='failed',error=repr(e)))
for key,path,url,version in old: rows.append(inspect(REPO/path,key,url,version,'reused_existing_no_download'))
out={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixed_new_downloads':len(new),'broad_collection_performed':False,'radar_packet_attempt':{'id':'1901.08706','outcome':'failed_without_network_retrieval','error':'Paper not found in local database; run collection first','resolution':'No broad collection per task; official proceedings PDF directly retrieved'},'entries':rows}
(ROOT/'download_index.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print([(r.get('key',r['path']),r['status']) for r in rows])
