"""Transparent fallback packets for papers absent from native collector database."""
from pathlib import Path
import json,hashlib,urllib.request,subprocess,time,unicodedata,re
from datetime import datetime,timezone
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
RADAR='/Users/xia/.codex/plugins/cache/zhenhao-arxiv-tools/arxiv-watcher/0.2.0/skills/arxiv-paper-radar/scripts/radar.py'
(HERE/'papers').mkdir(exist_ok=True);(HERE/'packets').mkdir(exist_ok=True)
pool=json.loads((HERE/'candidate_pool20.json').read_text())
selected=[p for p in pool['papers'] if p['triage_decision']=='full_text_packet']
manifest=[]
for paper in selected:
 aid=paper['arxiv_id']; title=paper['title'];start=time.time()
 row=dict(paper_id=paper['paper_id'],title=title,authors=paper['authors'],version=paper['version'],official_page=paper['abs_url'],attempt_at=datetime.now(timezone.utc).isoformat())
 if aid:
  cmd=['python3',RADAR,'packet',aid,'--config',str(HERE/'radar.toml'),'--output',str(HERE/'packets'/f'{aid}_native.md')]
  done=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
  row['native_packet_attempt']=dict(command=cmd,exit_code=done.returncode,stdout=done.stdout,stderr=done.stderr)
 else:row['native_packet_attempt']=dict(status='not_applicable',reason='Official NeurIPS paper has no verified arXiv ID; do not invent one for an arXiv-only CLI.')
 if aid=='2604.03266':
  path=WORK/'research_program/literature_audit_2026-09-15/papers/2026_Kaszynski_Emergent Compositional Communication for Latent World Properties_arXiv2604.03266v1.pdf'
  url='https://arxiv.org/pdf/2604.03266v1';row['status']='reused_existing_pdf';row['new_pdf_download']=False
 elif aid=='2508.06659':
  path=HERE/'papers/2025_MartinezLopez_CORAL_In_Context_Reinforcement_Learning_v2_2026.pdf'
  url='https://arxiv.org/pdf/2508.06659v2';row['status']='download_requested';row['new_pdf_download']=True
 else:
  path=HERE/'papers/2024_Gualdoni_Bridging_Semantics_and_Pragmatics_NeurIPS.pdf'
  url='https://proceedings.neurips.cc/paper_files/paper/2024/file/2548fbe155ed2405488b7d5373013a64-Paper-Conference.pdf'
  row['status']='download_requested';row['new_pdf_download']=True
 row['official_pdf_url']=url;row['local_path']=str(path)
 try:
  if row['new_pdf_download']:
   assert not path.exists()
   req=urllib.request.Request(url,headers={'User-Agent':'ResearchLiteratureAudit/1.0 (bounded primary-source paper download)'})
   with urllib.request.urlopen(req,timeout=70) as resp:
    data=resp.read();row.update(http_status=resp.status,resolved_url=resp.url,content_type=resp.headers.get('Content-Type'))
   if not data.startswith(b'%PDF-'):raise ValueError('Response lacks PDF header')
   path.write_bytes(data);row['status']='downloaded_new'
  data=path.read_bytes();assert data.startswith(b'%PDF-')
  reader=PdfReader(path);pages=[p.extract_text() or '' for p in reader.pages]
  text='\n\n'.join(f'## PDF page {i+1}\n\n{x}' for i,x in enumerate(pages))
  normalize=lambda x: re.sub(r'\W','',unicodedata.normalize('NFKD',x).casefold())
  row.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),pdf_header=data[:8].decode(),page_count=len(pages),text_chars_by_page=list(map(len,pages)),all_pages_extractable=all(len(p.strip())>40 for p in pages),first_page_title_normalized_match=normalize(title) in normalize(pages[0]),first_page_authors={a:normalize(a) in normalize(pages[0]) for a in paper['authors']},pdf_metadata={str(k):str(v) for k,v in (reader.metadata or {}).items()})
  name=aid or 'gualdoni_neurips2024'
  packet=HERE/'packets'/f'{name}.md'
  packet.write_text('# Full-text packet: '+title+'\n\n'+'- Packet builder: explicit official/local fallback, NOT native collector packet.\n- Official page: '+paper['abs_url']+'\n- PDF: '+str(path)+'\n- Version: '+paper['version']+'\n- PDF SHA256: '+row['sha256']+'\n- Extractor: pypdf; all pages retained, no character truncation.\n\n'+text+'\n')
  row.update(packet_path=str(packet),packet_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),packet_characters=len(text))
 except Exception as exc:
  row.update(status='failed',error=repr(exc))
 row['elapsed_seconds']=time.time()-start;manifest.append(row)
 (HERE/'download_packet_index.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:row[k] for k in ('title','status','elapsed_seconds')}),flush=True)
